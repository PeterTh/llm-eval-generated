#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <tuple>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Thread placement
//
// The kernel is memory bound, so where the threads run decides how much of the
// machine's bandwidth (and L3) it can reach. If the environment does not
// request a binding, spread the threads over physical cores of all sockets
// ourselves; an explicit OMP_PROC_BIND/OMP_PLACES setting is left untouched.
// ---------------------------------------------------------------------------

namespace {

struct CpuInfo {
    int cpu = 0;
    int package = 0;  // socket
    int core = 0;     // physical core within the socket
    int smt = 0;      // index of this cpu among the siblings of its core
};

int readTopologyInt(const int cpu, const char* file, const int fallback) {
    char pathBuf[128];
    snprintf(pathBuf, sizeof(pathBuf), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, file);
    FILE* const f = fopen(pathBuf, "r");
    if (f == nullptr) {
        return fallback;
    }
    int value = fallback;
    if (fscanf(f, "%d", &value) != 1) {
        value = fallback;
    }
    fclose(f);
    return value;
}

// Position of `cpu` in its own thread_siblings_list, i.e. its SMT index.
int readSmtIndex(const int cpu) {
    char pathBuf[128];
    snprintf(pathBuf, sizeof(pathBuf), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE* const f = fopen(pathBuf, "r");
    if (f == nullptr) {
        return 0;
    }
    int index = 0;
    int sibling = 0;
    int position = 0;
    while (fscanf(f, "%d", &sibling) == 1) {
        if (sibling == cpu) {
            index = position;
            break;
        }
        ++position;
        int next = fgetc(f);
        if (next != ',' && next != '-') {
            break;
        }
    }
    fclose(f);
    return index;
}

// CPUs of the process affinity mask, ordered so that taking the first N entries
// yields N distinct physical cores spread evenly over all sockets, and only
// starts reusing cores via SMT once every core is busy.
std::vector<CpuInfo> buildCpuOrder() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    std::vector<CpuInfo> cpus;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        return cpus;
    }

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) {
            continue;
        }
        CpuInfo info;
        info.cpu = cpu;
        info.package = readTopologyInt(cpu, "physical_package_id", 0);
        info.core = readTopologyInt(cpu, "core_id", cpu);
        info.smt = readSmtIndex(cpu);
        cpus.push_back(info);
    }

    // Rank the packages and, inside each package, the cores, so that the sort
    // below can interleave sockets instead of filling them one after another.
    std::vector<int> packages;
    for (const CpuInfo& info : cpus) {
        packages.push_back(info.package);
    }
    std::sort(packages.begin(), packages.end());
    packages.erase(std::unique(packages.begin(), packages.end()), packages.end());

    std::vector<std::vector<int>> coresPerPackage(packages.size());
    for (const CpuInfo& info : cpus) {
        const size_t p = std::lower_bound(packages.begin(), packages.end(), info.package) - packages.begin();
        coresPerPackage[p].push_back(info.core);
    }
    for (std::vector<int>& cores : coresPerPackage) {
        std::sort(cores.begin(), cores.end());
        cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
    }

    const auto rankOf = [&](const CpuInfo& info) {
        const size_t p = std::lower_bound(packages.begin(), packages.end(), info.package) - packages.begin();
        const std::vector<int>& cores = coresPerPackage[p];
        const size_t c = std::lower_bound(cores.begin(), cores.end(), info.core) - cores.begin();
        return std::make_tuple(info.smt, c, p, info.cpu);
    };
    std::sort(cpus.begin(), cpus.end(), [&](const CpuInfo& a, const CpuInfo& b) {
        return rankOf(a) < rankOf(b);
    });

    return cpus;
}

size_t countPhysicalCores(const std::vector<CpuInfo>& cpus) {
    size_t cores = 0;
    for (const CpuInfo& info : cpus) {
        if (info.smt == 0) {
            ++cores;
        }
    }
    return cores > 0 ? cores : cpus.size();
}

// Pins the OpenMP thread pool. Called once, before any timed work.
void setupThreadPlacement() {
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        return;  // the environment already asked for a specific binding
    }

    const std::vector<CpuInfo> cpus = buildCpuOrder();
    if (cpus.empty()) {
        return;
    }

    // Without an explicit request, one thread per physical core beats using the
    // SMT siblings as well for this bandwidth-bound kernel.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int cores = static_cast<int>(countPhysicalCores(cpus));
        omp_set_num_threads(std::min(omp_get_max_threads(), cores));
    }

    const int numThreads = omp_get_max_threads();
    #pragma omp parallel num_threads(numThreads)
    {
        const size_t slot = static_cast<size_t>(omp_get_thread_num()) % cpus.size();
        cpu_set_t threadMask;
        CPU_ZERO(&threadMask);
        CPU_SET(cpus[slot].cpu, &threadMask);
        pthread_setaffinity_np(pthread_self(), sizeof(threadMask), &threadMask);
    }
}

}  // namespace

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Number of destination nodes handled per vectorized block in the inner loop.
constexpr size_t BLOCK_J = 64;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Allocator that hands out cache-line aligned storage and performs the initial
// page touch in parallel, so that on NUMA machines every matrix row ends up in
// the memory of the socket whose threads will later own that row (the OpenMP
// loops below all use a matching static schedule). Construction is a no-op
// because allocate() already zero-fills the storage.
template <typename T>
struct FirstTouchAllocator {
    using value_type = T;

    FirstTouchAllocator() noexcept = default;
    template <typename U>
    FirstTouchAllocator(const FirstTouchAllocator<U>&) noexcept {}

    T* allocate(const size_t count) {
        const size_t bytes = count * sizeof(T);
        void* const mem = ::operator new(bytes, std::align_val_t(64));

        char* const raw = static_cast<char*>(mem);
        const ptrdiff_t signedBytes = static_cast<ptrdiff_t>(bytes);
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t offset = 0; offset < signedBytes; offset += 4096) {
            const size_t span = std::min<size_t>(4096, bytes - static_cast<size_t>(offset));
            memset(raw + offset, 0, span);
        }

        return static_cast<T*>(mem);
    }

    void deallocate(T* ptr, const size_t) noexcept {
        ::operator delete(static_cast<void*>(ptr), std::align_val_t(64));
    }

    template <typename U, typename... Args>
    void construct(U*, Args&&...) noexcept {}

    template <typename U>
    void destroy(U*) noexcept {}

    template <typename U>
    bool operator==(const FirstTouchAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const FirstTouchAllocator<U>&) const noexcept { return false; }
};

using Matrix = std::vector<unsigned int, FirstTouchAllocator<unsigned int>>;

void initializeDistanceMatrix(Matrix& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Kept sequential: rand_r() is a serial recurrence and the exact sequence of
    // values defines the benchmark input. The pages were already first-touched
    // by the allocator, so writing them here does not change their NUMA home.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(Matrix& path, const size_t numNodes) {
    // The original nested loop writes path[idx2(i, j, n)] = j and
    // path[idx2(j, i, n)] = i, i.e. every element ends up holding the index of
    // the row (in the j*n+i layout: the second index) it belongs to.
    unsigned int* const p = path.data();
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        unsigned int* const row = p + j * numNodes;
        const unsigned int value = static_cast<unsigned int>(j);
        for (size_t i = 0; i < numNodes; ++i) {
            row[i] = value;
        }
    }
}

void floydWarshall(Matrix& dist,
                   Matrix& path,
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm.
    //
    // With the j*n+i indexing above, dist[idx2(j, i, n)] is element j of row i,
    // so rows (the source node i) are contiguous. Each iteration of k updates
    // every row from row k alone; rows are therefore distributed over the
    // threads while row k stays read-only for the whole iteration:
    //   * row k cannot update itself, because that would require
    //     dist[k][k] + dist[k][j] < dist[k][j] with dist[k][k] == 0, and
    //   * no thread ever writes outside the rows it owns.
    // A single parallel region spans the whole k loop, with the barrier at the
    // end of each `omp for` providing the ordering between successive k.
    unsigned int* const distData = dist.data();
    unsigned int* const pathData = path.data();

    #pragma omp parallel
    {
        for (size_t k = 0; k < numNodes; ++k) {
            const unsigned int* const kRow = distData + k * numNodes;
            const unsigned int kValue = static_cast<unsigned int>(k);

            // For each source node i
            #pragma omp for schedule(static)
            for (size_t i = 0; i < numNodes; ++i) {
                if (i == k) {
                    continue;  // row k is invariant during iteration k (see above)
                }

                unsigned int* const __restrict distRow = distData + i * numNodes;
                unsigned int* const __restrict pathRow = pathData + i * numNodes;
                const unsigned int distIK = distRow[k];

                // For each destination node j, in blocks: after the first few
                // values of k almost no element still improves, so a block that
                // has no update at all is left untouched instead of being
                // written back (halves the store traffic in the common case).
                size_t j = 0;
                for (; j + BLOCK_J <= numNodes; j += BLOCK_J) {
                    unsigned int updated = 0;
                    #pragma omp simd reduction(| : updated)
                    for (size_t t = 0; t < BLOCK_J; ++t) {
                        updated |= (distIK + kRow[j + t] < distRow[j + t]) ? 1u : 0u;
                    }
                    if (updated == 0) {
                        continue;
                    }
                    #pragma omp simd
                    for (size_t t = 0; t < BLOCK_J; ++t) {
                        const unsigned int distIJ = distRow[j + t];
                        const unsigned int newDist = distIK + kRow[j + t];
                        const bool improves = newDist < distIJ;
                        distRow[j + t] = improves ? newDist : distIJ;
                        pathRow[j + t] = improves ? kValue : pathRow[j + t];
                    }
                }
                for (; j < numNodes; ++j) {
                    const unsigned int distIJ = distRow[j];
                    const unsigned int newDist = distIK + kRow[j];
                    if (newDist < distIJ) {
                        distRow[j] = newDist;
                        pathRow[j] = kValue;
                    }
                }
            }
        }
    }
}

bool validateResult(const Matrix& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Pin the thread pool before anything is allocated, so that the allocator's
    // parallel first touch places each row on the socket that will own it.
    setupThreadPlacement();
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Allocate matrices
    Matrix dist(numNodes * numNodes);
    Matrix path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        // print_results_int() takes a std::vector with the default allocator.
        print_results_int(std::vector<unsigned int>(dist.begin(), dist.end()), "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
