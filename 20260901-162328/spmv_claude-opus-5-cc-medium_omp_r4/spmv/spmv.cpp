#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#endif

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Raw, uninitialized aligned storage.
//
// std::vector zero-initializes on construction, which faults every page in on
// the NUMA node of the (single) constructing thread. Using uninitialized
// storage lets us perform the first touch from the thread that will later own
// the data, so pages end up local to their consumer on multi-socket systems.
// ****************************************************************************
struct FreeDeleter {
    void operator()(void* p) const { std::free(p); }
};

template <typename T>
using aligned_array = std::unique_ptr<T[], FreeDeleter>;

template <typename T>
aligned_array<T> make_aligned_array(const size_t n) {
    constexpr size_t alignment = 64;
    const size_t bytes = ((n * sizeof(T) + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, bytes == 0 ? alignment : bytes);
    if (p == nullptr) {
        printf("Allocation of %zu bytes failed\n", bytes);
        exit(1);
    }
    return aligned_array<T>(static_cast<T*>(p));
}

#ifdef __linux__
// Returns the CPUs this process may run on, ordered by decreasing preference:
// one hardware thread of every physical core first (alternating between
// sockets, so that a small team already uses all memory controllers), then the
// second hardware thread of every core, and so on. Falls back to the plain
// ascending CPU list if the topology cannot be read.
std::vector<int> preferredCpuOrder(const std::vector<int>& cpus) {
    // Group the CPUs by physical core, keeping one bucket list per socket
    std::vector<int> packageIds;                     // socket id per bucket group
    std::vector<std::vector<std::vector<int>>> byPackage;  // [package][core][hwthread]
    std::vector<std::vector<int>> coreIdsByPackage;  // core id of each bucket, per package

    for (const int cpu : cpus) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                 cpu);
        FILE* f = fopen(path, "r");
        int pkg = -1;
        if (f == nullptr || fscanf(f, "%d", &pkg) != 1) {
            if (f != nullptr) fclose(f);
            return cpus;  // unknown topology
        }
        fclose(f);

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        f = fopen(path, "r");
        int core = -1;
        if (f == nullptr || fscanf(f, "%d", &core) != 1) {
            if (f != nullptr) fclose(f);
            return cpus;  // unknown topology
        }
        fclose(f);

        auto pkgIt = std::find(packageIds.begin(), packageIds.end(), pkg);
        if (pkgIt == packageIds.end()) {
            packageIds.push_back(pkg);
            byPackage.emplace_back();
            coreIdsByPackage.emplace_back();
            pkgIt = packageIds.end() - 1;
        }
        const size_t p = static_cast<size_t>(pkgIt - packageIds.begin());

        auto coreIt = std::find(coreIdsByPackage[p].begin(), coreIdsByPackage[p].end(), core);
        if (coreIt == coreIdsByPackage[p].end()) {
            coreIdsByPackage[p].push_back(core);
            byPackage[p].emplace_back();
            coreIt = coreIdsByPackage[p].end() - 1;
        }
        byPackage[p][static_cast<size_t>(coreIt - coreIdsByPackage[p].begin())].push_back(cpu);
    }

    // Round-robin over the sockets to get a socket-interleaved list of cores
    std::vector<const std::vector<int>*> cores;
    size_t maxCores = 0;
    for (const auto& pkg : byPackage) {
        maxCores = std::max(maxCores, pkg.size());
    }
    for (size_t c = 0; c < maxCores; ++c) {
        for (const auto& pkg : byPackage) {
            if (c < pkg.size()) {
                cores.push_back(&pkg[c]);
            }
        }
    }

    // Physical cores first, hardware threads of the same core last
    std::vector<int> order;
    order.reserve(cpus.size());
    size_t maxSiblings = 0;
    for (const auto* core : cores) {
        maxSiblings = std::max(maxSiblings, core->size());
    }
    for (size_t s = 0; s < maxSiblings; ++s) {
        for (const auto* core : cores) {
            if (s < core->size()) {
                order.push_back((*core)[s]);
            }
        }
    }
    return order;
}
#endif

// ****************************************************************************
// Function: bindThreads
//
// Purpose:
//   SpMV is bandwidth bound, so thread placement decides the achievable
//   performance: threads have to be spread over all sockets (to use all memory
//   controllers) and they must not migrate afterwards, or the NUMA-aware first
//   touch below is pointless.
//
//   If the user did not request a binding policy (OMP_PROC_BIND unset), pin
//   each thread ourselves, spreading the team evenly over the CPUs this
//   process is allowed to run on. An explicit OMP_PROC_BIND/OMP_PLACES setting
//   always wins, and the affinity mask of the process is respected, so running
//   under taskset/cgroups keeps working.
// ****************************************************************************
void bindThreads([[maybe_unused]] const int numThreads) {
#ifdef __linux__
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        return;  // runtime handles binding
    }

    cpu_set_t allowedMask;
    CPU_ZERO(&allowedMask);
    if (sched_getaffinity(0, sizeof(allowedMask), &allowedMask) != 0) {
        return;
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowedMask)) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.size() < 2) {
        return;
    }

    const std::vector<int> order = preferredCpuOrder(cpus);
    const int numCpus = static_cast<int>(order.size());
#pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        // Consecutive threads take distinct physical cores as long as there are
        // any left; oversubscribed teams wrap around.
        const int cpu = order[static_cast<size_t>(tid % numCpus)];
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
#endif
}

// Touch every page of [data, data+n) from the threads that will use it, so that
// the first-touch policy of the OS places the pages on the right NUMA node.
template <typename T>
void firstTouch(T* data, const size_t n) {
    constexpr size_t pageElems = 4096 / sizeof(T);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += pageElems) {
        data[i] = T{};
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//   rowPartition: precomputed row ranges, one per thread (size numThreads+1)
//   numThreads: size of the thread team rowPartition was computed for
//
// Rows are distributed statically, but the split points are chosen so that each
// thread gets (nearly) the same number of non-zeroes rather than the same
// number of rows. This balances the load for arbitrary row length
// distributions while keeping the memory access pattern of each thread
// contiguous and stable across iterations (important for NUMA locality).
// Each row is still summed by a single thread in the original order, so the
// results are bitwise identical to the sequential version.
// ****************************************************************************

// Computes the rows [rowBegin, rowEnd) of the product. Called from inside a
// parallel region; the row ranges of the threads are disjoint, so no
// synchronization is needed.
void spmvRows(const double* val, const index_t* cols, const index_t* rowDelimiters,
              const double* vec, double* out, const index_t rowBegin, const index_t rowEnd) {
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, [[maybe_unused]] const index_t dim, double* out,
             const index_t* rowPartition, const int numThreads) {
#pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        spmvRows(val, cols, rowDelimiters, vec, out, rowPartition[tid], rowPartition[tid + 1]);
    }
}

// ****************************************************************************
// Function: computeRowPartition
//
// Purpose:
//   Splits the rows of the matrix into numThreads contiguous chunks holding
//   approximately equal numbers of non-zero elements.
// ****************************************************************************
std::vector<index_t> computeRowPartition(const index_t* rowDelimiters, const index_t dim,
                                         const int numThreads) {
    std::vector<index_t> partition(static_cast<size_t>(numThreads) + 1);
    const index_t nnz = rowDelimiters[dim];

    partition[0] = 0;
    partition[numThreads] = dim;
    for (int t = 1; t < numThreads; ++t) {
        const index_t target =
            static_cast<index_t>((static_cast<uint64_t>(nnz) * t) / numThreads);
        const index_t* pos = std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target);
        index_t row = static_cast<index_t>(pos - rowDelimiters);
        // Keep the partition monotone even for degenerate matrices
        if (row < partition[t - 1]) {
            row = partition[t - 1];
        }
        if (row > dim) {
            row = dim;
        }
        partition[t] = row;
    }
    return partition;
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const int numThreads = omp_get_max_threads();
    printf("OpenMP threads: %d\n", numThreads);

    // Pin the threads before any data is touched
    bindThreads(numThreads);

    // Allocate and initialize data structures
    auto h_val = make_aligned_array<double>(nItems);              // Non-zero values
    auto h_cols = make_aligned_array<index_t>(nItems);            // Column indices
    auto h_rowDelimiters = make_aligned_array<index_t>(numRows + 1);  // Row delimiters
    auto h_vec = make_aligned_array<double>(numRows);             // Dense vector
    auto h_out = make_aligned_array<double>(numRows);             // Output vector

    // First touch from the threads that will later own the data. The rows are
    // split by non-zero count, which for a contiguous CSR layout corresponds to
    // an (almost) even split of h_val/h_cols, so a plain static distribution
    // places the pages on the right NUMA node. h_vec is accessed with random
    // indices by all threads, so spreading it evenly acts like interleaving.
    firstTouch(h_val.get(), nItems);
    firstTouch(h_cols.get(), nItems);
    firstTouch(h_rowDelimiters.get(), static_cast<size_t>(numRows) + 1);
    firstTouch(h_vec.get(), numRows);
    firstTouch(h_out.get(), numRows);

    printf("Initializing data structures...\n");
    fill(h_vec.get(), numRows, maxVal);
    fill(h_val.get(), nItems, maxVal);
    initRandomMatrix(h_cols.get(), h_rowDelimiters.get(), nItems, numRows);

    // Distribute the rows over the threads, balanced by non-zero count
    const std::vector<index_t> rowPartition =
        computeRowPartition(h_rowDelimiters.get(), numRows, numThreads);

    // For validation, compute reference solution
    auto h_reference = make_aligned_array<double>(validate ? numRows : 0);
    if (validate) {
        printf("Computing reference solution...\n");
        firstTouch(h_reference.get(), numRows);
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.get(),
                h_vec.get(), numRows, h_reference.get(), rowPartition.data(), numThreads);
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // The parallel region is opened once for the whole iteration loop: every
    // thread recomputes its own, fixed set of rows and nothing is shared
    // between the threads, so neither the fork/join nor a barrier per
    // iteration is needed.
#pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();
        const index_t rowBegin = rowPartition[tid];
        const index_t rowEnd = rowPartition[tid + 1];
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvRows(h_val.get(), h_cols.get(), h_rowDelimiters.get(), h_vec.get(), h_out.get(),
                     rowBegin, rowEnd);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> outVec(h_out.get(), h_out.get() + numRows);
        print_results(outVec, "OutputVector");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.get(), h_out.get(), numRows);

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
