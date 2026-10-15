#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// glibc rand_r advances its state with a fixed affine map mod 2^32 per call,
// so the generator can be jumped ahead to allow a parallel, identical sequence.
struct Affine {
    unsigned int a, c;
};

inline Affine compose(const Affine f, const Affine g) noexcept {  // f(g(x))
    return Affine{f.a * g.a, f.a * g.c + f.c};
}

Affine randRStep() {
    unsigned int s0 = 0, s1 = 1;
    rand_r(&s0);
    rand_r(&s1);
    return Affine{s1 - s0, s0};
}

unsigned int randRState(const unsigned int seed, unsigned long long steps) {
    Affine result{1, 0};
    Affine base = randRStep();
    while (steps) {
        if (steps & 1) result = compose(base, result);
        base = compose(base, base);
        steps >>= 1;
    }
    return result.a * seed + result.c;
}

// Initializes rows [rowBegin, rowEnd) of the distance matrix (row-major, local storage)
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t rowBegin, const size_t rowEnd,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t first = rowBegin * numNodes;
    const size_t count = (rowEnd - rowBegin) * numNodes;

    #pragma omp parallel
    {
        const size_t nt = omp_get_num_threads();
        const size_t t = omp_get_thread_num();
        const size_t lo = count * t / nt;
        const size_t hi = count * (t + 1) / nt;
        unsigned int state = randRState(seed, first + lo);
        for (size_t i = lo; i < hi; ++i) {
            dist[i] = rangeMin + (unsigned int)(range * rand_r(&state) / (double)RAND_MAX);
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

// Block-row distributed Floyd-Warshall. Row k of the matrix is broadcast each step.
// Row k and column k are unchanged during step k, so all cells update independently.
__global__ void fwKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const unsigned int* __restrict__ rowK, const unsigned int k,
                         const unsigned int n, const unsigned int rowBase,
                         const unsigned int skipRow) {
    const unsigned int row = rowBase + blockIdx.x;
    if (row == skipRow) return;
    const unsigned int j = (blockIdx.y * blockDim.x + threadIdx.x) * 4u;
    if (j >= n) return;
    const size_t off = (size_t)row * n + j;
    const unsigned int distIK = dist[(size_t)row * n + k];
    if (j + 4 <= n && (off & 3) == 0) {
        uint4 d = *reinterpret_cast<const uint4*>(dist + off);
        const uint4 r = *reinterpret_cast<const uint4*>(rowK + j);
        uint4 p = *reinterpret_cast<const uint4*>(path + off);
        bool ch = false;
        unsigned int nd;
        nd = distIK + r.x; if (nd < d.x) { d.x = nd; p.x = k; ch = true; }
        nd = distIK + r.y; if (nd < d.y) { d.y = nd; p.y = k; ch = true; }
        nd = distIK + r.z; if (nd < d.z) { d.z = nd; p.z = k; ch = true; }
        nd = distIK + r.w; if (nd < d.w) { d.w = nd; p.w = k; ch = true; }
        if (ch) {
            *reinterpret_cast<uint4*>(dist + off) = d;
            *reinterpret_cast<uint4*>(path + off) = p;
        }
    } else {
        for (unsigned int u = 0; u < 4 && j + u < n; ++u) {
            const unsigned int nd = distIK + rowK[j + u];
            if (nd < dist[off + u]) {
                dist[off + u] = nd;
                path[off + u] = k;
            }
        }
    }
}

__global__ void initPathKernel(unsigned int* path, const size_t rows, const unsigned int n,
                               const unsigned int rowBegin) {
    const size_t total = rows * n;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        path[i] = rowBegin + (unsigned int)(i / n);
    }
}

// Computes the full result; the final distance matrix is gathered into `dist` on rank 0.
void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& localDist,
                   const size_t numNodes, const std::vector<int>& rowStart,
                   const int rank, const int size) {
    const unsigned int n = (unsigned int)numNodes;
    const size_t rb = rowStart[rank], re = rowStart[rank + 1];
    const size_t rows = re - rb;

    auto ownerOf = [&](size_t row) {
        int lo = 0, hi = size - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if ((size_t)rowStart[mid] <= row) lo = mid; else hi = mid - 1;
        }
        return lo;
    };

    unsigned int *dDist = nullptr, *dPath = nullptr, *dRow[2] = {nullptr, nullptr};
    unsigned int* hRow = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(rows * n, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(rows * n, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRow[0], n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRow[1], n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hRow, n * sizeof(unsigned int)));

    cudaStream_t sMain, sRow;
    int lo, hi;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sMain, cudaStreamNonBlocking, lo));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sRow, cudaStreamNonBlocking, hi));
    std::vector<cudaEvent_t> evMain(numNodes), evRow(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evMain[k], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evRow[k], cudaEventDisableTiming));
    }
    cudaEvent_t evBuf[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&evBuf[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evBuf[1], cudaEventDisableTiming));

    if (rows) {
        CUDA_CHECK(cudaMemcpyAsync(dDist, localDist.data(), rows * n * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, sMain));
        initPathKernel<<<1024, 256, 0, sMain>>>(dPath, rows, n, (unsigned int)rb);
    }
    CUDA_CHECK(cudaStreamSynchronize(sMain));

    // Broadcast row r from its owner into device buffer dRow[r & 1] (on sRow)
    auto bcastRow = [&](size_t r) {
        const int owner = ownerOf(r);
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpyAsync(hRow, dDist + (r - rb) * n, n * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, sRow));
            CUDA_CHECK(cudaStreamSynchronize(sRow));
        }
        MPI_Bcast(hRow, (int)n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(dRow[r & 1], hRow, n * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, sRow));
        CUDA_CHECK(cudaEventRecord(evBuf[r & 1], sRow));
    };

    const unsigned int colBlocks = (n + 4 * 256 - 1) / (4 * 256);
    const unsigned int NONE = 0xFFFFFFFFu;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bcastRow(0);
    for (unsigned int k = 0; k < n; ++k) {
        const size_t next = (size_t)k + 1;
        const bool nextLocal = next < numNodes && next >= rb && next < re;
        const unsigned int skip = nextLocal ? (unsigned int)(next - rb) : NONE;

        // Main update of all local rows except the one needed next
        if (rows) {
            CUDA_CHECK(cudaStreamWaitEvent(sMain, evBuf[k & 1], 0));
            if (k > 0) CUDA_CHECK(cudaStreamWaitEvent(sMain, evRow[k - 1], 0));
            fwKernel<<<dim3((unsigned int)rows, colBlocks), 256, 0, sMain>>>(
                dDist, dPath, dRow[k & 1], k, n, 0, skip);
            CUDA_CHECK(cudaEventRecord(evMain[k], sMain));
        }

        if (next < numNodes) {
            // Update next row first (if owned) so that it can be shipped while the main kernel runs
            if (nextLocal) {
                if (k > 0) CUDA_CHECK(cudaStreamWaitEvent(sRow, evMain[k - 1], 0));
                fwKernel<<<dim3(1, colBlocks), 256, 0, sRow>>>(
                    dDist, dPath, dRow[k & 1], k, n, skip, NONE);
                CUDA_CHECK(cudaEventRecord(evRow[k], sRow));
            } else if (rows) {
                CUDA_CHECK(cudaEventRecord(evRow[k], sRow));
            }
            // Buffer dRow[(k+1)&1] was read by step k-1's main kernel
            if (rows && k > 0) CUDA_CHECK(cudaStreamWaitEvent(sRow, evMain[k - 1], 0));
            bcastRow(next);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaStreamSynchronize(sRow));

    // Gather final distance matrix on rank 0
    if (rows) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, rows * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = (rowStart[r + 1] - rowStart[r]) * (int)n;
        displs[r] = rowStart[r] * (int)n;
    }
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? dist.data() : nullptr,
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        cudaEventDestroy(evMain[k]);
        cudaEventDestroy(evRow[k]);
    }
    cudaEventDestroy(evBuf[0]);
    cudaEventDestroy(evBuf[1]);
    cudaStreamDestroy(sMain);
    cudaStreamDestroy(sRow);
    cudaFreeHost(hRow);
    cudaFree(dRow[0]);
    cudaFree(dRow[1]);
    cudaFree(dDist);
    cudaFree(dPath);
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // One GPU per rank, assigned round-robin within a node
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank, numDevices = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    }

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Block-row decomposition across ranks
    std::vector<int> rowStart(size + 1);
    for (int r = 0; r <= size; ++r) rowStart[r] = (int)(numNodes * r / size);
    const size_t rb = rowStart[rank], re = rowStart[rank + 1];

    std::vector<unsigned int> localDist((re - rb) * numNodes);
    std::vector<unsigned int> dist;
    if (rank == 0) dist.resize(numNodes * numNodes);
    initializeDistanceMatrix(localDist, numNodes, rb, re, 1, MAX_DISTANCE);

    if (rank == 0) printf("Computing shortest paths...\n");
    floydWarshall(dist, localDist, numNodes, rowStart, rank, size);

    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }
    MPI_Finalize();
    return ret;
}
