#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
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

// Path matrix initial value: path[idx2(i,j)] = j for the (row j, col i) layout, i.e.
// every element of row r holds r (the diagonal is also r).
__global__ void initPathKernel(unsigned int* path, const size_t rows, const size_t n,
                               const size_t rowStart) {
    size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t r = blockIdx.y;
    if (c < n && r < rows) path[r * n + c] = (unsigned int)(rowStart + r);
}

// One Floyd-Warshall relaxation step for local rows [rowBegin, rowBegin+rowCount),
// skipping local row `skipRow` (already updated elsewhere). rowK is the global row k.
__global__ void fwStepKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             const unsigned int* __restrict__ rowK, const size_t n,
                             const unsigned int k, const size_t rowBegin, const long long skipRow) {
    const size_t r = rowBegin + blockIdx.y;
    if ((long long)r == skipRow) return;
    const unsigned int distIK = dist[r * n + k];
    const size_t j0 = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    unsigned int* d = dist + r * n;
    unsigned int* p = path + r * n;
    if (j0 + 4 <= n && (n & 3) == 0) {
        uint4 kj = *reinterpret_cast<const uint4*>(rowK + j0);
        uint4 dv = *reinterpret_cast<uint4*>(d + j0);
        uint4 pv = *reinterpret_cast<uint4*>(p + j0);
        unsigned int nd;
        nd = distIK + kj.x; if (nd < dv.x) { dv.x = nd; pv.x = k; }
        nd = distIK + kj.y; if (nd < dv.y) { dv.y = nd; pv.y = k; }
        nd = distIK + kj.z; if (nd < dv.z) { dv.z = nd; pv.z = k; }
        nd = distIK + kj.w; if (nd < dv.w) { dv.w = nd; pv.w = k; }
        *reinterpret_cast<uint4*>(d + j0) = dv;
        *reinterpret_cast<uint4*>(p + j0) = pv;
    } else {
        for (size_t j = j0; j < j0 + 4 && j < n; ++j) {
            const unsigned int nd = distIK + rowK[j];
            if (nd < d[j]) { d[j] = nd; p[j] = k; }
        }
    }
}

// Distributed (row-block over MPI ranks) GPU Floyd-Warshall.
// localDist holds this rank's rows [rowStart, rowStart+rows) on entry and exit.
void floydWarshallHybrid(std::vector<unsigned int>& localDist, const size_t numNodes,
                         const std::vector<size_t>& starts, const int rank) {
    const size_t n = numNodes;
    const size_t rowStart = starts[rank];
    const size_t rows = starts[rank + 1] - rowStart;

    auto ownerOf = [&](size_t row) {
        int o = (int)(std::upper_bound(starts.begin(), starts.end(), row) - starts.begin()) - 1;
        return o;
    };

    unsigned int *dDist = nullptr, *dPath = nullptr, *dRow[2] = {nullptr, nullptr};
    unsigned int* hRow[2] = {nullptr, nullptr};
    const size_t localBytes = std::max<size_t>(rows, 1) * n * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, localBytes));
    CUDA_CHECK(cudaMalloc(&dPath, localBytes));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dRow[b], n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&hRow[b], n * sizeof(unsigned int)));
    }

    cudaStream_t sMain, sRow;
    int lo, hi;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sMain, cudaStreamNonBlocking, lo));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sRow, cudaStreamNonBlocking, hi));

    if (rows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(dDist, localDist.data(), rows * n * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, sMain));
        dim3 blk(256), grd((unsigned)((n + 255) / 256), (unsigned)rows);
        initPathKernel<<<grd, blk, 0, sMain>>>(dPath, rows, n, rowStart);
    }
    CUDA_CHECK(cudaStreamSynchronize(sMain));
    MPI_Barrier(MPI_COMM_WORLD);

    const int threads = 64;  // each thread handles 4 columns
    const unsigned gx = (unsigned)((n + 4ull * threads - 1) / (4ull * threads));

    // Row 0 for step 0
    {
        const int o = ownerOf(0);
        if (rank == o) {
            CUDA_CHECK(cudaMemcpy(hRow[0], dDist + (0 - rowStart) * n, n * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hRow[0], (int)n, MPI_UNSIGNED, o, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dRow[0], hRow[0], n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    for (size_t k = 0; k < n; ++k) {
        const int cur = (int)(k & 1), nxt = cur ^ 1;
        const bool hasNext = (k + 1 < n);
        const int nextOwner = hasNext ? ownerOf(k + 1) : -1;
        const bool iOwnNext = hasNext && rank == nextOwner;
        long long skip = -1;
        MPI_Request req = MPI_REQUEST_NULL;

        if (iOwnNext) {
            // Update the next pivot row first so it can be broadcast while the rest computes.
            const size_t lr = k + 1 - rowStart;
            skip = (long long)lr;
            dim3 grd(gx, 1), blk(threads);
            fwStepKernel<<<grd, blk, 0, sRow>>>(dDist, dPath, dRow[cur], n, (unsigned)k, lr, -1);
            CUDA_CHECK(cudaMemcpyAsync(dRow[nxt], dDist + lr * n, n * sizeof(unsigned int),
                                       cudaMemcpyDeviceToDevice, sRow));
            CUDA_CHECK(cudaMemcpyAsync(hRow[nxt], dDist + lr * n, n * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, sRow));
        }
        if (rows > 0) {
            dim3 grd(gx, (unsigned)rows), blk(threads);
            fwStepKernel<<<grd, blk, 0, sMain>>>(dDist, dPath, dRow[cur], n, (unsigned)k, 0, skip);
        }
        if (hasNext) {
            if (iOwnNext) CUDA_CHECK(cudaStreamSynchronize(sRow));
            MPI_Ibcast(hRow[nxt], (int)n, MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (!iOwnNext) {
                CUDA_CHECK(cudaMemcpyAsync(dRow[nxt], hRow[nxt], n * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, sMain));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(sMain));
    }

    if (rows > 0) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, rows * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaStreamDestroy(sMain));
    CUDA_CHECK(cudaStreamDestroy(sRow));
    for (int b = 0; b < 2; ++b) {
        cudaFree(dRow[b]);
        cudaFreeHost(hRow[b]);
    }
    cudaFree(dDist);
    cudaFree(dPath);
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    long long badDiag = -1;
    #pragma omp parallel for reduction(max : badDiag)
    for (long long i = 0; i < (long long)numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) badDiag = std::max(badDiag, i);
    }
    if (badDiag >= 0) {
        printf("Validation failed: diagonal element [%lld,%lld] is not zero\n", badDiag, badDiag);
        return false;
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

    // Bind each rank on a node to a GPU
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
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

    // Row-block decomposition
    std::vector<size_t> starts(size + 1);
    for (int r = 0; r <= size; ++r) {
        starts[r] = (numNodes / size) * r + std::min<size_t>(r, numNodes % size);
    }
    const size_t myRows = starts[rank + 1] - starts[rank];

    // Rank 0 generates the (sequential RNG) matrix and scatters row blocks
    std::vector<unsigned int> dist;
    std::vector<unsigned int> localDist(myRows * numNodes);
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = (int)((starts[r + 1] - starts[r]) * numNodes);
        displs[r] = (int)(starts[r] * numNodes);
    }
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(dist.data(), counts.data(), displs.data(), MPI_UNSIGNED, localDist.data(),
                 (int)localDist.size(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallHybrid(localDist, numNodes, starts, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MPI_Gatherv(localDist.data(), (int)localDist.size(), MPI_UNSIGNED, dist.data(), counts.data(),
                displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Floyd-Warshall has O(n^3) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
                status = 1;
            }
        }
    }

    MPI_Finalize();
    return status;
}
