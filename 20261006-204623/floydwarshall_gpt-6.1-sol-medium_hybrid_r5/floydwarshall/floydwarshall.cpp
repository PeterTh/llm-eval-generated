#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]; int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
// Chunk transfers so matrix dimensions are not limited by MPI's int counts.
void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) mpiCheck(MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else mpiCheck(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk; count -= chunk;
    }
}
void broadcast(unsigned int* data, size_t count, int owner) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(data, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        data += chunk; count -= chunk;
    }
}

// Turn a raw pivot panel into row-k snapshots at the instant k is visited.
// Only future pivot rows are updated: later improvements to earlier rows
// must not be used prematurely. Each column tile independently evolves the
// small diagonal block, avoiding a device-wide barrier for every vertex.
__global__ void preparePanel(const unsigned int* raw, unsigned int* panel, size_t n, size_t first, int width) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int values[TILE][TILE];
    int lane = threadIdx.x;
    size_t colBase = size_t(blockIdx.x) * TILE;
    for (int e = lane; e < TILE * TILE; e += blockDim.x) {
        int r = e / TILE, c = e % TILE;
        diagonal[r][c] = r < width && c < width ? raw[size_t(r)*n + first+c] : INF;
        values[r][c] = r < width && colBase+c < n ? raw[size_t(r)*n+colBase+c] : INF;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        for (int e = lane; e < TILE*TILE; e += blockDim.x) {
            int r = e / TILE, c = e % TILE;
            if (r > k && r < width) {
                values[r][c] = min(values[r][c], diagonal[r][k] + values[k][c]);
            }
        }
        __syncthreads();
        // Finish panel reads before modifying the diagonal coefficients.
        for (int e = lane; e < TILE*TILE; e += blockDim.x) {
            int r = e / TILE, c = e % TILE;
            if (r > k && r < width && c > k && c < width)
                diagonal[r][c] = min(diagonal[r][c], diagonal[r][k]+diagonal[k][c]);
        }
        __syncthreads();
    }
    for (int e = lane; e < TILE*TILE; e += blockDim.x) {
        int r = e / TILE, c = e % TILE;
        if (r < width && colBase+c < n) panel[size_t(r)*n+colBase+c] = values[r][c];
    }
}

// Capture each local row's d[i,k] in original k order. These coefficients
// allow independent column tiles to run a whole batch without global sync.
__global__ void prepareCoefficients(const unsigned int* dist, const unsigned int* panel,
                                   unsigned int* coefficients, size_t n, size_t rows,
                                   size_t first, int width) {
    __shared__ unsigned int diagonal[TILE][TILE];
    for (int e = threadIdx.x; e < TILE*TILE; e += blockDim.x) {
        int r = e / TILE, c = e % TILE;
        diagonal[r][c] = r < width && c < width ? panel[size_t(r)*n+first+c] : INF;
    }
    __syncthreads();
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    unsigned int values[TILE];
    for (int k = 0; k < width; ++k) values[k] = dist[row*n+first+k];
    for (int k = 0; k < width; ++k) {
        unsigned int v = values[k];
        coefficients[row*TILE+k] = v;
        for (int c = k+1; c < width; ++c) values[c] = min(values[c], v+diagonal[k][c]);
    }
}

__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* panel, const unsigned int* coefficients,
                           size_t n, size_t rows, size_t first, int width) {
    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int coeff[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y, lane = y*TILE+x;
    size_t rowBase = size_t(blockIdx.y)*TILE, colBase = size_t(blockIdx.x)*TILE;
    for (int e = lane; e < TILE*TILE; e += TILE*8) {
        int r = e/TILE, c = e%TILE;
        pivot[r][c] = r < width && colBase+c < n ? panel[size_t(r)*n+colBase+c] : INF;
        coeff[r][c] = rowBase+r < rows && c < width ? coefficients[(rowBase+r)*TILE+c] : INF;
    }
    __syncthreads();
    unsigned int d[4], p[4];
    size_t col = colBase+x;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        size_t row = rowBase+y+t*8;
        d[t] = row < rows && col < n ? dist[row*n+col] : INF;
        p[t] = row < rows && col < n ? path[row*n+col] : 0;
    }
    for (int k = 0; k < width; ++k) {
        unsigned int v = pivot[k][x];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned int candidate = coeff[y+t*8][k]+v;
            if (candidate < d[t]) { d[t] = candidate; p[t] = static_cast<unsigned int>(first+k); }
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        size_t row = rowBase+y+t*8;
        if (row < rows && col < n) { dist[row*n+col] = d[t]; path[row*n+col] = p[t]; }
    }
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
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (*arg == '-' || !*arg || *end || value > UINT_MAX ||
                value > std::numeric_limits<size_t>::max() ||
                (value && value > std::numeric_limits<size_t>::max()/value/sizeof(unsigned int))) {
                if (!rank) fprintf(stderr, "Invalid matrix dimension: %s\n", arg);
                MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    try {
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int localRank, devices;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "A CUDA GPU is required on every MPI rank.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        // Respect scheduler GPU visibility; ranks sharing a node rotate over visible GPUs.
        cudaCheck(cudaSetDevice(localRank % devices));
        mpiCheck(MPI_Comm_free(&local));
        size_t tiles = (n+TILE-1)/TILE;
        std::vector<size_t> starts(ranks+1);
        for (int r = 0; r <= ranks; ++r)
            starts[r] = std::min(n, (tiles/size_t(ranks)*r + std::min(size_t(r), tiles%ranks))*TILE);
        size_t rows = starts[rank+1]-starts[rank], elements = rows*n;
        std::vector<unsigned int> dist(elements), path(elements);
        if (!rank) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        // Preserve the original rand_r stream exactly, without replicating the
        // global matrix. Rank zero generates contiguous partitions in order.
        if (!rank) {
            unsigned int seed = 42;
            for (int r = 0; r < ranks; ++r) {
                size_t count = (starts[r+1]-starts[r])*n;
                std::vector<unsigned int> temporary(r ? count : 0);
                unsigned int* data = r ? temporary.data() : dist.data();
                for (size_t e = 0; e < count; ++e)
                    data[e] = 1 + static_cast<unsigned int>(double(MAX_DISTANCE)*rand_r(&seed)/double(RAND_MAX));
                if (r) transfer(data, count, r, true);
            }
        } else transfer(dist.data(), elements, 0, false);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row) {
            dist[row*n+starts[rank]+row] = 0;
            for (size_t col = 0; col < n; ++col) path[row*n+col] = static_cast<unsigned int>(starts[rank]+row);
        }
        unsigned int *deviceDist = nullptr, *devicePath = nullptr, *panel = nullptr, *rawPanel = nullptr, *coefficients = nullptr, *hostPanel = nullptr;
        cudaStream_t stream;
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (elements) {
            cudaCheck(cudaMalloc(&deviceDist, elements*sizeof(unsigned int)));
            cudaCheck(cudaMalloc(&devicePath, elements*sizeof(unsigned int)));
            cudaCheck(cudaMalloc(&coefficients, rows*TILE*sizeof(unsigned int)));
            cudaCheck(cudaMemcpyAsync(deviceDist, dist.data(), elements*sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
            cudaCheck(cudaMemcpyAsync(devicePath, path.data(), elements*sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        }
        if (n) {
            cudaCheck(cudaMalloc(&panel, TILE*n*sizeof(unsigned int)));
            cudaCheck(cudaMalloc(&rawPanel, TILE*n*sizeof(unsigned int)));
            cudaCheck(cudaMallocHost(&hostPanel, TILE*n*sizeof(unsigned int)));
        }
        cudaCheck(cudaStreamSynchronize(stream));
        if (!rank) printf("Computing shortest paths...\n");
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        double start = MPI_Wtime();
        int owner = 0;
        for (size_t first = 0; first < n; first += TILE) {
            while (starts[owner+1] <= first) ++owner;
            int width = static_cast<int>(std::min(size_t(TILE), n-first));
            size_t bytes = size_t(width)*n*sizeof(unsigned int);
            if (rank == owner) {
                cudaCheck(cudaMemcpyAsync(hostPanel, deviceDist+(first-starts[rank])*n, bytes, cudaMemcpyDeviceToHost, stream));
                cudaCheck(cudaStreamSynchronize(stream));
            }
            broadcast(hostPanel, size_t(width)*n, owner);
            if (rows) {
                cudaCheck(cudaMemcpyAsync(rawPanel, hostPanel, bytes, cudaMemcpyHostToDevice, stream));
                preparePanel<<<static_cast<unsigned int>(tiles), 256, 0, stream>>>(rawPanel, panel, n, first, width);
                cudaCheck(cudaGetLastError());
                prepareCoefficients<<<static_cast<unsigned int>((rows+127)/128), 128, 0, stream>>>(deviceDist, panel, coefficients, n, rows, first, width);
                cudaCheck(cudaGetLastError());
                updateRows<<<dim3(static_cast<unsigned int>(tiles), static_cast<unsigned int>((rows+31)/32)), dim3(32,8), 0, stream>>>(deviceDist, devicePath, panel, coefficients, n, rows, first, width);
                cudaCheck(cudaGetLastError());
                // The pinned staging panel is reused by the next collective.
                cudaCheck(cudaStreamSynchronize(stream));
            }
        }
        double elapsed = MPI_Wtime()-start, seconds = 0;
        mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (!rank) {
            printf("Computation time: %ld ms\n", static_cast<long>(seconds*1000));
            printf("Performance: %.3f GOPS\n", seconds > 0 ? double(n)*n*n/seconds/1e9 : 0);
        }
        if (validate || printResults) {
            if (elements) cudaCheck(cudaMemcpy(dist.data(), deviceDist, elements*sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (!rank) {
                dist.resize(n*n);
                for (int r = 1; r < ranks; ++r) {
                    size_t count = (starts[r+1]-starts[r])*n;
                    if (count) transfer(dist.data()+starts[r]*n, count, r, false);
                }
            } else transfer(dist.data(), elements, 0, true);
        }
        cudaCheck(cudaFree(deviceDist)); cudaCheck(cudaFree(devicePath));
        cudaCheck(cudaFree(coefficients)); cudaCheck(cudaFree(panel)); cudaCheck(cudaFree(rawPanel));
        if (hostPanel) cudaCheck(cudaFreeHost(hostPanel));
        cudaCheck(cudaStreamDestroy(stream));
        int status = 0;
        if (!rank) {
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(dist, n) ? 0 : 1;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
        mpiCheck(MPI_Finalize());
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Allocation/runtime failure: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
