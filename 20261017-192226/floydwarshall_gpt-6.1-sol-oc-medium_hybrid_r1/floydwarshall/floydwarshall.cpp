#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>
#include <climits>
#include <exception>
#include <limits>
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
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Tiles are block-cyclic over a two-dimensional MPI process grid. The path
// scratch matrix in the original benchmark was never read; omit its allocation
// and writes. All observable results are the exact integer distance matrix.
constexpr int TILE = 64;
constexpr int CELLS = TILE * TILE;
constexpr int THREADS = 256;

void cudaCheck(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, expression,
                cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call, __LINE__)

void mpiCheck(int result) {
    if (result != MPI_SUCCESS) {
        char text[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(result, text, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, text);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk counts so large matrices do not overflow MPI's int count interface.
void broadcast(unsigned int* data, size_t count, int root, MPI_Comm comm) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(data + offset, chunk, MPI_UNSIGNED, root, comm));
        offset += chunk;
    }
}

void transfer(unsigned int* data, size_t count, int peer, bool send) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send)
            mpiCheck(MPI_Send(data + offset, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else
            mpiCheck(MPI_Recv(data + offset, chunk, MPI_UNSIGNED, peer, 0,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

struct Buffer {
    unsigned int* device = nullptr;
    unsigned int* host = nullptr;
    size_t count;
    explicit Buffer(size_t elements) : count(elements) {
        size_t bytes = std::max(elements, size_t(1)) * sizeof(unsigned int);
        CUDA(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
        CUDA(cudaMallocHost(reinterpret_cast<void**>(&host), bytes));
    }
    ~Buffer() {
        CUDA(cudaFree(device));
        CUDA(cudaFreeHost(host));
    }
    void upload(cudaStream_t stream) {
        CUDA(cudaMemcpyAsync(device, host, count * sizeof(unsigned int),
                             cudaMemcpyHostToDevice, stream));
    }
    void download(cudaStream_t stream) {
        CUDA(cudaMemcpyAsync(host, device, count * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost, stream));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__global__ void pivotKernel(unsigned int* matrix, size_t tile) {
    __shared__ unsigned int d[CELLS];
    for (int t = threadIdx.x; t < CELLS; t += THREADS)
        d[t] = matrix[tile * CELLS + t];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int q = 0; q < CELLS / THREADS; ++q) {
            int t = threadIdx.x + q * THREADS;
            // With nonnegative weights and zero diagonal, the pivot row and
            // column cannot improve. Leave them read-only during this step.
            if (t / TILE != k && t % TILE != k)
                d[t] = min(d[t], d[(t / TILE) * TILE + k] + d[k * TILE + t % TILE]);
        }
        __syncthreads();
    }
    for (int t = threadIdx.x; t < CELLS; t += THREADS)
        matrix[tile * CELLS + t] = d[t];
}

// Update either the pivot row or the pivot column, and pack it for MPI.
template<bool Row>
__global__ void panelKernel(unsigned int* matrix, const unsigned int* pivot,
                            unsigned int* panel, int localColumns, int fixed,
                            int coordinate, int stride, int kBlock) {
    int block = blockIdx.x;
    size_t tile = Row ? size_t(fixed) * localColumns + block
                      : size_t(block) * localColumns + fixed;
    __shared__ unsigned int d[CELLS];
    __shared__ unsigned int p[CELLS];
    for (int t = threadIdx.x; t < CELLS; t += THREADS) {
        d[t] = matrix[tile * CELLS + t];
        p[t] = pivot[t];
    }
    __syncthreads();
    bool isPivot = block * stride + coordinate == kBlock;
    unsigned int value[CELLS / THREADS];
    #pragma unroll
    for (int q = 0; q < CELLS / THREADS; ++q)
        value[q] = d[threadIdx.x + q * THREADS];
    if (!isPivot) {
        // The pivot tile is already transitively closed: one min-plus product
        // completes the panel without a barrier for every intermediate vertex.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            #pragma unroll
            for (int q = 0; q < CELLS / THREADS; ++q) {
                int t = threadIdx.x + q * THREADS;
                unsigned int candidate = Row ? p[(t / TILE) * TILE + k] + d[k * TILE + t % TILE]
                                             : d[(t / TILE) * TILE + k] + p[k * TILE + t % TILE];
                value[q] = min(value[q], candidate);
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < CELLS / THREADS; ++q) {
        int t = threadIdx.x + q * THREADS;
        if (!isPivot) matrix[tile * CELLS + t] = value[q];
        panel[size_t(block) * CELLS + t] = value[q];
    }
}

__global__ void remainderKernel(unsigned int* matrix, const unsigned int* column,
                                const unsigned int* row, int localRows, int localColumns,
                                int processRow, int processColumn, int rows, int columns,
                                int kBlock) {
    // A grid-stride tile loop also supports matrices beyond CUDA grid limits.
    for (size_t tile = blockIdx.x; tile < size_t(localRows) * localColumns; tile += gridDim.x) {
        int i = tile / localColumns, j = tile % localColumns;
        if (i * rows + processRow == kBlock || j * columns + processColumn == kBlock)
            continue;
        __shared__ unsigned int a[CELLS];
        __shared__ unsigned int b[CELLS];
        unsigned int value[CELLS / THREADS];
        #pragma unroll
        for (int q = 0; q < CELLS / THREADS; ++q) {
            int t = threadIdx.x + q * THREADS;
            a[t] = column[size_t(i) * CELLS + t];
            b[t] = row[size_t(j) * CELLS + t];
            value[q] = matrix[tile * CELLS + t];
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            #pragma unroll
            for (int q = 0; q < CELLS / THREADS; ++q) {
                int t = threadIdx.x + q * THREADS;
                value[q] = min(value[q], a[(t / TILE) * TILE + k] + b[k * TILE + t % TILE]);
            }
        }
        #pragma unroll
        for (int q = 0; q < CELLS / THREADS; ++q)
            matrix[tile * CELLS + threadIdx.x + q * THREADS] = value[q];
        __syncthreads();
    }
}

int localBlocks(int total, int coordinate, int stride) {
    return total / stride + (coordinate < total % stride);
}

// OpenMP packing preserves the original row-major order, including partial tiles.
void pack(std::vector<unsigned int>& dist, unsigned int* tiles, size_t n,
          int localRows, int localColumns, int pr, int pc, int rows, int columns, bool unpack) {
    #pragma omp parallel for schedule(static)
    for (size_t tile = 0; tile < size_t(localRows) * localColumns; ++tile) {
        size_t baseRow = (tile / localColumns * rows + pr) * TILE;
        size_t baseColumn = (tile % localColumns * columns + pc) * TILE;
        for (int i = 0; i < TILE; ++i) {
            for (int j = 0; j < TILE; ++j) {
                size_t r = baseRow + i, c = baseColumn + j;
                size_t t = tile * CELLS + i * TILE + j;
                if (unpack) {
                    if (r < n && c < n) dist[r * n + c] = tiles[t];
                } else {
                    tiles[t] = r < n && c < n ? dist[r * n + c] : (r == c ? 0 : INF);
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n, bool collect,
                   int rank, int processes, const int* dims, double& seconds) {
    int pr = rank / dims[1], pc = rank % dims[1];
    int blocks = (n + TILE - 1) / TILE;
    int nr = localBlocks(blocks, pr, dims[0]);
    int nc = localBlocks(blocks, pc, dims[1]);
    MPI_Comm rowComm, columnComm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &rowComm));
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &columnComm));
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Buffer matrix(size_t(nr) * nc * CELLS), pivot(CELLS);
    Buffer row(size_t(nc) * CELLS), column(size_t(nr) * CELLS);

    // Root generates the identical rand_r sequence; distribute only local tiles.
    if (rank == 0) {
        for (int p = 0; p < processes; ++p) {
            int r = p / dims[1], c = p % dims[1];
            int rcount = localBlocks(blocks, r, dims[0]);
            int ccount = localBlocks(blocks, c, dims[1]);
            if (p == 0) {
                pack(dist, matrix.host, n, rcount, ccount, r, c, dims[0], dims[1], false);
            } else {
                std::vector<unsigned int> staging(size_t(rcount) * ccount * CELLS);
                pack(dist, staging.data(), n, rcount, ccount, r, c, dims[0], dims[1], false);
                transfer(staging.data(), staging.size(), p, true);
            }
        }
        // The full result is needed only for requested output or validation.
        if (!collect) std::vector<unsigned int>().swap(dist);
    } else {
        transfer(matrix.host, matrix.count, 0, false);
    }
    matrix.upload(stream);
    CUDA(cudaStreamSynchronize(stream));
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    for (int k = 0; k < blocks; ++k) {
        int ownerRow = k % dims[0], ownerColumn = k % dims[1];
        int owner = ownerRow * dims[1] + ownerColumn;
        const unsigned int* pivotDevice = pivot.device;
        if (rank == owner) {
            size_t tile = size_t(k / dims[0]) * nc + k / dims[1];
            pivotKernel<<<1, THREADS, 0, stream>>>(matrix.device, tile);
            CUDA(cudaGetLastError());
            pivotDevice = matrix.device + tile * CELLS;
            if (processes > 1) {
                CUDA(cudaMemcpyAsync(pivot.host, pivotDevice, CELLS * sizeof(unsigned int),
                                     cudaMemcpyDeviceToHost, stream));
                CUDA(cudaStreamSynchronize(stream));
            }
        }
        if (processes > 1) {
            broadcast(pivot.host, CELLS, owner, MPI_COMM_WORLD);
            if (rank != owner) pivot.upload(stream);
        }
        if (pr == ownerRow && nc) {
            panelKernel<true><<<nc, THREADS, 0, stream>>>(matrix.device, pivotDevice,
                row.device, nc, k / dims[0], pc, dims[1], k);
            CUDA(cudaGetLastError());
            if (dims[0] > 1) row.download(stream);
        }
        if (pc == ownerColumn && nr) {
            panelKernel<false><<<nr, THREADS, 0, stream>>>(matrix.device, pivotDevice,
                column.device, nc, k / dims[1], pr, dims[0], k);
            CUDA(cudaGetLastError());
            if (dims[1] > 1) column.download(stream);
        }
        if (processes > 1) CUDA(cudaStreamSynchronize(stream));
        // Independent panel collectives progress together, without CUDA-aware MPI.
        // Each panel is O(n/sqrt(P)); no rank replicates the complete GPU matrix.
        if (row.count <= INT_MAX && column.count <= INT_MAX) {
            MPI_Request requests[2];
            mpiCheck(MPI_Ibcast(row.host, dims[0] > 1 ? static_cast<int>(row.count) : 0, MPI_UNSIGNED,
                               ownerRow, columnComm, &requests[0]));
            mpiCheck(MPI_Ibcast(column.host, dims[1] > 1 ? static_cast<int>(column.count) : 0, MPI_UNSIGNED,
                               ownerColumn, rowComm, &requests[1]));
            mpiCheck(MPI_Waitall(2, requests, MPI_STATUSES_IGNORE));
        } else {
            if (dims[0] > 1) broadcast(row.host, row.count, ownerRow, columnComm);
            if (dims[1] > 1) broadcast(column.host, column.count, ownerColumn, rowComm);
        }
        if (pr != ownerRow) row.upload(stream);
        if (pc != ownerColumn) column.upload(stream);
        if (nr && nc) {
            unsigned int grid = static_cast<unsigned int>(std::min(size_t(nr) * nc, size_t(65535)));
            remainderKernel<<<grid, THREADS, 0, stream>>>(matrix.device, column.device, row.device,
                nr, nc, pr, pc, dims[0], dims[1], k);
            CUDA(cudaGetLastError());
        }
    }
    CUDA(cudaStreamSynchronize(stream));
    double elapsed = MPI_Wtime() - start;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (collect) {
        matrix.download(stream);
        CUDA(cudaStreamSynchronize(stream));
        if (rank == 0) {
            pack(dist, matrix.host, n, nr, nc, pr, pc, dims[0], dims[1], true);
            for (int p = 1; p < processes; ++p) {
                int r = p / dims[1], c = p % dims[1];
                int rcount = localBlocks(blocks, r, dims[0]);
                int ccount = localBlocks(blocks, c, dims[1]);
                std::vector<unsigned int> staging(size_t(rcount) * ccount * CELLS);
                transfer(staging.data(), staging.size(), p, false);
                pack(dist, staging.data(), n, rcount, ccount, r, c, dims[0], dims[1], true);
            }
        } else {
            transfer(matrix.host, matrix.count, 0, true);
        }
    }
    CUDA(cudaStreamDestroy(stream));
    mpiCheck(MPI_Comm_free(&rowComm));
    mpiCheck(MPI_Comm_free(&columnComm));
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
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 1;
    int rank = 0, processes = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &processes));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Rank-local GPU selection works across nodes and with per-rank GPU masks.
    MPI_Comm nodeComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    mpiCheck(MPI_Comm_rank(nodeComm, &localRank));
    mpiCheck(MPI_Comm_size(nodeComm, &localSize));
    // Avoid one full host thread team per rank unless explicitly configured.
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    mpiCheck(MPI_Comm_free(&nodeComm));
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || !argument[0] || *end ||
                value > UINT_MAX || value > std::numeric_limits<size_t>::max() ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / value)) {
                if (rank == 0) fprintf(stderr, "Invalid node count: %s\n", argument);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(value);
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
    
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(processes, 2, dims));
    int status = 0;
    try {
        std::vector<unsigned int> dist;
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI grid: %d x %d, OpenMP threads per rank: %d, CUDA tile: %d\n",
                   dims[0], dims[1], omp_get_max_threads(), TILE);
            printf("Initializing graph...\n");
            dist.resize(numNodes * numNodes);
            initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
            printf("Computing shortest paths...\n");
        }
        double seconds = 0;
        floydWarshall(dist, numNodes, validate || printResults, rank, processes, dims, seconds);
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            double ops = double(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(dist, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return status;
}
