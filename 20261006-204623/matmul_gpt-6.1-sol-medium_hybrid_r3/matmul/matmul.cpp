#include <algorithm>
#include <cerrno>
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

namespace {
int worldRank = 0;

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "Rank %d: MPI error: %.*s\n", worldRank, length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA error: %s\n", worldRank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// Balanced contiguous intervals, including empty intervals when ranks exceed N.
size_t boundary(size_t N, int part, int parts) {
    return (N / parts) * part + std::min(static_cast<size_t>(part), N % parts);
}

// A is rows x N; B is N x columns. Each block computes 32 x 64 outputs,
// reusing shared-memory panels and keeping 4 x 4 outputs per thread in registers.
// The k loop is sequential and contraction is disabled to preserve the original
// double-precision multiply/add order, including for incomplete tiles.
__global__ void multiply(const double* __restrict__ A,
                         const double* __restrict__ B,
                         double* __restrict__ C,
                         size_t rows, size_t columns, size_t N) {
    __shared__ double aTile[32][17];
    __shared__ double bTile[16][65];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int lane = ty * 16 + tx;
    const size_t columnTiles = (columns + 63) / 64;
    const size_t tiles = ((rows + 31) / 32) * columnTiles;
    for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const size_t rowBase = (tile / columnTiles) * 32;
        const size_t columnBase = (tile % columnTiles) * 64;
        double sums[4][4] = {};
        for (size_t base = 0; base < N; base += 16) {
            for (int index = lane; index < 32 * 16; index += 128) {
                const int r = index / 16, k = index % 16;
                aTile[r][k] = rowBase + r < rows && base + k < N
                    ? A[(rowBase + r) * N + base + k] : 0.0;
            }
            for (int index = lane; index < 16 * 64; index += 128) {
                const int k = index / 64, c = index % 64;
                bTile[k][c] = base + k < N && columnBase + c < columns
                    ? B[(base + k) * columns + columnBase + c] : 0.0;
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (base + k < N) {
                    double a[4], b[4];
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) a[r] = aTile[ty + 8 * r][k];
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) b[c] = bTile[k][tx + 16 * c];
                    #pragma unroll
                    for (int r = 0; r < 4; ++r)
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) sums[r][c] += a[r] * b[c];
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const size_t row = rowBase + ty + 8 * r;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const size_t column = columnBase + tx + 16 * c;
                if (row < rows && column < columns)
                    C[row * columns + column] = sums[r][c];
            }
        }
        __syncthreads();
    }
}

bool validateResult(const std::vector<double>& C, size_t N,
                    size_t rowStart, size_t rows, size_t columnStart, size_t columns) {
    bool valid = true;
    // Preserve the original checks and also check every rank's block edges.
    const size_t points[] = {0, 1 % N, 2 % N, 3 % N, 4 % N,
                            rowStart, rows ? rowStart + rows - 1 : rowStart};
    const size_t columnPoints[] = {0, 1 % N, 2 % N, 3 % N, 4 % N,
                                  columnStart, columns ? columnStart + columns - 1 : columnStart};
    for (size_t i : points) {
        if (i < rowStart || i >= rowStart + rows) continue;
        for (size_t j : columnPoints) {
            if (j < columnStart || j >= columnStart + columns) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[(i - rowStart) * columns + j - columnStart];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f\n",
                        i, j, expected, actual);
                valid = false;
            }
        }
    }
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Chunked transfers avoid MPI's int count limit. Only -r needs a global matrix.
void gatherResults(const std::vector<double>& local, size_t N, const int dims[2],
                   int rank, int ranks) {
    std::vector<double> result;
    if (rank == 0) result.resize(N * N);
    for (int source = 0; source < ranks; ++source) {
        const size_t r0 = boundary(N, source / dims[1], dims[0]);
        const size_t nr = boundary(N, source / dims[1] + 1, dims[0]) - r0;
        const size_t c0 = boundary(N, source % dims[1], dims[1]);
        const size_t nc = boundary(N, source % dims[1] + 1, dims[1]) - c0;
        for (size_t r = 0; r < nr; ++r) {
            for (size_t c = 0; c < nc;) {
                const int count = static_cast<int>(std::min(nc - c, static_cast<size_t>(INT_MAX)));
                if (rank == 0) {
                    double* target = result.data() + (r0 + r) * N + c0 + c;
                    if (source == 0) std::copy_n(local.data() + r * nc + c, count, target);
                    else mpiCheck(MPI_Recv(target, count, MPI_DOUBLE, source, 0,
                                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                } else if (rank == source) {
                    mpiCheck(MPI_Send(local.data() + r * nc + c, count, MPI_DOUBLE,
                                      0, 0, MPI_COMM_WORLD));
                }
                c += count;
            }
        }
    }
    if (rank == 0) print_results(result, "MatrixC");
}

int run(int argc, char** argv) {
    int ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (worldRank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        if (worldRank == 0) fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node));
    int localRank, localRanks, devices = 0;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    mpiCheck(MPI_Comm_size(node, &localRanks));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> assigned;
    // Honor CUDA_VISIBLE_DEVICES, use all GPUs when there are fewer ranks than
    // GPUs, and share GPUs when an explicitly oversubscribed launch is used.
    if (localRanks >= devices) assigned.push_back(localRank % devices);
    else for (int d = localRank; d < devices; d += localRanks) assigned.push_back(d);
    mpiCheck(MPI_Comm_free(&node));

    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(ranks, 2, dims));
    const size_t rowStart = boundary(N, worldRank / dims[1], dims[0]);
    const size_t rows = boundary(N, worldRank / dims[1] + 1, dims[0]) - rowStart;
    const size_t columnStart = boundary(N, worldRank % dims[1], dims[1]);
    const size_t columns = boundary(N, worldRank % dims[1] + 1, dims[1]) - columnStart;
    std::vector<double> A(rows * N), B(N * columns), C(rows * columns);
    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < N; ++k)
                A[i * N + k] = getPseudoRndValue(N, rowStart + i, k);
        #pragma omp for schedule(static)
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < columns; ++j)
                B[k * columns + j] = getPseudoRndValue(N, k, columnStart + j);
    }

    // Create contexts and load/JIT the kernel before timing, like host setup.
    const int workers = static_cast<int>(assigned.size());
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int worker = 0; worker < workers; ++worker) {
        cudaCheck(cudaSetDevice(assigned[worker]));
        cudaCheck(cudaFree(nullptr));
        multiply<<<1, dim3(16, 8)>>>(nullptr, nullptr, nullptr, 0, 0, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
    }
    if (worldRank == 0) printf("Computing matrix multiplication...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int worker = 0; worker < workers; ++worker) {
        const size_t begin = boundary(rows, worker, workers);
        const size_t count = boundary(rows, worker + 1, workers) - begin;
        if (count == 0 || columns == 0) continue;
        cudaCheck(cudaSetDevice(assigned[worker]));
        double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
        cudaCheck(cudaMalloc(&deviceA, count * N * sizeof(double)));
        cudaCheck(cudaMalloc(&deviceB, B.size() * sizeof(double)));
        cudaCheck(cudaMalloc(&deviceC, count * columns * sizeof(double)));
        cudaCheck(cudaMemcpy(deviceA, A.data() + begin * N, count * N * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(deviceB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
        const size_t tiles = ((count + 31) / 32) * ((columns + 63) / 64);
        multiply<<<static_cast<unsigned>(std::min(tiles, size_t(65535))), dim3(16, 8)>>>(
            deviceA, deviceB, deviceC, count, columns, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(C.data() + begin * columns, deviceC,
                             count * columns * sizeof(double), cudaMemcpyDeviceToHost));
        cudaCheck(cudaFree(deviceC));
        cudaCheck(cudaFree(deviceB));
        cudaCheck(cudaFree(deviceA));
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (worldRank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }
    if (printResults) gatherResults(C, N, dims, worldRank, ranks);
    if (validate) {
        if (worldRank == 0) printf("Validating result...\n");
        int valid = validateResult(C, N, rowStart, rows, columnStart, columns) ? 1 : 0;
        int allValid = 0;
        mpiCheck(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (worldRank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return result;
}
