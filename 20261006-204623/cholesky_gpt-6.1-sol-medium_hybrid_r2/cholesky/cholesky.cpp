#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 128;
constexpr int generationBlock = 1024;

void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void check(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS operation failed (" + std::to_string(int(code)) + ")");
}
void mpiCheck(int code) {
    if (code != MPI_SUCCESS) throw std::runtime_error("MPI operation failed");
}

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) { check(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double))); }
    ~Pinned() { cudaFreeHost(p); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
};

// Column-major local rows; every rank owns complete, cyclically assigned row blocks.
__global__ void selectRows(const double* global, double* local, const int* rows,
                           int m, int n, int width) {
    const size_t total = size_t(m) * width;
    for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         x < total; x += size_t(gridDim.x) * blockDim.x)
        local[x] = global[rows[x % m] + (x / m) * n];
}
__global__ void diagonalShift(double* a, const int* rows, int m, int n) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m; r += gridDim.x * blockDim.x)
        a[r + size_t(rows[r]) * m] += n;
}
// One warp per row avoids the general BLAS triangular-solve setup cost on
// small panels. Each warp cooperatively reduces dot products in FP64.
__global__ void solveSmallPanel(double* a, int lda, int active,
                                const double* diagonal, int width) {
    __shared__ double solved[8][blockSize];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row = blockIdx.x * 8 + warp;
    if (row >= active) return;
    for (int j = 0; j < width; ++j) {
        double sum = 0.0;
        for (int k = lane; k < j; k += 32)
            sum += diagonal[j + k * width] * solved[warp][k];
        for (int offset = 16; offset; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            const double value = (a[row + size_t(j) * lda] - sum) / diagonal[j + j * width];
            solved[warp][j] = value;
            a[row + size_t(j) * lda] = value;
        }
        __syncwarp();
    }
}
__global__ void zeroUpper(double* a, const int* rows, int m, int n) {
    for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         x < size_t(m) * n; x += size_t(gridDim.x) * blockDim.x)
        if (int(x / m) > rows[x % m]) a[x] = 0.0;
}
__global__ void errors(const double* a, const double* original, size_t count, double* result) {
    double absolute = 0.0, relative = 0.0;
    for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         x < count; x += size_t(gridDim.x) * blockDim.x) {
        double e = fabs(a[x] - original[x]);
        if (!isfinite(e)) e = CUDART_INF;
        absolute = fmax(absolute, e);
        relative = fmax(relative, e / (fabs(original[x]) + 1e-10));
    }
    // Positive IEEE doubles have the same ordering as their unsigned bit patterns.
    atomicMax(reinterpret_cast<unsigned long long*>(result), __double_as_longlong(absolute));
    atomicMax(reinterpret_cast<unsigned long long*>(result + 1), __double_as_longlong(relative));
}

struct DistributedMatrix {
    int n, rank, ranks;
    std::vector<std::vector<int>> ownership;
    std::vector<int> rows;
    int m, lda;
    Device<int> deviceRows;
    Device<double> a, panel, diagonal;
    Pinned send, received, hostPanel, hostDiagonal;
    cudaStream_t stream{};
    cublasHandle_t blas{};

    static std::vector<std::vector<int>> partition(int n, int ranks) {
        std::vector<std::vector<int>> result(ranks);
        for (int i = 0; i < n; ++i) result[(i / blockSize) % ranks].push_back(i);
        return result;
    }
    DistributedMatrix(int size, int r, int p)
        : n(size), rank(r), ranks(p), ownership(partition(n, p)), rows(ownership[r]),
          m(int(rows.size())), lda(std::max(1, m)), deviceRows(m),
          a(size_t(lda) * n), panel(size_t(n) * generationBlock),
          diagonal(blockSize * blockSize), send(size_t(lda) * blockSize),
          received(size_t(n) * blockSize), hostPanel(size_t(n) * generationBlock),
          hostDiagonal(blockSize * blockSize) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cublasCreate(&blas));
        check(cublasSetStream(blas, stream));
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        if (m) check(cudaMemcpyAsync(deviceRows.p, rows.data(), m * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    ~DistributedMatrix() {
        cudaStreamSynchronize(stream);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
    int firstRow(int start) const {
        return int(std::lower_bound(rows.begin(), rows.end(), start) - rows.begin());
    }
    void sync() { check(cudaStreamSynchronize(stream)); }

    void generate() {
        // Record the actual rand_r state at each row boundary, preserving the original
        // random matrix on every libc without assuming a particular RNG recurrence.
        std::vector<unsigned int> seeds(n);
        if (rank == 0) {
            unsigned int seed = 42;
            for (int i = 0; i < n; ++i) {
                seeds[i] = seed;
                for (int j = 0; j < n; ++j) (void)rand_r(&seed);
            }
        }
        mpiCheck(MPI_Bcast(seeds.data(), n, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
        Device<double> localB(size_t(lda) * generationBlock);
        check(cudaMemsetAsync(a.p, 0, size_t(lda) * n * sizeof(double), stream));
        const double one = 1.0;
        for (int k = 0; k < n; k += generationBlock) {
            const int width = std::min(generationBlock, n - k);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                unsigned int seed = seeds[i];
                for (int j = 0; j < width; ++j)
                    hostPanel.p[i + size_t(j) * n] = rand_r(&seed) / double(RAND_MAX) - 0.5;
                seeds[i] = seed;
            }
            check(cudaMemcpyAsync(panel.p, hostPanel.p, size_t(n) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
            if (m) {
                selectRows<<<256, 256, 0, stream>>>(panel.p, localB.p, deviceRows.p, m, n, width);
                check(cudaGetLastError());
                check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, width,
                      &one, localB.p, lda, panel.p, n, &one, a.p, lda));
            }
            sync(); // hostPanel is reused by OpenMP in the next iteration.
        }
        if (m) diagonalShift<<<256, 256, 0, stream>>>(a.p, deviceRows.p, m, n);
        check(cudaGetLastError());
        sync();
    }

    // Host staging works with ordinary MPI, including MPI installations without
    // CUDA awareness. Only the active panel is communicated, never the full matrix.
    void gatherPanel(int column, int width, int start) {
        const int first = firstRow(start), active = m - first;
        if (active)
            check(cudaMemcpy2DAsync(send.p, active * sizeof(double), a.p + first + size_t(column) * lda,
                  lda * sizeof(double), active * sizeof(double), width, cudaMemcpyDeviceToHost, stream));
        sync();
        std::vector<int> counts(ranks), offsets(ranks), starts(ranks);
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            starts[r] = int(std::lower_bound(ownership[r].begin(), ownership[r].end(), start) - ownership[r].begin());
            counts[r] = (int(ownership[r].size()) - starts[r]) * width;
            offsets[r] = total;
            total += counts[r];
        }
        mpiCheck(MPI_Allgatherv(send.p, active * width, MPI_DOUBLE, received.p,
                              counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD));
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < width; ++c) {
            for (int r = 0; r < ranks; ++r) {
                const int count = int(ownership[r].size()) - starts[r];
                for (int i = 0; i < count; ++i)
                    hostPanel.p[ownership[r][starts[r] + i] + size_t(c) * n] =
                        received.p[offsets[r] + size_t(c) * count + i];
            }
        }
        // Rows preceding start are unused in the trailing update.
        check(cudaMemcpyAsync(panel.p, hostPanel.p, size_t(n) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    bool factor() {
        const double one = 1.0, minusOne = -1.0;
        for (int k = 0; k < n; k += blockSize) {
            const int width = std::min(blockSize, n - k), owner = (k / blockSize) % ranks;
            int status = 0;
            if (rank == owner) {
                const int first = firstRow(k);
                check(cudaMemcpy2DAsync(hostDiagonal.p, width * sizeof(double), a.p + first + size_t(k) * lda,
                      lda * sizeof(double), width * sizeof(double), width, cudaMemcpyDeviceToHost, stream));
                sync();
                // A small cache-resident panel is faster on the CPU than a sequence
                // of dependent GPU launches. SIMD updates use contiguous columns.
                for (int j = 0; j < width; ++j) {
                    double& pivot = hostDiagonal.p[j + size_t(j) * width];
                    if (!(pivot > 0.0) || !std::isfinite(pivot)) {
                        status = j + 1;
                        break;
                    }
                    pivot = std::sqrt(pivot);
                    for (int i = j + 1; i < width; ++i)
                        hostDiagonal.p[i + size_t(j) * width] /= pivot;
                    for (int c = j + 1; c < width; ++c) {
                        const double scale = hostDiagonal.p[c + size_t(j) * width];
                        #pragma omp simd
                        for (int i = c; i < width; ++i)
                            hostDiagonal.p[i + size_t(c) * width] -=
                                hostDiagonal.p[i + size_t(j) * width] * scale;
                    }
                }
            }
            mpiCheck(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (status != 0) {
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + status - 1);
                return false;
            }
            mpiCheck(MPI_Bcast(hostDiagonal.p, width * width, MPI_DOUBLE, owner, MPI_COMM_WORLD));
            check(cudaMemcpyAsync(diagonal.p, hostDiagonal.p, size_t(width) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
            if (rank == owner) {
                const int first = firstRow(k);
                check(cudaMemcpy2DAsync(a.p + first + size_t(k) * lda, lda * sizeof(double), diagonal.p,
                      width * sizeof(double), width * sizeof(double), width, cudaMemcpyDeviceToDevice, stream));
            }
            const int first = firstRow(k + width), active = m - first;
            if (active > 0 && active <= 1024) {
                solveSmallPanel<<<(active + 7) / 8, 256, 0, stream>>>(
                    a.p + first + size_t(k) * lda, lda, active, diagonal.p, width);
                check(cudaGetLastError());
            } else if (active) {
                check(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, active, width, &one,
                      diagonal.p, width, a.p + first + size_t(k) * lda, lda));
            }
            if (k + width < n) {
                gatherPanel(k, width, k + width);
                if (active)
                    check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, active, n - k - width, width,
                          &minusOne, a.p + first + size_t(k) * lda, lda, panel.p + k + width, n,
                          &one, a.p + first + size_t(k + width) * lda, lda));
            }
        }
        if (m) zeroUpper<<<256, 256, 0, stream>>>(a.p, deviceRows.p, m, n);
        check(cudaGetLastError());
        sync();
        return true;
    }

    bool validate(const double* original) {
        Device<double> reconstructed(size_t(lda) * n), maxima(2);
        check(cudaMemsetAsync(reconstructed.p, 0, size_t(lda) * n * sizeof(double), stream));
        const double one = 1.0;
        for (int k = 0; k < n; k += blockSize) {
            const int width = std::min(blockSize, n - k);
            gatherPanel(k, width, 0);
            if (m)
                check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, width, &one,
                      a.p + size_t(k) * lda, lda, panel.p, n, &one, reconstructed.p, lda));
        }
        check(cudaMemsetAsync(maxima.p, 0, 2 * sizeof(double), stream));
        errors<<<256, 256, 0, stream>>>(reconstructed.p, original, size_t(m) * n, maxima.p);
        check(cudaGetLastError());
        double local[2], global[2];
        check(cudaMemcpyAsync(local, maxima.p, sizeof(local), cudaMemcpyDeviceToHost, stream));
        sync();
        mpiCheck(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (rank == 0) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
        }
        return std::isfinite(global[1]) && global[1] <= 1e-6;
    }

    void printResults() {
        // Gather one row block at a time to avoid MPI's int count limit and keep
        // the full row-major result exclusively on rank zero, only when requested.
        std::vector<double> result;
        if (rank == 0) result.resize(size_t(n) * n);
        Pinned block(size_t(blockSize) * n);
        for (int k = 0; k < n; k += blockSize) {
            const int width = std::min(blockSize, n - k), owner = (k / blockSize) % ranks;
            if (rank == owner) {
                check(cudaMemcpy2DAsync(block.p, width * sizeof(double), a.p + firstRow(k),
                      lda * sizeof(double), width * sizeof(double), n, cudaMemcpyDeviceToHost, stream));
                sync();
                if (owner != 0) mpiCheck(MPI_Send(block.p, width * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD));
            }
            if (rank == 0) {
                if (owner != 0) mpiCheck(MPI_Recv(block.p, width * n, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < n; ++j)
                        result[size_t(k + i) * n + j] = block.p[i + size_t(j) * width];
            }
        }
        if (rank == 0) print_results(result, "CholeskyL");
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int exitCode = 0;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        int n = 512;
        bool validation = false, results = false, help = false;
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "-n") && i + 1 < argc) {
                char* end = nullptr;
                errno = 0;
                long long value = strtoll(argv[++i], &end, 10);
                if (errno || end == argv[i] || *end || value < 0 || value > INT_MAX / blockSize)
                    throw std::runtime_error("Invalid matrix size (must be nonnegative and fit MPI panel counts)");
                n = int(value);
            } else if (!strcmp(argv[i], "-v")) validation = true;
            else if (!strcmp(argv[i], "-r")) results = true;
            else if (!strcmp(argv[i], "-h")) help = true;
            else throw std::runtime_error(std::string("Unknown option: ") + argv[i]);
        }
        if (help) {
            if (rank == 0) printUsage(argv[0]);
        } else {
            MPI_Comm local;
            mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
            int localRank = 0, localRanks = 1, devices = 0;
            mpiCheck(MPI_Comm_rank(local, &localRank));
            mpiCheck(MPI_Comm_size(local, &localRanks));
            check(cudaGetDeviceCount(&devices));
            if (devices == 0) throw std::runtime_error("A CUDA GPU is required on every MPI rank");
            check(cudaSetDevice(localRank % devices));
            mpiCheck(MPI_Comm_free(&local));
            // Panel packing is bandwidth bound: a small team avoids oversubscribing
            // many-core nodes. Explicit OMP_NUM_THREADS always takes precedence.
            if (!getenv("OMP_NUM_THREADS"))
                omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / localRanks)));
            if (rank == 0) {
                printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
                printf("Validation: %s\nMPI ranks: %d; OpenMP threads per rank: %d\n",
                       validation ? "enabled" : "disabled", ranks, omp_get_max_threads());
                printf("Generating positive definite matrix...\n");
            }
            DistributedMatrix matrix(n, rank, ranks);
            matrix.generate();
            Device<double> original(validation ? size_t(matrix.lda) * n : 0);
            if (validation) {
                check(cudaMemcpyAsync(original.p, matrix.a.p, size_t(matrix.lda) * n * sizeof(double),
                      cudaMemcpyDeviceToDevice, matrix.stream));
                matrix.sync();
            }
            if (rank == 0) printf("Computing Cholesky decomposition...\n");
            mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
            const double start = MPI_Wtime();
            const bool success = matrix.factor();
            const double elapsed = MPI_Wtime() - start;
            double seconds = 0.0;
            mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            if (!success) {
                if (rank == 0) printf("Cholesky decomposition failed\n");
                exitCode = 1;
            } else {
                if (rank == 0) {
                    printf("Computation time: %.3f ms\n", seconds * 1000.0);
                    printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0 * std::max(seconds, 1e-12) * 1e9));
                }
                if (results) matrix.printResults();
                if (validation) {
                    if (rank == 0) printf("Validating result...\n");
                    const bool valid = matrix.validate(original.p);
                    if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    exitCode = valid ? 0 : 1;
                }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        // A failed allocation or GPU operation on one rank must not strand peers
        // inside a collective. Abort also handles unexpected asymmetric failures.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exitCode;
}
