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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static inline void dieMPI(const char* msg) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::fprintf(stderr, "%s\n", msg);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "CUDA error in %s: %s", what, cudaGetErrorString(err));
        dieMPI(buf);
    }
}

static inline void decomposeRows(const size_t N, const int size, const int rank,
                                size_t& row_start, size_t& local_rows) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    local_rows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    row_start = base * static_cast<size_t>(rank) + std::min(rem, static_cast<size_t>(rank));
}

static inline int ownerOfRow(const size_t N, const int size, const size_t i) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t cutoff = (base + 1) * rem;
    if (i < cutoff) {
        return static_cast<int>(i / (base + 1));
    }
    return static_cast<int>(rem + (i - cutoff) / base);
}

static inline size_t localIndexOfRow(const size_t N, const int size, const size_t i) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t cutoff = (base + 1) * rem;
    if (i < cutoff) {
        return i % (base + 1);
    }
    return (i - cutoff) % base;
}

static void initMatrixFull(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t ii = 0; ii < static_cast<ptrdiff_t>(N); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static void initMatrixRows(std::vector<double>& mat, const size_t N,
                           const size_t row_start, const size_t local_rows) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t lii = 0; lii < static_cast<ptrdiff_t>(local_rows); ++lii) {
        const size_t li = static_cast<size_t>(lii);
        const size_t i = row_start + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CUDA tiled GEMM: C[MxN] = A[MxK] * B[KxN]
template <int TILE>
__global__ void matmul_tiled_kernel(const double* __restrict__ A,
                                    const double* __restrict__ B,
                                    double* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    double acc = 0.0;

    for (int t = 0; t < K; t += TILE) {
        const int a_col = t + threadIdx.x;
        const int b_row = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

static void gpuMatMul(const std::vector<double>& A_local,
                      const std::vector<double>& B,
                      std::vector<double>& C_local,
                      const size_t local_rows,
                      const size_t N) {
    const int M = static_cast<int>(local_rows);
    const int K = static_cast<int>(N);
    const int NN = static_cast<int>(N);

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    cudaCheck(cudaMalloc(&dA, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double)), "cudaMalloc(dA)");
    cudaCheck(cudaMalloc(&dB, static_cast<size_t>(K) * static_cast<size_t>(NN) * sizeof(double)), "cudaMalloc(dB)");
    cudaCheck(cudaMalloc(&dC, static_cast<size_t>(M) * static_cast<size_t>(NN) * sizeof(double)), "cudaMalloc(dC)");

    cudaCheck(cudaMemcpy(dA, A_local.data(), static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double), cudaMemcpyHostToDevice),
              "cudaMemcpy(A)");
    cudaCheck(cudaMemcpy(dB, B.data(), static_cast<size_t>(K) * static_cast<size_t>(NN) * sizeof(double), cudaMemcpyHostToDevice),
              "cudaMemcpy(B)");

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((NN + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel<TILE><<<grid, block>>>(dA, dB, dC, M, NN, K);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaCheck(cudaMemcpy(C_local.data(), dC, static_cast<size_t>(M) * static_cast<size_t>(NN) * sizeof(double), cudaMemcpyDeviceToHost),
              "cudaMemcpy(C)");

    cudaCheck(cudaFree(dA), "cudaFree(dA)");
    cudaCheck(cudaFree(dB), "cudaFree(dB)");
    cudaCheck(cudaFree(dC), "cudaFree(dC)");
}

static bool validateDistributed(const std::vector<double>& C_local,
                                const size_t N,
                                const size_t row_start,
                                const size_t local_rows,
                                const int rank,
                                const int size) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    bool ok = true;
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            const int owner = ownerOfRow(N, size, i);
            double actual = 0.0;
            if (rank == owner) {
                const size_t li = localIndexOfRow(N, size, i);
                if (!(i >= row_start && i < row_start + local_rows)) {
                    dieMPI("Internal error: owner row mapping inconsistent");
                }
                actual = C_local[li * N + j];
            }

            MPI_Bcast(&actual, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            if (rank == 0) {
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                }
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    std::printf(
                        "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, relError);
                    ok = false;
                }
            }
        }
    }

    int ok_i = ok ? 1 : 0;
    MPI_Bcast(&ok_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok_i == 1;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation (gathers to rank 0)\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices found; this benchmark requires CUDA.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    size_t row_start = 0, local_rows = 0;
    decomposeRows(N, size, rank, row_start, local_rows);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("MPI ranks: %d\n", size);
        std::printf("OpenMP threads (rank 0): %d\n", omp_get_max_threads());
        std::printf("CUDA devices visible: %d\n", deviceCount);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate per-rank matrices: A_local (rows owned by rank), full B, and C_local.
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N);

    if (rank == 0) {
        std::printf("Initializing matrices...\n");
    }
    initMatrixRows(A_local, N, row_start, local_rows);
    initMatrixFull(B, N);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    gpuMatMul(A_local, B, C_local, local_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_s = t1 - t0;
    double max_s = 0.0;
    MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / max_s / 1e9;
        std::printf("Computation time (max over ranks): %.3f ms\n", max_s * 1e3);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> C;
        std::vector<int> recvcounts;
        std::vector<int> displs;

        if (rank == 0) {
            C.resize(N * N);
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                size_t rs = 0, lr = 0;
                decomposeRows(N, size, r, rs, lr);
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(lr * N);
                displs[static_cast<size_t>(r)] = static_cast<int>(rs * N);
            }
        }

        MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(C, "MatrixC");
        }
    }

    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateDistributed(C_local, N, row_start, local_rows, rank, size);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
