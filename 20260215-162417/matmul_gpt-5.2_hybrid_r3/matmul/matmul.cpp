#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define MPI_CHECK(call)                                                                 \
    do {                                                                                \
        const int err_ = (call);                                                        \
        if (err_ != MPI_SUCCESS) {                                                      \
            char msg[MPI_MAX_ERROR_STRING];                                             \
            int len = 0;                                                                \
            MPI_Error_string(err_, msg, &len);                                          \
            fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len, msg); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                               \
        }                                                                               \
    } while (0)

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        const cublasStatus_t st_ = (call);                                               \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)st_); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static inline size_t blockStart(const size_t N, const int rank, const int size) {
    return (N * static_cast<size_t>(rank)) / static_cast<size_t>(size);
}
static inline size_t blockEnd(const size_t N, const int rank, const int size) {
    return (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(size);
}

static void initLocalA(std::vector<double>& A, const size_t N, const size_t globalRowStart) {
    const size_t rows = A.empty() ? 0 : (A.size() / N);
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < rows; ++ii) {
        for (size_t j = 0; j < N; ++j) {
            const size_t i = globalRowStart + ii;
            A[ii * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static void initB(std::vector<double>& B, const size_t N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static void selectCudaDevice() {
    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&localComm));

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int dev = localRank % devCount;
    CUDA_CHECK(cudaSetDevice(dev));
}

// Compute C = A * B for the local row block of A (rows x N), using cuBLAS DGEMM.
// Matrices are stored in row-major order; we compute C^T = B^T * A^T in column-major so the
// result lands in the row-major output buffer without an explicit transpose.
static void gpuMatmulLocal(const double* hA, const double* hB, double* hC, const size_t rows, const size_t N) {
    if (rows == 0 || N == 0) {
        return;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    const size_t bytesA = rows * N * sizeof(double);
    const size_t bytesB = N * N * sizeof(double);
    const size_t bytesC = rows * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), bytesA));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), bytesB));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), bytesC));

    CUDA_CHECK(cudaMemcpyAsync(dA, hA, bytesA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, hB, bytesB, cudaMemcpyHostToDevice, stream));

    const double alpha = 1.0;
    const double beta = 0.0;
    const int n = static_cast<int>(N);
    const int m = static_cast<int>(rows);
    const int k = static_cast<int>(N);

    // C^T (n x m) = B^T (n x k) * A^T (k x m)
    CUBLAS_CHECK(cublasDgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             dB,
                             n,
                             dA,
                             k,
                             &beta,
                             dC,
                             n));

    CUDA_CHECK(cudaMemcpyAsync(hC, dC, bytesC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

static double expectedElement(const size_t N, const size_t i, const size_t j) {
    double sum = 0.0;
    for (size_t k = 0; k < N; ++k) {
        sum += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
    }
    return sum;
}

static bool validateDistributed(const std::vector<double>& C_local,
                                const size_t N,
                                const size_t globalRowStart,
                                const size_t globalRowEnd,
                                const int worldRank,
                                const int worldSize) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    if (worldRank == 0) {
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = checkPoints[pi] % N;
                const size_t j = checkPoints[pj] % N;
                const int owner = static_cast<int>(((i + 1) * static_cast<size_t>(worldSize) - 1) / N);

                double actual = 0.0;
                if (owner == 0) {
                    const size_t localI = i - globalRowStart;
                    actual = C_local[localI * N + j];
                } else {
                    MPI_CHECK(MPI_Recv(&actual, 1, MPI_DOUBLE, owner, static_cast<int>(i), MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                }

                const double expected = expectedElement(N, i, j);
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    return false;
                }
            }
        }
        return true;
    }

    // Non-root: send any needed check rows owned by this rank.
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = checkPoints[pi] % N;
        const int owner = static_cast<int>(((i + 1) * static_cast<size_t>(worldSize) - 1) / N);
        if (owner != worldRank) {
            continue;
        }
        if (!(i >= globalRowStart && i < globalRowEnd)) {
            continue;
        }

        const size_t localI = i - globalRowStart;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            const double actual = C_local[localI * N + j];
            MPI_CHECK(MPI_Send(&actual, 1, MPI_DOUBLE, 0, static_cast<int>(i), MPI_COMM_WORLD));
        }
    }

    return true;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int worldRank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    unsigned long long N_ull = 512;
    int validate = 0;
    int printResults = 0;
    int help = 0;
    int parseFailed = 0;

    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N_ull = static_cast<unsigned long long>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                help = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseFailed = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&help, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (help) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 0;
    }

    MPI_CHECK(MPI_Bcast(&parseFailed, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (parseFailed) {
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    MPI_CHECK(MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD));

    const size_t N = static_cast<size_t>(N_ull);

    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Parallelization: MPI + OpenMP + CUDA (cuBLAS)\n");
        printf("MPI ranks: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t rowStart = blockStart(N, worldRank, worldSize);
    const size_t rowEnd = blockEnd(N, worldRank, worldSize);
    const size_t rows = (rowEnd > rowStart) ? (rowEnd - rowStart) : 0;

    std::vector<double> A_local(rows * N);
    std::vector<double> C_local(rows * N);
    std::vector<double> B;
    if (rows > 0) {
        B.resize(N * N);
    }

    if (worldRank == 0) {
        printf("Initializing matrices...\n");
        printf("Computing matrix multiplication...\n");
    }

    if (rows > 0) {
        initLocalA(A_local, N, rowStart);
        initB(B, N);
    }

    // One GPU per (node-local) MPI rank (only ranks that actually compute need a GPU).
    if (rows > 0) {
        selectCudaDevice();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t0 = MPI_Wtime();

    if (rows > 0) {
        gpuMatmulLocal(A_local.data(), B.data(), C_local.data(), rows, N);
    }

    const double t1 = MPI_Wtime();
    const double localSec = t1 - t0;

    double maxSec = 0.0;
    MPI_CHECK(MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (worldRank == 0) {
        const long ms = static_cast<long>(maxSec * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / maxSec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> C_full;
        if (worldRank == 0) {
            counts.resize(worldSize);
            displs.resize(worldSize);
            int disp = 0;
            for (int r = 0; r < worldSize; ++r) {
                const size_t rs = blockStart(N, r, worldSize);
                const size_t re = blockEnd(N, r, worldSize);
                const size_t rrows = (re > rs) ? (re - rs) : 0;
                const size_t cnt = rrows * N;
                counts[r] = static_cast<int>(cnt);
                displs[r] = disp;
                disp += counts[r];
            }
            C_full.resize(N * N);
        }

        const int sendCount = static_cast<int>(rows * N);
        MPI_CHECK(MPI_Gatherv(C_local.empty() ? nullptr : C_local.data(),
                              sendCount,
                              MPI_DOUBLE,
                              (worldRank == 0) ? C_full.data() : nullptr,
                              (worldRank == 0) ? counts.data() : nullptr,
                              (worldRank == 0) ? displs.data() : nullptr,
                              MPI_DOUBLE,
                              0,
                              MPI_COMM_WORLD));

        if (worldRank == 0) {
            print_results(C_full, "MatrixC");
        }
    }

    int validAll = 1;
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
        }

        const bool validLocal = validateDistributed(C_local, N, rowStart, rowEnd, worldRank, worldSize);

        int valid = 1;
        if (worldRank == 0) {
            valid = validLocal ? 1 : 0;
        }
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
        validAll = valid;

        if (worldRank == 0) {
            printf("Validation: %s\n", validAll ? "PASSED" : "FAILED");
        }
    }

    MPI_CHECK(MPI_Finalize());
    return validAll ? 0 : 1;
}
