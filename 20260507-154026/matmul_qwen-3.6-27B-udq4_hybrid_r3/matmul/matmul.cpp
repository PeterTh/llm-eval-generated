#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = call;                                                 \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(_err));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

/* ------------------------------------------------------------------ */
/*  CUDA tiled matmul kernel (16×16 tiles, double precision)          */
/* ------------------------------------------------------------------ */
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N, const size_t local_M) {
    constexpr unsigned TILE = 16;
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const size_t row = blockIdx.y * blockDim.y + ty;
    const size_t col = blockIdx.x * blockDim.x + tx;

    if (row >= local_M || col >= N) return;

    double sum = 0.0;
    const unsigned tiles = static_cast<unsigned>((N + TILE - 1) / TILE);

    for (unsigned t = 0; t < tiles; ++t) {
        /* Load tile of A: each thread loads one element per tile */
        sA[ty][tx] = (t * TILE + tx < N)
                         ? A[row * N + t * TILE + tx]
                         : 0.0;
        /* Load tile of B: transpose layout for coalesced access */
        sB[ty][tx] = (t * TILE + ty < N)
                         ? B[(t * TILE + ty) * N + col]
                         : 0.0;
        __syncthreads();

        for (unsigned k = 0; k < TILE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

/* ------------------------------------------------------------------ */
/*  Host helpers                                                       */
/* ------------------------------------------------------------------ */
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

/* OpenMP-parallelised initialisation */
void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got "
                    "%.10f (error: %.10e)\n",
                    i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

/* ------------------------------------------------------------------ */
/*  Main – hybrid MPI + OpenMP + CUDA                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* ---- parse CLI on rank 0, broadcast to all ranks ---- */
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d | OpenMP threads: %d\n", num_ranks,
               omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- 1-D block row distribution ---- */
    const size_t base_M    = N / static_cast<size_t>(num_ranks);
    const size_t remainder = N % static_cast<size_t>(num_ranks);
    const size_t local_M   = base_M +
                             (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t my_row_start =
        static_cast<size_t>(rank) * base_M +
        std::min(static_cast<size_t>(rank), remainder);

    /* Scatter / gather counts and displacements (all ranks) */
    std::vector<int> counts(num_ranks), displs(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        const size_t lm = base_M + (static_cast<size_t>(r) < remainder ? 1 : 0);
        counts[r] = static_cast<int>(lm * N);
        displs[r] = static_cast<int>(
            (static_cast<size_t>(r) * base_M +
             std::min(static_cast<size_t>(r), remainder)) *
            N);
    }

    /* ---- allocate local buffers ---- */
    std::vector<double> A_local(local_M * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_M * N, 0.0);

    /* ---- initialise A on rank 0, scatter to all ranks ---- */
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(N * N);
        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B, N);
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, counts.data(),
                 displs.data(), MPI_DOUBLE, A_local.data(),
                 static_cast<int>(local_M * N), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    /* ---- broadcast B to every rank ---- */
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    /* ---- CUDA: allocate device memory & transfer data ---- */
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (local_M > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, local_M * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, local_M * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(),
                              local_M * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    /* ---- synchronise ranks, then time the kernel ---- */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");

    auto start = std::chrono::high_resolution_clock::now();

    if (local_M > 0) {
        const dim3 block(16, 16);
        const dim3 grid((N + 15) / 16, (local_M + 15) / 16);
        matmulKernel<<<grid, block>>>(d_A, d_B, d_C, N, local_M);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    /* ---- copy results back ---- */
    if (local_M > 0) {
        CUDA_CHECK(cudaMemcpy(C_local.data(), d_C,
                              local_M * N * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    /* ---- gather C back to rank 0 ---- */
    std::vector<double> C_full;
    if (rank == 0) C_full.resize(N * N);

    MPI_Gatherv(C_local.data(), static_cast<int>(local_M * N), MPI_DOUBLE,
                rank == 0 ? C_full.data() : nullptr, counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- rank 0: report, validate ---- */
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        double gflops =
            (2.0 * N * N * N) / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A_full, B, C_full, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
