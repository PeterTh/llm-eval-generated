#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static constexpr int TILE = 32;

__global__ void matmul_tiled_kernel(const double* __restrict__ A, const double* __restrict__ B,
                                   double* __restrict__ C, int N, int rowsA) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    const int col = (int)blockIdx.x * TILE + (int)threadIdx.x;

    double sum = 0.0;

    for (int k0 = 0; k0 < N; k0 += TILE) {
        const int a_col = k0 + (int)threadIdx.x;
        const int b_row = k0 + (int)threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < rowsA && a_col < N) ? A[(size_t)row * (size_t)N + (size_t)a_col] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[(size_t)b_row * (size_t)N + (size_t)col] : 0.0;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsA && col < N) {
        C[(size_t)row * (size_t)N + (size_t)col] = sum;
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned long long N_ull = 512;
    int validate_i = 0;
    int printResults_i = 0;

    if (world_rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N_ull = std::strtoull(argv[++i], nullptr, 10);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
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

        if (N_ull == 0) {
            printf("Matrix size N must be >= 1\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t N = (size_t)N_ull;
    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    // Pick a GPU per rank based on intra-node rank.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (world_rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = local_rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
        printf("OpenMP threads (max): %d\n", omp_get_max_threads());
    }

    // Row-block distribution of A across ranks
    const size_t base = N / (size_t)world_size;
    const size_t rem = N % (size_t)world_size;

    const size_t local_rows = base + ((size_t)world_rank < rem ? 1 : 0);

    std::vector<int> sendcounts(world_size, 0);
    std::vector<int> displs(world_size, 0);
    for (int r = 0; r < world_size; ++r) {
        const size_t rows_r = base + ((size_t)r < rem ? 1 : 0);
        const size_t off_r = base * (size_t)r + (size_t)((r < (int)rem) ? r : (int)rem);
        const size_t cnt = rows_r * N;
        sendcounts[r] = (int)cnt;
        displs[r] = (int)(off_r * N);
    }

    std::vector<double> A_full;
    std::vector<double> C_full;
    std::vector<double> B((size_t)N * (size_t)N);

    if (world_rank == 0) {
        A_full.resize((size_t)N * (size_t)N);
        if (validate || printResults) {
            C_full.resize((size_t)N * (size_t)N);
        }

        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B, N);
    }

    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);

    MPI_Scatterv(world_rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), (int)(local_rows * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), (int)((size_t)N * (size_t)N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Device allocations & H2D (not timed)
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    const size_t bytesB = (size_t)N * (size_t)N * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&dB, bytesB));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));

    if (local_rows > 0) {
        const size_t bytesA = local_rows * N * sizeof(double);
        const size_t bytesC = local_rows * N * sizeof(double);
        CUDA_CHECK(cudaMalloc((void**)&dA, bytesA));
        CUDA_CHECK(cudaMalloc((void**)&dC, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, A_local.data(), bytesA, cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    double local_sec = 0.0;
    if (local_rows > 0) {
        const dim3 block(TILE, TILE);
        const dim3 grid((unsigned)((N + TILE - 1) / TILE), (unsigned)((local_rows + TILE - 1) / TILE));

        const double t0 = MPI_Wtime();
        matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, (int)N, (int)local_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t1 = MPI_Wtime();
        local_sec = (t1 - t0);
    }

    double max_sec = 0.0;
    MPI_Reduce(&local_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results back / gather only when needed
    if (validate || printResults) {
        if (local_rows > 0) {
            const size_t bytesC = local_rows * N * sizeof(double);
            CUDA_CHECK(cudaMemcpy(C_local.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        }

        MPI_Gatherv(C_local.data(), (int)(local_rows * N), MPI_DOUBLE,
                    world_rank == 0 ? C_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        const double denom = (max_sec > 0.0) ? max_sec : 1e-12;
        const long ms = (long)llround(denom * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (2.0 * (double)N * (double)N * (double)N) / (denom) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A_full, B, C_full, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }

            CUDA_CHECK(cudaFree(dB));
            if (dA) CUDA_CHECK(cudaFree(dA));
            if (dC) CUDA_CHECK(cudaFree(dC));

            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFree(dB));
    if (dA) CUDA_CHECK(cudaFree(dA));
    if (dC) CUDA_CHECK(cudaFree(dC));

    MPI_Finalize();
    return 0;
}
