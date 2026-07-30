#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Tile size for CUDA shared memory tiling
#define TILE_SIZE 32

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,            \
                    __LINE__, cudaGetErrorString(err));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                    const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CUDA tiled matrix multiplication kernel using shared memory
__global__ void matmul_kernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t M, size_t K,
                               size_t N_dim) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

    for (int t = 0; t < (static_cast<int>(K) + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A (M x K) into shared memory
        if (row < static_cast<int>(M) && t * TILE_SIZE + tx < static_cast<int>(K))
            tileA[ty][tx] = A[static_cast<size_t>(row) * K +
                              static_cast<size_t>(t * TILE_SIZE + tx)];
        else
            tileA[ty][tx] = 0.0;

        // Load tile of B (K x N_dim) into shared memory
        if (col < static_cast<int>(N_dim) && t * TILE_SIZE + ty < static_cast<int>(K))
            tileB[ty][tx] =
                B[static_cast<size_t>(t * TILE_SIZE + ty) * N_dim +
                  static_cast<size_t>(col)];
        else
            tileB[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (row < static_cast<int>(M) && col < static_cast<int>(N_dim))
        C[static_cast<size_t>(row) * N_dim + static_cast<size_t>(col)] = sum;
}

// Validation: compute reference elements on CPU and compare
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
                printf("Validation failed at (%zu, %zu): expected %.10f, "
                       "got %.10f (error: %.10e)\n",
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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Parse command line arguments on rank 0, then broadcast
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
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

    // Broadcast parameters
    unsigned long N_ul = N;
    MPI_Bcast(&N_ul, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    N = N_ul;
    int val = validate ? 1 : 0;
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (val != 0);
    val = printResults ? 1 : 0;
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printResults = (val != 0);

    // Compute row distribution across MPI processes
    size_t rows_per_proc = N / num_procs;
    size_t rem = N % num_procs;
    size_t local_rows =
        rows_per_proc + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Build scatterv/gatherv counts and displacements
    std::vector<int> send_counts(num_procs);
    std::vector<int> displs(num_procs);
    {
        size_t offset = 0;
        for (int i = 0; i < num_procs; ++i) {
            size_t r = rows_per_proc + (static_cast<size_t>(i) < rem ? 1 : 0);
            send_counts[i] = static_cast<int>(r * N);
            displs[i] = static_cast<int>(offset);
            offset += send_counts[i];
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    // Select GPU: round-robin assignment across available devices on node
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(rank % num_gpus);
    }

    // Rank 0 holds the full A and B matrices
    std::vector<double> A, B;
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        printf("Initializing matrices...\n");
        fflush(stdout);
        initMatrix(A, N);
        initMatrix(B, N);
    } else {
        B.resize(N * N);
    }

    // Local buffers for this MPI rank
    std::vector<double> local_A(local_rows * N);
    std::vector<double> local_C(local_rows * N);

    // Distribute rows of A across processes, broadcast B to all
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, send_counts.data(),
                 displs.data(), MPI_DOUBLE, local_A.data(),
                 static_cast<int>(local_A.size()), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    MPI_Bcast(B.data(), static_cast<int>(B.size()), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // ---- CUDA computation on each MPI rank ----
    if (local_rows > 0) {
        double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        CUDA_CHECK(cudaMalloc(&d_A, local_rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, local_rows * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_A, local_A.data(),
                               local_rows * N * sizeof(double),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double),
                               cudaMemcpyHostToDevice));

        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize(static_cast<unsigned int>((N + TILE_SIZE - 1) / TILE_SIZE),
                      static_cast<unsigned int>(
                          (local_rows + TILE_SIZE - 1) / TILE_SIZE));

        matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, local_rows, N,
                                               N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(local_C.data(), d_C,
                               local_rows * N * sizeof(double),
                               cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    double end_time = MPI_Wtime();

    // Gather the result matrix C back to rank 0
    std::vector<double> full_C;
    if (rank == 0) {
        full_C.resize(N * N);
    }

    MPI_Gatherv(local_C.data(), static_cast<int>(local_C.size()), MPI_DOUBLE,
                rank == 0 ? full_C.data() : nullptr, send_counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0: report results and optionally validate
    if (rank == 0) {
        double elapsed = end_time - start_time;
        double elapsed_ms = elapsed * 1000.0;

        printf("Computation time: %.0f ms\n", elapsed_ms);

        // Calculate GFLOPS: 2*N^3 operations / time_in_seconds / 1e9
        double gflops = (2.0 * static_cast<double>(N) *
                         static_cast<double>(N) * static_cast<double>(N)) /
                        elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(full_C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            fflush(stdout);
            bool valid = validateResult(A, B, full_C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
