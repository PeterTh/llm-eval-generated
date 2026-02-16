#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixFull(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < N * N; ++idx) {
        size_t i = idx / N;
        size_t j = idx % N;
        mat[idx] = getPseudoRndValue(N, i, j);
    }
}

void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t start_row, const size_t rows) {
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < rows; ++ii) {
        for (size_t j = 0; j < N; ++j) {
            mat[ii * N + j] = getPseudoRndValue(N, start_row + ii, j);
        }
    }
}

// CUDA kernel: tiled matrix multiplication (A is rows x N, B is N x N, C is rows x N)
extern "C" __global__ void matmul_kernel(const double* A, const double* B, double* C, int N, int row_offset) {
    const int TILE = 16;
    __shared__ double Asub[TILE][TILE];
    __shared__ double Bsub[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // local row within A block
    int col = blockIdx.x * TILE + threadIdx.x;
    int global_row = row_offset + row; // absolute row index

    double sum = 0.0;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int A_col = t * TILE + threadIdx.x;
        int B_row = t * TILE + threadIdx.y;

        // Load Asub
        if (row < blockDim.y * gridDim.y && A_col < N && row >=0) {
            Asub[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load Bsub
        if (B_row < N && col < N) {
            Bsub[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < blockDim.y * gridDim.y && col < N) {
        C[row * N + col] = sum;
    }
}

bool validateResultCPU(const std::vector<double>& A_full, const std::vector<double>& B_full,
                       const std::vector<double>& C_full, const size_t N) {
    // Check a few positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A_full[i * N + k] * B_full[k * N + j];
            const double actual = C_full[i * N + j];
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
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Parse command line arguments (only rank 0 prints usage)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d, Threads per process: %d\n", world_size, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine the row partitioning for each rank
    std::vector<int> rows_per_rank(world_size, 0);
    int base = N / world_size;
    int rem = N % world_size;
    for (int r = 0; r < world_size; ++r) rows_per_rank[r] = base + (r < rem ? 1 : 0);

    int start_row = 0;
    for (int r = 0; r < world_rank; ++r) start_row += rows_per_rank[r];
    int my_rows = rows_per_rank[world_rank];

    // Allocate local matrices
    std::vector<double> A_local(my_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(my_rows * N);

    // Initialize A_local and B
    initMatrixRows(A_local, N, start_row, my_rows);
    initMatrixFull(B, N); // everyone has full B for simplicity and locality

    // Root also keeps full A and full C for validation / printing
    std::vector<double> A_full, C_full;
    if (world_rank == 0) {
        A_full.resize(N * N);
        C_full.resize(N * N);
        initMatrixFull(A_full, N);
    }

    // Synchronize and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // CUDA device selection: assign one device per MPI rank if available
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int device_id = 0;
    if (device_count > 0) device_id = world_rank % device_count;
    cudaSetDevice(device_id);

    // Allocate device memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t A_bytes = (size_t)my_rows * N * sizeof(double);
    size_t B_bytes = (size_t)N * N * sizeof(double);
    size_t C_bytes = (size_t)my_rows * N * sizeof(double);

    cudaMalloc((void**)&d_A, A_bytes);
    cudaMalloc((void**)&d_B, B_bytes);
    cudaMalloc((void**)&d_C, C_bytes);

    cudaMemcpy(d_A, A_local.data(), A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (my_rows + TILE - 1) / TILE);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, (int)N, start_row);
    cudaDeviceSynchronize();

    cudaMemcpy(C_local.data(), d_C, C_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Gather C_local into root process
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        recvcounts[r] = rows_per_rank[r] * N;
    }
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) displs[r] = displs[r - 1] + recvcounts[r - 1];

    MPI_Gatherv(C_local.data(), my_rows * N, MPI_DOUBLE,
                world_rank == 0 ? C_full.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (world_rank == 0) {
        double duration = (t1 - t0);
        printf("Computation time: %.3f s\n", duration);
        double gflops = (2.0 * (double)N * (double)N * (double)N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(C_full, "MatrixC");

        if (validate) {
            printf("Validating result...\n");
            bool ok = validateResultCPU(A_full, B, C_full, N);
            if (ok) printf("Validation: PASSED\n"); else printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
