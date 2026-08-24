#include <algorithm>
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

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CUDA tiled matrix multiplication kernel with shared memory for data reuse
template <int BLOCK_SIZE>
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N,
                             const size_t local_rows) {
    __shared__ double sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sB[BLOCK_SIZE][BLOCK_SIZE];

    const size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double sum = 0.0;
    const size_t numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t t = 0; t < numTiles; ++t) {
        // Load tile from A into shared memory
        const size_t a_col = t * BLOCK_SIZE + threadIdx.x;
        if (row < local_rows && a_col < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        // Load tile from B into shared memory
        const size_t b_row = t * BLOCK_SIZE + threadIdx.y;
        if (b_row < N && col < N)
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        // Compute partial dot product using shared memory tiles
        #pragma unroll
        for (size_t k = 0; k < BLOCK_SIZE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < local_rows && col < N)
        C[row * N + col] = sum;
}

// Launch the CUDA tiled matrix multiplication kernel
void launchMatmul(const double* d_A, const double* d_B, double* d_C,
                  const size_t N, const size_t local_rows, cudaStream_t stream = 0) {
    if (local_rows == 0) return;
    constexpr int BS = 16;
    dim3 blockDim(BS, BS);
    dim3 gridDim((N + BS - 1) / BS, (local_rows + BS - 1) / BS);
    matmulKernel<BS><<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N, local_rows);
}

// Simple validation: compute a single element and compare
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Parse arguments on rank 0 and broadcast
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                N = strtoul(argv[++i], nullptr, 10);
            else if (strcmp(argv[i], "-v") == 0)
                validate = true;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
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

    int validate_int = validate ? 1 : 0;
    int printRes_int = printResults ? 1 : 0;

    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printRes_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_int != 0);
    printResults = (printRes_int != 0);

    // Compute row distribution across MPI ranks
    std::vector<int> scounts(num_ranks);
    std::vector<int> sdispls(num_ranks);
    std::vector<int> rcounts(num_ranks);
    std::vector<int> rdispls(num_ranks);
    size_t displacement = 0;
    size_t local_rows = 0;

    for (int r = 0; r < num_ranks; ++r) {
        size_t rows = N / num_ranks + (r < (int)(N % num_ranks) ? 1 : 0);
        scounts[r] = rows * N;
        sdispls[r] = displacement * N;
        rcounts[r] = rows * N;
        rdispls[r] = displacement * N;
        if (r == rank)
            local_rows = rows;
        displacement += rows;
    }

    // Allocate matrices on rank 0
    std::vector<double> A, B, C;

    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);

        int omp_threads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            omp_threads = omp_get_num_threads();
        }

        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);

        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", num_ranks);
        printf("OpenMP threads: %d\n", omp_threads);
        printf("CUDA devices detected: %d\n", num_devices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");

        initMatrix(A, N);
        initMatrix(B, N);
    }

    // Setup CUDA device (round-robin assignment)
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0 && rank == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (num_devices > 0)
        cudaSetDevice(rank % num_devices);

    // Allocate local row buffers
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);

    // Scatter rows of A from rank 0 to all ranks
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_DOUBLE,
                 A_local.data(), local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast full matrix B to all ranks
    std::vector<double> B_local(N * N);
    if (rank == 0)
        MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Bcast(B_local.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double* B_ptr = (rank == 0) ? B.data() : B_local.data();

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    if (local_rows > 0) {
        cudaMalloc(&d_A, local_rows * N * sizeof(double));
        cudaMalloc(&d_C, local_rows * N * sizeof(double));
    }
    cudaMalloc(&d_B, N * N * sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_B, B_ptr, N * N * sizeof(double), cudaMemcpyHostToDevice);
    if (local_rows > 0)
        cudaMemcpy(d_A, A_local.data(), local_rows * N * sizeof(double), cudaMemcpyHostToDevice);

    // Synchronize all ranks before computation begins
    if (rank == 0)
        printf("Computing matrix multiplication...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    auto compute_start = std::chrono::high_resolution_clock::now();

    // Launch CUDA kernel
    launchMatmul(d_A, d_B, d_C, N, local_rows);

    cudaDeviceSynchronize();
    auto compute_end = std::chrono::high_resolution_clock::now();

    long local_duration = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(
        compute_end - compute_start).count());
    long duration = 0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy result back from device to host
    if (local_rows > 0)
        cudaMemcpy(C_local.data(), d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Gather all local C blocks into rank 0's full C matrix
    MPI_Gatherv(C_local.data(), local_rows * N, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                rcounts.data(), rdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Output and validation on rank 0
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);

        double gflops = (2.0 * N * N * N) / (duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults)
            print_results(C, "MatrixC");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            cudaFree(d_B);
            if (local_rows > 0) { cudaFree(d_A); cudaFree(d_C); }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    // Cleanup
    cudaFree(d_B);
    if (local_rows > 0) { cudaFree(d_A); cudaFree(d_C); }
    MPI_Finalize();
    return 0;
}
