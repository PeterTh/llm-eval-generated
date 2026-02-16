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

// CUDA tiled kernel (row-blocked for local rows)
constexpr int TILE = 16;
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N, int rows_local) {
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    int local_row = blockIdx.y * TILE + threadIdx.y; // 0..rows_local-1
    int col = blockIdx.x * TILE + threadIdx.x;       // 0..N-1

    double sum = 0.0;
    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (local_row < rows_local && a_col < N)
            sA[threadIdx.y][threadIdx.x] = A[local_row * N + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if (b_row < N && col < N)
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (local_row < rows_local && col < N) {
        C[local_row * N + col] = sum;
    }
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
    // Initialize MPI with thread support for OpenMP interactions
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

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
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine rows assigned to each rank
    std::vector<int> rows_per_rank(world_size);
    std::vector<int> row_offsets(world_size);
    int base = N / world_size;
    int rem = N % world_size;
    for (int r = 0; r < world_size; ++r) {
        rows_per_rank[r] = base + (r < rem ? 1 : 0);
    }
    row_offsets[0] = 0;
    for (int r = 1; r < world_size; ++r) row_offsets[r] = row_offsets[r-1] + rows_per_rank[r-1];

    const int local_rows = rows_per_rank[world_rank];
    const int local_row_offset = row_offsets[world_rank];

    // Allocate host matrices (local A, full B, local C)
    std::vector<double> localA(static_cast<size_t>(local_rows) * N);
    std::vector<double> B(static_cast<size_t>(N) * N);
    std::vector<double> localC(static_cast<size_t>(local_rows) * N);

    // Initialize local A (only the rows this rank owns) and full B in parallel using OpenMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_rows; ++i) {
        size_t gi = static_cast<size_t>(local_row_offset + i);
        for (size_t j = 0; j < N; ++j) {
            localA[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, gi, j);
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }

    // Create device buffers and copy data to GPU
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    size_t bytesA = static_cast<size_t>(local_rows) * N * sizeof(double);
    size_t bytesB = static_cast<size_t>(N) * N * sizeof(double);
    size_t bytesC = static_cast<size_t>(local_rows) * N * sizeof(double);

    cudaError_t cerr;
    cerr = cudaMalloc(&dA, bytesA); if (cerr != cudaSuccess) { fprintf(stderr, "CUDA malloc dA failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }
    cerr = cudaMalloc(&dB, bytesB); if (cerr != cudaSuccess) { fprintf(stderr, "CUDA malloc dB failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }
    cerr = cudaMalloc(&dC, bytesC); if (cerr != cudaSuccess) { fprintf(stderr, "CUDA malloc dC failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }

    cerr = cudaMemcpy(dA, localA.data(), bytesA, cudaMemcpyHostToDevice); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMemcpy dA failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }
    cerr = cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMemcpy dB failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }

    // Synchronize before timed region
    MPI_Barrier(MPI_COMM_WORLD);
    auto tstart = std::chrono::high_resolution_clock::now();

    // Launch kernel
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (local_rows + TILE - 1) / TILE);

    matmulKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), local_rows);
    cerr = cudaGetLastError(); if (cerr != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }

    cerr = cudaDeviceSynchronize(); if (cerr != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }

    // Copy result back
    cerr = cudaMemcpy(localC.data(), dC, bytesC, cudaMemcpyDeviceToHost); if (cerr != cudaSuccess) { fprintf(stderr, "cudaMemcpy dC->host failed: %s\n", cudaGetErrorString(cerr)); MPI_Abort(MPI_COMM_WORLD, -1); }

    auto tend = std::chrono::high_resolution_clock::now();
    auto loc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();

    // Reduce to get max duration across ranks
    long long max_duration = 0;
    long long loc_d = static_cast<long long>(loc_duration);
    MPI_Reduce(&loc_d, &max_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute performance on rank 0
    if (world_rank == 0) {
        double seconds = static_cast<double>(max_duration) / 1000.0;
        double gflops = (2.0 * static_cast<double>(N) * N * N) / seconds / 1e9;
        printf("Computation time (max over ranks): %lld ms\n", max_duration);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // If printResults requested, gather full C on rank 0 and print
    if (printResults) {
        std::vector<int> recvcounts(world_size);
        std::vector<int> displs(world_size);
        for (int r = 0; r < world_size; ++r) {
            recvcounts[r] = rows_per_rank[r] * static_cast<int>(N);
            displs[r] = row_offsets[r] * static_cast<int>(N);
        }
        std::vector<double> fullC;
        if (world_rank == 0) fullC.resize(static_cast<size_t>(N) * N);

        MPI_Gatherv(localC.data(), local_rows * static_cast<int>(N), MPI_DOUBLE,
                    fullC.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (world_rank == 0) print_results(fullC, "MatrixC");
    }

    // Validation: each rank validates a few sample points in its local rows and reduce
    bool local_valid = true;
    if (validate) {
        const int checkPoints[5] = {0,1,2,3,4};
        for (int pi = 0; pi < 5 && local_valid; ++pi) {
            for (int pj = 0; pj < 5 && local_valid; ++pj) {
                size_t gi = static_cast<size_t>(checkPoints[pi]) ;
                size_t gj = static_cast<size_t>(checkPoints[pj]) ;
                // Only validate if the global row belongs to this rank
                if (gi >= static_cast<size_t>(local_row_offset) && gi < static_cast<size_t>(local_row_offset + local_rows)) {
                    int local_i = static_cast<int>(gi - local_row_offset);
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) expected += localA[static_cast<size_t>(local_i) * N + k] * B[k * N + gj];
                    double actual = localC[static_cast<size_t>(local_i) * N + gj];
                    double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        fprintf(stderr, "Rank %d: Validation failed at (%zu,%zu): expected %.10f got %.10f (err %.10e)\n",
                                world_rank, gi, gj, expected, actual, relError);
                        local_valid = false;
                        break;
                    }
                }
            }
        }
    }

    int global_valid_int = 1;
    int local_valid_int = (local_valid ? 1 : 0);
    MPI_Reduce(&local_valid_int, &global_valid_int, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world_rank == 0 && validate) {
        if (global_valid_int) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    MPI_Finalize();
    return 0;
}
