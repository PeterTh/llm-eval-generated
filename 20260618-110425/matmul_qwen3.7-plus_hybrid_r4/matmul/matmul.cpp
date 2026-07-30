#include <chrono>
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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N,
                const size_t start_row, const size_t num_rows) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

bool validateResult(const std::vector<double>& B, const std::vector<double>& C,
                    const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    int validate_flag = 0;
    int print_flag = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0);
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }

    unsigned long long N_ull = static_cast<unsigned long long>(N);
    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_ull);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    bool validate = validate_flag != 0;
    bool printResults = print_flag != 0;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution across MPI ranks
    const size_t total_rows = N;
    const size_t base_rows = total_rows / static_cast<size_t>(nprocs);
    const size_t remainder = total_rows % static_cast<size_t>(nprocs);

    size_t local_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t start_row;
    if (static_cast<size_t>(rank) < remainder) {
        start_row = static_cast<size_t>(rank) * (base_rows + 1);
    } else {
        start_row = remainder * (base_rows + 1)
                  + (static_cast<size_t>(rank) - remainder) * base_rows;
    }

    // Allocate local matrices
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N, 0.0);

    // Initialize matrices with OpenMP parallelism
    if (rank == 0) printf("Initializing matrices...\n");

    initMatrix(A_local, N, start_row, local_rows);
    initMatrix(B, N, 0, N);

    // Register host memory for faster DMA transfers
    if (local_rows > 0) {
        cudaHostRegister(A_local.data(), A_local.size() * sizeof(double), cudaHostRegisterDefault);
        cudaHostRegister(C_local.data(), C_local.size() * sizeof(double), cudaHostRegisterDefault);
    }
    cudaHostRegister(B.data(), B.size() * sizeof(double), cudaHostRegisterDefault);

    // Determine local rank for GPU assignment
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        if (rank == 0) fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    cudaSetDevice(local_rank % num_devices);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate GPU memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (local_rows > 0) {
        cudaMalloc(&d_A, local_rows * N * sizeof(double));
        cudaMalloc(&d_C, local_rows * N * sizeof(double));
    }
    cudaMalloc(&d_B, N * N * sizeof(double));

    // Copy input data to GPU
    if (local_rows > 0) {
        cudaMemcpy(d_A, A_local.data(), local_rows * N * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix multiplication with cuBLAS on GPU
    if (rank == 0) printf("Computing matrix multiplication...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (local_rows > 0 && N > 0) {
        const double alpha = 1.0;
        const double beta = 0.0;

        // Row-major C_local = A_local * B
        // Equivalent col-major: C_local^T = B^T * A_local^T
        // cuBLAS sees row-major data as transposed col-major
        cublasStatus_t status = cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(local_rows),
            static_cast<int>(N),
            &alpha,
            d_B, static_cast<int>(N),
            d_A, static_cast<int>(N),
            &beta,
            d_C, static_cast<int>(N));

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "Rank %d: cuBLAS error %d\n", rank, status);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long local_ms = static_cast<long long>(duration.count());
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy result back to host
    if (local_rows > 0) {
        cudaMemcpy(C_local.data(), d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Gather results on rank 0 using MPI_Gatherv
    std::vector<int> recv_counts(nprocs);
    std::vector<int> recv_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t rows_r = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recv_counts[r] = static_cast<int>(rows_r * N);
    }
    recv_displs[0] = 0;
    for (int r = 1; r < nprocs; ++r) {
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }

    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 outputs results
    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);

        double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N))
                        / (static_cast<double>(max_ms) / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    // Broadcast exit code to all ranks
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    cublasDestroy(handle);
    if (d_A) cudaFree(d_A);
    if (d_C) cudaFree(d_C);
    cudaFree(d_B);

    if (local_rows > 0) {
        cudaHostUnregister(A_local.data());
        cudaHostUnregister(C_local.data());
    }
    cudaHostUnregister(B.data());

    MPI_Finalize();
    return exit_code;
}
