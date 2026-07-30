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

// Generate pseudo-random values for matrix initialization (same as original)
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// OpenMP-parallel matrix initialization for a contiguous row range
void initMatrixBlock(double* mat, const size_t N, const size_t startRow, const size_t numRows) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, startRow + i, j);
        }
    }
}

// CPU fallback matrix multiplication with OpenMP (cache-friendly ikj order)
void cpuMatMul(const double* __restrict__ A, const double* __restrict__ B,
               double* __restrict__ C, const size_t myRows, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < myRows; ++i) {
        // Zero-initialize the row
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] = 0.0;
        }
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = A[i * N + k];
            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const double* A, const double* B,
                    const double* C, const size_t N) {
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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse command line arguments on rank 0
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

    // Broadcast parameters from rank 0
    unsigned long long N_ull = static_cast<unsigned long long>(N);
    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_ull);

    int validate_int = validate ? 1 : 0;
    int printResults_int = printResults ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_int != 0);
    printResults = (printResults_int != 0);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP threads: %d\n", omp_get_max_threads());

        // Check CUDA device
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        printf("CUDA devices: %d\n", deviceCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution across MPI ranks
    // Each rank gets a contiguous block of rows
    const size_t totalRows = N;
    const size_t rowsPerProc = totalRows / nprocs;
    const size_t extraRows = totalRows % nprocs;

    size_t myStartRow, myRows;
    if (static_cast<size_t>(rank) < extraRows) {
        myRows = rowsPerProc + 1;
        myStartRow = static_cast<size_t>(rank) * myRows;
    } else {
        myRows = rowsPerProc;
        myStartRow = extraRows * (rowsPerProc + 1) +
                     (static_cast<size_t>(rank) - extraRows) * rowsPerProc;
    }

    // Initialize CUDA on this rank
    bool useGPU = true;
    cublasHandle_t cublasH = nullptr;

    // Select GPU device (round-robin for multi-GPU nodes)
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int deviceId = rank % deviceCount;
        cudaSetDevice(deviceId);

        cublasStatus_t status = cublasCreate(&cublasH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (rank == 0) {
                printf("cuBLAS initialization failed (status %d), falling back to CPU+OpenMP\n",
                       static_cast<int>(status));
            }
            useGPU = false;
        } else {
            // Use default math mode (double precision)
            cublasSetMathMode(cublasH, CUBLAS_DEFAULT_MATH);
        }
    } else {
        if (rank == 0) {
            printf("No CUDA devices found, using CPU+OpenMP fallback\n");
        }
        useGPU = false;
    }

    // Allocate host matrices
    // Local A: myRows x N (this rank's portion of A)
    // B: N x N (full matrix, replicated on all ranks)
    // Local C: myRows x N (this rank's portion of result)
    std::vector<double> localA(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(myRows * N, 0.0);

    // Initialize matrices using OpenMP
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    // Initialize local portion of A
    initMatrixBlock(localA.data(), N, myStartRow, myRows);

    // Initialize full B matrix (needed for the multiplication)
    initMatrixBlock(B.data(), N, 0, N);

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (useGPU) {
        // GPU path: use cuBLAS DGEMM
        // Allocate device memory
        double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        cudaMalloc(&d_A, myRows * N * sizeof(double));
        cudaMalloc(&d_B, N * N * sizeof(double));
        cudaMalloc(&d_C, myRows * N * sizeof(double));

        // Use pinned memory for faster host-device transfers
        double *h_A_pinned = nullptr, *h_B_pinned = nullptr, *h_C_pinned = nullptr;
        cudaHostAlloc(&h_A_pinned, myRows * N * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc(&h_B_pinned, N * N * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc(&h_C_pinned, myRows * N * sizeof(double), cudaHostAllocDefault);

        // Copy data to pinned memory then to device
        memcpy(h_A_pinned, localA.data(), myRows * N * sizeof(double));
        memcpy(h_B_pinned, B.data(), N * N * sizeof(double));

        cudaMemcpy(d_A, h_A_pinned, myRows * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_pinned, N * N * sizeof(double), cudaMemcpyHostToDevice);

        // cuBLAS DGEMM: compute C = A * B
        // cuBLAS uses column-major order. Our row-major matrices are interpreted as:
        //   A_rm (myRows x N) = A_cm^T (N x myRows in col-major)
        //   B_rm (N x N) = B_cm^T (N x N in col-major)
        //   C_rm (myRows x N) = C_cm^T (N x myRows in col-major)
        // We want C_rm = A_rm * B_rm
        // Equivalently: C_cm^T = B_cm^T * A_cm^T => C_cm = B_cm * A_cm
        // cublasDgemm: C = alpha*op(A)*op(B) + beta*C
        // So: pass B_cm as first matrix, A_cm as second matrix
        const double alpha = 1.0;
        const double beta = 0.0;

        cublasStatus_t stat = cublasDgemm(cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(N),        // m: rows of op(B_cm) and C_cm
            static_cast<int>(myRows),   // n: cols of op(A_cm) and C_cm
            static_cast<int>(N),        // k: cols of op(B_cm), rows of op(A_cm)
            &alpha,
            d_B, static_cast<int>(N),   // B_cm (N x N), lda = N
            d_A, static_cast<int>(N),   // A_cm (N x myRows), ldb = N
            &beta,
            d_C, static_cast<int>(N));  // C_cm (N x myRows), ldc = N

        if (stat != CUBLAS_STATUS_SUCCESS) {
            if (rank == 0) {
                printf("cuBLAS DGEMM failed (status %d), falling back to CPU\n",
                       static_cast<int>(stat));
            }
            // CPU fallback
            cpuMatMul(localA.data(), B.data(), localC.data(), myRows, N);
        } else {
            // Copy result back
            cudaMemcpy(h_C_pinned, d_C, myRows * N * sizeof(double), cudaMemcpyDeviceToHost);
            memcpy(localC.data(), h_C_pinned, myRows * N * sizeof(double));
        }

        // Cleanup GPU resources
        cudaFreeHost(h_A_pinned);
        cudaFreeHost(h_B_pinned);
        cudaFreeHost(h_C_pinned);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    } else {
        // CPU fallback with OpenMP
        cpuMatMul(localA.data(), B.data(), localC.data(), myRows, N);
    }

    // Synchronize to get accurate timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Get max duration across all ranks for consistent reporting
    long long dur_ll = duration.count();
    long long max_dur_ll = 0;
    MPI_Reduce(&dur_ll, &max_dur_ll, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0 using MPI_Gatherv
    // Compute displacements and counts
    std::vector<int> recvCounts(nprocs);
    std::vector<int> recvDispls(nprocs);

    // Each rank sends myRows * N doubles
    int localCount = static_cast<int>(myRows * N);
    MPI_Allgather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    recvDispls[0] = 0;
    for (int i = 1; i < nprocs; ++i) {
        recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
    }

    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE,
                C.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Report results on rank 0
    if (rank == 0) {
        long long maxDuration = max_dur_ll;
        printf("Computation time: %lld ms\n", maxDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");

            // Reconstruct full A for validation (deterministic initialization)
            std::vector<double> A(N * N);
            initMatrixBlock(A.data(), N, 0, N);

            bool valid = validateResult(A.data(), B.data(), C.data(), N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    if (cublasH) {
        cublasDestroy(cublasH);
    }

    MPI_Finalize();
    return 0;
}
