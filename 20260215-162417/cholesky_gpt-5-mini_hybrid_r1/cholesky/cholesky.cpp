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

// Simple hybrid Cholesky: CUDA is used to form A = B*B^T, OpenMP accelerates reductions,
// and MPI is used for process orchestration (broadcast and reduction of timings).

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

// GPU kernel: compute A = B * B^T (row-major). Each thread computes one entry.
extern "C" __global__ void matMulKernel(const double* B, double* A, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += B[row * n + k] * B[col * n + k];
        }
        A[row * n + col] = sum;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order. Use OpenMP to parallelize reduction loops where safe.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;

            if (i == j) {
                // Diagonal element: sum of squares of row j, columns 0..j-1
                #pragma omp parallel for reduction(+:sum)
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    if (MPI::COMM_WORLD.Get_rank() == 0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    }
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                #pragma omp parallel for reduction(+:sum)
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }

        // Zero out upper triangular part (parallelizable)
        #pragma omp parallel for
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix using GPU to accelerate B * B^T
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Allocate device memory
    double* d_B = nullptr;
    double* d_A = nullptr;
    size_t bytes = sizeof(double) * n * n;
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));

    // Copy B to device
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    matMulKernel<<<grid, block>>>(d_B, d_A, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy A back
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (parallelized)
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    if (MPI::COMM_WORLD.Get_rank() == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    if (relError > 1e-6) {
        if (MPI::COMM_WORLD.Get_rank() == 0) {
            printf("Validation failed: relative error too large\n");
        }
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        printf("Usage: %s [options]\n", progName);
        printf("Options:\n");
        printf("  -n <num>     Matrix size (default: 512)\n");
        printf("  -v           Enable validation\n");
        printf("  -r           Print results for external validation\n");
        printf("  -h           Show this help message\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 needed but broadcast for consistency)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            }
        }
    }

    // Broadcast options and n to all ranks
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d, OpenMP threads: %d\n", size, omp_get_max_threads());
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix on rank 0 using GPU and broadcast to all ranks
    if (rank == 0) {
        if (cudaGetDeviceCount(NULL) > 0) {
            // let runtime pick device
            CUDA_CHECK(cudaSetDevice(0));
        }
        printf("Generating positive definite matrix on GPU (rank 0)...\n");
        generatePositiveDefiniteMatrix(A, n);
    }

    // Broadcast matrix A to all ranks
    MPI_Bcast(A.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (validate && rank == 0) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition on each rank (work is duplicated across ranks to keep semantics simple)
    if (rank == 0) {
        printf("Computing Cholesky decomposition (each rank performs computation locally for correctness)...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    bool success = choleskyDecomposition(A, n);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    // Reduce to get minimum time across ranks (best-case)
    double min_time = 0.0;
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        long ms = (long)(min_time * 1000.0);
        printf("Computation time (best rank): %ld ms\n", ms);

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / min_time / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    // Validation (only rank 0 prints)
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
