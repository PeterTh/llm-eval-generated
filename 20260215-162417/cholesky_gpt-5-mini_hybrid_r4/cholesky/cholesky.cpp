#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + optional CUDA implementation (row-cyclic ownership)

#ifdef USE_CUDA
static inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

extern "C" __global__ void dot_kernel(const double* a, const double* b, double* partials, int n) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    // Strided loop
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// Compute dot product using CUDA (copies inputs to device each call)
static double dot_cuda(const double* a_host, const double* b_host, int n) {
    if (n <= 0) return 0.0;
    int threads = 256;
    int blocks = std::min((n + threads - 1) / threads, 1024);

    double *d_a = nullptr, *d_b = nullptr, *d_part = nullptr;
    size_t bytes = sizeof(double) * n;
    cudaCheck(cudaMalloc((void**)&d_a, bytes), "malloc a");
    cudaCheck(cudaMalloc((void**)&d_b, bytes), "malloc b");
    cudaCheck(cudaMemcpy(d_a, a_host, bytes, cudaMemcpyHostToDevice), "copy a");
    cudaCheck(cudaMemcpy(d_b, b_host, bytes, cudaMemcpyHostToDevice), "copy b");

    cudaCheck(cudaMalloc((void**)&d_part, sizeof(double) * blocks), "malloc parts");

    size_t shared = threads * sizeof(double);
    dot_kernel<<<blocks, threads, shared>>>(d_a, d_b, d_part, n);
    cudaCheck(cudaGetLastError(), "kernel launch");

    std::vector<double> parts(blocks);
    cudaCheck(cudaMemcpy(parts.data(), d_part, sizeof(double) * blocks, cudaMemcpyDeviceToHost), "copy parts");

    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += parts[i];

    cudaCheck(cudaFree(d_a), "free a");
    cudaCheck(cudaFree(d_b), "free b");
    cudaCheck(cudaFree(d_part), "free parts");

    return sum;
}
#else
// Fallback when CUDA is not available: use OpenMP parallel reduction
static double dot_cuda(const double* a_host, const double* b_host, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) sum += a_host[i] * b_host[i];
    return sum;
}
#endif

// Compute dot product with hybrid strategy: CUDA for large vectors, OpenMP for small
static double compute_dot(const double* a, const double* b, int n) {
    const int cuda_threshold = 512; // threshold to prefer cuda
    if (n >= cuda_threshold) {
        return dot_cuda(a, b, n);
    } else {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
}

// Generate positive definite matrix (only rank 0 needs to do this)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T with OpenMP parallelization
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
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

    if (MPI_COMM_WORLD != MPI_COMM_NULL) {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
            int world_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            if (world_rank == 0) {
                printf("Max absolute error: %.10e\n", maxError);
                printf("Max relative error: %.10e\n", relError);
            }
        }
    }

    if (relError > 1e-6) return false;
    return true;
}

void printUsage(const char* progName) {
    if (MPI_COMM_WORLD != MPI_COMM_NULL) {
        int initialized = 0; MPI_Initialized(&initialized);
        if (initialized) {
            int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank != 0) return;
        }
    }
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints usage/errors)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Ranks: %d, Matrix size: %zu x %zu\n", world_size, n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix on all ranks to keep simple memory model; ownership is row-cyclic
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (world_rank == 0) {
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
    }

    // Broadcast original matrix from rank 0 to all ranks
    MPI_Bcast(A.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank != 0 && validate) {
        // non-root ranks don't need original copy, but keep it empty
    }

    // Start timed parallel Cholesky (synchronized across ranks)
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    bool error_flag = false;

    for (size_t i = 0; i < n; ++i) {
        int owner = (int)(i % world_size);

        if (world_rank == owner) {
            // compute row i elements A[i*n + j] for j = 0..i
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                if (j > 0) {
                    // compute dot product of length j: A[i*n + 0..j-1] and A[j*n + 0..j-1]
                    sum = compute_dot(&A[i * n], &A[j * n], (int)j);
                }

                if (i == j) {
                    double val = A[j * n + j] - sum;
                    if (val <= 0.0) {
                        if (world_rank == 0) printf("Error: Matrix not positive definite at %zu\n", j);
                        error_flag = true;
                    }
                    A[j * n + j] = sqrt(val);
                } else {
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
            }

            // zero upper triangular entries for row i
            for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
        }

        // Broadcast the computed row segment [0..i] from owner to all ranks for next iterations
        MPI_Bcast(A.data() + i * n, (int)(i + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // After broadcast, continue to next i
        if (error_flag) break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (world_rank == 0) {
        long ms = (long)((t1 - t0) * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / ((t1 - t0)) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Only rank 0 prints or validates
    if (printResults && world_rank == 0) {
        print_results(A, "CholeskyL");
    }

    if (validate && world_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return error_flag ? 1 : 0;
}
