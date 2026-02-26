#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

static inline size_t rows_for_rank(const size_t n, const int rank, const int size) {
    const size_t base = n / size;
    const size_t rem = n % size;
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

static inline size_t row_start_for_rank(const size_t n, const int rank, const int size) {
    const size_t base = n / size;
    const size_t rem = n % size;
    return base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
}

static inline int owner_for_row(const size_t n, const size_t row, const int size) {
    const size_t base = n / size;
    const size_t rem = n % size;
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>((row - cutoff) / base + rem);
}

__global__ void updateColumnKernel(double* A, const double* row_k, const size_t n,
                                   const size_t k, const size_t row_start,
                                   const size_t local_rows) {
    const size_t local_i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_i >= local_rows) {
        return;
    }
    const size_t global_i = row_start + local_i;
    if (global_i <= k) {
        return;
    }
    double sum = 0.0;
    const double* row_i = A + local_i * n;
    for (size_t j = 0; j < k; ++j) {
        sum += row_i[j] * row_k[j];
    }
    A[local_i * n + k] = (row_i[k] - sum) / row_k[k];
}

bool choleskyDecompositionHybrid(std::vector<double>& A_local, const size_t n,
                                 const size_t row_start, const size_t local_rows,
                                 const int rank, const int size,
                                 double* d_A, double* d_row_k) {
    (void)A_local;
    std::vector<double> row_k(n, 0.0);
    for (size_t k = 0; k < n; ++k) {
        const int owner = owner_for_row(n, k, size);
        int local_ok = 1;

        if (rank == owner) {
            const size_t local_index = k - row_start;
            CUDA_CHECK(cudaMemcpy(row_k.data(), d_A + local_index * n,
                                  (k + 1) * sizeof(double), cudaMemcpyDeviceToHost));

            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) schedule(static)
            for (size_t j = 0; j < k; ++j) {
                sum += row_k[j] * row_k[j];
            }

            const double val = row_k[k] - sum;
            if (val <= 0.0) {
                local_ok = 0;
                row_k[k] = 0.0;
            } else {
                row_k[k] = sqrt(val);
            }
            CUDA_CHECK(cudaMemcpy(d_A + local_index * n + k, &row_k[k],
                                  sizeof(double), cudaMemcpyHostToDevice));
        }

        MPI_Bcast(row_k.data(), static_cast<int>(k + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!global_ok) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            }
            return false;
        }

        CUDA_CHECK(cudaMemcpy(d_row_k, row_k.data(), (k + 1) * sizeof(double),
                              cudaMemcpyHostToDevice));

        if (local_rows > 0 && k + 1 < n) {
            const int threads = 256;
            const int blocks = static_cast<int>((local_rows + threads - 1) / threads);
            updateColumnKernel<<<blocks, threads>>>(d_A, d_row_k, n, k, row_start, local_rows);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "Error: MPI does not provide required thread support\n");
        }
        MPI_Finalize();
        return 1;
    }

    size_t n = 512;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int parseOk = 1;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseOk = 0;
            }
        }
        if (!parseOk || showHelp) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOk) {
        MPI_Finalize();
        return 1;
    }
    if (showHelp) {
        MPI_Finalize();
        return 0;
    }

    uint64_t n64 = static_cast<uint64_t>(n);
    MPI_Bcast(&n64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n64);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: No CUDA devices available\n");
        }
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    const size_t row_start = row_start_for_rank(n, rank, size);
    const size_t local_rows = rows_for_rank(n, rank, size);
    const size_t local_elems = local_rows * n;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
        const size_t rows_r = rows_for_rank(n, r, size);
        counts[r] = static_cast<int>(rows_r * n);
        displs[r] = static_cast<int>(row_start_for_rank(n, r, size) * n);
    }

    std::vector<double> A_local(local_elems);
    std::vector<double> A_full;
    std::vector<double> A_orig;

    if (rank == 0) {
        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, counts.data(), displs.data(),
                 MPI_DOUBLE,
                 local_elems > 0 ? A_local.data() : nullptr, counts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double* d_A = nullptr;
    double* d_row_k = nullptr;
    const size_t alloc_elems = std::max<size_t>(local_elems, 1);
    CUDA_CHECK(cudaMalloc(&d_A, alloc_elems * sizeof(double)));
    if (local_elems > 0) {
        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), local_elems * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_row_k, std::max<size_t>(n, 1) * sizeof(double)));

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool success = choleskyDecompositionHybrid(A_local, n, row_start, local_rows,
                                                     rank, size, d_A, d_row_k);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_row_k));
        MPI_Finalize();
        return 1;
    }

    if (local_elems > 0) {
        CUDA_CHECK(cudaMemcpy(A_local.data(), d_A, local_elems * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for schedule(static)
    for (size_t local_i = 0; local_i < local_rows; ++local_i) {
        const size_t global_i = row_start + local_i;
        double* row = A_local.data() + local_i * n;
        for (size_t j = global_i + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_time * 1000.0);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / max_time / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (validate || printResults) {
        if (rank == 0) {
            A_full.assign(n * n, 0.0);
        }
        MPI_Gatherv(local_elems > 0 ? A_local.data() : nullptr, counts[rank], MPI_DOUBLE,
                    rank == 0 ? A_full.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exit_code = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(A_full, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_row_k));

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
