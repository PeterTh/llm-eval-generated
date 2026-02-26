#include <algorithm>
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

__global__ void cholesky_update_kernel(double* A, const double* row_k, size_t n, size_t k,
                                       size_t row_offset, size_t local_rows) {
    const size_t local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= local_rows) {
        return;
    }
    const size_t global_i = row_offset + local_i;
    if (global_i <= k) {
        return;
    }
    const size_t base = local_i * n;
    double sum = 0.0;
    for (size_t j = 0; j < k; ++j) {
        sum += A[base + j] * row_k[j];
    }
    A[base + k] = (A[base + k] - sum) / row_k[k];
}

static inline size_t row_offset_for_rank(int rank, size_t base, size_t rem) {
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

static inline size_t local_rows_for_rank(int rank, size_t base, size_t rem) {
    return base + (static_cast<size_t>(rank) < rem ? 1u : 0u);
}

static inline int owner_for_row(size_t row, size_t base, size_t rem) {
    if (base == 0) {
        return static_cast<int>(row);
    }
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - cutoff) / base);
}

static bool check_cuda(cudaError_t status, const char* message, MPI_Comm comm, int rank) {
    if (status != cudaSuccess) {
        if (rank == 0) {
            fprintf(stderr, "CUDA error (%s): %s\n", message, cudaGetErrorString(status));
        }
        MPI_Abort(comm, 1);
        return false;
    }
    return true;
}

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (unblocked algorithm)
bool choleskyDecompositionHybrid(std::vector<double>& local_A, const size_t n, const size_t row_offset,
                                 const size_t local_rows, const size_t base, const size_t rem,
                                 MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", comm, rank) ||
        device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: No CUDA devices available\n");
        }
        MPI_Abort(comm, 1);
        return false;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    const int device = local_rank % device_count;
    if (!check_cuda(cudaSetDevice(device), "cudaSetDevice", comm, rank)) {
        return false;
    }

    double* d_A = nullptr;
    double* d_row_k = nullptr;
    const size_t bytes = local_rows * n * sizeof(double);
    if (local_rows > 0) {
        if (!check_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc matrix", comm, rank)) {
            return false;
        }
        if (!check_cuda(cudaMemcpy(d_A, local_A.data(), bytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy matrix", comm, rank)) {
            cudaFree(d_A);
            return false;
        }
    }
    if (!check_cuda(cudaMalloc(&d_row_k, n * sizeof(double)), "cudaMalloc row buffer", comm, rank)) {
        if (d_A) {
            cudaFree(d_A);
        }
        return false;
    }

    std::vector<double> row_k(n, 0.0);

    for (size_t k = 0; k < n; ++k) {
        const int owner = owner_for_row(k, base, rem);
        int local_ok = 1;

        if (rank == owner) {
            const size_t local_index = k - row_offset;
            if (!check_cuda(cudaMemcpy(local_A.data() + local_index * n, d_A + local_index * n,
                                        n * sizeof(double), cudaMemcpyDeviceToHost),
                            "cudaMemcpy pivot row", comm, rank)) {
                local_ok = 0;
            } else {
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (size_t j = 0; j < k; ++j) {
                    const double v = local_A[local_index * n + j];
                    sum += v * v;
                }
                const double val = local_A[local_index * n + k] - sum;
                if (val <= 0.0) {
                    local_ok = 0;
                } else {
                    local_A[local_index * n + k] = std::sqrt(val);
                    std::fill(row_k.begin(), row_k.end(), 0.0);
                    for (size_t j = 0; j <= k; ++j) {
                        row_k[j] = local_A[local_index * n + j];
                    }
                    for (size_t j = k + 1; j < n; ++j) {
                        local_A[local_index * n + j] = 0.0;
                    }
                    if (!check_cuda(cudaMemcpy(d_A + local_index * n, local_A.data() + local_index * n,
                                                n * sizeof(double), cudaMemcpyHostToDevice),
                                    "cudaMemcpy pivot update", comm, rank)) {
                        local_ok = 0;
                    }
                }
            }
        }

        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, comm);
        if (!global_ok) {
            if (rank == 0) {
                fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", k);
            }
            if (d_A) {
                cudaFree(d_A);
            }
            cudaFree(d_row_k);
            return false;
        }

        MPI_Bcast(row_k.data(), static_cast<int>(n), MPI_DOUBLE, owner, comm);

        if (!check_cuda(cudaMemcpy(d_row_k, row_k.data(), n * sizeof(double), cudaMemcpyHostToDevice),
                        "cudaMemcpy row buffer", comm, rank)) {
            if (d_A) {
                cudaFree(d_A);
            }
            cudaFree(d_row_k);
            return false;
        }

        if (local_rows > 0) {
            const dim3 block(256);
            const dim3 grid((local_rows + block.x - 1) / block.x);
            cholesky_update_kernel<<<grid, block>>>(d_A, d_row_k, n, k, row_offset, local_rows);
            if (!check_cuda(cudaGetLastError(), "cuda kernel launch", comm, rank)) {
                if (d_A) {
                    cudaFree(d_A);
                }
                cudaFree(d_row_k);
                return false;
            }
        }

        if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", comm, rank)) {
            if (d_A) {
                cudaFree(d_A);
            }
            cudaFree(d_row_k);
            return false;
        }
    }

    if (local_rows > 0) {
        if (!check_cuda(cudaMemcpy(local_A.data(), d_A, bytes, cudaMemcpyDeviceToHost),
                        "cudaMemcpy result", comm, rank)) {
            cudaFree(d_A);
            cudaFree(d_row_k);
            return false;
        }
        cudaFree(d_A);
    }
    cudaFree(d_row_k);

    #pragma omp parallel for
    for (size_t local_i = 0; local_i < local_rows; ++local_i) {
        const size_t global_i = row_offset + local_i;
        for (size_t j = global_i + 1; j < n; ++j) {
            local_A[local_i * n + j] = 0.0;
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
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
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

void compute_counts_displs(const size_t n, const int world_size,
                           std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(world_size);
    displs.resize(world_size);
    const size_t base = n / static_cast<size_t>(world_size);
    const size_t rem = n % static_cast<size_t>(world_size);
    size_t offset = 0;
    for (int rank = 0; rank < world_size; ++rank) {
        const size_t rows = local_rows_for_rank(rank, base, rem);
        const size_t elems = rows * n;
        counts[rank] = static_cast<int>(elems);
        displs[rank] = static_cast<int>(offset);
        offset += elems;
    }
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int parse_ok = 1;
    int show_help = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                show_help = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parse_ok = 0;
            }
        }
        if (!parse_ok || show_help) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&show_help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok) {
        MPI_Finalize();
        return 1;
    }
    if (show_help) {
        MPI_Finalize();
        return 0;
    }

    uint64_t n_value = static_cast<uint64_t>(n);
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&n_value, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n_value);
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    const size_t base = n / static_cast<size_t>(world_size);
    const size_t rem = n % static_cast<size_t>(world_size);
    const size_t local_rows = local_rows_for_rank(rank, base, rem);
    const size_t row_offset = row_offset_for_rank(rank, base, rem);

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

    std::vector<double> local_A(local_rows * n);
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        compute_counts_displs(n, world_size, counts, displs);
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 counts.empty() ? nullptr : counts.data(),
                 displs.empty() ? nullptr : displs.data(),
                 MPI_DOUBLE,
                 local_A.data(),
                 static_cast<int>(local_rows * n),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    const bool success = choleskyDecompositionHybrid(local_A, n, row_offset, local_rows, base, rem, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    int success_i = success ? 1 : 0;
    int global_success = 0;
    MPI_Allreduce(&success_i, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!global_success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Gatherv(local_A.data(),
                static_cast<int>(local_rows * n),
                MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr,
                counts.empty() ? nullptr : counts.data(),
                displs.empty() ? nullptr : displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = (end - start) * 1000.0;
        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
        }
    }

    int validation_ok = 1;
    if (validate) {
        if (rank == 0) {
            const bool valid = validateCholesky(A_full, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            validation_ok = valid ? 1 : 0;
        }
        MPI_Bcast(&validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!validation_ok) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
