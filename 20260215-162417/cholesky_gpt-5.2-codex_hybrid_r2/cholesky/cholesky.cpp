#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {
inline void cudaCheck(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(val) cudaCheck((val), __FILE__, __LINE__)

__global__ void rank1UpdateKernel(double* A, const double* col, int n, int row_start, int rows_local, int k) {
    const int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    if (local_i >= rows_local || j >= n) {
        return;
    }
    const int global_i = row_start + local_i;
    if (global_i <= k || j > global_i) {
        return;
    }
    const double update = col[global_i] * col[j];
    A[static_cast<size_t>(local_i) * n + j] -= update;
}

int ownerOfRow(int row, int base, int rem) {
    const int cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return row / (base + 1);
    }
    if (base == 0) {
        return 0;
    }
    return rem + (row - cutoff) / base;
}
} // namespace

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

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

#pragma omp parallel for reduction(max : maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

bool choleskyDecompositionHybrid(double* d_A,
                                 const size_t n,
                                 const int row_start,
                                 const int row_count,
                                 const std::vector<int>& row_counts,
                                 const std::vector<int>& row_displs,
                                 const int rank,
                                 const int size) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Error: matrix dimension exceeds int limits for GPU kernels.\n");
        }
        return false;
    }

    const int n_int = static_cast<int>(n);
    const int base = n_int / size;
    const int rem = n_int % size;

    std::vector<double> col_local(static_cast<size_t>(row_count));
    std::vector<double> col_full(n);

    double* d_col = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col, n * sizeof(double)));

    bool ok = true;
    for (int k = 0; k < n_int; ++k) {
        if (row_count > 0) {
            CUDA_CHECK(cudaMemcpy2D(col_local.data(),
                                    sizeof(double),
                                    d_A + k,
                                    n * sizeof(double),
                                    sizeof(double),
                                    row_count,
                                    cudaMemcpyDeviceToHost));
        }

        double diag = 0.0;
        const int owner = ownerOfRow(k, base, rem);
        if (rank == owner) {
            const int local_k = k - row_start;
            diag = col_local[local_k];
            if (diag <= 0.0) {
                ok = false;
            } else {
                diag = std::sqrt(diag);
                col_local[local_k] = diag;
            }
        }

        MPI_Bcast(&diag, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (diag <= 0.0) {
            ok = false;
        }
        if (!ok) {
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < row_count; ++i) {
            const int global_i = row_start + i;
            if (global_i > k) {
                col_local[i] /= diag;
            } else if (global_i == k) {
                col_local[i] = diag;
            }
        }

        if (row_count > 0) {
            CUDA_CHECK(cudaMemcpy2D(d_A + k,
                                    n * sizeof(double),
                                    col_local.data(),
                                    sizeof(double),
                                    sizeof(double),
                                    row_count,
                                    cudaMemcpyHostToDevice));
        }

        MPI_Allgatherv(col_local.data(),
                       row_count,
                       MPI_DOUBLE,
                       col_full.data(),
                       row_counts.data(),
                       row_displs.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_col, col_full.data(), n * sizeof(double), cudaMemcpyHostToDevice));

        if (row_count > 0 && k + 1 < n_int) {
            const dim3 block(16, 16);
            const dim3 grid((n_int - (k + 1) + block.x - 1) / block.x,
                            (row_count + block.y - 1) / block.y);
            rank1UpdateKernel<<<grid, block>>>(d_A, d_col, n_int, row_start, row_count, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaFree(d_col));
    return ok;
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    int validate = 0;
    int printResults = 0;
    bool showHelp = false;
    bool parseOk = true;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseOk = false;
                break;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        }
    }

    int showHelpInt = showHelp ? 1 : 0;
    int parseOkInt = parseOk ? 1 : 0;
    MPI_Bcast(&showHelpInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseOkInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (showHelpInt || !parseOkInt) {
        MPI_Finalize();
        return showHelpInt ? 0 : 1;
    }

    uint64_t n64 = static_cast<uint64_t>(n);
    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n64);

    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Error: matrix size too large for this implementation.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: no CUDA devices detected.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const int n_int = static_cast<int>(n);
    const int base = n_int / size;
    const int rem = n_int % size;
    const int row_count = base + (rank < rem ? 1 : 0);
    const int row_start = rank * base + std::min(rank, rem);

    std::vector<int> row_counts(size);
    std::vector<int> row_displs(size);
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    int offset_rows = 0;
    int offset_elems = 0;
    for (int r = 0; r < size; ++r) {
        const int rows = base + (r < rem ? 1 : 0);
        row_counts[r] = rows;
        row_displs[r] = offset_rows;
        counts[r] = rows * n_int;
        displs[r] = offset_elems;
        offset_rows += rows;
        offset_elems += counts[r];
    }

    std::vector<double> A_full;
    std::vector<double> A_orig;
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        A_full.resize(static_cast<size_t>(n_int) * n_int);
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
    }

    std::vector<double> A_local(static_cast<size_t>(row_count) * n_int);
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 A_local.data(),
                 row_count * n_int,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    double* d_A = nullptr;
    if (row_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(row_count) * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_A,
                              A_local.data(),
                              static_cast<size_t>(row_count) * n * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    bool success = choleskyDecompositionHybrid(d_A,
                                               n,
                                               row_start,
                                               row_count,
                                               row_counts,
                                               row_displs,
                                               rank,
                                               size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double local_time = MPI_Wtime() - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int success_int = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &success_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!success_int) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        if (row_count > 0) {
            CUDA_CHECK(cudaFree(d_A));
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_time * 1000.0);
        const double ops = static_cast<double>(n_int) * n_int * n_int / 3.0;
        const double gflops = max_time > 0.0 ? ops / max_time / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exit_code = 0;
    if (validate || printResults) {
        if (row_count > 0) {
            CUDA_CHECK(cudaMemcpy(A_local.data(),
                                  d_A,
                                  static_cast<size_t>(row_count) * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<double> A_result;
        if (rank == 0) {
            A_result.resize(static_cast<size_t>(n_int) * n_int);
        }

        MPI_Gatherv(A_local.data(),
                    row_count * n_int,
                    MPI_DOUBLE,
                    rank == 0 ? A_result.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A_result[i * n + j] = 0.0;
                }
            }

            if (printResults) {
                print_results(A_result, "CholeskyL");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateCholesky(A_result, A_orig, n);
                if (valid) {
                    printf("Validation: PASSED\n");
                    exit_code = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exit_code = 1;
                }
            }
        }
    }

    if (row_count > 0) {
        CUDA_CHECK(cudaFree(d_A));
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
