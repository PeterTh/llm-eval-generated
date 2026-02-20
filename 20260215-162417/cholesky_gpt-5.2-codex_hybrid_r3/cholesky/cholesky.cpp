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

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (blocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {
constexpr size_t kBlockSize = 64;
constexpr int kCudaBlockDim = 16;
} // namespace

static void cudaCheck(cudaError_t err, const char* msg, int rank) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Rank %d CUDA error (%s): %s\n", rank, msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void computeRowDistribution(size_t n, int size, std::vector<int>& rowCounts, std::vector<int>& rowDispls) {
    rowCounts.assign(size, 0);
    rowDispls.assign(size, 0);
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t rows = base + ((static_cast<size_t>(r) < rem) ? 1 : 0);
        rowCounts[r] = static_cast<int>(rows);
        rowDispls[r] = static_cast<int>(offset);
        offset += rows;
    }
}

static bool factorizeDiagonalBlock(std::vector<double>& A, size_t n, size_t k, size_t bk) {
    for (size_t j = 0; j < bk; ++j) {
        const size_t jj = k + j;
        double sum = A[jj * n + jj];
        for (size_t p = k; p < jj; ++p) {
            sum -= A[jj * n + p] * A[jj * n + p];
        }
        if (sum <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", jj);
            return false;
        }
        A[jj * n + jj] = sqrt(sum);

        #pragma omp parallel for schedule(static)
        for (size_t i = jj + 1; i < k + bk; ++i) {
            double s = A[i * n + jj];
            for (size_t p = k; p < jj; ++p) {
                s -= A[i * n + p] * A[jj * n + p];
            }
            A[i * n + jj] = s / A[jj * n + jj];
        }
    }
    return true;
}

static void computePanel(std::vector<double>& A, size_t n, size_t k, size_t bk, size_t startRow, size_t endRow) {
    const size_t rowStart = std::max(startRow, k + bk);
    if (rowStart >= endRow) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = rowStart; i < endRow; ++i) {
        for (size_t j = k; j < k + bk; ++j) {
            double sum = A[i * n + j];
            for (size_t p = k; p < j; ++p) {
                sum -= A[i * n + p] * A[j * n + p];
            }
            A[i * n + j] = sum / A[j * n + j];
        }
    }
}

__global__ void trailingUpdateKernel(double* A, size_t n, size_t k, size_t bk, size_t rowStart, size_t rowEnd) {
    const size_t i = rowStart + static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t j = k + bk + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rowEnd || j >= n || j > i) {
        return;
    }

    double sum = 0.0;
    for (size_t p = k; p < k + bk; ++p) {
        sum += A[i * n + p] * A[j * n + p];
    }
    A[i * n + j] -= sum;
}

static bool choleskyDecompositionHybrid(std::vector<double>& A, const size_t n, int rank, int size, MPI_Comm comm) {
    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    computeRowDistribution(n, size, rowCounts, rowDispls);

    const int localRows = rowCounts[rank];
    const size_t startRow = static_cast<size_t>(rowDispls[rank]);
    const size_t endRow = startRow + static_cast<size_t>(localRows);

    std::vector<int> countsElems(size, 0);
    std::vector<int> displsElems(size, 0);
    for (int r = 0; r < size; ++r) {
        countsElems[r] = rowCounts[r] * static_cast<int>(n);
        displsElems[r] = rowDispls[r] * static_cast<int>(n);
    }

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices detected.\n");
        }
        MPI_Abort(comm, 1);
    }
    const int device = rank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice", rank);

    double* d_A = nullptr;
    cudaCheck(cudaMalloc(&d_A, n * n * sizeof(double)), "cudaMalloc", rank);

    for (size_t k = 0; k < n; k += kBlockSize) {
        const size_t bk = std::min(kBlockSize, n - k);
        int ok = 1;
        if (rank == 0) {
            ok = factorizeDiagonalBlock(A, n, k, bk) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
        if (!ok) {
            cudaCheck(cudaFree(d_A), "cudaFree", rank);
            return false;
        }

        MPI_Bcast(A.data() + k * n, static_cast<int>(bk * n), MPI_DOUBLE, 0, comm);

        computePanel(A, n, k, bk, startRow, endRow);

        const int localCount = localRows * static_cast<int>(n);
        MPI_Allgatherv(A.data() + startRow * n, localCount, MPI_DOUBLE,
                       A.data(), countsElems.data(), displsElems.data(), MPI_DOUBLE, comm);

        const size_t updateRowStart = std::max(startRow, k + bk);
        const size_t updateRowEnd = endRow;
        if (updateRowStart < updateRowEnd && k + bk < n) {
            const size_t numRows = updateRowEnd - updateRowStart;
            const size_t numCols = n - (k + bk);

            cudaCheck(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice),
                      "cudaMemcpy H2D", rank);

            dim3 block(kCudaBlockDim, kCudaBlockDim);
            dim3 grid((numCols + block.x - 1) / block.x,
                      (numRows + block.y - 1) / block.y);
            trailingUpdateKernel<<<grid, block>>>(d_A, n, k, bk, updateRowStart, updateRowEnd);
            cudaCheck(cudaGetLastError(), "trailingUpdateKernel", rank);
            cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);

            cudaCheck(cudaMemcpy(A.data() + updateRowStart * n, d_A + updateRowStart * n,
                                 numRows * n * sizeof(double), cudaMemcpyDeviceToHost),
                      "cudaMemcpy D2H", rank);
        }

        MPI_Allgatherv(A.data() + startRow * n, localCount, MPI_DOUBLE,
                       A.data(), countsElems.data(), displsElems.data(), MPI_DOUBLE, comm);
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = startRow; i < endRow; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    const int localCount = localRows * static_cast<int>(n);
    MPI_Allgatherv(A.data() + startRow * n, localCount, MPI_DOUBLE,
                   A.data(), countsElems.data(), displsElems.data(), MPI_DOUBLE, comm);

    cudaCheck(cudaFree(d_A), "cudaFree", rank);
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
    bool validate = false;
    bool printResults = false;
    int shouldExit = 0;
    int exitCode = 0;

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
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 1;
            }
        }
    }

    uint64_t nValue = static_cast<uint64_t>(n);
    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;
    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nValue, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    n = static_cast<size_t>(nValue);
    validate = validateFlag != 0;
    printResults = printFlag != 0;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }

    MPI_Bcast(A.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool localSuccess = choleskyDecompositionHybrid(A, n, rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    int successFlag = localSuccess ? 1 : 0;
    int successGlobal = 0;
    MPI_Allreduce(&successFlag, &successGlobal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!successGlobal) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation
    int validFlag = 1;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        validFlag = validateCholesky(A, A_orig, n) ? 1 : 0;
        if (validFlag) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validFlag ? 0 : 1;
}
