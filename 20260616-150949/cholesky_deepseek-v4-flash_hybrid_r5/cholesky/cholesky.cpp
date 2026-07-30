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

// ============================================================
// CUDA error checking macro
// ============================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ============================================================
// CUDA Kernels
// ============================================================

// Compute diagonal element L[j,j] for the pivot row
// rowPtr points to the start of row j on the device
__global__ void choleskyDiagonalKernel(double* rowPtr, int j) {
    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
        sum += rowPtr[k] * rowPtr[k];
    }
    rowPtr[j] = sqrt(rowPtr[j] - sum);
}

// Update column j for all local rows below the pivot (globalRow > j)
// Each thread handles one local row
// d_localRows: device pointer to the local rows (myNumRows * n)
// d_pivotRow: device pointer to the broadcast pivot row (elements 0..j)
__global__ void choleskyColumnKernel(
    double* d_localRows,
    const double* d_pivotRow,
    int j,
    int n,
    int myStartRow,
    int myNumRows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= myNumRows) return;

    int globalRow = myStartRow + tid;
    if (globalRow <= j) return; // skip the pivot row and rows above

    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
        sum += d_localRows[tid * n + k] * d_pivotRow[k];
    }
    d_localRows[tid * n + j] = (d_localRows[tid * n + j] - sum) / d_pivotRow[j];
}

// ============================================================
// Row distribution helpers for MPI
// ============================================================

// Compute the number of rows each rank gets (block distribution)
static void computeRowDistribution(
    int n, int numRanks,
    std::vector<int>& sendcounts,
    std::vector<int>& displs)
{
    int base = n / numRanks;
    int rem = n % numRanks;
    sendcounts.resize(numRanks);
    displs.resize(numRanks);
    int offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        sendcounts[r] = rows * n;
        displs[r] = offset;
        offset += rows * n;
    }
}

// Return the rank that owns a given global row
static inline int getRowOwner(int row, int n, int numRanks) {
    int base = n / numRanks;
    int rem = n % numRanks;
    int acc = 0;
    for (int r = 0; r < numRanks; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        acc += rows;
        if (row < acc) return r;
    }
    return numRanks - 1;
}

// ============================================================
// Matrix generation (rank 0 only, parallelized with OpenMP)
// ============================================================

// Generate a symmetric positive definite matrix on rank 0
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add diagonal to make it strictly positive definite

    std::vector<double> B(n * n);

    // Generate random matrix B (parallelized with OpenMP)
    // Each row uses its own seed for thread safety
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        unsigned int seed = 42u + (unsigned int)i * 7919u;
        for (size_t j = 0; j < n; ++j) {
            B[i * n + j] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    }

    // Compute A = B * B^T (parallelized with OpenMP)
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

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += (double)n;
    }
}

// ============================================================
// Hybrid MPI+OpenMP+CUDA Cholesky Decomposition
// ============================================================
// Each MPI rank owns a contiguous block of rows.
// The algorithm proceeds column-by-column:
//   1. The rank owning the pivot row computes L[j,j] on GPU
//   2. The pivot row (elements 0..j) is broadcast via MPI
//   3. All ranks update their local rows for column j on GPU
// ============================================================

static bool choleskyDecompositionHybrid(
    double* d_localMatrix,   // device pointer: [myNumRows * n] doubles
    double* d_pivotRow,      // device pointer: buffer for pivot row (size n)
    std::vector<double>& h_pivotRow,  // host buffer for MPI broadcast
    int n,
    int numRanks,
    int rank,
    int myStartRow,
    int myNumRows)
{
    if (myNumRows == 0) {
        // Rank has no rows; still needs to participate in broadcasts
        for (int j = 0; j < n; ++j) {
            int owner = getRowOwner(j, n, numRanks);
            MPI_Bcast(h_pivotRow.data(), j + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        }
        return true;
    }

    for (int j = 0; j < n; ++j) {
        int owner = getRowOwner(j, n, numRanks);

        // --- Step 1: diagonal computation (by the owner) ---
        if (rank == owner) {
            int localJ = j - myStartRow;
            double* rowPtr = d_localMatrix + localJ * n;

            // Launch single-thread kernel to compute L[j,j]
            choleskyDiagonalKernel<<<1, 1>>>(rowPtr, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy row j elements 0..j from device to host for broadcast
            CUDA_CHECK(cudaMemcpy(
                h_pivotRow.data(), rowPtr,
                (j + 1) * sizeof(double),
                cudaMemcpyDeviceToHost));
        }

        // --- Step 2: broadcast pivot row (elements 0..j) ---
        MPI_Bcast(h_pivotRow.data(), j + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // --- Step 3: copy pivot row to all GPUs ---
        CUDA_CHECK(cudaMemcpy(
            d_pivotRow, h_pivotRow.data(),
            (j + 1) * sizeof(double),
            cudaMemcpyHostToDevice));

        // --- Step 4: column update kernel on GPU ---
        // Update rows i > j in this rank's local block
        const int blockSize = 256;
        int numThreads = myNumRows;
        int gridSize = (numThreads + blockSize - 1) / blockSize;

        if (gridSize > 0) {
            choleskyColumnKernel<<<gridSize, blockSize>>>(
                d_localMatrix, d_pivotRow, j, n, myStartRow, myNumRows);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

// ============================================================
// Validation (rank 0 only, parallelized with OpenMP)
// ============================================================

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (parallelized with OpenMP)
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

    // Compare with original (parallelized with OpenMP)
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max:maxError,relError)
    for (size_t idx = 0; idx < n * n; ++idx) {
        const double error = fabs(reconstructed[idx] - A_orig[idx]);
        if (error > maxError) maxError = error;

        const double rel = error / (fabs(A_orig[idx]) + 1e-10);
        if (rel > relError) relError = rel;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse command line arguments (all ranks parse the same args)
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Set up CUDA device (round-robin across ranks)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0)
            fprintf(stderr, "Error: No CUDA-capable devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = rank % numDevices;
    CUDA_CHECK(cudaSetDevice(dev));

    // Compute row distribution
    std::vector<int> sendcounts, displs;
    computeRowDistribution(n, numRanks, sendcounts, displs);
    int myStartRow = displs[rank] / (int)n;
    int myNumRows = sendcounts[rank] / (int)n;

    // Print banner from rank 0
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices available: %d\n", numDevices);
        printf("Rank %d using GPU %d (rows %d..%d)\n",
               rank, dev, myStartRow, myStartRow + myNumRows - 1);
    } else {
        printf("Rank %d using GPU %d (rows %d..%d)\n",
               rank, dev, myStartRow, myStartRow + myNumRows - 1);
    }

    // Allocate host buffers
    std::vector<double> A_local(myNumRows * (size_t)n);
    std::vector<double> h_pivotRow(n);
    std::vector<double> A_orig;  // only populated on rank 0 if validate

    // ---- Generate matrix (rank 0) and scatter to all ranks ----
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        std::vector<double> A_full(n * n);
        generatePositiveDefiniteMatrix(A_full, n);

        // Save original for validation (rank 0 only)
        if (validate) {
            A_orig = A_full;
        }

        // Scatter rows to all ranks
        MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(),
                      MPI_DOUBLE,
                      A_local.data(), myNumRows * (int)n, MPI_DOUBLE,
                      0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                      A_local.data(), myNumRows * (int)n, MPI_DOUBLE,
                      0, MPI_COMM_WORLD);
    }

    // ---- Allocate GPU memory ----
    double* d_localMatrix = nullptr;
    double* d_pivotRow = nullptr;
    size_t localSize = (size_t)myNumRows * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_localMatrix, localSize));
    CUDA_CHECK(cudaMalloc(&d_pivotRow, n * sizeof(double)));

    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_localMatrix, A_local.data(), localSize,
                           cudaMemcpyHostToDevice));

    // ---- Perform Cholesky decomposition (timed) ----
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    bool success = choleskyDecompositionHybrid(
        d_localMatrix, d_pivotRow, h_pivotRow,
        n, numRanks, rank, myStartRow, myNumRows);

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(d_localMatrix));
        CUDA_CHECK(cudaFree(d_pivotRow));
        MPI_Finalize();
        return 1;
    }

    double duration = endTime - startTime;  // seconds

    // ---- Copy results back from GPU to host ----
    CUDA_CHECK(cudaMemcpy(A_local.data(), d_localMatrix, localSize,
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_localMatrix));
    CUDA_CHECK(cudaFree(d_pivotRow));

    // ---- Gather all rows at rank 0 ----
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize(n * n);
    }
    MPI_Gatherv(A_local.data(), myNumRows * (int)n, MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- Rank 0: output results ----
    if (rank == 0) {
        // Zero out upper triangular part (result L is lower triangular)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A_full[i * n + j] = 0.0;
            }
        }

        printf("Computation time: %.0f ms\n", duration * 1000.0);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A_full, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
