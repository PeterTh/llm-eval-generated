#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Row blocks belong to MPI ranks. Each rank updates its trailing rows on its GPU.
// A panel is exchanged once, so the cubic work stays local to the accelerators.
constexpr int blockSize = 32;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void factorDiagonal(double* a, double* diagonal, int n, int localRow,
                               int k, int width, int* failed) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int j = 0; j < width; ++j) {
        double value = a[(localRow + j) * n + k + j];
        if (!(value > 0.0)) { *failed = k + j + 1; return; }
        double pivot = sqrt(value);
        a[(localRow + j) * n + k + j] = pivot;
        for (int i = j + 1; i < width; ++i)
            a[(localRow + i) * n + k + j] /= pivot;
        for (int i = j + 1; i < width; ++i) {
            double li = a[(localRow + i) * n + k + j];
            for (int c = j + 1; c <= i; ++c)
                a[(localRow + i) * n + k + c] -= li * a[(localRow + c) * n + k + j];
        }
    }
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j)
            diagonal[i * blockSize + j] = j <= i ? a[(localRow + i) * n + k + j] : 0.0;
}

__global__ void solvePanel(double* a, const double* diagonal, int n, int firstRow,
                           int localRows, int k, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= localRows || firstRow + row < k + width) return;
    for (int j = 0; j < width; ++j) {
        double value = a[row * n + k + j];
        for (int t = 0; t < j; ++t)
            value -= a[row * n + k + t] * diagonal[j * blockSize + t];
        a[row * n + k + j] = value / diagonal[j * blockSize + j];
    }
}

__global__ void updateTrailing(double* a, const double* panel, int n, int firstRow,
                               int localRows, int k, int width) {
    int col = k + width + blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= localRows || col > firstRow + row || col >= n || firstRow + row < k + width) return;
    double sum = 0.0;
    for (int j = 0; j < width; ++j)
        sum += panel[(firstRow + row) * blockSize + j] * panel[col * blockSize + j];
    a[row * n + col] -= sum;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    const int N = static_cast<int>(n);
    const int tiles = (N + blockSize - 1) / blockSize;
    std::vector<int> starts(ranks + 1), counts(ranks), displacements(ranks);
    for (int r = 0; r <= ranks; ++r)
        starts[r] = std::min(N, (tiles * r / ranks) * blockSize);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = (starts[r + 1] - starts[r]) * N;
        displacements[r] = starts[r] * N;
    }
    int first = starts[rank], rows = starts[rank + 1] - first;
    std::vector<double> local(static_cast<size_t>(rows) * n);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *deviceA = nullptr, *deviceDiagonal = nullptr, *devicePanel = nullptr;
    int* deviceFailed = nullptr;
    cudaCheck(cudaMalloc(&deviceA, std::max<size_t>(1, local.size()) * sizeof(double)), "allocate matrix");
    cudaCheck(cudaMalloc(&deviceDiagonal, blockSize * blockSize * sizeof(double)), "allocate diagonal");
    cudaCheck(cudaMalloc(&devicePanel, std::max<size_t>(1, n * blockSize) * sizeof(double)), "allocate panel");
    cudaCheck(cudaMalloc(&deviceFailed, sizeof(int)), "allocate status");
    if (!local.empty()) cudaCheck(cudaMemcpy(deviceA, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice), "copy matrix");
    std::vector<double> diagonal(blockSize * blockSize), panel(n * blockSize);
    std::vector<int> panelCounts(ranks), panelOffsets(ranks);
    bool success = true;
    for (int k = 0; k < N; k += blockSize) {
        int width = std::min(blockSize, N - k);
        int owner = static_cast<int>(std::upper_bound(starts.begin(), starts.end(), k) - starts.begin()) - 1;
        // Empty ranks may share a boundary; upper_bound finds the actual owner.
        int failed = 0;
        if (rank == owner) {
            cudaCheck(cudaMemset(deviceFailed, 0, sizeof(int)), "reset status");
            factorDiagonal<<<1, 1>>>(deviceA, deviceDiagonal, N, k - first, k, width, deviceFailed);
            cudaCheck(cudaGetLastError(), "factor diagonal launch");
            cudaCheck(cudaMemcpy(&failed, deviceFailed, sizeof(int), cudaMemcpyDeviceToHost), "check diagonal");
            if (!failed) cudaCheck(cudaMemcpy(diagonal.data(), deviceDiagonal, diagonal.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy diagonal");
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", failed - 1);
            success = false;
            break;
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(diagonal.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(deviceDiagonal, diagonal.data(), diagonal.size() * sizeof(double), cudaMemcpyHostToDevice), "broadcast diagonal to GPU");
        if (rows) {
            solvePanel<<<(rows + 127) / 128, 128>>>(deviceA, deviceDiagonal, N, first, rows, k, width);
            cudaCheck(cudaGetLastError(), "solve panel launch");
        }
        int sendFirst = std::max(first, k + width);
        int sendRows = std::max(0, starts[rank + 1] - sendFirst);
        if (sendRows) cudaCheck(cudaMemcpy2D(panel.data() + static_cast<size_t>(sendFirst) * blockSize,
                                             blockSize * sizeof(double), deviceA + static_cast<size_t>(sendFirst - first) * N + k,
                                             N * sizeof(double), width * sizeof(double), sendRows,
                                             cudaMemcpyDeviceToHost), "copy panel from GPU");
        for (int r = 0; r < ranks; ++r) {
            int begin = std::max(starts[r], k + width);
            panelCounts[r] = std::max(0, starts[r + 1] - begin) * blockSize;
            panelOffsets[r] = begin * blockSize;
        }
        // The final, short panel still uses a fixed stride for MPI and GPU indexing.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, panel.data(), panelCounts.data(),
                       panelOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        if (k + width < N && rows) {
            cudaCheck(cudaMemcpy(devicePanel, panel.data(), panel.size() * sizeof(double), cudaMemcpyHostToDevice), "copy panel to GPU");
            dim3 threads(32, 8);
            dim3 blocks((N - k - width + 31) / 32, (rows + 7) / 8);
            updateTrailing<<<blocks, threads>>>(deviceA, devicePanel, N, first, rows, k, width);
            cudaCheck(cudaGetLastError(), "update trailing matrix launch");
        }
    }
    if (success) {
        if (!local.empty()) cudaCheck(cudaMemcpy(local.data(), deviceA, local.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy result");
        MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE, rank == 0 ? A.data() : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i)
                std::fill(A.begin() + static_cast<size_t>(i) * n + i + 1,
                          A.begin() + static_cast<size_t>(i + 1) * n, 0.0);
        }
    }
    cudaCheck(cudaFree(deviceFailed), "free status");
    cudaCheck(cudaFree(devicePanel), "free panel");
    cudaCheck(cudaFree(deviceDiagonal), "free diagonal");
    cudaCheck(cudaFree(deviceA), "free matrix");
    return success;
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank = 0, gpuCount = 0;
    MPI_Comm_rank(shared, &localRank);
    cudaCheck(cudaGetDeviceCount(&gpuCount), "count GPUs");
    if (gpuCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % gpuCount), "select GPU");
    MPI_Comm_free(&shared);
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    if (n > static_cast<size_t>(INT_MAX) || n * n > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Matrix size exceeds MPI count range\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(rank == 0 ? n * n : 0);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }
    
    if (validate && rank == 0) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, ranks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
