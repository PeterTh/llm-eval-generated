#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// One CUDA thread owns one local matrix row.  The row-wise right-looking
// formulation has no writes shared by threads, which makes it a good fit for
// a rank-local GPU while MPI distributes the rows between ranks.
__global__ void factorDiagonal(double* matrix, size_t n, size_t rowStart,
                               size_t rowCount, size_t j) {
    const size_t localRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= rowCount) return;
    const size_t row = rowStart + localRow;
    if (row != j) return;
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k) {
        const double x = matrix[localRow * n + k];
        sum += x * x;
    }
    const double diagonal = matrix[localRow * n + j] - sum;
    matrix[localRow * n + j] = diagonal > 0.0 ? sqrt(diagonal) : -1.0;
}

__global__ void updatePivot(double* matrix, const double* pivot, size_t n,
                            size_t rowStart, size_t rowCount, size_t j) {
    const size_t localRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= rowCount) return;
    const size_t row = rowStart + localRow;
    if (row <= j) return;
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k)
        sum += matrix[localRow * n + k] * pivot[k];
    matrix[localRow * n + j] =
        (matrix[localRow * n + j] - sum) / pivot[j];
}

__global__ void zeroUpper(double* matrix, size_t n, size_t rowStart,
                          size_t rowCount) {
    const size_t localRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= rowCount) return;
    const size_t row = rowStart + localRow;
    for (size_t column = row + 1; column < n; ++column)
        matrix[localRow * n + column] = 0.0;
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// Generate the same deterministic positive definite input as the original.
// The O(n^3) product is threaded with OpenMP; keeping B's generation serial
// preserves the benchmark's original seed and input values exactly.
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(
                relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    return relError <= 1e-6;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n = strtoull(argv[++i], nullptr, 10);
        } else if (!strcmp(argv[i], "-v")) {
            validate = true;
        } else if (!strcmp(argv[i], "-r")) {
            printResults = true;
        } else if (!strcmp(argv[i], "-h")) {
            printUsage(argv[0]);
            return 0;
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Matrix size must be in the range 1..%d\n",
                std::numeric_limits<int>::max());
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    const size_t rowStart = n * static_cast<size_t>(rank) / ranks;
    const size_t rowEnd = n * static_cast<size_t>(rank + 1) / ranks;
    const size_t localRows = rowEnd - rowStart;
    std::vector<double> globalA, original;
    if (rank == 0) {
        globalA.resize(n * n);
        printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu, MPI ranks: %d, OpenMP threads/rank: %d\n",
               n, n, ranks, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(globalA, n);
        if (validate) original = globalA;
    }

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t first = n * static_cast<size_t>(r) / ranks;
        const size_t last = n * static_cast<size_t>(r + 1) / ranks;
        counts[r] = static_cast<int>((last - first) * n);
        displacements[r] = static_cast<int>(first * n);
    }
    std::vector<double> local(localRows * n);
    MPI_Scatterv(rank == 0 ? globalA.data() : nullptr, counts.data(),
                 displacements.data(), MPI_DOUBLE, local.data(),
                 static_cast<int>(local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "discovering GPUs");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "selecting GPU");
    double *deviceMatrix = nullptr, *devicePivot = nullptr;
    cudaCheck(cudaMalloc(&deviceMatrix, local.size() * sizeof(double)), "allocating matrix");
    cudaCheck(cudaMalloc(&devicePivot, n * sizeof(double)), "allocating pivot");
    cudaCheck(cudaMemcpy(deviceMatrix, local.data(), local.size() * sizeof(double),
                         cudaMemcpyHostToDevice), "uploading matrix");
    std::vector<double> pivot(n);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    constexpr int blockSize = 256;
    for (size_t j = 0; j < n; ++j) {
        const int blocks = std::max(1, static_cast<int>((localRows + blockSize - 1) / blockSize));
        factorDiagonal<<<blocks, blockSize>>>(deviceMatrix, n, rowStart,
                                              localRows, j);
        cudaCheck(cudaGetLastError(), "launching pivot kernel");
        cudaCheck(cudaDeviceSynchronize(), "completing pivot kernel");
        if (rowStart <= j && j < rowEnd)
            cudaCheck(cudaMemcpy(&pivot[j], deviceMatrix + (j - rowStart) * n + j,
                                 sizeof(double), cudaMemcpyDeviceToHost), "reading diagonal");
        MPI_Bcast(&pivot[j], 1, MPI_DOUBLE, static_cast<int>(j * ranks / n),
                  MPI_COMM_WORLD);
        if (!(pivot[j] > 0.0)) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            MPI_Finalize();
            return 1;
        }
        if (rowStart <= j && j < rowEnd)
            cudaCheck(cudaMemcpy(pivot.data(), deviceMatrix + (j - rowStart) * n,
                                 n * sizeof(double), cudaMemcpyDeviceToHost), "reading pivot row");
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE,
                  static_cast<int>(j * ranks / n), MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(devicePivot, pivot.data(), n * sizeof(double),
                             cudaMemcpyHostToDevice), "uploading pivot row");
        updatePivot<<<blocks, blockSize>>>(deviceMatrix, devicePivot, n,
                                           rowStart, localRows, j);
        cudaCheck(cudaGetLastError(), "launching trailing update kernel");
        cudaCheck(cudaDeviceSynchronize(), "completing trailing update kernel");
    }
    zeroUpper<<<std::max(1, static_cast<int>((localRows + blockSize - 1) /
                                             blockSize)), blockSize>>>(
        deviceMatrix, n, rowStart, localRows);
    cudaCheck(cudaGetLastError(), "launching upper-triangle cleanup");
    cudaCheck(cudaDeviceSynchronize(), "completing upper-triangle cleanup");
    cudaCheck(cudaMemcpy(local.data(), deviceMatrix, local.size() * sizeof(double),
                         cudaMemcpyDeviceToHost), "downloading factor");
    cudaFree(devicePivot);
    cudaFree(deviceMatrix);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Gatherv(local.data(), counts[0], MPI_DOUBLE, globalA.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, nullptr,
                    nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n",
               maxElapsed * 1000.0, (n * n * n / 3.0) / maxElapsed / 1e9);
        if (printResults) print_results(globalA, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(globalA, original, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
