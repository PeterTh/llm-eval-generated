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

// Each MPI rank owns rows r, r + nranks, ... .  Matrices are intentionally
// row-distributed (rather than replicated) after construction; this keeps the
// O(n^2) storage and O(n^3) update work evenly spread over all accelerators.
// The completed pivot row is broadcast before every update, which is the only
// data needed by other ranks for a right-looking Cholesky step.

static void cudaCheck(cudaError_t error, const char* where, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure at %s: %s\n", rank, where,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void updateColumn(double* matrix, const double* pivotRow, size_t n,
                             size_t column, int rank, int ranks) {
    const size_t localRow = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(rank) + localRow * static_cast<size_t>(ranks);
    if (row <= column || row >= n) return;

    const double* const current = matrix + row * n;
    double dot = 0.0;
    // This loop is the computationally dominant part of the algorithm.  The
    // compiler keeps the reduction in registers; each thread owns one row.
    for (size_t k = 0; k < column; ++k)
        dot += current[k] * pivotRow[k];
    matrix[row * n + column] = (current[column] - dot) / pivotRow[column];
}

static bool distributedCholesky(std::vector<double>& matrix, size_t n, int rank,
                                int ranks) {
    double* deviceMatrix = nullptr;
    double* devicePivot = nullptr;
    const size_t bytes = n * n * sizeof(double);
    const size_t rowBytes = n * sizeof(double);

    cudaCheck(cudaMalloc(&deviceMatrix, bytes), "cudaMalloc(matrix)", rank);
    cudaCheck(cudaMalloc(&devicePivot, rowBytes), "cudaMalloc(pivot)", rank);
    cudaCheck(cudaMemcpy(deviceMatrix, matrix.data(), bytes, cudaMemcpyHostToDevice),
              "copy matrix to device", rank);

    std::vector<double> pivot(n);
    bool positiveDefinite = true;
    constexpr int threads = 256;
    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(ranks));
        if (rank == owner) {
            // The owning GPU has produced this row's preceding columns.
            // Transfer that compact prefix only when it becomes a panel row.
            cudaCheck(cudaMemcpy(matrix.data() + j * n, deviceMatrix + j * n,
                                 (j + 1) * sizeof(double), cudaMemcpyDeviceToHost),
                      "copy pivot row from device", rank);
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) schedule(static)
            for (long long k = 0; k < static_cast<long long>(j); ++k)
                sum += matrix[j * n + static_cast<size_t>(k)] * matrix[j * n + static_cast<size_t>(k)];
            const double diagonal = matrix[j * n + j] - sum;
            if (diagonal <= 0.0 || !std::isfinite(diagonal)) {
                positiveDefinite = false;
            } else {
                matrix[j * n + j] = std::sqrt(diagonal);
                std::copy_n(matrix.data() + j * n, j + 1, pivot.data());
                cudaCheck(cudaMemcpy(deviceMatrix + j * n + j, &matrix[j * n + j], sizeof(double),
                                     cudaMemcpyHostToDevice), "copy pivot diagonal", rank);
            }
        }
        int localOK = positiveDefinite ? 1 : 0, globalOK = 0;
        MPI_Allreduce(&localOK, &globalOK, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!globalOK) {
            if (rank == owner)
                std::fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", j);
            cudaFree(devicePivot); cudaFree(deviceMatrix);
            return false;
        }

        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(devicePivot, pivot.data(), (j + 1) * sizeof(double),
                             cudaMemcpyHostToDevice), "copy pivot row", rank);

        const size_t first = static_cast<size_t>(rank);
        const size_t localRows = first >= n ? 0 : (n - 1 - first) / static_cast<size_t>(ranks) + 1;
        const size_t startLocal = (j < first) ? 0 : (j - first) / static_cast<size_t>(ranks) + 1;
        if (startLocal < localRows) {
            // Offset the grid by adjusting rank encoded through a temporary
            // launch range is unnecessary: inactive early rows cheaply exit.
            const int blocks = static_cast<int>((localRows + threads - 1) / threads);
            updateColumn<<<blocks, threads>>>(deviceMatrix, devicePivot, n, j, rank, ranks);
            cudaCheck(cudaGetLastError(), "updateColumn launch", rank);
            cudaCheck(cudaDeviceSynchronize(), "updateColumn", rank);
        }
    }
    // Return each rank's final row ownership to host memory for the MPI reduce.
    for (size_t row = static_cast<size_t>(rank); row < n; row += static_cast<size_t>(ranks))
        cudaCheck(cudaMemcpy(matrix.data() + row * n, deviceMatrix + row * n, rowBytes,
                             cudaMemcpyDeviceToHost), "copy factor rows", rank);
    cudaFree(devicePivot);
    cudaFree(deviceMatrix);
    return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& matrix, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += b[static_cast<size_t>(i) * n + k] * b[j * n + k];
            matrix[static_cast<size_t>(i) * n + j] = sum;
        }
        matrix[static_cast<size_t>(i) * n + static_cast<size_t>(i)] += n;
    }
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, size_t n) {
    double maxError = 0.0, maxRelativeError = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:maxError,maxRelativeError) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) for (long long j = 0; j < static_cast<long long>(n); ++j) {
        double sum = 0.0;
        for (size_t k = 0; k <= std::min(static_cast<size_t>(i), static_cast<size_t>(j)); ++k)
            sum += l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
        const double error = std::fabs(sum - original[static_cast<size_t>(i) * n + static_cast<size_t>(j)]);
        maxError = std::max(maxError, error);
        maxRelativeError = std::max(maxRelativeError, error / (std::fabs(original[static_cast<size_t>(i) * n + static_cast<size_t>(j)]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, maxRelativeError);
    return maxRelativeError <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num> Matrix size (default: 512)\n  -v Enable validation\n  -r Print results\n  -h Show help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0; cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) { if (rank == 0) std::fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
    if (rank == 0) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nMPI ranks: %d, OpenMP threads/rank: %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, ranks, omp_get_max_threads(), validate ? "enabled" : "disabled");

    std::vector<double> matrix(n * n), original;
    generatePositiveDefiniteMatrix(matrix, n);
    if (validate) original = matrix;
    // Retain only each rank's assigned rows.  MPI_SUM reconstructs L on root.
    #pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(n); ++row)
        if (static_cast<int>(static_cast<size_t>(row) % static_cast<size_t>(ranks)) != rank)
            std::fill_n(matrix.data() + static_cast<size_t>(row) * n, n, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const bool success = distributedCholesky(matrix, n, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    if (!success) { MPI_Finalize(); return 1; }
    std::vector<double> factor(rank == 0 ? n * n : 0);
    MPI_Reduce(matrix.data(), rank == 0 ? factor.data() : nullptr, static_cast<int>(n * n), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        const double seconds = std::chrono::duration<double>(end - start).count();
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", seconds * 1000.0, (static_cast<double>(n) * n * n / 3.0) / seconds / 1e9);
        if (printResults) print_results(factor, "CholeskyL");
        if (validate) {
            result = validateCholesky(factor, original, n) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return result;
}
