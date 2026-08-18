#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous rows.  A right-looking blocked
// factorization keeps the expensive trailing update local and exchanges only
// one compact panel per block.
bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                           size_t firstRow, size_t localRows, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    constexpr size_t blockSize = 128;
    const size_t rowEnd = firstRow + localRows;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    int worldSize = 0;
    MPI_Comm_size(comm, &worldSize);
    rowCounts.resize(worldSize);
    rowDisplacements.resize(worldSize);
    int localCount = static_cast<int>(localRows);
    MPI_Allgather(&localCount, 1, MPI_INT, rowCounts.data(), 1, MPI_INT, comm);
    for (int r = 0; r < worldSize; ++r)
        rowDisplacements[r] = (r == 0) ? 0 : rowDisplacements[r - 1] + rowCounts[r - 1];
    for (int r = 0; r < worldSize; ++r) {
        rowCounts[r] *= static_cast<int>(blockSize);
        rowDisplacements[r] *= static_cast<int>(blockSize);
    }

    std::vector<double> gathered(n * blockSize);
    std::vector<double> panel(n * blockSize);
    std::vector<double> blockSend(localRows * blockSize);
    std::vector<double> diagonal(blockSize * blockSize);

    for (size_t k = 0; k < n; k += blockSize) {
        const size_t b = std::min(blockSize, n - k);
        const int sendCount = static_cast<int>(localRows * blockSize);

        // Every rank contributes the current block columns.  The root needs
        // the diagonal block; the same exchange avoids a separate gather.
        std::fill(blockSend.begin(), blockSend.end(), 0.0);
        for (size_t i = 0; i < localRows; ++i)
            std::memcpy(blockSend.data() + i * blockSize,
                        localA.data() + i * n + k, b * sizeof(double));
        MPI_Allgatherv(blockSend.data(), sendCount,
                       MPI_DOUBLE, gathered.data(), rowCounts.data(),
                       rowDisplacements.data(), MPI_DOUBLE, comm);

        bool success = true;
        if (rank == 0) {
            std::fill(diagonal.begin(), diagonal.end(), 0.0);
            for (size_t i = 0; i < b; ++i)
                for (size_t j = 0; j <= i; ++j)
                    diagonal[i * blockSize + j] = gathered[(k + i) * blockSize + j];

            for (size_t i = 0; i < b && success; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t t = 0; t < j; ++t)
                        sum += diagonal[i * blockSize + t] * diagonal[j * blockSize + t];
                    if (i == j) {
                        const double value = diagonal[i * blockSize + i] - sum;
                        if (value <= 0.0) {
                            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", k + i);
                            success = false;
                            break;
                        }
                        diagonal[i * blockSize + i] = std::sqrt(value);
                    } else {
                        diagonal[i * blockSize + j] =
                            (diagonal[i * blockSize + j] - sum) / diagonal[j * blockSize + j];
                    }
                }
            }
        }
        MPI_Bcast(&success, 1, MPI_C_BOOL, 0, comm);
        if (!success) return false;
        MPI_Bcast(diagonal.data(), static_cast<int>(b * blockSize), MPI_DOUBLE, 0, comm);

        // Install the factored diagonal block and solve the panel below it.
        for (size_t i = std::max(firstRow, k); i < rowEnd && i < k + b; ++i) {
            for (size_t j = 0; j < b; ++j)
                localA[(i - firstRow) * n + k + j] =
                    (j <= i - k) ? diagonal[(i - k) * blockSize + j] : 0.0;
        }
        for (size_t i = std::max(firstRow, k + b); i < rowEnd; ++i) {
            for (size_t j = 0; j < b; ++j) {
                double value = localA[(i - firstRow) * n + k + j];
                for (size_t t = 0; t < j; ++t)
                    value -= localA[(i - firstRow) * n + k + t] * diagonal[j * blockSize + t];
                localA[(i - firstRow) * n + k + j] = value / diagonal[j * blockSize + j];
            }
        }

        // Make L21 available by global row index, then perform the local
        // symmetric rank-b update on the lower triangle only.
        std::fill(panel.begin(), panel.end(), 0.0);
        for (size_t i = std::max(firstRow, k + b); i < rowEnd; ++i)
            std::memcpy(panel.data() + i * blockSize,
                        localA.data() + (i - firstRow) * n + k, b * sizeof(double));
        MPI_Allgatherv(panel.data() + firstRow * blockSize, sendCount, MPI_DOUBLE,
                       panel.data(), rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE, comm);
        for (size_t i = std::max(firstRow, k + b); i < rowEnd; ++i) {
            for (size_t j = k + b; j <= i; ++j) {
                double value = localA[(i - firstRow) * n + j];
                for (size_t t = 0; t < b; ++t)
                    value -= panel[i * blockSize + t] * panel[j * blockSize + t];
                localA[(i - firstRow) * n + j] = value;
            }
        }
    }

    for (size_t i = firstRow; i < rowEnd; ++i)
        for (size_t j = i + 1; j < n; ++j)
            localA[(i - firstRow) * n + j] = 0.0;
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, size_t n) {
    std::vector<double> reconstructed(n * n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k) reconstructed[i * n + j] += L[i * n + k] * L[j * n + k];
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - original[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(original[i]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) { std::printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    const size_t base = n / static_cast<size_t>(worldSize);
    const size_t remainder = n % static_cast<size_t>(worldSize);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min<size_t>(rank, remainder);
    const size_t localRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    std::vector<double> globalA, original, localA(localRows * n);
    if (rank == 0) {
        globalA.resize(n * n);
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled");
        generatePositiveDefiniteMatrix(globalA, n);
        if (validate) original = globalA;
    }
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        size_t rows = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        size_t start = static_cast<size_t>(r) * base + std::min<size_t>(r, remainder);
        counts[r] = static_cast<int>(rows * n);
        displacements[r] = static_cast<int>(start * n);
    }
    MPI_Scatterv(rank == 0 ? globalA.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 localA.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(localA, n, firstRow, localRows, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!success) { MPI_Finalize(); return 1; }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n", (n * n * n / 3.0) / elapsed / 1e9);
    }
    std::vector<double> result;
    if (rank == 0) result.resize(n * n);
    MPI_Gatherv(localA.data(), counts[rank], MPI_DOUBLE, rank == 0 ? result.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int exitCode = 0;
    if (rank == 0) {
        if (printResults) print_results(result, "CholeskyL");
        if (validate) { std::printf("Validating result...\n"); if (validateCholesky(result, original, n)) std::printf("Validation: PASSED\n"); else { std::printf("Validation: FAILED\n"); exitCode = 1; } }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
