#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, i, j);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N, int rank) {
    bool valid = true;
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    if (rank == 0) {
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = checkPoints[pi] % N, j = checkPoints[pj] % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) expected += A[i * N + k] * B[k * N + j];
                const double actual = C[i * N + j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    valid = false;
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0 || parsed > std::numeric_limits<size_t>::max()) parseStatus = 1;
            else N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseStatus = 1; }
    }

    int gridDim = static_cast<int>(std::sqrt(static_cast<double>(worldSize)));
    while (worldSize % gridDim != 0) --gridDim;
    const int gridRows = gridDim, gridCols = worldSize / gridDim;
    const int row = rank / gridCols, col = rank % gridCols;
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

    size_t block = 0;
    if (!parseStatus) {
        block = (N + static_cast<size_t>(std::min(gridRows, gridCols)) - 1) / static_cast<size_t>(std::min(gridRows, gridCols));
        if (block > static_cast<size_t>(std::numeric_limits<int>::max()) / block) parseStatus = 1;
    }
    if (parseStatus || N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or arguments\n");
        MPI_Comm_free(&rowComm); MPI_Comm_free(&colComm); MPI_Finalize(); return 1;
    }
    // A square process grid enables the standard SUMMA schedule.
    if (gridRows != gridCols) {
        if (rank == 0) fprintf(stderr, "MPI process count must be a perfect square\n");
        MPI_Comm_free(&rowComm); MPI_Comm_free(&colComm); MPI_Finalize(); return 1;
    }
    const int count = static_cast<int>(block * block);
    std::vector<double> localA(static_cast<size_t>(count)), localB(static_cast<size_t>(count));
    std::vector<double> localC(static_cast<size_t>(count), 0.0);
    std::vector<double> aPanel(static_cast<size_t>(count)), bPanel(static_cast<size_t>(count));
    std::vector<double> A, B, C;
    if (rank == 0) {
        A.resize(N * N); B.resize(N * N); C.resize(N * N);
        initMatrix(A, N); initMatrix(B, N);
    }
    if (rank == 0) printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nInitializing matrices...\n", N, N, validate ? "enabled" : "disabled");

    // Distribute square blocks, padding only the final block rows/columns.
    for (int pr = 0; pr < gridRows; ++pr) {
        for (int pc = 0; pc < gridCols; ++pc) {
            std::vector<double> sendA, sendB;
            if (rank == 0) {
                sendA.assign(static_cast<size_t>(count), 0.0); sendB.assign(static_cast<size_t>(count), 0.0);
                for (size_t i = 0; i < block && static_cast<size_t>(pr) * block + i < N; ++i)
                    for (size_t j = 0; j < block && static_cast<size_t>(pc) * block + j < N; ++j) {
                        sendA[i * block + j] = A[(static_cast<size_t>(pr) * block + i) * N + static_cast<size_t>(pc) * block + j];
                        sendB[i * block + j] = B[(static_cast<size_t>(pr) * block + i) * N + static_cast<size_t>(pc) * block + j];
                    }
            }
            const int target = pr * gridCols + pc;
            if (rank == 0 && target == 0) { localA = std::move(sendA); localB = std::move(sendB); }
            else if (rank == 0) {
                MPI_Send(sendA.data(), count, MPI_DOUBLE, target, 10, MPI_COMM_WORLD);
                MPI_Send(sendB.data(), count, MPI_DOUBLE, target, 11, MPI_COMM_WORLD);
            } else if (rank == target) {
                MPI_Recv(localA.data(), count, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(localB.data(), count, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < gridCols; ++k) {
        if (col == k) aPanel = localA;
        if (row == k) bPanel = localB;
        MPI_Bcast(aPanel.data(), count, MPI_DOUBLE, k, rowComm);
        MPI_Bcast(bPanel.data(), count, MPI_DOUBLE, k, colComm);
        for (size_t i = 0; i < block; ++i)
            for (size_t kk = 0; kk < block; ++kk) {
                const double aik = aPanel[i * block + kk];
                for (size_t j = 0; j < block; ++j) localC[i * block + j] += aik * bPanel[kk * block + j];
            }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    for (int pr = 0; pr < gridRows; ++pr) for (int pc = 0; pc < gridCols; ++pc) {
        const int source = pr * gridCols + pc;
        std::vector<double> recv;
        if (rank == 0 && source != 0) recv.resize(static_cast<size_t>(count));
        double* data = rank == 0 && source != 0 ? recv.data() : localC.data();
        if (rank == 0 && source != 0) MPI_Recv(data, count, MPI_DOUBLE, source, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else if (rank == source && source != 0) MPI_Send(localC.data(), count, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
        if (rank == 0) {
            const double* tile = source == 0 ? localC.data() : recv.data();
            for (size_t i = 0; i < block && static_cast<size_t>(pr) * block + i < N; ++i)
                for (size_t j = 0; j < block && static_cast<size_t>(pc) * block + j < N; ++j)
                    C[(static_cast<size_t>(pr) * block + i) * N + static_cast<size_t>(pc) * block + j] = tile[i * block + j];
        }
    }
    if (rank == 0) {
        printf("Computing matrix multiplication...\nComputation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
        if (printResults) print_results(C, "MatrixC");
    }
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(A, B, C, N, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm); MPI_Comm_free(&colComm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
