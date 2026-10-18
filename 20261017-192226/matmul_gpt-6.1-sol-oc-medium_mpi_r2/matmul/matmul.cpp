#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous output tiles, including when N is not divisible by the grid.
size_t blockStart(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(N % parts, size_t(coordinate));
}

// Register-blocked local update. Each output element accumulates k in the
// original order; independent output columns are available for SIMD execution.
// Explicit FMA gives the same rounding in full tiles and boundary tiles.
void multiplyPanel(const double* A, const double* B, double* C,
                   size_t rows, size_t cols, size_t width) {
    size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
            double c0[8], c1[8], c2[8], c3[8];
            for (size_t t = 0; t < 8; ++t) {
                c0[t] = C[i * cols + j + t];
                c1[t] = C[(i + 1) * cols + j + t];
                c2[t] = C[(i + 2) * cols + j + t];
                c3[t] = C[(i + 3) * cols + j + t];
            }
            for (size_t k = 0; k < width; ++k) {
                const double a0 = A[i * width + k];
                const double a1 = A[(i + 1) * width + k];
                const double a2 = A[(i + 2) * width + k];
                const double a3 = A[(i + 3) * width + k];
                const double* b = B + k * cols + j;
                #pragma omp simd
                for (size_t t = 0; t < 8; ++t) {
                    c0[t] = std::fma(a0, b[t], c0[t]);
                    c1[t] = std::fma(a1, b[t], c1[t]);
                    c2[t] = std::fma(a2, b[t], c2[t]);
                    c3[t] = std::fma(a3, b[t], c3[t]);
                }
            }
            for (size_t t = 0; t < 8; ++t) {
                C[i * cols + j + t] = c0[t];
                C[(i + 1) * cols + j + t] = c1[t];
                C[(i + 2) * cols + j + t] = c2[t];
                C[(i + 3) * cols + j + t] = c3[t];
            }
        }
        for (size_t r = i; r < i + 4; ++r) {
            for (size_t k = 0; k < width; ++k) {
                const double a = A[r * width + k];
                for (size_t t = j; t < cols; ++t)
                    C[r * cols + t] = std::fma(a, B[k * cols + t], C[r * cols + t]);
            }
        }
    }
    for (; i < rows; ++i) {
        for (size_t k = 0; k < width; ++k) {
            const double a = A[i * width + k];
            for (size_t j = 0; j < cols; ++j)
                C[i * cols + j] = std::fma(a, B[k * cols + j], C[i * cols + j]);
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t N, size_t rowStart,
                    size_t colStart, size_t rows, size_t cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N;
            const size_t j = pj % N;
            if (i < rowStart || i >= rowStart + rows ||
                j < colStart || j >= colStart + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = C[(i - rowStart) * cols + j - colStart];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

// Only -r needs the full matrix. Transfer packed tiles in bounded messages so
// ordinary benchmark runs never allocate a global matrix on any rank.
std::vector<double> gatherResult(const std::vector<double>& local, size_t N,
                                 int rank, int size, const int dims[2]) {
    std::vector<double> result;
    std::vector<double> tile;
    if (rank == 0) {
        result.resize(N * N);
        tile.resize((N / dims[0] + (N % dims[0] != 0)) *
                    (N / dims[1] + (N % dims[1] != 0)));
    }
    for (int source = 0; source < size; ++source) {
        if (rank != 0 && rank != source) continue;
        const int r = source / dims[1], c = source % dims[1];
        const size_t row = blockStart(N, r, dims[0]);
        const size_t col = blockStart(N, c, dims[1]);
        const size_t rows = blockStart(N, r + 1, dims[0]) - row;
        const size_t cols = blockStart(N, c + 1, dims[1]) - col;
        if (source != 0) {
            for (size_t offset = 0; offset < rows * cols;) {
                const int count = static_cast<int>(std::min(size_t(INT_MAX), rows * cols - offset));
                if (rank == 0)
                    MPI_Recv(tile.data() + offset, count, MPI_DOUBLE, source, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                else
                    MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += count;
            }
        }
        if (rank == 0 && cols != 0) {
            const double* data = source == 0 ? local.data() : tile.data();
            for (size_t i = 0; i < rows; ++i)
                std::copy_n(data + i * cols, cols, result.data() + (row + i) * N + col);
        }
    }
    return result;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int size) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(text, &end, 10);
            if (errno != 0 || text[0] == '-' || end == text || *end != '\0' ||
                parsed == 0 || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", text);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    const int row = rank / dims[1], col = rank % dims[1];
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    const size_t rowStart = blockStart(N, row, dims[0]);
    const size_t colStart = blockStart(N, col, dims[1]);
    const size_t rows = blockStart(N, row + 1, dims[0]) - rowStart;
    const size_t cols = blockStart(N, col + 1, dims[1]) - colStart;
    // Every broadcast count fits the MPI int count interface on every rank.
    const size_t maxExtent = std::max((N + dims[0] - 1) / dims[0],
                                     (N + dims[1] - 1) / dims[1]);
    const size_t panelSize = std::min(size_t(128), size_t(INT_MAX) / maxExtent);
    const size_t stages = (N + panelSize - 1) / panelSize;
    std::vector<double> C(rows * cols, 0.0);
    std::vector<std::vector<double>> A(stages), B(stages);
    // Block-cyclic ownership of k panels balances input storage. Generate only
    // locally owned inputs directly from their global indices, without scatter.
    for (size_t s = 0; s < stages; ++s) {
        const size_t kStart = s * panelSize;
        const size_t width = std::min(panelSize, N - kStart);
        if (s % dims[1] == size_t(col)) {
            A[s].resize(rows * width);
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < width; ++k)
                    A[s][i * width + k] = getPseudoRndValue(N, rowStart + i, kStart + k);
        }
        if (s % dims[0] == size_t(row)) {
            B[s].resize(width * cols);
            for (size_t k = 0; k < width; ++k)
                for (size_t j = 0; j < cols; ++j)
                    B[s][k * cols + j] = getPseudoRndValue(N, kStart + k, colStart + j);
        }
    }
    std::vector<double> aPanel[2], bPanel[2];
    for (int slot = 0; slot < 2; ++slot) {
        aPanel[slot].resize(rows * panelSize);
        bPanel[slot].resize(cols * panelSize);
    }
    MPI_Request requests[2][2];
    auto startPanel = [&](size_t s) {
        const int slot = s % 2;
        const size_t width = std::min(panelSize, N - s * panelSize);
        const int aRoot = s % dims[1], bRoot = s % dims[0];
        if (col == aRoot) std::copy(A[s].begin(), A[s].end(), aPanel[slot].begin());
        if (row == bRoot) std::copy(B[s].begin(), B[s].end(), bPanel[slot].begin());
        MPI_Ibcast(aPanel[slot].data(), static_cast<int>(rows * width), MPI_DOUBLE,
                   aRoot, rowComm, &requests[slot][0]);
        MPI_Ibcast(bPanel[slot].data(), static_cast<int>(cols * width), MPI_DOUBLE,
                   bRoot, colComm, &requests[slot][1]);
    };

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // Pipelined SUMMA: broadcast the next panel while updating this output tile.
    startPanel(0);
    for (size_t s = 0; s < stages; ++s) {
        const int slot = s % 2;
        MPI_Waitall(2, requests[slot], MPI_STATUSES_IGNORE);
        if (s + 1 < stages) startPanel(s + 1);
        multiplyPanel(aPanel[slot].data(), bPanel[slot].data(), C.data(), rows, cols,
                      std::min(panelSize, N - s * panelSize));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        const auto result = gatherResult(C, N, rank, size, dims);
        if (rank == 0) print_results(result, "MatrixC");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(C, N, rowStart, colStart, rows, cols);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
