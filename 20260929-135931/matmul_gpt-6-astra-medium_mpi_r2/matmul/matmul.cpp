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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

constexpr size_t panelSize = 128;

// Balanced contiguous ownership, including empty blocks when there are more
// process rows/columns than matrix rows/columns.
size_t blockStart(size_t n, int coordinate, int processes) {
    return n / processes * coordinate + std::min(n % processes, size_t(coordinate));
}

// Vectorize independent columns, retaining the original ascending k summation
// order. Four rows share each loaded B vector; small tiles stay in registers.
void multiplyPanel(const double* A, const double* B, double* C,
                   size_t rows, size_t cols, size_t depth) {
    constexpr size_t tileCols = 16;
    size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        size_t j = 0;
        for (; j + tileCols <= cols; j += tileCols) {
            double c0[tileCols], c1[tileCols], c2[tileCols], c3[tileCols];
            for (size_t x = 0; x < tileCols; ++x) {
                c0[x] = C[i * cols + j + x];
                c1[x] = C[(i + 1) * cols + j + x];
                c2[x] = C[(i + 2) * cols + j + x];
                c3[x] = C[(i + 3) * cols + j + x];
            }
            for (size_t k = 0; k < depth; ++k) {
                const double a0 = A[i * depth + k], a1 = A[(i + 1) * depth + k];
                const double a2 = A[(i + 2) * depth + k], a3 = A[(i + 3) * depth + k];
                for (size_t x = 0; x < tileCols; ++x) {
                    const double b = B[k * cols + j + x];
                    c0[x] += a0 * b;
                    c1[x] += a1 * b;
                    c2[x] += a2 * b;
                    c3[x] += a3 * b;
                }
            }
            for (size_t x = 0; x < tileCols; ++x) {
                C[i * cols + j + x] = c0[x];
                C[(i + 1) * cols + j + x] = c1[x];
                C[(i + 2) * cols + j + x] = c2[x];
                C[(i + 3) * cols + j + x] = c3[x];
            }
        }
        for (size_t r = i; r < i + 4; ++r)
            for (size_t k = 0; k < depth; ++k)
                for (size_t x = j; x < cols; ++x)
                    C[r * cols + x] += A[r * depth + k] * B[k * cols + x];
    }
    for (; i < rows; ++i)
        for (size_t k = 0; k < depth; ++k)
            for (size_t j = 0; j < cols; ++j)
                C[i * cols + j] += A[i * depth + k] * B[k * cols + j];
}

bool validateResult(const std::vector<double>& C, size_t N, size_t rowStart,
                    size_t colStart, size_t rows, size_t cols) {
    // Retain the original probes and also check each rank's block boundaries.
    if (rows == 0 || cols == 0) return true;
    const size_t rowChecks[] = {0, 1 % N, 2 % N, 3 % N, 4 % N,
                               rowStart, rowStart + rows / 2, rowStart + rows - 1};
    const size_t colChecks[] = {0, 1 % N, 2 % N, 3 % N, 4 % N,
                               colStart, colStart + cols / 2, colStart + cols - 1};
    for (size_t i : rowChecks) {
        if (i < rowStart || i >= rowStart + rows) continue;
        for (size_t j : colChecks) {
            if (j < colStart || j >= colStart + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            double actual = C[(i - rowStart) * cols + j - colStart];
            double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

// Only -r needs a global matrix. Chunk transfers to respect MPI's int counts.
void printDistributedResult(const std::vector<double>& local, size_t N,
                            int rank, int processes, const int* dims) {
    if (rank != 0) {
        for (size_t offset = 0; offset < local.size();) {
            int count = static_cast<int>(std::min(local.size() - offset, size_t(INT_MAX)));
            MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += count;
        }
        return;
    }
    std::vector<double> full(N * N), received;
    for (int source = 0; source < processes; ++source) {
        int r = source / dims[1], c = source % dims[1];
        size_t rowStart = blockStart(N, r, dims[0]), colStart = blockStart(N, c, dims[1]);
        size_t rows = blockStart(N, r + 1, dims[0]) - rowStart;
        size_t cols = blockStart(N, c + 1, dims[1]) - colStart;
        const double* data = local.data();
        if (source != 0) {
            received.resize(rows * cols);
            for (size_t offset = 0; offset < received.size();) {
                int count = static_cast<int>(std::min(received.size() - offset, size_t(INT_MAX)));
                MPI_Recv(received.data() + offset, count, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += count;
            }
            data = received.data();
        }
        if (cols != 0)
            for (size_t i = 0; i < rows; ++i)
                std::copy_n(data + i * cols, cols, full.data() + (rowStart + i) * N + colStart);
    }
    print_results(full, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t N = 512;
    bool validate = false, printResults = false;
    int status = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > size_t(INT_MAX) / panelSize ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                status = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            status = 1;
            break;
        }
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return status;
    }

    try {
        int dims[2] = {0, 0};
        MPI_Dims_create(processes, 2, dims);
        int row = rank / dims[1], col = rank % dims[1];
        MPI_Comm rowComm, colComm;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        size_t rowStart = blockStart(N, row, dims[0]), colStart = blockStart(N, col, dims[1]);
        size_t rows = blockStart(N, row + 1, dims[0]) - rowStart;
        size_t cols = blockStart(N, col + 1, dims[1]) - colStart;
        size_t stages = (N + panelSize - 1) / panelSize;
        if (rank == 0) {
            printf("Matrix Multiplication Benchmark\n");
            printf("Matrix size: %zu x %zu\n", N, N);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing matrices...\n");
        }
        // SUMMA: A panels cycle across process columns, B panels across rows.
        // Generate only owned input blocks, with no centralized initialization.
        size_t aPanels = stages / dims[1] + (size_t(col) < stages % dims[1]);
        size_t bPanels = stages / dims[0] + (size_t(row) < stages % dims[0]);
        std::vector<double> A(aPanels * rows * panelSize), B(bPanels * cols * panelSize);
        std::vector<double> C(rows * cols, 0.0);
        for (size_t s = col; s < stages; s += dims[1]) {
            size_t depth = std::min(panelSize, N - s * panelSize);
            size_t base = (s / dims[1]) * rows * panelSize;
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < depth; ++k)
                    A[base + i * depth + k] = getPseudoRndValue(N, rowStart + i, s * panelSize + k);
        }
        for (size_t s = row; s < stages; s += dims[0]) {
            size_t depth = std::min(panelSize, N - s * panelSize);
            size_t base = (s / dims[0]) * cols * panelSize;
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < cols; ++j)
                    B[base + k * cols + j] = getPseudoRndValue(N, s * panelSize + k, colStart + j);
        }
        std::vector<double> aBuffer[2], bBuffer[2];
        for (int slot = 0; slot < 2; ++slot) {
            aBuffer[slot].resize(std::max(size_t(1), rows * panelSize));
            bBuffer[slot].resize(std::max(size_t(1), cols * panelSize));
        }
        MPI_Request requests[2][2];
        auto startPanel = [&](size_t s) {
            size_t depth = std::min(panelSize, N - s * panelSize);
            int slot = s % 2;
            if (s % dims[1] == size_t(col) && rows != 0)
                std::copy_n(A.data() + (s / dims[1]) * rows * panelSize,
                            rows * depth, aBuffer[slot].data());
            if (s % dims[0] == size_t(row) && cols != 0)
                std::copy_n(B.data() + (s / dims[0]) * cols * panelSize,
                            cols * depth, bBuffer[slot].data());
            MPI_Ibcast(aBuffer[slot].data(), static_cast<int>(rows * depth), MPI_DOUBLE,
                       s % dims[1], rowComm, &requests[slot][0]);
            MPI_Ibcast(bBuffer[slot].data(), static_cast<int>(cols * depth), MPI_DOUBLE,
                       s % dims[0], colComm, &requests[slot][1]);
        };
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        startPanel(0);
        for (size_t s = 0; s < stages; ++s) {
            int slot = s % 2;
            MPI_Waitall(2, requests[slot], MPI_STATUSES_IGNORE);
            if (s + 1 < stages) startPanel(s + 1);
            multiplyPanel(aBuffer[slot].data(), bBuffer[slot].data(), C.data(),
                          rows, cols, std::min(panelSize, N - s * panelSize));
        }
        double elapsed = MPI_Wtime() - start, duration = 0.0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
            printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / duration / 1e9);
        }
        if (printResults) printDistributedResult(C, N, rank, processes, dims);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            int failed = !validateResult(C, N, rowStart, colStart, rows, cols);
            MPI_Allreduce(&failed, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
