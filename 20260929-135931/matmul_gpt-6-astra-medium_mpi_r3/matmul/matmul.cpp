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

// Balanced contiguous output blocks, including empty blocks when N < grid size.
size_t boundary(size_t n, int part, int parts) {
    return (n / parts) * part + std::min(n % parts, static_cast<size_t>(part));
}

// Keep k in its original order; vectorize across independent output columns.
// Four rows share each B load and small tiles keep the output in cache.
void multiplyPanel(const double* a, const double* b, double* c,
                   size_t rows, size_t cols, size_t depth) {
    for (size_t ii = 0; ii < rows; ii += 32) {
        for (size_t jj = 0; jj < cols; jj += 128) {
            const size_t iend = std::min(rows, ii + 32);
            const size_t jend = std::min(cols, jj + 128);
            size_t i = ii;
            for (; i + 3 < iend; i += 4) {
                double* c0 = c + i * cols;
                double* c1 = c0 + cols;
                double* c2 = c1 + cols;
                double* c3 = c2 + cols;
                for (size_t k = 0; k < depth; ++k) {
                    const double a0 = a[i * depth + k];
                    const double a1 = a[(i + 1) * depth + k];
                    const double a2 = a[(i + 2) * depth + k];
                    const double a3 = a[(i + 3) * depth + k];
                    const double* bk = b + k * cols;
                    for (size_t j = jj; j < jend; ++j) {
                        c0[j] += a0 * bk[j];
                        c1[j] += a1 * bk[j];
                        c2[j] += a2 * bk[j];
                        c3[j] += a3 * bk[j];
                    }
                }
            }
            for (; i < iend; ++i) {
                for (size_t k = 0; k < depth; ++k) {
                    const double aik = a[i * depth + k];
                    for (size_t j = jj; j < jend; ++j)
                        c[i * cols + j] += aik * b[k * cols + j];
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t n,
                    size_t rowBegin, size_t rowEnd, size_t colBegin, size_t colEnd) {
    for (size_t i = rowBegin; i < std::min(n, size_t(5)) && i < rowEnd; ++i) {
        for (size_t j = colBegin; j < std::min(n, size_t(5)) && j < colEnd; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            const double actual = c[(i - rowBegin) * (colEnd - colBegin) + j - colBegin];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

// Chunk transfers so neither matrix sizes nor displacements are limited to MPI int.
void transfer(double* data, size_t count, int peer, bool send) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send) MPI_Send(data + offset, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data + offset, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int processes) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
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
    if (N && N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        if (rank == 0) fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    int dims[2] = {0, 0};
    MPI_Dims_create(processes, 2, dims);
    const int row = rank / dims[1], col = rank % dims[1];
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    const size_t rowBegin = boundary(N, row, dims[0]);
    const size_t rowEnd = boundary(N, row + 1, dims[0]);
    const size_t colBegin = boundary(N, col, dims[1]);
    const size_t colEnd = boundary(N, col + 1, dims[1]);
    const size_t rows = rowEnd - rowBegin, cols = colEnd - colBegin;
    // A uniform panel width ensures all collective calls have matching counts.
    const size_t maxExtent = std::max(boundary(N, 1, dims[0]), boundary(N, 1, dims[1]));
    if (maxExtent > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Local matrix extent exceeds MPI count limit\n");
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        return 1;
    }
    const size_t width = std::min(size_t(128), size_t(INT_MAX) / std::max(size_t(1), maxExtent));
    const size_t panels = N / width + (N % width != 0);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    // SUMMA: cyclic ownership of inner-dimension panels on a 2D process grid.
    // Only owners initialize each input entry, using its global coordinates.
    std::vector<std::vector<double>> a(panels), b(panels);
    for (size_t p = 0; p < panels; ++p) {
        const size_t depth = std::min(width, N - p * width);
        if (p % dims[1] == static_cast<size_t>(col)) {
            a[p].resize(rows * depth);
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < depth; ++k)
                    a[p][i * depth + k] = getPseudoRndValue(N, rowBegin + i, p * width + k);
        }
        if (p % dims[0] == static_cast<size_t>(row)) {
            b[p].resize(depth * cols);
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < cols; ++j)
                    b[p][k * cols + j] = getPseudoRndValue(N, p * width + k, colBegin + j);
        }
    }
    std::vector<double> c(rows * cols, 0.0);
    std::vector<double> aBuffer[2], bBuffer[2];
    for (int slot = 0; slot < 2; ++slot) {
        aBuffer[slot].resize(std::max(size_t(1), rows * width));
        bBuffer[slot].resize(std::max(size_t(1), cols * width));
    }
    MPI_Request requests[2][2];
    auto broadcast = [&](size_t p) {
        const size_t depth = std::min(width, N - p * width);
        const size_t slot = p % 2;
        double* ap = p % dims[1] == static_cast<size_t>(col) && !a[p].empty()
                   ? a[p].data() : aBuffer[slot].data();
        double* bp = p % dims[0] == static_cast<size_t>(row) && !b[p].empty()
                   ? b[p].data() : bBuffer[slot].data();
        MPI_Ibcast(ap, static_cast<int>(rows * depth), MPI_DOUBLE,
                   static_cast<int>(p % dims[1]), rowComm, &requests[slot][0]);
        MPI_Ibcast(bp, static_cast<int>(cols * depth), MPI_DOUBLE,
                   static_cast<int>(p % dims[0]), colComm, &requests[slot][1]);
    };
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (panels) broadcast(0);
    for (size_t p = 0; p < panels; ++p) {
        MPI_Waitall(2, requests[p % 2], MPI_STATUSES_IGNORE);
        if (p + 1 < panels) broadcast(p + 1);
        const double* ap = p % dims[1] == static_cast<size_t>(col) ? a[p].data() : aBuffer[p % 2].data();
        const double* bp = p % dims[0] == static_cast<size_t>(row) ? b[p].data() : bBuffer[p % 2].data();
        multiplyPanel(ap, bp, c.data(), rows, cols, std::min(width, N - p * width));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? (2.0 * N * N * N) / duration / 1e9 : 0.0);
    }
    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(N * N), received;
            for (int source = 0; source < processes; ++source) {
                const size_t r0 = boundary(N, source / dims[1], dims[0]);
                const size_t r1 = boundary(N, source / dims[1] + 1, dims[0]);
                const size_t c0 = boundary(N, source % dims[1], dims[1]);
                const size_t c1 = boundary(N, source % dims[1] + 1, dims[1]);
                const std::vector<double>* block = &c;
                if (source != 0) {
                    received.resize((r1 - r0) * (c1 - c0));
                    transfer(received.data(), received.size(), source, false);
                    block = &received;
                }
                if (c1 != c0)
                    for (size_t i = r0; i < r1; ++i)
                        std::copy_n(block->data() + (i - r0) * (c1 - c0), c1 - c0,
                                    result.data() + i * N + c0);
            }
            print_results(result, "MatrixC");
        } else {
            transfer(c.data(), c.size(), 0, true);
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int localValid = validateResult(c, N, rowBegin, rowEnd, colBegin, colEnd);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = run(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
