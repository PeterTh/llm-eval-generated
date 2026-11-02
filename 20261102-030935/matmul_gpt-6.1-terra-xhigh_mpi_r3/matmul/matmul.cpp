#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Every rank generates its own rows, so no input matrix needs to be scattered.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initRows(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
}

// Store B as consecutive eight-column panels, so each microkernel streams
// its entire k dimension without strided cache and TLB accesses.
void initB(std::vector<double>& b, size_t N) {
    size_t offset = 0;
    for (size_t j = 0; j < N; j += 8) {
        const size_t width = std::min(size_t{8}, N - j);
        for (size_t k = 0; k < N; ++k)
            for (size_t x = 0; x < width; ++x)
                b[offset++] = getPseudoRndValue(N, k, j + x);
    }
}

size_t bIndex(size_t N, size_t k, size_t j) {
    const size_t panel = j / 8;
    const size_t width = std::min(size_t{8}, N - panel * 8);
    return panel * 8 * N + k * width + j % 8;
}

// Keep k in ascending order for every output. The four-row, eight-column
// microkernel keeps partial sums in vector registers and reuses each B load.
using Vec4 = double __attribute__((vector_size(32)));

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows) {
    const double* __restrict__ a = A.data();
    const double* __restrict__ b = B.data();
    double* __restrict__ c = C.data();
    const size_t paired = N - N % 2;
    for (size_t i = 0; i < rows; i += 4) {
        const size_t active = std::min(size_t{4}, rows - i);
        const double* a0row = a + i * N;
        const double* a1row = a + (i + std::min(size_t{1}, active - 1)) * N;
        const double* a2row = a + (i + std::min(size_t{2}, active - 1)) * N;
        const double* a3row = a + (i + std::min(size_t{3}, active - 1)) * N;
        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            Vec4 s00{}, s01{}, s10{}, s11{}, s20{}, s21{}, s30{}, s31{};
            for (size_t k = 0; k < paired; ++k) {
                Vec4 b0, b1;
                std::memcpy(&b0, b + j * N + k * 8, sizeof(b0));
                std::memcpy(&b1, b + j * N + k * 8 + 4, sizeof(b1));
                const double a0 = a0row[k];
                const double a1 = a1row[k];
                const double a2 = a2row[k];
                const double a3 = a3row[k];
                const Vec4 v0{a0, a0, a0, a0}, v1{a1, a1, a1, a1};
                const Vec4 v2{a2, a2, a2, a2}, v3{a3, a3, a3, a3};
                s00 += v0 * b0; s01 += v0 * b1;
                s10 += v1 * b0; s11 += v1 * b1;
                s20 += v2 * b0; s21 += v2 * b1;
                s30 += v3 * b0; s31 += v3 * b1;
            }
            std::memcpy(c + (i + 0) * N + j, &s00, sizeof(s00));
            std::memcpy(c + (i + 0) * N + j + 4, &s01, sizeof(s01));
            if (active > 1) {
                std::memcpy(c + (i + 1) * N + j, &s10, sizeof(s10));
                    std::memcpy(c + (i + 1) * N + j + 4, &s11, sizeof(s11));
            }
            if (active > 2) {
                std::memcpy(c + (i + 2) * N + j, &s20, sizeof(s20));
                    std::memcpy(c + (i + 2) * N + j + 4, &s21, sizeof(s21));
            }
            if (active > 3) {
                std::memcpy(c + (i + 3) * N + j, &s30, sizeof(s30));
                    std::memcpy(c + (i + 3) * N + j + 4, &s31, sizeof(s31));
            }
            // The original optimized scalar dot product contracts its odd
            // final multiply and add on platforms with hardware FMA.
            if (N % 2) {
                const size_t k = N - 1;
                for (size_t r = 0; r < active; ++r)
                    for (size_t x = 0; x < 8; ++x)
                        c[(i + r) * N + j + x] = std::fma(a[(i + r) * N + k],
                            b[bIndex(N, k, j + x)], c[(i + r) * N + j + x]);
            }
        }
        for (; j < N; ++j) {
            for (size_t r = 0; r < active; ++r) {
                double sum = 0.0;
                for (size_t k = 0; k < paired; ++k)
                    sum += a[(i + r) * N + k] * b[bIndex(N, k, j)];
                if (N % 2) sum = std::fma(a[(i + r) * N + N - 1], b[bIndex(N, N - 1, j)], sum);
                c[(i + r) * N + j] = sum;
            }
        }
    }
}

bool validateRows(const std::vector<double>& A, const std::vector<double>& B,
                  const std::vector<double>& C, size_t N, size_t firstRow, size_t rows) {
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = pi % N;
        if (i < firstRow || i - firstRow >= rows) continue;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[(i - firstRow) * N + k] * B[bIndex(N, k, j)];
            const double actual = C[(i - firstRow) * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Split rows evenly, including when there are more ranks than rows.
size_t rowCount(size_t N, int ranks, int rank) {
    return N / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < N % static_cast<size_t>(ranks));
}
size_t rowStart(size_t N, int ranks, int rank) {
    return static_cast<size_t>(rank) * (N / static_cast<size_t>(ranks)) +
           std::min(static_cast<size_t>(rank), N % static_cast<size_t>(ranks));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(arg, &end, 10);
            if (errno || end == arg || *end != '\0' || *arg == '-' || value == 0 ||
                value > std::numeric_limits<size_t>::max() ||
                value > std::numeric_limits<size_t>::max() / value ||
                value * value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                parseStatus = 1;
                break;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const size_t rows = rowCount(N, ranks, rank);
    const size_t firstRow = rowStart(N, ranks, rank);
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
    initRows(A, N, firstRow, rows);
    if (rows) initB(B, N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        const double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        // Chunk transfers to avoid MPI's int count limit. Only rank zero needs
        // the complete result, and only when the user requests its statistics.
        if (rank == 0) {
            std::vector<double> result(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t count = rowCount(N, ranks, source) * N;
                const size_t offset = rowStart(N, ranks, source) * N;
                for (size_t pos = 0; pos < count;) {
                    const int chunk = static_cast<int>(std::min(count - pos, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + offset + pos, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += static_cast<size_t>(chunk);
                }
            }
            print_results(result, "MatrixC");
        } else {
            for (size_t pos = 0; pos < C.size();) {
                const int chunk = static_cast<int>(std::min(C.size() - pos, static_cast<size_t>(INT_MAX)));
                MPI_Send(C.data() + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                pos += static_cast<size_t>(chunk);
            }
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateRows(A, B, C, N, firstRow, rows) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
