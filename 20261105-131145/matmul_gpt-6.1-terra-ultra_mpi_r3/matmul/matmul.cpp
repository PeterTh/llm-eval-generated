#include <mpi.h>

#include <algorithm>
#include <climits>
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

// Contiguous, nearly equal partitions, including when there are more parts than elements.
size_t partBegin(size_t n, int part, int parts) {
    return (n / parts) * part + std::min(n % parts, static_cast<size_t>(part));
}

// Each output element is accumulated in increasing k order. Blocking keeps a
// panel of B in cache and the small output tile in registers.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t rows, size_t cols, size_t N) {
    constexpr size_t KB = 128, IB = 32, JB = 64, MR = 4, NR = 8;
    const size_t evenN = N & ~size_t(1);
    for (size_t kk = 0; kk < evenN; kk += KB) {
        const size_t kend = std::min(kk + KB, evenN);
        for (size_t jj = 0; jj < cols; jj += JB) {
            const size_t jend = std::min(jj + JB, cols);
            for (size_t ii = 0; ii < rows; ii += IB) {
                const size_t iend = std::min(ii + IB, rows);
                for (size_t i = ii; i < iend; i += MR) {
                    const size_t nr = std::min(MR, iend - i);
                    for (size_t j = jj; j < jend; j += NR) {
                        const size_t nc = std::min(NR, jend - j);
                        if (nr == MR && nc == NR) {
                            // Four adjacent doubles per vector; memcpy permits unaligned tiles.
                            using Vec = double __attribute__((vector_size(32)));
                            Vec acc[MR][2];
                            for (size_t r = 0; r < MR; ++r)
                                for (size_t v = 0; v < 2; ++v)
                                    std::memcpy(&acc[r][v], &C[(i + r) * cols + j + v * 4], sizeof(Vec));
                            for (size_t k = kk; k < kend; ++k) {
                                Vec b0, b1;
                                std::memcpy(&b0, &B[k * cols + j], sizeof(Vec));
                                std::memcpy(&b1, &B[k * cols + j + 4], sizeof(Vec));
                                for (size_t r = 0; r < MR; ++r) {
                                    const double a = A[(i + r) * N + k];
                                    const Vec av = {a, a, a, a};
                                    acc[r][0] += av * b0;
                                    acc[r][1] += av * b1;
                                }
                            }
                            for (size_t r = 0; r < MR; ++r)
                                for (size_t v = 0; v < 2; ++v)
                                    std::memcpy(&C[(i + r) * cols + j + v * 4], &acc[r][v], sizeof(Vec));
                        } else {
                            for (size_t r = 0; r < nr; ++r) {
                                for (size_t k = kk; k < kend; ++k) {
                                    const double a = A[(i + r) * N + k];
                                    for (size_t x = 0; x < nc; ++x)
                                        C[(i + r) * cols + j + x] += a * B[k * cols + j + x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // The original dot product's scalar remainder is fused when available.
    if (N != evenN) {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                C[i * cols + j] = std::fma(A[i * N + evenN], B[evenN * cols + j], C[i * cols + j]);
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

// MPI's count is an int; split large tiles into legal message sizes.
void transfer(double* data, size_t count, int peer, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long n = std::strtoull(arg, &end, 10);
            if (arg[0] == '-' || arg[0] == '\0' || *end != '\0' || n == 0 ||
                n > std::numeric_limits<size_t>::max() / n ||
                n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(n);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    const int pr = rank / dims[1], pc = rank % dims[1];
    const size_t row0 = partBegin(N, pr, dims[0]), row1 = partBegin(N, pr + 1, dims[0]);
    const size_t col0 = partBegin(N, pc, dims[1]), col1 = partBegin(N, pc + 1, dims[1]);
    const size_t rows = row1 - row0, cols = col1 - col0;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * cols), C(rows * cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t k = 0; k < N; ++k) A[i * N + k] = getPseudoRndValue(N, row0 + i, k);
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < cols; ++j) B[k * cols + j] = getPseudoRndValue(N, k, col0 + j);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, rows, cols, N);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(N * N);
            for (int p = 0; p < size; ++p) {
                const int r = p / dims[1], c = p % dims[1];
                const size_t rb = partBegin(N, r, dims[0]), re = partBegin(N, r + 1, dims[0]);
                const size_t cb = partBegin(N, c, dims[1]), ce = partBegin(N, c + 1, dims[1]);
                const size_t width = ce - cb;
                std::vector<double> tile;
                const double* source = C.data();
                if (p != 0) {
                    tile.resize((re - rb) * width);
                    transfer(tile.data(), tile.size(), p, false);
                    source = tile.data();
                }
                if (width != 0)
                    for (size_t i = rb; i < re; ++i)
                        std::copy_n(source + (i - rb) * width, width, result.data() + i * N + cb);
            }
            print_results(result, "MatrixC");
        } else transfer(C.data(), C.size(), 0, true);
    }

    int valid = 1;
    if (validate) {
        double local[25] = {}, actual[25] = {};
        for (size_t pi = 0; pi < 5; ++pi)
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = pi % N, j = pj % N;
                if (i >= row0 && i < row1 && j >= col0 && j < col1)
                    local[pi * 5 + pj] = C[(i - row0) * cols + j - col0];
            }
        MPI_Reduce(local, actual, 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            for (size_t pi = 0; pi < 5 && valid; ++pi)
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t i = pi % N, j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    const double value = actual[pi * 5 + pj];
                    const double relError = std::abs((value - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, value, relError);
                        valid = 0;
                        break;
                    }
                }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
