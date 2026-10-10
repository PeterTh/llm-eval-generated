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

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Keep the original initialization, including its size_t arithmetic.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Block {
    size_t row, col, rows, cols;
};

// Balanced contiguous partitions also handle N smaller than the process grid.
size_t partitionStart(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(N % parts, static_cast<size_t>(coordinate));
}

Block rankBlock(size_t N, int rank, const int dims[2]) {
    const int row = rank / dims[1], col = rank % dims[1];
    const size_t firstRow = partitionStart(N, row, dims[0]);
    const size_t firstCol = partitionStart(N, col, dims[1]);
    return {firstRow, firstCol,
            partitionStart(N, row + 1, dims[0]) - firstRow,
            partitionStart(N, col + 1, dims[1]) - firstCol};
}

// An odd number of cache lines per B row avoids power-of-two cache conflicts.
size_t bLeadingDimension(const Block& block) {
    return (((block.cols + 7) / 8) | size_t{1}) * 8;
}

void initMatrices(std::vector<double>& A, std::vector<double>& B,
                  size_t N, const Block& block) {
    for (size_t i = 0; i < block.rows; ++i)
        for (size_t k = 0; k < N; ++k)
            A[i * N + k] = getPseudoRndValue(N, block.row + i, k);
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < block.cols; ++j)
            B[k * bLeadingDimension(block) + j] = getPseudoRndValue(N, k, block.col + j);
}

inline double multiplyAdd(double a, double b, double c) {
#if defined(__AVX2__) && defined(__FMA__)
    // Match the SIMD kernel even on edge rows and columns.
    return std::fma(a, b, c);
#else
    return c + a * b;
#endif
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, const Block& block) {
    // Four rows share each contiguous B vector. Blocking keeps the working
    // panels in cache, and vectorization across columns preserves k order.
    constexpr size_t rowTile = 32, colTile = 128, innerTile = 128;
    const size_t cols = block.cols;
    const size_t bStride = bLeadingDimension(block);
    for (size_t ii = 0; ii < block.rows; ii += rowTile) {
        const size_t iEnd = std::min(ii + rowTile, block.rows);
        for (size_t jj = 0; jj < cols; jj += colTile) {
            const size_t jEnd = std::min(jj + colTile, cols);
            for (size_t kk = 0; kk < N; kk += innerTile) {
                const size_t kEnd = std::min(kk + innerTile, N);
                size_t i = ii;
                for (; i + 4 <= iEnd; i += 4) {
                    double* c0 = C.data() + i * cols;
                    double* c1 = c0 + cols;
                    double* c2 = c1 + cols;
                    double* c3 = c2 + cols;
                    size_t j = jj;
#if defined(__AVX2__) && defined(__FMA__)
                    for (; j + 8 <= jEnd; j += 8) {
                        // Eight SIMD accumulators hold a 4x8 output tile.
                        // Each dot product still accumulates k in order.
                        __m256d s00 = _mm256_loadu_pd(c0 + j);
                        __m256d s01 = _mm256_loadu_pd(c0 + j + 4);
                        __m256d s10 = _mm256_loadu_pd(c1 + j);
                        __m256d s11 = _mm256_loadu_pd(c1 + j + 4);
                        __m256d s20 = _mm256_loadu_pd(c2 + j);
                        __m256d s21 = _mm256_loadu_pd(c2 + j + 4);
                        __m256d s30 = _mm256_loadu_pd(c3 + j);
                        __m256d s31 = _mm256_loadu_pd(c3 + j + 4);
                        for (size_t k = kk; k < kEnd; ++k) {
                            const double* b = B.data() + k * bStride + j;
                            const __m256d b0 = _mm256_loadu_pd(b);
                            const __m256d b1 = _mm256_loadu_pd(b + 4);
                            const __m256d a0 = _mm256_set1_pd(A[i * N + k]);
                            const __m256d a1 = _mm256_set1_pd(A[(i + 1) * N + k]);
                            const __m256d a2 = _mm256_set1_pd(A[(i + 2) * N + k]);
                            const __m256d a3 = _mm256_set1_pd(A[(i + 3) * N + k]);
                            s00 = _mm256_fmadd_pd(a0, b0, s00);
                            s01 = _mm256_fmadd_pd(a0, b1, s01);
                            s10 = _mm256_fmadd_pd(a1, b0, s10);
                            s11 = _mm256_fmadd_pd(a1, b1, s11);
                            s20 = _mm256_fmadd_pd(a2, b0, s20);
                            s21 = _mm256_fmadd_pd(a2, b1, s21);
                            s30 = _mm256_fmadd_pd(a3, b0, s30);
                            s31 = _mm256_fmadd_pd(a3, b1, s31);
                        }
                        _mm256_storeu_pd(c0 + j, s00);
                        _mm256_storeu_pd(c0 + j + 4, s01);
                        _mm256_storeu_pd(c1 + j, s10);
                        _mm256_storeu_pd(c1 + j + 4, s11);
                        _mm256_storeu_pd(c2 + j, s20);
                        _mm256_storeu_pd(c2 + j + 4, s21);
                        _mm256_storeu_pd(c3 + j, s30);
                        _mm256_storeu_pd(c3 + j + 4, s31);
                    }
#endif
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a0 = A[i * N + k];
                        const double a1 = A[(i + 1) * N + k];
                        const double a2 = A[(i + 2) * N + k];
                        const double a3 = A[(i + 3) * N + k];
                        const double* b = B.data() + k * bStride;
                        for (size_t tail = j; tail < jEnd; ++tail) {
                            c0[tail] = multiplyAdd(a0, b[tail], c0[tail]);
                            c1[tail] = multiplyAdd(a1, b[tail], c1[tail]);
                            c2[tail] = multiplyAdd(a2, b[tail], c2[tail]);
                            c3[tail] = multiplyAdd(a3, b[tail], c3[tail]);
                        }
                    }
                }
                for (; i < iEnd; ++i) {
                    double* c = C.data() + i * cols;
                    for (size_t k = kk; k < kEnd; ++k) {
                        const double a = A[i * N + k];
                        const double* b = B.data() + k * bStride;
                        for (size_t j = jj; j < jEnd; ++j)
                            c[j] = multiplyAdd(a, b[j], c[j]);
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, const Block& block) {
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t row = pi % N;
        if (row < block.row || row - block.row >= block.rows) continue;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t col = pj % N;
            if (col < block.col || col - block.col >= block.cols) continue;
            const size_t i = row - block.row, j = col - block.col;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[i * N + k] * B[k * bLeadingDimension(block) + j];
            const double actual = C[i * block.cols + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       row, col, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

// Transfer in chunks to avoid the int count limit of MPI implementations
// predating MPI-4. Only -r requires a full matrix on rank zero.
void gatherResults(const std::vector<double>& local, size_t N, int rank,
                   int ranks, const int dims[2]) {
    if (rank != 0) {
        for (size_t offset = 0; offset < local.size();) {
            const int count = static_cast<int>(std::min(local.size() - offset,
                                                       static_cast<size_t>(INT_MAX)));
            MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += count;
        }
        return;
    }
    std::vector<double> full(N * N);
    std::vector<double> received;
    for (int source = 0; source < ranks; ++source) {
        const Block block = rankBlock(N, source, dims);
        const size_t count = block.rows * block.cols;
        const double* data = local.data();
        if (source != 0) {
            received.resize(count);
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset,
                                                           static_cast<size_t>(INT_MAX)));
                MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
            data = received.data();
        }
        if (count != 0)
            for (size_t i = 0; i < block.rows; ++i)
                std::copy_n(data + i * block.cols, block.cols,
                            full.data() + (block.row + i) * N + block.col);
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end || value == 0 ||
                value > std::numeric_limits<size_t>::max() ||
                value > std::vector<double>().max_size() / value) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                parseStatus = 1;
                break;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
            break;
        }
    }
    if (help || parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus;
    }

    int status = 0;
    try {
        int dims[2] = {0, 0};
        MPI_Dims_create(ranks, 2, dims);
        const Block block = rankBlock(N, rank, dims);
        const bool nonempty = block.rows != 0 && block.cols != 0;
        std::vector<double> A(nonempty ? block.rows * N : 0);
        std::vector<double> B(nonempty ? N * bLeadingDimension(block) : 0);
        std::vector<double> C(block.rows * block.cols, 0.0);
        if (rank == 0) {
            printf("Matrix Multiplication Benchmark\n");
            printf("Matrix size: %zu x %zu\n", N, N);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing matrices...\n");
        }
        if (nonempty) initMatrices(A, B, N, block);
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (nonempty) matrixMultiply(A, B, C, N, block);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
            const double gflops = duration > 0 ? (2.0 * N * N * N) / duration / 1e9 : 0.0;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }
        if (printResults) gatherResults(C, N, rank, ranks, dims);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const int localValid = validateResult(A, B, C, N, block) ? 1 : 0;
            int valid = 0;
            MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
