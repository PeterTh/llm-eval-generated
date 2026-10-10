#include <mpi.h>

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

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous partitions, including empty tiles when there are more ranks
// than matrix rows/columns. Avoid N * coordinate overflow.
size_t boundary(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(N % parts, static_cast<size_t>(coordinate));
}

// Four rows share each loaded B vector. Independent columns can be vectorized
// without changing the order of additions along the reduction dimension.
void multiplyPanel(const double* A, const double* B, double* C,
                   size_t rows, size_t cols, size_t depth) {
    size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        size_t j = 0;
        for (; j + 8 <= cols; j += 8) {
#if defined(__AVX__)
            // Explicit SIMD keeps the accumulators in registers on x86.
            __m256d c00 = _mm256_loadu_pd(C + (i + 0) * cols + j + 0);
            __m256d c01 = _mm256_loadu_pd(C + (i + 0) * cols + j + 4);
            __m256d c10 = _mm256_loadu_pd(C + (i + 1) * cols + j + 0);
            __m256d c11 = _mm256_loadu_pd(C + (i + 1) * cols + j + 4);
            __m256d c20 = _mm256_loadu_pd(C + (i + 2) * cols + j + 0);
            __m256d c21 = _mm256_loadu_pd(C + (i + 2) * cols + j + 4);
            __m256d c30 = _mm256_loadu_pd(C + (i + 3) * cols + j + 0);
            __m256d c31 = _mm256_loadu_pd(C + (i + 3) * cols + j + 4);
            for (size_t k = 0; k < depth; ++k) {
                const __m256d b0 = _mm256_loadu_pd(B + k * cols + j + 0);
                const __m256d b1 = _mm256_loadu_pd(B + k * cols + j + 4);
                const __m256d a0 = _mm256_set1_pd(A[(i + 0) * depth + k]);
                c00 = _mm256_add_pd(c00, _mm256_mul_pd(a0, b0));
                c01 = _mm256_add_pd(c01, _mm256_mul_pd(a0, b1));
                const __m256d a1 = _mm256_set1_pd(A[(i + 1) * depth + k]);
                c10 = _mm256_add_pd(c10, _mm256_mul_pd(a1, b0));
                c11 = _mm256_add_pd(c11, _mm256_mul_pd(a1, b1));
                const __m256d a2 = _mm256_set1_pd(A[(i + 2) * depth + k]);
                c20 = _mm256_add_pd(c20, _mm256_mul_pd(a2, b0));
                c21 = _mm256_add_pd(c21, _mm256_mul_pd(a2, b1));
                const __m256d a3 = _mm256_set1_pd(A[(i + 3) * depth + k]);
                c30 = _mm256_add_pd(c30, _mm256_mul_pd(a3, b0));
                c31 = _mm256_add_pd(c31, _mm256_mul_pd(a3, b1));
            }
            _mm256_storeu_pd(C + (i + 0) * cols + j + 0, c00);
            _mm256_storeu_pd(C + (i + 0) * cols + j + 4, c01);
            _mm256_storeu_pd(C + (i + 1) * cols + j + 0, c10);
            _mm256_storeu_pd(C + (i + 1) * cols + j + 4, c11);
            _mm256_storeu_pd(C + (i + 2) * cols + j + 0, c20);
            _mm256_storeu_pd(C + (i + 2) * cols + j + 4, c21);
            _mm256_storeu_pd(C + (i + 3) * cols + j + 0, c30);
            _mm256_storeu_pd(C + (i + 3) * cols + j + 4, c31);
#else
            double c0[8], c1[8], c2[8], c3[8];
            for (size_t x = 0; x < 8; ++x) {
                c0[x] = C[i * cols + j + x];
                c1[x] = C[(i + 1) * cols + j + x];
                c2[x] = C[(i + 2) * cols + j + x];
                c3[x] = C[(i + 3) * cols + j + x];
            }
            for (size_t k = 0; k < depth; ++k) {
                const double a0 = A[i * depth + k], a1 = A[(i + 1) * depth + k];
                const double a2 = A[(i + 2) * depth + k], a3 = A[(i + 3) * depth + k];
                for (size_t x = 0; x < 8; ++x) {
                    const double b = B[k * cols + j + x];
                    c0[x] += a0 * b;
                    c1[x] += a1 * b;
                    c2[x] += a2 * b;
                    c3[x] += a3 * b;
                }
            }
            for (size_t x = 0; x < 8; ++x) {
                C[i * cols + j + x] = c0[x];
                C[(i + 1) * cols + j + x] = c1[x];
                C[(i + 2) * cols + j + x] = c2[x];
                C[(i + 3) * cols + j + x] = c3[x];
            }
#endif
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

// Split broadcasts to support panels exceeding MPI's int count limit.
void broadcastPanel(double* data, size_t count, int root, MPI_Comm comm,
                    std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, root, comm, &requests.back());
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

int run(int argc, char** argv, int rank, int ranks) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max() ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
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

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const int row = rank / dims[1], col = rank % dims[1];
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    const size_t rowStart = boundary(N, row, dims[0]);
    const size_t colStart = boundary(N, col, dims[1]);
    const size_t rows = boundary(N, row + 1, dims[0]) - rowStart;
    const size_t cols = boundary(N, col + 1, dims[1]) - colStart;
    const size_t block = std::min(size_t{128}, (N - 1) / std::max(dims[0], dims[1]) + 1);
    const size_t panels = (N - 1) / block + 1;
    std::vector<double> C(rows * cols, 0.0);
    std::vector<std::vector<double>> A(panels), B(panels);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    // Block-cyclic ownership along k keeps input storage distributed as well.
    // The deterministic initializer needs no root allocation or scatter.
    for (size_t p = 0; p < panels; ++p) {
        const size_t start = p * block, depth = std::min(block, N - start);
        if (static_cast<int>(p % dims[1]) == col) {
            A[p].resize(rows * depth);
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < depth; ++k)
                    A[p][i * depth + k] = getPseudoRndValue(N, rowStart + i, start + k);
        }
        if (static_cast<int>(p % dims[0]) == row) {
            B[p].resize(depth * cols);
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < cols; ++j)
                    B[p][k * cols + j] = getPseudoRndValue(N, start + k, colStart + j);
        }
    }
    std::vector<double> aBuffer[2], bBuffer[2];
    std::vector<MPI_Request> requests[2];
    for (int slot = 0; slot < 2; ++slot) {
        aBuffer[slot].resize(rows * block);
        bBuffer[slot].resize(block * cols);
    }
    auto post = [&](size_t p) {
        const size_t depth = std::min(block, N - p * block);
        const int slot = p % 2;
        requests[slot].clear();
        if (static_cast<int>(p % dims[1]) == col)
            std::copy(A[p].begin(), A[p].end(), aBuffer[slot].begin());
        if (static_cast<int>(p % dims[0]) == row)
            std::copy(B[p].begin(), B[p].end(), bBuffer[slot].begin());
        broadcastPanel(aBuffer[slot].data(), rows * depth, p % dims[1], rowComm, requests[slot]);
        broadcastPanel(bBuffer[slot].data(), depth * cols, p % dims[0], colComm, requests[slot]);
    };
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    post(0);
    for (size_t p = 0; p < panels; ++p) {
        const int slot = p % 2;
        for (auto& request : requests[slot]) MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (p + 1 < panels) post(p + 1);
        multiplyPanel(aBuffer[slot].data(), bBuffer[slot].data(), C.data(),
                      rows, cols, std::min(block, N - p * block));
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(elapsed * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / elapsed / 1e9);
    }

    // Assemble the original row-major result only when explicitly requested.
    // Chunked tile transfers avoid MPI count/displacement overflow.
    if (printResults) {
        std::vector<double> full, tile;
        if (rank == 0) full.resize(N * N);
        for (int source = 0; source < ranks; ++source) {
            const int sr = source / dims[1], sc = source % dims[1];
            const size_t r0 = boundary(N, sr, dims[0]), c0 = boundary(N, sc, dims[1]);
            const size_t nr = boundary(N, sr + 1, dims[0]) - r0;
            const size_t nc = boundary(N, sc + 1, dims[1]) - c0;
            if (rank == 0 && source != 0) tile.resize(nr * nc);
            for (size_t offset = 0; offset < nr * nc;) {
                const int count = static_cast<int>(std::min(nr * nc - offset, static_cast<size_t>(INT_MAX)));
                if (source != 0 && rank == source)
                    MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if (source != 0 && rank == 0)
                    MPI_Recv(tile.data() + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += count;
            }
            if (rank == 0 && nc != 0) {
                const auto& data = source == 0 ? C : tile;
                for (size_t i = 0; i < nr; ++i)
                    std::copy_n(data.data() + i * nc, nc, full.data() + (r0 + i) * N + c0);
            }
        }
        if (rank == 0) print_results(full, "MatrixC");
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        for (size_t pi = 0; pi < 5; ++pi) {
            const size_t i = pi % N;
            if (i < rowStart || i >= rowStart + rows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = pj % N;
                if (j < colStart || j >= colStart + cols) continue;
                double expected = 0;
                for (size_t k = 0; k < N; ++k)
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                const double actual = C[(i - rowStart) * cols + j - colStart];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(actual) || error > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, error);
                    valid = 0;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
