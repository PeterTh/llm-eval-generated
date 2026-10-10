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
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous partitions, including dimensions not divisible by the grid.
size_t boundary(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(N % parts, size_t(coordinate));
}

constexpr size_t columnTile = 16;

struct Panel {
    std::vector<double> a, b;
    std::vector<MPI_Request> requests;
};

// Chunk messages so MPI's int count does not limit matrix sizes.
void broadcast(std::vector<double>& values, size_t count, int root,
               MPI_Comm comm, std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(values.data() + offset, chunk, MPI_DOUBLE, root, comm,
                   &requests.back());
        offset += chunk;
    }
}

// Fixed-size output tiles keep accumulators in registers across the k loop.
// The compiler can vectorize columns without changing ascending-k sum order.
template <size_t Rows>
void multiplyTile(const double* a, const double* b, double* c,
                  size_t cols, size_t depth, size_t column, size_t bCols) {
#if defined(__AVX__)
    __m256d low[Rows], high[Rows];
    for (size_t i = 0; i < Rows; ++i) {
        low[i] = _mm256_loadu_pd(c + i * cols + column);
        high[i] = _mm256_loadu_pd(c + i * cols + column + 4);
    }
    for (size_t k = 0; k < depth; ++k) {
        const __m256d bLow = _mm256_loadu_pd(b + k * bCols);
        const __m256d bHigh = _mm256_loadu_pd(b + k * bCols + 4);
        for (size_t i = 0; i < Rows; ++i) {
            const __m256d av = _mm256_set1_pd(a[i * depth + k]);
            low[i] = _mm256_add_pd(low[i], _mm256_mul_pd(av, bLow));
            high[i] = _mm256_add_pd(high[i], _mm256_mul_pd(av, bHigh));
        }
    }
    for (size_t i = 0; i < Rows; ++i) {
        _mm256_storeu_pd(c + i * cols + column, low[i]);
        _mm256_storeu_pd(c + i * cols + column + 4, high[i]);
    }
#else
    double sums[Rows][8];
    for (size_t i = 0; i < Rows; ++i)
        for (size_t j = 0; j < 8; ++j)
            sums[i][j] = c[i * cols + column + j];
    for (size_t k = 0; k < depth; ++k) {
        for (size_t i = 0; i < Rows; ++i) {
            const double av = a[i * depth + k];
            for (size_t j = 0; j < 8; ++j)
                sums[i][j] += av * b[k * bCols + j];
        }
    }
    for (size_t i = 0; i < Rows; ++i)
        for (size_t j = 0; j < 8; ++j)
            c[i * cols + column + j] = sums[i][j];
#endif
}

void matrixMultiply(const double* a, const double* b, double* c,
                    size_t rows, size_t cols, size_t depth) {
    constexpr size_t colTile = columnTile;
    for (size_t jj = 0; jj < cols; jj += colTile) {
        const size_t jEnd = std::min(cols, jj + colTile);
        const size_t bCols = jEnd - jj;
        const double* blockB = b + jj * depth;
        size_t i = 0;
        for (; i + 4 <= rows; i += 4) {
            size_t j = jj;
            for (; j + 8 <= jEnd; j += 8)
                multiplyTile<4>(a + i * depth, blockB + j - jj, c + i * cols, cols, depth, j, bCols);
            for (; j < jEnd; ++j) {
                for (size_t r = i; r < i + 4; ++r) {
                    double sum = c[r * cols + j];
                    for (size_t k = 0; k < depth; ++k)
                        sum += a[r * depth + k] * blockB[k * bCols + j - jj];
                    c[r * cols + j] = sum;
                }
            }
        }
        for (; i < rows; ++i) {
            size_t j = jj;
            for (; j + 8 <= jEnd; j += 8)
                multiplyTile<1>(a + i * depth, blockB + j - jj, c + i * cols, cols, depth, j, bCols);
            for (; j < jEnd; ++j) {
                double sum = c[i * cols + j];
                for (size_t k = 0; k < depth; ++k)
                    sum += a[i * depth + k] * blockB[k * bCols + j - jj];
                c[i * cols + j] = sum;
            }
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t N,
                    size_t rowStart, size_t colStart, size_t rows, size_t cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N, j = pj % N;
            if (i < rowStart || i >= rowStart + rows ||
                j < colStart || j >= colStart + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = C[(i - rowStart) * cols + j - colStart];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) return false;
        }
    }
    return true;
}

// Reconstruct row-major output only when external validation is requested.
void printDistributedResults(const std::vector<double>& local, size_t N,
                             const int dims[2], int rank, int ranks, MPI_Comm comm) {
    std::vector<double> full, received;
    if (rank == 0) full.resize(N * N);
    for (int source = 0; source < ranks; ++source) {
        const int r = source / dims[1], c = source % dims[1];
        const size_t r0 = boundary(N, r, dims[0]), c0 = boundary(N, c, dims[1]);
        const size_t rows = boundary(N, r + 1, dims[0]) - r0;
        const size_t cols = boundary(N, c + 1, dims[1]) - c0;
        const size_t count = rows * cols;
        if (rank == 0 && source != 0) received.resize(count);
        for (size_t offset = 0; offset < count;) {
            const int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
            if (rank == source && source != 0)
                MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 0, comm);
            if (rank == 0 && source != 0)
                MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, source, 0,
                         comm, MPI_STATUS_IGNORE);
            offset += chunk;
        }
        if (rank == 0) {
            const auto& block = source == 0 ? local : received;
            for (size_t i = 0; i < rows; ++i)
                std::copy_n(block.data() + i * cols, cols, full.data() + (r0 + i) * N + c0);
        }
    }
    if (rank == 0) print_results(full, "MatrixC");
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end || errno || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (worldRank == 0) printf("Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        if (worldRank == 0) printf("Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    // Avoid empty blocks when there are more ranks than useful matrix partitions.
    int ranks = static_cast<int>(std::min(size_t(worldSize), N * N));
    int dims[2];
    do {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(ranks, 2, dims);
        if (size_t(dims[0]) <= N && size_t(dims[1]) <= N) break;
        --ranks;
    } while (ranks > 0);
    MPI_Comm grid = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                   worldRank, &grid);
    int status = 0;
    if (worldRank < ranks) {
        try {
            int rank;
            MPI_Comm_rank(grid, &rank);
            const int gridRow = rank / dims[1], gridCol = rank % dims[1];
            MPI_Comm rowComm, colComm;
            MPI_Comm_split(grid, gridRow, gridCol, &rowComm);
            MPI_Comm_split(grid, gridCol, gridRow, &colComm);
            const size_t r0 = boundary(N, gridRow, dims[0]);
            const size_t c0 = boundary(N, gridCol, dims[1]);
            const size_t rows = boundary(N, gridRow + 1, dims[0]) - r0;
            const size_t cols = boundary(N, gridCol + 1, dims[1]) - c0;
            // Enough cyclic panels to distribute inputs even on large process grids.
            const size_t width = std::min(size_t(128),
                std::max(size_t(1), N / (2 * size_t(std::max(dims[0], dims[1])))));
            const size_t panels = (N + width - 1) / width;
            std::vector<Panel> owned(panels);
            std::vector<double> C(rows * cols, 0.0);
            Panel buffers[2];
            for (auto& buffer : buffers) {
                buffer.a.resize(rows * width);
                buffer.b.resize(width * cols);
                buffer.requests.reserve(2 + (rows * width) / INT_MAX + (width * cols) / INT_MAX);
            }
            if (rank == 0) {
                printf("Matrix Multiplication Benchmark\n");
                printf("Matrix size: %zu x %zu\n", N, N);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("Initializing matrices...\n");
            }
            // Generate each input element once, on the rank that owns its panel.
            for (size_t p = 0; p < panels; ++p) {
                const size_t k0 = p * width, depth = std::min(width, N - k0);
                if (p % dims[1] == size_t(gridCol)) {
                    owned[p].a.resize(rows * depth);
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t k = 0; k < depth; ++k)
                            owned[p].a[i * depth + k] = getPseudoRndValue(N, r0 + i, k0 + k);
                }
                if (p % dims[0] == size_t(gridRow)) {
                    owned[p].b.resize(depth * cols);
                    // Pack column tiles to avoid large-stride cache conflicts in the kernel.
                    for (size_t jj = 0; jj < cols; jj += columnTile) {
                        const size_t tileCols = std::min(columnTile, cols - jj);
                        for (size_t k = 0; k < depth; ++k)
                            for (size_t j = 0; j < tileCols; ++j)
                                owned[p].b[jj * depth + k * tileCols + j] =
                                    getPseudoRndValue(N, k0 + k, c0 + jj + j);
                    }
                }
            }
            auto startPanel = [&](size_t p, Panel& buffer) {
                const size_t depth = std::min(width, N - p * width);
                const int aRoot = static_cast<int>(p % dims[1]);
                const int bRoot = static_cast<int>(p % dims[0]);
                if (gridCol == aRoot)
                    std::copy(owned[p].a.begin(), owned[p].a.end(), buffer.a.begin());
                if (gridRow == bRoot)
                    std::copy(owned[p].b.begin(), owned[p].b.end(), buffer.b.begin());
                buffer.requests.clear();
                broadcast(buffer.a, rows * depth, aRoot, rowComm, buffer.requests);
                broadcast(buffer.b, cols * depth, bRoot, colComm, buffer.requests);
            };
            if (rank == 0) printf("Computing matrix multiplication...\n");
            MPI_Barrier(grid);
            const double start = MPI_Wtime();
            startPanel(0, buffers[0]);
            for (size_t p = 0; p < panels; ++p) {
                Panel& current = buffers[p % 2];
                MPI_Waitall(static_cast<int>(current.requests.size()),
                            current.requests.data(), MPI_STATUSES_IGNORE);
                // Separate buffers allow the next broadcasts to overlap this computation.
                if (p + 1 < panels) startPanel(p + 1, buffers[(p + 1) % 2]);
                matrixMultiply(current.a.data(), current.b.data(), C.data(), rows, cols,
                               std::min(width, N - p * width));
            }
            const double elapsed = MPI_Wtime() - start;
            double seconds = 0.0;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
            if (rank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
                printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
            }
            if (printResults) printDistributedResults(C, N, dims, rank, ranks, grid);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                int valid = validateResult(C, N, r0, c0, rows, cols) ? 1 : 0;
                int allValid;
                MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, grid);
                status = allValid ? 0 : 1;
                if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
            }
            MPI_Comm_free(&rowComm);
            MPI_Comm_free(&colComm);
            MPI_Comm_free(&grid);
        } catch (const std::exception& e) {
            fprintf(stderr, "Rank %d: %s\n", worldRank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
