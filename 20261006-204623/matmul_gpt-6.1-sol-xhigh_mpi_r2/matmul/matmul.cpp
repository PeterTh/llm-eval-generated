#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

namespace {

constexpr size_t tileRows = 4;
constexpr size_t tileCols = 8;
constexpr size_t panelSize = 256;

size_t blockStart(size_t N, int parts, int coordinate) {
    return (N / parts) * coordinate + std::min(N % parts, size_t(coordinate));
}

size_t paddedSize(size_t n, size_t tile) {
    return ((n + tile - 1) / tile) * tile;
}

struct Grid {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm colComm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[2] = {0, 0};
    int coords[2] = {0, 0};
    size_t rowStart = 0, colStart = 0, rows = 0, cols = 0;
    size_t paddedRows = 0, paddedCols = 0;

    explicit Grid(size_t N) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dims);
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);
        MPI_Comm_split(comm, coords[0], coords[1], &rowComm);
        MPI_Comm_split(comm, coords[1], coords[0], &colComm);
        rowStart = blockStart(N, dims[0], coords[0]);
        colStart = blockStart(N, dims[1], coords[1]);
        rows = blockStart(N, dims[0], coords[0] + 1) - rowStart;
        cols = blockStart(N, dims[1], coords[1] + 1) - colStart;
        paddedRows = paddedSize(rows, tileRows);
        paddedCols = paddedSize(cols, tileCols);
    }

    ~Grid() {
        // A rank that throws must reach MPI_Abort instead of entering collective frees.
        if (std::uncaught_exceptions()) return;
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&comm);
    }
};

void initMatrix(std::vector<double>& mat, size_t N, const Grid& grid) {
    for (size_t i = 0; i < grid.rows; ++i) {
        for (size_t j = 0; j < grid.cols; ++j) {
            mat[i * grid.cols + j] = getPseudoRndValue(N, grid.rowStart + i,
                                                       grid.colStart + j);
        }
    }
}

struct Panel {
    size_t start, width;
    int aRoot, bRoot;
};

struct AlignedBuffer {
    std::vector<double> storage;
    size_t offset = 0;

    void resize(size_t count) {
        // Overallocate within the vector so packed SIMD loads cannot cross cache lines.
        constexpr size_t alignment = 64;
        storage.resize(count + alignment / sizeof(double) - 1);
        offset = ((alignment - reinterpret_cast<std::uintptr_t>(storage.data()) % alignment)
                  % alignment) / sizeof(double);
    }

    double* data() { return storage.data() + offset; }
    const double* data() const { return storage.data() + offset; }
};

struct PanelBuffer {
    AlignedBuffer a, b;
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
};

std::vector<Panel> makePanels(size_t N, const Grid& grid) {
    // Every panel lies within both an A column block and a B row block.
    // Bounding the padded message sizes also keeps MPI counts within INT_MAX.
    const size_t largestBlock = std::max(
        paddedSize(blockStart(N, grid.dims[0], 1), tileRows),
        paddedSize(blockStart(N, grid.dims[1], 1), tileCols));
    const size_t width = std::min(panelSize, size_t(INT_MAX) / largestBlock);
    std::vector<Panel> panels;
    int aRoot = 0, bRoot = 0;
    for (size_t k = 0; k < N;) {
        while (blockStart(N, grid.dims[1], aRoot + 1) <= k) ++aRoot;
        while (blockStart(N, grid.dims[0], bRoot + 1) <= k) ++bRoot;
        const size_t end = std::min({N, k + width,
            blockStart(N, grid.dims[1], aRoot + 1),
            blockStart(N, grid.dims[0], bRoot + 1)});
        panels.push_back({k, end - k, aRoot, bRoot});
        k = end;
    }
    return panels;
}

void broadcastPanel(const std::vector<double>& A, const std::vector<double>& B,
                    const Grid& grid, const Panel& panel, PanelBuffer& buffer) {
    // Broadcast in microkernel layout, avoiding a second packing pass at receivers.
    if (grid.coords[1] == panel.aRoot) {
        const size_t offset = panel.start - grid.colStart;
        for (size_t i = 0; i < grid.paddedRows; i += tileRows) {
            double* packed = buffer.a.data() + i * panel.width;
            for (size_t k = 0; k < panel.width; ++k) {
                for (size_t r = 0; r < tileRows; ++r) {
                    packed[k * tileRows + r] = i + r < grid.rows
                        ? A[(i + r) * grid.cols + offset + k] : 0.0;
                }
            }
        }
    }
    if (grid.coords[0] == panel.bRoot) {
        const size_t offset = panel.start - grid.rowStart;
        for (size_t j = 0; j < grid.paddedCols; j += tileCols) {
            double* packed = buffer.b.data() + j * panel.width;
            for (size_t k = 0; k < panel.width; ++k) {
                for (size_t c = 0; c < tileCols; ++c) {
                    packed[k * tileCols + c] = j + c < grid.cols
                        ? B[(offset + k) * grid.cols + j + c] : 0.0;
                }
            }
        }
    }
    MPI_Ibcast(buffer.a.data(), int(grid.paddedRows * panel.width), MPI_DOUBLE,
               panel.aRoot, grid.rowComm, &buffer.requests[0]);
    MPI_Ibcast(buffer.b.data(), int(grid.paddedCols * panel.width), MPI_DOUBLE,
               panel.bRoot, grid.colComm, &buffer.requests[1]);
}

void multiplyTile(const double* a, const double* b, double* c,
                  size_t stride, size_t width) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c00 = _mm256_loadu_pd(c);
    __m256d c01 = _mm256_loadu_pd(c + 4);
    __m256d c10 = _mm256_loadu_pd(c + stride);
    __m256d c11 = _mm256_loadu_pd(c + stride + 4);
    __m256d c20 = _mm256_loadu_pd(c + 2 * stride);
    __m256d c21 = _mm256_loadu_pd(c + 2 * stride + 4);
    __m256d c30 = _mm256_loadu_pd(c + 3 * stride);
    __m256d c31 = _mm256_loadu_pd(c + 3 * stride + 4);
    for (size_t k = 0; k < width; ++k, a += tileRows, b += tileCols) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        const __m256d a0 = _mm256_broadcast_sd(a);
        const __m256d a1 = _mm256_broadcast_sd(a + 1);
        const __m256d a2 = _mm256_broadcast_sd(a + 2);
        const __m256d a3 = _mm256_broadcast_sd(a + 3);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
    }
    _mm256_storeu_pd(c, c00);
    _mm256_storeu_pd(c + 4, c01);
    _mm256_storeu_pd(c + stride, c10);
    _mm256_storeu_pd(c + stride + 4, c11);
    _mm256_storeu_pd(c + 2 * stride, c20);
    _mm256_storeu_pd(c + 2 * stride + 4, c21);
    _mm256_storeu_pd(c + 3 * stride, c30);
    _mm256_storeu_pd(c + 3 * stride + 4, c31);
#else
    // Portable kernel: vectorize across columns without reassociating the k sum.
    double sums[tileRows][tileCols];
    for (size_t r = 0; r < tileRows; ++r)
        std::copy_n(c + r * stride, tileCols, sums[r]);
    for (size_t k = 0; k < width; ++k, a += tileRows, b += tileCols)
        for (size_t r = 0; r < tileRows; ++r)
            for (size_t j = 0; j < tileCols; ++j)
                sums[r][j] += a[r] * b[j];
    for (size_t r = 0; r < tileRows; ++r)
        std::copy_n(sums[r], tileCols, c + r * stride);
#endif
}

void multiplyPanel(const PanelBuffer& buffer, std::vector<double>& C,
                   const Grid& grid, size_t width, MPI_Request* pending) {
    // Keep the working A/B panels and C tiles in the private cache.
    constexpr size_t cacheRows = 64, cacheCols = 64;
    for (size_t jj = 0; jj < grid.paddedCols; jj += cacheCols) {
        for (size_t ii = 0; ii < grid.paddedRows; ii += cacheRows) {
            const size_t endI = std::min(ii + cacheRows, grid.paddedRows);
            const size_t endJ = std::min(jj + cacheCols, grid.paddedCols);
            for (size_t j = jj; j < endJ; j += tileCols) {
                for (size_t i = ii; i < endI; i += tileRows) {
                    multiplyTile(buffer.a.data() + i * width,
                                 buffer.b.data() + j * width,
                                 C.data() + i * grid.paddedCols + j,
                                 grid.paddedCols, width);
                }
            }
            // MPI implementations without background progress still overlap transfers.
            if (pending) {
                int complete = 0;
                MPI_Testall(2, pending, &complete, MPI_STATUSES_IGNORE);
                if (complete) pending = nullptr;
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const Grid& grid,
                    const std::vector<Panel>& panels, std::array<PanelBuffer, 2>& buffers) {
    // Pipelined SUMMA: receive the next panels while computing the current ones.
    broadcastPanel(A, B, grid, panels[0], buffers[0]);
    for (size_t p = 0; p < panels.size(); ++p) {
        PanelBuffer& current = buffers[p % 2];
        PanelBuffer& next = buffers[(p + 1) % 2];
        MPI_Waitall(2, current.requests, MPI_STATUSES_IGNORE);
        const bool hasNext = p + 1 < panels.size();
        if (hasNext) broadcastPanel(A, B, grid, panels[p + 1], next);
        multiplyPanel(current, C, grid, panels[p].width,
                      hasNext ? next.requests : nullptr);
    }
}

bool validateResult(const std::vector<double>& C, size_t N, const Grid& grid) {
    int valid = 1;
    // Validate the same 25 positions on their owning ranks, without gathering A/B/C.
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N, j = pj % N;
            if (i < grid.rowStart || i >= grid.rowStart + grid.rows ||
                j < grid.colStart || j >= grid.colStart + grid.cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = C[(i - grid.rowStart) * grid.paddedCols + j - grid.colStart];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                valid = 0;
            }
        }
    }
    int allValid = 0;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, grid.comm);
    return allValid != 0;
}

void printDistributedResults(const std::vector<double>& C, size_t N, const Grid& grid) {
    // Only -r materializes the full result for row-major statistics and hashing.
    // Derived types avoid packing and the INT_MAX total-element limit of Gatherv.
    if (grid.rank == 0) {
        std::vector<double> result(N * N);
        for (size_t i = 0; i < grid.rows; ++i)
            std::copy_n(C.data() + i * grid.paddedCols, grid.cols, result.data() + i * N);
        for (int rank = 1; rank < grid.size; ++rank) {
            int coords[2];
            MPI_Cart_coords(grid.comm, rank, 2, coords);
            int sizes[2] = {int(N), int(N)};
            int starts[2] = {int(blockStart(N, grid.dims[0], coords[0])),
                             int(blockStart(N, grid.dims[1], coords[1]))};
            int subsizes[2] = {
                int(blockStart(N, grid.dims[0], coords[0] + 1)) - starts[0],
                int(blockStart(N, grid.dims[1], coords[1] + 1)) - starts[1]};
            if (!subsizes[0] || !subsizes[1]) continue;
            MPI_Datatype block;
            MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            MPI_Recv(result.data(), 1, block, rank, 0, grid.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
        print_results(result, "MatrixC");
    } else if (grid.rows && grid.cols) {
        MPI_Datatype block;
        MPI_Type_vector(int(grid.rows), int(grid.cols), int(grid.paddedCols), MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(C.data(), 1, block, 0, 0, grid.comm);
        MPI_Type_free(&block);
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

int parseArguments(int argc, char** argv, unsigned long long& N, int* options) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            N = std::strtoull(value, &end, 10);
            if (errno || value == end || *end || value[0] == '-' || N == 0 ||
                N > INT_MAX || N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
                printf("Invalid matrix size: %s\n", value);
                return 2;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            options[0] = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            options[1] = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }
    }
    return 0;
}

int runBenchmark(size_t N, bool validate, bool printResults) {
    Grid grid(N);
    if (grid.rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(grid.rows * grid.cols), B(grid.rows * grid.cols);
    std::vector<double> C(grid.paddedRows * grid.paddedCols, 0.0);
    initMatrix(A, N, grid);
    initMatrix(B, N, grid);
    const std::vector<Panel> panels = makePanels(N, grid);
    size_t maxWidth = 0;
    for (const Panel& panel : panels) maxWidth = std::max(maxWidth, panel.width);
    std::array<PanelBuffer, 2> buffers;
    for (PanelBuffer& buffer : buffers) {
        buffer.a.resize(grid.paddedRows * maxWidth);
        buffer.b.resize(grid.paddedCols * maxWidth);
    }

    if (grid.rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(grid.comm);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, grid, panels, buffers);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
    if (grid.rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        const double n = static_cast<double>(N);
        const double gflops = (2.0 * n * n * n) / std::max(duration, 1e-12) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) printDistributedResults(C, N, grid);
    if (validate) {
        if (grid.rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(C, N, grid);
        if (grid.rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    unsigned long long N = 512;
    int options[2] = {0, 0};
    int status = rank == 0 ? parseArguments(argc, argv, N, options) : 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status) {
        MPI_Finalize();
        return status == 1 ? 0 : 1;
    }
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(options, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int result = 1;
    try {
        result = runBenchmark(size_t(N), options[0] != 0, options[1] != 0);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
