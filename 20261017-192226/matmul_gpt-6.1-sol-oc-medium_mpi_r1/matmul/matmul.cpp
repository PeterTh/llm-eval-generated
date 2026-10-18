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

// Generate exactly the same entries as the serial benchmark, using global indices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    size_t begin;
    size_t size;
};

Range partition(size_t N, int coordinate, int parts) {
    const size_t base = N / parts;
    const size_t extra = N % parts;
    return {base * coordinate + std::min(static_cast<size_t>(coordinate), extra),
            base + (static_cast<size_t>(coordinate) < extra)};
}

struct Panel {
    std::vector<double> a;
    std::vector<double> b;
};

// Each C block is local to one process. A panels are distributed cyclically
// across process columns, and B panels across process rows (2D SUMMA).
std::vector<Panel> initPanels(size_t N, size_t panelWidth, Range rows, Range cols,
                              const int* coords, const int* dims) {
    const size_t count = (N + panelWidth - 1) / panelWidth;
    std::vector<Panel> panels(count);
    for (size_t s = 0; s < count; ++s) {
        const size_t first = s * panelWidth;
        const size_t width = std::min(panelWidth, N - first);
        if (s % dims[1] == static_cast<size_t>(coords[1])) {
            auto& a = panels[s].a;
            a.resize(rows.size * width);
            for (size_t i = 0; i < rows.size; ++i)
                for (size_t k = 0; k < width; ++k)
                    a[i * width + k] = getPseudoRndValue(N, rows.begin + i, first + k);
        }
        if (s % dims[0] == static_cast<size_t>(coords[0])) {
            auto& b = panels[s].b;
            b.resize(width * cols.size);
            for (size_t k = 0; k < width; ++k)
                for (size_t j = 0; j < cols.size; ++j)
                    b[k * cols.size + j] = getPseudoRndValue(N, first + k, cols.begin + j);
        }
    }
    return panels;
}

// Contiguous inner loops vectorize across independent output elements. Keep
// increasing k order and do not reassociate sums, preserving serial arithmetic.
void multiplyPanel(const double* a, const double* b, std::vector<double>& c,
                   size_t rows, size_t cols, size_t width) {
    constexpr size_t rowTile = 32;
    constexpr size_t colTile = 128;
    for (size_t ii = 0; ii < rows; ii += rowTile) {
        for (size_t jj = 0; jj < cols; jj += colTile) {
            const size_t iend = std::min(ii + rowTile, rows);
            const size_t jend = std::min(jj + colTile, cols);
            for (size_t k = 0; k < width; ++k) {
                const double* bp = b + k * cols;
                for (size_t i = ii; i < iend; ++i) {
                    const double av = a[i * width + k];
                    double* cp = c.data() + i * cols;
                    for (size_t j = jj; j < jend; ++j)
                        cp[j] += av * bp[j];
                }
            }
        }
    }
}

void matrixMultiply(std::vector<Panel>& panels, std::vector<double>& c,
                    size_t N, size_t panelWidth, Range rows, Range cols,
                    const int* coords, const int* dims,
                    MPI_Comm rowComm, MPI_Comm colComm) {
    struct Pending {
        std::vector<double> aBuffer, bBuffer;
        const double* a = nullptr;
        const double* b = nullptr;
        MPI_Request requests[2];
    } pending[2];
    for (auto& p : pending) {
        p.aBuffer.resize(rows.size * panelWidth);
        p.bBuffer.resize(cols.size * panelWidth);
    }
    auto start = [&](size_t s) {
        auto& p = pending[s % 2];
        const size_t width = std::min(panelWidth, N - s * panelWidth);
        const int aRoot = static_cast<int>(s % dims[1]);
        const int bRoot = static_cast<int>(s % dims[0]);
        double* a = coords[1] == aRoot ? panels[s].a.data() : p.aBuffer.data();
        double* b = coords[0] == bRoot ? panels[s].b.data() : p.bBuffer.data();
        p.a = a;
        p.b = b;
        MPI_Ibcast(a, static_cast<int>(rows.size * width), MPI_DOUBLE,
                   aRoot, rowComm, &p.requests[0]);
        MPI_Ibcast(b, static_cast<int>(cols.size * width), MPI_DOUBLE,
                   bRoot, colComm, &p.requests[1]);
    };
    start(0);
    for (size_t s = 0; s < panels.size(); ++s) {
        auto& p = pending[s % 2];
        MPI_Waitall(2, p.requests, MPI_STATUSES_IGNORE);
        // Double buffering allows the next broadcasts to overlap this multiply.
        if (s + 1 < panels.size()) start(s + 1);
        multiplyPanel(p.a, p.b, c, rows.size, cols.size,
                      std::min(panelWidth, N - s * panelWidth));
    }
}

bool validateResult(const std::vector<double>& c, size_t N, Range rows, Range cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N;
            const size_t j = pj % N;
            if (i < rows.begin || i >= rows.begin + rows.size ||
                j < cols.begin || j >= cols.begin + cols.size) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = c[(i - rows.begin) * cols.size + j - cols.begin];
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

// Gather only for -r. Chunk messages to avoid MPI's int count limit, and unpack
// one rank at a time so the root needs only one additional local-sized buffer.
void printDistributedResults(const std::vector<double>& c, size_t N,
                             MPI_Comm grid, int rank, int processes, const int* dims) {
    if (rank != 0) {
        for (size_t offset = 0; offset < c.size();) {
            const int count = static_cast<int>(std::min(c.size() - offset,
                                                       static_cast<size_t>(INT_MAX)));
            MPI_Send(c.data() + offset, count, MPI_DOUBLE, 0, 0, grid);
            offset += count;
        }
        return;
    }
    std::vector<double> result(N * N), received;
    for (int source = 0; source < processes; ++source) {
        int coords[2];
        MPI_Cart_coords(grid, source, 2, coords);
        const Range rows = partition(N, coords[0], dims[0]);
        const Range cols = partition(N, coords[1], dims[1]);
        const double* block = c.data();
        if (source != 0) {
            received.resize(rows.size * cols.size);
            for (size_t offset = 0; offset < received.size();) {
                const int count = static_cast<int>(std::min(received.size() - offset,
                                                           static_cast<size_t>(INT_MAX)));
                MPI_Recv(received.data() + offset, count, MPI_DOUBLE, source, 0,
                         grid, MPI_STATUS_IGNORE);
                offset += count;
            }
            block = received.data();
        }
        if (cols.size != 0)
            for (size_t i = 0; i < rows.size; ++i)
                std::copy_n(block + i * cols.size, cols.size,
                            result.data() + (rows.begin + i) * N + cols.begin);
    }
    print_results(result, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int processes) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(text, &end, 10);
            if (errno || text == end || *end || *text == '-' || value == 0 ||
                value > INT_MAX || value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", text);
                return 1;
            }
            N = static_cast<size_t>(value);
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
    int periods[2] = {0, 0};
    MPI_Dims_create(processes, 2, dims);
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
    MPI_Comm_split(grid, coords[1], coords[0], &colComm);
    const Range rows = partition(N, coords[0], dims[0]);
    const Range cols = partition(N, coords[1], dims[1]);
    const size_t maxExtent = std::max((N + dims[0] - 1) / dims[0],
                                      (N + dims[1] - 1) / dims[1]);
    const size_t panelWidth = std::min(size_t{128}, static_cast<size_t>(INT_MAX) / maxExtent);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d grid)\n", processes, dims[0], dims[1]);
        printf("Initializing matrices...\n");
    }
    auto panels = initPanels(N, panelWidth, rows, cols, coords, dims);
    std::vector<double> c(rows.size * cols.size, 0.0);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    matrixMultiply(panels, c, N, panelWidth, rows, cols, coords, dims, rowComm, colComm);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) printDistributedResults(c, N, grid, rank, processes, dims);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(c, N, rows, cols);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, grid);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
