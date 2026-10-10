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

#include "../common/results_output.hpp"

// Keep the original initialization, including its size_t arithmetic.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Partition {
    size_t begin;
    size_t size;
};

Partition partition(size_t n, int parts, int coordinate) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    return {base * coordinate + std::min(extra, static_cast<size_t>(coordinate)),
            base + (static_cast<size_t>(coordinate) < extra)};
}

constexpr size_t panelWidth = 128;

struct Panel {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<MPI_Request> requests;
    size_t width = 0;
};

// Chunk messages to support matrices larger than MPI's int count limit.
void broadcast(std::vector<double>& values, size_t count, int root,
               MPI_Comm communicator, std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        requests.emplace_back(MPI_REQUEST_NULL);
        MPI_Ibcast(values.data() + offset, chunk, MPI_DOUBLE, root, communicator,
                   &requests.back());
        offset += chunk;
    }
}

void startPanel(Panel& panel, size_t start, size_t n, Partition rows, Partition cols,
                const int* coords, const int* dims, MPI_Comm rowComm, MPI_Comm colComm) {
    panel.width = std::min(panelWidth, n - start);
    panel.requests.clear();
    const size_t stage = start / panelWidth;
    const int aOwner = static_cast<int>(stage % dims[1]);
    const int bOwner = static_cast<int>(stage % dims[0]);
    if (coords[1] == aOwner) {
        for (size_t i = 0; i < rows.size; ++i)
            for (size_t k = 0; k < panel.width; ++k)
                panel.a[i * panel.width + k] = getPseudoRndValue(n, rows.begin + i, start + k);
    }
    if (coords[0] == bOwner) {
        for (size_t k = 0; k < panel.width; ++k)
            for (size_t j = 0; j < cols.size; ++j)
                panel.b[k * cols.size + j] = getPseudoRndValue(n, start + k, cols.begin + j);
    }
    // Cartesian subcommunicator ranks are obtained explicitly, rather than
    // assuming their rank numbering matches the grid coordinates.
    int aRoot, bRoot;
    MPI_Cart_rank(rowComm, &aOwner, &aRoot);
    MPI_Cart_rank(colComm, &bOwner, &bRoot);
    broadcast(panel.a, rows.size * panel.width, aRoot, rowComm, panel.requests);
    broadcast(panel.b, panel.width * cols.size, bRoot, colComm, panel.requests);
}

void waitPanel(Panel& panel) {
    // Individual waits also avoid an int limit on the number of requests.
    for (auto& request : panel.requests)
        MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void matrixMultiply(const Panel& panel, std::vector<double>& c, size_t rows, size_t cols) {
    // Cache-resident output tiles, with the contiguous column loop available
    // for SIMD. Each element still accumulates k in the original order.
    constexpr size_t rowTile = 32;
    constexpr size_t colTile = 64;
    for (size_t ii = 0; ii < rows; ii += rowTile) {
        const size_t iEnd = std::min(rows, ii + rowTile);
        for (size_t jj = 0; jj < cols; jj += colTile) {
            const size_t jEnd = std::min(cols, jj + colTile);
            size_t i = ii;
            for (; i + 4 <= iEnd; i += 4) {
                size_t j = jj;
                for (; j + 8 <= jEnd; j += 8) {
                    // Four rows share each B vector; keep the output in
                    // registers throughout this panel instead of loading
                    // and storing it at every k.
                    double c0[8], c1[8], c2[8], c3[8];
                    std::copy_n(c.data() + i * cols + j, 8, c0);
                    std::copy_n(c.data() + (i + 1) * cols + j, 8, c1);
                    std::copy_n(c.data() + (i + 2) * cols + j, 8, c2);
                    std::copy_n(c.data() + (i + 3) * cols + j, 8, c3);
                    for (size_t k = 0; k < panel.width; ++k) {
                        const double a0 = panel.a[i * panel.width + k];
                        const double a1 = panel.a[(i + 1) * panel.width + k];
                        const double a2 = panel.a[(i + 2) * panel.width + k];
                        const double a3 = panel.a[(i + 3) * panel.width + k];
                        const double* b = panel.b.data() + k * cols + j;
                        for (size_t t = 0; t < 8; ++t) {
                            c0[t] += a0 * b[t];
                            c1[t] += a1 * b[t];
                            c2[t] += a2 * b[t];
                            c3[t] += a3 * b[t];
                        }
                    }
                    std::copy_n(c0, 8, c.data() + i * cols + j);
                    std::copy_n(c1, 8, c.data() + (i + 1) * cols + j);
                    std::copy_n(c2, 8, c.data() + (i + 2) * cols + j);
                    std::copy_n(c3, 8, c.data() + (i + 3) * cols + j);
                }
                for (size_t k = 0; k < panel.width; ++k)
                    for (size_t r = i; r < i + 4; ++r)
                        for (size_t t = j; t < jEnd; ++t)
                            c[r * cols + t] += panel.a[r * panel.width + k] * panel.b[k * cols + t];
            }
            for (size_t k = 0; k < panel.width; ++k) {
                const double* b = panel.b.data() + k * cols;
                for (size_t r = i; r < iEnd; ++r) {
                    const double a = panel.a[r * panel.width + k];
                    double* out = c.data() + r * cols;
                    for (size_t j = jj; j < jEnd; ++j)
                        out[j] += a * b[j];
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t n, Partition rows, Partition cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % n;
            const size_t j = pj % n;
            if (i < rows.begin || i >= rows.begin + rows.size ||
                j < cols.begin || j >= cols.begin + cols.size)
                continue;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
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

void printDistributedResult(const std::vector<double>& local, size_t n, int rank,
                            int ranks, const int* dims, MPI_Comm grid) {
    // Preserve the exact row-major result statistics and hash. Only -r needs
    // the full matrix, and only rank zero allocates it.
    if (rank == 0) {
        std::vector<double> full(n * n);
        std::vector<double> received;
        for (int source = 0; source < ranks; ++source) {
            int coordinates[2];
            MPI_Cart_coords(grid, source, 2, coordinates);
            const auto rows = partition(n, dims[0], coordinates[0]);
            const auto cols = partition(n, dims[1], coordinates[1]);
            const size_t count = rows.size * cols.size;
            const double* block = local.data();
            if (source != 0) {
                received.resize(count);
                for (size_t offset = 0; offset < count;) {
                    const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, source, 0, grid, MPI_STATUS_IGNORE);
                    offset += chunk;
                }
                block = received.data();
            }
            if (cols.size != 0)
                for (size_t i = 0; i < rows.size; ++i)
                    std::copy_n(block + i * cols.size, cols.size,
                                full.data() + (rows.begin + i) * n + cols.begin);
        }
        print_results(full, "MatrixC");
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const int chunk = static_cast<int>(std::min(local.size() - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 0, grid);
            offset += chunk;
        }
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
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max() ||
                parsed > std::numeric_limits<size_t>::max() / parsed ||
                parsed * parsed > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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
    const int periods[2] = {0, 0};
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    const int rowRemain[2] = {0, 1};
    const int colRemain[2] = {1, 0};
    MPI_Cart_sub(grid, rowRemain, &rowComm);
    MPI_Cart_sub(grid, colRemain, &colComm);
    const auto rows = partition(n, dims[0], coords[0]);
    const auto cols = partition(n, dims[1], coords[1]);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> c(rows.size * cols.size, 0.0);
    Panel panels[2];
    for (auto& panel : panels) {
        panel.a.resize(rows.size * std::min(n, panelWidth));
        panel.b.resize(std::min(n, panelWidth) * cols.size);
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    startPanel(panels[0], 0, n, rows, cols, coords, dims, rowComm, colComm);
    waitPanel(panels[0]);
    int current = 0;
    for (size_t k = 0; k < n;) {
        const size_t next = k + panels[current].width;
        if (next < n)
            startPanel(panels[1 - current], next, n, rows, cols, coords, dims, rowComm, colComm);
        matrixMultiply(panels[current], c, rows.size, cols.size);
        if (next < n) waitPanel(panels[1 - current]);
        current = 1 - current;
        k = next;
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        const double gflops = (2.0 * n * n * n) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) printDistributedResult(c, n, rank, ranks, dims, grid);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(c, n, rows, cols) ? 1 : 0;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
