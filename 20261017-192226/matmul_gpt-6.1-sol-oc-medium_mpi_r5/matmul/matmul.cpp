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

// Balanced contiguous ownership, including empty blocks when there are more ranks
// than matrix rows or columns.
size_t blockStart(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(N % parts, static_cast<size_t>(coordinate));
}

struct Panel {
    std::vector<double> a, b;
};

// k always advances in the original order. Vectorization is across independent
// output columns, so blocking does not reorder any dot product's additions.
void matrixMultiply(const double* a, const double* b, std::vector<double>& c,
                    size_t rows, size_t cols, size_t width) {
    constexpr size_t rowTile = 32;
    constexpr size_t colTile = 256;
    for (size_t ii = 0; ii < rows; ii += rowTile) {
        for (size_t jj = 0; jj < cols; jj += colTile) {
            const size_t iEnd = std::min(ii + rowTile, rows);
            const size_t jEnd = std::min(jj + colTile, cols);
            for (size_t k = 0; k < width; ++k) {
                const double* bp = b + k * cols;
                for (size_t i = ii; i < iEnd; ++i) {
                    const double av = a[i * width + k];
                    double* cp = c.data() + i * cols;
                    for (size_t j = jj; j < jEnd; ++j) {
                        cp[j] += av * bp[j];
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t N, size_t rowStart,
                    size_t colStart, size_t rows, size_t cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N, j = pj % N;
            if (i < rowStart || i >= rowStart + rows ||
                j < colStart || j >= colStart + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = c[(i - rowStart) * cols + j - colStart];
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

// Only -r needs a full matrix. Transfer packed rank blocks in bounded messages
// and unpack into global row-major order to retain the original statistics/hash.
void printDistributedResults(const std::vector<double>& local, size_t N,
                             int rank, int size, const int* dims, MPI_Comm grid) {
    constexpr size_t chunk = 1024 * 1024;
    if (rank != 0) {
        for (size_t offset = 0; offset < local.size(); offset += chunk) {
            const int count = static_cast<int>(std::min(chunk, local.size() - offset));
            MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, grid);
        }
        return;
    }
    std::vector<double> result(N * N), received;
    for (int source = 0; source < size; ++source) {
        int coords[2];
        MPI_Cart_coords(grid, source, 2, coords);
        const size_t row = blockStart(N, coords[0], dims[0]);
        const size_t col = blockStart(N, coords[1], dims[1]);
        const size_t rows = blockStart(N, coords[0] + 1, dims[0]) - row;
        const size_t cols = blockStart(N, coords[1] + 1, dims[1]) - col;
        const double* data = local.data();
        if (source != 0) {
            received.resize(rows * cols);
            for (size_t offset = 0; offset < received.size(); offset += chunk) {
                const int count = static_cast<int>(std::min(chunk, received.size() - offset));
                MPI_Recv(received.data() + offset, count, MPI_DOUBLE, source, 0,
                         grid, MPI_STATUS_IGNORE);
            }
            data = received.data();
        }
        if (cols != 0) {
            for (size_t i = 0; i < rows; ++i) {
                std::copy_n(data + i * cols, cols, result.data() + (row + i) * N + col);
            }
        }
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

int run(int argc, char** argv, int rank, int size) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
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

    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    MPI_Dims_create(size, 2, dims);
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    MPI_Cart_coords(grid, rank, 2, coords);
    MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
    MPI_Comm_split(grid, coords[1], coords[0], &colComm);
    const size_t rowStart = blockStart(N, coords[0], dims[0]);
    const size_t colStart = blockStart(N, coords[1], dims[1]);
    const size_t rows = blockStart(N, coords[0] + 1, dims[0]) - rowStart;
    const size_t cols = blockStart(N, coords[1] + 1, dims[1]) - colStart;
    // The common width keeps every broadcast count within MPI's int limit.
    const size_t panelWidth = std::min(size_t(128), size_t(INT_MAX) / N);
    const size_t stages = (N + panelWidth - 1) / panelWidth;
    std::vector<Panel> panels(stages);
    std::vector<double> c(rows * cols, 0.0);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    // Each input panel is initialized only by its owner, without serial scatter
    // or replicated full input matrices on any rank.
    for (size_t s = 0; s < stages; ++s) {
        const size_t kStart = s * panelWidth;
        const size_t width = std::min(panelWidth, N - kStart);
        if (s % dims[1] == static_cast<size_t>(coords[1])) {
            panels[s].a.resize(rows * width);
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < width; ++k)
                    panels[s].a[i * width + k] = getPseudoRndValue(N, rowStart + i, kStart + k);
        }
        if (s % dims[0] == static_cast<size_t>(coords[0])) {
            panels[s].b.resize(width * cols);
            for (size_t k = 0; k < width; ++k)
                for (size_t j = 0; j < cols; ++j)
                    panels[s].b[k * cols + j] = getPseudoRndValue(N, kStart + k, colStart + j);
        }
    }
    Panel buffers[2];
    for (auto& buffer : buffers) {
        buffer.a.resize(rows * panelWidth);
        buffer.b.resize(cols * panelWidth);
    }
    MPI_Request requests[2][2];
    auto broadcast = [&](size_t s) {
        const size_t width = std::min(panelWidth, N - s * panelWidth);
        Panel& buffer = buffers[s % 2];
        if (s % dims[1] == static_cast<size_t>(coords[1]))
            std::copy(panels[s].a.begin(), panels[s].a.end(), buffer.a.begin());
        if (s % dims[0] == static_cast<size_t>(coords[0]))
            std::copy(panels[s].b.begin(), panels[s].b.end(), buffer.b.begin());
        MPI_Ibcast(buffer.a.data(), static_cast<int>(rows * width), MPI_DOUBLE,
                   static_cast<int>(s % dims[1]), rowComm, &requests[s % 2][0]);
        MPI_Ibcast(buffer.b.data(), static_cast<int>(cols * width), MPI_DOUBLE,
                   static_cast<int>(s % dims[0]), colComm, &requests[s % 2][1]);
    };

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    broadcast(0);
    for (size_t s = 0; s < stages; ++s) {
        MPI_Waitall(2, requests[s % 2], MPI_STATUSES_IGNORE);
        if (s + 1 < stages) broadcast(s + 1);
        const Panel& buffer = buffers[s % 2];
        matrixMultiply(buffer.a.data(), buffer.b.data(), c, rows, cols,
                       std::min(panelWidth, N - s * panelWidth));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / duration / 1e9);
    }
    if (printResults) printDistributedResults(c, N, rank, size, dims, grid);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int localValid = validateResult(c, N, rowStart, colStart, rows, cols);
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int result = 1;
    try {
        result = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
