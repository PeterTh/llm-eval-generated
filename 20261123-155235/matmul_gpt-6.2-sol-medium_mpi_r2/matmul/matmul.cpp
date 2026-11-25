#include <mpi.h>

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int tileRows = 4;
constexpr int tileCols = 8;
constexpr int maxPanelDepth = 128;

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

struct Block {
    int start;
    int size;
};

Block partition(int n, int parts, int index) {
    const int remainder = n % parts;
    return {index * (n / parts) + std::min(index, remainder),
            n / parts + (index < remainder)};
}

size_t tileCount(int count, int width) {
    return (static_cast<size_t>(count) + width - 1) / width;
}

// Store each group of rows or columns contiguously at every step of the dot product.
template <int Width, bool IsA>
std::vector<double> packPanel(int n, Block outer, int start, int depth) {
    std::vector<double> values(tileCount(outer.size, Width) * depth * Width);
    for (int i = 0; i < outer.size; ++i) {
        double* tile = values.data() + (static_cast<size_t>(i / Width) * depth) * Width;
        for (int k = 0; k < depth; ++k) {
            tile[k * Width + i % Width] = IsA
                ? getPseudoRndValue(n, outer.start + i, start + k)
                : getPseudoRndValue(n, start + k, outer.start + i);
        }
    }
    return values;
}

void multiplyTile(const double* a, const double* b, double* c, size_t stride,
                  int depth, bool fusedTail) {
    double sums[tileRows][tileCols];
    for (int i = 0; i < tileRows; ++i) {
        for (int j = 0; j < tileCols; ++j) {
            sums[i][j] = c[i * stride + j];
        }
    }

    for (int k = 0; k < depth; ++k) {
        for (int i = 0; i < tileRows; ++i) {
            for (int j = 0; j < tileCols; ++j) {
                sums[i][j] += a[k * tileRows + i] * b[k * tileCols + j];
            }
        }
    }

    if (fusedTail) {
        for (int i = 0; i < tileRows; ++i) {
            for (int j = 0; j < tileCols; ++j) {
                sums[i][j] = std::fma(a[depth * tileRows + i], b[depth * tileCols + j], sums[i][j]);
            }
        }
    }

    for (int i = 0; i < tileRows; ++i) {
        for (int j = 0; j < tileCols; ++j) {
            c[i * stride + j] = sums[i][j];
        }
    }
}

void multiplyPanel(const double* a, const double* b, double* c, size_t rowTiles,
                   size_t colTiles, int depth, bool fusedTail) {
    const size_t stride = colTiles * tileCols;
    // Reuse a strip of B across the local rows before moving to the next strip.
    constexpr size_t stripTiles = 16;
    for (size_t strip = 0; strip < colTiles; strip += stripTiles) {
        const size_t end = std::min(strip + stripTiles, colTiles);
        for (size_t i = 0; i < rowTiles; ++i) {
            const double* aTile = a + i * depth * tileRows;
            for (size_t j = strip; j < end; ++j) {
                multiplyTile(aTile, b + j * depth * tileCols,
                             c + i * tileRows * stride + j * tileCols,
                             stride, depth - fusedTail, fusedTail);
            }
        }
    }
}

struct Panel {
    std::vector<double> a;
    std::vector<double> b;
};

struct Broadcast {
    const double* a;
    const double* b;
    MPI_Request requests[2];
};

Broadcast broadcastPanel(Panel& owned, Panel& scratch, int depth, size_t rowTiles,
                         size_t colTiles, int row, int col, int rowRoot, int colRoot,
                         MPI_Comm rowComm, MPI_Comm colComm) {
    double* a = col == rowRoot ? owned.a.data() : scratch.a.data();
    double* b = row == colRoot ? owned.b.data() : scratch.b.data();
    Broadcast pending{a, b, {MPI_REQUEST_NULL, MPI_REQUEST_NULL}};
    if (rowComm != MPI_COMM_NULL) {
        MPI_Ibcast(a, static_cast<int>(rowTiles * tileRows * depth), MPI_DOUBLE,
                   rowRoot, rowComm, &pending.requests[0]);
    }
    if (colComm != MPI_COMM_NULL) {
        MPI_Ibcast(b, static_cast<int>(colTiles * tileCols * depth), MPI_DOUBLE,
                   colRoot, colComm, &pending.requests[1]);
    }
    return pending;
}

std::vector<double> gatherResult(const std::vector<double>& local, int n, int rank,
                                 const int* grid, Block rows, Block cols, size_t stride) {
    std::vector<double> result;
    if (rank == 0) {
        result.resize(static_cast<size_t>(n) * n);
        for (int i = 0; i < rows.size; ++i) {
            std::copy_n(local.data() + i * stride, cols.size,
                        result.data() + static_cast<size_t>(i) * n);
        }
    }

    // Write each block directly into row-major order for the original result printer.
    std::vector<MPI_Request> requests;
    const int firstSource = rank == 0 ? 1 : rank;
    const int endSource = rank == 0 ? grid[0] * grid[1] : rank + 1;
    for (int source = firstSource; source < endSource; ++source) {
        const Block sourceRows = partition(n, grid[0], source / grid[1]);
        const Block sourceCols = partition(n, grid[1], source % grid[1]);
        if (sourceRows.size > 0 && sourceCols.size > 0) {
            const size_t leading = rank == 0 ? n : stride;
            const int chunkRows = std::max(1, (INT_MAX / static_cast<int>(sizeof(double))) / sourceCols.size);
            for (int first = 0; first < sourceRows.size;) {
                const int count = std::min(chunkRows, sourceRows.size - first);
                MPI_Datatype blockType;
                MPI_Type_create_hvector(count, sourceCols.size, leading * sizeof(double),
                                        MPI_DOUBLE, &blockType);
                MPI_Type_commit(&blockType);
                if (rank == 0) {
                    double* destination = result.data() + static_cast<size_t>(sourceRows.start + first) * n
                                          + sourceCols.start;
                    requests.push_back(MPI_REQUEST_NULL);
                    MPI_Irecv(destination, 1, blockType, source, 0, MPI_COMM_WORLD, &requests.back());
                } else {
                    MPI_Send(local.data() + first * stride, 1, blockType, 0, 0, MPI_COMM_WORLD);
                }
                MPI_Type_free(&blockType);
                first += count;
            }
        }
    }
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    return result;
}

bool validateResult(const std::vector<double>& local, int n, int rank,
                    Block rows, Block cols, size_t stride) {
    if (n == 0) {
        return true;
    }

    double samples[5][5]{};
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const int i = pi % n;
            const int j = pj % n;
            if (i >= rows.start && i - rows.start < rows.size &&
                j >= cols.start && j - cols.start < cols.size) {
                samples[pi][pj] = local[(i - rows.start) * stride + j - cols.start];
            }
        }
    }

    double actual[5][5];
    MPI_Reduce(samples, actual, 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0) {
        for (int pi = 0; pi < 5 && valid; ++pi) {
            for (int pj = 0; pj < 5; ++pj) {
                const int i = pi % n;
                const int j = pj % n;
                double expected = 0.0;
                for (int k = 0; k < n; ++k) {
                    expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
                }
                const double error = std::abs((actual[pi][pj] - expected) / (expected + 1e-10));
                if (error > 1e-6) {
                    printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual[pi][pj], error);
                    valid = 0;
                    break;
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int parseOptions(int argc, char** argv, int& n, int& validate, int& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            const char* end = argument + strlen(argument);
            if (argument != end && *argument == '+') {
                ++argument;
            }
            const auto [ptr, error] = std::from_chars(argument, end, n);
            if (error != std::errc{} || ptr != end || n < 0) {
                printf("Invalid matrix size: %s\n", argv[i]);
                printUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    return -1;
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int options[] = {512, 0, 0, -1};
    if (rank == 0) {
        options[3] = parseOptions(argc, argv, options[0], options[1], options[2]);
    }
    MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD);
    const auto [n, validate, printResults, exitCode] = options;
    if (exitCode >= 0) {
        return exitCode;
    }

    int grid[2]{};
    MPI_Dims_create(ranks, 2, grid);
    const int row = rank / grid[1];
    const int col = rank % grid[1];
    const Block rows = partition(n, grid[0], row);
    const Block cols = partition(n, grid[1], col);
    const size_t rowTiles = tileCount(rows.size, tileRows);
    const size_t colTiles = tileCount(cols.size, tileCols);
    const size_t stride = colTiles * tileCols;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    const int gridWidth = std::max(grid[0], grid[1]);
    const size_t maxPanelStride = std::max(tileCount(partition(n, grid[0], 0).size, tileRows) * tileRows,
                                          tileCount(partition(n, grid[1], 0).size, tileCols) * tileCols);
    // Keep MPI-3 collective counts in range, including padding on the largest rank.
    if (maxPanelStride > INT_MAX) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size exceeds the MPI panel count limit.\n");
        }
        return 1;
    }
    const int panelWidth = std::min({maxPanelDepth, n == 0 ? 1 : 1 + (n - 1) / gridWidth,
                                     INT_MAX / static_cast<int>(std::max<size_t>(1, maxPanelStride))});
    const int panelCount = n == 0 ? 0 : 1 + (n - 1) / panelWidth;
    std::vector<Panel> panels(panelCount);
    for (int panel = 0; panel < panelCount; ++panel) {
        const int start = panel * panelWidth;
        const int depth = std::min(panelWidth, n - start);
        if (col == panel % grid[1]) {
            panels[panel].a = packPanel<tileRows, true>(n, rows, start, depth);
        }
        if (row == panel % grid[0]) {
            panels[panel].b = packPanel<tileCols, false>(n, cols, start, depth);
        }
    }

    std::vector<double> local(rowTiles * tileRows * stride);
    Panel scratch[2];
    for (Panel& buffer : scratch) {
        if (grid[1] > 1) {
            buffer.a.resize(rowTiles * tileRows * panelWidth);
        }
        if (grid[0] > 1) {
            buffer.b.resize(colTiles * tileCols * panelWidth);
        }
    }
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm colComm = MPI_COMM_NULL;
    if (grid[1] > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    }
    if (grid[0] > 1) {
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    // SUMMA broadcasts A across each process row and B down each process column.
    // Start the next broadcast before multiplying the current pair of panels.
    auto startPanel = [&](int panel) {
        return broadcastPanel(panels[panel], scratch[panel % 2],
                              std::min(panelWidth, n - panel * panelWidth),
                              rowTiles, colTiles, row, col,
                              panel % grid[1], panel % grid[0], rowComm, colComm);
    };
    Broadcast current{};
    if (panelCount > 0) {
        current = startPanel(0);
    }
    for (int panel = 0; panel < panelCount; ++panel) {
        MPI_Waitall(2, current.requests, MPI_STATUSES_IGNORE);
        Broadcast next{};
        if (panel + 1 < panelCount) {
            next = startPanel(panel + 1);
        }
        if (rows.size > 0 && cols.size > 0) {
            bool fusedTail = false;
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
            // GCC's original two-lane reduction contracts its scalar remainder.
            fusedTail = n % 2 != 0 && panel + 1 == panelCount;
#endif
            multiplyPanel(current.a, current.b, local.data(), rowTiles, colTiles,
                          std::min(panelWidth, n - panel * panelWidth), fusedTail);
        }
        current = next;
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rowComm != MPI_COMM_NULL) {
        MPI_Comm_free(&rowComm);
    }
    if (colComm != MPI_COMM_NULL) {
        MPI_Comm_free(&colComm);
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        const double gflops = duration > 0.0 ? (2.0 * n * n * n) / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        const auto result = gatherResult(local, n, rank, grid, rows, cols, stride);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(local, n, rank, rows, cols, stride);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int exitCode;
    try {
        exitCode = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exitCode;
}
