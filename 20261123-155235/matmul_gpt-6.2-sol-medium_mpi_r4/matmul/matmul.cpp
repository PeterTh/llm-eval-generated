#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr size_t columnBlock = 8;
constexpr size_t maxPanelDepth = 128;

#if defined(__AVX512F__)
constexpr size_t vectorWidth = 8;
#elif defined(__AVX__)
constexpr size_t vectorWidth = 4;
#else
constexpr size_t vectorWidth = 2;
#endif
constexpr size_t rowBlock = vectorWidth >= 4 ? 4 : 2;
constexpr size_t vectorsPerBlock = columnBlock / vectorWidth;
using Vector = double __attribute__((vector_size(vectorWidth * sizeof(double))));

#if defined(__GNUC__) && !defined(__clang__) && defined(FP_FAST_FMA)
constexpr bool fuseOddTail = true;
#else
constexpr bool fuseOddTail = false;
#endif

struct Range {
    size_t begin;
    size_t count;
};

Range partition(size_t n, int parts, int index) {
    const size_t base = n / parts;
    const size_t extra = n % parts;
    const size_t part = index;
    return {part * base + std::min(part, extra), base + (part < extra)};
}

size_t paddedColumns(size_t n) {
    return (n + columnBlock - 1) / columnBlock * columnBlock;
}

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

inline Vector loadVector(const double* data) {
    Vector value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

inline void storeVector(double* data, Vector value) {
    std::memcpy(data, &value, sizeof(value));
}

// Each accumulator visits k in ascending order, with SIMD across output columns.
template <size_t... I>
void multiplyBlock(const double* __restrict__ a, const double* __restrict__ b,
                   double* __restrict__ c, size_t depth, size_t stride, size_t terms,
                   std::index_sequence<I...>) {
    // Constant accumulator indices let the compiler keep the tile in registers.
    const auto column = [](size_t i) { return i % vectorsPerBlock * vectorWidth; };
    const auto row = [](size_t i) { return i / vectorsPerBlock; };
    Vector sums[] = {loadVector(c + row(I) * stride + column(I))...};

    for (size_t k = 0; k < terms; ++k) {
        const double* bRow = b + k * columnBlock;
        ((sums[I] += a[row(I) * depth + k] * loadVector(bRow + column(I))), ...);
    }

    (storeVector(c + row(I) * stride + column(I), sums[I]), ...);
}

void multiplyPanel(const double* a, const double* b, double* c, size_t rows,
                   size_t stride, size_t depth, bool finalTerm, MPI_Request* next) {
    // Reuse packed B blocks across groups of rows while keeping C accesses local.
    constexpr size_t rowTile = 64;
    const size_t terms = depth - finalTerm;
    for (size_t firstRow = 0; firstRow < rows; firstRow += rowTile) {
        const size_t endRow = std::min(firstRow + rowTile, rows);
        for (size_t j = 0; j < stride; j += columnBlock) {
            const double* bBlock = b + j * depth;
            size_t i = firstRow;
            for (; i + rowBlock <= endRow; i += rowBlock) {
                multiplyBlock(a + i * depth, bBlock, c + i * stride + j, depth, stride, terms,
                              std::make_index_sequence<rowBlock * vectorsPerBlock>{});
            }
            for (; i < endRow; ++i) {
                multiplyBlock(a + i * depth, bBlock, c + i * stride + j, depth, stride, terms,
                              std::make_index_sequence<vectorsPerBlock>{});
            }
            // GCC can contract the original scalar odd tail on hardware with FMA.
            if (finalTerm) {
                const double* bRow = bBlock + terms * columnBlock;
                for (i = firstRow; i < endRow; ++i) {
                    const double aValue = a[i * depth + terms];
                    double* cRow = c + i * stride + j;
                    for (size_t lane = 0; lane < columnBlock; ++lane) {
                        cRow[lane] = std::fma(aValue, bRow[lane], cRow[lane]);
                    }
                }
            }
        }
        // Give MPI implementations that need application calls a chance to progress.
        if (next != nullptr && endRow < rows) {
            int complete;
            MPI_Testall(2, next, &complete, MPI_STATUSES_IGNORE);
            if (complete) {
                next = nullptr;
            }
        }
    }
}

struct Panels {
    // Owners are assigned cyclically within a row or column communicator.
    std::vector<double> owned;
    std::array<std::vector<double>, 2> incoming;
    size_t capacity;
    int coordinate;
    int peers;
    MPI_Comm comm;

    Panels(size_t count, size_t capacity, int coordinate, int peers, MPI_Comm comm)
        : capacity(capacity), coordinate(coordinate), peers(peers), comm(comm) {
        const size_t ownedCount = (count + peers - 1 - coordinate) / peers;
        owned.resize(ownedCount * capacity);
        if (peers > 1) {
            for (auto& buffer : incoming) {
                buffer.resize(capacity);
            }
        }
    }

    double* local(size_t panel) {
        return owned.data() + panel / peers * capacity;
    }

    double* broadcast(size_t panel, int slot, size_t count, MPI_Request& request) {
        const int owner = panel % peers;
        double* data = owner == coordinate ? local(panel) : incoming[slot].data();
        request = MPI_REQUEST_NULL;
        if (peers > 1) {
            MPI_Ibcast(data, static_cast<int>(count), MPI_DOUBLE, owner, comm, &request);
        }
        return data;
    }
};

// Only external result reporting needs the full, unpadded row-major matrix.
void printDistributedResults(const std::vector<double>& c, size_t n, Range rows,
                             Range cols, size_t stride, int rank, const int dims[2],
                             MPI_Comm comm) {
    if (rank != 0) {
        MPI_Datatype tile;
        MPI_Type_vector(rows.count, cols.count, stride, MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Send(c.data(), 1, tile, 0, 0, comm);
        MPI_Type_free(&tile);
        return;
    }

    std::vector<double> result(n * n);
    for (size_t i = 0; i < rows.count; ++i) {
        std::copy_n(c.data() + i * stride, cols.count, result.data() + i * n);
    }

    const int ranks = dims[0] * dims[1];
    std::vector<MPI_Request> requests(ranks - 1);
    for (int source = 1; source < ranks; ++source) {
        const Range sourceRows = partition(n, dims[0], source / dims[1]);
        const Range sourceCols = partition(n, dims[1], source % dims[1]);
        MPI_Datatype tile;
        MPI_Type_vector(sourceRows.count, sourceCols.count, n, MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Irecv(result.data() + sourceRows.begin * n + sourceCols.begin,
                  1, tile, source, 0, comm, &requests[source - 1]);
        MPI_Type_free(&tile);
    }
    if (!requests.empty()) {
        MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE);
    }
    print_results(result, "MatrixC");
}

bool validateResult(const std::vector<double>& c, size_t n, Range rows,
                    Range cols, size_t stride, int rank, MPI_Comm comm) {
    constexpr size_t maxChecks = 5;
    const size_t checks = std::min(maxChecks, n);
    std::array<double, maxChecks * maxChecks> local{};
    std::array<double, maxChecks * maxChecks> actual{};

    for (size_t i = rows.begin; i < std::min(rows.begin + rows.count, checks); ++i) {
        for (size_t j = cols.begin; j < std::min(cols.begin + cols.count, checks); ++j) {
            local[i * checks + j] = c[(i - rows.begin) * stride + j - cols.begin];
        }
    }
    MPI_Reduce(local.data(), actual.data(), checks * checks, MPI_DOUBLE, MPI_SUM, 0, comm);

    int valid = 1;
    if (rank == 0) {
        for (size_t i = 0; i < checks && valid; ++i) {
            for (size_t j = 0; j < checks; ++j) {
                double expected = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
                }
                const double value = actual[i * checks + j];
                const double relError = std::abs((value - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, value, relError);
                    valid = 0;
                    break;
                }
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
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

struct Options {
    int n = 512;
    int validate = 0;
    int printResults = 0;
    int exitStatus = -1;
};

Options readOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            const long n = strtol(arg, &end, 10);
            if (arg == end || *end != '\0' || n < 0 || n > INT_MAX - static_cast<long>(columnBlock)) {
                fprintf(stderr, "Invalid matrix size: %s\n", arg);
                options.exitStatus = 1;
                break;
            }
            options.n = static_cast<int>(n);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.printResults = 1;
        } else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (!help) {
                printf("Unknown option: %s\n", argv[i]);
            }
            printUsage(argv[0]);
            options.exitStatus = help ? 0 : 1;
            break;
        }
    }
    return options;
}

int runBenchmark(size_t n, bool validate, bool printResults, int rank,
                 const int dims[2], MPI_Comm comm) {
    const int row = rank / dims[1];
    const int col = rank % dims[1];
    const Range rows = partition(n, dims[0], row);
    const Range cols = partition(n, dims[1], col);
    const size_t stride = paddedColumns(cols.count);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    MPI_Comm rowComm, colComm;
    MPI_Comm_split(comm, row, col, &rowComm);
    MPI_Comm_split(comm, col, row, &colComm);

    const size_t maxPeers = std::max(dims[0], dims[1]);
    const size_t maxExtent = std::max(partition(n, dims[0], 0).count,
                                      paddedColumns(partition(n, dims[1], 0).count));
    // Spread ownership across the grid and keep MPI counts within the int limit.
    const size_t panelDepth = std::min({maxPanelDepth, (n + maxPeers - 1) / maxPeers,
                                        INT_MAX / std::max(size_t{1}, maxExtent)});
    const size_t panelCount = n == 0 ? 0 : 1 + (n - 1) / panelDepth;
    const auto depth = [=](size_t p) { return std::min(panelDepth, n - p * panelDepth); };

    std::vector<double> c(rows.count * stride);
    Panels a(panelCount, rows.count * panelDepth, col, dims[1], rowComm);
    Panels b(panelCount, stride * panelDepth, row, dims[0], colComm);

    for (size_t p = col; p < panelCount; p += dims[1]) {
        const size_t d = depth(p);
        double* panel = a.local(p);
        for (size_t i = 0; i < rows.count; ++i) {
            for (size_t k = 0; k < d; ++k) {
                panel[i * d + k] = getPseudoRndValue(n, rows.begin + i, p * panelDepth + k);
            }
        }
    }
    for (size_t p = row; p < panelCount; p += dims[0]) {
        const size_t d = depth(p);
        double* panel = b.local(p);
        // Pack each SIMD column block so successive k values are contiguous.
        for (size_t j = 0; j < cols.count; j += columnBlock) {
            const size_t width = std::min(columnBlock, cols.count - j);
            for (size_t k = 0; k < d; ++k) {
                for (size_t lane = 0; lane < width; ++lane) {
                    panel[j * d + k * columnBlock + lane] =
                        getPseudoRndValue(n, p * panelDepth + k, cols.begin + j + lane);
                }
            }
        }
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    struct Transfer {
        const double* a;
        const double* b;
        MPI_Request requests[2];
    };
    std::array<Transfer, 2> transfers;
    const auto post = [&](size_t p) {
        const int slot = p % transfers.size();
        auto& transfer = transfers[slot];
        transfer.a = a.broadcast(p, slot, rows.count * depth(p), transfer.requests[0]);
        transfer.b = b.broadcast(p, slot, stride * depth(p), transfer.requests[1]);
    };

    if (panelCount != 0) {
        post(0);
    }
    for (size_t p = 0; p < panelCount; ++p) {
        auto& current = transfers[p % transfers.size()];
        MPI_Waitall(2, current.requests, MPI_STATUSES_IGNORE);
        const bool last = p + 1 == panelCount;
        MPI_Request* next = nullptr;
        if (!last) {
            post(p + 1);
            if (maxPeers > 1) {
                next = transfers[(p + 1) % transfers.size()].requests;
            }
        }
        multiplyPanel(current.a, current.b, c.data(), rows.count, stride, depth(p),
                      fuseOddTail && n % 2 != 0 && last, next);
    }

    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? 2.0 * n * n * n / seconds / 1e9 : 0.0);
    }

    if (printResults) {
        printDistributedResults(c, n, rows, cols, stride, rank, dims, comm);
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        return validateResult(c, n, rows, cols, stride, rank, comm) ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::array<int, 4> config{};
    if (rank == 0) {
        const Options options = readOptions(argc, argv);
        config = {options.n, options.validate, options.printResults, options.exitStatus};
    }
    MPI_Bcast(config.data(), config.size(), MPI_INT, 0, MPI_COMM_WORLD);
    const auto [n, validate, printResults, exitStatus] = config;

    int status = exitStatus;
    if (status < 0) {
        int dims[2] = {0, 0};
        MPI_Dims_create(ranks, 2, dims);
        for (int& dim : dims) {
            dim = std::min(dim, std::max(n, 1));
        }

        // Only use ranks that can own a nonempty tile.
        const int active = dims[0] * dims[1];
        MPI_Comm comm = MPI_COMM_WORLD;
        if (active < ranks) {
            MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
        }
        status = 0;
        if (rank < active) {
            try {
                status = runBenchmark(n, validate, printResults, rank, dims, comm);
            } catch (const std::exception& error) {
                fprintf(stderr, "Rank %d: %s\n", rank, error.what());
                MPI_Abort(MPI_COMM_WORLD, 1);
                status = 1;
            }
            if (active < ranks) {
                MPI_Comm_free(&comm);
            }
        }
    }

    MPI_Finalize();
    return status;
}
