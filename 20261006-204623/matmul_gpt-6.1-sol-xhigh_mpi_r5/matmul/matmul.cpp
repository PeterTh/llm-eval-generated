#include <mpi.h>

#include <algorithm>
#include <array>
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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

namespace {

constexpr size_t vectorWidth = 8;
constexpr size_t maxPanelWidth = 256;

size_t ceilDivide(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

// Balanced contiguous partitions, including empty blocks when N is small.
size_t blockSize(size_t N, int parts, int coordinate) {
    return N / parts + (static_cast<size_t>(coordinate) < N % parts);
}

size_t blockStart(size_t N, int parts, int coordinate) {
    return (N / parts) * coordinate +
           std::min(N % parts, static_cast<size_t>(coordinate));
}

struct ProcessGrid {
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm row = MPI_COMM_NULL;
    MPI_Comm column = MPI_COMM_NULL;
    int dimensions[2] = {0, 0};
    int coordinates[2] = {0, 0};
    int rank = 0;
    int processes = 0;

    ProcessGrid() {
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dimensions);
        int periodic[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periodic, 0, &cart);
        MPI_Comm_rank(cart, &rank);
        MPI_Cart_coords(cart, rank, 2, coordinates);
        int keepRow[2] = {0, 1};
        int keepColumn[2] = {1, 0};
        MPI_Cart_sub(cart, keepRow, &row);
        MPI_Cart_sub(cart, keepColumn, &column);
    }

    ~ProcessGrid() {
        MPI_Comm_free(&row);
        MPI_Comm_free(&column);
        MPI_Comm_free(&cart);
    }
};

// Each rank holds one C rectangle and cyclically owned A/B panels. B is
// packed into eight-column strips so the inner kernel streams through L1.
struct LocalMatrices {
    size_t rows;
    size_t columns;
    size_t firstRow;
    size_t firstColumn;
    size_t stride;
    size_t panelWidth;
    size_t panels;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;
    std::array<std::vector<double>, 2> receiveA;
    std::array<std::vector<double>, 2> receiveB;

    LocalMatrices(size_t N, const ProcessGrid& grid)
        : rows(blockSize(N, grid.dimensions[0], grid.coordinates[0])),
          columns(blockSize(N, grid.dimensions[1], grid.coordinates[1])),
          firstRow(blockStart(N, grid.dimensions[0], grid.coordinates[0])),
          firstColumn(blockStart(N, grid.dimensions[1], grid.coordinates[1])),
          stride(ceilDivide(columns, vectorWidth) * vectorWidth),
          panelWidth(1), panels(0) {
        if (N != 0) {
            // Enough panels to balance their ownership, and MPI counts that
            // remain representable even for matrices larger than INT_MAX.
            const size_t largestBlock = ceilDivide(N, std::min(grid.dimensions[0],
                                                              grid.dimensions[1]));
            const size_t largestStride = ceilDivide(largestBlock, vectorWidth) * vectorWidth;
            panelWidth = std::min({maxPanelWidth,
                                  ceilDivide(N, std::max(grid.dimensions[0], grid.dimensions[1])),
                                  static_cast<size_t>(INT_MAX) / largestStride});
            panels = ceilDivide(N, panelWidth);
        }

        size_t ownedColumns = 0;
        size_t ownedRows = 0;
        for (size_t panel = 0; panel < panels; ++panel) {
            const size_t width = std::min(panelWidth, N - panel * panelWidth);
            if (panel % grid.dimensions[1] == static_cast<size_t>(grid.coordinates[1])) {
                ownedColumns += width;
            }
            if (panel % grid.dimensions[0] == static_cast<size_t>(grid.coordinates[0])) {
                ownedRows += width;
            }
        }
        A.resize(rows * ownedColumns);
        B.resize(stride * ownedRows);
        C.resize(rows * stride);
        for (size_t slot = 0; slot < 2; ++slot) {
            receiveA[slot].resize(rows * panelWidth);
            receiveB[slot].resize(stride * panelWidth);
        }

        for (size_t panel = 0; panel < panels; ++panel) {
            const size_t firstK = panel * panelWidth;
            const size_t width = std::min(panelWidth, N - firstK);
            if (rows != 0 && panel % grid.dimensions[1] == static_cast<size_t>(grid.coordinates[1])) {
                double* values = A.data() + (panel / grid.dimensions[1]) * panelWidth * rows;
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t k = 0; k < width; ++k) {
                        values[i * width + k] = getPseudoRndValue(N, firstRow + i, firstK + k);
                    }
                }
            }
            if (stride != 0 && panel % grid.dimensions[0] == static_cast<size_t>(grid.coordinates[0])) {
                double* values = B.data() + (panel / grid.dimensions[0]) * panelWidth * stride;
                for (size_t j = 0; j < columns; j += vectorWidth) {
                    for (size_t k = 0; k < width; ++k) {
                        for (size_t lane = 0; lane < std::min(vectorWidth, columns - j); ++lane) {
                            values[j * width + k * vectorWidth + lane] =
                                getPseudoRndValue(N, firstK + k, firstColumn + j + lane);
                        }
                    }
                }
            }
        }
    }
};

// Independent columns vectorize without reassociating the k reduction.
// A 4x8 microtile provides independent accumulators and reuses each B load.
template <size_t Rows>
void multiplyRows(const double* A, const double* B, double* C,
                  size_t width, size_t stride) {
    for (size_t j = 0; j < stride; j += vectorWidth) {
#if defined(__AVX__)
        __m256d sums[Rows][2];
        for (size_t i = 0; i < Rows; ++i) {
            sums[i][0] = _mm256_loadu_pd(C + i * stride + j);
            sums[i][1] = _mm256_loadu_pd(C + i * stride + j + 4);
        }
        const double* packedB = B + j * width;
        for (size_t k = 0; k < width; ++k) {
            const __m256d b0 = _mm256_loadu_pd(packedB + k * vectorWidth);
            const __m256d b1 = _mm256_loadu_pd(packedB + k * vectorWidth + 4);
            for (size_t i = 0; i < Rows; ++i) {
                const __m256d a = _mm256_broadcast_sd(A + i * width + k);
                sums[i][0] = _mm256_add_pd(sums[i][0], _mm256_mul_pd(a, b0));
                sums[i][1] = _mm256_add_pd(sums[i][1], _mm256_mul_pd(a, b1));
            }
        }
        for (size_t i = 0; i < Rows; ++i) {
            _mm256_storeu_pd(C + i * stride + j, sums[i][0]);
            _mm256_storeu_pd(C + i * stride + j + 4, sums[i][1]);
        }
#else
        alignas(64) double sums[Rows][vectorWidth];
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t lane = 0; lane < vectorWidth; ++lane) {
                sums[i][lane] = C[i * stride + j + lane];
            }
        }
        const double* packedB = B + j * width;
        for (size_t k = 0; k < width; ++k) {
            for (size_t i = 0; i < Rows; ++i) {
                const double a = A[i * width + k];
                for (size_t lane = 0; lane < vectorWidth; ++lane) {
                    sums[i][lane] += a * packedB[k * vectorWidth + lane];
                }
            }
        }
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t lane = 0; lane < vectorWidth; ++lane) {
                C[i * stride + j + lane] = sums[i][lane];
            }
        }
#endif
    }
}

void multiplyPanel(const double* A, const double* B, LocalMatrices& matrices, size_t width,
                   MPI_Request* progressRequests) {
    size_t i = 0;
    for (; i + 4 <= matrices.rows; i += 4) {
        // Drive the next broadcasts on MPI implementations without a
        // background progress thread, amortizing the polling overhead.
        if (progressRequests != nullptr && i % 64 == 0) {
            int completed = 0;
            MPI_Testall(2, progressRequests, &completed, MPI_STATUSES_IGNORE);
            if (completed) {
                progressRequests = nullptr;
            }
        }
        multiplyRows<4>(A + i * width, B, matrices.C.data() + i * matrices.stride,
                        width, matrices.stride);
    }
    for (; i < matrices.rows; ++i) {
        multiplyRows<1>(A + i * width, B, matrices.C.data() + i * matrices.stride,
                        width, matrices.stride);
    }
}

void matrixMultiply(LocalMatrices& matrices, size_t N, const ProcessGrid& grid) {
    if (matrices.panels == 0) {
        return;
    }
    struct PendingPanel {
        double* A = nullptr;
        double* B = nullptr;
        size_t width = 0;
        MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    } pending[2];

    auto postPanel = [&](size_t panel, size_t slot) {
        auto& current = pending[slot];
        current.width = std::min(matrices.panelWidth, N - panel * matrices.panelWidth);
        const int ownerA = static_cast<int>(panel % grid.dimensions[1]);
        const int ownerB = static_cast<int>(panel % grid.dimensions[0]);
        current.A = matrices.receiveA[slot].data();
        current.B = matrices.receiveB[slot].data();
        // Broadcast directly from owned panels, avoiding packing/copying in
        // the timed section. Subcommunicator ranks equal their coordinates.
        if (ownerA == grid.coordinates[1] && matrices.rows != 0) {
            current.A = matrices.A.data() +
                        (panel / grid.dimensions[1]) * matrices.panelWidth * matrices.rows;
        }
        if (ownerB == grid.coordinates[0] && matrices.stride != 0) {
            current.B = matrices.B.data() +
                        (panel / grid.dimensions[0]) * matrices.panelWidth * matrices.stride;
        }
        MPI_Ibcast(current.A, static_cast<int>(matrices.rows * current.width),
                   MPI_DOUBLE, ownerA, grid.row, &current.requests[0]);
        MPI_Ibcast(current.B, static_cast<int>(matrices.stride * current.width),
                   MPI_DOUBLE, ownerB, grid.column, &current.requests[1]);
    };

    postPanel(0, 0);
    for (size_t panel = 0; panel < matrices.panels; ++panel) {
        auto& current = pending[panel % 2];
        MPI_Waitall(2, current.requests, MPI_STATUSES_IGNORE);
        if (panel + 1 < matrices.panels) {
            postPanel(panel + 1, (panel + 1) % 2);
        }
        if (matrices.rows != 0 && matrices.columns != 0) {
            MPI_Request* progressRequests = panel + 1 < matrices.panels
                ? pending[(panel + 1) % 2].requests : nullptr;
            multiplyPanel(current.A, current.B, matrices, current.width, progressRequests);
        }
    }
}

// Only -r requires a full matrix. Derived datatypes place each rectangle
// directly into row-major C and avoid both int-sized element counts and a
// second root-side receive buffer.
std::vector<double> gatherResult(const LocalMatrices& matrices, size_t N,
                                 const ProcessGrid& grid) {
    std::vector<double> result;
    if (grid.rank == 0) {
        result.resize(N * N);
        for (size_t i = 0; i < matrices.rows; ++i) {
            std::copy_n(matrices.C.data() + i * matrices.stride, matrices.columns,
                        result.data() + (matrices.firstRow + i) * N + matrices.firstColumn);
        }
        for (int source = 1; source < grid.processes; ++source) {
            int coordinates[2];
            MPI_Cart_coords(grid.cart, source, 2, coordinates);
            const size_t rows = blockSize(N, grid.dimensions[0], coordinates[0]);
            const size_t columns = blockSize(N, grid.dimensions[1], coordinates[1]);
            if (rows == 0 || columns == 0) {
                continue;
            }
            MPI_Datatype rectangle;
            MPI_Type_vector(static_cast<int>(rows), static_cast<int>(columns),
                            static_cast<int>(N), MPI_DOUBLE, &rectangle);
            MPI_Type_commit(&rectangle);
            const size_t offset = blockStart(N, grid.dimensions[0], coordinates[0]) * N +
                                  blockStart(N, grid.dimensions[1], coordinates[1]);
            MPI_Recv(result.data() + offset, 1, rectangle, source, 0, grid.cart, MPI_STATUS_IGNORE);
            MPI_Type_free(&rectangle);
        }
    } else if (matrices.rows != 0 && matrices.columns != 0) {
        MPI_Datatype rectangle;
        MPI_Type_vector(static_cast<int>(matrices.rows), static_cast<int>(matrices.columns),
                        static_cast<int>(matrices.stride), MPI_DOUBLE, &rectangle);
        MPI_Type_commit(&rectangle);
        MPI_Send(matrices.C.data(), 1, rectangle, 0, 0, grid.cart);
        MPI_Type_free(&rectangle);
    }
    return result;
}

bool validateResult(const LocalMatrices& matrices, size_t N, const ProcessGrid& grid) {
    if (N == 0) {
        return true;
    }
    std::array<double, 25> local{};
    std::array<double, 25> actual{};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N;
            const size_t j = pj % N;
            if (i >= matrices.firstRow && i - matrices.firstRow < matrices.rows &&
                j >= matrices.firstColumn && j - matrices.firstColumn < matrices.columns) {
                local[pi * 5 + pj] = matrices.C[(i - matrices.firstRow) * matrices.stride +
                                               j - matrices.firstColumn];
            }
        }
    }
    MPI_Reduce(local.data(), actual.data(), 25, MPI_DOUBLE, MPI_SUM, 0, grid.cart);
    if (grid.rank == 0) {
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = pi % N;
                const size_t j = pj % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                }
                const double value = actual[pi * 5 + pj];
                const double relError = std::abs((value - expected) / (expected + 1e-10));
                if (!std::isfinite(value) || relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, value, relError);
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse once and broadcast so every rank takes the same control path.
    int argumentStatus = 0;
    for (int i = 1; rank == 0 && i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(text, &end, 10);
            const size_t maxElements = std::vector<double>().max_size();
            if (text[0] == '-' || text[0] == '\0' || *end != '\0' || errno == ERANGE ||
                value > std::numeric_limits<size_t>::max() || value > INT_MAX - vectorWidth ||
                (value != 0 && value > maxElements / value)) {
                fprintf(stderr, "Invalid matrix size: %s\n", text);
                argumentStatus = 1;
                break;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            argumentStatus = 2;
            break;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            argumentStatus = 1;
            break;
        }
    }
    MPI_Bcast(&argumentStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (argumentStatus != 0) {
        return argumentStatus == 2 ? 0 : 1;
    }
    unsigned long long options[3] = {static_cast<unsigned long long>(N), validate, printResults};
    MPI_Bcast(options, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(options[0]);
    validate = options[1] != 0;
    printResults = options[2] != 0;

    ProcessGrid grid;
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    LocalMatrices matrices(N, grid);
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(grid.cart);
    const double start = MPI_Wtime();
    matrixMultiply(matrices, N, grid);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid.cart);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        const double size = static_cast<double>(N);
        const double gflops = duration > 0.0 ? (2.0 * size * size * size) / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        const auto C = gatherResult(matrices, N, grid);
        if (rank == 0) {
            print_results(C, "MatrixC");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        int valid = validateResult(matrices, N, grid) ? 1 : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, grid.cart);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
