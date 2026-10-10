#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
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

// Preserve the original initialization, including unsigned integer arithmetic.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

namespace {

constexpr size_t columnTile = 8;
constexpr size_t rowTile = 4;
constexpr size_t rowBlock = 64;
constexpr size_t maxPanelWidth = 256;
constexpr size_t maxMessageCount = std::numeric_limits<int>::max();

struct Range {
    size_t begin;
    size_t count;
};

Range partition(size_t N, int parts, int coordinate) {
    const size_t base = N / static_cast<size_t>(parts);
    const size_t remainder = N % static_cast<size_t>(parts);
    const size_t index = static_cast<size_t>(coordinate);
    return {index * base + std::min(index, remainder), base + (index < remainder)};
}

struct ProcessGrid {
    int dimensions[2] = {0, 0};
    int row;
    int column;
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm columnComm = MPI_COMM_NULL;

    ProcessGrid(int rank, int processes) {
        MPI_Dims_create(processes, 2, dimensions);
        row = rank / dimensions[1];
        column = rank % dimensions[1];
        // Keys make communicator ranks equal to the panel owner coordinates.
        MPI_Comm_split(MPI_COMM_WORLD, row, column, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, column, row, &columnComm);
    }

    ~ProcessGrid() {
        // main aborts all ranks on exceptions; do not enter collectives while
        // unwinding a local allocation failure.
        if (std::uncaught_exceptions() != 0) {
            return;
        }
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&columnComm);
    }
};

size_t ownedPanels(size_t panels, int parts, int coordinate) {
    return panels / static_cast<size_t>(parts) +
           (static_cast<size_t>(coordinate) < panels % static_cast<size_t>(parts));
}

struct Panel {
    std::vector<double> A;
    std::vector<double> B;
    double* aData = nullptr;
    double* bData = nullptr;
    std::vector<MPI_Request> requests;

    Panel(size_t aCount, size_t bCount) : A(aCount), B(bCount) {
        requests.reserve(aCount / maxMessageCount + bCount / maxMessageCount + 2);
    }
};

// Chunk messages so the MPI int count limit never truncates a panel.
void broadcastPanel(double* data, size_t count, int owner, MPI_Comm comm,
                    std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, maxMessageCount));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, owner, comm, &requests.back());
        offset += static_cast<size_t>(chunk);
    }
}

// A 4x8 register tile vectorizes across independent output columns. Every
// element accumulates k in ascending order; no parallel floating-point reduction
// or reassociation is used. B is packed into contiguous k-by-8 micro-panels.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void multiplyTile(const double* A, const double* B, double* C, size_t strideA,
                  size_t strideC, size_t depth, size_t rows) {
#if defined(__AVX2__) && defined(__FMA__)
    if (rows == rowTile) {
        __m256d c00 = _mm256_loadu_pd(C);
        __m256d c01 = _mm256_loadu_pd(C + 4);
        __m256d c10 = _mm256_loadu_pd(C + strideC);
        __m256d c11 = _mm256_loadu_pd(C + strideC + 4);
        __m256d c20 = _mm256_loadu_pd(C + 2 * strideC);
        __m256d c21 = _mm256_loadu_pd(C + 2 * strideC + 4);
        __m256d c30 = _mm256_loadu_pd(C + 3 * strideC);
        __m256d c31 = _mm256_loadu_pd(C + 3 * strideC + 4);
        for (size_t k = 0; k < depth; ++k) {
            __m256d b0 = _mm256_loadu_pd(B + k * columnTile);
            __m256d b1 = _mm256_loadu_pd(B + k * columnTile + 4);
#if defined(__GNUC__)
            // Keep B in registers instead of reloading it for every FMA.
            __asm__("" : "+x"(b0), "+x"(b1));
#endif
            const __m256d a0 = _mm256_broadcast_sd(A + k);
            const __m256d a1 = _mm256_broadcast_sd(A + strideA + k);
            const __m256d a2 = _mm256_broadcast_sd(A + 2 * strideA + k);
            const __m256d a3 = _mm256_broadcast_sd(A + 3 * strideA + k);
            c00 = _mm256_fmadd_pd(a0, b0, c00);
            c01 = _mm256_fmadd_pd(a0, b1, c01);
            c10 = _mm256_fmadd_pd(a1, b0, c10);
            c11 = _mm256_fmadd_pd(a1, b1, c11);
            c20 = _mm256_fmadd_pd(a2, b0, c20);
            c21 = _mm256_fmadd_pd(a2, b1, c21);
            c30 = _mm256_fmadd_pd(a3, b0, c30);
            c31 = _mm256_fmadd_pd(a3, b1, c31);
        }
        _mm256_storeu_pd(C, c00);
        _mm256_storeu_pd(C + 4, c01);
        _mm256_storeu_pd(C + strideC, c10);
        _mm256_storeu_pd(C + strideC + 4, c11);
        _mm256_storeu_pd(C + 2 * strideC, c20);
        _mm256_storeu_pd(C + 2 * strideC + 4, c21);
        _mm256_storeu_pd(C + 3 * strideC, c30);
        _mm256_storeu_pd(C + 3 * strideC + 4, c31);
        return;
    }
    for (size_t i = 0; i < rows; ++i) {
        __m256d c0 = _mm256_loadu_pd(C + i * strideC);
        __m256d c1 = _mm256_loadu_pd(C + i * strideC + 4);
        for (size_t k = 0; k < depth; ++k) {
            const __m256d a = _mm256_broadcast_sd(A + i * strideA + k);
            c0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(B + k * columnTile), c0);
            c1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(B + k * columnTile + 4), c1);
        }
        _mm256_storeu_pd(C + i * strideC, c0);
        _mm256_storeu_pd(C + i * strideC + 4, c1);
    }
#else
    // Portable kernel for architectures without AVX2/FMA.
    for (size_t i = 0; i < rows; ++i) {
        std::array<double, columnTile> sums;
        std::copy_n(C + i * strideC, columnTile, sums.begin());
        for (size_t k = 0; k < depth; ++k) {
            const double a = A[i * strideA + k];
            for (size_t j = 0; j < columnTile; ++j) {
                sums[j] += a * B[k * columnTile + j];
            }
        }
        std::copy_n(sums.begin(), columnTile, C + i * strideC);
    }
#endif
}

class DistributedMatrices {
public:
    const Range rows;
    const Range columns;
    const size_t columnStride;
    const size_t panelWidth;
    const size_t panelCount;
    std::vector<double> C;

    DistributedMatrices(size_t N, const ProcessGrid& grid)
        : rows(partition(N, grid.dimensions[0], grid.row)),
          columns(partition(N, grid.dimensions[1], grid.column)),
          columnStride(((columns.count + columnTile - 1) / columnTile) * columnTile),
          panelWidth(std::min(maxPanelWidth,
                             N / static_cast<size_t>(grid.dimensions[0]) +
                             (N % static_cast<size_t>(grid.dimensions[0]) != 0))),
          panelCount(N / panelWidth + (N % panelWidth != 0)),
          C(rows.count * columnStride),
          aPanelSize(rows.count * panelWidth),
          bPanelSize(columnStride * panelWidth),
          localA(ownedPanels(panelCount, grid.dimensions[1], grid.column) * aPanelSize),
          localB(ownedPanels(panelCount, grid.dimensions[0], grid.row) * bPanelSize),
          panels{Panel(grid.dimensions[1] > 1 ? aPanelSize : 0,
                       grid.dimensions[0] > 1 ? bPanelSize : 0),
                 Panel(panelCount > 1 && grid.dimensions[1] > 1 ? aPanelSize : 0,
                       panelCount > 1 && grid.dimensions[0] > 1 ? bPanelSize : 0)} {
        // Block-cyclic ownership of the k panels keeps input storage O(N^2/P).
        // Initialization is local and uses global indices, so there is no
        // serial input allocation or scatter on rank zero.
        if (rows.count != 0) {
            for (size_t step = grid.column; step < panelCount; step += grid.dimensions[1]) {
                const size_t beginK = step * panelWidth;
                const size_t depth = std::min(panelWidth, N - beginK);
                double* panel = localA.data() + (step / grid.dimensions[1]) * aPanelSize;
                for (size_t i = 0; i < rows.count; ++i) {
                    for (size_t k = 0; k < depth; ++k) {
                        panel[i * panelWidth + k] = getPseudoRndValue(N, rows.begin + i, beginK + k);
                    }
                }
            }
        }
        if (columns.count != 0) {
            for (size_t step = grid.row; step < panelCount; step += grid.dimensions[0]) {
                const size_t beginK = step * panelWidth;
                const size_t depth = std::min(panelWidth, N - beginK);
                double* panel = localB.data() + (step / grid.dimensions[0]) * bPanelSize;
                for (size_t j = 0; j < columns.count; j += columnTile) {
                    for (size_t k = 0; k < depth; ++k) {
                        for (size_t lane = 0; lane < std::min(columnTile, columns.count - j); ++lane) {
                            panel[j * panelWidth + k * columnTile + lane] =
                                getPseudoRndValue(N, beginK + k, columns.begin + j + lane);
                        }
                    }
                }
            }
        }
    }

    void multiply(size_t N, const ProcessGrid& grid) {
        startPanel(0, grid);
        for (size_t step = 0; step < panelCount; ++step) {
            Panel& current = panels[step % 2];
            MPI_Waitall(static_cast<int>(current.requests.size()), current.requests.data(),
                        MPI_STATUSES_IGNORE);
            std::vector<MPI_Request>* pending = nullptr;
            if (step + 1 < panelCount) {
                startPanel(step + 1, grid);
                pending = &panels[(step + 1) % 2].requests;
            }
            const size_t depth = std::min(panelWidth, N - step * panelWidth);
            size_t tilesSinceProgress = 0;
            // Keep a small A row block in cache while streaming B micro-panels.
            for (size_t block = 0; block < rows.count; block += rowBlock) {
                const size_t end = std::min(block + rowBlock, rows.count);
                for (size_t j = 0; j < columnStride; j += columnTile) {
                    for (size_t i = block; i < end; i += rowTile) {
                        multiplyTile(current.aData + i * panelWidth,
                                     current.bData + j * panelWidth,
                                     C.data() + i * columnStride + j,
                                     panelWidth, columnStride, depth,
                                     std::min(rowTile, end - i));
                        // Explicit progress also overlaps communication on MPI
                        // implementations without a background progress thread.
                        if (pending != nullptr && ++tilesSinceProgress == 64) {
                            int complete = 0;
                            MPI_Testall(static_cast<int>(pending->size()), pending->data(),
                                        &complete, MPI_STATUSES_IGNORE);
                            if (complete) {
                                pending = nullptr;
                            }
                            tilesSinceProgress = 0;
                        }
                    }
                }
            }
        }
    }

private:
    const size_t aPanelSize;
    const size_t bPanelSize;
    std::vector<double> localA;
    std::vector<double> localB;
    std::array<Panel, 2> panels;

    void startPanel(size_t step, const ProcessGrid& grid) {
        Panel& panel = panels[step % 2];
        panel.requests.clear();
        const int aOwner = static_cast<int>(step % grid.dimensions[1]);
        const int bOwner = static_cast<int>(step % grid.dimensions[0]);
        panel.aData = panel.A.data();
        panel.bData = panel.B.data();
        if (grid.column == aOwner && aPanelSize != 0) {
            panel.aData = localA.data() + (step / grid.dimensions[1]) * aPanelSize;
        }
        if (grid.row == bOwner && bPanelSize != 0) {
            panel.bData = localB.data() + (step / grid.dimensions[0]) * bPanelSize;
        }
        broadcastPanel(panel.aData, aPanelSize, aOwner, grid.rowComm, panel.requests);
        broadcastPanel(panel.bData, bPanelSize, bOwner, grid.columnComm, panel.requests);
    }
};

// Gather only when the original complete, row-major results are requested.
// Chunked transfers support matrices larger than MPI's int element limit.
std::vector<double> gatherResult(const DistributedMatrices& matrices, size_t N,
                                 const ProcessGrid& grid, int rank, int processes) {
    if (rank != 0) {
        for (size_t offset = 0; offset < matrices.C.size();) {
            const int count = static_cast<int>(std::min(matrices.C.size() - offset, maxMessageCount));
            MPI_Send(matrices.C.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += static_cast<size_t>(count);
        }
        return {};
    }

    std::vector<double> result(N * N);
    std::vector<double> buffer(matrices.C.size()); // Rank zero owns the largest tile.
    for (int source = 0; source < processes; ++source) {
        const Range rows = partition(N, grid.dimensions[0], source / grid.dimensions[1]);
        const Range columns = partition(N, grid.dimensions[1], source % grid.dimensions[1]);
        const size_t stride = ((columns.count + columnTile - 1) / columnTile) * columnTile;
        const size_t elements = rows.count * stride;
        const double* data = matrices.C.data();
        if (source != 0) {
            for (size_t offset = 0; offset < elements;) {
                const int count = static_cast<int>(std::min(elements - offset, maxMessageCount));
                MPI_Recv(buffer.data() + offset, count, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(count);
            }
            data = buffer.data();
        }
        if (columns.count != 0) {
            for (size_t i = 0; i < rows.count; ++i) {
                std::copy_n(data + i * stride, columns.count,
                            result.data() + (rows.begin + i) * N + columns.begin);
            }
        }
    }
    return result;
}

bool validateResult(const DistributedMatrices& matrices, size_t N) {
    int valid = 1;
    // Validate the same 25 positions as the original, on their owning ranks.
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N;
            const size_t j = pj % N;
            if (i < matrices.rows.begin || i >= matrices.rows.begin + matrices.rows.count ||
                j < matrices.columns.begin || j >= matrices.columns.begin + matrices.columns.count) {
                continue;
            }
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = matrices.C[(i - matrices.rows.begin) * matrices.columnStride +
                                             j - matrices.columns.begin];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                valid = 0;
            }
        }
    }
    int globallyValid = 0;
    MPI_Allreduce(&valid, &globallyValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globallyValid != 0;
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
            const char* argument = argv[++i];
            const char* end = argument + strlen(argument);
            const auto parsed = std::from_chars(argument, end, N);
            if (parsed.ec != std::errc{} || parsed.ptr != end || N == 0 ||
                N > std::vector<double>().max_size() / N) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid matrix size: %s\n", argument);
                }
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    ProcessGrid grid(rank, processes);
    DistributedMatrices matrices(N, grid);
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrices.multiply(N, grid);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        // Use the full timer precision even for runs shorter than a millisecond.
        const double gflops = (2.0 * static_cast<double>(N) * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        const std::vector<double> result = gatherResult(matrices, N, grid, rank, processes);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(matrices, N);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }
    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // A local allocation failure must not leave other ranks in collectives.
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    MPI_Finalize();
    return status;
}
