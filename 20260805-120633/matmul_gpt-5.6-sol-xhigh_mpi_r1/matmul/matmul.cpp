#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

namespace {

constexpr size_t kTargetPanelWidth = 256;
constexpr size_t kRowTile = 16;
constexpr size_t kColumnTile = 128;
constexpr int kResultTag = 117;

constexpr size_t roundUp(const size_t value, const size_t multiple) {
    return value == 0 ? 0 : ((value + multiple - 1) / multiple) * multiple;
}

struct Block {
    size_t begin;
    size_t size;
};

struct PanelLayout {
    size_t begin;
    size_t size;
};

// Generate the same pseudo-random values as the original benchmark.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

Block getBlock(const size_t N, const int parts, const int coordinate) {
    const size_t partCount = static_cast<size_t>(parts);
    const size_t coord = static_cast<size_t>(coordinate);
    const size_t base = N / partCount;
    const size_t remainder = N % partCount;
    return {coord * base + std::min(coord, remainder),
            base + (coord < remainder ? 1U : 0U)};
}

PanelLayout getPanel(const size_t N, const size_t panelCount,
                     const size_t panel) {
    const size_t base = N / panelCount;
    const size_t remainder = N % panelCount;
    return {panel * base + std::min(panel, remainder),
            base + (panel < remainder ? 1U : 0U)};
}

int mpiCount(const size_t count, const int worldRank) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "An MPI panel is too large for this MPI implementation.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return static_cast<int>(count);
}

#if defined(__AVX2__)
// A 3x16 register tile uses twelve AVX2 accumulators, leaving enough registers
// for B and broadcast A values without spills. Smaller instantiations cover
// the final one or two rows.
template <size_t Rows>
void multiplyRowsAvx2(const double* const A, const double* const B,
                      double* const C, const size_t columns,
                      const size_t columnBegin, const size_t columnEnd,
                      const size_t inner, const bool fusedLastValue) {
    const size_t ordinaryInner = inner - (fusedLastValue ? 1U : 0U);
    size_t j = columnBegin;
    for (; j + 16 <= columnEnd; j += 16) {
        __m256d accumulators[Rows][4];
        for (size_t row = 0; row < Rows; ++row) {
            accumulators[row][0] =
                _mm256_loadu_pd(C + row * columns + j);
            accumulators[row][1] =
                _mm256_loadu_pd(C + row * columns + j + 4);
            accumulators[row][2] =
                _mm256_loadu_pd(C + row * columns + j + 8);
            accumulators[row][3] =
                _mm256_loadu_pd(C + row * columns + j + 12);
        }

        for (size_t k = 0; k < ordinaryInner; ++k) {
            const double* const b = B + k * columns + j;
            const __m256d b0 = _mm256_loadu_pd(b);
            const __m256d b1 = _mm256_loadu_pd(b + 4);
            const __m256d b2 = _mm256_loadu_pd(b + 8);
            const __m256d b3 = _mm256_loadu_pd(b + 12);
            for (size_t row = 0; row < Rows; ++row) {
                const __m256d a = _mm256_broadcast_sd(A + row * inner + k);
                accumulators[row][0] =
                    _mm256_add_pd(accumulators[row][0],
                                  _mm256_mul_pd(a, b0));
                accumulators[row][1] =
                    _mm256_add_pd(accumulators[row][1],
                                  _mm256_mul_pd(a, b1));
                accumulators[row][2] =
                    _mm256_add_pd(accumulators[row][2],
                                  _mm256_mul_pd(a, b2));
                accumulators[row][3] =
                    _mm256_add_pd(accumulators[row][3],
                                  _mm256_mul_pd(a, b3));
            }
        }
        if (fusedLastValue) {
            const size_t k = ordinaryInner;
            const double* const b = B + k * columns + j;
            const __m256d b0 = _mm256_loadu_pd(b);
            const __m256d b1 = _mm256_loadu_pd(b + 4);
            const __m256d b2 = _mm256_loadu_pd(b + 8);
            const __m256d b3 = _mm256_loadu_pd(b + 12);
            for (size_t row = 0; row < Rows; ++row) {
                const __m256d a = _mm256_broadcast_sd(A + row * inner + k);
                accumulators[row][0] =
                    _mm256_fmadd_pd(a, b0, accumulators[row][0]);
                accumulators[row][1] =
                    _mm256_fmadd_pd(a, b1, accumulators[row][1]);
                accumulators[row][2] =
                    _mm256_fmadd_pd(a, b2, accumulators[row][2]);
                accumulators[row][3] =
                    _mm256_fmadd_pd(a, b3, accumulators[row][3]);
            }
        }

        for (size_t row = 0; row < Rows; ++row) {
            _mm256_storeu_pd(C + row * columns + j, accumulators[row][0]);
            _mm256_storeu_pd(C + row * columns + j + 4,
                             accumulators[row][1]);
            _mm256_storeu_pd(C + row * columns + j + 8,
                             accumulators[row][2]);
            _mm256_storeu_pd(C + row * columns + j + 12,
                             accumulators[row][3]);
        }
    }

    for (; j + 4 <= columnEnd; j += 4) {
        __m256d accumulators[Rows];
        for (size_t row = 0; row < Rows; ++row) {
            accumulators[row] = _mm256_loadu_pd(C + row * columns + j);
        }
        for (size_t k = 0; k < ordinaryInner; ++k) {
            const __m256d b = _mm256_loadu_pd(B + k * columns + j);
            for (size_t row = 0; row < Rows; ++row) {
                const __m256d a = _mm256_broadcast_sd(A + row * inner + k);
                accumulators[row] = _mm256_add_pd(
                    accumulators[row], _mm256_mul_pd(a, b));
            }
        }
        if (fusedLastValue) {
            const size_t k = ordinaryInner;
            const __m256d b = _mm256_loadu_pd(B + k * columns + j);
            for (size_t row = 0; row < Rows; ++row) {
                const __m256d a = _mm256_broadcast_sd(A + row * inner + k);
                accumulators[row] =
                    _mm256_fmadd_pd(a, b, accumulators[row]);
            }
        }
        for (size_t row = 0; row < Rows; ++row) {
            _mm256_storeu_pd(C + row * columns + j, accumulators[row]);
        }
    }

    for (; j < columnEnd; ++j) {
        double accumulators[Rows];
        for (size_t row = 0; row < Rows; ++row) {
            accumulators[row] = C[row * columns + j];
        }
        for (size_t k = 0; k < ordinaryInner; ++k) {
            const double b = B[k * columns + j];
            for (size_t row = 0; row < Rows; ++row) {
                accumulators[row] += A[row * inner + k] * b;
            }
        }
        if (fusedLastValue) {
            const size_t k = ordinaryInner;
            const double b = B[k * columns + j];
            for (size_t row = 0; row < Rows; ++row) {
                accumulators[row] =
                    std::fma(A[row * inner + k], b, accumulators[row]);
            }
        }
        for (size_t row = 0; row < Rows; ++row) {
            C[row * columns + j] = accumulators[row];
        }
    }
}
#endif

// C += A * B for one SUMMA panel. The AVX2 path keeps each output micro-tile
// in registers for the entire panel; the portable path is cache blocked and
// exposes a contiguous, vectorizable innermost loop.
void multiplyPanel(const double* const A, const double* const B,
                   std::vector<double>& C, const size_t rows,
                   const size_t columns, const size_t inner,
                   const bool fusedLastValue) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#if defined(__AVX2__)
    for (size_t columnBlock = 0; columnBlock < columns;
         columnBlock += kColumnTile) {
        const size_t columnEnd =
            std::min(columnBlock + kColumnTile, columns);
        size_t row = 0;
        for (; row + 3 <= rows; row += 3) {
            multiplyRowsAvx2<3>(A + row * inner, B,
                                C.data() + row * columns, columns,
                                columnBlock, columnEnd, inner,
                                fusedLastValue);
        }
        if (rows - row == 2) {
            multiplyRowsAvx2<2>(A + row * inner, B,
                                C.data() + row * columns, columns,
                                columnBlock, columnEnd, inner,
                                fusedLastValue);
        } else if (rows - row == 1) {
            multiplyRowsAvx2<1>(A + row * inner, B,
                                C.data() + row * columns, columns,
                                columnBlock, columnEnd, inner,
                                fusedLastValue);
        }
    }
#else
    for (size_t columnBlock = 0; columnBlock < columns;
         columnBlock += kColumnTile) {
        for (size_t rowBlock = 0; rowBlock < rows; rowBlock += kRowTile) {
            const size_t rowEnd = std::min(rowBlock + kRowTile, rows);
            const size_t columnEnd =
                std::min(columnBlock + kColumnTile, columns);
            const size_t ordinaryInner =
                inner - (fusedLastValue ? 1U : 0U);
            for (size_t k = 0; k < ordinaryInner; ++k) {
                const double* const bRow = B + k * columns;
                for (size_t i = rowBlock; i < rowEnd; ++i) {
                    const double a = A[i * inner + k];
                    double* const cRow = C.data() + i * columns;
                    for (size_t j = columnBlock; j < columnEnd; ++j) {
                        cRow[j] += a * bRow[j];
                    }
                }
            }
            if (fusedLastValue) {
                const size_t k = ordinaryInner;
                const double* const bRow = B + k * columns;
                for (size_t i = rowBlock; i < rowEnd; ++i) {
                    const double a = A[i * inner + k];
                    double* const cRow = C.data() + i * columns;
                    for (size_t j = columnBlock; j < columnEnd; ++j) {
                        cRow[j] = std::fma(a, bRow[j], cRow[j]);
                    }
                }
            }
        }
    }
#endif
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

class DistributedMatrices {
  public:
    DistributedMatrices(const size_t matrixSize, const int processRows,
                        const int processColumns, const int processRow,
                        const int processColumn, const int rank)
        : N(matrixSize), gridRows(processRows), gridColumns(processColumns),
          gridRow(processRow), gridColumn(processColumn), worldRank(rank),
          rowBlock(getBlock(N, gridRows, gridRow)),
          columnBlock(getBlock(N, gridColumns, gridColumn)),
          paddedRows(roundUp(rowBlock.size, 3)),
          paddedColumns(roundUp(columnBlock.size, 16)),
          panelCount(std::min(
              N, std::max((N + kTargetPanelWidth - 1) / kTargetPanelWidth,
                          static_cast<size_t>(std::max(gridRows,
                                                       gridColumns))))),
          aOffsets(panelCount, 0), bOffsets(panelCount, 0),
          C(paddedRows * paddedColumns, 0.0) {
        MPI_Comm_split(MPI_COMM_WORLD, gridRow, gridColumn, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, gridColumn, gridRow, &columnComm);

        initializeDistributedInputs();

        const size_t maximumPanel = (N + panelCount - 1) / panelCount;
        for (auto& buffer : aReceive) {
            buffer.resize(paddedRows * maximumPanel);
        }
        for (auto& buffer : bReceive) {
            buffer.resize(maximumPanel * paddedColumns);
        }
    }

    DistributedMatrices(const DistributedMatrices&) = delete;
    DistributedMatrices& operator=(const DistributedMatrices&) = delete;

    ~DistributedMatrices() {
        if (rowComm != MPI_COMM_NULL) {
            MPI_Comm_free(&rowComm);
        }
        if (columnComm != MPI_COMM_NULL) {
            MPI_Comm_free(&columnComm);
        }
    }

    double multiply() {
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();

        MPI_Request aRequests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Request bRequests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        postPanel(0, aRequests[0], bRequests[0]);
        for (size_t panel = 0; panel < panelCount; ++panel) {
            const size_t slot = panel & 1U;
            MPI_Request requests[2] = {aRequests[slot], bRequests[slot]};
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            aRequests[slot] = MPI_REQUEST_NULL;
            bRequests[slot] = MPI_REQUEST_NULL;

            if (panel + 1 < panelCount) {
                const size_t nextSlot = (panel + 1) & 1U;
                postPanel(panel + 1, aRequests[nextSlot],
                          bRequests[nextSlot]);
            }

            const PanelLayout layout = getPanel(N, panelCount, panel);
            multiplyPanel(aPanelData(panel, slot), bPanelData(panel, slot), C,
                          paddedRows, paddedColumns, layout.size,
                          (N & 1U) != 0 && layout.begin + layout.size == N);
        }

        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        return elapsed;
    }

    std::vector<double> gatherResult() const {
        std::vector<double> packed(rowBlock.size * columnBlock.size);
        if (columnBlock.size != 0) {
            for (size_t i = 0; i < rowBlock.size; ++i) {
                std::copy_n(C.data() + i * paddedColumns, columnBlock.size,
                            packed.data() + i * columnBlock.size);
            }
        }

        if (worldRank != 0) {
            sendLarge(packed.data(), packed.size(), 0);
            return {};
        }

        std::vector<double> result(N * N);
        std::vector<double> received;
        for (int source = 0; source < gridRows * gridColumns; ++source) {
            const int sourceRow = source / gridColumns;
            const int sourceColumn = source % gridColumns;
            const Block rows = getBlock(N, gridRows, sourceRow);
            const Block columns = getBlock(N, gridColumns, sourceColumn);
            const size_t count = rows.size * columns.size;

            const double* sourceData = nullptr;
            if (source == 0) {
                sourceData = packed.data();
            } else {
                received.resize(count);
                receiveLarge(received.data(), count, source);
                sourceData = received.data();
            }

            if (columns.size != 0) {
                for (size_t i = 0; i < rows.size; ++i) {
                    std::copy_n(sourceData + i * columns.size, columns.size,
                                result.data() + (rows.begin + i) * N +
                                    columns.begin);
                }
            }
        }
        return result;
    }

    bool validate(double& failedExpected, double& failedActual,
                  size_t& failedRow, size_t& failedColumn) const {
        constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
        int localFailure = 25;

        for (int pi = 0; pi < 5 && localFailure == 25; ++pi) {
            for (int pj = 0; pj < 5; ++pj) {
                const size_t i = checkPoints[pi] % N;
                const size_t j = checkPoints[pj] % N;
                if (i < rowBlock.begin || i >= rowBlock.begin + rowBlock.size ||
                    j < columnBlock.begin ||
                    j >= columnBlock.begin + columnBlock.size) {
                    continue;
                }

                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += getPseudoRndValue(N, i, k) *
                                getPseudoRndValue(N, k, j);
                }
                const double actual =
                    C[(i - rowBlock.begin) * paddedColumns +
                      (j - columnBlock.begin)];
                const double relativeError =
                    std::abs((actual - expected) / (expected + 1e-10));
                if (relativeError > 1e-6) {
                    localFailure = pi * 5 + pj;
                    break;
                }
            }
        }

        int globalFailure = 25;
        MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (globalFailure == 25) {
            return true;
        }

        failedRow = checkPoints[globalFailure / 5] % N;
        failedColumn = checkPoints[globalFailure % 5] % N;
        failedExpected = 0.0;
        for (size_t k = 0; k < N; ++k) {
            failedExpected += getPseudoRndValue(N, failedRow, k) *
                              getPseudoRndValue(N, k, failedColumn);
        }

        double localActual = 0.0;
        if (failedRow >= rowBlock.begin &&
            failedRow < rowBlock.begin + rowBlock.size &&
            failedColumn >= columnBlock.begin &&
            failedColumn < columnBlock.begin + columnBlock.size) {
            localActual =
                C[(failedRow - rowBlock.begin) * paddedColumns +
                  (failedColumn - columnBlock.begin)];
        }
        MPI_Reduce(&localActual, &failedActual, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        return false;
    }

    const Block& localRows() const { return rowBlock; }
    const Block& localColumns() const { return columnBlock; }
    size_t panels() const { return panelCount; }

  private:
    void initializeDistributedInputs() {
        // Panels are stored contiguously on their broadcast owner. Across the
        // process grid, each element of A and B is stored exactly once.
        for (size_t panel = 0; panel < panelCount; ++panel) {
            const PanelLayout layout = getPanel(N, panelCount, panel);
            if (static_cast<int>(panel % static_cast<size_t>(gridColumns)) ==
                gridColumn) {
                aOffsets[panel] = localA.size();
                localA.resize(localA.size() + paddedRows * layout.size, 0.0);
                double* const destination =
                    aOffsets[panel] == 0
                        ? localA.data()
                        : localA.data() + aOffsets[panel];
                for (size_t i = 0; i < rowBlock.size; ++i) {
                    for (size_t k = 0; k < layout.size; ++k) {
                        destination[i * layout.size + k] =
                            getPseudoRndValue(N, rowBlock.begin + i,
                                              layout.begin + k);
                    }
                }
            }

            if (static_cast<int>(panel % static_cast<size_t>(gridRows)) ==
                gridRow) {
                bOffsets[panel] = localB.size();
                localB.resize(localB.size() + layout.size * paddedColumns,
                              0.0);
                double* const destination =
                    bOffsets[panel] == 0
                        ? localB.data()
                        : localB.data() + bOffsets[panel];
                for (size_t k = 0; k < layout.size; ++k) {
                    for (size_t j = 0; j < columnBlock.size; ++j) {
                        destination[k * paddedColumns + j] =
                            getPseudoRndValue(N, layout.begin + k,
                                              columnBlock.begin + j);
                    }
                }
            }
        }
    }

    double* aPanelData(const size_t panel, const size_t slot) {
        const int owner =
            static_cast<int>(panel % static_cast<size_t>(gridColumns));
        if (owner != gridColumn) {
            return aReceive[slot].data();
        }
        return aOffsets[panel] == 0 ? localA.data()
                                    : localA.data() + aOffsets[panel];
    }

    const double* aPanelData(const size_t panel, const size_t slot) const {
        const int owner =
            static_cast<int>(panel % static_cast<size_t>(gridColumns));
        if (owner != gridColumn) {
            return aReceive[slot].data();
        }
        return aOffsets[panel] == 0 ? localA.data()
                                    : localA.data() + aOffsets[panel];
    }

    double* bPanelData(const size_t panel, const size_t slot) {
        const int owner =
            static_cast<int>(panel % static_cast<size_t>(gridRows));
        if (owner != gridRow) {
            return bReceive[slot].data();
        }
        return bOffsets[panel] == 0 ? localB.data()
                                    : localB.data() + bOffsets[panel];
    }

    const double* bPanelData(const size_t panel, const size_t slot) const {
        const int owner =
            static_cast<int>(panel % static_cast<size_t>(gridRows));
        if (owner != gridRow) {
            return bReceive[slot].data();
        }
        return bOffsets[panel] == 0 ? localB.data()
                                    : localB.data() + bOffsets[panel];
    }

    void postPanel(const size_t panel, MPI_Request& aRequest,
                   MPI_Request& bRequest) {
        const size_t slot = panel & 1U;
        const PanelLayout layout = getPanel(N, panelCount, panel);
        const int aOwner =
            static_cast<int>(panel % static_cast<size_t>(gridColumns));
        const int bOwner =
            static_cast<int>(panel % static_cast<size_t>(gridRows));

        MPI_Ibcast(aPanelData(panel, slot),
                   mpiCount(paddedRows * layout.size, worldRank), MPI_DOUBLE,
                   aOwner, rowComm, &aRequest);
        MPI_Ibcast(bPanelData(panel, slot),
                   mpiCount(layout.size * paddedColumns, worldRank),
                   MPI_DOUBLE, bOwner, columnComm, &bRequest);
    }

    void sendLarge(const double* data, size_t count, const int destination) const {
        const size_t chunkLimit =
            static_cast<size_t>(std::numeric_limits<int>::max());
        while (count != 0) {
            const int chunk = static_cast<int>(std::min(count, chunkLimit));
            MPI_Send(data, chunk, MPI_DOUBLE, destination, kResultTag,
                     MPI_COMM_WORLD);
            data += chunk;
            count -= static_cast<size_t>(chunk);
        }
    }

    void receiveLarge(double* data, size_t count, const int source) const {
        const size_t chunkLimit =
            static_cast<size_t>(std::numeric_limits<int>::max());
        while (count != 0) {
            const int chunk = static_cast<int>(std::min(count, chunkLimit));
            MPI_Recv(data, chunk, MPI_DOUBLE, source, kResultTag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data += chunk;
            count -= static_cast<size_t>(chunk);
        }
    }

    size_t N;
    int gridRows;
    int gridColumns;
    int gridRow;
    int gridColumn;
    int worldRank;
    Block rowBlock;
    Block columnBlock;
    size_t paddedRows;
    size_t paddedColumns;
    size_t panelCount;
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm columnComm = MPI_COMM_NULL;
    std::vector<size_t> aOffsets;
    std::vector<size_t> bOffsets;
    std::vector<double> localA;
    std::vector<double> localB;
    std::vector<double> aReceive[2];
    std::vector<double> bReceive[2];
    std::vector<double> C;
};

void printUsage(const char* const programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf(
        "  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* const value, size_t& result) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (value[0] == '\0' || value[0] == '-' || end == nullptr || *end != '\0' ||
        parsed == 0 ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<size_t>::max())) {
        return false;
    }
    result = static_cast<size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            argumentsValid = parseSize(argv[++i], N) && argumentsValid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    int allArgumentsValid = argumentsValid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &allArgumentsValid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (showHelp || allArgumentsValid == 0) {
        if (worldRank == 0) {
            if (allArgumentsValid == 0) {
                std::printf("Matrix size must be a positive integer.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return allArgumentsValid == 0 ? 1 : 0;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int gridRow = worldRank / dimensions[1];
    const int gridColumn = worldRank % dimensions[1];

    int returnCode = 0;
    if (worldRank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n",
                    validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d process grid)\n", worldSize,
                    dimensions[0], dimensions[1]);
        std::printf("Initializing distributed matrices...\n");
    }

    {
        DistributedMatrices matrices(N, dimensions[0], dimensions[1], gridRow,
                                     gridColumn, worldRank);

        if (worldRank == 0) {
            std::printf("Computing matrix multiplication...\n");
        }

        const double elapsed = matrices.multiply();
        if (worldRank == 0) {
            const long long milliseconds =
                static_cast<long long>(elapsed * 1000.0);
            const double operations =
                2.0 * static_cast<double>(N) * static_cast<double>(N) *
                static_cast<double>(N);
            const double gflops = operations / elapsed / 1.0e9;
            std::printf("Computation time: %lld ms\n", milliseconds);
            std::printf("Performance: %.3f GFLOPS\n", gflops);
        }

        if (printResults) {
            std::vector<double> result = matrices.gatherResult();
            if (worldRank == 0) {
                print_results(result, "MatrixC");
            }
        }

        if (validate) {
            if (worldRank == 0) {
                std::printf("Validating result...\n");
            }
            double expected = 0.0;
            double actual = 0.0;
            size_t failedRow = 0;
            size_t failedColumn = 0;
            const bool valid = matrices.validate(expected, actual, failedRow,
                                                 failedColumn);
            if (worldRank == 0) {
                if (valid) {
                    std::printf("Validation: PASSED\n");
                } else {
                    const double relativeError =
                        std::abs((actual - expected) / (expected + 1e-10));
                    std::printf(
                        "Validation failed at (%zu, %zu): expected %.10f, got "
                        "%.10f (error: %.10e)\n",
                        failedRow, failedColumn, expected, actual,
                        relativeError);
                    std::printf("Validation: FAILED\n");
                    returnCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
