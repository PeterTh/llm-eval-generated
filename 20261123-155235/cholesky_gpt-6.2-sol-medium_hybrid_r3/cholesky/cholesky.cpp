#include <algorithm>
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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

constexpr double one = 1.0;
constexpr double zero = 0.0;
constexpr double minusOne = -1.0;

[[noreturn]] void fail(const char* message, int status) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s (%d)\n", rank, message, status);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CHECK(call) do {                              \
    const auto status = (call);                       \
    if (status != 0) fail(#call, status);              \
} while (false)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fail("CUDA allocation size overflows", 1);
        }
        if (count) CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) {
        if (count) CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(double)));
    }
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    double* data = nullptr;
};

size_t elements(int rows, int cols) {
    return static_cast<size_t>(rows) * cols;
}

int mpiCount(size_t count) {
    if (count > INT_MAX) fail("MPI panel exceeds the supported message size", 1);
    return static_cast<int>(count);
}

struct Grid {
    int rank, size;
    int dims[2] = {};
    int row, col;
    MPI_Comm rowComm, colComm;

    Grid() {
        CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        CHECK(MPI_Dims_create(size, 2, dims));
        row = rank / dims[1];
        col = rank % dims[1];
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));

        MPI_Comm local;
        CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local));
        int localRank, devices;
        CHECK(MPI_Comm_rank(local, &localRank));
        CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) fail("A CUDA device is required", 1);
        CHECK(cudaSetDevice(localRank % devices));
        CHECK(MPI_Comm_free(&local));
    }

    ~Grid() {
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&rowComm);
    }
};

struct Axis {
    int n, block, parts, coord;

    int tileCount() const { return n / block + (n % block != 0); }
    int width(int tile) const { return std::min(block, n - tile * block); }
    int first(int tile) const {
        return tile + (coord - tile % parts + parts) % parts;
    }
    // Local offset of the first owned tile at or after the given global tile.
    int offset(int tile) const {
        const auto index = static_cast<int64_t>(first(std::min(tile, tileCount()))) / parts;
        return static_cast<int>(std::min(index * block, static_cast<int64_t>(length())));
    }
    int length() const {
        const int full = n / block;
        const int ownedFull = full / parts + (coord < full % parts);
        return ownedFull * block + (coord == full % parts ? n % block : 0);
    }
    int globalIndex(int local) const {
        return ((local / block) * parts + coord) * block + local % block;
    }
};

struct Gpu {
    cudaStream_t panel, bulk;
    cublasHandle_t panelBlas, bulkBlas;
    cusolverDnHandle_t solver;
    cudaEvent_t bulkDone;

    Gpu() {
        int greatestPriority;
        CHECK(cudaDeviceGetStreamPriorityRange(nullptr, &greatestPriority));
        CHECK(cudaStreamCreateWithPriority(&panel, cudaStreamNonBlocking, greatestPriority));
        CHECK(cudaStreamCreateWithFlags(&bulk, cudaStreamNonBlocking));
        CHECK(cudaEventCreateWithFlags(&bulkDone, cudaEventDisableTiming));
        CHECK(cublasCreate(&panelBlas));
        CHECK(cublasCreate(&bulkBlas));
        CHECK(cublasSetStream(panelBlas, panel));
        CHECK(cublasSetStream(bulkBlas, bulk));
        CHECK(cusolverDnCreate(&solver));
        CHECK(cusolverDnSetStream(solver, panel));
    }

    ~Gpu() {
        cusolverDnDestroy(solver);
        cublasDestroy(bulkBlas);
        cublasDestroy(panelBlas);
        cudaEventDestroy(bulkDone);
        cudaStreamDestroy(bulk);
        cudaStreamDestroy(panel);
    }

    void synchronize() const {
        CHECK(cudaStreamSynchronize(panel));
        CHECK(cudaStreamSynchronize(bulk));
    }
};

void copyMatrix(double* dest, int ldDest, const double* source, int ldSource,
                int height, int columns, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!height || !columns) return;
    if (ldDest == height && ldSource == height) {
        CHECK(cudaMemcpyAsync(dest, source, elements(height, columns) * sizeof(double),
                              kind, stream));
    } else {
        CHECK(cudaMemcpy2DAsync(dest, elements(ldDest, 1) * sizeof(double),
                                source, elements(ldSource, 1) * sizeof(double),
                                elements(height, 1) * sizeof(double), columns, kind, stream));
    }
}

// glibc rand_r advances a 32-bit LCG three times per output. Compose
// those affine steps to jump directly to any row segment of the original B.
unsigned int randomSeed(uint64_t position) {
#if defined(__GLIBC__)
    uint32_t scale = 1103515245u;
    uint32_t shift = 12345u;
    uint32_t seed = 42u;
    for (position *= 3; position; position >>= 1) {
        if (position & 1) seed = scale * seed + shift;
        shift = scale * shift + shift;
        scale *= scale;
    }
    return seed;
#else
    unsigned int seed = 42;
    for (uint64_t i = 0; i < position; ++i) rand_r(&seed);
    return seed;
#endif
}

void fillRandom(double* data, const Axis& axis, int begin, int width) {
    #pragma omp parallel for schedule(static) if(elements(axis.length(), width) >= 16384)
    for (int i = 0; i < axis.length(); ++i) {
        unsigned int seed = randomSeed(static_cast<uint64_t>(axis.globalIndex(i)) * axis.n + begin);
        double* segment = data + elements(i, width);
        for (int k = 0; k < width; ++k) {
            segment[k] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        }
    }
}

void generateMatrix(double* matrix, const Axis& rows, const Axis& cols, const Gpu& gpu) {
    const int ld = rows.length();
    if (!ld || !cols.length()) return;

    // Bound workspace even when the local rows span a large fraction of B.
    const int chunk = std::min(rows.n, 2048);
    const bool shared = rows.parts == cols.parts && rows.coord == cols.coord;
    PinnedBuffer hostRows(elements(chunk, ld));
    PinnedBuffer hostCols(shared ? 0 : elements(chunk, cols.length()));
    DeviceBuffer<double> deviceRows(elements(chunk, ld));
    DeviceBuffer<double> deviceCols(shared ? 0 : elements(chunk, cols.length()));
    cudaEvent_t uploaded;
    CHECK(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
    for (int begin = 0; begin < rows.n;) {
        // The CPU can refill the pinned buffers as soon as the upload ends.
        // GPU work for the next chunk remains ordered on the same stream.
        CHECK(cudaEventSynchronize(uploaded));
        const int width = std::min(chunk, rows.n - begin);
        fillRandom(hostRows.data, rows, begin, width);
        if (!shared) fillRandom(hostCols.data, cols, begin, width);
        copyMatrix(deviceRows.data, width, hostRows.data, width, width, ld,
                   cudaMemcpyHostToDevice, gpu.panel);
        if (!shared) {
            copyMatrix(deviceCols.data, width, hostCols.data, width, width, cols.length(),
                       cudaMemcpyHostToDevice, gpu.panel);
        }
        CHECK(cudaEventRecord(uploaded, gpu.panel));
        const double* right = shared ? deviceRows.data : deviceCols.data;
        const double* beta = begin == 0 ? &zero : &one;
        if (shared) {
            CHECK(cublasDsyrk(gpu.panelBlas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                              ld, width, &one, deviceRows.data, width, beta, matrix, ld));
        }
        for (int j = cols.coord; !shared && j < cols.tileCount(); j += cols.parts) {
            const int row = rows.offset(j);
            if (row == ld) break;
            const int col = cols.offset(j);
            CHECK(cublasDgemm(gpu.panelBlas, CUBLAS_OP_T, CUBLAS_OP_N,
                              ld - row, cols.width(j), width, &one,
                              deviceRows.data + elements(row, width), width,
                              right + elements(col, width), width, beta,
                              matrix + elements(col, ld) + row, ld));
        }
        begin += width;
    }

    std::vector<double> shift(rows.block, static_cast<double>(rows.n));
    DeviceBuffer<double> diagonalShift(shift.size());
    CHECK(cudaMemcpyAsync(diagonalShift.data, shift.data(), shift.size() * sizeof(double),
                          cudaMemcpyHostToDevice, gpu.panel));
    for (int j = cols.coord; j < cols.tileCount(); j += cols.parts) {
        if (j % rows.parts != rows.coord) continue;
        double* diagonal = matrix + elements(cols.offset(j), ld) + rows.offset(j);
        CHECK(cublasDaxpy(gpu.panelBlas, cols.width(j), &one,
                         diagonalShift.data, 1, diagonal, ld + 1));
    }
    CHECK(cudaStreamSynchronize(gpu.panel));
    CHECK(cudaEventDestroy(uploaded));
}

struct Panel {
    PinnedBuffer hostRows, hostCols, packedCols;
    DeviceBuffer<double> deviceRows, deviceCols;
    cudaEvent_t ready, consumed;
    const double* rowData;
    const double* colData;
    int rowLd, colLd;

    Panel(const Axis& rows, const Axis& cols)
        : hostRows(elements(rows.length(), rows.block)),
          hostCols(rows.parts == cols.parts && rows.coord == cols.coord
                   ? 0 : elements(cols.length(), rows.block)),
          packedCols(rows.parts == cols.parts ? 0 : elements(cols.length(), rows.block)),
          deviceRows(elements(rows.length(), rows.block)),
          deviceCols(rows.parts == cols.parts && rows.coord == cols.coord
                     ? 0 : elements(cols.length(), rows.block)) {
        CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
    }

    ~Panel() {
        cudaEventDestroy(consumed);
        cudaEventDestroy(ready);
    }
};

void exchangePanel(Panel& panel, const Grid& grid, const Axis& rows, const Axis& cols,
                   int first, int ownerCol, int width, const Gpu& gpu,
                   const double* localPanel = nullptr) {
    const int rowStart = rows.offset(first), colStart = cols.offset(first);
    const int rowCount = rows.length() - rowStart, colCount = cols.length() - colStart;
    if (rowCount) {
        CHECK(MPI_Bcast(panel.hostRows.data, mpiCount(elements(rowCount, width)),
                        MPI_DOUBLE, ownerCol, grid.rowComm));
    }

    // Keep a device slot alive until the bulk stream finishes reading it.
    CHECK(cudaStreamWaitEvent(gpu.panel, panel.consumed, 0));
    panel.rowData = localPanel ? localPanel : panel.deviceRows.data;
    panel.rowLd = localPanel ? rows.length() : rowCount;
    if (!localPanel) {
        copyMatrix(panel.deviceRows.data, rowCount, panel.hostRows.data, rowCount,
                   rowCount, width, cudaMemcpyHostToDevice, gpu.panel);
    }

    const bool shared = rows.parts == cols.parts && grid.row == grid.col;
    if (colCount && rows.parts == cols.parts) {
        // Each column's entire panel comes from the process row with that index.
        double* buffer = shared ? panel.hostRows.data : panel.hostCols.data;
        CHECK(MPI_Bcast(buffer, mpiCount(elements(colCount, width)), MPI_DOUBLE,
                        grid.col, grid.colComm));
    } else if (colCount) {
        // Pack each producer's tiles together, then restore local column order.
        std::vector<int> counts(rows.parts, 0), offsets(rows.parts);
        const int firstColTile = cols.first(first);
        for (int tile = firstColTile; tile < cols.tileCount(); tile += cols.parts) {
            const int producer = tile % rows.parts;
            counts[producer] = mpiCount(static_cast<size_t>(counts[producer])
                                       + elements(cols.width(tile), width));
        }
        mpiCount(elements(colCount, width));
        for (int r = 1; r < rows.parts; ++r) offsets[r] = offsets[r - 1] + counts[r - 1];
        std::vector<int> next = offsets;
        std::vector<int> tileOffsets;
        for (int tile = firstColTile; tile < cols.tileCount(); tile += cols.parts) {
            const int producer = tile % rows.parts;
            tileOffsets.push_back(next[producer]);
            next[producer] += cols.width(tile) * width;
        }

        const int tiles = static_cast<int>(tileOffsets.size());
        #pragma omp parallel for schedule(static) if(tiles > 1 && elements(colCount, width) >= 65536)
        for (int t = 0; t < tiles; ++t) {
            const int tile = firstColTile + t * cols.parts;
            if (tile % rows.parts != grid.row) continue;
            const int height = cols.width(tile);
            const int localRow = rows.offset(tile) - rowStart;
            double* dest = panel.packedCols.data + tileOffsets[t];
            for (int k = 0; k < width; ++k) {
                std::copy_n(panel.hostRows.data + elements(k, rowCount) + localRow,
                            height, dest + elements(k, height));
            }
        }

        CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, panel.packedCols.data,
                             counts.data(), offsets.data(), MPI_DOUBLE, grid.colComm));

        #pragma omp parallel for schedule(static) if(tiles > 1 && elements(colCount, width) >= 65536)
        for (int t = 0; t < tiles; ++t) {
            const int tile = firstColTile + t * cols.parts;
            const int height = cols.width(tile);
            const int localCol = cols.offset(tile) - colStart;
            const double* source = panel.packedCols.data + tileOffsets[t];
            for (int k = 0; k < width; ++k) {
                std::copy_n(source + elements(k, height), height,
                            panel.hostCols.data + elements(k, colCount) + localCol);
            }
        }
    }

    panel.colData = shared ? panel.rowData : panel.deviceCols.data;
    panel.colLd = shared ? panel.rowLd : colCount;
    if (!shared) {
        copyMatrix(panel.deviceCols.data, colCount, panel.hostCols.data, colCount,
                   colCount, width, cudaMemcpyHostToDevice, gpu.panel);
    }
    CHECK(cudaEventRecord(panel.ready, gpu.panel));
}

void updateTileColumn(double* matrix, const Axis& rows, const Axis& cols, const Panel& panel,
                      int first, int width, int tile, const double* alpha, cublasHandle_t blas) {
    const int row = rows.offset(tile), col = cols.offset(tile), ld = rows.length();
    if (row == ld) return;
    const int rowStart = rows.offset(first), colStart = cols.offset(first);
    CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, ld - row, cols.width(tile),
                      width, alpha, panel.rowData + (row - rowStart), panel.rowLd,
                      panel.colData + (col - colStart), panel.colLd,
                      &one, matrix + elements(col, ld) + row, ld));
}

int factor(double* matrix, const Grid& grid, const Axis& rows, const Axis& cols,
           const Gpu& gpu, double* work, int workSize, int* deviceInfo,
           PinnedBuffer& diagonal, Panel* even, Panel* odd) {
    const int ld = rows.length();
    if (grid.size == 1) {
        int info;
        CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, rows.n,
                              matrix, ld, work, workSize, deviceInfo));
        CHECK(cudaMemcpyAsync(&info, deviceInfo, sizeof(info), cudaMemcpyDeviceToHost,
                              gpu.panel));
        CHECK(cudaStreamSynchronize(gpu.panel));
        return info;
    }

    for (int k = 0; k < rows.tileCount(); ++k) {
        const int width = rows.width(k), ownerRow = k % rows.parts, ownerCol = k % cols.parts;
        const int first = k + 1, rowStart = rows.offset(first);
        const int rowCount = ld - rowStart;
        int info = 0;
        double* diagonalTile = nullptr;
        if (grid.row == ownerRow && grid.col == ownerCol) {
            diagonalTile = matrix + elements(cols.offset(k), ld) + rows.offset(k);
            CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, width, diagonalTile,
                                  ld, work, workSize, deviceInfo));
            CHECK(cudaMemcpyAsync(&info, deviceInfo, sizeof(info), cudaMemcpyDeviceToHost,
                                  gpu.panel));
            if (rows.parts > 1 && first < rows.tileCount()) {
                copyMatrix(diagonal.data, width, diagonalTile, ld, width, width,
                           cudaMemcpyDeviceToHost, gpu.panel);
            }
            CHECK(cudaStreamSynchronize(gpu.panel));
            if (info > 0) info += k * rows.block;
        }
        CHECK(MPI_Bcast(&info, 1, MPI_INT, ownerRow * cols.parts + ownerCol, MPI_COMM_WORLD));
        if (info || first == rows.tileCount()) {
            gpu.synchronize();
            return info;
        }

        Panel& panel = k % 2 == 0 ? *even : *odd;
        // The previous use of this host slot may still be in flight.
        CHECK(cudaEventSynchronize(panel.ready));
        double* localPanel = nullptr;
        if (grid.col == ownerCol) {
            if (rows.parts > 1) {
                CHECK(MPI_Bcast(diagonal.data, width * width, MPI_DOUBLE, ownerRow, grid.colComm));
            }
            if (rowCount) {
                const double* triangular = diagonalTile;
                int triangularLd = ld;
                if (grid.row != ownerRow) {
                    // The bulk stream never reads this diagonal staging buffer.
                    copyMatrix(work, width, diagonal.data, width, width, width,
                               cudaMemcpyHostToDevice, gpu.panel);
                    triangular = work;
                    triangularLd = width;
                }
                localPanel = matrix + elements(cols.offset(k), ld) + rowStart;
                CHECK(cublasDtrsm(gpu.panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rowCount, width, &one,
                                  triangular, triangularLd, localPanel, ld));
                copyMatrix(panel.hostRows.data, rowCount, localPanel, ld, rowCount, width,
                           cudaMemcpyDeviceToHost, gpu.panel);
                CHECK(cudaStreamSynchronize(gpu.panel));
            }
        }

        exchangePanel(panel, grid, rows, cols, first, ownerCol, width, gpu, localPanel);
        // Update the next panel column first. Its factorization and exchange
        // can overlap the updates to the rest of the trailing matrix.
        if (first % cols.parts == grid.col) {
            CHECK(cudaStreamWaitEvent(gpu.panel, gpu.bulkDone, 0));
            updateTileColumn(matrix, rows, cols, panel, first, width, first,
                             &minusOne, gpu.panelBlas);
        }
        CHECK(cudaStreamWaitEvent(gpu.bulk, panel.ready, 0));
        for (int j = cols.first(first + 1); j < cols.tileCount(); j += cols.parts) {
            updateTileColumn(matrix, rows, cols, panel, first, width, j,
                             &minusOne, gpu.bulkBlas);
        }
        CHECK(cudaEventRecord(panel.consumed, gpu.bulk));
        CHECK(cudaEventRecord(gpu.bulkDone, gpu.bulk));
    }
    return 0;
}

std::vector<double> download(const double* matrix, const Axis& rows, const Axis& cols,
                             const Gpu& gpu) {
    const int ld = rows.length();
    std::vector<double> result(elements(ld, cols.length()));
    if (ld && cols.length()) {
        for (int j = cols.coord; j < cols.tileCount(); j += cols.parts) {
            const int row = rows.offset(j);
            if (row == ld) break;
            const size_t offset = elements(cols.offset(j), ld) + row;
            copyMatrix(result.data() + offset, ld, matrix + offset, ld,
                       ld - row, cols.width(j), cudaMemcpyDeviceToHost, gpu.panel);
        }
        CHECK(cudaStreamSynchronize(gpu.panel));

        #pragma omp parallel for schedule(static) if(cols.length() >= 1024)
        for (int j = 0; j < cols.length(); ++j) {
            const int globalCol = cols.globalIndex(j);
            const int tile = globalCol / cols.block;
            if (tile % rows.parts == rows.coord) {
                const int row = rows.offset(tile);
                std::fill_n(result.data() + elements(j, ld) + row, globalCol % cols.block, 0.0);
            }
        }
    }
    return result;
}

void printFactor(const std::vector<double>& local, const Grid& grid,
                 const Axis& rows, const Axis& cols) {
    // Move local rectangles in bounded messages, preserving the shared
    // result printer's original row-major layout and zero upper triangle.
    constexpr size_t messageSize = 1 << 20;
    if (grid.rank != 0) {
        for (size_t offset = 0; offset < local.size(); offset += messageSize) {
            const auto count = std::min(messageSize, local.size() - offset);
            CHECK(MPI_Send(local.data() + offset, static_cast<int>(count), MPI_DOUBLE,
                           0, 0, MPI_COMM_WORLD));
        }
        return;
    }

    std::vector<double> result(elements(rows.n, rows.n));
    std::vector<double> received;
    for (int rank = 0; rank < grid.size; ++rank) {
        const Axis sourceRows{rows.n, rows.block, rows.parts, rank / cols.parts};
        const Axis sourceCols{cols.n, cols.block, cols.parts, rank % cols.parts};
        const int ld = sourceRows.length(), colsLocal = sourceCols.length();
        if (!ld || !colsLocal) continue;
        const double* data = local.data();
        if (rank != 0) {
            received.resize(elements(ld, colsLocal));
            for (size_t offset = 0; offset < received.size(); offset += messageSize) {
                const auto count = std::min(messageSize, received.size() - offset);
                CHECK(MPI_Recv(received.data() + offset, static_cast<int>(count), MPI_DOUBLE,
                               rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            data = received.data();
        }

        #pragma omp parallel for schedule(static) if(elements(ld, colsLocal) >= 65536)
        for (int i = 0; i < ld; ++i) {
            const int row = sourceRows.globalIndex(i);
            for (int j = 0; j < colsLocal; ++j) {
                const int col = sourceCols.globalIndex(j);
                if (col > row) break;
                result[elements(row, rows.n) + col] = data[elements(j, ld) + i];
            }
        }
    }
    print_results(result, "CholeskyL");
}

bool validateFactor(double* matrix, const std::vector<double>& local,
                    const std::vector<double>& original, const Grid& grid,
                    const Axis& rows, const Axis& cols, const Gpu& gpu, Panel* panel) {
    const int ld = rows.length();
    if (rows.n && grid.size == 1) {
        DeviceBuffer<double> lower(local.size());
        CHECK(cudaMemcpyAsync(lower.data, local.data(), local.size() * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.panel));
        CHECK(cublasDsyrk(gpu.panelBlas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                          rows.n, rows.n, &one, lower.data, ld, &zero, matrix, ld));
        CHECK(cudaStreamSynchronize(gpu.panel));
    } else if (rows.n) {
        if (!local.empty()) {
            CHECK(cudaMemsetAsync(matrix, 0, local.size() * sizeof(double), gpu.panel));
        }
        for (int k = 0; k < rows.tileCount(); ++k) {
            const int width = rows.width(k), row = rows.offset(k);
            const int count = ld - row;
            CHECK(cudaEventSynchronize(panel->ready));
            if (grid.col == k % cols.parts && count) {
                const double* source = local.data() + elements(cols.offset(k), ld) + row;
                #pragma omp parallel for schedule(static) if(elements(count, width) >= 65536)
                for (int j = 0; j < width; ++j) {
                    std::copy_n(source + elements(j, ld), count,
                                panel->hostRows.data + elements(j, count));
                }
            }
            exchangePanel(*panel, grid, rows, cols, k, k % cols.parts, width, gpu);
            for (int j = cols.first(k); j < cols.tileCount(); j += cols.parts) {
                updateTileColumn(matrix, rows, cols, *panel, k, width, j, &one, gpu.panelBlas);
            }
        }
        CHECK(cudaStreamSynchronize(gpu.panel));
    }

    const auto reconstructed = download(matrix, rows, cols, gpu);
    double absError = 0.0, relError = 0.0;
    if (ld) {
        #pragma omp parallel for schedule(static) reduction(max:absError,relError) if(local.size() >= 65536)
        for (int j = 0; j < cols.length(); ++j) {
            const int globalCol = cols.globalIndex(j), tile = globalCol / cols.block;
            int row = rows.offset(tile);
            if (tile % rows.parts == rows.coord) row += globalCol % cols.block;
            for (int i = row; i < ld; ++i) {
                const size_t index = elements(j, ld) + i;
                const double error = std::abs(reconstructed[index] - original[index]);
                if (!std::isfinite(error)) {
                    absError = relError = std::numeric_limits<double>::infinity();
                    continue;
                }
                absError = std::max(absError, error);
                relError = std::max(relError, error / (std::abs(original[index]) + 1e-10));
            }
        }
    }

    const double errors[2] = {absError, relError};
    double maxima[2];
    CHECK(MPI_Allreduce(errors, maxima, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (grid.rank == 0) {
        printf("Max absolute error: %.10e\n", maxima[0]);
        printf("Max relative error: %.10e\n", maxima[1]);
        if (maxima[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return maxima[1] <= 1e-6;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end;
            errno = 0;
            const long size = strtol(argument, &end, 10);
            if (end == argument || *end || errno || size < 0 || size >= INT_MAX) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return EXIT_FAILURE;
            }
            n = static_cast<int>(size);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return EXIT_FAILURE;
        }
    }

    const Grid grid;
    const Gpu gpu;
    // Use enough blocks to spread small matrices across the process grid.
    const int block = std::clamp((n / std::max(grid.dims[0], grid.dims[1]) / 2) / 32 * 32,
                                 64, 256);
    const Axis rows{n, block, grid.dims[0], grid.row};
    const Axis cols{n, block, grid.dims[1], grid.col};
    DeviceBuffer<double> matrix(elements(rows.length(), cols.length()));
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    if (matrix.data) {
        CHECK(cudaMemsetAsync(matrix.data, 0, elements(rows.length(), cols.length()) * sizeof(double),
                              gpu.panel));
    }
    generateMatrix(matrix.data, rows, cols, gpu);
    const auto original = validate ? download(matrix.data, rows, cols, gpu) : std::vector<double>{};

    const bool distributed = n && grid.size > 1;
    const int diagonalSize = distributed ? std::min(n, block) : n;
    DeviceBuffer<double> queryDummy(n && !matrix.data ? 1 : 0);
    DeviceBuffer<int> deviceInfo(n ? 1 : 0);
    int workSize = 0;
    if (n) {
        CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, diagonalSize,
                                         matrix.data ? matrix.data : queryDummy.data,
                                         std::max(rows.length(), diagonalSize), &workSize));
    }
    DeviceBuffer<double> work(std::max(static_cast<size_t>(workSize),
                                      distributed ? elements(diagonalSize, diagonalSize) : 0));
    PinnedBuffer diagonal(distributed && rows.parts > 1 ? elements(diagonalSize, diagonalSize) : 0);
    // A single rank calls the full-matrix CUDA factorization directly.
    const Axis empty{0, block, 1, 0};
    Panel even(distributed ? rows : empty, distributed ? cols : empty);
    Panel odd(distributed ? rows : empty, distributed ? cols : empty);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const int info = n ? factor(matrix.data, grid, rows, cols, gpu, work.data, workSize,
                                deviceInfo.data, diagonal, &even, &odd) : 0;
    const double elapsed = MPI_Wtime() - start;
    double duration;
    CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (info) {
        if (rank == 0) {
            if (info > 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
            } else {
                fprintf(stderr, "CUDA Cholesky reported an invalid argument (%d)\n", -info);
            }
            printf("Cholesky decomposition failed\n");
        }
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
    }

    const auto local = printResults || validate ? download(matrix.data, rows, cols, gpu)
                                               : std::vector<double>{};
    if (printResults) printFactor(local, grid, rows, cols);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateFactor(matrix.data, local, original, grid, rows, cols, gpu, &even);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required", provided);
    int rank;
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int status = EXIT_FAILURE;
    try {
        status = run(argc, argv, rank);
    } catch (const std::exception& error) {
        fail(error.what(), 1);
    }
    return MPI_Finalize() == MPI_SUCCESS ? status : EXIT_FAILURE;
}
