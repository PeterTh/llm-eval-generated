#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void fail(const char* operation, const char* detail) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char detail[32];
        snprintf(detail, sizeof(detail), "cuBLAS status %d", static_cast<int>(status));
        fail(operation, detail);
    }
}

void check(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char detail[32];
        snprintf(detail, sizeof(detail), "cuSOLVER status %d", static_cast<int>(status));
        fail(operation, detail);
    }
}

template<class T, bool pinned = false>
class Buffer {
public:
    explicit Buffer(size_t count) {
        if (count == 0) return;
        if (count > SIZE_MAX / sizeof(T)) fail("CUDA allocation", "buffer size overflow");
        if constexpr (pinned) {
            check(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(T)), "cudaMallocHost");
        } else {
            check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "cudaMalloc");
        }
    }

    ~Buffer() {
        if constexpr (pinned) {
            if (data) cudaFreeHost(data);
        } else {
            if (data) cudaFree(data);
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    T* data = nullptr;
};

void copyMatrix(double* to, int toLd, const double* from, int fromLd,
                int rows, int cols, cudaMemcpyKind kind, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;
    check(cudaMemcpy2DAsync(to, size_t(toLd) * sizeof(double),
                            from, size_t(fromLd) * sizeof(double),
                            size_t(rows) * sizeof(double), cols, kind, stream),
          "cudaMemcpy2DAsync");
}

int threadsFor(size_t elements, size_t grain = 16384) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + elements / grain));
}

struct Grid {
    Grid(int rank, int ranks, int tiles) {
        // Keep the grid close to square even when the rank count is prime.
        // When tiles are scarce, grid dimensions cannot exceed their count.
        for (int r = 1; r <= std::max(1, std::min(tiles, ranks)); ++r) {
            const int limit = ranks >= 4 ? std::min(ranks / r, 2 * r) : ranks / r;
            const int c = std::max(1, std::min(tiles, limit));
            if (c < r) break;
            if (r * c > rows * cols ||
                (r * c == rows * cols && std::abs(r - c) < std::abs(rows - cols))) {
                rows = r;
                cols = c;
            }
        }
        MPI_Comm_split(MPI_COMM_WORLD, rank < rows * cols ? 0 : MPI_UNDEFINED, rank, &all);
        if (all == MPI_COMM_NULL) return;
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(all, row, col, &acrossRow);
        MPI_Comm_split(all, col, row, &downCol);
    }

    ~Grid() {
        if (all == MPI_COMM_NULL) return;
        MPI_Comm_free(&acrossRow);
        MPI_Comm_free(&downCol);
        MPI_Comm_free(&all);
    }

    bool single() const { return rows == 1 && cols == 1; }

    int rows = 1;
    int cols = 1;
    int row = 0;
    int col = 0;
    MPI_Comm all = MPI_COMM_NULL;
    MPI_Comm acrossRow = MPI_COMM_NULL;
    MPI_Comm downCol = MPI_COMM_NULL;
};

struct Axis {
    Axis(int dimension, int block, int processes, int coordinate)
        : n(dimension), tile(block), stride(processes), coord(coordinate) {
        const int full = n / tile;
        const int complete = (full + stride - 1 - coord) / stride;
        size = complete * tile + (full % stride == coord ? n % tile : 0);
    }

    int start(int globalTile) const {
        const int localTiles = (globalTile + stride - 1 - coord) / stride;
        return static_cast<int>(std::min<size_t>(size, size_t(localTiles) * tile));
    }

    int globalTile(int local) const { return local / tile * stride + coord; }
    int globalIndex(int local) const { return globalTile(local) * tile + local % tile; }

    int n;
    int tile;
    int stride;
    int coord;
    int size;
};

class Gpu {
public:
    explicit Gpu(const Grid& grid) {
        MPI_Comm local;
        MPI_Comm_split_type(grid.all, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices == 0) fail("CUDA", "no visible GPU");
        check(cudaSetDevice(localRank % devices), "cudaSetDevice");
        check(cudaStreamCreateWithPriority(&panel, cudaStreamNonBlocking, highestPriority()), "create panel stream");
        check(cudaStreamCreateWithFlags(&bulk, cudaStreamNonBlocking), "create update stream");
        check(cublasCreate(&panelBlas), "create panel cuBLAS handle");
        check(cublasCreate(&bulkBlas), "create update cuBLAS handle");
        check(cublasSetStream(panelBlas, panel), "set panel cuBLAS stream");
        check(cublasSetStream(bulkBlas, bulk), "set update cuBLAS stream");
        check(cusolverDnCreate(&solver), "create cuSOLVER handle");
        check(cusolverDnSetStream(solver, panel), "set cuSOLVER stream");
    }

    ~Gpu() {
        cusolverDnDestroy(solver);
        cublasDestroy(bulkBlas);
        cublasDestroy(panelBlas);
        cudaStreamDestroy(bulk);
        cudaStreamDestroy(panel);
    }

    cudaStream_t panel;
    cudaStream_t bulk;
    cublasHandle_t panelBlas;
    cublasHandle_t bulkBlas;
    cusolverDnHandle_t solver;

private:
    static int highestPriority() {
        int priority = 0;
        check(cudaDeviceGetStreamPriorityRange(nullptr, &priority), "query CUDA stream priority");
        return priority;
    }
};

struct Event {
    Event() { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create CUDA event"); }
    ~Event() { cudaEventDestroy(event); }

    void record(cudaStream_t stream) { check(cudaEventRecord(event, stream), "record CUDA event"); }
    void waitOn(cudaStream_t stream) const {
        check(cudaStreamWaitEvent(stream, event, 0), "wait for CUDA event on stream");
    }
    void synchronize() const { check(cudaEventSynchronize(event), "synchronize CUDA event"); }

    cudaEvent_t event;
};

constexpr int randomChunk = 2048;

// glibc advances its 32-bit rand_r state three times per output. Compose
// the affine state transitions to jump straight to any row and column.
unsigned int seedAt(uint64_t offset) {
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    uint32_t seed = 42;
    for (uint64_t steps = 3 * offset; steps; steps >>= 1) {
        if (steps & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return seed;
}

void randomRows(double* buffer, const Axis& axis, int first, int width,
                const std::vector<unsigned int>& rowSeeds) {
    #pragma omp parallel for schedule(static) num_threads(threadsFor(size_t(axis.size) * width))
    for (int local = 0; local < axis.size; ++local) {
        const int row = axis.globalIndex(local);
        unsigned int seed;
        if (rowSeeds.empty()) {
            seed = seedAt(uint64_t(row) * axis.n + first);
        } else {
            seed = rowSeeds[size_t(first / randomChunk) * axis.n + row];
        }
        double* output = buffer + size_t(local) * width;
        for (int j = 0; j < width; ++j) output[j] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
}

void generateMatrix(double* matrix, const Axis& rows, const Axis& cols, const Gpu& gpu) {
    const int n = rows.n;
    if (!rows.size || !cols.size) return;
    const int capacity = std::min(n, randomChunk);
    const bool same = rows.stride == cols.stride && rows.coord == cols.coord;
    Buffer<double, true> hostRows(size_t(rows.size) * capacity);
    Buffer<double, true> hostCols(same ? 0 : size_t(cols.size) * capacity);
    Buffer<double> randomRowData(size_t(rows.size) * capacity);
    Buffer<double> randomColData(same ? 0 : size_t(cols.size) * capacity);
    std::vector<unsigned int> rowSeeds;
#ifndef __GLIBC__
    // On other libcs, discover the start of each chunk through rand_r itself.
    rowSeeds.resize(size_t((n - 1) / randomChunk + 1) * n);
    unsigned int seed = 42;
    for (int row = 0; row < n; ++row) {
        for (int j = 0; j < n; ++j) {
            if (j % randomChunk == 0) rowSeeds[size_t(j / randomChunk) * n + row] = seed;
            rand_r(&seed);
        }
    }
#endif

    check(cudaMemsetAsync(matrix, 0, size_t(rows.size) * cols.size * sizeof(double), gpu.panel),
          "initialize matrix storage");
    const double one = 1.0;
    Event uploaded;
    for (int first = 0; first < n;) {
        const int width = std::min(randomChunk, n - first);
        if (first) uploaded.synchronize();
        randomRows(hostRows.data, rows, first, width, rowSeeds);
        if (!same) randomRows(hostCols.data, cols, first, width, rowSeeds);
        copyMatrix(randomRowData.data, width, hostRows.data, width, width, rows.size,
                   cudaMemcpyHostToDevice, gpu.panel);
        if (!same) {
            copyMatrix(randomColData.data, width, hostCols.data, width, width, cols.size,
                       cudaMemcpyHostToDevice, gpu.panel);
        }
        uploaded.record(gpu.panel);
        const double beta = first ? 1.0 : 0.0;
        if (same) {
            check(cublasDsyrk(gpu.panelBlas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, rows.size,
                              width, &one, randomRowData.data, width, &beta, matrix, rows.size),
                  "generate symmetric B * B^T");
        } else {
            for (int col = 0; col < cols.size;) {
                const int row = rows.start(cols.globalTile(col));
                const int height = rows.size - row;
                const int columns = std::min(4 * cols.tile, cols.size - col);
                if (height) {
                    check(cublasDgemm(gpu.panelBlas, CUBLAS_OP_T, CUBLAS_OP_N, height, columns,
                                      width, &one, randomRowData.data + size_t(row) * width, width,
                                      randomColData.data + size_t(col) * width, width, &beta,
                                      matrix + size_t(col) * rows.size + row, rows.size),
                          "generate lower B * B^T");
                }
                col += columns;
            }
        }
        first += width;
    }

    // Add n to the diagonal of the generated matrix.
    std::vector<double> ones(rows.tile, 1.0);
    Buffer<double> deviceOnes(ones.size());
    check(cudaMemcpy(deviceOnes.data, ones.data(), ones.size() * sizeof(double),
                     cudaMemcpyHostToDevice), "copy diagonal increment");
    const double shift = n;
    for (int local = 0; local < cols.size; local += std::min(cols.tile, cols.size - local)) {
        const int global = cols.globalTile(local);
        if (global % rows.stride == rows.coord) {
            const int width = std::min(cols.tile, cols.size - local);
            check(cublasDaxpy(gpu.panelBlas, width, &shift, deviceOnes.data, 1,
                              matrix + size_t(local) * rows.size + rows.start(global),
                              rows.size + 1), "shift matrix diagonal");
        }
    }
    check(cudaStreamSynchronize(gpu.panel), "finish matrix generation");
}

struct Panel {
    const double* left = nullptr;
    const double* right = nullptr;
    int leftLd = 0;
    int rightLd = 0;
};

class Panels {
public:
    Panels(const Grid& grid, const Axis& rows, const Axis& cols, const Gpu& gpu)
        : grid(grid), rows(rows), cols(cols), gpu(gpu),
          rowCapacity(grid.single() ? 0 : size_t(rows.size) * rows.tile),
          colCapacity(grid.single() ? 0 : size_t(cols.size) * cols.tile),
          rowHost(grid.single() ? 0 : rowCapacity + 1), colHost(colCapacity),
          sendHost(grid.rows == grid.cols ? 0 : colCapacity),
          receiveHost(grid.rows == grid.cols ? 0 : colCapacity),
          rowDevice(2 * rowCapacity), colDevice(2 * colCapacity),
          counts(grid.rows), displacements(grid.rows) {
        if (rowCapacity >= INT_MAX || colCapacity > INT_MAX) {
            fail("MPI panel", "local panel exceeds the supported MPI message size");
        }
    }

    // The owning process column sends a row panel along each process row.
    // Redistribute the corresponding tile rows down each process column.
    // A status is carried with the first broadcast to propagate POTRF failures.
    int exchange(const double* matrix, int k, int first, int width, int info, Panel& result) {
        const int rowStart = rows.start(first);
        const int colStart = cols.start(first);
        const int m = rows.size - rowStart;
        const int n = cols.size - colStart;
        if (grid.single()) {
            if (m) {
                const double* local = matrix + size_t(cols.start(k)) * rows.size + rowStart;
                result = {local, local, rows.size, rows.size};
            }
            return info;
        }

        ready.synchronize();
        if (grid.col == k % grid.cols) {
            copyMatrix(rowHost.data + 1, m, matrix + size_t(cols.start(k)) * rows.size + rowStart,
                       rows.size, m, width, cudaMemcpyDeviceToHost, gpu.panel);
            if (m) check(cudaStreamSynchronize(gpu.panel), "download solved panel");
            rowHost.data[0] = info;
        }
        if (grid.cols > 1) {
            MPI_Bcast(rowHost.data, 1 + m * width, MPI_DOUBLE, k % grid.cols, grid.acrossRow);
        }
        info = static_cast<int>(rowHost.data[0]);
        if (info != 0 || (first * int64_t(rows.tile)) >= rows.n) return info;

        const int slot = k % 2;
        double* left = rowCapacity ? rowDevice.data + slot * rowCapacity : nullptr;
        double* right = colCapacity ? colDevice.data + slot * colCapacity : nullptr;
        copyMatrix(left, m, rowHost.data + 1, m, m, width, cudaMemcpyHostToDevice, gpu.panel);

        if (grid.rows == grid.cols) {
            double* host = grid.row == grid.col ? rowHost.data + 1 : colHost.data;
            if (n) MPI_Bcast(host, n * width, MPI_DOUBLE, grid.col, grid.downCol);
            if (grid.row == grid.col) right = left;
            else copyMatrix(right, n, host, n, n, width, cudaMemcpyHostToDevice, gpu.panel);
        } else if (n) {
            redistribute(rowStart, colStart, m, n, width);
            copyMatrix(right, n, colHost.data, n, n, width, cudaMemcpyHostToDevice, gpu.panel);
        }

        result = {left, right, m, n};
        ready.record(gpu.panel);
        return 0;
    }

    Event ready;

private:
    struct Slice {
        int offset;
        int global;
        int size;
        int packedOffset;
    };

    void redistribute(int rowStart, int colStart, int m, int n, int width) {
        std::fill(counts.begin(), counts.end(), 0);
        slices.clear();
        for (int local = colStart; local < cols.size;) {
            const int global = cols.globalTile(local);
            const int size = std::min(cols.tile, cols.size - local);
            const int source = global % grid.rows;
            slices.push_back({local - colStart, global, size, counts[source]});
            counts[source] += size * width;
            local += size;
        }
        std::exclusive_scan(counts.begin(), counts.end(), displacements.begin(), 0);
        const int workers = threadsFor(size_t(n) * width);
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (size_t i = 0; i < slices.size(); ++i) {
            const Slice& slice = slices[i];
            if (slice.global % grid.rows != grid.row) continue;
            const int from = rows.start(slice.global) - rowStart;
            for (int j = 0; j < width; ++j) {
                std::memcpy(sendHost.data + slice.packedOffset + size_t(j) * slice.size,
                            rowHost.data + 1 + size_t(j) * m + from,
                            size_t(slice.size) * sizeof(double));
            }
        }
        const double* packed = sendHost.data;
        if (grid.rows > 1) {
            MPI_Allgatherv(sendHost.data, counts[grid.row], MPI_DOUBLE, receiveHost.data,
                           counts.data(), displacements.data(), MPI_DOUBLE, grid.downCol);
            packed = receiveHost.data;
        }
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (size_t i = 0; i < slices.size(); ++i) {
            const Slice& slice = slices[i];
            const int source = slice.global % grid.rows;
            const double* from = packed + displacements[source] + slice.packedOffset;
            for (int j = 0; j < width; ++j) {
                std::memcpy(colHost.data + size_t(j) * n + slice.offset,
                            from + size_t(j) * slice.size, size_t(slice.size) * sizeof(double));
            }
        }
    }

    const Grid& grid;
    const Axis& rows;
    const Axis& cols;
    const Gpu& gpu;
    size_t rowCapacity;
    size_t colCapacity;
    Buffer<double, true> rowHost;
    Buffer<double, true> colHost;
    Buffer<double, true> sendHost;
    Buffer<double, true> receiveHost;
    Buffer<double> rowDevice;
    Buffer<double> colDevice;
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<Slice> slices;
};

// Group adjacent local tile columns into GEMMs. Skip the upper triangle
// above each group and ignore the unused upper entries inside the group.
void update(double* matrix, const Axis& rows, const Axis& cols, const Panel& panel,
            int first, int begin, int end, int width, int group, double alpha,
            cublasHandle_t blas) {
    if (!rows.size || !cols.size) return;
    const int rowBase = rows.start(first);
    const int colBase = cols.start(first);
    const int colEnd = cols.start(end);
    for (int col = cols.start(begin); col < colEnd;) {
        const int row = rows.start(cols.globalTile(col));
        const int m = rows.size - row;
        const int n = std::min(group * cols.tile, colEnd - col);
        const double beta = 1.0;
        if (m) {
            check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, width, &alpha,
                              panel.left + (row - rowBase), panel.leftLd,
                              panel.right + (col - colBase), panel.rightLd, &beta,
                              matrix + size_t(col) * rows.size + row, rows.size),
                  "update trailing matrix");
        }
        col += n;
    }
}

class Factorization {
public:
    Factorization(double* matrix, const Grid& grid, const Axis& rows, const Axis& cols,
                  const Gpu& gpu, int tiles)
        : matrix(matrix), grid(grid), rows(rows), cols(cols), gpu(gpu), tiles(tiles),
          direct(grid.single() && rows.n <= 1024), panels(grid, rows, cols, gpu),
          diagHost(grid.rows == 1 ? 1 : 1 + size_t(rows.tile) * rows.tile),
          diagDevice(grid.rows == 1 ? 0 : size_t(rows.tile) * rows.tile), devInfo(1),
          workspace(workspaceSize()) {}

    void warmUp() const {
        const int width = std::min(rows.tile, rows.n);
        if (!width) return;
        const size_t elements = size_t(width) * width;
        std::vector<double> identity(elements, 0.0);
        for (int j = 0; j < width; ++j) identity[size_t(j) * (width + 1)] = 1.0;
        Buffer<double> diagonal(elements);
        Buffer<double> scratch(elements);
        check(cudaMemcpy(diagonal.data, identity.data(), elements * sizeof(double),
                         cudaMemcpyHostToDevice), "initialize accelerator warm-up");
        int required = 0;
        check(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, width, diagonal.data,
                                          width, &required), "query warm-up workspace");
        Buffer<double> work(required);
        check(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, width, diagonal.data, width,
                               work.data, required, devInfo.data), "warm up POTRF");
        const double one = 1.0;
        const double zero = 0.0;
        check(cublasDgemm(gpu.panelBlas, CUBLAS_OP_N, CUBLAS_OP_T, width, width, width,
                          &one, diagonal.data, width, diagonal.data, width, &zero, scratch.data, width),
              "warm up panel GEMM");
        check(cublasDtrsm(gpu.panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, width, width, &one,
                          diagonal.data, width, scratch.data, width), "warm up TRSM");
        check(cudaStreamSynchronize(gpu.panel), "finish panel warm-up");
        check(cublasDgemm(gpu.bulkBlas, CUBLAS_OP_N, CUBLAS_OP_T, width, width, width,
                          &one, diagonal.data, width, diagonal.data, width, &zero, scratch.data, width),
              "warm up trailing GEMM");
        check(cudaStreamSynchronize(gpu.bulk), "finish update warm-up");
    }

    int run() {
        // For small single-GPU problems, cuSOLVER can factor the local matrix in one call.
        if (direct) {
            if (!tiles) return 0;
            factorDiagonal(rows.n, matrix);
            return hostInfo;
        }
        for (int k = 0; k < tiles; ++k) {
            const int width = std::min(rows.tile, rows.n - k * rows.tile);
            const int info = factorPanel(k, width);
            Panel panel;
            const int status = panels.exchange(matrix, k, k + 1, width, info, panel);
            if (status != 0) return status > 0 ? k * rows.tile + status : status;
            if (k + 1 == tiles) break;

            // Update the next panel first. The bulk stream can work on later
            // columns while the panel stream factors and exchanges that panel.
            if (grid.single()) panels.ready.record(gpu.panel);
            bulkDone.waitOn(gpu.panel);
            update(matrix, rows, cols, panel, k + 1, k + 1, k + 2, width, 1, -1.0, gpu.panelBlas);
            panels.ready.waitOn(gpu.bulk);
            update(matrix, rows, cols, panel, k + 1, k + 2, tiles, width, 4, -1.0, gpu.bulkBlas);
            bulkDone.record(gpu.bulk);
        }
        synchronize();
        return 0;
    }

    void reconstruct(double* output) {
        // Mask each diagonal tile's upper entries before using L in GEMMs.
        Buffer<double, true> tile(size_t(rows.tile) * rows.tile);
        for (int k = 0; k < tiles; ++k) {
            if (grid.row != k % grid.rows || grid.col != k % grid.cols) continue;
            const int width = std::min(rows.tile, rows.n - k * rows.tile);
            double* diagonal = matrix + size_t(cols.start(k)) * rows.size + rows.start(k);
            copyMatrix(tile.data, width, diagonal, rows.size, width, width, cudaMemcpyDeviceToHost, gpu.panel);
            check(cudaStreamSynchronize(gpu.panel), "download diagonal factor");
            for (int j = 0; j < width; ++j) std::fill_n(tile.data + size_t(j) * width, j, 0.0);
            copyMatrix(diagonal, rows.size, tile.data, width, width, width, cudaMemcpyHostToDevice, gpu.panel);
        }
        for (int k = 0; k < tiles; ++k) {
            const int width = std::min(rows.tile, rows.n - k * rows.tile);
            Panel panel;
            panels.exchange(matrix, k, k, width, 0, panel);
            update(output, rows, cols, panel, k, k, tiles, width, 4, 1.0, gpu.panelBlas);
        }
        check(cudaStreamSynchronize(gpu.panel), "finish reconstruction");
    }

    void synchronize() const {
        check(cudaStreamSynchronize(gpu.panel), "finish panel stream");
        check(cudaStreamSynchronize(gpu.bulk), "finish update stream");
    }

private:
    int diagonalWorkspace(int width) const {
        int required = 0;
        check(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, width, matrix,
                                          rows.size, &required), "query POTRF workspace");
        return required;
    }

    int workspaceSize() const {
        if (direct && tiles) return diagonalWorkspace(rows.n);
        int size = 0;
        // Any owned diagonal tile has the same leading dimension.
        for (int k = 0; k < tiles; ++k) {
            if (grid.row != k % grid.rows || grid.col != k % grid.cols) continue;
            const int width = std::min(rows.tile, rows.n - k * rows.tile);
            size = std::max(size, diagonalWorkspace(width));
            if (width == rows.tile && (k + 1 == tiles || rows.n % rows.tile == 0)) break;
        }
        return size;
    }

    void factorDiagonal(int width, double* local, bool downloadTile = false) {
        check(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, width, local, rows.size,
                               workspace.data, workspace.size, devInfo.data), "factor diagonal tile");
        if (downloadTile) {
            copyMatrix(diagHost.data + 1, width, local, rows.size, width, width,
                       cudaMemcpyDeviceToHost, gpu.panel);
        }
        check(cudaMemcpyAsync(&hostInfo, devInfo.data, sizeof(int), cudaMemcpyDeviceToHost,
                              gpu.panel), "download POTRF status");
        check(cudaStreamSynchronize(gpu.panel), "finish diagonal factorization");
    }

    int factorPanel(int k, int width) {
        if (grid.col != k % grid.cols) return 0;
        const int root = k % grid.rows;
        const int row = rows.start(k);
        const int col = cols.start(k);
        const int below = rows.start(k + 1);
        const int height = rows.size - below;
        const bool last = k + 1 == tiles;
        const double* diagonal = diagDevice.data;
        int ld = width;

        if (grid.row == root) {
            double* local = matrix + size_t(col) * rows.size + row;
            factorDiagonal(width, local, grid.rows > 1 && !last);
            diagonal = local;
            ld = rows.size;
            diagHost.data[0] = hostInfo;
        }

        if (grid.rows > 1) {
            MPI_Bcast(diagHost.data, last ? 1 : 1 + width * width, MPI_DOUBLE, root, grid.downCol);
        }
        const int info = static_cast<int>(diagHost.data[0]);
        if (info == 0 && height > 0) {
            if (grid.row != root) {
                copyMatrix(diagDevice.data, width, diagHost.data + 1, width, width, width,
                           cudaMemcpyHostToDevice, gpu.panel);
            }
            const double one = 1.0;
            check(cublasDtrsm(gpu.panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                             CUBLAS_DIAG_NON_UNIT, height, width, &one, diagonal, ld,
                             matrix + size_t(col) * rows.size + below, rows.size), "solve panel");
        }
        return info;
    }

    double* matrix;
    const Grid& grid;
    const Axis& rows;
    const Axis& cols;
    const Gpu& gpu;
    int tiles;
    bool direct;
    Panels panels;
    Buffer<double, true> diagHost;
    Buffer<double> diagDevice;
    Event bulkDone;
    Buffer<int> devInfo;
    int hostInfo = 0;

    struct Workspace : Buffer<double> {
        explicit Workspace(int count) : Buffer<double>(count), size(count) {}
        int size;
    } workspace;
};

void collectResults(double* matrix, const Axis& rows, const Axis& cols, const Grid& grid,
                    const Gpu& gpu, int rank) {
    // Transfer bounded column slabs to root and restore the original row-major layout.
    constexpr int maxSlab = 1 << 20;
    const int mySlabCols = std::max(1, maxSlab / std::max(1, rows.size));
    Buffer<double, true> staging(rank == 0 ? std::max(maxSlab, rows.size) :
                                 size_t(rows.size) * std::min(cols.size, mySlabCols));
    if (rank != 0) {
        for (int j = 0; j < cols.size; j += mySlabCols) {
            const int width = std::min(mySlabCols, cols.size - j);
            copyMatrix(staging.data, rows.size, matrix + size_t(j) * rows.size, rows.size,
                       rows.size, width, cudaMemcpyDeviceToHost, gpu.panel);
            check(cudaStreamSynchronize(gpu.panel), "download results");
            MPI_Send(staging.data, rows.size * width, MPI_DOUBLE, 0, 0, grid.all);
        }
        return;
    }

    std::vector<double> result(size_t(rows.n) * rows.n, 0.0);
    for (int source = 0; source < grid.rows * grid.cols; ++source) {
        const Axis srcRows(rows.n, rows.tile, rows.stride, source / grid.cols);
        const Axis srcCols(cols.n, cols.tile, cols.stride, source % grid.cols);
        const int widthLimit = std::max(1, maxSlab / std::max(1, srcRows.size));
        if (srcRows.size == 0) continue;
        for (int j = 0; j < srcCols.size; j += widthLimit) {
            const int width = std::min(widthLimit, srcCols.size - j);
            if (source == 0) {
                copyMatrix(staging.data, rows.size, matrix + size_t(j) * rows.size, rows.size,
                           rows.size, width, cudaMemcpyDeviceToHost, gpu.panel);
                check(cudaStreamSynchronize(gpu.panel), "download root results");
            } else {
                MPI_Recv(staging.data, srcRows.size * width, MPI_DOUBLE, source, 0, grid.all, MPI_STATUS_IGNORE);
            }
            #pragma omp parallel for schedule(static) num_threads(threadsFor(size_t(srcRows.size) * width))
            for (int i = 0; i < srcRows.size; ++i) {
                const int row = srcRows.globalIndex(i);
                for (int local = 0; local < width; ++local) {
                    const int col = srcCols.globalIndex(j + local);
                    if (col > row) break;
                    result[size_t(row) * rows.n + col] = staging.data[size_t(local) * srcRows.size + i];
                }
            }
        }
    }
    print_results(result, "CholeskyL");
}

bool validateFactor(Factorization& factor, const std::vector<double>& original, const Axis& rows,
                    const Axis& cols, const Grid& grid, const Gpu& gpu, int rank) {
    const size_t elements = size_t(rows.size) * cols.size;
    Buffer<double> reconstructed(elements);
    if (elements) check(cudaMemsetAsync(reconstructed.data, 0, elements * sizeof(double), gpu.panel),
                        "initialize reconstruction");
    factor.reconstruct(reconstructed.data);
    const int slab = std::max(1, (1 << 20) / std::max(1, rows.size));
    Buffer<double, true> host(size_t(rows.size) * std::min(slab, cols.size));
    double maxError = 0.0;
    double relError = 0.0;
    for (int col = 0; rows.size && col < cols.size; col += slab) {
        const int width = std::min(slab, cols.size - col);
        copyMatrix(host.data, rows.size, reconstructed.data + size_t(col) * rows.size, rows.size,
                   rows.size, width, cudaMemcpyDeviceToHost, gpu.panel);
        check(cudaStreamSynchronize(gpu.panel), "download reconstruction");
        #pragma omp parallel for schedule(static) reduction(max:maxError,relError) num_threads(threadsFor(size_t(rows.size) * width))
        for (int j = 0; j < width; ++j) {
            const int globalCol = cols.globalIndex(col + j);
            const int from = rows.start(globalCol / rows.tile);
            for (int i = from; i < rows.size; ++i) {
                if (rows.globalIndex(i) < globalCol) continue;
                const double reference = original[size_t(col + j) * rows.size + i];
                const double error = std::abs(host.data[size_t(j) * rows.size + i] - reference);
                const double relative = std::isfinite(error) ? error / (std::abs(reference) + 1e-10) : INFINITY;
                maxError = std::max(maxError, std::isfinite(error) ? error : double(INFINITY));
                relError = std::max(relError, relative);
            }
        }
    }
    const double local[] = {maxError, relError};
    double errors[2];
    MPI_Allreduce(local, errors, 2, MPI_DOUBLE, MPI_MAX, grid.all);
    const bool valid = errors[1] <= 1e-6;
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (!valid) printf("Validation failed: relative error too large\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const long size = std::strtol(value, &end, 10);
            if (errno || end == value || *end || size < 0 || size >= INT_MAX) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<int>(size);
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

    const int tile = n < 2048 ? 128 : 256;
    const int tiles = n / tile + (n % tile != 0);
    Grid grid(rank, ranks, tiles);
    if (grid.all == MPI_COMM_NULL) return 0;
    Gpu gpu(grid);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    const Axis rows(n, tile, grid.rows, grid.row);
    const Axis cols(n, tile, grid.cols, grid.col);
    const size_t elements = size_t(rows.size) * cols.size;
    Buffer<double> matrix(elements);
    generateMatrix(matrix.data, rows, cols, gpu);
    std::vector<double> original;
    if (validate && elements) {
        original.resize(elements);
        check(cudaMemcpy(original.data(), matrix.data, elements * sizeof(double),
                         cudaMemcpyDeviceToHost), "save original matrix");
    }
    Factorization factor(matrix.data, grid, rows, cols, gpu, tiles);
    factor.warmUp();

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(grid.all);
    const double start = MPI_Wtime();
    const int status = factor.run();
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid.all);

    if (status != 0) {
        factor.synchronize();
        if (rank == 0) {
            if (status > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", status - 1);
            else fprintf(stderr, "Invalid POTRF argument %d\n", -status);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? double(n) * n * n / (3e9 * seconds) : 0.0);
    }
    if (printResults) collectResults(matrix.data, rows, cols, grid, gpu, rank);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        if (!validateFactor(factor, original, rows, cols, grid, gpu, rank)) return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_Init_thread", "MPI_THREAD_FUNNELED is required");
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail("Cholesky benchmark", error.what());
    }
    MPI_Finalize();
    return result;
}
