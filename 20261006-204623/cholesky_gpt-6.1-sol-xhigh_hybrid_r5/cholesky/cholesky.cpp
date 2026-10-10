#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED). Matrices
// remain on GPUs in a 2-D block-cyclic, column-major layout. OpenMP produces
// the original random rows and packs the pinned MPI staging buffers.
namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(status));
}

void check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER status " + std::to_string(status));
}

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string("MPI: ") + std::string(message, length));
    }
}

template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("Device allocation size overflow");
        check(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template<class T> class PinnedBuffer {
public:
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("Host allocation size overflow");
        check(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

// Chunk large transfers instead of truncating MPI's int counts.
void broadcast(double* buffer, size_t count, int root, MPI_Comm comm) {
    constexpr size_t chunk = 1u << 26;
    for (size_t offset = 0; offset < count; offset += chunk)
        mpiCheck(MPI_Bcast(buffer + offset, static_cast<int>(std::min(chunk, count - offset)),
                           MPI_DOUBLE, root, comm));
}

int prefixBlocks(int end, int coordinate, int stride) {
    return end <= coordinate ? 0 : 1 + (end - 1 - coordinate) / stride;
}

struct Grid {
    int rank, size, rows, cols, row, col;
    int n, block, tiles, localRows, localCols, ld;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;

    Grid(int dimension, int processRank, int processes) : rank(processRank), size(processes), n(dimension) {
        int dims[2] = {0, 0};
        mpiCheck(MPI_Dims_create(size, 2, dims));
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        // Keep tiles large for level-3 BLAS, but expose work to small grids
        // and avoid excessive padding of the final tile.
        const int wanted = static_cast<int>(std::max((n + 255LL) / 256,
                                                     static_cast<long long>(std::max(rows, cols))));
        const int width = static_cast<int>((n + static_cast<long long>(wanted) - 1) / wanted);
        block = n == 0 ? 1 : std::min(n, std::max(32, ((width + 31) / 32) * 32));
        tiles = static_cast<int>((n + static_cast<long long>(block) - 1) / block);
        localRows = prefixBlocks(tiles, row, rows) * block;
        localCols = prefixBlocks(tiles, col, cols) * block;
        ld = std::max(1, localRows);
        mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
    }
    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int rowStart(int first) const { return prefixBlocks(first, row, rows) * block; }
    int colStart(int first) const { return prefixBlocks(first, col, cols) * block; }
    int globalRow(int local) const { return (local / block * rows + row) * block + local % block; }
    int globalCol(int local) const { return (local / block * cols + col) * block + local % block; }
    size_t elements() const { return size_t(localRows) * localCols; }
    size_t offset(int tileRow, int tileCol) const {
        return size_t(tileRow / rows * block) + size_t(tileCol / cols * block) * ld;
    }
};

struct Accelerator {
    cudaStream_t panelStream = nullptr, updateStream = nullptr;
    cublasHandle_t panelBlas = nullptr, updateBlas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cudaEvent_t ready[2] = {}, done[2] = {};

    Accelerator() {
        int lowest, highest;
        check(cudaDeviceGetStreamPriorityRange(&lowest, &highest));
        check(cudaStreamCreateWithPriority(&panelStream, cudaStreamNonBlocking, highest));
        check(cudaStreamCreateWithPriority(&updateStream, cudaStreamNonBlocking, lowest));
        check(cublasCreate(&panelBlas));
        check(cublasCreate(&updateBlas));
        check(cublasSetStream(panelBlas, panelStream));
        check(cublasSetStream(updateBlas, updateStream));
        check(cusolverDnCreate(&solver));
        check(cusolverDnSetStream(solver, panelStream));
        for (int s = 0; s < 2; ++s) {
            check(cudaEventCreateWithFlags(&ready[s], cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
            check(cudaEventRecord(done[s], updateStream));
        }
    }
    ~Accelerator() {
        cudaStreamSynchronize(panelStream);
        cudaStreamSynchronize(updateStream);
        for (int s = 0; s < 2; ++s) {
            cudaEventDestroy(ready[s]);
            cudaEventDestroy(done[s]);
        }
        cusolverDnDestroy(solver);
        cublasDestroy(panelBlas);
        cublasDestroy(updateBlas);
        cudaStreamDestroy(panelStream);
        cudaStreamDestroy(updateStream);
    }
    void synchronize() const {
        check(cudaStreamSynchronize(panelStream));
        check(cudaStreamSynchronize(updateStream));
    }
};

// glibc rand_r advances its 32-bit LCG three times per result. Jump ahead by
// whole results using affine exponentiation, then call the ORIGINAL rand_r
// for each value. This preserves seed=42 and the row-major sequence exactly.
#if defined(__GLIBC__)
unsigned int randomSeedAt(uint64_t results) {
    constexpr uint32_t a = 1103515245u, c = 12345u;
    uint32_t multiplier = a * a * a, increment = (a * a + a + 1u) * c;
    uint32_t state = 42u;
    while (results) {
        if (results & 1u) state = multiplier * state + increment;
        increment *= multiplier + 1u;
        multiplier *= multiplier;
        results >>= 1;
    }
    return state;
}
#endif

std::vector<unsigned int> rowSeeds(const Grid& grid, bool forColumns) {
    const int count = forColumns ? grid.localCols : grid.localRows;
    std::vector<unsigned int> seeds(count);
#if defined(__GLIBC__)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const int global = forColumns ? grid.globalCol(i) : grid.globalRow(i);
        seeds[i] = randomSeedAt(uint64_t(global) * grid.n);
    }
#else
    // POSIX does not specify rand_r's recurrence. Retain its actual sequence
    // on other libc implementations rather than silently changing the input.
    std::vector<unsigned int> starts(grid.n);
    unsigned int seed = 42;
    for (int i = 0; i < grid.n; ++i) {
        starts[i] = seed;
        for (int j = 0; j < grid.n; ++j) (void)rand_r(&seed);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const int global = forColumns ? grid.globalCol(i) : grid.globalRow(i);
        seeds[i] = global < grid.n ? starts[global] : 42;
    }
#endif
    return seeds;
}

void generateRows(double* buffer, std::vector<unsigned int>& seeds,
                  const Grid& grid, bool forColumns, int depth, int pitch) {
    const int count = static_cast<int>(seeds.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        double* destination = buffer + size_t(i) * pitch;
        const int global = forColumns ? grid.globalCol(i) : grid.globalRow(i);
        if (global >= grid.n) {
            std::fill(destination, destination + depth, 0.0);
        } else {
            unsigned int seed = seeds[i];
            for (int k = 0; k < depth; ++k)
                destination[k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
            seeds[i] = seed;
        }
    }
}

void generatePositiveDefiniteMatrix(double* matrix, const Grid& grid, Accelerator& gpu) {
    if (!grid.localRows || !grid.localCols || !grid.n) return;
    // Stream B in reduction-dimension chunks: no rank holds the entire B.
    // Double buffering overlaps OpenMP generation with GPU GEMM and DMA.
    const int depthLimit = std::min(grid.n, 2048);
    const bool sameRows = grid.rows == grid.cols && grid.row == grid.col;
    const size_t rowCount = size_t(grid.localRows) * depthLimit;
    const size_t colCount = sameRows ? 1 : size_t(grid.localCols) * depthLimit;
    PinnedBuffer<double> hr0(rowCount), hr1(rowCount), hc0(colCount), hc1(colCount);
    DeviceBuffer<double> dr0(rowCount), dr1(rowCount), dc0(colCount), dc1(colCount);
    double* hostRows[2] = {hr0.data, hr1.data};
    double* hostCols[2] = {hc0.data, hc1.data};
    double* deviceRows[2] = {dr0.data, dr1.data};
    double* deviceCols[2] = {dc0.data, dc1.data};
    auto seedsR = rowSeeds(grid, false);
    auto seedsC = sameRows ? std::vector<unsigned int>() : rowSeeds(grid, true);
    const double one = 1.0, zero = 0.0;
    int iteration = 0;
    for (int begin = 0; begin < grid.n; begin += depthLimit, ++iteration) {
        const int s = iteration % 2, depth = std::min(depthLimit, grid.n - begin);
        check(cudaEventSynchronize(gpu.done[s]));
        generateRows(hostRows[s], seedsR, grid, false, depth, depthLimit);
        if (!sameRows) generateRows(hostCols[s], seedsC, grid, true, depth, depthLimit);
        check(cudaMemcpy2DAsync(deviceRows[s], size_t(depthLimit) * sizeof(double),
                                hostRows[s], size_t(depthLimit) * sizeof(double),
                                size_t(depth) * sizeof(double), grid.localRows,
                                cudaMemcpyHostToDevice, gpu.panelStream));
        if (!sameRows)
            check(cudaMemcpy2DAsync(deviceCols[s], size_t(depthLimit) * sizeof(double),
                                    hostCols[s], size_t(depthLimit) * sizeof(double),
                                    size_t(depth) * sizeof(double), grid.localCols,
                                    cudaMemcpyHostToDevice, gpu.panelStream));
        check(cublasDgemm(gpu.panelBlas, CUBLAS_OP_T, CUBLAS_OP_N,
                          grid.localRows, grid.localCols, depth, &one,
                          deviceRows[s], depthLimit, sameRows ? deviceRows[s] : deviceCols[s],
                          depthLimit, begin == 0 ? &zero : &one, matrix, grid.ld));
        check(cudaEventRecord(gpu.done[s], gpu.panelStream));
    }
    PinnedBuffer<double> diagonal(grid.block);
    DeviceBuffer<double> dDiagonal(grid.block);
    for (int j = grid.col; j < grid.tiles; j += grid.cols) {
        if (j % grid.rows != grid.row) continue;
        for (int i = 0; i < grid.block; ++i)
            diagonal.data[i] = j * grid.block + i < grid.n ? double(grid.n) : 1.0;
        check(cudaMemcpyAsync(dDiagonal.data, diagonal.data, size_t(grid.block) * sizeof(double),
                               cudaMemcpyHostToDevice, gpu.panelStream));
        check(cublasDaxpy(gpu.panelBlas, grid.block, &one, dDiagonal.data, 1,
                          matrix + grid.offset(j, j), grid.ld + 1));
        // Protect the pinned source from being overwritten by the next tile.
        check(cudaStreamSynchronize(gpu.panelStream));
    }
    check(cudaStreamSynchronize(gpu.panelStream));
}

struct Panel {
    PinnedBuffer<double> hostRows, hostCols, send, receive;
    DeviceBuffer<double> deviceRows, deviceCols;
    Panel(const Grid& g) : hostRows(size_t(g.localRows) * g.block),
        hostCols(size_t(g.localCols) * g.block),
        send(size_t(g.localCols) * g.block), receive(size_t(g.localCols) * g.block),
        deviceRows(size_t(g.localRows) * g.block), deviceCols(size_t(g.localCols) * g.block) {}
};

// Redistribute the tall panel into the row/column panels needed by each rank.
// On square grids this is two broadcasts of contiguous buffers. Rectangular
// grids use one column allgather and OpenMP packing of the intersections.
void exchangePanel(double* matrix, int k, int first, bool triangular,
                   const Grid& g, Accelerator& gpu, Panel& panel) {
    const int startR = g.rowStart(first), startC = g.colStart(first);
    const int height = g.localRows - startR, width = g.localCols - startC;
    if (g.size == 1 && !triangular) {
        // With one rank both panel distributions coincide; avoid staging
        // device data through host memory for a self broadcast.
        if (height) {
            check(cudaMemcpy2DAsync(panel.deviceRows.data, size_t(height) * sizeof(double),
                                    matrix + startR + size_t(k * g.block) * g.ld,
                                    size_t(g.ld) * sizeof(double), size_t(height) * sizeof(double),
                                    g.block, cudaMemcpyDeviceToDevice, gpu.panelStream));
            check(cudaMemcpyAsync(panel.deviceCols.data, panel.deviceRows.data,
                                   size_t(height) * g.block * sizeof(double),
                                   cudaMemcpyDeviceToDevice, gpu.panelStream));
        }
        return;
    }
    if (g.col == k % g.cols && height) {
        check(cudaMemcpy2DAsync(panel.hostRows.data, size_t(height) * sizeof(double),
                                matrix + startR + size_t(k / g.cols * g.block) * g.ld,
                                size_t(g.ld) * sizeof(double), size_t(height) * sizeof(double),
                                g.block, cudaMemcpyDeviceToHost, gpu.panelStream));
        check(cudaStreamSynchronize(gpu.panelStream));
        if (triangular && g.row == k % g.rows) {
            // Only diagonal tiles contain an unused upper triangle.
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < g.block; ++j)
                for (int i = 0; i < j; ++i)
                    panel.hostRows.data[size_t(j) * height + i] = 0.0;
        }
    }
    broadcast(panel.hostRows.data, size_t(height) * g.block, k % g.cols, g.rowComm);
    if (height)
        check(cudaMemcpyAsync(panel.deviceRows.data, panel.hostRows.data,
                               size_t(height) * g.block * sizeof(double),
                               cudaMemcpyHostToDevice, gpu.panelStream));
    if (g.rows == g.cols) {
        if (g.row == g.col && width)
            std::memcpy(panel.hostCols.data, panel.hostRows.data, size_t(width) * g.block * sizeof(double));
        broadcast(panel.hostCols.data, size_t(width) * g.block, g.col, g.colComm);
    } else {
        std::vector<int> counts(g.rows, 0), displacements(g.rows, 0), offsets(g.localCols / g.block, 0);
        const size_t tileElements = size_t(g.block) * g.block;
        for (int local = startC / g.block; local < g.localCols / g.block; ++local) {
            const int global = local * g.cols + g.col, source = global % g.rows;
            offsets[local] = counts[source];
            if (size_t(counts[source]) + tileElements > size_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("MPI panel exceeds its count limit");
            counts[source] += static_cast<int>(tileElements);
        }
        for (int r = 1; r < g.rows; ++r) {
            const size_t displacement = size_t(displacements[r - 1]) + counts[r - 1];
            if (displacement > size_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("MPI panel exceeds its displacement limit");
            displacements[r] = static_cast<int>(displacement);
        }
        #pragma omp parallel for schedule(static)
        for (int local = startC / g.block; local < g.localCols / g.block; ++local) {
            const int global = local * g.cols + g.col;
            if (global % g.rows != g.row) continue;
            const int rowOffset = global / g.rows * g.block - startR;
            for (int j = 0; j < g.block; ++j)
                std::memcpy(panel.send.data + offsets[local] + size_t(j) * g.block,
                            panel.hostRows.data + rowOffset + size_t(j) * height,
                            size_t(g.block) * sizeof(double));
        }
        mpiCheck(MPI_Allgatherv(panel.send.data, counts[g.row], MPI_DOUBLE, panel.receive.data,
                               counts.data(), displacements.data(), MPI_DOUBLE, g.colComm));
        #pragma omp parallel for schedule(static)
        for (int local = startC / g.block; local < g.localCols / g.block; ++local) {
            const int global = local * g.cols + g.col;
            const size_t source = size_t(displacements[global % g.rows]) + offsets[local];
            const int colOffset = local * g.block - startC;
            for (int j = 0; j < g.block; ++j)
                std::memcpy(panel.hostCols.data + colOffset + size_t(j) * width,
                            panel.receive.data + source + size_t(j) * g.block,
                            size_t(g.block) * sizeof(double));
        }
    }
    if (width)
        check(cudaMemcpyAsync(panel.deviceCols.data, panel.hostCols.data,
                               size_t(width) * g.block * sizeof(double),
                               cudaMemcpyHostToDevice, gpu.panelStream));
}

// Update exactly the lower triangle. Each off-diagonal GEMM spans all local
// rows below a tile column, avoiding tiny per-tile GEMM launches and the
// doubled FLOP count of updating the unused upper triangle.
void updateColumn(double* matrix, int j, int first, const Grid& g,
                  const Panel& panel, cublasHandle_t blas) {
    if (j % g.cols != g.col) return;
    const int startR = g.rowStart(first), startC = g.colStart(first);
    const int height = g.localRows - startR, width = g.localCols - startC;
    if (!height || !width) return;
    const double minusOne = -1.0, one = 1.0;
    if (j % g.rows == g.row) {
        const int diagonalRow = j / g.rows * g.block;
        check(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                          g.block, g.block, &minusOne, panel.deviceRows.data + diagonalRow - startR,
                          height, &one, matrix + g.offset(j, j), g.ld));
    }
    const int below = g.rowStart(j + 1), count = g.localRows - below;
    if (count) {
        const int localCol = j / g.cols * g.block;
        check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                          count, g.block, g.block, &minusOne,
                          panel.deviceRows.data + below - startR, height,
                          panel.deviceCols.data + localCol - startC, width, &one,
                          matrix + below + size_t(localCol) * g.ld, g.ld));
    }
}

bool choleskyDecomposition(double* matrix, const Grid& g, Accelerator& gpu,
                           Panel& p0, Panel& p1, DeviceBuffer<double>& diagonal,
                           PinnedBuffer<double>& hostDiagonal, DeviceBuffer<double>& workspace,
                           int workspaceSize, DeviceBuffer<int>& deviceInfo) {
    Panel* panels[2] = {&p0, &p1};
    const double one = 1.0;
    for (int k = 0; k < g.tiles; ++k) {
        const int slot = k % 2;
        // Double buffers stay alive until their previous trailing GEMMs end.
        // A stream wait protects device storage; pinned host storage needs
        // a host-side wait before MPI can overwrite its previous DMA source.
        if (k >= 2) check(cudaEventSynchronize(gpu.ready[slot]));
        check(cudaStreamWaitEvent(gpu.panelStream, gpu.done[slot], 0));
        int info = 0;
        const int owner = (k % g.rows) * g.cols + k % g.cols;
        if (g.rank == owner) {
            check(cudaMemcpy2DAsync(diagonal.data, size_t(g.block) * sizeof(double),
                                    matrix + g.offset(k, k), size_t(g.ld) * sizeof(double),
                                    size_t(g.block) * sizeof(double), g.block,
                                    cudaMemcpyDeviceToDevice, gpu.panelStream));
            check(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, g.block,
                                   diagonal.data, g.block, workspace.data, workspaceSize, deviceInfo.data));
            check(cudaMemcpyAsync(&info, deviceInfo.data, sizeof(int), cudaMemcpyDeviceToHost, gpu.panelStream));
            check(cudaMemcpyAsync(hostDiagonal.data, diagonal.data,
                                   size_t(g.block) * g.block * sizeof(double),
                                   cudaMemcpyDeviceToHost, gpu.panelStream));
            check(cudaMemcpy2DAsync(matrix + g.offset(k, k), size_t(g.ld) * sizeof(double),
                                    diagonal.data, size_t(g.block) * sizeof(double),
                                    size_t(g.block) * sizeof(double), g.block,
                                    cudaMemcpyDeviceToDevice, gpu.panelStream));
            check(cudaStreamSynchronize(gpu.panelStream));
        }
        mpiCheck(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info != 0) {
            if (g.rank == 0) {
                if (info > 0)
                    std::printf("Error: Matrix is not positive definite at diagonal element %lld\n",
                                static_cast<long long>(k) * g.block + info - 1);
                else std::printf("Error: cuSOLVER rejected factorization argument %d\n", -info);
            }
            gpu.synchronize();
            return false;
        }
        if (g.col == k % g.cols && k + 1 < g.tiles) {
            broadcast(hostDiagonal.data, size_t(g.block) * g.block, k % g.rows, g.colComm);
            const int start = g.rowStart(k + 1), count = g.localRows - start;
            if (count) {
                if (g.rank != owner)
                    check(cudaMemcpyAsync(diagonal.data, hostDiagonal.data,
                                           size_t(g.block) * g.block * sizeof(double),
                                           cudaMemcpyHostToDevice, gpu.panelStream));
                check(cublasDtrsm(gpu.panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, count, g.block, &one,
                                  diagonal.data, g.block,
                                  matrix + start + size_t(k / g.cols * g.block) * g.ld, g.ld));
            }
        }
        if (k + 1 == g.tiles) break;
        Panel& panel = *panels[slot];
        exchangePanel(matrix, k, k + 1, false, g, gpu, panel);
        check(cudaEventRecord(gpu.ready[slot], gpu.panelStream));
        check(cudaStreamWaitEvent(gpu.updateStream, gpu.ready[slot], 0));
        for (int j = k + 2; j < g.tiles; ++j)
            updateColumn(matrix, j, k + 1, g, panel, gpu.updateBlas);
        check(cudaEventRecord(gpu.done[slot], gpu.updateStream));
        // One-step lookahead: update the next panel on the priority stream.
        // Its previous contribution must finish first; the rest of this
        // step's trailing updates overlap the next POTRF/TRSM and MPI work.
        if (k > 0) check(cudaStreamWaitEvent(gpu.panelStream, gpu.done[1 - slot], 0));
        updateColumn(matrix, k + 1, k + 1, g, panel, gpu.panelBlas);
    }
    gpu.synchronize();
    return true;
}

bool validateCholesky(double* matrix, double* original, const Grid& g,
                      Accelerator& gpu, Panel& panel) {
    DeviceBuffer<double> residual(g.elements());
    if (g.elements())
        check(cudaMemcpyAsync(residual.data, original, g.elements() * sizeof(double),
                               cudaMemcpyDeviceToDevice, gpu.panelStream));
    // Form A - L*L^T distributively using the same BLAS-3 panel exchanges.
    // Only the lower triangle is needed because the input is symmetric.
    for (int k = 0; k < g.tiles; ++k) {
        exchangePanel(matrix, k, k, true, g, gpu, panel);
        for (int j = k; j < g.tiles; ++j)
            updateColumn(residual.data, j, k, g, panel, gpu.panelBlas);
        // The pinned buffers are reused in the next exchange.
        check(cudaStreamSynchronize(gpu.panelStream));
    }
    gpu.synchronize();
    std::vector<double> hostOriginal(g.elements()), hostResidual(g.elements());
    if (g.elements()) {
        check(cudaMemcpy(hostOriginal.data(), original, g.elements() * sizeof(double), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hostResidual.data(), residual.data, g.elements() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    double absolute = 0.0, relative = 0.0;
    int invalid = 0;
    #pragma omp parallel for reduction(max:absolute,relative,invalid) schedule(static)
    for (int j = 0; j < g.localCols; ++j) {
        const int globalJ = g.globalCol(j);
        if (globalJ >= g.n) continue;
        for (int i = 0; i < g.localRows; ++i) {
            const int globalI = g.globalRow(i);
            if (globalI < globalJ || globalI >= g.n) continue;
            const size_t index = size_t(j) * g.ld + i;
            const double error = std::abs(hostResidual[index]);
            const double rel = error / (std::abs(hostOriginal[index]) + 1e-10);
            if (!std::isfinite(error) || !std::isfinite(rel)) invalid = 1;
            absolute = std::max(absolute, error);
            relative = std::max(relative, rel);
        }
    }
    double local[2] = {absolute, relative}, global[2] = {};
    mpiCheck(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (g.rank == 0) {
        std::printf("Max absolute error: %.10e\n", global[0]);
        std::printf("Max relative error: %.10e\n", global[1]);
        if (invalid || global[1] > 1e-6)
            std::printf("Validation failed: relative error too large or non-finite result\n");
    }
    return !invalid && global[1] <= 1e-6;
}

void printFactor(double* matrix, const Grid& g) {
    std::vector<double> local(g.elements());
    if (g.elements())
        check(cudaMemcpy(local.data(), matrix, g.elements() * sizeof(double), cudaMemcpyDeviceToHost));
    constexpr size_t chunk = 1u << 26;
    constexpr int tag = 71;
    if (g.rank != 0) {
        for (size_t offset = 0; offset < local.size(); offset += chunk)
            mpiCheck(MPI_Send(local.data() + offset, static_cast<int>(std::min(chunk, local.size() - offset)),
                              MPI_DOUBLE, 0, tag, MPI_COMM_WORLD));
        return;
    }
    std::vector<double> factor(size_t(g.n) * g.n, 0.0);
    for (int source = 0; source < g.size; ++source) {
        const int sourceRow = source / g.cols, sourceCol = source % g.cols;
        const int m = prefixBlocks(g.tiles, sourceRow, g.rows) * g.block;
        const int n = prefixBlocks(g.tiles, sourceCol, g.cols) * g.block;
        if (source != 0) {
            local.resize(size_t(m) * n);
            for (size_t offset = 0; offset < local.size(); offset += chunk)
                mpiCheck(MPI_Recv(local.data() + offset, static_cast<int>(std::min(chunk, local.size() - offset)),
                                  MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j) {
            const int globalJ = (j / g.block * g.cols + sourceCol) * g.block + j % g.block;
            if (globalJ >= g.n) continue;
            for (int i = 0; i < m; ++i) {
                const int globalI = (i / g.block * g.rows + sourceRow) * g.block + i % g.block;
                if (globalI >= globalJ && globalI < g.n)
                    factor[size_t(globalI) * g.n + globalJ] = local[size_t(j) * m + i];
            }
        }
    }
    print_results(factor, "CholeskyL");
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int processes, MPI_Comm nodeComm) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(text, &end, 10);
            if (text[0] == '-' || !text[0] || *end || errno ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max() - 1024) ||
                (value > 0 && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                if (rank == 0) std::printf("Invalid matrix size: %s\n", text);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    int localRank = 0, localSize = 1, devices = 0;
    mpiCheck(MPI_Comm_rank(nodeComm, &localRank));
    mpiCheck(MPI_Comm_size(nodeComm, &localSize));
    check(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("MPI/OpenMP/CUDA execution requires an available CUDA GPU");
    check(cudaSetDevice(localRank % devices));
    // Respect explicit OpenMP settings. Otherwise avoid creating a full
    // machine's worth of threads for every MPI rank on a shared node.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    Grid grid(n, rank, processes);
    Accelerator gpu;
    DeviceBuffer<double> matrix(grid.elements()), original(validate ? grid.elements() : 0);
    Panel p0(grid), p1(grid);
    DeviceBuffer<double> diagonal(size_t(grid.block) * grid.block);
    PinnedBuffer<double> hostDiagonal(size_t(grid.block) * grid.block);
    DeviceBuffer<int> deviceInfo(1);
    int workspaceSize = 0;
    check(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, grid.block,
                                     diagonal.data, grid.block, &workspaceSize));
    DeviceBuffer<double> workspace(workspaceSize);
    // Warm the solver outside the timed section, just as generation warms
    // cuBLAS. This also establishes consistent startup costs across ranks.
    std::fill(hostDiagonal.data, hostDiagonal.data + size_t(grid.block) * grid.block, 0.0);
    for (int i = 0; i < grid.block; ++i) hostDiagonal.data[size_t(i) * (grid.block + 1)] = 1.0;
    check(cudaMemcpyAsync(diagonal.data, hostDiagonal.data, size_t(grid.block) * grid.block * sizeof(double),
                           cudaMemcpyHostToDevice, gpu.panelStream));
    check(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, grid.block, diagonal.data,
                           grid.block, workspace.data, workspaceSize, deviceInfo.data));
    check(cudaStreamSynchronize(gpu.panelStream));
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI grid: %d x %d; tile size: %d; OpenMP threads per rank: %d\n",
                    grid.rows, grid.cols, grid.block, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(matrix.data, grid, gpu);
    if (validate && grid.elements())
        check(cudaMemcpyAsync(original.data, matrix.data, grid.elements() * sizeof(double),
                               cudaMemcpyDeviceToDevice, gpu.panelStream));
    gpu.synchronize();
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(matrix.data, grid, gpu, p0, p1, diagonal,
                                               hostDiagonal, workspace, workspaceSize, deviceInfo);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    mpiCheck(MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double operations = double(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? operations / elapsed / 1e9 : 0.0);
    }
    if (printResults) printFactor(matrix.data, grid);
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateCholesky(matrix.data, original.data, grid, gpu, p0);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, processes = 1, result = 1;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
        mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &processes));
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
        result = runBenchmark(argc, argv, rank, processes, nodeComm);
        mpiCheck(MPI_Comm_free(&nodeComm));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
