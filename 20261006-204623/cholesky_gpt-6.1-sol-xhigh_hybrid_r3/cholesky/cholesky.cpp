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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
int worldRank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation,
                     cudaGetErrorString(status));
        fail("CUDA operation failed");
    }
}
void checkLibrary(int status, const char* operation) {
    if (status != 0) {
        std::fprintf(stderr, "Rank %d: %s returned %d\n", worldRank, operation, status);
        fail("CUDA library operation failed");
    }
}
void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(status, message, &length);
        std::fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
        fail("MPI operation failed");
    }
}
#define CUDA_CALL(x) checkCuda((x), #x)
#define BLAS_CALL(x) checkLibrary((x), #x)
#define SOLVER_CALL(x) checkLibrary((x), #x)
#define MPI_CALL(x) checkMpi((x), #x)

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct HostBuffer {
    double* data = nullptr;
    void allocate(size_t count) {
        CUDA_CALL(cudaMallocHost(reinterpret_cast<void**>(&data),
                                 std::max(size_t(1), count) * sizeof(double)));
    }
    ~HostBuffer() { if (data) cudaFreeHost(data); }
    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

// Tile-cyclic axes are packed into local dense column-major matrices. Thus
// a complete local panel can be solved/updated with a single BLAS operation.
struct Grid {
    int n, block, ranks, rows, cols, row, col, localRows, localCols, lda;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;

    Grid(int size, int processes) : n(size), ranks(processes) {
        int dims[2] = {0, 0};
        MPI_CALL(MPI_Dims_create(ranks, 2, dims));
        rows = dims[0]; cols = dims[1];
        row = worldRank / cols; col = worldRank % cols;
        // Smaller panels shorten the critical path for small local problems;
        // wider tiles improve GEMM efficiency for large trailing matrices.
        block = n / std::max(rows, cols) > 4096 ? 256 : 128;
        // Retain enough tiles to spread small inputs over the process grid.
        while (block > 32 && (n + block - 1LL) / block < 2 * std::max(rows, cols))
            block /= 2;
        localRows = axisSize(row, rows);
        localCols = axisSize(col, cols);
        lda = std::max(1, localRows);
        if (static_cast<size_t>(std::max(localRows, localCols)) * block > INT_MAX)
            fail("Matrix panels exceed the MPI count limit");
        MPI_CALL(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        MPI_CALL(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
    }
    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int tiles() const { return static_cast<int>((n + block - 1LL) / block); }
    int width(int tile) const { return std::min(block, n - tile * block); }
    int axisSize(int coordinate, int processes) const {
        const int full = n / block, tail = n % block;
        return (full / processes + (coordinate < full % processes)) * block
             + (coordinate == full % processes ? tail : 0);
    }
    int globalIndex(int local, int coordinate, int processes) const {
        return (local / block * processes + coordinate) * block + local % block;
    }
    int firstAt(int tile, int coordinate, int processes, int count) const {
        const int first = tile + (coordinate - tile % processes + processes) % processes;
        return std::min(count, first / processes * block);
    }
    int rowAt(int tile) const { return firstAt(tile, row, rows, localRows); }
    int colAt(int tile) const { return firstAt(tile, col, cols, localCols); }
    bool sameAxes() const { return rows == cols && row == col; }
};

// glibc rand_r advances this 32-bit LCG three times per output. Jumping the
// affine recurrence preserves the original seed-42 sequence exactly, while
// allowing independent OpenMP generation of arbitrary global rows/panels.
uint32_t randomSeed(uint64_t index) {
    uint32_t multiplier = 1103515245u, increment = 12345u;
    uint32_t accumulatedMultiplier = 1, accumulatedIncrement = 0;
    for (uint64_t steps = index * 3; steps; steps >>= 1) {
        if (steps & 1) {
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
            accumulatedMultiplier *= multiplier;
        }
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return accumulatedMultiplier * 42u + accumulatedIncrement;
}

double nextRandom(unsigned int& state) {
#if defined(__GLIBC__)
    state = state * 1103515245u + 12345u;
    unsigned int value = (state >> 16) & 2047u;
    state = state * 1103515245u + 12345u;
    value = (value << 10) ^ ((state >> 16) & 1023u);
    state = state * 1103515245u + 12345u;
    value = (value << 10) ^ ((state >> 16) & 1023u);
    return value / static_cast<double>(RAND_MAX) - 0.5;
#else
    return rand_r(&state) / static_cast<double>(RAND_MAX) - 0.5;
#endif
}

__global__ void addDiagonal(double* a, int nr, int nc, int n, int block,
                            int row, int col, int pr, int pc) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr; i += blockDim.x * gridDim.x) {
        const int global = (i / block * pr + row) * block + i % block;
        if (global / block % pc == col) {
            const int j = global / block / pc * block + global % block;
            if (j < nc) a[static_cast<size_t>(j) * nr + i] += n;
        }
    }
}

__global__ void zeroUpper(double* a, size_t count, int nr, int block,
                          int row, int col, int pr, int pc) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int i = index % nr, j = index / nr;
        const int gi = (i / block * pr + row) * block + i % block;
        const int gj = (j / block * pc + col) * block + j % block;
        if (gi < gj) a[index] = 0.0;
    }
}

__global__ void maximumErrors(const double* a, const double* original, size_t count,
                              double* errors) {
    __shared__ double absolute[256], relative[256];
    double absMax = 0, relMax = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const double error = fabs(a[i] - original[i]);
        const double rel = error / (fabs(original[i]) + 1e-10);
        absMax = isfinite(error) ? fmax(absMax, error) : INFINITY;
        relMax = isfinite(rel) ? fmax(relMax, rel) : INFINITY;
    }
    absolute[threadIdx.x] = absMax; relative[threadIdx.x] = relMax;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            absolute[threadIdx.x] = fmax(absolute[threadIdx.x], absolute[threadIdx.x + stride]);
            relative[threadIdx.x] = fmax(relative[threadIdx.x], relative[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        errors[2 * blockIdx.x] = absolute[0];
        errors[2 * blockIdx.x + 1] = relative[0];
    }
}

__global__ void identityTile(double* a, int width, int leading) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * width)
        a[static_cast<size_t>(index / width) * leading + index % width] =
            index / width == index % width ? 1.0 : 0.0;
}

struct Accelerator {
    const Grid& g;
    DeviceBuffer<double> a, original, rowPanel[2], colPanel[2], diagonal, workspace;
    DeviceBuffer<int> info;
    HostBuffer hostRow[2], hostCol[2], hostDiagonal, sendPanel, receivePanel;
    cudaStream_t panelStream, updateStream;
    cudaEvent_t uploaded[2], updated[2];
    cublasHandle_t panelBlas, updateBlas;
    cusolverDnHandle_t solver;
    int workspaceSize = 0;
    static constexpr int generationWidth = 1024;

    explicit Accelerator(const Grid& grid, bool validate) : g(grid) {
        int low, high;
        CUDA_CALL(cudaDeviceGetStreamPriorityRange(&low, &high));
        CUDA_CALL(cudaStreamCreateWithPriority(&panelStream, cudaStreamNonBlocking, high));
        CUDA_CALL(cudaStreamCreateWithPriority(&updateStream, cudaStreamNonBlocking, low));
        BLAS_CALL(cublasCreate(&panelBlas)); BLAS_CALL(cublasCreate(&updateBlas));
        BLAS_CALL(cublasSetStream(panelBlas, panelStream));
        BLAS_CALL(cublasSetStream(updateBlas, updateStream));
        SOLVER_CALL(cusolverDnCreate(&solver));
        SOLVER_CALL(cusolverDnSetStream(solver, panelStream));
        const size_t count = static_cast<size_t>(g.localRows) * g.localCols;
        a.allocate(count);
        if (validate) original.allocate(count);
        for (int slot = 0; slot < 2; ++slot) {
            rowPanel[slot].allocate(static_cast<size_t>(g.localRows) * generationWidth);
            colPanel[slot].allocate(static_cast<size_t>(g.localCols) * generationWidth);
            hostRow[slot].allocate(static_cast<size_t>(g.localRows) * generationWidth);
            hostCol[slot].allocate(static_cast<size_t>(g.localCols) * generationWidth);
            CUDA_CALL(cudaEventCreateWithFlags(&uploaded[slot], cudaEventDisableTiming));
            CUDA_CALL(cudaEventCreateWithFlags(&updated[slot], cudaEventDisableTiming));
            CUDA_CALL(cudaEventRecord(uploaded[slot], panelStream));
            CUDA_CALL(cudaEventRecord(updated[slot], updateStream));
        }
        diagonal.allocate(static_cast<size_t>(g.block) * g.block);
        hostDiagonal.allocate(static_cast<size_t>(g.block) * g.block);
        if (g.rows != g.cols) {
            sendPanel.allocate(static_cast<size_t>(g.localCols) * g.block);
            receivePanel.allocate(static_cast<size_t>(g.localCols) * g.block);
        }
        info.allocate(1);
        const int width = std::min({g.block, g.localRows, g.localCols});
        if (width) {
            SOLVER_CALL(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                width, a.data, g.lda, &workspaceSize));
        }
        workspace.allocate(workspaceSize);
        // Force lazy solver/TRSM kernel loading before the timed factorization.
        // Matrix generation overwrites this temporary identity tile.
        if (width) {
            identityTile<<<(width * width + 255) / 256, 256, 0, panelStream>>>(a.data, width, g.lda);
            CUDA_CALL(cudaGetLastError());
            SOLVER_CALL(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width,
                a.data, g.lda, workspace.data, workspaceSize, info.data));
            CUDA_CALL(cudaMemsetAsync(rowPanel[0].data, 0, width * width * sizeof(double), panelStream));
            const double one = 1;
            BLAS_CALL(cublasDtrsm(panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, width, width, &one,
                a.data, g.lda, rowPanel[0].data, width));
            synchronize();
        }
    }
    ~Accelerator() {
        cudaStreamSynchronize(panelStream); cudaStreamSynchronize(updateStream);
        for (int slot = 0; slot < 2; ++slot) {
            cudaEventDestroy(uploaded[slot]); cudaEventDestroy(updated[slot]);
        }
        cusolverDnDestroy(solver);
        cublasDestroy(panelBlas); cublasDestroy(updateBlas);
        cudaStreamDestroy(panelStream); cudaStreamDestroy(updateStream);
    }
    void synchronize() {
        CUDA_CALL(cudaStreamSynchronize(panelStream));
        CUDA_CALL(cudaStreamSynchronize(updateStream));
    }

    void generatePositiveDefiniteMatrix() {
        const size_t count = static_cast<size_t>(g.localRows) * g.localCols;
        CUDA_CALL(cudaMemsetAsync(a.data, 0, count * sizeof(double), updateStream));
        std::vector<unsigned int> nativeSeeds;
#if !defined(__GLIBC__)
        // Other libc implementations may use a different rand_r recurrence.
        // Preserve their original sequence rather than substituting an RNG.
        nativeSeeds.resize(g.n);
        if (worldRank == 0) {
            unsigned int seed = 42;
            for (int i = 0; i < g.n; ++i) {
                nativeSeeds[i] = seed;
                for (int j = 0; j < g.n; ++j) rand_r(&seed);
            }
        }
        MPI_CALL(MPI_Bcast(nativeSeeds.data(), g.n, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
#endif
        int iteration = 0;
        for (int start = 0; start < g.n; ++iteration) {
            const int width = std::min(generationWidth, g.n - start), slot = iteration % 2;
            CUDA_CALL(cudaEventSynchronize(uploaded[slot]));
            const int totalRows = g.localRows + (g.sameAxes() ? 0 : g.localCols);
            #pragma omp parallel for schedule(static)
            for (int index = 0; index < totalRows; ++index) {
                const bool isRow = index < g.localRows;
                const int local = isRow ? index : index - g.localRows;
                const int global = g.globalIndex(local, isRow ? g.row : g.col,
                                                 isRow ? g.rows : g.cols);
                const int leading = isRow ? g.localRows : g.localCols;
                double* buffer = isRow ? hostRow[slot].data : hostCol[slot].data;
#if defined(__GLIBC__)
                unsigned int seed = randomSeed(static_cast<uint64_t>(global) * g.n + start);
#else
                unsigned int seed = nativeSeeds[global];
                for (int j = 0; j < start; ++j) rand_r(&seed);
#endif
                for (int j = 0; j < width; ++j)
                    buffer[static_cast<size_t>(j) * leading + local] = nextRandom(seed);
            }
            CUDA_CALL(cudaStreamWaitEvent(panelStream, updated[slot], 0));
            CUDA_CALL(cudaMemcpyAsync(rowPanel[slot].data, hostRow[slot].data,
                static_cast<size_t>(g.localRows) * width * sizeof(double),
                cudaMemcpyHostToDevice, panelStream));
            if (!g.sameAxes())
                CUDA_CALL(cudaMemcpyAsync(colPanel[slot].data, hostCol[slot].data,
                    static_cast<size_t>(g.localCols) * width * sizeof(double),
                    cudaMemcpyHostToDevice, panelStream));
            CUDA_CALL(cudaEventRecord(uploaded[slot], panelStream));
            CUDA_CALL(cudaStreamWaitEvent(updateStream, uploaded[slot], 0));
            if (g.localRows && g.localCols) {
                const double one = 1;
                const double* right = g.sameAxes() ? rowPanel[slot].data : colPanel[slot].data;
                BLAS_CALL(cublasDgemm(updateBlas, CUBLAS_OP_N, CUBLAS_OP_T,
                    g.localRows, g.localCols, width, &one, rowPanel[slot].data, g.localRows,
                    right, g.localCols, &one, a.data, g.lda));
            }
            CUDA_CALL(cudaEventRecord(updated[slot], updateStream));
            start += width;
        }
        if (g.localRows) {
            addDiagonal<<<std::min(65535LL, (g.localRows + 255LL) / 256), 256, 0, updateStream>>>
                (a.data, g.localRows, g.localCols, g.n, g.block, g.row, g.col, g.rows, g.cols);
            CUDA_CALL(cudaGetLastError());
        }
        if (original.data)
            CUDA_CALL(cudaMemcpyAsync(original.data, a.data, count * sizeof(double),
                                      cudaMemcpyDeviceToDevice, updateStream));
        synchronize();
    }

    // Row communicators distribute L(i,k). Column communicators redistribute
    // the matching L(j,k) for local column indices. Square grids need just one
    // broadcast per axis; rectangular grids use one packed all-gather.
    void exchangeColumnPanel(int slot, int width, int rowBegin, int colBegin) {
        const int nr = g.localRows - rowBegin, nc = g.localCols - colBegin;
        double* input = hostRow[slot].data;
        double* output = hostCol[slot].data;
        if (g.rows == g.cols) {
            if (g.row == g.col)
                std::memcpy(output, input, static_cast<size_t>(nc) * width * sizeof(double));
            MPI_CALL(MPI_Bcast(output, nc * width, MPI_DOUBLE, g.col, g.colComm));
            return;
        }
        std::vector<int> sizes(g.rows, 0), counts(g.rows), displacements(g.rows);
        std::vector<int> sourceRow(nc), sourceRank(nc), sendRows;
        for (int j = 0; j < nc; ++j) {
            const int global = g.globalIndex(colBegin + j, g.col, g.cols);
            const int root = global / g.block % g.rows;
            sourceRank[j] = root;
            sourceRow[j] = sizes[root]++;
            if (root == g.row)
                sendRows.push_back(global / g.block / g.rows * g.block + global % g.block - rowBegin);
        }
        int total = 0;
        for (int r = 0; r < g.rows; ++r) {
            counts[r] = sizes[r] * width;
            displacements[r] = total;
            total += counts[r];
        }
        const int owned = sizes[g.row];
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < width; ++k)
            for (int i = 0; i < owned; ++i)
                sendPanel.data[static_cast<size_t>(k) * owned + i] = input[static_cast<size_t>(k) * nr + sendRows[i]];
        MPI_CALL(MPI_Allgatherv(sendPanel.data, counts[g.row], MPI_DOUBLE,
            receivePanel.data, counts.data(), displacements.data(), MPI_DOUBLE, g.colComm));
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < width; ++k)
            for (int j = 0; j < nc; ++j) {
                const int root = sourceRank[j];
                output[static_cast<size_t>(k) * nc + j] =
                    receivePanel.data[displacements[root] + static_cast<size_t>(k) * sizes[root] + sourceRow[j]];
            }
    }

    void updateColumn(int tile, int rowBegin, int colBegin, int width, int slot,
                      cublasHandle_t handle) {
        const int first = g.rowAt(tile), nr = g.localRows - first;
        if (!nr) return;
        const int column = tile / g.cols * g.block;
        const int rowLeading = g.localRows - rowBegin, colLeading = g.localCols - colBegin;
        const double minusOne = -1, one = 1;
        const double* right = g.sameAxes() ? rowPanel[slot].data : colPanel[slot].data;
        BLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nr, g.width(tile), width,
            &minusOne, rowPanel[slot].data + first - rowBegin, rowLeading,
            right + column - colBegin, colLeading, &one,
            a.data + static_cast<size_t>(column) * g.lda + first, g.lda));
    }

    bool choleskyDecomposition() {
        for (int tile = 0; tile < g.tiles(); ++tile) {
            const int slot = tile % 2, width = g.width(tile);
            const int ownerRow = tile % g.rows, ownerCol = tile % g.cols;
            const int owner = ownerRow * g.cols + ownerCol;
            const int rowBegin = g.rowAt(tile + 1), colBegin = g.colAt(tile + 1);
            const int nr = g.localRows - rowBegin, nc = g.localCols - colBegin;
            CUDA_CALL(cudaEventSynchronize(uploaded[slot]));
            int status = 0;
            if (worldRank == owner) {
                double* diagonalTile = a.data + static_cast<size_t>(tile / g.cols * g.block) * g.lda
                                              + tile / g.rows * g.block;
                SOLVER_CALL(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width,
                    diagonalTile, g.lda, workspace.data, workspaceSize, info.data));
                CUDA_CALL(cudaMemcpyAsync(&status, info.data, sizeof(int), cudaMemcpyDeviceToHost, panelStream));
                CUDA_CALL(cudaMemcpy2DAsync(hostDiagonal.data, width * sizeof(double), diagonalTile,
                    g.lda * sizeof(double), width * sizeof(double), width, cudaMemcpyDeviceToHost, panelStream));
                CUDA_CALL(cudaStreamSynchronize(panelStream));
            }
            MPI_CALL(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (status) {
                if (worldRank == 0) {
                    if (status > 0)
                        std::printf("Error: Matrix is not positive definite at diagonal element %lld\n",
                                    static_cast<long long>(tile) * g.block + status - 1);
                    else std::printf("Error: Cholesky solver rejected argument %d\n", -status);
                }
                synchronize();
                return false;
            }
            if (g.col == ownerCol) {
                MPI_CALL(MPI_Bcast(hostDiagonal.data, width * width, MPI_DOUBLE, ownerRow, g.colComm));
                CUDA_CALL(cudaMemcpyAsync(diagonal.data, hostDiagonal.data, width * width * sizeof(double),
                                          cudaMemcpyHostToDevice, panelStream));
                if (nr) {
                    double* panel = a.data + static_cast<size_t>(tile / g.cols * g.block) * g.lda + rowBegin;
                    const double one = 1;
                    BLAS_CALL(cublasDtrsm(panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, nr, width, &one,
                        diagonal.data, width, panel, g.lda));
                    CUDA_CALL(cudaMemcpy2DAsync(hostRow[slot].data, nr * sizeof(double), panel,
                        g.lda * sizeof(double), nr * sizeof(double), width, cudaMemcpyDeviceToHost, panelStream));
                }
                CUDA_CALL(cudaStreamSynchronize(panelStream));
            }
            MPI_CALL(MPI_Bcast(hostRow[slot].data, nr * width, MPI_DOUBLE, ownerCol, g.rowComm));
            exchangeColumnPanel(slot, width, rowBegin, colBegin);
            // Ping-pong buffers protect the previous asynchronous update.
            CUDA_CALL(cudaStreamWaitEvent(panelStream, updated[slot], 0));
            CUDA_CALL(cudaMemcpyAsync(rowPanel[slot].data, hostRow[slot].data,
                static_cast<size_t>(nr) * width * sizeof(double), cudaMemcpyHostToDevice, panelStream));
            if (!g.sameAxes())
                CUDA_CALL(cudaMemcpyAsync(colPanel[slot].data, hostCol[slot].data,
                    static_cast<size_t>(nc) * width * sizeof(double), cudaMemcpyHostToDevice, panelStream));
            CUDA_CALL(cudaEventRecord(uploaded[slot], panelStream));
            CUDA_CALL(cudaStreamWaitEvent(updateStream, uploaded[slot], 0));
            // Finish the next panel first. Its factorization/communication in
            // the next iteration overlaps all remaining trailing-column GEMMs.
            if (tile) CUDA_CALL(cudaStreamWaitEvent(panelStream, updated[1 - slot], 0));
            if (tile + 1 < g.tiles() && (tile + 1) % g.cols == g.col)
                updateColumn(tile + 1, rowBegin, colBegin, width, slot, panelBlas);
            for (int next = tile + 2 + (g.col - (tile + 2) % g.cols + g.cols) % g.cols;
                 next < g.tiles(); next += g.cols)
                updateColumn(next, rowBegin, colBegin, width, slot, updateBlas);
            CUDA_CALL(cudaEventRecord(updated[slot], updateStream));
        }
        synchronize();
        const size_t count = static_cast<size_t>(g.localRows) * g.localCols;
        if (count) {
            zeroUpper<<<std::min(size_t(65535), (count + 255) / 256), 256, 0, panelStream>>>
                (a.data, count, g.localRows, g.block, g.row, g.col, g.rows, g.cols);
            CUDA_CALL(cudaGetLastError());
        }
        synchronize();
        return true;
    }

    bool validateCholesky() {
        const size_t count = static_cast<size_t>(g.localRows) * g.localCols;
        DeviceBuffer<double> reconstructed;
        reconstructed.allocate(count);
        CUDA_CALL(cudaMemsetAsync(reconstructed.data, 0, count * sizeof(double), updateStream));
        for (int tile = 0; tile < g.tiles(); ++tile) {
            const int slot = tile % 2, width = g.width(tile), ownerCol = tile % g.cols;
            CUDA_CALL(cudaEventSynchronize(uploaded[slot]));
            if (g.col == ownerCol && g.localRows) {
                CUDA_CALL(cudaMemcpy2DAsync(hostRow[slot].data, g.localRows * sizeof(double),
                    a.data + static_cast<size_t>(tile / g.cols * g.block) * g.lda,
                    g.lda * sizeof(double), g.localRows * sizeof(double), width,
                    cudaMemcpyDeviceToHost, panelStream));
                CUDA_CALL(cudaStreamSynchronize(panelStream));
            }
            MPI_CALL(MPI_Bcast(hostRow[slot].data, g.localRows * width, MPI_DOUBLE, ownerCol, g.rowComm));
            exchangeColumnPanel(slot, width, 0, 0);
            CUDA_CALL(cudaStreamWaitEvent(panelStream, updated[slot], 0));
            CUDA_CALL(cudaMemcpyAsync(rowPanel[slot].data, hostRow[slot].data,
                static_cast<size_t>(g.localRows) * width * sizeof(double), cudaMemcpyHostToDevice, panelStream));
            if (!g.sameAxes())
                CUDA_CALL(cudaMemcpyAsync(colPanel[slot].data, hostCol[slot].data,
                    static_cast<size_t>(g.localCols) * width * sizeof(double), cudaMemcpyHostToDevice, panelStream));
            CUDA_CALL(cudaEventRecord(uploaded[slot], panelStream));
            CUDA_CALL(cudaStreamWaitEvent(updateStream, uploaded[slot], 0));
            if (count) {
                const double one = 1;
                const double* right = g.sameAxes() ? rowPanel[slot].data : colPanel[slot].data;
                BLAS_CALL(cublasDgemm(updateBlas, CUBLAS_OP_N, CUBLAS_OP_T,
                    g.localRows, g.localCols, width, &one, rowPanel[slot].data, g.localRows,
                    right, g.localCols, &one, reconstructed.data, g.lda));
            }
            CUDA_CALL(cudaEventRecord(updated[slot], updateStream));
        }
        synchronize();
        double localErrors[2] = {0, 0}, errors[2];
        if (count) {
            const int blocks = static_cast<int>(std::min(size_t(1024), (count + 255) / 256));
            DeviceBuffer<double> deviceErrors;
            deviceErrors.allocate(2 * blocks);
            maximumErrors<<<blocks, 256, 0, updateStream>>>(reconstructed.data, original.data, count, deviceErrors.data);
            CUDA_CALL(cudaGetLastError());
            std::vector<double> partial(2 * blocks);
            CUDA_CALL(cudaMemcpyAsync(partial.data(), deviceErrors.data, partial.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost, updateStream));
            CUDA_CALL(cudaStreamSynchronize(updateStream));
            for (int i = 0; i < blocks; ++i) {
                localErrors[0] = std::max(localErrors[0], partial[2 * i]);
                localErrors[1] = std::max(localErrors[1], partial[2 * i + 1]);
            }
        }
        MPI_CALL(MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (worldRank == 0) {
            std::printf("Max absolute error: %.10e\n", errors[0]);
            std::printf("Max relative error: %.10e\n", errors[1]);
            if (errors[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
        }
        return errors[1] <= 1e-6;
    }

    void printResults() {
        // Gather only when requested. Normal runs and validation never
        // replicate the full matrix or require O(n^2) storage on rank zero.
        std::vector<double> result;
        if (worldRank == 0) result.resize(static_cast<size_t>(g.n) * g.n);
        HostBuffer packed;
        packed.allocate(static_cast<size_t>(g.localRows) * g.localCols);
        CUDA_CALL(cudaMemcpy(packed.data, a.data,
            static_cast<size_t>(g.localRows) * g.localCols * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> received;
        for (int rank = 0; rank < g.ranks; ++rank) {
            const int r = rank / g.cols, c = rank % g.cols;
            const int nr = g.axisSize(r, g.rows), nc = g.axisSize(c, g.cols);
            const size_t count = static_cast<size_t>(nr) * nc;
            double* source = packed.data;
            if (worldRank == 0 && rank) { received.resize(count); source = received.data(); }
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(size_t(INT_MAX), count - offset));
                if (rank && worldRank == rank)
                    MPI_CALL(MPI_Send(packed.data + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                if (rank && worldRank == 0)
                    MPI_CALL(MPI_Recv(source + offset, chunk, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                offset += chunk;
            }
            if (worldRank == 0) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nr; ++i) {
                    const int globalRow = g.globalIndex(i, r, g.rows);
                    for (int j = 0; j < nc; ++j) {
                        const int globalCol = g.globalIndex(j, c, g.cols);
                        result[static_cast<size_t>(globalRow) * g.n + globalCol] = source[static_cast<size_t>(j) * nr + i];
                    }
                }
            }
        }
        if (worldRank == 0) print_results(result, "CholeskyL");
    }
};

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int ranks) {
    int options[4] = {512, 0, 0, 0};
    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* argument = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long parsed = std::strtoull(argument, &end, 10);
                if (*argument == '-' || end == argument || *end || errno || parsed > INT_MAX ||
                    parsed > std::sqrt(static_cast<long double>(std::numeric_limits<size_t>::max() / sizeof(double)))) {
                    std::printf("Invalid matrix size: %s\n", argument);
                    options[3] = 1; break;
                }
                options[0] = static_cast<int>(parsed);
            } else if (std::strcmp(argv[i], "-v") == 0) options[1] = 1;
            else if (std::strcmp(argv[i], "-r") == 0) options[2] = 1;
            else if (std::strcmp(argv[i], "-h") == 0) { options[3] = -1; break; }
            else {
                std::printf("Unknown option: %s\n", argv[i]);
                options[3] = 1; break;
            }
        }
        if (options[3]) printUsage(argv[0]);
    }
    MPI_CALL(MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD));
    if (options[3]) return options[3] < 0 ? 0 : 1;

    MPI_Comm localComm;
    MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm));
    int localRank, localSize, devices;
    MPI_CALL(MPI_Comm_rank(localComm, &localRank));
    MPI_CALL(MPI_Comm_size(localComm, &localSize));
    MPI_CALL(MPI_Comm_free(&localComm));
    CUDA_CALL(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA accelerator is required on every MPI rank");
    CUDA_CALL(cudaSetDevice(localRank % devices));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(16, omp_get_num_procs() / localSize)));

    Grid grid(options[0], ranks);
    Accelerator accelerator(grid, options[1] != 0);
    if (worldRank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", grid.n, grid.n);
        std::printf("Validation: %s\n", options[1] ? "enabled" : "disabled");
        std::printf("MPI grid: %d x %d; OpenMP threads per rank: %d; tile size: %d\n",
                    grid.rows, grid.cols, omp_get_max_threads(), grid.block);
        std::printf("Generating positive definite matrix...\n");
    }
    accelerator.generatePositiveDefiniteMatrix();
    if (worldRank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success = accelerator.choleskyDecomposition();
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_CALL(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (worldRank == 0) std::printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000);
        const double operations = static_cast<double>(grid.n) * grid.n * grid.n / 3;
        std::printf("Performance: %.3f GFLOPS\n", seconds > 0 ? operations / seconds / 1e9 : 0);
    }
    if (options[2]) accelerator.printResults();
    if (options[1]) {
        if (worldRank == 0) std::printf("Validating result...\n");
        const bool valid = accelerator.validateCholesky();
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    int ranks;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status;
    try {
        status = run(argc, argv, ranks);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    MPI_CALL(MPI_Finalize());
    return status;
}
