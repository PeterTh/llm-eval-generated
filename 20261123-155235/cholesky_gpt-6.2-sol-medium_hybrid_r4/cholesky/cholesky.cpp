#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <vector>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void fail(const char* operation, int status) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s failed (status %d)\n", rank, operation, status);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CHECK(call) do { const auto status = (call); \
    if (status != 0) fail(#call, static_cast<int>(status)); } while (false)

template <typename T, bool pinned = false>
class Buffer {
public:
    explicit Buffer(size_t count = 0) {
        if (count == 0) return;
        if constexpr (pinned) {
            CHECK(cudaMallocHost(&data, count * sizeof(T)));
        } else {
            CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }
    ~Buffer() {
        if (!data) return;
        if constexpr (pinned) cudaFreeHost(data);
        else cudaFree(data);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data = nullptr;
};

struct Gpu {
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;

    explicit Gpu(int localRank) {
        int count;
        CHECK(cudaGetDeviceCount(&count));
        if (!count) fail("finding a CUDA device", count);
        CHECK(cudaSetDevice(localRank % count));
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cublasSetStream(blas, stream));
        CHECK(cusolverDnCreate(&solver));
        CHECK(cusolverDnSetStream(solver, stream));
    }
    ~Gpu() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
    void sync() const { CHECK(cudaStreamSynchronize(stream)); }
};

// Number of locally owned elements preceding the given global tile.
__host__ __device__ int prefix(int tile, int coordinate, int stride, int block, int n) {
    const int count = tile / stride + (coordinate < tile % stride);
    const auto length = static_cast<int64_t>(count) * block;
    const auto last = (static_cast<int64_t>(count - 1) * stride + coordinate) * block;
    return static_cast<int>(count && last + block > n ? length - (last + block - n) : length);
}

__host__ __device__ int globalIndex(int index, int coordinate, int stride, int block) {
    return ((index / block) * stride + coordinate) * block + index % block;
}

struct Grid {
    int rank, size, row, col;
    int dims[2] = {};
    MPI_Comm rowComm, colComm;
    int n, block, tiles, rows, cols;

    explicit Grid(int order) : n(order) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        CHECK(MPI_Dims_create(size, 2, dims));
        row = rank / dims[1];
        col = rank % dims[1];
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
        // Keep enough tiles to spread small matrices across the process grid.
        const int target = static_cast<int>((static_cast<int64_t>(n) + 2LL * dims[0] - 1)
                                            / (2LL * dims[0]));
        block = size == 1 ? std::max(n, 1) : std::clamp(target, 64, 256);
        tiles = n / block + (n % block != 0);
        rows = rowPrefix(tiles);
        cols = colPrefix(tiles);
    }
    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int rowPrefix(int tile) const { return prefix(tile, row, dims[0], block, n); }
    int colPrefix(int tile) const { return prefix(tile, col, dims[1], block, n); }
    int width(int tile) const { return std::min(block, n - tile * block); }
    size_t elements() const { return static_cast<size_t>(rows) * cols; }
};

#ifdef __GLIBC__
// glibc rand_r advances a 32-bit linear congruential state three times per value.
// Exponentiate that transition to start any row or column stripe independently.
__host__ __device__ uint32_t advanceRandom(uint32_t state, uint64_t count) {
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    for (count *= 3; count; count >>= 1) {
        if (count & 1) state = multiplier * state + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return state;
}

__device__ double nextRandom(uint32_t& state) {
    state = state * 1103515245u + 12345u;
    uint32_t value = (state >> 16) & 2047u;
    state = state * 1103515245u + 12345u;
    value = (value << 10) ^ ((state >> 16) & 1023u);
    state = state * 1103515245u + 12345u;
    value = (value << 10) ^ ((state >> 16) & 1023u);
    return __dadd_rn(__ddiv_rn(static_cast<double>(value), RAND_MAX), -0.5);
}

constexpr int randomStripe = 16;

__global__ void randomSlab(double* slab, const uint32_t* seeds, int rows, int begin,
                           int width, size_t stripes) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < stripes;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int row = index % rows;
        const int first = index / rows * randomStripe;
        uint32_t seed = advanceRandom(seeds[row], static_cast<uint64_t>(begin) + first);
        for (int j = first; j < min(first + randomStripe, width); ++j) {
            slab[row + static_cast<size_t>(j) * rows] = nextRandom(seed);
        }
    }
}
#endif

class RandomRows {
public:
    RandomRows(int rows, int coordinate, int stride, const Grid& grid, const Gpu& gpu,
               int slabWidth)
        : rows(rows), gpu(gpu), slab(static_cast<size_t>(rows) * slabWidth), seeds(rows)
#ifdef __GLIBC__
        , deviceSeeds(rows)
#else
        , hostSlab(static_cast<size_t>(rows) * slabWidth)
#endif
    {
#ifdef __GLIBC__
        #pragma omp parallel for if(rows >= 128) schedule(static)
        for (int i = 0; i < rows; ++i) {
            const int global = globalIndex(i, coordinate, stride, grid.block);
            seeds[i] = advanceRandom(42, static_cast<uint64_t>(global) * grid.n);
        }
        if (rows) CHECK(cudaMemcpyAsync(deviceSeeds.data, seeds.data(), rows * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice, gpu.stream));
#else
        // On other C libraries, obtain the same row checkpoints through rand_r.
        unsigned int seed = 42;
        int previous = 0;
        for (int i = 0; i < rows; ++i) {
            const int global = globalIndex(i, coordinate, stride, grid.block);
            for (uint64_t skip = static_cast<uint64_t>(global - previous) * grid.n; skip; --skip) {
                rand_r(&seed);
            }
            seeds[i] = seed;
            previous = global;
        }
#endif
    }

    void generate(int begin, int width) {
#ifdef __GLIBC__
        const size_t stripes = static_cast<size_t>(rows) * ((width + randomStripe - 1) / randomStripe);
        const int blocks = std::min<size_t>((stripes + 255) / 256, 65535);
        randomSlab<<<blocks, 256, 0, gpu.stream>>>(slab.data, deviceSeeds.data, rows, begin,
                                                  width, stripes);
        CHECK(cudaGetLastError());
#else
        (void)begin;
        gpu.sync();
        #pragma omp parallel for if(rows >= 128) schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < width; ++j) {
                hostSlab.data[i + static_cast<size_t>(j) * rows] =
                    rand_r(&seeds[i]) / static_cast<double>(RAND_MAX) - 0.5;
            }
        }
        CHECK(cudaMemcpyAsync(slab.data, hostSlab.data, static_cast<size_t>(rows) * width *
                              sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
#endif
    }

    int rows;
    const Gpu& gpu;
    Buffer<double> slab;

private:
    std::vector<unsigned int> seeds;
#ifdef __GLIBC__
    Buffer<uint32_t> deviceSeeds;
#else
    Buffer<double, true> hostSlab;
#endif
};

__global__ void addDiagonal(double* matrix, int rows, int cols, int n, int block,
                            int row, int col, int processRows, int processCols) {
    for (int64_t j = blockIdx.x * blockDim.x + threadIdx.x; j < cols;
         j += blockDim.x * gridDim.x) {
        const int index = globalIndex(j, col, processCols, block);
        const int tile = index / block;
        if (tile % processRows == row) {
            const int i = (tile / processRows) * block + index % block;
            matrix[i + j * rows] += n;
        }
    }
}

void generateMatrix(const Grid& grid, const Gpu& gpu, double* matrix) {
    if (!grid.elements()) return;
    const int slabWidth = std::min(grid.n, 2048);
    const bool sameRows = grid.dims[0] == grid.dims[1] && grid.row == grid.col;
    RandomRows left(grid.rows, grid.row, grid.dims[0], grid, gpu, slabWidth);
    RandomRows right(sameRows ? 0 : grid.cols, grid.col, grid.dims[1], grid, gpu, slabWidth);
    if (sameRows) {
        // Grouped trailing updates may also touch some entries above the diagonal.
        CHECK(cudaMemsetAsync(matrix, 0, grid.elements() * sizeof(double), gpu.stream));
    }
    const double one = 1.0;
    for (int begin = 0; begin < grid.n;) {
        const int width = std::min(slabWidth, grid.n - begin);
        const double beta = begin ? 1.0 : 0.0;
        left.generate(begin, width);
        if (sameRows) {
            CHECK(cublasDsyrk(gpu.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                              grid.rows, width, &one, left.slab.data, grid.rows, &beta,
                              matrix, grid.rows));
        } else {
            right.generate(begin, width);
            CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T, grid.rows, grid.cols, width,
                              &one, left.slab.data, grid.rows, right.slab.data, grid.cols, &beta,
                              matrix, grid.rows));
        }
        begin += width;
    }
    const int blocks = std::min((grid.cols - 1) / 256 + 1, 65535);
    addDiagonal<<<blocks, 256, 0, gpu.stream>>>(matrix, grid.rows, grid.cols, grid.n,
        grid.block, grid.row, grid.col, grid.dims[0], grid.dims[1]);
    CHECK(cudaGetLastError());
    gpu.sync();
}

__global__ void clearUpper(double* diagonal, int width, int ld) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < static_cast<size_t>(width) * width;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int i = index % width;
        const int j = index / width;
        if (i < j) diagonal[i + static_cast<size_t>(j) * ld] = 0.0;
    }
}

// The two views of each solved panel match the rows and columns owned by this rank.
// Pinned staging permits the same communication path with any MPI implementation.
class Panels {
public:
    Panels(const Grid& grid, const Gpu& gpu)
        : grid(grid), gpu(gpu), rowDevice(panelSize(grid.rows)), colDevice(panelSize(grid.cols)),
          rowHost(panelSize(grid.rows)), colHost(panelSize(grid.cols)),
          gathered(grid.dims[0] == grid.dims[1] ? 0 : panelSize(grid.cols)),
          diagDevice(panelSize(grid.block)), diagHost(panelSize(grid.block)),
          counts(grid.dims[0]), displacements(grid.dims[0]), cursors(grid.dims[0]),
          offsets(grid.cols / grid.block + 1) {}

    void solve(double* matrix, int k, int width, int ownerRow, int ownerCol,
               double* diagonal) {
        if (grid.col != ownerCol) return;
        const int first = grid.rowPrefix(k + 1);
        const int remaining = grid.rows - first;
        // All ranks in this process column participate even when their local tail is empty.
        if (grid.row == ownerRow && grid.dims[0] > 1) {
            CHECK(cudaMemcpy2DAsync(diagHost.data, width * sizeof(double), diagonal,
                grid.rows * sizeof(double), width * sizeof(double), width,
                cudaMemcpyDeviceToHost, gpu.stream));
        }
        if (grid.dims[0] > 1) {
            if (grid.row == ownerRow) gpu.sync();
            CHECK(MPI_Bcast(diagHost.data, width * width, MPI_DOUBLE, ownerRow, grid.colComm));
        }
        if (!remaining) return;
        int ld = grid.rows;
        if (grid.row != ownerRow) {
            CHECK(cudaMemcpyAsync(diagDevice.data, diagHost.data, width * width * sizeof(double),
                                  cudaMemcpyHostToDevice, gpu.stream));
            diagonal = diagDevice.data;
            ld = width;
        }
        const double one = 1.0;
        CHECK(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                          CUBLAS_DIAG_NON_UNIT, remaining, width, &one, diagonal, ld,
                          matrix + first + static_cast<size_t>(grid.colPrefix(k)) * grid.rows,
                          grid.rows));
    }

    void exchange(const double* matrix, int k, int firstTile) {
        const int width = grid.width(k);
        const int ownerCol = k % grid.dims[1];
        const int rowStart = grid.rowPrefix(firstTile);
        rowCount = grid.rows - rowStart;
        colCount = grid.cols - grid.colPrefix(firstTile);
        left = rowDevice.data;
        leftLd = rowCount;
        right = colDevice.data;
        rightLd = colCount;

        if (rowCount && grid.col == ownerCol) {
            left = matrix + rowStart + static_cast<size_t>(grid.colPrefix(k)) * grid.rows;
            leftLd = grid.rows;
            CHECK(cudaMemcpy2DAsync(rowHost.data, rowCount * sizeof(double), left,
                grid.rows * sizeof(double), rowCount * sizeof(double), width,
                cudaMemcpyDeviceToHost, gpu.stream));
        }
        // Complete any transfers from a previous exchange before reusing its buffers.
        gpu.sync();
        if (rowCount) {
            CHECK(MPI_Bcast(rowHost.data, rowCount * width, MPI_DOUBLE, ownerCol, grid.rowComm));
            if (grid.col != ownerCol && colCount) {
                CHECK(cudaMemcpyAsync(rowDevice.data, rowHost.data, rowCount * width * sizeof(double),
                                      cudaMemcpyHostToDevice, gpu.stream));
            }
        }

        if (!colCount) return;
        if (grid.dims[0] == grid.dims[1]) {
            // With a square grid, the rank on the grid diagonal already has this column view.
            double* source = grid.row == grid.col ? rowHost.data : colHost.data;
            CHECK(MPI_Bcast(source, colCount * width, MPI_DOUBLE, grid.col, grid.colComm));
            if (grid.row == grid.col) {
                right = left;
                rightLd = leftLd;
                return;
            }
        } else {
            exchangeRectangular(firstTile, width, rowStart);
        }
        if (rowCount) {
            CHECK(cudaMemcpyAsync(colDevice.data, colHost.data, colCount * width * sizeof(double),
                                  cudaMemcpyHostToDevice, gpu.stream));
        }
    }

    void update(double* matrix, int firstTile, int width, double alpha, double beta,
                cublasHandle_t blas, int begin, int end) const {
        const int colStart = grid.colPrefix(firstTile);
        // Group neighboring local tiles so GEMMs stay large while skipping most of the upper half.
        constexpr int stripes = 4;
        const int groupWidth = stripes * grid.block;
        for (int offset = begin; offset < end;) {
            const int j = colStart + offset;
            const int globalTile = (j / grid.block) * grid.dims[1] + grid.col;
            const int i = grid.rowPrefix(globalTile);
            const int m = grid.rows - i;
            const int n = std::min(groupWidth, end - offset);
            if (m) {
                CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, width,
                    &alpha, left + (rowCount - m), leftLd, right + offset, rightLd,
                    &beta, matrix + i + static_cast<size_t>(j) * grid.rows, grid.rows));
            }
            offset += n;
        }
    }

    int columns() const { return colCount; }

private:
    size_t panelSize(int length) const {
        if (grid.size == 1) return 0;
        const auto size = static_cast<size_t>(length) * grid.block;
        if (size > INT_MAX) fail("MPI panel size exceeds the supported count", MPI_ERR_COUNT);
        return size;
    }

    void exchangeRectangular(int firstTile, int width, int rowStart) {
        const int colStart = grid.colPrefix(firstTile);
        const int first = (colStart / grid.block) * grid.dims[1] + grid.col;
        std::fill(counts.begin(), counts.end(), 0);
        for (int tile = first; tile < grid.tiles; tile += grid.dims[1]) {
            counts[tile % grid.dims[0]] += grid.width(tile) * width;
        }
        int total = 0;
        for (int row = 0; row < grid.dims[0]; ++row) {
            displacements[row] = total;
            total += counts[row];
        }
        std::copy(displacements.begin(), displacements.end(), cursors.begin());
        const int localTiles = (colCount - 1) / grid.block + 1;
        for (int index = 0; index < localTiles; ++index) {
            const int tile = first + index * grid.dims[1];
            const int owner = tile % grid.dims[0];
            const int height = grid.width(tile);
            offsets[index] = cursors[owner];
            cursors[owner] += height * width;
            if (owner == grid.row) {
                const int i = grid.rowPrefix(tile) - rowStart;
                for (int j = 0; j < width; ++j) {
                    std::memcpy(gathered.data + offsets[index] + j * height,
                                rowHost.data + i + j * rowCount, height * sizeof(double));
                }
            }
        }
        CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, gathered.data, counts.data(),
                            displacements.data(), MPI_DOUBLE, grid.colComm));
        if (!rowCount) return;
        #pragma omp parallel for if(colCount * width >= 65536) schedule(static)
        for (int j = 0; j < width; ++j) {
            for (int index = 0; index < localTiles; ++index) {
                const int start = index * grid.block;
                const int height = std::min(grid.block, colCount - start);
                std::memcpy(colHost.data + start + j * colCount,
                            gathered.data + offsets[index] + j * height, height * sizeof(double));
            }
        }
    }

    const Grid& grid;
    const Gpu& gpu;
    Buffer<double> rowDevice, colDevice;
    Buffer<double, true> rowHost, colHost, gathered;
    Buffer<double> diagDevice;
    Buffer<double, true> diagHost;
    std::vector<int> counts, displacements, cursors, offsets;
    const double* left = nullptr;
    const double* right = nullptr;
    int leftLd = 0, rightLd = 0, rowCount = 0, colCount = 0;
};

struct Lookahead {
    cudaStream_t stream;
    cublasHandle_t blas;
    cudaEvent_t ready, updated;

    Lookahead() {
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cublasSetStream(blas, stream));
        CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&updated, cudaEventDisableTiming));
    }
    ~Lookahead() {
        cudaEventDestroy(updated);
        cudaEventDestroy(ready);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
};

class Factorization {
public:
    Factorization(const Grid& grid, const Gpu& gpu, double* matrix)
        : grid(grid), gpu(gpu), matrix(matrix), panels(grid, gpu), info(1), hostInfo(1),
          workspace(workSize()) {
        if (grid.tiles >= 3) {
            alternate.emplace(grid, gpu);
            lookahead.emplace();
        }
    }

    int run(bool validate) {
        int firstError = INT_MAX;
        for (int k = 0; k < grid.tiles; ++k) {
            const int ownerRow = k % grid.dims[0];
            const int ownerCol = k % grid.dims[1];
            const int width = grid.width(k);
            const bool owner = grid.row == ownerRow && grid.col == ownerCol;
            double* diagonal = nullptr;
            if (owner) {
                diagonal = matrix + grid.rowPrefix(k) + static_cast<size_t>(grid.colPrefix(k)) * grid.rows;
                CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, width, diagonal,
                    grid.rows, workspace.data, workspaceSize, info.data));
                CHECK(cudaMemcpyAsync(hostInfo.data, info.data, sizeof(int), cudaMemcpyDeviceToHost,
                                      gpu.stream));
                if (validate) {
                    const int blocks = std::min<size_t>((static_cast<size_t>(width) * width + 255) / 256,
                                                        65535);
                    clearUpper<<<blocks, 256, 0, gpu.stream>>>(diagonal, width, grid.rows);
                    CHECK(cudaGetLastError());
                }
            }
            if (k + 1 < grid.tiles) {
                Panels& current = alternate && k % 2 ? *alternate : panels;
                current.solve(matrix, k, width, ownerRow, ownerCol, diagonal);
                current.exchange(matrix, k, k + 1);
                updateTrailing(current, k, width);
            } else {
                gpu.sync();
            }
            if (owner) {
                if (*hostInfo.data < 0) fail("cuSOLVER Cholesky arguments", *hostInfo.data);
                if (*hostInfo.data > 0) firstError = std::min(firstError, k * grid.block + *hostInfo.data - 1);
            }
        }
        if (lookahead) CHECK(cudaStreamSynchronize(lookahead->stream));
        int error;
        CHECK(MPI_Allreduce(&firstError, &error, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        return error;
    }

    void reconstruct(double* reconstructed) {
        if (grid.size == 1 && grid.n) {
            const double one = 1.0, zero = 0.0;
            CHECK(cublasDsyrk(gpu.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, grid.n, grid.n,
                &one, matrix, grid.rows, &zero, reconstructed, grid.rows));
        } else {
            if (grid.elements()) {
                CHECK(cudaMemsetAsync(reconstructed, 0, grid.elements() * sizeof(double), gpu.stream));
            }
            for (int k = 0; k < grid.tiles; ++k) {
                panels.exchange(matrix, k, k);
                panels.update(reconstructed, k, grid.width(k), 1.0, 1.0,
                              gpu.blas, 0, panels.columns());
            }
        }
    }

private:
    void updateTrailing(const Panels& current, int k, int width) {
        const int next = k + 1;
        if (!lookahead) {
            current.update(matrix, next, width, -1.0, 1.0, gpu.blas, 0, current.columns());
            return;
        }

        // Prioritize the next panel on the main stream and update other columns separately.
        // Two sets of panel buffers let the next exchange overlap the deferred GEMMs.
        auto& deferred = *lookahead;
        const int critical = grid.col == next % grid.dims[1] ? grid.width(next) : 0;
        CHECK(cudaEventRecord(deferred.ready, gpu.stream));
        CHECK(cudaStreamWaitEvent(deferred.stream, deferred.ready, 0));
        if (k) CHECK(cudaStreamWaitEvent(gpu.stream, deferred.updated, 0));
        current.update(matrix, next, width, -1.0, 1.0, gpu.blas, 0, critical);
        current.update(matrix, next, width, -1.0, 1.0, deferred.blas, critical, current.columns());
        CHECK(cudaEventRecord(deferred.updated, deferred.stream));
    }

    int workSize() {
        if (!grid.elements()) return 0;
        // Workspace is sized for the largest diagonal tile this rank can own.
        for (int k = grid.row; k < grid.tiles; k += grid.dims[0]) {
            if (k % grid.dims[1] == grid.col) {
                CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER,
                    grid.width(k), matrix, grid.rows, &workspaceSize));
                break;
            }
        }
        return workspaceSize;
    }

    const Grid& grid;
    const Gpu& gpu;
    double* matrix;
    Panels panels;
    Buffer<int> info;
    Buffer<int, true> hostInfo;
    int workspaceSize = 0;
    Buffer<double> workspace;
    std::optional<Panels> alternate;
    std::optional<Lookahead> lookahead;
};

__global__ void measureErrors(const double* actual, const double* original, size_t length,
                              int rows, int block, int row, int col, int pr, int pc,
                              unsigned long long* maxima) {
    double absolute = 0.0, relative = 0.0;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < length;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (globalIndex(index % rows, row, pr, block) >= globalIndex(index / rows, col, pc, block)) {
            const double expected = original[index];
            const double error = fabs(actual[index] - expected);
            absolute = fmax(absolute, isfinite(error) ? error : CUDART_INF);
            relative = fmax(relative, isfinite(error) ? error / (fabs(expected) + 1e-10) : CUDART_INF);
        }
    }
    __shared__ double absBlock[256], relBlock[256];
    const int thread = threadIdx.x;
    absBlock[thread] = absolute;
    relBlock[thread] = relative;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            absBlock[thread] = fmax(absBlock[thread], absBlock[thread + offset]);
            relBlock[thread] = fmax(relBlock[thread], relBlock[thread + offset]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        atomicMax(maxima, static_cast<unsigned long long>(__double_as_longlong(absBlock[0])));
        atomicMax(maxima + 1, static_cast<unsigned long long>(__double_as_longlong(relBlock[0])));
    }
}

bool validateFactor(const Grid& grid, const Gpu& gpu, Factorization& factor, const double* original) {
    Buffer<double> reconstructed(grid.elements());
    Buffer<unsigned long long> maxima(2);
    factor.reconstruct(reconstructed.data);
    double local[2] = {};
    if (grid.elements()) {
        CHECK(cudaMemsetAsync(maxima.data, 0, sizeof(local), gpu.stream));
        const int blocks = std::min<size_t>((grid.elements() + 255) / 256, 4096);
        measureErrors<<<blocks, 256, 0, gpu.stream>>>(reconstructed.data, original, grid.elements(),
            grid.rows, grid.block, grid.row, grid.col, grid.dims[0], grid.dims[1], maxima.data);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpyAsync(local, maxima.data, sizeof(local), cudaMemcpyDeviceToHost, gpu.stream));
    }
    gpu.sync();
    double errors[2];
    CHECK(MPI_Allreduce(local, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    const bool valid = errors[1] <= 1e-6;
    if (grid.rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
        if (!valid) printf("Validation failed: relative error too large\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void reportFactor(const Grid& grid, const double* matrix) {
    const int maxRows = prefix(grid.tiles, 0, grid.dims[0], grid.block, grid.n);
    const int maxCols = prefix(grid.tiles, 0, grid.dims[1], grid.block, grid.n);
    Buffer<double, true> local(grid.rank == 0 ? static_cast<size_t>(maxRows) * maxCols : grid.elements());
    if (grid.elements()) {
        CHECK(cudaMemcpy(local.data, matrix, grid.elements() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (grid.rank != 0) {
        for (size_t offset = 0; offset < grid.elements();) {
            const int count = std::min<size_t>(grid.elements() - offset, INT_MAX);
            CHECK(MPI_Send(local.data + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            offset += count;
        }
        return;
    }

    std::vector<double> factor(static_cast<size_t>(grid.n) * grid.n, 0.0);
    for (int rank = 0; rank < grid.size; ++rank) {
        const int row = rank / grid.dims[1];
        const int col = rank % grid.dims[1];
        const int rows = prefix(grid.tiles, row, grid.dims[0], grid.block, grid.n);
        const int cols = prefix(grid.tiles, col, grid.dims[1], grid.block, grid.n);
        const size_t length = static_cast<size_t>(rows) * cols;
        for (size_t offset = 0; rank && offset < length;) {
            const int count = std::min<size_t>(length - offset, INT_MAX);
            CHECK(MPI_Recv(local.data + offset, count, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
            offset += count;
        }
        #pragma omp parallel for if(length >= 16384) schedule(static)
        for (int i = 0; i < rows; ++i) {
            const int globalRow = globalIndex(i, row, grid.dims[0], grid.block);
            const int end = prefix(globalRow / grid.block, col, grid.dims[1], grid.block, grid.n)
                + (globalRow / grid.block % grid.dims[1] == col ? globalRow % grid.block + 1 : 0);
            double* output = factor.data() + static_cast<size_t>(globalRow) * grid.n;
            for (int j = 0; j < end; ++j) {
                output[globalIndex(j, col, grid.dims[1], grid.block)] =
                    local.data[i + static_cast<size_t>(j) * rows];
            }
        }
    }
    print_results(factor, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            const char* value = argv[++i];
            errno = 0;
            const long order = strtol(value, &end, 10);
            if (errno || end == value || *end || order < 0 || order > INT_MAX) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<int>(order);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm localComm;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank, localSize;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    CHECK(MPI_Comm_free(&localComm));
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::clamp(omp_get_num_procs() / localSize, 1, 8));
    }

    const Grid grid(n);
    const Gpu gpu(localRank);
    Buffer<double> matrix(grid.elements());
    Buffer<double> original(validate ? grid.elements() : 0);
    Factorization factor(grid, gpu, matrix.data);
    if (!rank) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    generateMatrix(grid, gpu, matrix.data);
    if (original.data) {
        CHECK(cudaMemcpyAsync(original.data, matrix.data, grid.elements() * sizeof(double),
                              cudaMemcpyDeviceToDevice, gpu.stream));
        gpu.sync();
    }
    if (!rank) printf("Computing Cholesky decomposition...\n");
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const int error = factor.run(validate);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (error != INT_MAX) {
        if (!rank) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", error);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }
    if (!rank) {
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", ops / seconds / 1e9);
    }
    if (printResults) reportFactor(grid, matrix.data);
    if (validate) {
        if (!rank) printf("Validating result...\n");
        return validateFactor(grid, gpu, factor, original.data) ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) fail("initializing MPI with MPI_THREAD_FUNNELED", provided);
    int result;
    try {
        result = benchmark(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what(), 1);
    }
    MPI_Finalize();
    return result;
}
