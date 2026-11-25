#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

void check(int status, const char* operation, const char* detail = "") {
    if (status != 0) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s failed (%d) %s\n", rank, operation, status, detail);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA(call) do { const cudaError_t error = (call); \
    check(error, #call, cudaGetErrorString(error)); } while (false)
#define BLAS(call) check((call), #call)
#define SOLVER(call) check((call), #call)

enum class Memory { device, pinned };

template <typename T, Memory memory = Memory::device>
class Buffer {
public:
    explicit Buffer(size_t size) : data_(nullptr) {
        if (size == 0) return;
        if constexpr (memory == Memory::device) {
            CUDA(cudaMalloc(&data_, size * sizeof(T)));
        } else {
            CUDA(cudaMallocHost(&data_, size * sizeof(T)));
        }
    }

    ~Buffer() {
        if (!data_) return;
        if constexpr (memory == Memory::device) {
            CUDA(cudaFree(data_));
        } else {
            CUDA(cudaFreeHost(data_));
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() const { return data_; }

private:
    T* data_;
};

struct Gpu {
    cudaStream_t stream, trailingStream;
    cudaEvent_t panelReady;
    cublasHandle_t blas, trailingBlas;

    explicit Gpu(int localRank) {
        int devices;
        CUDA(cudaGetDeviceCount(&devices));
        check(devices == 0, "CUDA device selection", "No CUDA device is visible");
        CUDA(cudaSetDevice(localRank % devices));
        int highestPriority;
        CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &highestPriority));
        CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, highestPriority));
        CUDA(cudaStreamCreateWithFlags(&trailingStream, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&panelReady, cudaEventDisableTiming));
        BLAS(cublasCreate(&blas));
        BLAS(cublasSetStream(blas, stream));
        BLAS(cublasCreate(&trailingBlas));
        BLAS(cublasSetStream(trailingBlas, trailingStream));
    }

    ~Gpu() {
        BLAS(cublasDestroy(trailingBlas));
        BLAS(cublasDestroy(blas));
        CUDA(cudaEventDestroy(panelReady));
        CUDA(cudaStreamDestroy(trailingStream));
        CUDA(cudaStreamDestroy(stream));
    }
};

struct Grid {
    int rank, row, col, rows, cols, localRank;
    MPI_Comm rowComm, colComm;

    Grid() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);
    }

    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
};

struct Layout {
    int n, tile, blocks, rows, cols, ld;
    const Grid& grid;

    Layout(int n, const Grid& grid)
        : n(n), tile(n < 2048 ? 128 : 256), blocks((n + tile - 1) / tile),
          rows(extent(blocks, grid.row, grid.rows)),
          cols(extent(blocks, grid.col, grid.cols)), ld(std::max(1, rows)), grid(grid) {}

    int width(int block) const { return std::min(tile, n - block * tile); }

    // Number of local elements preceding a global block index.
    int extent(int block, int coordinate, int stride) const {
        const int count = (block + stride - 1 - coordinate) / stride;
        const int last = (count - 1) * stride + coordinate;
        return count ? (count - 1) * tile + width(last) : 0;
    }

    int rowStart(int block) const { return extent(block, grid.row, grid.rows); }
    int colStart(int block) const { return extent(block, grid.col, grid.cols); }
    size_t elements() const { return static_cast<size_t>(rows) * cols; }

    double* at(double* matrix, int rowBlock, int colBlock) const {
        return matrix + rowStart(rowBlock) + static_cast<size_t>(colStart(colBlock)) * ld;
    }
};

#if defined(__GLIBC__)
// Each glibc rand_r draw advances the 32-bit LCG three times.
unsigned int seedAt(size_t position) {
    uint32_t seed = 42;
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    for (size_t steps = 3 * position; steps != 0; steps >>= 1) {
        if (steps & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return seed;
}
#endif

void fillRandomSlab(double* slab, int localRows, int coordinate, int stride,
                    int start, int depth, const Layout& layout,
                    const std::vector<unsigned int>& rowSeeds) {
    const int threads = std::min(omp_get_max_threads(), std::max(1, localRows / 8));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int r = 0; r < localRows; ++r) {
        const int globalRow = (r / layout.tile * stride + coordinate) * layout.tile
                            + r % layout.tile;
#if defined(__GLIBC__)
        (void)rowSeeds;
        unsigned int seed = seedAt(static_cast<size_t>(globalRow) * layout.n + start);
#else
        unsigned int seed = rowSeeds[globalRow];
        for (int k = 0; k < start; ++k) rand_r(&seed);
#endif
        double* values = slab + static_cast<size_t>(r) * depth;
        for (int k = 0; k < depth; ++k) values[k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
}

__global__ void addDiagonal(double* matrix, int ld, int localCols, int n,
                            int tile, int processRow, int processCol, int rows, int cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < localCols) {
        const int block = c / tile * cols + processCol;
        if (block % rows == processRow) {
            const int r = block / rows * tile + c % tile;
            matrix[r + static_cast<size_t>(c) * ld] += n;
        }
    }
}

void generateMatrix(double* matrix, const Layout& layout, const Gpu& gpu) {
    const auto& grid = layout.grid;
    if (layout.rows == 0 || layout.cols == 0) return;
    const int chunk = std::min(layout.n, 2048);
    const bool sameRows = grid.rows == grid.cols && grid.row == grid.col;
    const size_t leftSize = static_cast<size_t>(layout.rows) * chunk;
    const size_t rightSize = sameRows ? 0 : static_cast<size_t>(layout.cols) * chunk;
    Buffer<double, Memory::pinned> hostLeft(leftSize), hostRight(rightSize);
    Buffer<double> left(leftSize), right(rightSize);
    std::vector<unsigned int> rowSeeds;
#if !defined(__GLIBC__)
    rowSeeds.resize(layout.n);
    unsigned int seed = 42;
    for (int r = 0; r < layout.n; ++r) {
        rowSeeds[r] = seed;
        for (int k = 0; k < layout.n; ++k) rand_r(&seed);
    }
#endif
    const double one = 1.0, zero = 0.0;
    for (int start = 0; start < layout.n;) {
        const int depth = std::min(chunk, layout.n - start);
        fillRandomSlab(hostLeft.data(), layout.rows, grid.row, grid.rows, start, depth,
                       layout, rowSeeds);
        CUDA(cudaMemcpyAsync(left.data(), hostLeft.data(),
                             static_cast<size_t>(layout.rows) * depth * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.stream));
        if (!sameRows) {
            fillRandomSlab(hostRight.data(), layout.cols, grid.col, grid.cols, start, depth,
                           layout, rowSeeds);
            CUDA(cudaMemcpyAsync(right.data(), hostRight.data(),
                                 static_cast<size_t>(layout.cols) * depth * sizeof(double),
                                 cudaMemcpyHostToDevice, gpu.stream));
        }
        BLAS(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, layout.rows, layout.cols,
                         depth, &one, left.data(), depth, sameRows ? left.data() : right.data(),
                         depth, start == 0 ? &zero : &one, matrix, layout.ld));
        CUDA(cudaStreamSynchronize(gpu.stream));
        start += depth;
    }
    addDiagonal<<<(layout.cols - 1) / 256 + 1, 256, 0, gpu.stream>>>(
        matrix, layout.ld, layout.cols, layout.n, layout.tile, grid.row, grid.col,
        grid.rows, grid.cols);
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(gpu.stream));
}

__global__ void clearUpper(double* tile, int ld, int width) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < width * width && i % width < i / width) {
        tile[i % width + static_cast<size_t>(i / width) * ld] = 0.0;
    }
}

__device__ int packedIndex(int row, int col, int width) {
    return col * (2 * width - col + 1) / 2 + row - col;
}

// Keep a small diagonal tile in shared memory throughout the rank-one updates.
__global__ void factorSmallTile(double* tile, int ld, int width, int* info) {
    extern __shared__ double lower[];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    for (int c = warp; c < width; c += warps) {
        for (int r = c + lane; r < width; r += 32) {
            lower[packedIndex(r, c, width)] = tile[r + static_cast<size_t>(c) * ld];
        }
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        double* column = lower + packedIndex(k, k, width);
        const double pivot = column[0];
        // All threads must read the pivot before the diagonal is overwritten.
        __syncthreads();
        if (pivot <= 0.0) {
            if (threadIdx.x == 0) *info = k + 1;
            return;
        }
        const double diagonal = sqrt(pivot);
        for (int offset = threadIdx.x; offset < width - k; offset += blockDim.x) {
            column[offset] = offset == 0 ? diagonal : column[offset] / diagonal;
        }
        __syncthreads();
        for (int c = k + 1 + warp; c < width; c += warps) {
            const double value = column[c - k];
            for (int r = c + lane; r < width; r += 32) {
                const int index = packedIndex(r, c, width);
                lower[index] = fma(-column[r - k], value, lower[index]);
            }
        }
        __syncthreads();
    }
    for (int c = warp; c < width; c += warps) {
        for (int r = lane; r < width; r += 32) {
            tile[r + static_cast<size_t>(c) * ld] = r >= c ? lower[packedIndex(r, c, width)] : 0.0;
        }
    }
    if (threadIdx.x == 0) *info = 0;
}

class Panels {
public:
    Panels(const Layout& layout, const Gpu& gpu)
        : layout(layout), gpu(gpu), grid(layout.grid),
          diagonal(grid.rows > 1 ? static_cast<size_t>(layout.tile) * layout.tile : 0),
          hostDiagonal(grid.rows > 1 ? static_cast<size_t>(layout.tile) * layout.tile : 0),
          hostRows(grid.cols > 1 || grid.rows != grid.cols ? panelSize(layout.rows) : 0),
          deviceRows(grid.cols > 1 ? panelSize(layout.rows) : 0),
          hostColumns(grid.rows > 1 ? panelSize(layout.cols) : 0),
          deviceColumns(grid.rows > 1 ? panelSize(layout.cols) : 0),
          counts(grid.rows), displacements(grid.rows), offsets(layout.blocks) {}

    void stageDiagonal(double* matrix, int k) {
        if (grid.rows == 1 || k + 1 == layout.blocks) return;
        const int width = layout.width(k);
        const size_t pitch = width * sizeof(double);
        CUDA(cudaMemcpy2DAsync(hostDiagonal.data(), pitch, layout.at(matrix, k, k),
                               layout.ld * sizeof(double), pitch, width,
                               cudaMemcpyDeviceToHost, gpu.stream));
    }

    double* broadcastDiagonal(double* matrix, int k) {
        if (grid.rows == 1) return layout.at(matrix, k, k);
        const int width = layout.width(k);
        const bool owner = grid.row == k % grid.rows;
        MPI_Bcast(hostDiagonal.data(), width * width, MPI_DOUBLE, k % grid.rows, grid.colComm);
        if (owner) return layout.at(matrix, k, k);
        if (layout.rows > layout.rowStart(k + 1)) {
            CUDA(cudaMemcpyAsync(diagonal.data(), hostDiagonal.data(),
                                 static_cast<size_t>(width) * width * sizeof(double),
                                 cudaMemcpyHostToDevice, gpu.stream));
        }
        return diagonal.data();
    }

    void exchange(double* matrix, int k, int first) {
        firstRow = layout.rowStart(first);
        firstCol = layout.colStart(first);
        rowLeading = layout.rows - firstRow;
        colLeading = layout.cols - firstCol;
        width = layout.width(k);
        rowPanel = nullptr;
        if (rowLeading > 0 && grid.col == k % grid.cols) {
            rowPanel = layout.at(matrix, first, k);
        }
        if (grid.rows == 1 && grid.cols == 1) {
            rowLeading = layout.ld;
            colLeading = layout.ld;
            columnPanel = rowPanel;
            return;
        }

        // A row broadcast distributes the solved block column across process columns.
        const int rowCount = rowLeading * width;
        if (rowCount > 0) {
            const size_t pitch = rowLeading * sizeof(double);
            if (rowPanel) {
                CUDA(cudaMemcpy2DAsync(hostRows.data(), pitch, rowPanel,
                                       layout.ld * sizeof(double), pitch, width,
                                       cudaMemcpyDeviceToHost, gpu.stream));
                CUDA(cudaStreamSynchronize(gpu.stream));
            }
            MPI_Bcast(hostRows.data(), rowCount, MPI_DOUBLE, k % grid.cols, grid.rowComm);
            if (rowPanel) {
                rowLeading = layout.ld;
            } else {
                CUDA(cudaMemcpyAsync(deviceRows.data(), hostRows.data(), rowCount * sizeof(double),
                                     cudaMemcpyHostToDevice, gpu.stream));
                rowPanel = deviceRows.data();
            }
        }

        // The other GEMM operand needs rows indexed by local matrix columns.
        // On a square grid these already match on the diagonal ranks.
        const int colCount = colLeading * width;
        if (colCount == 0) return;
        if (grid.rows == grid.cols) {
            double* staging = grid.row == grid.col ? hostRows.data() : hostColumns.data();
            MPI_Bcast(staging, colCount, MPI_DOUBLE, grid.col, grid.colComm);
            if (grid.row == grid.col) {
                columnPanel = rowPanel;
                colLeading = rowLeading;
                return;
            }
        } else {
            gatherColumns(first);
        }
        CUDA(cudaMemcpyAsync(deviceColumns.data(), hostColumns.data(),
                             colCount * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
        columnPanel = deviceColumns.data();
    }

    void update(double* target, int first, int end, double scale, bool lowerOnly,
                cublasHandle_t blas) const {
        const int localRows = layout.rows - firstRow;
        const int localCols = layout.cols - firstCol;
        if (localRows == 0 || localCols == 0) return;
        const double one = 1.0;
        const bool square = grid.rows == grid.cols;
        if (!lowerOnly && square) {
            BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, localRows, localCols, width,
                             &scale, rowPanel, rowLeading, columnPanel, colLeading, &one,
                             target + firstRow + static_cast<size_t>(firstCol) * layout.ld,
                             layout.ld));
            return;
        }
        for (int j = first + (grid.col - first % grid.cols + grid.cols) % grid.cols; j < end;) {
            const int start = lowerOnly ? layout.rowStart(j) : firstRow;
            if (start == layout.rows) break;
            // Amortize launches and reuse the left operand on long trailing
            // columns. The extra entries above these diagonal tiles are unused.
            const int group = square && layout.rows - start >= 8 * layout.tile
                            ? std::min(4, (end - 1 - j) / grid.cols + 1) : 1;
            const int last = j + (group - 1) * grid.cols;
            const int outputWidth = layout.colStart(last) - layout.colStart(j)
                                  + layout.width(last);
            const double* right = columnPanel
                + (square ? layout.colStart(j) - firstCol : offsets[j]);
            const int rightLeading = square ? colLeading : layout.width(j);
            BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, layout.rows - start,
                             outputWidth, width, &scale, rowPanel + start - firstRow,
                             rowLeading, right, rightLeading, &one, target + start
                             + static_cast<size_t>(layout.colStart(j)) * layout.ld, layout.ld));
            j += group * grid.cols;
        }
    }

private:
    size_t panelSize(int elements) const {
        const size_t size = static_cast<size_t>(elements) * layout.tile;
        check(size > INT_MAX, "MPI panel size", "Panel exceeds the MPI count limit");
        return size;
    }

    void gatherColumns(int first) {
        std::fill(counts.begin(), counts.end(), 0);
        const int firstBlock = first + (grid.col - first % grid.cols + grid.cols) % grid.cols;
        for (int j = firstBlock; j < layout.blocks; j += grid.cols) {
            const int owner = j % grid.rows;
            offsets[j] = counts[owner];
            counts[owner] += layout.width(j) * width;
        }
        int displacement = 0;
        for (int r = 0; r < grid.rows; ++r) {
            displacements[r] = displacement;
            displacement += counts[r];
        }
        const int blocks = (layout.blocks - 1 - firstBlock) / grid.cols + 1;
        const int threads = std::min(omp_get_max_threads(), std::max(1, blocks / grid.rows));
        const int hostLeading = layout.rows - firstRow;
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int j = firstBlock; j < layout.blocks; j += grid.cols) {
            const int owner = j % grid.rows;
            offsets[j] += displacements[owner];
            if (owner == grid.row) {
                const int height = layout.width(j);
                const double* source = hostRows.data() + layout.rowStart(j) - firstRow;
                double* destination = hostColumns.data() + offsets[j];
                for (int c = 0; c < width; ++c) {
                    std::copy_n(source + static_cast<size_t>(c) * hostLeading, height,
                                destination + c * height);
                }
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostColumns.data(),
                       counts.data(), displacements.data(), MPI_DOUBLE, grid.colComm);
    }

    const Layout& layout;
    const Gpu& gpu;
    const Grid& grid;
    Buffer<double> diagonal;
    Buffer<double, Memory::pinned> hostDiagonal, hostRows;
    Buffer<double> deviceRows;
    Buffer<double, Memory::pinned> hostColumns;
    Buffer<double> deviceColumns;
    std::vector<int> counts, displacements, offsets;
    int firstRow, firstCol, width, rowLeading, colLeading;
    double* rowPanel;
    double* columnPanel;
};

class Factorization {
public:
    Factorization(const Layout& layout, const Gpu& gpu, double* matrix)
        : layout(layout), gpu(gpu), panels(layout, gpu), info(1),
          hostInfo(1), smallTileLimit(configureSmallTile()), workspace(workspaceSize(matrix)) {}

    ~Factorization() {
        if (solver) SOLVER(cusolverDnDestroy(solver));
    }

    bool run(double* matrix) {
        const auto& grid = layout.grid;
        const double one = 1.0;
        for (int k = 0; k < layout.blocks; ++k) {
            const int rootRow = k % grid.rows;
            const int rootCol = k % grid.cols;
            const int root = rootRow * grid.cols + rootCol;
            const int width = layout.width(k);
            if (grid.rank == root) {
                double* diagonal = layout.at(matrix, k, k);
                if (width <= smallTileLimit) {
                    factorSmallTile<<<1, 256, sharedBytes(width), gpu.stream>>>(
                        diagonal, layout.ld, width, info.data());
                } else {
                    SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width,
                                           diagonal, layout.ld, workspace.data(), workSize,
                                           info.data()));
                    clearUpper<<<(width * width - 1) / 256 + 1, 256, 0, gpu.stream>>>(
                        diagonal, layout.ld, width);
                }
                CUDA(cudaGetLastError());
                panels.stageDiagonal(matrix, k);
                CUDA(cudaMemcpyAsync(hostInfo.data(), info.data(), sizeof(int),
                                     cudaMemcpyDeviceToHost, gpu.stream));
                CUDA(cudaStreamSynchronize(gpu.stream));
            }
            MPI_Bcast(hostInfo.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
            check(*hostInfo.data() < 0, "cuSOLVER Cholesky factorization");
            if (*hostInfo.data() > 0) {
                if (grid.rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %d\n",
                           k * layout.tile + *hostInfo.data() - 1);
                }
                CUDA(cudaStreamSynchronize(gpu.trailingStream));
                return false;
            }
            if (k + 1 == layout.blocks) break;
            if (grid.col == rootCol) {
                const double* diagonal = panels.broadcastDiagonal(matrix, k);
                const int height = layout.rows - layout.rowStart(k + 1);
                if (height > 0) {
                    const int diagonalLd = grid.row == rootRow ? layout.ld : width;
                    BLAS(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, height, width, &one,
                                     diagonal, diagonalLd, layout.at(matrix, k + 1, k),
                                     layout.ld));
                }
            }
            // The previous bulk update can run during POTRF and TRSM above.
            // Finish it before publishing new panel data into the shared buffers.
            CUDA(cudaStreamSynchronize(gpu.trailingStream));
            panels.exchange(matrix, k, k + 1);
            CUDA(cudaEventRecord(gpu.panelReady, gpu.stream));
            CUDA(cudaStreamWaitEvent(gpu.trailingStream, gpu.panelReady, 0));
            panels.update(matrix, k + 1, k + 2, -1.0, true, gpu.blas);
            panels.update(matrix, k + 2, layout.blocks, -1.0, true, gpu.trailingBlas);
        }
        CUDA(cudaStreamSynchronize(gpu.stream));
        CUDA(cudaStreamSynchronize(gpu.trailingStream));
        return true;
    }

    void reconstruct(double* reconstructed, double* matrix) {
        if (layout.elements() > 0) {
            CUDA(cudaMemsetAsync(reconstructed, 0, layout.elements() * sizeof(double), gpu.stream));
        }
        for (int k = 0; k < layout.blocks; ++k) {
            panels.exchange(matrix, k, k);
            panels.update(reconstructed, k, layout.blocks, 1.0, false, gpu.blas);
            CUDA(cudaStreamSynchronize(gpu.stream));
        }
    }

private:
    static int sharedBytes(int width) {
        return width * (width + 1) / 2 * sizeof(double);
    }

    int configureSmallTile() const {
        int device, available, defaultLimit;
        CUDA(cudaGetDevice(&device));
        CUDA(cudaDeviceGetAttribute(&defaultLimit, cudaDevAttrMaxSharedMemoryPerBlock, device));
        CUDA(cudaDeviceGetAttribute(&available, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        available = std::max(defaultLimit, available);
        int limit = std::min({layout.tile, layout.n, 128});
        while (sharedBytes(limit) > available) --limit;
        if (sharedBytes(limit) > defaultLimit) {
            CUDA(cudaFuncSetAttribute(factorSmallTile, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      sharedBytes(limit)));
        }
        return limit;
    }

    int workspaceSize(double* matrix) {
        const auto& grid = layout.grid;
        for (int k = grid.row; k < layout.blocks; k += grid.rows) {
            const int width = layout.width(k);
            if (k % grid.cols == grid.col && width > smallTileLimit) {
                SOLVER(cusolverDnCreate(&solver));
                SOLVER(cusolverDnSetStream(solver, gpu.stream));
                SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, width,
                                                 layout.at(matrix, k, k), layout.ld, &workSize));
                break;
            }
        }
        return workSize;
    }

    const Layout& layout;
    const Gpu& gpu;
    Panels panels;
    Buffer<int> info;
    Buffer<int, Memory::pinned> hostInfo;
    int smallTileLimit;
    cusolverDnHandle_t solver = nullptr;
    int workSize = 0;
    Buffer<double> workspace;
};

__global__ void measureError(const double* reconstructed, const double* original,
                             int localRows, size_t count, int tile, int processRow,
                             int processCol, int rows, int cols, unsigned long long* maxima) {
    double absolute = 0.0, relative = 0.0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int r = i % localRows;
        const int c = i / localRows;
        const int globalRow = (r / tile * rows + processRow) * tile + r % tile;
        const int globalCol = (c / tile * cols + processCol) * tile + c % tile;
        if (globalRow >= globalCol) {
            const double error = fabs(reconstructed[i] - original[i]);
            absolute = fmax(absolute, isfinite(error) ? error : CUDART_INF);
            relative = fmax(relative, isfinite(error) ? error / (fabs(original[i]) + 1e-10)
                                                      : CUDART_INF);
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        absolute = fmax(absolute, __shfl_down_sync(0xffffffff, absolute, offset));
        relative = fmax(relative, __shfl_down_sync(0xffffffff, relative, offset));
    }
    if (threadIdx.x % 32 == 0) {
        atomicMax(maxima, static_cast<unsigned long long>(__double_as_longlong(absolute)));
        atomicMax(maxima + 1, static_cast<unsigned long long>(__double_as_longlong(relative)));
    }
}

bool validateResult(const Layout& layout, const Gpu& gpu, const double* reconstructed,
                    const double* original) {
    double local[2] = {};
    if (layout.rows > 0 && layout.cols > 0) {
        Buffer<unsigned long long> maxima(2);
        CUDA(cudaMemsetAsync(maxima.data(), 0, 2 * sizeof(unsigned long long), gpu.stream));
        const auto& grid = layout.grid;
        const int blocks = std::min(size_t{1024}, (layout.elements() + 255) / 256);
        measureError<<<blocks, 256, 0, gpu.stream>>>(reconstructed, original, layout.rows,
            layout.elements(), layout.tile, grid.row, grid.col, grid.rows, grid.cols,
            maxima.data());
        CUDA(cudaGetLastError());
        CUDA(cudaMemcpyAsync(local, maxima.data(), sizeof(local), cudaMemcpyDeviceToHost,
                             gpu.stream));
        CUDA(cudaStreamSynchronize(gpu.stream));
    }
    double global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (layout.grid.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void reportFactor(double* matrix, const Layout& layout, const Gpu& gpu) {
    const auto& grid = layout.grid;
    std::vector<double> factor;
    if (grid.rank == 0) factor.resize(static_cast<size_t>(layout.n) * layout.n);
    const int capacity = grid.rank == 0 ? layout.n : layout.rows;
    Buffer<double, Memory::pinned> host(static_cast<size_t>(capacity) * layout.tile);
    // Transfer one block column at a time, without allocating host copies on every rank.
    for (int j = 0; j < layout.blocks; ++j) {
        const int width = layout.width(j);
        for (int r = 0; r < grid.rows; ++r) {
            const int first = layout.extent(j, r, grid.rows);
            const int height = layout.extent(layout.blocks, r, grid.rows) - first;
            if (height == 0) continue;
            const int owner = r * grid.cols + j % grid.cols;
            if (grid.rank == owner) {
                const size_t pitch = height * sizeof(double);
                CUDA(cudaMemcpy2DAsync(host.data(), pitch, layout.at(matrix, j, j),
                                       layout.ld * sizeof(double), pitch, width,
                                       cudaMemcpyDeviceToHost, gpu.stream));
                CUDA(cudaStreamSynchronize(gpu.stream));
                if (owner != 0) {
                    MPI_Send(host.data(), height * width, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
            if (grid.rank == 0) {
                if (owner != 0) {
                    MPI_Recv(host.data(), height * width, MPI_DOUBLE, owner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                const int threads = std::min(omp_get_max_threads(), std::max(1, height / 32));
                #pragma omp parallel for schedule(static) num_threads(threads)
                for (int i = 0; i < height; ++i) {
                    const int local = first + i;
                    const int global = (local / layout.tile * grid.rows + r) * layout.tile
                                     + local % layout.tile;
                    double* destination = factor.data() + static_cast<size_t>(global) * layout.n
                                        + j * layout.tile;
                    for (int c = 0; c < width; ++c) {
                        destination[c] = host.data()[i + static_cast<size_t>(c) * height];
                    }
                }
            }
        }
    }
    if (grid.rank == 0) print_results(factor, "CholeskyL");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int rank) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            const char* value = argv[++i];
            const long parsed = strtol(value, &end, 10);
            if (errno || end == value || *end || parsed < 0 || parsed > INT_MAX / 256) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<int>(parsed);
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
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const Grid grid;
    const Gpu gpu(grid.localRank);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
    }

    const Layout layout(n, grid);
    Buffer<double> matrix(layout.elements());
    generateMatrix(matrix.data(), layout, gpu);
    Buffer<double> original(validate ? layout.elements() : 0);
    if (validate && layout.elements() > 0) {
        CUDA(cudaMemcpyAsync(original.data(), matrix.data(), layout.elements() * sizeof(double),
                             cudaMemcpyDeviceToDevice, gpu.stream));
        CUDA(cudaStreamSynchronize(gpu.stream));
    }
    Factorization factorization(layout, gpu, matrix.data());
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = factorization.run(matrix.data());
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }

    if (printResults) reportFactor(matrix.data(), layout, gpu);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        Buffer<double> reconstructed(layout.elements());
        factorization.reconstruct(reconstructed.data(), matrix.data());
        const bool valid = validateResult(layout, gpu, reconstructed.data(), original.data());
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    check(provided < MPI_THREAD_FUNNELED, "MPI thread initialization");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 1;
    try {
        result = benchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        check(1, "Cholesky benchmark", error.what());
    }
    MPI_Finalize();
    return result;
}
