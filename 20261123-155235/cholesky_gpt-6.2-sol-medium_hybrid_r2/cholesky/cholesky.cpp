#include <algorithm>
#include <array>
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
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr uint32_t rngMultiplier = 1103515245u;
constexpr uint32_t rngIncrement = 12345u;

[[noreturn]] void fail(const char* operation, const char* reason) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, reason);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fail("CUDA", cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail("cuBLAS", cublasGetStatusString(status));
    }
}

void check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char reason[32];
        snprintf(reason, sizeof(reason), "status %d", static_cast<int>(status));
        fail("cuSOLVER", reason);
    }
}

template <typename T, bool Pinned = false>
class Buffer {
public:
    explicit Buffer(size_t count) {
        if (count == 0) {
            return;
        }
        if (count > SIZE_MAX / sizeof(T)) {
            fail("CUDA", "allocation size exceeds address space");
        }
        if constexpr (Pinned) {
            check(cudaMallocHost(&data, count * sizeof(T)));
        } else {
            check(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    ~Buffer() {
        if (data) {
            if constexpr (Pinned) {
                cudaFreeHost(data);
            } else {
                cudaFree(data);
            }
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data = nullptr;
};

[[maybe_unused]] int threadsFor(size_t work) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + work / 32768));
}

struct Grid {
    explicit Grid(int n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int dims[2] = {};
        MPI_Dims_create(size, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

        // Launcher-masked GPUs appear as device zero to their assigned rank.
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int devices;
        check(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fail("CUDA", "no visible GPUs");
        }
        check(cudaSetDevice(localRank % devices));

        block = 256;
        while (block > 64 && static_cast<long long>(n) < 2LL * block * std::max(rows, cols)) {
            block /= 2;
        }
    }

    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }

    int rank, size, rows, cols, row, col, block;
    MPI_Comm rowComm, colComm;
};

struct Axis {
    int n, block, period, coord;

    int tiles() const { return n / block + (n % block != 0); }
    int width(int tile) const { return std::min(block, n - tile * block); }

    int firstTile(int tile) const {
        return (tile + period - 1 - coord) / period * period + coord;
    }

    int count() const {
        const int full = n / block;
        return (firstTile(full) / period) * block + (full % period == coord ? n % block : 0);
    }

    // Local offset of the first element in any tile at or after the given tile.
    int offset(int tile) const {
        const int next = firstTile(tile);
        return next < tiles() ? next / period * block : count();
    }

    __host__ __device__ int global(int index) const {
        return (index / block * period + coord) * block + index % block;
    }
};

#ifdef __GLIBC__
// glibc rand_r advances a 32-bit affine state three times per sample.
// Jump to each row's initial state while keeping the original random sequence.
unsigned int seedAt(uint64_t index) {
    uint32_t seed = 42;
    uint32_t multiplier = rngMultiplier;
    uint32_t increment = rngIncrement;
    for (uint64_t steps = 3 * index; steps != 0; steps >>= 1) {
        if (steps & 1) {
            seed = multiplier * seed + increment;
        }
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return seed;
}
#endif

unsigned int nextRandom(unsigned int& state) {
#ifdef __GLIBC__
    // Match rand_r's three output fields and allow SIMD across independent rows.
    uint32_t x = rngMultiplier * state + rngIncrement;
    const unsigned int high = (x >> 16) & 2047u;
    x = rngMultiplier * x + rngIncrement;
    const unsigned int middle = (x >> 16) & 1023u;
    x = rngMultiplier * x + rngIncrement;
    state = x;
    return (high << 20) | (middle << 10) | ((x >> 16) & 1023u);
#else
    return rand_r(&state);
#endif
}

void initializeSeeds(const Axis& axis, std::vector<unsigned int>& seeds) {
#ifdef __GLIBC__
    #pragma omp parallel for schedule(static) num_threads(threadsFor(seeds.size() * 64))
    for (int i = 0; i < axis.count(); ++i) {
        seeds[i] = seedAt(static_cast<uint64_t>(axis.global(i)) * axis.n);
    }
#else
    unsigned int seed = 42;
    int previous = 0;
    for (int i = 0; i < axis.count(); ++i) {
        const int row = axis.global(i);
        for (uint64_t steps = static_cast<uint64_t>(row - previous) * axis.n; steps; --steps) {
            rand_r(&seed);
        }
        seeds[i] = seed;
        previous = row;
    }
#endif
}

void generateSlab(double* slab, std::vector<unsigned int>& seeds, int depth) {
    const int count = static_cast<int>(seeds.size());
    // Transpose small groups of RNG rows into column-major storage for GEMM.
    constexpr int group = 16;
    #pragma omp parallel for schedule(static) num_threads(threadsFor(static_cast<size_t>(count) * depth))
    for (int first = 0; first < count; first += group) {
        const int width = std::min(group, count - first);
        unsigned int state[group];
        std::copy_n(seeds.data() + first, width, state);
        for (int j = 0; j < depth; ++j) {
            double* column = slab + static_cast<size_t>(j) * count + first;
            #pragma omp simd
            for (int i = 0; i < width; ++i) {
                column[i] = nextRandom(state[i]) / static_cast<double>(RAND_MAX) - 0.5;
            }
        }
        std::copy_n(state, width, seeds.data() + first);
    }
}

__global__ void prepareDiagonal(double* matrix, int ld, int n, int block,
                                int first, int period, int rowPeriod, int colPeriod,
                                bool zeroUpper) {
    const int tile = first + blockIdx.x * period;
    const int width = min(block, n - tile * block);
    const size_t offset = static_cast<size_t>(tile / rowPeriod) * block;
    const size_t column = static_cast<size_t>(tile / colPeriod) * block + threadIdx.x;
    if (threadIdx.x >= width) {
        return;
    }
    double* diagonalColumn = matrix + column * ld + offset;
    if (zeroUpper) {
        for (int i = 0; i < threadIdx.x; ++i) {
            diagonalColumn[i] = 0.0;
        }
    } else {
        diagonalColumn[threadIdx.x] += n;
    }
}

__global__ void zeroUpperTriangle(double* matrix, int n, size_t elements) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (i % n < i / n) {
            matrix[i] = 0.0;
        }
    }
}

__global__ void residual(const double* reconstructed, const double* original, double* errors,
                         Axis rows, Axis cols, int ld, size_t elements) {
    constexpr int blockSize = 256;
    __shared__ double absMax[blockSize];
    __shared__ double relMax[blockSize];
    double absolute = 0.0;
    double relative = 0.0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (rows.global(i % ld) >= cols.global(i / ld)) {
            const double error = fabs(reconstructed[i] - original[i]);
            absolute = fmax(absolute, error);
            relative = fmax(relative, error / (fabs(original[i]) + 1e-10));
        }
    }
    absMax[threadIdx.x] = absolute;
    relMax[threadIdx.x] = relative;
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            absMax[threadIdx.x] = fmax(absMax[threadIdx.x], absMax[threadIdx.x + stride]);
            relMax[threadIdx.x] = fmax(relMax[threadIdx.x], relMax[threadIdx.x + stride]);
        }
    }
    if (threadIdx.x == 0) {
        errors[2 * blockIdx.x] = absMax[0];
        errors[2 * blockIdx.x + 1] = relMax[0];
    }
}

class Cholesky {
public:
    Cholesky(const Grid& grid, int n, bool validate)
        : grid(grid), rows{n, grid.block, grid.rows, grid.row},
          cols{n, grid.block, grid.cols, grid.col}, ld(rows.count()),
          elements(static_cast<size_t>(ld) * cols.count()), matrix(elements),
          original(validate ? elements : 0), info(rows.tiles()),
          diagonal(grid.rows > 1 ? tileElements() : 0),
          hostDiagonal(grid.rows > 1 ? tileElements() : 0),
          rowPanel(grid.cols > 1 ? static_cast<size_t>(ld) * rows.block : 0),
          hostRows(grid.size > 1 ? static_cast<size_t>(ld) * rows.block : 0),
          colPanel(grid.rows > 1 ? static_cast<size_t>(cols.count()) * rows.block : 0),
          hostCols(grid.rows > 1 ? static_cast<size_t>(cols.count()) * rows.block : 0),
          colOffsets(cols.tiles()), groupOffsets(grid.rows + 1), requests(grid.rows) {
        int priority;
        check(cudaDeviceGetStreamPriorityRange(nullptr, &priority));
        check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
        check(cublasCreate(&blas));
        check(cublasSetStream(blas, stream));
        check(cusolverDnCreate(&solver));
        check(cusolverDnSetStream(solver, stream));
        if (grid.size > 1) {
            check(cudaStreamCreateWithFlags(&updateStream, cudaStreamNonBlocking));
            check(cublasCreate(&updateBlas));
            check(cublasSetStream(updateBlas, updateStream));
            for (auto event : {&diagonalUploaded, &rowsUploaded, &colsUploaded,
                               &panelReady, &updatesDone}) {
                check(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
            }
        }
    }

    ~Cholesky() {
        if (grid.size > 1) {
            for (auto event : {diagonalUploaded, rowsUploaded, colsUploaded, panelReady, updatesDone}) {
                cudaEventDestroy(event);
            }
            cublasDestroy(updateBlas);
            cudaStreamDestroy(updateStream);
        }
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }

    void generate() {
        if (elements == 0) {
            return;
        }
        const bool sameAxis = grid.rows == grid.cols && grid.row == grid.col;
        const int slabDepth = std::min(rows.n, 2048);
        Buffer<double> bRows(static_cast<size_t>(ld) * slabDepth);
        Buffer<double> bCols(sameAxis ? 0 : static_cast<size_t>(cols.count()) * slabDepth);
        // Double buffering overlaps OpenMP RNG with the preceding GPU work.
        Buffer<double, true> hostRowSlabs[2] = {
            Buffer<double, true>(static_cast<size_t>(ld) * slabDepth),
            Buffer<double, true>(static_cast<size_t>(ld) * slabDepth)
        };
        Buffer<double, true> hostColSlabs[2] = {
            Buffer<double, true>(sameAxis ? 0 : static_cast<size_t>(cols.count()) * slabDepth),
            Buffer<double, true>(sameAxis ? 0 : static_cast<size_t>(cols.count()) * slabDepth)
        };
        cudaEvent_t uploaded[2];
        for (auto& event : uploaded) {
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        std::vector<unsigned int> rowSeeds(ld), colSeeds(sameAxis ? 0 : cols.count());
        initializeSeeds(rows, rowSeeds);
        if (!sameAxis) {
            initializeSeeds(cols, colSeeds);
        }

        for (int start = 0, slot = 0; start < rows.n; slot ^= 1) {
            const int depth = std::min(slabDepth, rows.n - start);
            if (start >= 2 * slabDepth) {
                check(cudaEventSynchronize(uploaded[slot]));
            }
            generateSlab(hostRowSlabs[slot].data, rowSeeds, depth);
            upload(bRows.data, hostRowSlabs[slot].data, static_cast<size_t>(ld) * depth);
            if (!sameAxis) {
                generateSlab(hostColSlabs[slot].data, colSeeds, depth);
                upload(bCols.data, hostColSlabs[slot].data, static_cast<size_t>(cols.count()) * depth);
            }
            check(cudaEventRecord(uploaded[slot], stream));
            const double alpha = 1.0;
            const double beta = start == 0 ? 0.0 : 1.0;
            if (grid.size == 1) {
                check(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, ld, depth,
                                  &alpha, bRows.data, ld, &beta, matrix.data, ld));
            } else {
                const double* right = sameAxis ? bRows.data : bCols.data;
                for (int tile = grid.col; tile < cols.tiles(); tile += grid.cols) {
                    const int rowStart = rows.offset(tile);
                    const int colStart = cols.offset(tile);
                    multiply(blas, ld - rowStart, cols.width(tile), depth, alpha,
                             bRows.data + rowStart, ld, right + colStart, cols.count(),
                             beta, matrix.data + static_cast<size_t>(colStart) * ld + rowStart);
                }
            }
            start += depth;
        }
        adjustDiagonal(false);
        if (original.data) {
            check(cudaMemcpyAsync(original.data, matrix.data, elements * sizeof(double),
                                  cudaMemcpyDeviceToDevice, stream));
        }
        check(cudaStreamSynchronize(stream));
        for (auto event : uploaded) {
            check(cudaEventDestroy(event));
        }
    }

    void prepareFactorization() {
        if (rows.tiles() == 0) {
            return;
        }
        int workSize = 0;
        if (grid.size == 1 && rows.n != 0) {
            check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                             rows.n, matrix.data, ld, &workSize));
        } else {
            for (int tile = 0; tile < rows.tiles(); ++tile) {
                if (tile % grid.rows == grid.row && tile % grid.cols == grid.col) {
                    int needed;
                    check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                                                     rows.width(tile), tilePointer(tile), ld, &needed));
                    workSize = std::max(workSize, needed);
                }
            }
        }
        workspaceSize = workSize;
        if (workSize > 0) {
            check(cudaMalloc(&workspace.data, static_cast<size_t>(workSize) * sizeof(double)));
        }
        check(cudaMemsetAsync(info.data, 0, static_cast<size_t>(rows.tiles()) * sizeof(int), stream));
        check(cudaStreamSynchronize(stream));
    }

    int factor() {
        if (grid.size == 1 && rows.n != 0) {
            potrf(0, rows.n);
        } else {
            for (int tile = 0; tile < rows.tiles(); ++tile) {
                factorPanel(tile);
                if (tile + 1 < rows.tiles()) {
                    applyPanel(tile, tile + 1, -1.0, matrix.data, true);
                }
            }
            if (updateStream) {
                check(cudaStreamSynchronize(updateStream));
            }
        }

        int firstFailure = INT_MAX;
        if (rows.tiles() != 0) {
            std::vector<int> status(rows.tiles());
            download(status.data(), info.data, status.size());
            check(cudaStreamSynchronize(stream));
            for (int tile = 0; tile < rows.tiles(); ++tile) {
                if (status[tile] < 0) {
                    fail("cuSOLVER", "invalid POTRF argument");
                }
                if (status[tile] > 0) {
                    firstFailure = std::min(firstFailure, tile * rows.block + status[tile] - 1);
                }
            }
        }
        int failure;
        MPI_Allreduce(&firstFailure, &failure, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return failure;
    }

    std::array<double, 2> validate() {
        std::array<double, 2> local{};
        if (grid.size == 1 && elements != 0) {
            constexpr int threads = 256;
            const int blocks = std::min<size_t>(1024, (elements + threads - 1) / threads);
            zeroUpperTriangle<<<blocks, threads, 0, stream>>>(matrix.data, ld, elements);
            check(cudaGetLastError());
        } else {
            adjustDiagonal(true);
        }
        Buffer<double> reconstructed(elements);
        if (elements) {
            check(cudaMemsetAsync(reconstructed.data, 0, elements * sizeof(double), stream));
        }
        if (grid.size == 1 && ld != 0) {
            const double one = 1.0;
            check(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, ld, ld, &one,
                              matrix.data, ld, &one, reconstructed.data, ld));
        } else {
            for (int tile = 0; tile < rows.tiles(); ++tile) {
                applyPanel(tile, tile, 1.0, reconstructed.data, false);
            }
        }
        if (elements) {
            constexpr int threads = 256;
            const int blocks = std::min<size_t>(1024, (elements + threads - 1) / threads);
            Buffer<double> deviceErrors(2 * blocks);
            std::vector<double> hostErrors(2 * blocks);
            residual<<<blocks, threads, 0, stream>>>(reconstructed.data, original.data,
                                                     deviceErrors.data, rows, cols, ld, elements);
            check(cudaGetLastError());
            download(hostErrors.data(), deviceErrors.data, hostErrors.size());
            check(cudaStreamSynchronize(stream));
            for (int i = 0; i < blocks; ++i) {
                local[0] = std::max(local[0], hostErrors[2 * i]);
                local[1] = std::max(local[1], hostErrors[2 * i + 1]);
            }
        }
        std::array<double, 2> global{};
        MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return global;
    }

    void printResult() {
        std::vector<double> local(elements);
        if (elements) {
            download(local.data(), matrix.data, elements);
            check(cudaStreamSynchronize(stream));
        }
        constexpr int chunk = 1 << 25;
        if (grid.rank != 0) {
            for (size_t start = 0; start < elements; start += chunk) {
                const int count = std::min<size_t>(chunk, elements - start);
                MPI_Send(local.data() + start, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            return;
        }

        std::vector<double> result(static_cast<size_t>(rows.n) * rows.n, 0.0);
        std::vector<double> incoming;
        for (int rank = 0; rank < grid.size; ++rank) {
            const Axis sourceRows{rows.n, rows.block, grid.rows, rank / grid.cols};
            const Axis sourceCols{cols.n, cols.block, grid.cols, rank % grid.cols};
            const int height = sourceRows.count();
            const int width = sourceCols.count();
            const double* source = local.data();
            if (rank != 0) {
                const size_t count = static_cast<size_t>(height) * width;
                incoming.resize(count);
                for (size_t start = 0; start < count; start += chunk) {
                    const int length = std::min<size_t>(chunk, count - start);
                    MPI_Recv(incoming.data() + start, length, MPI_DOUBLE, rank, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                source = incoming.data();
            }
            #pragma omp parallel for schedule(static) num_threads(threadsFor(static_cast<size_t>(height) * width))
            for (int i = 0; i < height; ++i) {
                const int globalRow = sourceRows.global(i);
                double* output = result.data() + static_cast<size_t>(globalRow) * rows.n;
                for (int j = 0; j < width; ++j) {
                    const int globalCol = sourceCols.global(j);
                    if (globalCol > globalRow) {
                        break;
                    }
                    output[globalCol] = source[static_cast<size_t>(j) * height + i];
                }
            }
        }
        print_results(result, "CholeskyL");
    }

private:
    size_t tileElements() const { return static_cast<size_t>(rows.block) * rows.block; }

    template <typename T>
    void download(T* host, const T* device, size_t count) {
        check(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    void upload(double* device, const double* host, size_t count) {
        check(cudaMemcpyAsync(device, host, count * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    void uploadPanel(double* device, const double* host, size_t count, cudaEvent_t ready) {
        upload(device, host, count);
        check(cudaEventRecord(ready, stream));
    }

    void downloadTile(double* host, const double* device, int height, int width) {
        check(cudaMemcpy2DAsync(host, static_cast<size_t>(height) * sizeof(double),
                               device, static_cast<size_t>(ld) * sizeof(double),
                               static_cast<size_t>(height) * sizeof(double), width,
                               cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
    }

    double* tilePointer(int tile) const {
        return matrix.data + static_cast<size_t>(cols.offset(tile)) * ld + rows.offset(tile);
    }

    void multiply(cublasHandle_t handle, int height, int width, int depth, double alpha,
                  const double* a, int lda, const double* b, int ldb, double beta, double* c) {
        if (height != 0) {
            check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, height, width, depth, &alpha,
                              a, lda, b, ldb, &beta, c, ld));
        }
    }

    void potrf(int tile, int width) {
        check(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, tilePointer(tile), ld,
                               workspace.data, workspaceSize, info.data + tile));
    }

    void factorPanel(int tile) {
        if (grid.col != tile % grid.cols) {
            return;
        }
        const int owner = tile % grid.rows;
        const int width = rows.width(tile);
        const int start = rows.offset(tile + 1);
        const int height = ld - start;
        const double* d = diagonal.data;
        int diagonalLd = width;
        if (grid.row == owner) {
            potrf(tile, width);
            d = tilePointer(tile);
            diagonalLd = ld;
        }

        if (grid.rows > 1 && tile + 1 < rows.tiles()) {
            if (grid.row == owner) {
                downloadTile(hostDiagonal.data, d, width, width);
            } else {
                check(cudaEventSynchronize(diagonalUploaded));
            }
            MPI_Bcast(hostDiagonal.data, width * width, MPI_DOUBLE, owner, grid.colComm);
            if (height != 0 && grid.row != owner) {
                uploadPanel(diagonal.data, hostDiagonal.data, width * width, diagonalUploaded);
            }
        }
        if (height != 0) {
            const double one = 1.0;
            double* panel = matrix.data + static_cast<size_t>(cols.offset(tile)) * ld + start;
            check(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                              CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, height, width,
                              &one, d, diagonalLd, panel, ld));
        }
    }

    // Redistribute the row panel into a column panel. Within a process column,
    // each source row broadcasts only the tiles used by that process column.
    void exchangeColumns(int first, int width, int rowStart, int height) {
        check(cudaEventSynchronize(colsUploaded));
        std::fill(groupOffsets.begin(), groupOffsets.end(), 0);
        const int firstLocalTile = cols.firstTile(first);
        for (int tile = firstLocalTile; tile < cols.tiles(); tile += grid.cols) {
            groupOffsets[tile % grid.rows + 1] += cols.width(tile) * static_cast<size_t>(width);
        }
        for (int row = 0; row < grid.rows; ++row) {
            groupOffsets[row + 1] += groupOffsets[row];
        }
        // Assign each tile its packed location, leaving each group cursor at its end.
        for (int tile = firstLocalTile; tile < cols.tiles(); tile += grid.cols) {
            const int row = tile % grid.rows;
            colOffsets[tile] = groupOffsets[row];
            groupOffsets[row] += cols.width(tile) * static_cast<size_t>(width);
        }
        #pragma omp parallel for schedule(static) num_threads(threadsFor(groupOffsets[grid.row] - (grid.row ? groupOffsets[grid.row - 1] : 0)))
        for (int tile = firstLocalTile; tile < cols.tiles(); tile += grid.cols) {
            if (tile % grid.rows == grid.row) {
                const int tileWidth = cols.width(tile);
                const int source = rows.offset(tile) - rowStart;
                double* out = hostCols.data + colOffsets[tile];
                for (int j = 0; j < width; ++j) {
                    std::copy_n(hostRows.data + static_cast<size_t>(j) * height + source,
                                tileWidth, out + static_cast<size_t>(j) * tileWidth);
                }
            }
        }
        // A group is typically a single message. Chunk counts for MPI interfaces using int.
        constexpr size_t chunk = 1 << 27;
        for (size_t offset = 0; offset < groupOffsets.back(); offset += chunk) {
            int pending = 0;
            for (int row = 0; row < grid.rows; ++row) {
                const size_t begin = std::max(offset, row ? groupOffsets[row - 1] : 0);
                const size_t end = std::min(offset + chunk, groupOffsets[row]);
                if (begin < end) {
                    MPI_Ibcast(hostCols.data + begin, static_cast<int>(end - begin), MPI_DOUBLE,
                               row, grid.colComm, &requests[pending++]);
                }
            }
            MPI_Waitall(pending, requests.data(), MPI_STATUSES_IGNORE);
        }
        if (height != 0 && groupOffsets.back()) {
            uploadPanel(colPanel.data, hostCols.data, groupOffsets.back(), colsUploaded);
        }
    }

    void applyPanel(int tile, int first, double alpha, double* target, bool lookahead) {
        const int owner = tile % grid.cols;
        const int width = rows.width(tile);
        const int rowStart = rows.offset(first);
        const int height = ld - rowStart;
        const int firstLocalTile = cols.firstTile(first);
        const double* a = rowPanel.data;
        int panelLd = height;
        if (height != 0) {
            if (grid.col == owner) {
                a = matrix.data + static_cast<size_t>(cols.offset(tile)) * ld + rowStart;
                panelLd = ld;
                if (grid.size > 1) {
                    downloadTile(hostRows.data, a, height, width);
                }
            }
            if (grid.cols > 1) {
                if (grid.col != owner) {
                    check(cudaEventSynchronize(rowsUploaded));
                }
                const size_t count = static_cast<size_t>(height) * width;
                constexpr size_t chunk = 1 << 27;
                for (size_t offset = 0; offset < count; offset += chunk) {
                    MPI_Bcast(hostRows.data + offset, std::min(chunk, count - offset),
                              MPI_DOUBLE, owner, grid.rowComm);
                }
            }
        }
        if (lookahead) {
            // The panel solve and row broadcast can overlap previous trailing GEMMs.
            check(cudaStreamWaitEvent(stream, updatesDone));
        }
        if (height != 0 && grid.col != owner && firstLocalTile < cols.tiles()) {
            uploadPanel(rowPanel.data, hostRows.data, static_cast<size_t>(height) * width, rowsUploaded);
        }
        if (grid.rows > 1) {
            exchangeColumns(first, width, rowStart, height);
        }
        if (height == 0) {
            return;
        }
        if (lookahead) {
            check(cudaEventRecord(panelReady, stream));
            check(cudaStreamWaitEvent(updateStream, panelReady));
        }
        for (int j = firstLocalTile; j < cols.tiles(); j += grid.cols) {
            const int start = rows.offset(j);
            if (start == ld) {
                break;
            }
            const int tileWidth = cols.width(j);
            const double* b = grid.rows > 1 ? colPanel.data + colOffsets[j] : a + start - rowStart;
            const int ldb = grid.rows > 1 ? tileWidth : panelLd;
            // The next column is needed for POTRF and TRSM on the high-priority stream.
            const auto handle = lookahead && j != first ? updateBlas : blas;
            multiply(handle, ld - start, tileWidth, width, alpha, a + start - rowStart, panelLd, b, ldb,
                     1.0, target + static_cast<size_t>(cols.offset(j)) * ld + start);
        }
        if (lookahead) {
            check(cudaEventRecord(updatesDone, updateStream));
        }
    }

    void adjustDiagonal(bool zeroUpper) {
        int first = grid.row;
        while (first < rows.tiles() && first % grid.cols != grid.col) {
            first += grid.rows;
        }
        if (first < rows.tiles()) {
            const int period = std::lcm(grid.rows, grid.cols);
            const int count = 1 + (rows.tiles() - 1 - first) / period;
            prepareDiagonal<<<count, rows.block, 0, stream>>>(
                matrix.data, ld, rows.n, rows.block, first, period, grid.rows, grid.cols, zeroUpper);
            check(cudaGetLastError());
        }
    }

    const Grid& grid;
    Axis rows, cols;
    int ld;
    size_t elements;
    Buffer<double> matrix, original;
    Buffer<int> info;
    Buffer<double> diagonal;
    Buffer<double, true> hostDiagonal;
    Buffer<double> rowPanel;
    Buffer<double, true> hostRows;
    Buffer<double> colPanel;
    Buffer<double, true> hostCols;
    std::vector<size_t> colOffsets, groupOffsets;
    std::vector<MPI_Request> requests;
    Buffer<double> workspace{0};
    int workspaceSize = 0;
    cudaStream_t stream;
    cudaStream_t updateStream = nullptr;
    cudaEvent_t diagonalUploaded, rowsUploaded, colsUploaded, panelReady, updatesDone;
    cublasHandle_t blas;
    cublasHandle_t updateBlas;
    cusolverDnHandle_t solver;
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            errno = 0;
            const long parsed = strtol(value, nullptr, 10);
            if (errno || parsed < 0 || parsed > INT_MAX) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid matrix size: %s\n", value);
                }
                return 1;
            }
            n = static_cast<int>(parsed);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    const Grid grid(n);
    Cholesky benchmark(grid, n, validate);
    benchmark.generate();
    benchmark.prepareFactorization();
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int failure = benchmark.factor();
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (failure != INT_MAX) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failure);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }
    if (rank == 0) {
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
    }

    if (printResults) {
        benchmark.printResult();
    }
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const auto errors = benchmark.validate();
        const bool valid = errors[1] <= 1e-6;
        if (rank == 0) {
            printf("Max absolute error: %.10e\n", errors[0]);
            printf("Max relative error: %.10e\n", errors[1]);
            if (!valid) {
                printf("Validation failed: relative error too large\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fail("MPI", "MPI_THREAD_FUNNELED is required");
    }
    int status;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        fail("Cholesky", error.what());
    }
    MPI_Finalize();
    return status;
}
