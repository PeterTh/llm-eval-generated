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

[[noreturn]] void fatal(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

void checkMPI(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        std::fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, error);
        MPI_Abort(MPI_COMM_WORLD, status);
        std::exit(1);
    }
}

void checkCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::exit(1);
    }
}

void checkBLAS(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: %s: cuBLAS error %d\n", worldRank, operation,
                     static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::exit(1);
    }
}

void checkSolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: %s: cuSOLVER error %d\n", worldRank, operation,
                     static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::exit(1);
    }
}

#define MPI_CHECK(call) checkMPI((call), #call)
#define CUDA_CHECK(call) checkCUDA((call), #call)
#define BLAS_CHECK(call) checkBLAS((call), #call)
#define SOLVER_CHECK(call) checkSolver((call), #call)

template<class T, bool Pinned = false>
class Buffer {
public:
    T* data = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            fatal("Allocation size overflow");
        count = std::max(size_t{1}, count);
        if (Pinned)
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(T)));
        else
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Buffer() {
        if (data) {
            if (Pinned) cudaFreeHost(data);
            else cudaFree(data);
        }
    }
};

int localTiles(int tiles, int coordinate, int processes) {
    return tiles > coordinate ? (tiles - 1 - coordinate) / processes + 1 : 0;
}

int firstAfter(int tile, int coordinate, int processes) {
    return tile < coordinate ? 0 : (tile - coordinate) / processes + 1;
}

// Large output matrices can exceed MPI's int count even though panels do not.
void broadcastDoubles(double* data, size_t count, int root, MPI_Comm comm) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, size_t{INT_MAX}));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm));
        data += chunk;
        count -= chunk;
    }
}

__device__ size_t globalIndex(int local, int block, int coordinate, int processes) {
    return (static_cast<size_t>(local / block) * processes + coordinate) * block
           + local % block;
}

__global__ void finishMatrix(double* matrix, int ld, int rows, int cols,
                             int block, int prow, int pcol, int nprows,
                             int npcols, size_t n, bool zeroUpper) {
    const size_t count = static_cast<size_t>(rows) * cols;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index % rows);
        const int col = static_cast<int>(index / rows);
        const size_t i = globalIndex(row, block, prow, nprows);
        const size_t j = globalIndex(col, block, pcol, npcols);
        double& value = matrix[row + static_cast<size_t>(col) * ld];
        if (zeroUpper) {
            if (i < j) value = 0.0;
        } else if (i == j) {
            // Identity padding permits fixed-size BLAS tiles on edge blocks.
            value += i < n ? static_cast<double>(n) : 1.0;
        }
    }
}

__global__ void reconstructionErrors(const double* reconstructed,
                                     const double* original, double* errors,
                                     int ld, int rows, int cols, int block,
                                     int prow, int pcol, int nprows,
                                     int npcols, size_t n) {
    __shared__ double absolute[256];
    __shared__ double relative[256];
    double maxAbsolute = 0.0, maxRelative = 0.0;
    const size_t count = static_cast<size_t>(rows) * cols;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index % rows);
        const int col = static_cast<int>(index / rows);
        if (globalIndex(row, block, prow, nprows) >= n ||
            globalIndex(col, block, pcol, npcols) >= n) continue;
        const size_t offset = row + static_cast<size_t>(col) * ld;
        const double a = original[offset], r = reconstructed[offset];
        const bool finite = isfinite(a) && isfinite(r);
        const double error = finite ? fabs(r - a) : INFINITY;
        maxAbsolute = fmax(maxAbsolute, error);
        maxRelative = fmax(maxRelative, finite ? error / (fabs(a) + 1e-10) : INFINITY);
    }
    absolute[threadIdx.x] = maxAbsolute;
    relative[threadIdx.x] = maxRelative;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
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

class HybridCholesky {
    size_t n;
    int ranks, nprows, npcols, prow, pcol, block, tiles;
    int rowTiles, colTiles, rows, cols, ld;
    size_t matrixCount, batchCapacity;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;
    cudaStream_t panelStream = nullptr, updateStream = nullptr;
    cudaEvent_t uploaded[2]{}, bulkDone[2]{};
    cublasHandle_t panelBLAS = nullptr, updateBLAS = nullptr;
    cusolverDnHandle_t solver = nullptr;
    int workspaceSize = 0;
    Buffer<double> matrix, original, diagonal, workspace;
    Buffer<int> infoDevice;
    Buffer<int, true> infoHost;
    Buffer<double, true> diagonalHost, rowHost[2], colHost[2], sendHost, gatherHost;
    Buffer<double> rowDevice[2], colDevice[2];
    Buffer<double*, true> pointerHost[2];
    Buffer<double*> pointerDevice[2];
#ifndef __GLIBC__
    std::vector<unsigned int> rowSeeds;
#endif

    double* tileAddress(int localRow, int localCol) {
        return matrix.data + localRow * block + static_cast<size_t>(localCol) * block * ld;
    }

    void launchFinish(bool zeroUpper) {
        const size_t count = static_cast<size_t>(rows) * cols;
        if (!count) return;
        const int grid = static_cast<int>(std::min(size_t{65535}, (count + 255) / 256));
        finishMatrix<<<grid, 256, 0, panelStream>>>(
            matrix.data, ld, rows, cols, block, prow, pcol, nprows, npcols, n, zeroUpper);
        CUDA_CHECK(cudaGetLastError());
    }

#ifdef __GLIBC__
    // glibc rand_r advances this 32-bit LCG three times per returned number.
    // Skip ahead without changing the original seed-42 random sequence.
    static unsigned int randomSeed(uint64_t calls) {
        uint32_t multiplier = 1103515245u, increment = 12345u;
        uint32_t accumulatedMultiplier = 1u, accumulatedIncrement = 0u;
        for (uint64_t steps = 3 * calls; steps; steps >>= 1) {
            if (steps & 1) {
                accumulatedIncrement = accumulatedIncrement * multiplier + increment;
                accumulatedMultiplier *= multiplier;
            }
            increment *= multiplier + 1u;
            multiplier *= multiplier;
        }
        return accumulatedMultiplier * 42u + accumulatedIncrement;
    }
#endif

    void randomPanel(double* panel, int coordinate, int processes,
                     int localRows, int start, int width) {
        #pragma omp parallel for schedule(static) if(localRows >= 256)
        for (int local = 0; local < localRows; ++local) {
            const size_t global = (static_cast<size_t>(local / block) * processes
                                  + coordinate) * block + local % block;
            if (global >= n) {
                for (int k = 0; k < width; ++k)
                    panel[local + static_cast<size_t>(k) * localRows] = 0.0;
                continue;
            }
#ifdef __GLIBC__
            unsigned int seed = randomSeed(static_cast<uint64_t>(global) * n + start);
#else
            unsigned int seed = rowSeeds[global];
            for (int k = 0; k < start; ++k) (void)rand_r(&seed);
#endif
            for (int k = 0; k < width; ++k)
                panel[local + static_cast<size_t>(k) * localRows] =
                    rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
        }
    }

    // Broadcast L_ik along process rows, then gather just the L_jk tiles
    // needed by each process column. Staging is portable to non-CUDA-aware MPI.
    void redistributePanel(int k, int parity, bool trailing, int& rowFirst, int& colFirst) {
        rowFirst = trailing ? firstAfter(k, prow, nprows) : 0;
        colFirst = trailing ? firstAfter(k, pcol, npcols) : 0;
        const int m = (rowTiles - rowFirst) * block;
        const int nc = (colTiles - colFirst) * block;
        if (pcol == k % npcols && m)
            CUDA_CHECK(cudaMemcpy2DAsync(rowHost[parity].data, m * sizeof(double),
                tileAddress(rowFirst, k / npcols), ld * sizeof(double),
                m * sizeof(double), block, cudaMemcpyDeviceToHost, panelStream));
        // Also ensures prior asynchronous host-to-device copies have finished.
        CUDA_CHECK(cudaStreamSynchronize(panelStream));
        broadcastDoubles(rowHost[parity].data, static_cast<size_t>(m) * block,
                         k % npcols, rowComm);

        std::vector<int> counts(nprows, 0), displacements(nprows, 0);
        std::vector<int> sources(colTiles - colFirst), selected;
        for (int j = colFirst; j < colTiles; ++j) {
            const int owner = (j * npcols + pcol) % nprows;
            sources[j - colFirst] = counts[owner];
            counts[owner] += block * block;
            if (owner == prow) selected.push_back(j);
        }
        for (int r = 1; r < nprows; ++r)
            displacements[r] = displacements[r - 1] + counts[r - 1];
        #pragma omp parallel for schedule(static) if(selected.size() >= 8)
        for (size_t t = 0; t < selected.size(); ++t) {
            const int global = selected[t] * npcols + pcol;
            const int offset = (global / nprows - rowFirst) * block;
            for (int c = 0; c < block; ++c)
                std::copy_n(rowHost[parity].data + offset + static_cast<size_t>(c) * m,
                            block, sendHost.data + (t * block + c) * block);
        }
        MPI_CHECK(MPI_Allgatherv(sendHost.data, counts[prow], MPI_DOUBLE, gatherHost.data,
                                counts.data(), displacements.data(), MPI_DOUBLE, colComm));
        #pragma omp parallel for schedule(static) if(colTiles - colFirst >= 8)
        for (int j = colFirst; j < colTiles; ++j) {
            const int owner = (j * npcols + pcol) % nprows;
            const size_t source = displacements[owner] + sources[j - colFirst];
            for (int c = 0; c < block; ++c)
                std::copy_n(gatherHost.data + source + c * block, block,
                    colHost[parity].data + (j - colFirst) * block
                    + static_cast<size_t>(c) * nc);
        }
    }

    void updateTrailing(int k, int parity, int rowFirst, int colFirst) {
        const int m = (rowTiles - rowFirst) * block;
        const int nc = (colTiles - colFirst) * block;
        size_t count = 0, lookahead = 0;
        double** pointers = pointerHost[parity].data;
        // Place the next panel's tiles first for a separate high-priority stream.
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = colFirst; j < colTiles; ++j) {
                const int globalCol = j * npcols + pcol;
                if ((globalCol == k + 1) != (pass == 0)) continue;
                for (int i = rowFirst; i < rowTiles; ++i) {
                    if (i * nprows + prow < globalCol) continue;
                    pointers[count] = rowDevice[parity].data + (i - rowFirst) * block;
                    pointers[batchCapacity + count] =
                        colDevice[parity].data + (j - colFirst) * block;
                    pointers[2 * batchCapacity + count] = tileAddress(i, j);
                    ++count;
                }
            }
            if (pass == 0) lookahead = count;
        }
        if (m && nc) {
            CUDA_CHECK(cudaMemcpyAsync(rowDevice[parity].data, rowHost[parity].data,
                static_cast<size_t>(m) * block * sizeof(double),
                cudaMemcpyHostToDevice, panelStream));
            CUDA_CHECK(cudaMemcpyAsync(colDevice[parity].data, colHost[parity].data,
                static_cast<size_t>(nc) * block * sizeof(double),
                cudaMemcpyHostToDevice, panelStream));
        }
        for (int p = 0; p < 3; ++p)
            if (count)
                CUDA_CHECK(cudaMemcpyAsync(pointerDevice[parity].data + p * batchCapacity,
                    pointers + p * batchCapacity, count * sizeof(double*),
                    cudaMemcpyHostToDevice, panelStream));
        CUDA_CHECK(cudaEventRecord(uploaded[parity], panelStream));
        // Earlier bulk updates can touch the next panel; wait only on that work.
        if (k) CUDA_CHECK(cudaStreamWaitEvent(panelStream, bulkDone[(k - 1) & 1], 0));
        const double minusOne = -1.0, one = 1.0;
        double** devicePointers = pointerDevice[parity].data;
        if (lookahead)
            BLAS_CHECK(cublasDgemmBatched(panelBLAS, CUBLAS_OP_N, CUBLAS_OP_T,
                block, block, block, &minusOne, devicePointers, m,
                devicePointers + batchCapacity, nc, &one,
                devicePointers + 2 * batchCapacity, ld, static_cast<int>(lookahead)));
        CUDA_CHECK(cudaStreamWaitEvent(updateStream, uploaded[parity], 0));
        if (count > lookahead)
            BLAS_CHECK(cublasDgemmBatched(updateBLAS, CUBLAS_OP_N, CUBLAS_OP_T,
                block, block, block, &minusOne, devicePointers + lookahead, m,
                devicePointers + batchCapacity + lookahead, nc, &one,
                devicePointers + 2 * batchCapacity + lookahead, ld,
                static_cast<int>(count - lookahead)));
        CUDA_CHECK(cudaEventRecord(bulkDone[parity], updateStream));
    }

public:
    HybridCholesky(size_t dimension, bool validate) : n(dimension) {
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
        int dims[2] = {0, 0};
        MPI_CHECK(MPI_Dims_create(ranks, 2, dims));
        nprows = dims[0]; npcols = dims[1];
        prow = worldRank / npcols; pcol = worldRank % npcols;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, prow, pcol, &rowComm));
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, pcol, prow, &colComm));
        MPI_CHECK(MPI_Comm_set_errhandler(rowComm, MPI_ERRORS_RETURN));
        MPI_CHECK(MPI_Comm_set_errhandler(colComm, MPI_ERRORS_RETURN));

        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                     worldRank, MPI_INFO_NULL, &nodeComm));
        int localRank = 0, localRanks = 0, devices = 0;
        MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
        MPI_CHECK(MPI_Comm_size(nodeComm, &localRanks));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) fatal("The MPI/OpenMP/CUDA benchmark requires a CUDA device on every rank");
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localRanks)));

        block = n < 2048 ? 128 : 256;
        if (n < static_cast<size_t>(block))
            block = std::max(32, static_cast<int>((n + 31) / 32) * 32);
        tiles = static_cast<int>((n + block - 1) / block);
        rowTiles = localTiles(tiles, prow, nprows);
        colTiles = localTiles(tiles, pcol, npcols);
        rows = rowTiles * block; cols = colTiles * block;
        ld = std::max(block, rows);
        matrixCount = static_cast<size_t>(ld) * std::max(block, cols);
        if (static_cast<size_t>(std::max(rows, cols)) * block > INT_MAX)
            fatal("A communication panel exceeds the MPI count limit");
        batchCapacity = 0;
        for (int j = 0; j < colTiles; ++j)
            for (int i = 0; i < rowTiles; ++i)
                if (i * nprows + prow >= j * npcols + pcol) ++batchCapacity;
        if (batchCapacity > INT_MAX) fatal("Too many local tiles for batched BLAS");
        batchCapacity = std::max(size_t{1}, batchCapacity);

        int lowPriority, highPriority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&panelStream, cudaStreamNonBlocking, highPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&updateStream, cudaStreamNonBlocking, lowPriority));
        BLAS_CHECK(cublasCreate(&panelBLAS));
        BLAS_CHECK(cublasCreate(&updateBLAS));
        BLAS_CHECK(cublasSetStream(panelBLAS, panelStream));
        BLAS_CHECK(cublasSetStream(updateBLAS, updateStream));
        SOLVER_CHECK(cusolverDnCreate(&solver));
        SOLVER_CHECK(cusolverDnSetStream(solver, panelStream));
        matrix.allocate(matrixCount);
        if (validate) original.allocate(matrixCount);
        diagonal.allocate(static_cast<size_t>(block) * block);
        diagonalHost.allocate(static_cast<size_t>(block) * block);
        infoDevice.allocate(1); infoHost.allocate(1);
        SOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
            block, matrix.data, ld, &workspaceSize));
        workspace.allocate(workspaceSize);
        const size_t rowCount = static_cast<size_t>(rows) * block;
        const size_t colCount = static_cast<size_t>(cols) * block;
        sendHost.allocate(colCount); gatherHost.allocate(colCount);
        for (int p = 0; p < 2; ++p) {
            CUDA_CHECK(cudaEventCreateWithFlags(&uploaded[p], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&bulkDone[p], cudaEventDisableTiming));
            rowHost[p].allocate(rowCount); colHost[p].allocate(colCount);
            rowDevice[p].allocate(rowCount); colDevice[p].allocate(colCount);
            pointerHost[p].allocate(3 * batchCapacity);
            pointerDevice[p].allocate(3 * batchCapacity);
        }
        // Initialize library kernels before the timed factorization. CUDA context
        // setup and lazy solver/BLAS loading are benchmark setup costs.
        std::fill_n(diagonalHost.data, static_cast<size_t>(block) * block, 0.0);
        for (int i = 0; i < block; ++i) diagonalHost.data[i * block + i] = 1.0;
        CUDA_CHECK(cudaMemcpyAsync(diagonal.data, diagonalHost.data,
            static_cast<size_t>(block) * block * sizeof(double),
            cudaMemcpyHostToDevice, panelStream));
        SOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, block,
            diagonal.data, block, workspace.data, workspaceSize, infoDevice.data));
        const double one = 1.0, zero = 0.0;
        CUDA_CHECK(cudaMemsetAsync(matrix.data, 0,
            static_cast<size_t>(block) * block * sizeof(double), panelStream));
        BLAS_CHECK(cublasDtrsm(panelBLAS, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, block, block, &one,
            diagonal.data, block, matrix.data, block));
        pointerHost[0].data[0] = diagonal.data;
        pointerHost[0].data[batchCapacity] = diagonal.data;
        pointerHost[0].data[2 * batchCapacity] = matrix.data;
        for (int p = 0; p < 3; ++p)
            CUDA_CHECK(cudaMemcpyAsync(pointerDevice[0].data + p * batchCapacity,
                pointerHost[0].data + p * batchCapacity, sizeof(double*),
                cudaMemcpyHostToDevice, panelStream));
        BLAS_CHECK(cublasDgemmBatched(panelBLAS, CUBLAS_OP_N, CUBLAS_OP_T,
            block, block, block, &one, pointerDevice[0].data, block,
            pointerDevice[0].data + batchCapacity, block, &zero,
            pointerDevice[0].data + 2 * batchCapacity, block, 1));
        CUDA_CHECK(cudaStreamSynchronize(panelStream));
        if (worldRank == 0)
            std::printf("Parallel configuration: %d MPI ranks, %d x %d process grid, "
                        "%d OpenMP threads/rank, %d x %d CUDA tiles\n",
                        ranks, nprows, npcols, omp_get_max_threads(), block, block);
    }

    ~HybridCholesky() {
        if (panelStream) cudaStreamSynchronize(panelStream);
        if (updateStream) cudaStreamSynchronize(updateStream);
        if (solver) cusolverDnDestroy(solver);
        if (panelBLAS) cublasDestroy(panelBLAS);
        if (updateBLAS) cublasDestroy(updateBLAS);
        for (int p = 0; p < 2; ++p) {
            if (uploaded[p]) cudaEventDestroy(uploaded[p]);
            if (bulkDone[p]) cudaEventDestroy(bulkDone[p]);
        }
        if (panelStream) cudaStreamDestroy(panelStream);
        if (updateStream) cudaStreamDestroy(updateStream);
        if (rowComm != MPI_COMM_NULL) MPI_Comm_free(&rowComm);
        if (colComm != MPI_COMM_NULL) MPI_Comm_free(&colComm);
    }

    void generatePositiveDefiniteMatrix() {
#ifndef __GLIBC__
        // Portable fallback for a libc with a different rand_r recurrence.
        // This preserves the original stream while keeping B distributed.
        rowSeeds.resize(n);
        if (worldRank == 0) {
            unsigned int seed = 42;
            for (size_t i = 0; i < n; ++i) {
                rowSeeds[i] = seed;
                for (size_t j = 0; j < n; ++j) (void)rand_r(&seed);
            }
        }
        if (n) MPI_CHECK(MPI_Bcast(rowSeeds.data(), static_cast<int>(n),
                                 MPI_UNSIGNED, 0, MPI_COMM_WORLD));
#endif
        CUDA_CHECK(cudaMemsetAsync(matrix.data, 0, matrixCount * sizeof(double), panelStream));
        const double one = 1.0;
        for (int k = 0; k < tiles; ++k) {
            const int parity = k & 1;
            if (k >= 2) CUDA_CHECK(cudaEventSynchronize(bulkDone[parity]));
            const int start = k * block;
            const int width = static_cast<int>(std::min(static_cast<size_t>(block), n - start));
            randomPanel(rowHost[parity].data, prow, nprows, rows, start, width);
            if (nprows == npcols && prow == pcol)
                std::copy_n(rowHost[parity].data, static_cast<size_t>(rows) * width,
                            colHost[parity].data);
            else
                randomPanel(colHost[parity].data, pcol, npcols, cols, start, width);
            if (rows && cols) {
                CUDA_CHECK(cudaMemcpyAsync(rowDevice[parity].data, rowHost[parity].data,
                    static_cast<size_t>(rows) * width * sizeof(double),
                    cudaMemcpyHostToDevice, panelStream));
                CUDA_CHECK(cudaMemcpyAsync(colDevice[parity].data, colHost[parity].data,
                    static_cast<size_t>(cols) * width * sizeof(double),
                    cudaMemcpyHostToDevice, panelStream));
                BLAS_CHECK(cublasDgemm(panelBLAS, CUBLAS_OP_N, CUBLAS_OP_T, rows, cols, width,
                    &one, rowDevice[parity].data, rows, colDevice[parity].data, cols,
                    &one, matrix.data, ld));
            }
            CUDA_CHECK(cudaEventRecord(bulkDone[parity], panelStream));
        }
        launchFinish(false);
        if (original.data)
            CUDA_CHECK(cudaMemcpyAsync(original.data, matrix.data, matrixCount * sizeof(double),
                                      cudaMemcpyDeviceToDevice, panelStream));
        CUDA_CHECK(cudaStreamSynchronize(panelStream));
    }

    bool choleskyDecomposition() {
        for (int k = 0; k < tiles; ++k) {
            const int parity = k & 1;
            // A panel buffer is reused only after the bulk update consuming it.
            if (k >= 2) CUDA_CHECK(cudaEventSynchronize(bulkDone[parity]));
            const int owner = (k % nprows) * npcols + k % npcols;
            int info = 0;
            if (worldRank == owner) {
                double* tile = tileAddress(k / nprows, k / npcols);
                SOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, block, tile, ld,
                                             workspace.data, workspaceSize, infoDevice.data));
                CUDA_CHECK(cudaMemcpyAsync(infoHost.data, infoDevice.data, sizeof(int),
                                          cudaMemcpyDeviceToHost, panelStream));
                CUDA_CHECK(cudaMemcpy2DAsync(diagonalHost.data, block * sizeof(double),
                    tile, ld * sizeof(double), block * sizeof(double), block,
                    cudaMemcpyDeviceToHost, panelStream));
                CUDA_CHECK(cudaStreamSynchronize(panelStream));
                info = *infoHost.data;
            }
            MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (info) {
                if (worldRank == 0) {
                    if (info > 0)
                        std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                                    static_cast<size_t>(k) * block + info - 1);
                    else
                        std::printf("Error: Cholesky solver rejected argument %d\n", -info);
                }
                return false;
            }
            const int rowFirst = firstAfter(k, prow, nprows);
            const int m = (rowTiles - rowFirst) * block;
            if (pcol == k % npcols)
                broadcastDoubles(diagonalHost.data, static_cast<size_t>(block) * block,
                                 k % nprows, colComm);
            if (pcol == k % npcols && m) {
                CUDA_CHECK(cudaMemcpyAsync(diagonal.data, diagonalHost.data,
                    static_cast<size_t>(block) * block * sizeof(double),
                    cudaMemcpyHostToDevice, panelStream));
                const double one = 1.0;
                BLAS_CHECK(cublasDtrsm(panelBLAS, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, block, &one,
                    diagonal.data, block, tileAddress(rowFirst, k / npcols), ld));
            }
            int firstRow, firstCol;
            redistributePanel(k, parity, true, firstRow, firstCol);
            updateTrailing(k, parity, firstRow, firstCol);
        }
        CUDA_CHECK(cudaStreamSynchronize(updateStream));
        CUDA_CHECK(cudaStreamSynchronize(panelStream));
        launchFinish(true);
        CUDA_CHECK(cudaStreamSynchronize(panelStream));
        return true;
    }

    bool validateCholesky() {
        Buffer<double> reconstructed;
        reconstructed.allocate(matrixCount);
        CUDA_CHECK(cudaMemsetAsync(reconstructed.data, 0, matrixCount * sizeof(double), panelStream));
        const double one = 1.0;
        for (int k = 0; k < tiles; ++k) {
            int firstRow, firstCol;
            redistributePanel(k, 0, false, firstRow, firstCol);
            if (rows && cols) {
                CUDA_CHECK(cudaMemcpyAsync(rowDevice[0].data, rowHost[0].data,
                    static_cast<size_t>(rows) * block * sizeof(double),
                    cudaMemcpyHostToDevice, panelStream));
                CUDA_CHECK(cudaMemcpyAsync(colDevice[0].data, colHost[0].data,
                    static_cast<size_t>(cols) * block * sizeof(double),
                    cudaMemcpyHostToDevice, panelStream));
                BLAS_CHECK(cublasDgemm(panelBLAS, CUBLAS_OP_N, CUBLAS_OP_T,
                    rows, cols, block, &one, rowDevice[0].data, rows,
                    colDevice[0].data, cols, &one, reconstructed.data, ld));
            }
        }
        const size_t count = static_cast<size_t>(rows) * cols;
        const int grid = static_cast<int>(std::min(size_t{65535}, (count + 255) / 256));
        Buffer<double> errors;
        Buffer<double, true> errorsHost;
        errors.allocate(static_cast<size_t>(grid) * 2);
        errorsHost.allocate(static_cast<size_t>(grid) * 2);
        if (grid) {
            reconstructionErrors<<<grid, 256, 0, panelStream>>>(reconstructed.data, original.data,
                errors.data, ld, rows, cols, block, prow, pcol, nprows, npcols, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(errorsHost.data, errors.data, grid * 2 * sizeof(double),
                                      cudaMemcpyDeviceToHost, panelStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(panelStream));
        double maxError = 0.0, relError = 0.0;
        #pragma omp parallel for reduction(max:maxError,relError) if(grid >= 256)
        for (int i = 0; i < grid; ++i) {
            maxError = std::max(maxError, errorsHost.data[2 * i]);
            relError = std::max(relError, errorsHost.data[2 * i + 1]);
        }
        double local[2] = {maxError, relError}, global[2];
        MPI_CHECK(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (worldRank == 0) {
            std::printf("Max absolute error: %.10e\n", global[0]);
            std::printf("Max relative error: %.10e\n", global[1]);
            if (!std::isfinite(global[1]) || global[1] > 1e-6)
                std::printf("Validation failed: relative error too large\n");
        }
        return std::isfinite(global[1]) && global[1] <= 1e-6;
    }

    void printResults() {
        std::vector<double> local(matrixCount);
        CUDA_CHECK(cudaMemcpy(local.data(), matrix.data, matrixCount * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::vector<double> result;
        if (worldRank == 0) result.resize(n * n);
        for (int rank = 0; rank < ranks; ++rank) {
            const int r = rank / npcols, c = rank % npcols;
            const int rankRows = localTiles(tiles, r, nprows) * block;
            const int rankCols = localTiles(tiles, c, npcols) * block;
            const int rankLD = std::max(block, rankRows);
            const size_t rankCount = static_cast<size_t>(rankLD) * std::max(block, rankCols);
            std::vector<double> received;
            double* source = nullptr;
            if (worldRank == 0) {
                if (rank == 0) source = local.data();
                else { received.resize(rankCount); source = received.data(); }
            }
            if (rank) {
                for (size_t offset = 0; offset < rankCount;) {
                    const int chunk = static_cast<int>(std::min(rankCount - offset, size_t{INT_MAX}));
                    if (worldRank == rank)
                        MPI_CHECK(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE,
                                           0, 0, MPI_COMM_WORLD));
                    if (worldRank == 0)
                        MPI_CHECK(MPI_Recv(source + offset, chunk, MPI_DOUBLE, rank, 0,
                                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    offset += chunk;
                }
            }
            if (worldRank == 0) {
                #pragma omp parallel for schedule(static) if(rankRows >= 256)
                for (int i = 0; i < rankRows; ++i) {
                    const size_t globalRow = (static_cast<size_t>(i / block) * nprows + r)
                                             * block + i % block;
                    if (globalRow >= n) continue;
                    for (int j = 0; j < rankCols; ++j) {
                        const size_t globalCol = (static_cast<size_t>(j / block) * npcols + c)
                                                 * block + j % block;
                        if (globalCol < n)
                            result[globalRow * n + globalCol] = source[i + static_cast<size_t>(j) * rankLD];
                    }
                }
            }
        }
        if (worldRank == 0) print_results(result, "CholeskyL");
    }
};

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end || errno ||
                parsed > static_cast<unsigned long long>(INT_MAX - 256) ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                if (worldRank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (worldRank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    HybridCholesky computation(n, validate);
    if (worldRank == 0) std::printf("Generating positive definite matrix...\n");
    computation.generatePositiveDefiniteMatrix();
    if (worldRank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success = computation.choleskyDecomposition();
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (worldRank == 0) std::printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (worldRank == 0) {
        const double operations = static_cast<double>(n) * n * n / 3.0;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", seconds > 0 ? operations / seconds / 1e9 : 0.0);
    }
    if (printResults) computation.printResults();
    if (validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        const bool valid = computation.validateCholesky();
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fatal("MPI_THREAD_FUNNELED support is required");
    int result = 1;
    try {
        result = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        fatal(error.what());
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
