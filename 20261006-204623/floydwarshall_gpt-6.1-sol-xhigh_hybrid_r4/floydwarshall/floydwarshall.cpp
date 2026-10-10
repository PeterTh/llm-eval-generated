#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 64;
constexpr int THREAD_ROWS = 8;
constexpr int THREAD_COLS = 32;
constexpr unsigned int UNCHANGED = UINT_MAX;

void mpiCheck(int error) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    fprintf(stderr, "MPI error: %.*s\n", length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void cudaCheck(cudaError_t error) {
    if (error == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

struct MpiEnvironment {
    MpiEnvironment(int& argc, char**& argv) {
        int provided = 0;
        mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    ~MpiEnvironment() { MPI_Finalize(); }
};

template<bool Pinned = false> struct CudaBuffer {
    unsigned int* data = nullptr;
    explicit CudaBuffer(size_t elements) {
        if constexpr (Pinned)
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), elements * sizeof(unsigned int)));
        else
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), elements * sizeof(unsigned int)));
    }
    ~CudaBuffer() {
        if constexpr (Pinned) cudaFreeHost(data);
        else cudaFree(data);
    }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
};

struct CudaStream {
    cudaStream_t value;
    CudaStream() { cudaCheck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(value); }
};

struct CudaEvent {
    cudaEvent_t value;
    CudaEvent() { cudaCheck(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~CudaEvent() { cudaEventDestroy(value); }
};

// glibc rand_r uses three LCG steps per output. Jump whole outputs so every
// distributed row reproduces the original, single seed-42 random sequence.
#if defined(__GLIBC__)
unsigned int rowSeed(size_t outputs) {
    uint32_t multiplier = 1, increment = 0;
    for (int step = 0; step < 3; ++step) {
        multiplier *= 1103515245u;
        increment = increment * 1103515245u + 12345u;
    }
    uint32_t seed = 42;
    while (outputs) {
        if (outputs & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1u;
        multiplier *= multiplier;
        outputs >>= 1;
    }
    return seed;
}
#endif

void initializeLocalDistances(std::vector<unsigned int>& dist, size_t n,
                              size_t pitch, int rowRank, int columnRank,
                              int rowRanks, int columnRanks) {
    const size_t rows = dist.size() / pitch;
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const size_t globalRow = (row / TILE * rowRanks + rowRank) * TILE + row % TILE;
        unsigned int* values = dist.data() + row * pitch;
        std::fill(values, values + pitch, INF);
#if defined(__GLIBC__)
        if (globalRow < n) {
            for (size_t block = 0; block < pitch / TILE; ++block) {
                const size_t globalColumn = (block * columnRanks + columnRank) * TILE;
                if (globalColumn >= n) continue;
                unsigned int seed = rowSeed(globalRow * n + globalColumn);
                const size_t columns = std::min(static_cast<size_t>(TILE), n - globalColumn);
                for (size_t col = 0; col < columns; ++col)
                    values[block * TILE + col] = 1u + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
            }
        }
#endif
        if (static_cast<int>(globalRow / TILE % columnRanks) == columnRank)
            values[globalRow / (TILE * columnRanks) * TILE + globalRow % TILE] = 0;
    }
#if !defined(__GLIBC__)
    // Other libcs can implement rand_r differently. Replay their actual RNG
    // without assuming an implementation-specific skip-ahead formula.
    unsigned int seed = 42;
    for (size_t row = 0; row < n; ++row) {
        const bool ownedRow = static_cast<int>(row / TILE % rowRanks) == rowRank;
        const size_t localRow = row / (TILE * rowRanks) * TILE + row % TILE;
        for (size_t col = 0; col < n; ++col) {
            const unsigned int value = 1u + static_cast<unsigned int>(
                static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
            if (ownedRow && static_cast<int>(col / TILE % columnRanks) == columnRank) {
                const size_t localColumn = col / (TILE * columnRanks) * TILE + col % TILE;
                dist[localRow * pitch + localColumn] = row == col ? 0 : value;
            }
        }
    }
#endif
}

__global__ void initializePaths(unsigned int* path, size_t elements,
                                size_t pitch, int rank, int ranks) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < elements; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = i / pitch;
        path[i] = static_cast<unsigned int>((row / TILE * ranks + rank) * TILE + row % TILE);
    }
}

// Record both operands and pivot-row snapshots at their original k iteration.
// Ordinary blocked Floyd-Warshall can change path tie breaking; these histories
// preserve the scalar algorithm's strict-improvement updates exactly.
__global__ void pivotHistory(const unsigned int* dist, size_t pitch,
                             size_t localBase, size_t globalBase,
                             unsigned int* operands, unsigned int* snapshots) {
    __shared__ unsigned int tile[TILE][TILE];
    #pragma unroll
    for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
        const int row = threadIdx.y + ry * THREAD_ROWS;
        #pragma unroll
        for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
            const int col = threadIdx.x + cx * THREAD_COLS;
            tile[row][col] = dist[(localBase + row) * pitch + globalBase + col];
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
            const int row = threadIdx.y + ry * THREAD_ROWS;
            #pragma unroll
            for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
                const int col = threadIdx.x + cx * THREAD_COLS;
                if (col == k) operands[row * TILE + k] = tile[row][col];
                if (row == k) snapshots[k * TILE + col] = tile[row][col];
                // The zero diagonal makes the pivot row/column immutable at k.
                // Do not write them while other threads read their operands.
                if (row != k && col != k)
                    tile[row][col] = min(tile[row][col], tile[row][k] + tile[k][col]);
            }
        }
        __syncthreads();
    }
}

__global__ void pivotRows(const unsigned int* dist, size_t pitch,
                          size_t localBase, int pivotBlock,
                          const unsigned int* operands,
                          const unsigned int* diagonal,
                          unsigned int* panel) {
    const int colBlock = blockIdx.x;
    const size_t colBase = static_cast<size_t>(colBlock) * TILE;
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int left[TILE][TILE];
    #pragma unroll
    for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
        const int row = threadIdx.y + ry * THREAD_ROWS;
        #pragma unroll
        for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
            const int col = threadIdx.x + cx * THREAD_COLS;
            if (colBlock == pivotBlock) {
                panel[row * pitch + colBase + col] = diagonal[row * TILE + col];
            } else {
                tile[row][col] = dist[(localBase + row) * pitch + colBase + col];
                left[row][col] = operands[row * TILE + col];
            }
        }
    }
    if (colBlock == pivotBlock) return;
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
            const int row = threadIdx.y + ry * THREAD_ROWS;
            #pragma unroll
            for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
                const int col = threadIdx.x + cx * THREAD_COLS;
                if (row == k) panel[k * pitch + colBase + col] = tile[row][col];
                else tile[row][col] = min(tile[row][col], left[row][k] + tile[k][col]);
            }
        }
        __syncthreads();
    }
}

__global__ void columnHistory(unsigned int* dist, unsigned int* path,
                              size_t pitch, size_t columnBase, size_t globalBase,
                              const unsigned int* diagonal, unsigned int* operands) {
    const size_t rowBase = static_cast<size_t>(blockIdx.x) * TILE;
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    unsigned int last[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    #pragma unroll
    for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
        const int row = threadIdx.y + ry * THREAD_ROWS;
        #pragma unroll
        for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
            const int col = threadIdx.x + cx * THREAD_COLS;
            tile[row][col] = dist[(rowBase + row) * pitch + columnBase + col];
            right[row][col] = diagonal[row * TILE + col];
            last[ry][cx] = UNCHANGED;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
            const int row = threadIdx.y + ry * THREAD_ROWS;
            #pragma unroll
            for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
                const int col = threadIdx.x + cx * THREAD_COLS;
                if (col == k) operands[(rowBase + row) * TILE + k] = tile[row][col];
                else {
                    const unsigned int candidate = tile[row][k] + right[k][col];
                    if (candidate < tile[row][col]) {
                        tile[row][col] = candidate;
                        last[ry][cx] = k;
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
        const int row = threadIdx.y + ry * THREAD_ROWS;
        #pragma unroll
        for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
            const int col = threadIdx.x + cx * THREAD_COLS;
            const size_t index = (rowBase + row) * pitch + columnBase + col;
            dist[index] = tile[row][col];
            if (last[ry][cx] != UNCHANGED) path[index] = globalBase + last[ry][cx];
        }
    }
}

__global__ void updateTiles(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path, size_t pitch,
                            int pivotBlock, int excludedColumn, size_t rowBlockOffset,
                            const unsigned int* __restrict__ operands,
                            const unsigned int* __restrict__ panel) {
    const int colBlock = blockIdx.x + (static_cast<int>(blockIdx.x) >= excludedColumn);
    const size_t colBase = static_cast<size_t>(colBlock) * TILE;
    const size_t rowBase = (rowBlockOffset + blockIdx.y) * TILE;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    unsigned int values[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    unsigned int last[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    #pragma unroll
    for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
        const int row = threadIdx.y + ry * THREAD_ROWS;
        #pragma unroll
        for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
            const int col = threadIdx.x + cx * THREAD_COLS;
            left[row][col] = operands[(rowBase + row) * TILE + col];
            right[row][col] = panel[row * pitch + colBase + col];
            values[ry][cx] = dist[(rowBase + row) * pitch + colBase + col];
            last[ry][cx] = UNCHANGED;
        }
    }
    __syncthreads();
    #pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
            const int row = threadIdx.y + ry * THREAD_ROWS;
            const unsigned int distanceIK = left[row][k];
            #pragma unroll
            for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
                const int col = threadIdx.x + cx * THREAD_COLS;
                const unsigned int candidate = distanceIK + right[k][col];
                if (candidate < values[ry][cx]) {
                    values[ry][cx] = candidate;
                    last[ry][cx] = k;
                }
            }
        }
    }
    #pragma unroll
    for (int ry = 0; ry < TILE / THREAD_ROWS; ++ry) {
        const int row = threadIdx.y + ry * THREAD_ROWS;
        #pragma unroll
        for (int cx = 0; cx < TILE / THREAD_COLS; ++cx) {
            const int col = threadIdx.x + cx * THREAD_COLS;
            const size_t index = (rowBase + row) * pitch + colBase + col;
            dist[index] = values[ry][cx];
            if (last[ry][cx] != UNCHANGED) path[index] = pivotBlock * TILE + last[ry][cx];
        }
    }
}

// Two-dimensional block-cyclic ownership gives each rank O(n^2/P) matrix
// storage and O(n^2/sqrt(P)) panel traffic over a balanced process grid.
struct HybridWorkspace {
    CudaBuffer<> operands, history, panel;
    CudaBuffer<true> hostOperands, hostHistory, hostPanel;
    CudaStream compute, broadcast;
    CudaEvent updated, diagonalReady, panelReady, panelConsumed, operandsConsumed;
    std::vector<MPI_Request> requests;

    HybridWorkspace(size_t rows, size_t columns)
        : operands(rows * TILE), history(2 * TILE * TILE), panel(TILE * columns),
          hostOperands(rows * TILE), hostHistory(2 * TILE * TILE), hostPanel(TILE * columns) {
        requests.reserve((rows * TILE + INT_MAX - 1) / INT_MAX +
                         (columns * TILE + INT_MAX - 1) / INT_MAX);
        cudaCheck(cudaEventRecord(updated.value, compute.value));
        cudaCheck(cudaEventRecord(diagonalReady.value, compute.value));
        cudaCheck(cudaEventRecord(panelConsumed.value, compute.value));
        cudaCheck(cudaEventRecord(operandsConsumed.value, compute.value));
        // Load kernels before timing, including CUDA's lazy module setup.
        cudaFuncAttributes attributes;
        cudaCheck(cudaFuncGetAttributes(&attributes, pivotHistory));
        cudaCheck(cudaFuncGetAttributes(&attributes, pivotRows));
        cudaCheck(cudaFuncGetAttributes(&attributes, columnHistory));
        cudaCheck(cudaFuncGetAttributes(&attributes, updateTiles));
    }

    void broadcastAsync(unsigned int* values, size_t count, int root, MPI_Comm comm) {
        for (size_t offset = 0; offset < count;) {
            const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
            requests.emplace_back();
            mpiCheck(MPI_Ibcast(values + offset, chunk, MPI_UNSIGNED, root, comm, &requests.back()));
            offset += chunk;
        }
    }

    void run(unsigned int* dist, unsigned int* path, int blocks,
             int localRowBlocks, int localColumnBlocks, int rowRank, int columnRank,
             int rowRanks, int columnRanks, MPI_Comm rowComm, MPI_Comm columnComm) {
        const size_t pitch = static_cast<size_t>(localColumnBlocks) * TILE;
        const size_t rowCount = static_cast<size_t>(localRowBlocks) * TILE;
        const dim3 threads(THREAD_COLS, THREAD_ROWS);
        for (int pivot = 0; pivot < blocks; ++pivot) {
            const int ownerRow = pivot % rowRanks, ownerColumn = pivot % columnRanks;
            const size_t localRow = static_cast<size_t>(pivot / rowRanks) * TILE;
            const size_t localColumn = static_cast<size_t>(pivot / columnRanks) * TILE;
            const size_t globalBase = static_cast<size_t>(pivot) * TILE;
            const bool diagonalOwner = rowRank == ownerRow && columnRank == ownerColumn;
            // Finish any prior DMA read before MPI reuses its pinned buffer.
            if ((rowRanks > 1 || columnRanks > 1) &&
                (rowRank == ownerRow || columnRank == ownerColumn))
                cudaCheck(cudaEventSynchronize(diagonalReady.value));
            if (diagonalOwner) {
                cudaCheck(cudaStreamWaitEvent(broadcast.value, updated.value, 0));
                pivotHistory<<<1, threads, 0, broadcast.value>>>(
                    dist, pitch, localRow, localColumn, history.data + TILE * TILE, history.data);
                cudaCheck(cudaGetLastError());
                if (rowRanks > 1 || columnRanks > 1)
                    cudaCheck(cudaMemcpyAsync(hostHistory.data, history.data, 2 * TILE * TILE * sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost, broadcast.value));
                cudaCheck(cudaEventRecord(diagonalReady.value, broadcast.value));
                if (rowRanks > 1 || columnRanks > 1)
                    cudaCheck(cudaEventSynchronize(diagonalReady.value));
            }
            // Only the pivot process row and column need diagonal histories.
            // Avoid a cluster-wide synchronization for every pivot batch.
            requests.clear();
            if (rowRank == ownerRow && columnRanks > 1)
                broadcastAsync(hostHistory.data, 2 * TILE * TILE, ownerColumn, rowComm);
            if (columnRank == ownerColumn && rowRanks > 1)
                broadcastAsync(hostHistory.data, 2 * TILE * TILE, ownerRow, columnComm);
            mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            if (!diagonalOwner && (rowRank == ownerRow || columnRank == ownerColumn)) {
                cudaCheck(cudaStreamWaitEvent(broadcast.value, updated.value, 0));
                cudaCheck(cudaMemcpyAsync(history.data, hostHistory.data, 2 * TILE * TILE * sizeof(unsigned int),
                                          cudaMemcpyHostToDevice, broadcast.value));
                cudaCheck(cudaEventRecord(diagonalReady.value, broadcast.value));
            }
            if (rowRank == ownerRow) {
                const int diagonalColumn = columnRank == ownerColumn ? pivot / columnRanks : INT_MAX;
                pivotRows<<<localColumnBlocks, threads, 0, broadcast.value>>>(
                    dist, pitch, localRow, diagonalColumn,
                    history.data + TILE * TILE, history.data, panel.data);
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaEventRecord(panelReady.value, broadcast.value));
                if (rowRanks > 1) {
                    cudaCheck(cudaEventSynchronize(panelConsumed.value));
                    cudaCheck(cudaMemcpyAsync(hostPanel.data, panel.data, TILE * pitch * sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost, broadcast.value));
                }
            }
            if (columnRank == ownerColumn) {
                cudaCheck(cudaStreamWaitEvent(compute.value, diagonalReady.value, 0));
                columnHistory<<<localRowBlocks, threads, 0, compute.value>>>(
                    dist, path, pitch, localColumn, globalBase, history.data, operands.data);
                cudaCheck(cudaGetLastError());
                if (columnRanks > 1) {
                    cudaCheck(cudaEventSynchronize(operandsConsumed.value));
                    cudaCheck(cudaMemcpyAsync(hostOperands.data, operands.data, rowCount * TILE * sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost, compute.value));
                }
            }

            // Independent row/column collectives overlap on the MPI process
            // grid. Pinned staging also supports MPI without CUDA awareness.
            requests.clear();
            if (columnRanks > 1) {
                if (columnRank == ownerColumn) cudaCheck(cudaStreamSynchronize(compute.value));
                cudaCheck(cudaEventSynchronize(operandsConsumed.value));
                broadcastAsync(hostOperands.data, rowCount * TILE, ownerColumn, rowComm);
            }
            if (rowRanks > 1) {
                if (rowRank == ownerRow) cudaCheck(cudaStreamSynchronize(broadcast.value));
                cudaCheck(cudaEventSynchronize(panelConsumed.value));
                broadcastAsync(hostPanel.data, TILE * pitch, ownerRow, columnComm);
            }
            mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            if (columnRank != ownerColumn) {
                cudaCheck(cudaMemcpyAsync(operands.data, hostOperands.data, rowCount * TILE * sizeof(unsigned int),
                                          cudaMemcpyHostToDevice, compute.value));
                cudaCheck(cudaEventRecord(operandsConsumed.value, compute.value));
            }
            if (rowRank != ownerRow) {
                cudaCheck(cudaMemcpyAsync(panel.data, hostPanel.data, TILE * pitch * sizeof(unsigned int),
                                          cudaMemcpyHostToDevice, compute.value));
                cudaCheck(cudaEventRecord(panelConsumed.value, compute.value));
            } else {
                cudaCheck(cudaStreamWaitEvent(compute.value, panelReady.value, 0));
            }
            const int excludedColumn = columnRank == ownerColumn ? pivot / columnRanks : INT_MAX;
            const int updateColumns = localColumnBlocks - (columnRank == ownerColumn);
            if (updateColumns > 0) {
                for (int rowOffset = 0; rowOffset < localRowBlocks; rowOffset += 65535) {
                    const dim3 grid(updateColumns, std::min(localRowBlocks - rowOffset, 65535));
                    updateTiles<<<grid, threads, 0, compute.value>>>(
                        dist, path, pitch, pivot, excludedColumn, rowOffset, operands.data, panel.data);
                    cudaCheck(cudaGetLastError());
                }
            }
            cudaCheck(cudaEventRecord(updated.value, compute.value));
        }
        cudaCheck(cudaStreamSynchronize(compute.value));
        cudaCheck(cudaStreamSynchronize(broadcast.value));
    }
};

// Gather columns within each process row first, then gather block-cyclic rows
// on column zero. Derived datatypes place panels directly in row-major order.
std::vector<unsigned int> gatherMatrix(const unsigned int* device, size_t n, int blocks,
                                       int localRowBlocks, int localColumnBlocks,
                                       int rowRank, int columnRank, int rowRanks, int columnRanks,
                                       MPI_Comm rowComm, MPI_Comm columnComm) {
    if (n == 0) return {};
    const size_t rows = static_cast<size_t>(localRowBlocks) * TILE;
    const size_t columns = static_cast<size_t>(localColumnBlocks) * TILE;
    const size_t pitch = static_cast<size_t>(blocks) * TILE;
    std::vector<unsigned int> local(rows * columns), rowMatrix;
    cudaCheck(cudaMemcpy(local.data(), device, local.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (columnRanks == 1) {
        rowMatrix.swap(local);
    } else if (columnRank == 0) {
        rowMatrix.resize(rows * pitch);
        std::vector<MPI_Request> requests(columnRanks - 1);
        for (int peer = 1; peer < columnRanks; ++peer) {
            const int peerBlocks = (blocks + columnRanks - 1 - peer) / columnRanks;
            MPI_Datatype tileColumns, resized, matrix;
            mpiCheck(MPI_Type_vector(peerBlocks, TILE, columnRanks * TILE, MPI_UNSIGNED, &tileColumns));
            mpiCheck(MPI_Type_create_resized(tileColumns, 0, static_cast<MPI_Aint>(pitch * sizeof(unsigned int)), &resized));
            mpiCheck(MPI_Type_contiguous(static_cast<int>(rows), resized, &matrix));
            mpiCheck(MPI_Type_commit(&matrix));
            mpiCheck(MPI_Irecv(rowMatrix.data() + peer * TILE, 1, matrix, peer, 0, rowComm, &requests[peer - 1]));
            mpiCheck(MPI_Type_free(&matrix));
            mpiCheck(MPI_Type_free(&resized));
            mpiCheck(MPI_Type_free(&tileColumns));
        }
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row) {
            for (int block = 0; block < localColumnBlocks; ++block)
                std::memcpy(rowMatrix.data() + row * pitch + static_cast<size_t>(block) * columnRanks * TILE,
                            local.data() + row * columns + block * TILE, TILE * sizeof(unsigned int));
        }
        mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    } else {
        MPI_Datatype matrix;
        mpiCheck(MPI_Type_vector(static_cast<int>(rows), static_cast<int>(columns),
                                 static_cast<int>(columns), MPI_UNSIGNED, &matrix));
        mpiCheck(MPI_Type_commit(&matrix));
        mpiCheck(MPI_Send(local.data(), 1, matrix, 0, 0, rowComm));
        mpiCheck(MPI_Type_free(&matrix));
    }
    if (columnRank != 0) return {};
    std::vector<unsigned int>().swap(local);
    MPI_Datatype tileRows, sendType, recvType;
    mpiCheck(MPI_Type_vector(TILE, static_cast<int>(n), static_cast<int>(pitch), MPI_UNSIGNED, &tileRows));
    mpiCheck(MPI_Type_create_resized(tileRows, 0, static_cast<MPI_Aint>(TILE * pitch * sizeof(unsigned int)), &sendType));
    mpiCheck(MPI_Type_free(&tileRows));
    mpiCheck(MPI_Type_commit(&sendType));
    mpiCheck(MPI_Type_vector(TILE, static_cast<int>(n), static_cast<int>(n), MPI_UNSIGNED, &tileRows));
    mpiCheck(MPI_Type_create_resized(tileRows, 0, static_cast<MPI_Aint>(rowRanks) * TILE * n * sizeof(unsigned int), &recvType));
    mpiCheck(MPI_Type_free(&tileRows));
    mpiCheck(MPI_Type_commit(&recvType));
    std::vector<unsigned int> result;
    if (rowRank == 0) {
        result.resize(pitch * n);
        std::vector<MPI_Request> requests(rowRanks - 1);
        for (int peer = 1; peer < rowRanks; ++peer) {
            const int peerBlocks = (blocks + rowRanks - 1 - peer) / rowRanks;
            mpiCheck(MPI_Irecv(result.data() + static_cast<size_t>(peer) * TILE * n,
                               peerBlocks, recvType, peer, 0, columnComm, &requests[peer - 1]));
        }
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row) {
            const size_t globalRow = (row / TILE * rowRanks) * TILE + row % TILE;
            std::memcpy(result.data() + globalRow * n, rowMatrix.data() + row * pitch, n * sizeof(unsigned int));
        }
        mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        result.resize(n * n);
    } else {
        mpiCheck(MPI_Send(rowMatrix.data(), localRowBlocks, sendType, 0, 0, columnComm));
    }
    mpiCheck(MPI_Type_free(&sendType));
    mpiCheck(MPI_Type_free(&recvType));
    return result;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    bool valid = true;
    #pragma omp parallel for reduction(&&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) valid = valid && dist[i * n + i] == 0;
    const size_t sample = std::min(n, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) reduction(&&:valid) schedule(static)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < n; ++k) {
                if (dist[i * n + k] < INF && dist[k * n + j] < INF)
                    valid = valid && dist[i * n + k] + dist[k * n + j] >= dist[i * n + j];
            }
        }
    }
    return valid;
}

bool validateExact(const std::vector<unsigned int>& dist,
                   const std::vector<unsigned int>& path, size_t n) {
    std::vector<unsigned int> reference(n * n), referencePath(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < reference.size(); ++i)
        reference[i] = 1u + static_cast<unsigned int>(
            static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) {
        reference[i * n + i] = 0;
        std::fill(referencePath.begin() + i * n, referencePath.begin() + (i + 1) * n, i);
    }
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = reference[i * n + k] + reference[k * n + j];
                if (candidate < reference[i * n + j]) {
                    reference[i * n + j] = candidate;
                    referencePath[i * n + j] = k;
                }
            }
        }
    }
    if (dist != reference || path != referencePath) {
        printf("Validation failed: distance or path matrix differs from the scalar algorithm\n");
        return false;
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MpiEnvironment mpi(argc, argv);
    int worldRank = 0, worldRanks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldRanks));
    try {
        size_t n = 512;
        bool validate = false, printResults = false;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* text = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long value = std::strtoull(text, &end, 10);
                if (errno || end == text || *end || *text == '-' || value > INT_MAX - TILE) {
                    if (worldRank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", text);
                    return 1;
                }
                n = static_cast<size_t>(value);
            } else if (strcmp(argv[i], "-v") == 0) validate = true;
            else if (strcmp(argv[i], "-r") == 0) printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
                if (worldRank == 0) printUsage(argv[0]);
                return 0;
            } else {
                if (worldRank == 0) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return 1;
            }
        }

        const int blocks = static_cast<int>(std::max(size_t{1}, (n + TILE - 1) / TILE));
        const size_t paddedNodes = static_cast<size_t>(blocks) * TILE;
        int ranks = static_cast<int>(std::min(static_cast<size_t>(worldRanks), static_cast<size_t>(blocks) * blocks));
        int dimensions[2] = {};
        for (;;) {
            dimensions[0] = dimensions[1] = 0;
            mpiCheck(MPI_Dims_create(ranks, 2, dimensions));
            if (dimensions[0] <= blocks && dimensions[1] <= blocks) break;
            --ranks;
        }
        MPI_Comm comm = MPI_COMM_NULL, shared = MPI_COMM_NULL;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &shared));
        int localRank = 0;
        mpiCheck(MPI_Comm_rank(shared, &localRank));
        mpiCheck(MPI_Comm_free(&shared));
        mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
        if (worldRank >= ranks) return 0;
        const int rowRank = worldRank / dimensions[1], columnRank = worldRank % dimensions[1];
        MPI_Comm rowComm, columnComm;
        mpiCheck(MPI_Comm_split(comm, rowRank, columnRank, &rowComm));
        mpiCheck(MPI_Comm_split(comm, columnRank, rowRank, &columnComm));

        // One GPU per MPI rank, respecting per-rank CUDA_VISIBLE_DEVICES.
        // Extra ranks on a node may share a device when explicitly oversubscribed.
        int deviceCount = 0;
        cudaCheck(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "Rank %d: a CUDA device is required.\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cudaCheck(cudaSetDevice(localRank % deviceCount));
        // Respect explicit OpenMP settings; avoid spawning hundreds of threads
        // by default for initialization and small result gathers.
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::min(8, omp_get_max_threads()));
        const int localRowBlocks = (blocks + dimensions[0] - 1 - rowRank) / dimensions[0];
        const int localColumnBlocks = (blocks + dimensions[1] - 1 - columnRank) / dimensions[1];
        const size_t localRows = static_cast<size_t>(localRowBlocks) * TILE;
        const size_t pitch = static_cast<size_t>(localColumnBlocks) * TILE;
        if (paddedNodes > std::numeric_limits<size_t>::max() / paddedNodes / sizeof(unsigned int)) {
            fprintf(stderr, "Matrix dimensions exceed the addressable size.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CudaBuffer<> dist(localRows * pitch), path(localRows * pitch);
        if (worldRank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", n);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI grid: %d x %d; OpenMP threads per rank: %d; CUDA tile: %d\n",
                   dimensions[0], dimensions[1], omp_get_max_threads(), TILE);
            printf("Initializing graph...\n");
        }
        {
            std::vector<unsigned int> initial(localRows * pitch);
            initializeLocalDistances(initial, n, pitch, rowRank, columnRank, dimensions[0], dimensions[1]);
            cudaCheck(cudaMemcpy(dist.data, initial.data(), initial.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        initializePaths<<<static_cast<unsigned int>(std::min(size_t{65535}, (localRows * pitch + 255) / 256)), 256>>>(
            path.data, localRows * pitch, pitch, rowRank, dimensions[0]);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        HybridWorkspace workspace(localRows, pitch);
        if (worldRank == 0) printf("Computing shortest paths...\n");
        mpiCheck(MPI_Barrier(comm));
        const double start = MPI_Wtime();
        workspace.run(dist.data, path.data, blocks, localRowBlocks, localColumnBlocks,
                      rowRank, columnRank, dimensions[0], dimensions[1], rowComm, columnComm);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0;
        mpiCheck(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (worldRank == 0) {
            printf("Computation time: %.3f ms\n", duration * 1000.0);
            const double operations = static_cast<double>(n) * n * n;
            printf("Performance: %.3f GOPS\n", duration > 0 ? operations / duration / 1e9 : 0.0);
        }
        std::vector<unsigned int> result, resultPath;
        if (printResults || validate)
            result = gatherMatrix(dist.data, n, blocks, localRowBlocks, localColumnBlocks,
                                  rowRank, columnRank, dimensions[0], dimensions[1], rowComm, columnComm);
        if (validate && n <= 256)
            resultPath = gatherMatrix(path.data, n, blocks, localRowBlocks, localColumnBlocks,
                                      rowRank, columnRank, dimensions[0], dimensions[1], rowComm, columnComm);
        int status = 0;
        if (worldRank == 0) {
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(result, n);
                if (n <= 256) valid = validateExact(result, resultPath, n) && valid;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, comm));
        mpiCheck(MPI_Comm_free(&rowComm));
        mpiCheck(MPI_Comm_free(&columnComm));
        mpiCheck(MPI_Comm_free(&comm));
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
