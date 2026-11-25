#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>

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
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;
constexpr int BLOCK_SIZE = 256;
constexpr val_t MAX_VALUE = std::numeric_limits<val_t>::max();

// Face order is also the original order of flux accumulation.
enum Face { Bottom, Top, Right, Left, FaceCount };
constexpr Face opposite[] = {Top, Bottom, Left, Right};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template <typename T, bool Host = false>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count) {
        if (count == 0) {
            return;
        }
        if constexpr (Host) {
            CUDA_CHECK(cudaMallocHost(&data, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    ~CudaBuffer() {
        if (data != nullptr) {
            if constexpr (Host) {
                CUDA_CHECK(cudaFreeHost(data));
            } else {
                CUDA_CHECK(cudaFree(data));
            }
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* data = nullptr;
};

struct Range {
    int begin;
    int size;
};

Range splitRange(int length, int parts, int coord) {
    const int extra = length % parts;
    return {coord * (length / parts) + std::min(coord, extra),
            length / parts + (coord < extra)};
}

struct Tile {
    Range rows;
    Range cols;
    int cells;
    int faceSize[FaceCount];
    int faceOffset[FaceCount];
    int peer[FaceCount];
    int haloSize = 0;
};

Tile makeTile(int side, MPI_Comm cart) {
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(cart, 2, dims, periods, coords);

    Tile tile{};
    tile.rows = splitRange(side, dims[0], coords[0]);
    tile.cols = splitRange(side, dims[1], coords[1]);
    const int64_t cells = int64_t{tile.rows.size} * tile.cols.size;
    MPI_Cart_shift(cart, 0, 1, &tile.peer[Top], &tile.peer[Bottom]);
    MPI_Cart_shift(cart, 1, 1, &tile.peer[Left], &tile.peer[Right]);
    int64_t haloSize = 0;
    for (int face = 0; face < FaceCount; ++face) {
        tile.faceSize[face] = face < Right ? tile.cols.size : tile.rows.size;
        if (tile.peer[face] != MPI_PROC_NULL) {
            tile.faceOffset[face] = static_cast<int>(haloSize);
            haloSize += tile.faceSize[face];
            if (cells + haloSize > INT_MAX) {
                break;
            }
        }
    }
    if (cells + haloSize > INT_MAX) {
        fprintf(stderr, "Local tile exceeds the CUDA connectivity index range. Use more MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    tile.cells = static_cast<int>(cells);
    tile.haloSize = static_cast<int>(haloSize);
    return tile;
}

void buildConnectivity(const Tile& tile, int* deviceNeighbors) {
    const size_t cells = tile.cells;
    // One plane per direction keeps the index loads coalesced on the GPU.
    std::unique_ptr<int[]> neighbors(new int[FaceCount * cells]);
    const int rows = tile.rows.size;
    const int cols = tile.cols.size;
    const int threads = std::min({omp_get_max_threads(), rows,
                                  1 + (tile.cells - 1) / 32768});
    int ghost[FaceCount];
    for (int face = 0; face < FaceCount; ++face) {
        ghost[face] = tile.peer[face] == MPI_PROC_NULL ? -1 : tile.cells + tile.faceOffset[face];
    }

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int cell = row * cols + col;
            neighbors[Bottom * cells + cell] = row + 1 < rows ? cell + cols :
                                               ghost[Bottom] < 0 ? -1 : ghost[Bottom] + col;
            neighbors[Top * cells + cell] = row > 0 ? cell - cols :
                                            ghost[Top] < 0 ? -1 : ghost[Top] + col;
            neighbors[Right * cells + cell] = col + 1 < cols ? cell + 1 :
                                              ghost[Right] < 0 ? -1 : ghost[Right] + row;
            neighbors[Left * cells + cell] = col > 0 ? cell - 1 :
                                             ghost[Left] < 0 ? -1 : ghost[Left] + row;
        }
    }
    CUDA_CHECK(cudaMemcpy(deviceNeighbors, neighbors.get(),
                          FaceCount * cells * sizeof(int), cudaMemcpyHostToDevice));
}

enum class Region { Whole, Interior, Boundary };

template <Region region>
__global__ void updateCells(const int* __restrict__ neighbors,
                            const val_t* __restrict__ energy,
                            val_t* __restrict__ nextEnergy,
                            val_t* __restrict__ accumulatedFlux,
                            int rows, int cols, int count,
                            val_t* sendRight, val_t* sendLeft) {
    const unsigned int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= static_cast<unsigned int>(count)) {
        return;
    }

    int cell = item;
    if constexpr (region == Region::Interior) {
        const int innerCols = cols - 2;
        cell = (item / innerCols + 1) * cols + item % innerCols + 1;
    } else if constexpr (region == Region::Boundary) {
        if (rows > 2 && cols > 2) {
            // The two full rows followed by the remaining cells on the two sides.
            if (cell < 2 * cols) {
                if (cell >= cols) {
                    cell += (rows - 2) * cols;
                }
            } else {
                const int edge = cell - 2 * cols;
                cell = (edge / 2 + 1) * cols + (edge % 2) * (cols - 1);
            }
        }
    }

    const size_t cells = rows * cols;
    const int bottom = neighbors[Bottom * cells + cell];
    const int top = neighbors[Top * cells + cell];
    const int right = neighbors[Right * cells + cell];
    const int left = neighbors[Left * cells + cell];
    const val_t current = energy[cell];
    val_t flux = 0.0;
    if constexpr (region != Region::Interior) {
        // All generated cells have transfer coefficient 0.8 and unit edge
        // weights. Only global corner cells have an external energy flow.
        if ((bottom < 0 || top < 0) && (right < 0 || left < 0)) {
            flux = (top < 0) == (left < 0) ? 0.5 : -0.5;
        }
    }

    const int adjacent[] = {bottom, top, right, left};
    #pragma unroll
    for (int face = 0; face < FaceCount; ++face) {
        if (region == Region::Interior || adjacent[face] >= 0) {
            // Match the original multiply and accumulation order.
            const val_t transfer = (energy[adjacent[face]] - current) * 0.8;
            flux = fma(transfer, 0.25, flux);
        }
    }
    const val_t next = current + flux;
    nextEnergy[cell] = next;
    accumulatedFlux[cell] += fabs(flux);

    if constexpr (region == Region::Boundary) {
        // Pack columns as their next values are produced. Rows are contiguous.
        if (sendRight != nullptr || sendLeft != nullptr) {
            const int row = cell / cols;
            const int col = cell % cols;
            if (sendRight != nullptr && col == cols - 1) {
                sendRight[row] = next;
            }
            if (sendLeft != nullptr && col == 0) {
                sendLeft[row] = next;
            }
        }
    }
}

int blocksFor(int count) {
    return 1 + (count - 1) / BLOCK_SIZE;
}

bool cudaAwareMpi() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

val_t* runSimulation(const Tile& tile, MPI_Comm cart, int iterations, val_t* current,
                     val_t* next, val_t* flux, const int* neighbors,
                     val_t* sendColumns, val_t* hostSend, val_t* hostReceive,
                     cudaStream_t compute, cudaStream_t halo) {
    const int rows = tile.rows.size;
    const int cols = tile.cols.size;
    if (tile.haloSize == 0) {
        for (int iter = 0; iter < iterations; ++iter) {
            updateCells<Region::Whole><<<blocksFor(tile.cells), BLOCK_SIZE, 0, compute>>>(
                neighbors, current, next, flux, rows, cols, tile.cells, nullptr, nullptr);
            std::swap(current, next);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(compute));
        return current;
    }

    const int interior = std::max(0, rows - 2) * std::max(0, cols - 2);
    const int boundary = tile.cells - interior;
    val_t* sendRight = tile.peer[Right] == MPI_PROC_NULL ? nullptr : sendColumns;
    val_t* sendLeft = tile.peer[Left] == MPI_PROC_NULL ? nullptr : sendColumns + rows;
    cudaEvent_t boundaryReady, interiorReady[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&boundaryReady, cudaEventDisableTiming));
    if (interior > 0) {
        for (auto& event : interiorReady) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
    }

    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[2 * FaceCount];
        int requestCount = 0;
        val_t* ghosts = current + tile.cells;
        // MPI can fill one pinned buffer while CUDA copies from the other.
        val_t* stagedReceive = hostReceive == nullptr ? nullptr :
                               hostReceive + static_cast<size_t>(iter % 2) * tile.haloSize;
        const val_t* send[] = {current + (rows - 1) * cols, current, sendRight, sendLeft};

        // The initial state and its halos are already zero.
        if (iter > 0) {
            val_t* receive = stagedReceive == nullptr ? ghosts : stagedReceive;
            for (int face = 0; face < FaceCount; ++face) {
                if (tile.peer[face] != MPI_PROC_NULL) {
                    MPI_Irecv(receive + tile.faceOffset[face], tile.faceSize[face], MPI_DOUBLE,
                              tile.peer[face], opposite[face], cart, &requests[requestCount++]);
                    if (hostSend != nullptr) {
                        CUDA_CHECK(cudaMemcpyAsync(hostSend + tile.faceOffset[face], send[face],
                                                  tile.faceSize[face] * sizeof(val_t),
                                                  cudaMemcpyDeviceToHost, halo));
                    }
                }
            }
        }

        if (interior > 0) {
            if (iter > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(compute, boundaryReady, 0));
            }
            updateCells<Region::Interior><<<blocksFor(interior), BLOCK_SIZE, 0, compute>>>(
                neighbors, current, next, flux, rows, cols, interior, nullptr, nullptr);
            CUDA_CHECK(cudaEventRecord(interiorReady[iter % 2], compute));
        }

        if (iter > 0) {
            // Only the preceding boundary update and the optional staging
            // copies must finish before sending. The interior can keep running.
            CUDA_CHECK(cudaStreamSynchronize(halo));
            for (int face = 0; face < FaceCount; ++face) {
                if (tile.peer[face] != MPI_PROC_NULL) {
                    const val_t* data = hostSend == nullptr ? send[face] :
                                        hostSend + tile.faceOffset[face];
                    MPI_Isend(data, tile.faceSize[face], MPI_DOUBLE, tile.peer[face], face,
                              cart, &requests[requestCount++]);
                }
            }
            MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
            if (stagedReceive != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(ghosts, stagedReceive, tile.haloSize * sizeof(val_t),
                                          cudaMemcpyHostToDevice, halo));
            }
            if (interior > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(halo, interiorReady[(iter - 1) % 2], 0));
            }
        }

        updateCells<Region::Boundary><<<blocksFor(boundary), BLOCK_SIZE, 0, halo>>>(
            neighbors, current, next, flux, rows, cols, boundary, sendRight, sendLeft);
        CUDA_CHECK(cudaEventRecord(boundaryReady, halo));
        std::swap(current, next);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(halo));
    CUDA_CHECK(cudaStreamSynchronize(compute));
    CUDA_CHECK(cudaEventDestroy(boundaryReady));
    if (interior > 0) {
        for (auto event : interiorReady) {
            CUDA_CHECK(cudaEventDestroy(event));
        }
    }
    return current;
}

struct Results {
    uint64_t hash;
    val_t energySum;
    val_t fluxSum;
    val_t energyMin;
    val_t energyMax;
};

template <bool validate>
struct CombineResults {
    __host__ __device__ Results operator()(const Results& a, const Results& b) const {
        Results result{};
        result.hash = a.hash ^ b.hash;
        if constexpr (validate) {
            result.energySum = a.energySum + b.energySum;
            result.fluxSum = a.fluxSum + b.fluxSum;
            result.energyMin = a.energyMin < b.energyMin ? a.energyMin : b.energyMin;
            result.energyMax = a.energyMax > b.energyMax ? a.energyMax : b.energyMax;
        }
        return result;
    }
};

template <bool validate>
__global__ void summarize(const val_t* energy, const val_t* flux, Results* partials,
                          int side, int rowStart, int colStart, int cols, int cells) {
    Results local{0, 0.0, 0.0, MAX_VALUE, -MAX_VALUE};
    for (unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;
         cell < static_cast<unsigned int>(cells); cell += gridDim.x * blockDim.x) {
        const uint64_t row = rowStart + cell / cols;
        const uint64_t index = row * side + colStart + cell % cols;
        const val_t e = energy[cell];
        const val_t f = flux[cell];
        local.hash ^= (__double_as_longlong(e) + index) * 0x9e3779b97f4a7c15ULL;
        local.hash ^= (__double_as_longlong(f) + index) * 0xbf58476d1ce4e5b9ULL;
        if constexpr (validate) {
            local.energySum += e;
            local.fluxSum += f;
            local.energyMin = fmin(local.energyMin, e);
            local.energyMax = fmax(local.energyMax, e);
        }
    }
    using Reduce = cub::BlockReduce<Results, BLOCK_SIZE>;
    __shared__ typename Reduce::TempStorage scratch;
    const Results result = Reduce(scratch).Reduce(local, CombineResults<validate>{});
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = result;
    }
}

Results collectResults(const Tile& tile, int side, const val_t* energy, const val_t* flux,
                       bool validate, int multiprocessors, MPI_Comm cart) {
    const int blocks = std::min(4 * multiprocessors, blocksFor(tile.cells));
    CudaBuffer<Results> devicePartials(blocks);
    if (validate) {
        summarize<true><<<blocks, BLOCK_SIZE>>>(energy, flux, devicePartials.data, side,
                                                tile.rows.begin, tile.cols.begin,
                                                tile.cols.size, tile.cells);
    } else {
        summarize<false><<<blocks, BLOCK_SIZE>>>(energy, flux, devicePartials.data, side,
                                                 tile.rows.begin, tile.cols.begin,
                                                 tile.cols.size, tile.cells);
    }
    CUDA_CHECK(cudaGetLastError());
    std::vector<Results> partials(blocks);
    CUDA_CHECK(cudaMemcpy(partials.data(), devicePartials.data, blocks * sizeof(Results),
                          cudaMemcpyDeviceToHost));
    Results local{0, 0.0, 0.0, MAX_VALUE, -MAX_VALUE};
    for (const auto& partial : partials) {
        local = validate ? CombineResults<true>{}(local, partial) :
                           CombineResults<false>{}(local, partial);
    }

    Results global{};
    MPI_Reduce(&local.hash, &global.hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart);
    if (validate) {
        const val_t sums[] = {local.energySum, local.fluxSum};
        val_t globalSums[2]{};
        MPI_Reduce(sums, globalSums, 2, MPI_DOUBLE, MPI_SUM, 0, cart);
        global.energySum = globalSums[0];
        global.fluxSum = globalSums[1];
        MPI_Reduce(&local.energyMin, &global.energyMin, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
        MPI_Reduce(&local.energyMax, &global.energyMax, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    }
    return global;
}

bool validateResults(const Results& result) {
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", result.energySum);
    printf("  Flux sum: %.2f\n", result.fluxSum);
    printf("  Energy range: [%.6f, %.6f]\n", result.energyMin, result.energyMax);

    if (!std::isfinite(result.energySum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(result.energySum) > 1e-8) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(result.fluxSum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(result.energyMax) || !std::isfinite(result.energyMin)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

// The external Kahan sum and FNV hash require the original row-major order.
void printEnergy(const Tile& tile, const val_t* energy, int side, MPI_Comm cart, int rank) {
    std::vector<val_t> local(tile.cells);
    CUDA_CHECK(cudaMemcpy(local.data(), energy, tile.cells * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (rank != 0) {
        MPI_Send(local.data(), tile.cells, MPI_DOUBLE, 0, 0, cart);
        return;
    }

    int size;
    MPI_Comm_size(cart, &size);
    if (size == 1) {
        print_results(local, "ElementEnergy");
        return;
    }
    std::vector<val_t> global(static_cast<size_t>(side) * side);
    for (int row = 0; row < tile.rows.size; ++row) {
        std::copy_n(local.data() + row * tile.cols.size, tile.cols.size,
                    global.data() + static_cast<size_t>(row) * side);
    }
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(cart, 2, dims, periods, coords);
    for (int peer = 1; peer < size; ++peer) {
        MPI_Cart_coords(cart, peer, 2, coords);
        const Range rows = splitRange(side, dims[0], coords[0]);
        const Range cols = splitRange(side, dims[1], coords[1]);
        MPI_Datatype tileType;
        MPI_Type_vector(rows.size, cols.size, side, MPI_DOUBLE, &tileType);
        MPI_Type_commit(&tileType);
        MPI_Recv(global.data() + static_cast<size_t>(rows.begin) * side + cols.begin,
                 1, tileType, peer, 0, cart, MPI_STATUS_IGNORE);
        MPI_Type_free(&tileType);
    }
    print_results(global, "ElementEnergy");
}

int runBenchmark(int side, int iterations, bool validate, bool printResults,
                 MPI_Comm cart, int rank, int localRank) {
    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices are visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    int multiprocessors;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));

    const Tile tile = makeTile(side, cart);
    const size_t cells = static_cast<size_t>(side) * side;
    const size_t localEnergySize = static_cast<size_t>(tile.cells) + tile.haloSize;
    const size_t sendSize = tile.peer[Left] != MPI_PROC_NULL || tile.peer[Right] != MPI_PROC_NULL ?
                            2 * static_cast<size_t>(tile.rows.size) : 0;
    const size_t staged = cudaAwareMpi() ? 0 : tile.haloSize;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", side, side, cells);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    CudaBuffer<int> neighbors(FaceCount * static_cast<size_t>(tile.cells));
    CudaBuffer<val_t> first(localEnergySize), second(localEnergySize), flux(tile.cells);
    CudaBuffer<val_t> sendColumns(sendSize);
    CudaBuffer<val_t, true> hostSend(staged), hostReceive(2 * staged);
    buildConnectivity(tile, neighbors.data);
    CUDA_CHECK(cudaMemset(first.data, 0, localEnergySize * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(flux.data, 0, tile.cells * sizeof(val_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    const uint64_t localDynamic = (2 * localEnergySize + tile.cells + sendSize + 3 * staged) *
                                  sizeof(val_t);
    uint64_t dynamicBytes = 0;
    MPI_Reduce(&localDynamic, &dynamicBytes, 1, MPI_UINT64_T, MPI_SUM, 0, cart);
    if (rank == 0) {
        const size_t staticBytes = cells * FaceCount * sizeof(int);
        constexpr double mib = 1024.0 * 1024.0;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (staticBytes + dynamicBytes) / mib, staticBytes / mib, dynamicBytes / mib);
        printf("Running simulation...\n");
    }

    cudaStream_t compute, halo;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&halo, cudaStreamNonBlocking));
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    val_t* energy = runSimulation(tile, cart, iterations, first.data, second.data, flux.data,
                                  neighbors.data, sendColumns.data, hostSend.data, hostReceive.data,
                                  compute, halo);
    const double localTime = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localTime, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    CUDA_CHECK(cudaStreamDestroy(halo));
    CUDA_CHECK(cudaStreamDestroy(compute));

    const Results result = collectResults(tile, side, energy, flux.data, validate,
                                          multiprocessors, cart);
    if (rank == 0) {
        const double ms = seconds * 1000.0;
        const double perIteration = iterations > 0 ? ms / iterations : 0.0;
        const double throughput = iterations > 0 && seconds > 0 ?
                                  (static_cast<double>(cells) * iterations) / seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", perIteration);
        printf("  Elements/sec: %.4f GigaElements/s\n", throughput);
        printf("  Performance: %.4f GFLOPS\n", throughput * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", result.hash);
    }
    if (printResults) {
        printEnergy(tile, energy, side, cart, rank);
    }
    return rank == 0 && validate && !validateResults(result) ? 1 : 0;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool readInt(const char* arg, int& value) {
    char* end;
    errno = 0;
    const long parsed = std::strtol(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required for the OpenMP host code\n");
        }
        MPI_Finalize();
        return 1;
    }

    int side = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if (strcmp(option, "-v") == 0) {
            validate = true;
        } else if (strcmp(option, "-r") == 0) {
            printResults = true;
        } else if (strcmp(option, "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            const bool gridSize = strcmp(option, "-n") == 0;
            const bool iterationCount = strcmp(option, "-i") == 0;
            if ((gridSize || iterationCount) && i + 1 < argc &&
                readInt(argv[++i], gridSize ? side : iterations) && (!gridSize || side > 0)) {
                continue;
            }
            if (rank == 0) {
                printf("Invalid option or argument: %s\n", option);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Use as many nonempty tiles as possible, breaking ties toward square tiles.
    // Prefer more row partitions when the dimensions differ: row halos are contiguous.
    int dims[2] = {1, 1};
    for (int rows = 1; rows <= side && rows <= size / rows; ++rows) {
        const int cols = std::min(side, size / rows);
        if (rows * cols >= dims[0] * dims[1]) {
            dims[0] = cols;
            dims[1] = rows;
        }
    }

    MPI_Comm node, cart;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    const int periods[] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int status = 0;
    if (cart != MPI_COMM_NULL) {
        try {
            status = runBenchmark(side, iterations, validate, printResults, cart, rank, localRank);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&cart);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
