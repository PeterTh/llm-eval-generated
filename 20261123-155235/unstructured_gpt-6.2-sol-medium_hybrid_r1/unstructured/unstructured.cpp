#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int blockWidth = 32;
constexpr int blockRows = 4;
constexpr int blockSize = blockWidth * blockRows;

// Direction order is part of the floating point update.
enum Face { NextRow, PrevRow, NextCol, PrevCol, FaceCount };

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T, bool pinned = false>
class Buffer {
public:
    explicit Buffer(size_t count) {
        if (count == 0) {
            return;
        }
        if constexpr (pinned) {
            CUDA_CHECK(cudaMallocHost(&data, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    ~Buffer() {
        if (data) {
            if constexpr (pinned) {
                CUDA_CHECK(cudaFreeHost(data));
            } else {
                CUDA_CHECK(cudaFree(data));
            }
        }
    }

    T* get() const { return data; }

private:
    T* data = nullptr;
};

struct Layout {
    int rows = 1;
    int cols = 1;

    int ranks() const { return rows * cols; }
};

Layout chooseLayout(int n, int ranks) {
    Layout best;
    for (int cols = 1; cols <= n && cols <= ranks / cols; ++cols) {
        const int rows = std::min(n, ranks / cols);
        const int used = rows * cols;
        if (used > best.ranks() || (used == best.ranks() && rows - cols < best.rows - best.cols)) {
            best = {rows, cols};
        }
    }
    return best;
}

struct Tile {
    int n;
    int firstRow;
    int firstCol;
    int rows;
    int cols;
    int neighbor[FaceCount];
    int offset[FaceCount];
    int halo = 0;

    __host__ __device__ int cells() const { return rows * cols; }

    __host__ __device__ int globalIndex(int index) const {
        return (firstRow + index / cols) * n + firstCol + index % cols;
    }

    int faceSize(int face) const { return face <= PrevRow ? cols : rows; }

    int haloIndex(int face, int position) const {
        return offset[face] >= 0 ? cells() + offset[face] + position : -1;
    }
};

Tile makeTile(int n, Layout layout, int rank) {
    const int row = rank / layout.cols;
    const int col = rank % layout.cols;
    const int firstRow = static_cast<int64_t>(row) * n / layout.rows;
    const int firstCol = static_cast<int64_t>(col) * n / layout.cols;
    const int endRow = static_cast<int64_t>(row + 1) * n / layout.rows;
    const int endCol = static_cast<int64_t>(col + 1) * n / layout.cols;
    Tile tile{n, firstRow, firstCol, endRow - firstRow, endCol - firstCol,
              {row + 1 < layout.rows ? rank + layout.cols : MPI_PROC_NULL,
               row > 0 ? rank - layout.cols : MPI_PROC_NULL,
               col + 1 < layout.cols ? rank + 1 : MPI_PROC_NULL,
               col > 0 ? rank - 1 : MPI_PROC_NULL},
              {-1, -1, -1, -1}};

    for (int face = 0; face < FaceCount; ++face) {
        if (tile.neighbor[face] != MPI_PROC_NULL) {
            tile.offset[face] = tile.halo;
            tile.halo += tile.faceSize(face);
        }
    }
    return tile;
}

int hostThreads(size_t count, size_t grain) {
    return std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, (count + grain - 1) / grain));
}

void buildMesh(Tile tile, int4* connections) {
    // Every connection in the generated mesh has unit weight. Missing slots
    // are marked -1. Off-tile connections index the appended receive halo.
    const int cells = tile.cells();
    const int threads = std::min(tile.rows, hostThreads(cells, 32768));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int row = 0; row < tile.rows; ++row) {
        for (int col = 0; col < tile.cols; ++col) {
            const int i = row * tile.cols + col;
            connections[i] = make_int4(
                row + 1 < tile.rows ? i + tile.cols : tile.haloIndex(NextRow, col),
                row > 0 ? i - tile.cols : tile.haloIndex(PrevRow, col),
                col + 1 < tile.cols ? i + 1 : tile.haloIndex(NextCol, row),
                col > 0 ? i - 1 : tile.haloIndex(PrevCol, row));
        }
    }
}

__device__ double externalFlow(Tile tile, int row, int col) {
    const int x = tile.firstRow + row;
    const int y = tile.firstCol + col;
    const int last = tile.n - 1;
    if ((x == 0 || x == last) && (y == 0 || y == last)) {
        return x == y ? 0.5 : -0.5;
    }
    return 0.0;
}

__device__ double addConnection(double total, double energy, double neighbor, bool scalarTail = false) {
    const double transfer = (neighbor - energy) * 0.8;
    return scalarTail ? __fma_rn(transfer, 0.25, total) : total + __dmul_rn(transfer, 0.25);
}

template <bool interior>
__device__ double updateElement(int i, double flow, const int4* connections,
                               const double* previous, double* next, double* cumulativeFlux) {
    const int4 neighbors = connections[i];
    const double energy = previous[i];
    // The optimized reference rounds paired connections separately and
    // contracts the third connection on a three-neighbor global edge.
    const bool scalarTail = !interior && ((neighbors.x >= 0) + (neighbors.y >= 0) +
                                          (neighbors.z >= 0) + (neighbors.w >= 0) == 3);
    if (interior || neighbors.x >= 0) {
        flow = addConnection(flow, energy, previous[neighbors.x]);
    }
    if (interior || neighbors.y >= 0) {
        flow = addConnection(flow, energy, previous[neighbors.y]);
    }
    if (interior || neighbors.z >= 0) {
        flow = addConnection(flow, energy, previous[neighbors.z], scalarTail && neighbors.w < 0);
    }
    if (interior || neighbors.w >= 0) {
        flow = addConnection(flow, energy, previous[neighbors.w], scalarTail);
    }
    const double updated = energy + flow;
    next[i] = updated;
    // The accumulated flux has a single writer and is never read by neighbors.
    cumulativeFlux[i] += fabs(flow);
    return updated;
}

__global__ void updateMesh(Tile tile, const int4* __restrict__ connections, const double* __restrict__ previous,
                           double* __restrict__ next, double* __restrict__ cumulativeFlux, double* __restrict__ send,
                           int interiorBlocks, int blocksPerRow, int perimeter) {
    const int block = blockIdx.x;
    if (block < interiorBlocks) {
        const int row = 1 + (block / blocksPerRow) * blockRows + threadIdx.y;
        const int col = 1 + (block % blocksPerRow) * blockWidth + threadIdx.x;
        if (row < tile.rows - 1 && col < tile.cols - 1) {
            updateElement<true>(row * tile.cols + col, 0.0, connections, previous, next, cumulativeFlux);
        }
        return;
    }

    const int i = (block - interiorBlocks) * blockSize + threadIdx.y * blockWidth + threadIdx.x;
    if (i >= perimeter) {
        return;
    }

    // Enumerate the end rows first, then the side columns without the corners.
    const int endRows = (tile.rows > 1 ? 2 : 1) * tile.cols;
    const int sides = tile.cols > 1 ? 2 : 1;
    const int row = i < endRows ? (i < tile.cols ? 0 : tile.rows - 1) : 1 + (i - endRows) / sides;
    const int col = i < endRows ? i % tile.cols : ((i - endRows) % sides) * (tile.cols - 1);
    const double updated = updateElement<false>(row * tile.cols + col, externalFlow(tile, row, col),
                                                connections, previous, next, cumulativeFlux);
    // Pack the next exchange while the boundary values are already in registers.
    if (send) {
        if (row == tile.rows - 1 && tile.offset[NextRow] >= 0) {
            send[tile.offset[NextRow] + col] = updated;
        }
        if (row == 0 && tile.offset[PrevRow] >= 0) {
            send[tile.offset[PrevRow] + col] = updated;
        }
        if (col == tile.cols - 1 && tile.offset[NextCol] >= 0) {
            send[tile.offset[NextCol] + row] = updated;
        }
        if (col == 0 && tile.offset[PrevCol] >= 0) {
            send[tile.offset[PrevCol] + row] = updated;
        }
    }
}

__device__ uint64_t warpXor(uint64_t value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value ^= __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void hashMesh(Tile tile, const double* __restrict__ energy, const double* __restrict__ flux,
                         uint64_t* __restrict__ partial) {
    uint64_t result = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < tile.cells(); i += gridDim.x * blockDim.x) {
        const uint64_t global = tile.globalIndex(i);
        result ^= (__double_as_longlong(energy[i]) + global) * 0x9e3779b97f4a7c15ULL;
        result ^= (__double_as_longlong(flux[i]) + global) * 0xbf58476d1ce4e5b9ULL;
    }

    __shared__ uint64_t warps[blockSize / blockWidth];
    result = warpXor(result);
    const int lane = threadIdx.x % blockWidth;
    if (lane == 0) {
        warps[threadIdx.x / blockWidth] = result;
    }
    __syncthreads();
    if (threadIdx.x < blockWidth) {
        result = warpXor(lane < blockSize / blockWidth ? warps[lane] : 0);
        if (lane == 0) {
            partial[blockIdx.x] = result;
        }
    }
}

class Simulation {
public:
    explicit Simulation(Tile tile)
        : tile(tile), connections(tile.cells()),
          energy{Buffer<double>(tile.cells() + tile.halo), Buffer<double>(tile.cells() + tile.halo)},
          flux(tile.cells()), packed(tile.halo), send(tile.halo), receive(2 * tile.halo) {
        const size_t cells = tile.cells();
        std::unique_ptr<int4[]> mesh(new int4[cells]);
        buildMesh(tile, mesh.get());
        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMemcpyAsync(connections.get(), mesh.get(), cells * sizeof(int4),
                                   cudaMemcpyHostToDevice, computeStream));
        CUDA_CHECK(cudaMemsetAsync(energy[0].get(), 0, (cells + tile.halo) * sizeof(double), computeStream));
        CUDA_CHECK(cudaMemsetAsync(flux.get(), 0, cells * sizeof(double), computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        // Force lazy CUDA module loading before timing the simulation.
        cudaFuncAttributes attributes;
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateMesh));
        if (tile.halo) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&interiorDone, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&boundaryDone, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&sendReady, cudaEventDisableTiming));
        }
    }

    ~Simulation() {
        if (tile.halo) {
            CUDA_CHECK(cudaEventDestroy(sendReady));
            CUDA_CHECK(cudaEventDestroy(boundaryDone));
            CUDA_CHECK(cudaEventDestroy(interiorDone));
            CUDA_CHECK(cudaStreamDestroy(haloStream));
        }
        CUDA_CHECK(cudaStreamDestroy(computeStream));
    }

    void run(int iterations, MPI_Comm comm) {
        const int blocksPerRow = (std::max(tile.cols - 2, 0) + blockWidth - 1) / blockWidth;
        const int interiorBlocks = blocksPerRow * ((std::max(tile.rows - 2, 0) + blockRows - 1) / blockRows);
        const int perimeter = std::min(tile.rows, 2) * tile.cols +
                              std::max(tile.rows - 2, 0) * std::min(tile.cols, 2);
        const int boundaryBlocks = (perimeter + blockSize - 1) / blockSize;
        const dim3 threads(blockWidth, blockRows);
        const size_t haloBytes = tile.halo * sizeof(double);

        for (int iter = 0; iter < iterations; ++iter) {
            double* previous = energy[iter % 2].get();
            double* next = energy[1 - iter % 2].get();
            if (!tile.halo) {
                updateMesh<<<interiorBlocks + boundaryBlocks, threads, 0, computeStream>>>(
                    tile, connections.get(), previous, next, flux.get(), nullptr,
                    interiorBlocks, blocksPerRow, perimeter);
                CUDA_CHECK(cudaGetLastError());
                continue;
            }

            // The initial energy everywhere is zero, including the receive halo.
            MPI_Request requests[2 * FaceCount];
            int pending = 0;
            double* incoming = receive.get() + (iter % 2) * tile.halo;
            if (iter > 0) {
                for (int face = 0; face < FaceCount; ++face) {
                    if (tile.offset[face] >= 0) {
                        MPI_Irecv(incoming + tile.offset[face], tile.faceSize(face), MPI_DOUBLE,
                                  tile.neighbor[face], 0, comm, &requests[pending++]);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(send.get(), packed.get(), haloBytes, cudaMemcpyDeviceToHost, haloStream));
                CUDA_CHECK(cudaEventRecord(sendReady, haloStream));
                // Boundary updates require the previous interior. Halo sends
                // only use previous boundary values, so they can start earlier.
                CUDA_CHECK(cudaStreamWaitEvent(haloStream, interiorDone, 0));
                CUDA_CHECK(cudaStreamWaitEvent(computeStream, boundaryDone, 0));
            }

            if (interiorBlocks) {
                updateMesh<<<interiorBlocks, threads, 0, computeStream>>>(
                    tile, connections.get(), previous, next, flux.get(), nullptr,
                    interiorBlocks, blocksPerRow, 0);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(interiorDone, computeStream));

            if (iter > 0) {
                CUDA_CHECK(cudaEventSynchronize(sendReady));
                for (int face = 0; face < FaceCount; ++face) {
                    if (tile.offset[face] >= 0) {
                        MPI_Isend(send.get() + tile.offset[face], tile.faceSize(face), MPI_DOUBLE,
                                  tile.neighbor[face], 0, comm, &requests[pending++]);
                    }
                }
                MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
                // By the next use of this host slot, the intervening sendReady
                // event will have completed this copy on the same stream.
                CUDA_CHECK(cudaMemcpyAsync(previous + tile.cells(), incoming, haloBytes,
                                           cudaMemcpyHostToDevice, haloStream));
            }

            updateMesh<<<boundaryBlocks, threads, 0, haloStream>>>(
                tile, connections.get(), previous, next, flux.get(),
                iter + 1 < iterations ? packed.get() : nullptr, 0, 0, perimeter);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(boundaryDone, haloStream));
        }

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        if (tile.halo) {
            CUDA_CHECK(cudaStreamSynchronize(haloStream));
        }
        current = std::max(iterations, 0) % 2;
    }

    uint64_t resultHash(int device) const {
        int multiprocessors;
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
        const int blocks = std::min((tile.cells() - 1) / blockSize + 1, 4 * multiprocessors);
        Buffer<uint64_t> deviceHashes(blocks);
        std::vector<uint64_t> hostHashes(blocks);
        hashMesh<<<blocks, blockSize, 0, computeStream>>>(
            tile, energy[current].get(), flux.get(), deviceHashes.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hostHashes.data(), deviceHashes.get(), blocks * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        uint64_t result = 0;
        for (uint64_t value : hostHashes) {
            result ^= value;
        }
        return result;
    }

    void download(std::vector<double>& hostEnergy, std::vector<double>& hostFlux, bool validate) const {
        hostEnergy.resize(tile.cells());
        CUDA_CHECK(cudaMemcpy(hostEnergy.data(), energy[current].get(), hostEnergy.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (validate) {
            hostFlux.resize(tile.cells());
            CUDA_CHECK(cudaMemcpy(hostFlux.data(), flux.get(), hostFlux.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
    }

private:
    Tile tile;
    Buffer<int4> connections;
    Buffer<double> energy[2];
    Buffer<double> flux;
    Buffer<double> packed;
    Buffer<double, true> send;
    Buffer<double, true> receive;
    cudaStream_t computeStream{};
    cudaStream_t haloStream{};
    cudaEvent_t interiorDone{};
    cudaEvent_t boundaryDone{};
    cudaEvent_t sendReady{};
    int current = 0;
};

void printGlobalResults(const std::vector<double>& local, Tile tile, Layout layout, int rank, MPI_Comm comm) {
    if (layout.ranks() == 1) {
        print_results(local, "ElementEnergy");
        return;
    }
    if (rank != 0) {
        MPI_Send(local.data(), tile.cells(), MPI_DOUBLE, 0, 1, comm);
        return;
    }

    // The external verifier needs the original row-major order for its hash
    // and compensated sum. Receive each tile directly into its global rows.
    std::vector<double> global(static_cast<size_t>(tile.n) * tile.n);
    std::vector<MPI_Request> requests(layout.ranks() - 1);
    for (int source = 1; source < layout.ranks(); ++source) {
        const Tile other = makeTile(tile.n, layout, source);
        MPI_Datatype rows;
        MPI_Type_vector(other.rows, other.cols, tile.n, MPI_DOUBLE, &rows);
        MPI_Type_commit(&rows);
        MPI_Irecv(global.data() + other.globalIndex(0), 1, rows, source, 1, comm, &requests[source - 1]);
        MPI_Type_free(&rows);
    }

    const int threads = std::min(tile.rows, hostThreads(tile.cells(), 32768));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int row = 0; row < tile.rows; ++row) {
        std::copy_n(local.data() + row * tile.cols, tile.cols,
                    global.data() + tile.globalIndex(row * tile.cols));
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    print_results(global, "ElementEnergy");
}

bool validateResults(const std::vector<double>& energy, const std::vector<double>& flux, int rank, MPI_Comm comm) {
    double energySum = 0.0;
    double fluxSum = 0.0;
    double energyMin = std::numeric_limits<double>::max();
    double energyMax = std::numeric_limits<double>::lowest();

    const int threads = hostThreads(energy.size(), 32768);
    #pragma omp parallel for schedule(static) num_threads(threads) \
        reduction(+:energySum, fluxSum) reduction(min:energyMin) reduction(max:energyMax)
    for (size_t i = 0; i < energy.size(); ++i) {
        energySum += energy[i];
        fluxSum += flux[i];
        energyMin = std::min(energyMin, energy[i]);
        energyMax = std::max(energyMax, energy[i]);
    }

    const double localSums[] = {energySum, fluxSum};
    const double localMinima[] = {energyMin, -energyMax};
    double sums[2], minima[2];
    MPI_Reduce(localSums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(localMinima, minima, 2, MPI_DOUBLE, MPI_MIN, 0, comm);
    if (rank != 0) {
        return true;
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", sums[0]);
    std::printf("  Flux sum: %.2f\n", sums[1]);
    std::printf("  Energy range: [%.6f, %.6f]\n", minima[0], -minima[1]);
    if (!std::isfinite(sums[0])) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(sums[0]) > 1e-8) {
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(sums[1])) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(minima[0]) || !std::isfinite(minima[1])) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    std::printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return EXIT_FAILURE;
        }
    }

    if (n <= 0 || n > INT_MAX / n) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid size must be positive and contain at most %d elements\n", INT_MAX);
        }
        return EXIT_FAILURE;
    }
    const int cells = n * n;
    const Layout layout = chooseLayout(n, ranks);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < layout.ranks() ? 0 : MPI_UNDEFINED, rank, &comm);
    int result = EXIT_SUCCESS;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);

        int devices;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            checkCuda(cudaErrorNoDevice, "GPU selection");
        }
        const int device = localRank % devices;
        CUDA_CHECK(cudaSetDevice(device));

        if (rank == 0) {
            std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
            std::printf("============================================\n");
            std::printf("Grid size: %d x %d = %d elements\n", n, n, cells);
            std::printf("Iterations: %d\n", iterations);
            std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
            std::printf("Building unstructured mesh...\n");
        }

        const Tile tile = makeTile(n, layout, rank);
        Simulation simulation(tile);
        const uint64_t localDynamic = (3ULL * tile.cells() + 6ULL * tile.halo) * sizeof(double);
        uint64_t dynamicBytes;
        MPI_Reduce(&localDynamic, &dynamicBytes, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
        if (rank == 0) {
            const double staticMB = cells * sizeof(int4) / (1024.0 * 1024.0);
            const double dynamicMB = dynamicBytes / (1024.0 * 1024.0);
            std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                        staticMB + dynamicMB, staticMB, dynamicMB);
            std::printf("Running simulation...\n");
        }

        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        simulation.run(iterations, comm);
        const double elapsed = MPI_Wtime() - start;
        double duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        const uint64_t localHash = simulation.resultHash(device);
        uint64_t hash;
        MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
        if (rank == 0) {
            const double timePerIteration = iterations > 0 ? duration * 1000.0 / iterations : 0.0;
            const double rate = iterations > 0 && duration > 0.0 ? double(iterations) * cells / duration / 1e9 : 0.0;
            std::printf("Computation time: %.3f ms\n", duration * 1000.0);
            std::printf("Performance:\n");
            std::printf("  Time per iteration: %.4f ms\n", timePerIteration);
            std::printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            std::printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
            std::printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }

        if (printResults || validate) {
            std::vector<double> energy, flux;
            simulation.download(energy, flux, validate);
            if (printResults) {
                printGlobalResults(energy, tile, layout, rank, comm);
            }
            if (validate && !validateResults(energy, flux, rank, comm)) {
                result = EXIT_FAILURE;
            }
        }
        MPI_Comm_free(&comm);
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int result = EXIT_FAILURE;
    try {
        result = benchmark(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Finalize();
    return result;
}
