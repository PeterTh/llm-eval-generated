#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using val_t = double;
constexpr int blockSize = 256;
constexpr int interiorBlockWidth = 32;
constexpr int interiorBlockHeight = blockSize / interiorBlockWidth;
constexpr int directions = 4;
// The accumulation order is part of the numerical result.
enum Direction { Down, Up, Right, Left };

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Tile {
    int row;
    int col;
    int height;
    int width;
};

Tile tileForRank(int n, const int dims[2], int rank) {
    const int x = rank / dims[1];
    const int y = rank % dims[1];
    const int row = static_cast<int>(int64_t{x} * n / dims[0]);
    const int col = static_cast<int>(int64_t{y} * n / dims[1]);
    return {row, col,
            static_cast<int>(int64_t{x + 1} * n / dims[0]) - row,
            static_cast<int>(int64_t{y + 1} * n / dims[1]) - col};
}

struct Halo {
    int offset[directions];
    int count[directions];
    int peer[directions];
    int size = 0;
};

Halo makeHalo(const Tile& tile, int n, int processColumns, int rank, int cells) {
    Halo halo{};
    const int peers[directions] = {
        tile.row + tile.height < n ? rank + processColumns : MPI_PROC_NULL,
        tile.row > 0 ? rank - processColumns : MPI_PROC_NULL,
        tile.col + tile.width < n ? rank + 1 : MPI_PROC_NULL,
        tile.col > 0 ? rank - 1 : MPI_PROC_NULL
    };

    for (int d = 0; d < directions; ++d) {
        halo.peer[d] = peers[d];
        halo.offset[d] = -1;
        if (peers[d] != MPI_PROC_NULL) {
            halo.offset[d] = halo.size;
            halo.count[d] = d < Right ? tile.width : tile.height;
            if (halo.count[d] > std::numeric_limits<int>::max() - cells - halo.size) {
                fprintf(stderr, "Rank %d: tile and halo exceed the local index range\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            halo.size += halo.count[d];
        }
    }
    return halo;
}

int hostThreads(size_t count) {
    // Avoid spawning a full node's worth of threads per GPU for memory-bound setup.
    return static_cast<int>(std::min<size_t>(std::min(omp_get_max_threads(), 16),
                                             1 + count / 32768));
}

void buildConnectivity(const Tile& tile, const Halo& halo, int cells, int* deviceLinks) {
    // Each slot is a coalesced array of local or ghost indices. Missing edges use -1.
    auto links = std::unique_ptr<int[]>(new int[size_t{directions} * cells]);
    #pragma omp parallel for schedule(static) num_threads(hostThreads(cells))
    for (int x = 0; x < tile.height; ++x) {
        for (int y = 0; y < tile.width; ++y) {
            const int i = x * tile.width + y;
            links[i] = x + 1 < tile.height ? i + tile.width :
                (halo.offset[Down] >= 0 ? cells + halo.offset[Down] + y : -1);
            links[size_t{Up} * cells + i] = x > 0 ? i - tile.width :
                (halo.offset[Up] >= 0 ? cells + halo.offset[Up] + y : -1);
            links[size_t{Right} * cells + i] = y + 1 < tile.width ? i + 1 :
                (halo.offset[Right] >= 0 ? cells + halo.offset[Right] + x : -1);
            links[size_t{Left} * cells + i] = y > 0 ? i - 1 :
                (halo.offset[Left] >= 0 ? cells + halo.offset[Left] + x : -1);
        }
    }
    checkCuda(cudaMemcpy(deviceLinks, links.get(), size_t{directions} * cells * sizeof(int),
                         cudaMemcpyHostToDevice));
}

enum class Region { All, Interior, Boundary };

template <Region region>
__global__ void updateEnergy(Tile tile, int n, int cells, int work, Halo halo,
                             const int* __restrict__ links, const val_t* __restrict__ current,
                             val_t* __restrict__ next, val_t* __restrict__ flux,
                             val_t* __restrict__ send) {
    int x = 0;
    int y = 0;
    int i;
    if constexpr (region == Region::Interior) {
        const unsigned int rowBlock = blockIdx.y + blockIdx.z * gridDim.y;
        const unsigned int row = 1 + rowBlock * interiorBlockHeight + threadIdx.y;
        const unsigned int col = 1 + blockIdx.x * interiorBlockWidth + threadIdx.x;
        if (row >= static_cast<unsigned int>(tile.height - 1) ||
            col >= static_cast<unsigned int>(tile.width - 1)) {
            return;
        }
        i = static_cast<int>(row) * tile.width + static_cast<int>(col);
    } else {
        const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
        if (thread >= static_cast<unsigned int>(work)) {
            return;
        }
        const int id = static_cast<int>(thread);
        if constexpr (region == Region::Boundary) {
            // Enumerate full first and last rows, then the ends of the intervening rows.
            if (id < tile.width) {
                y = id;
            } else if (id - tile.width < tile.width) {
                x = tile.height - 1;
                y = id - tile.width;
            } else if (tile.width == 1) {
                x = id - 1;
            } else {
                const int edge = id - 2 * tile.width;
                x = 1 + edge / 2;
                y = (edge % 2) * (tile.width - 1);
            }
            i = x * tile.width + y;
        } else {
            i = id;
        }
    }

    const val_t energy = current[i];
    val_t total = 0.0;
    if constexpr (region == Region::All) {
        if (i == 0 || i == cells - 1) {
            total = 0.5;
        } else if (i == tile.width - 1 || i == cells - tile.width) {
            total = -0.5;
        }
    } else if constexpr (region == Region::Boundary) {
        const int row = tile.row + x;
        const int col = tile.col + y;
        if ((row == 0 || row == n - 1) && (col == 0 || col == n - 1)) {
            total = row == col ? 0.5 : -0.5;
        }
    }
    // Every link in the generated mesh has weight 1 and every material has coefficient 0.8.
    #pragma unroll
    for (int d = 0; d < directions; ++d) {
        const int neighbor = links[static_cast<size_t>(d) * cells + i];
        if (region == Region::Interior || neighbor >= 0) {
            // Preserve the rounded material product before accumulating the quarter-step.
            const val_t transfer = (current[neighbor] - energy) * 0.8;
            total = __fma_rn(transfer, 0.25, total);
        }
    }
    const val_t updated = energy + total;
    next[i] = updated;
    flux[i] += fabs(total);

    if constexpr (region == Region::Boundary) {
        if (send) {
            if (x == tile.height - 1 && halo.offset[Down] >= 0) {
                send[halo.offset[Down] + y] = updated;
            }
            if (x == 0 && halo.offset[Up] >= 0) {
                send[halo.offset[Up] + y] = updated;
            }
            if (y == tile.width - 1 && halo.offset[Right] >= 0) {
                send[halo.offset[Right] + x] = updated;
            }
            if (y == 0 && halo.offset[Left] >= 0) {
                send[halo.offset[Left] + x] = updated;
            }
        }
    }
}

unsigned int blocksFor(int work) {
    return (static_cast<unsigned int>(work) + blockSize - 1) / blockSize;
}

bool cudaAwareMpi() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

val_t* runSimulation(Tile tile, int n, int cells, Halo halo, int iterations,
                     val_t* current, val_t* next, val_t* flux, const int* links,
                     val_t* outgoing, val_t* hostHalo, MPI_Comm comm,
                     cudaStream_t interiorStream, cudaStream_t boundaryStream,
                     cudaEvent_t interiorDone, cudaEvent_t boundaryDone, cudaEvent_t sendReady) {
    if (halo.size == 0) {
        for (int iter = 0; iter < iterations; ++iter) {
            updateEnergy<Region::All><<<blocksFor(cells), blockSize, 0, interiorStream>>>(
                tile, n, cells, cells, halo, links, current, next, flux, nullptr);
            std::swap(current, next);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaStreamSynchronize(interiorStream));
        return current;
    }

    const int interior = std::max(tile.height - 2, 0) * std::max(tile.width - 2, 0);
    const int boundary = cells - interior;
    const size_t haloBytes = static_cast<size_t>(halo.size) * sizeof(val_t);
    const dim3 interiorBlock(interiorBlockWidth, interiorBlockHeight);
    dim3 interiorGrid;
    if (interior != 0) {
        const auto rowBlocks = (static_cast<unsigned int>(tile.height - 2) +
                                interiorBlockHeight - 1) / interiorBlockHeight;
        interiorGrid.x = (static_cast<unsigned int>(tile.width - 2) +
                          interiorBlockWidth - 1) / interiorBlockWidth;
        interiorGrid.y = std::min(rowBlocks, 65535u);
        interiorGrid.z = (rowBlocks + interiorGrid.y - 1) / interiorGrid.y;
    }

    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[2 * directions];
        int requestCount = 0;
        // MPI may start receiving while the previous pinned receive buffer is still
        // feeding the boundary stream. Alternate buffers to keep those accesses disjoint.
        val_t* receive = hostHalo ? hostHalo + static_cast<size_t>(1 + iter % 2) * halo.size
                                 : current + cells;
        val_t* send = hostHalo ? hostHalo : outgoing;
        if (iter != 0) {
            for (int d = 0; d < directions; ++d) {
                if (halo.peer[d] != MPI_PROC_NULL) {
                    MPI_Irecv(receive + halo.offset[d], halo.count[d], MPI_DOUBLE,
                              halo.peer[d], d ^ 1, comm, &requests[requestCount++]);
                }
            }
            if (interior != 0) {
                // Each stream needs the other stream's previous updates. Capture both
                // previous event records before recording any events for this iteration.
                checkCuda(cudaStreamWaitEvent(boundaryStream, interiorDone));
                checkCuda(cudaStreamWaitEvent(interiorStream, boundaryDone));
            }
        }
        if (interior != 0) {
            updateEnergy<Region::Interior><<<interiorGrid, interiorBlock, 0, interiorStream>>>(
                tile, n, cells, interior, halo, links, current, next, flux, nullptr);
            checkCuda(cudaEventRecord(interiorDone, interiorStream));
        }

        if (iter != 0) {
            // GPU interiors execute while the host drives the exchange of previous edges.
            checkCuda(cudaEventSynchronize(sendReady));
            for (int d = 0; d < directions; ++d) {
                if (halo.peer[d] != MPI_PROC_NULL) {
                    MPI_Isend(send + halo.offset[d], halo.count[d], MPI_DOUBLE,
                              halo.peer[d], d, comm, &requests[requestCount++]);
                }
            }
            MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
            if (hostHalo) {
                checkCuda(cudaMemcpyAsync(current + cells, receive, haloBytes,
                                          cudaMemcpyHostToDevice, boundaryStream));
            }
        }

        const bool more = iter + 1 < iterations;
        updateEnergy<Region::Boundary><<<blocksFor(boundary), blockSize, 0, boundaryStream>>>(
            tile, n, cells, boundary, halo, links, current, next, flux, more ? outgoing : nullptr);
        if (more) {
            if (interior != 0) {
                checkCuda(cudaEventRecord(boundaryDone, boundaryStream));
            }
            if (hostHalo) {
                checkCuda(cudaMemcpyAsync(send, outgoing, haloBytes,
                                          cudaMemcpyDeviceToHost, boundaryStream));
            }
            checkCuda(cudaEventRecord(sendReady, boundaryStream));
        }
        std::swap(current, next);
    }

    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(interiorStream));
    checkCuda(cudaStreamSynchronize(boundaryStream));
    return current;
}

__global__ void hashState(Tile tile, int n, int cells, const val_t* energy, const val_t* flux,
                          uint64_t* partials) {
    uint64_t result = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<unsigned int>(cells); i += blockDim.x * gridDim.x) {
        const uint64_t row = tile.row + i / tile.width;
        const uint64_t col = tile.col + i % tile.width;
        const uint64_t global = row * n + col;
        result ^= (static_cast<uint64_t>(__double_as_longlong(energy[i])) + global)
                  * 0x9e3779b97f4a7c15ULL;
        result ^= (static_cast<uint64_t>(__double_as_longlong(flux[i])) + global)
                  * 0xbf58476d1ce4e5b9ULL;
    }
    __shared__ uint64_t warpHashes[blockSize / 32];
    for (int step = 16; step > 0; step /= 2) {
        result ^= __shfl_down_sync(0xffffffff, result, step);
    }
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warpHashes[threadIdx.x / 32] = result;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        result = lane < blockSize / 32 ? warpHashes[lane] : 0;
        for (int step = 16; step > 0; step /= 2) {
            result ^= __shfl_down_sync(0xffffffff, result, step);
        }
        if (lane == 0) {
            partials[blockIdx.x] = result;
        }
    }
}

uint64_t computeHash(Tile tile, int n, int cells, int multiprocessors,
                     const val_t* deviceEnergy, const val_t* deviceFlux) {
    const int blocks = std::min(blocksFor(cells), static_cast<unsigned int>(8 * multiprocessors));
    DeviceBuffer<uint64_t> deviceHashes(blocks);
    hashState<<<blocks, blockSize>>>(tile, n, cells, deviceEnergy, deviceFlux, deviceHashes.data);
    checkCuda(cudaGetLastError());
    std::vector<uint64_t> hashes(blocks);
    checkCuda(cudaMemcpy(hashes.data(), deviceHashes.data, blocks * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    uint64_t hash = 0;
    for (const uint64_t partial : hashes) {
        hash ^= partial;
    }
    return hash;
}

// The optional reports consume the original row-major order. Receive directly into
// the destination tile, in chunks that stay within MPI's int count limit.
std::vector<val_t> gatherField(const val_t* device, int n, const int dims[2],
                               int rank, MPI_Comm comm) {
    constexpr int tag = directions;
    constexpr int chunkElements = 1 << 20;
    const Tile local = tileForRank(n, dims, rank);
    const size_t cells = static_cast<size_t>(local.width) * local.height;
    std::vector<val_t> global;
    if (dims[0] * dims[1] == 1) {
        global.resize(cells);
        checkCuda(cudaMemcpy(global.data(), device, cells * sizeof(val_t), cudaMemcpyDeviceToHost));
        return global;
    }
    auto field = std::unique_ptr<val_t[]>(new val_t[cells]);
    checkCuda(cudaMemcpy(field.get(), device, cells * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        global.resize(static_cast<size_t>(n) * n);
        #pragma omp parallel for schedule(static) num_threads(hostThreads(cells))
        for (int x = 0; x < local.height; ++x) {
            std::copy_n(field.get() + static_cast<size_t>(x) * local.width, local.width,
                        global.data() + static_cast<size_t>(x) * n);
        }
        for (int peer = 1; peer < dims[0] * dims[1]; ++peer) {
            const Tile tile = tileForRank(n, dims, peer);
            const int rows = std::min(tile.height, std::max(1, chunkElements / tile.width));
            MPI_Datatype fullChunk;
            MPI_Type_vector(rows, tile.width, n, MPI_DOUBLE, &fullChunk);
            MPI_Type_commit(&fullChunk);
            int x = 0;
            for (; x <= tile.height - rows; x += rows) {
                MPI_Recv(global.data() + static_cast<size_t>(tile.row + x) * n + tile.col,
                         1, fullChunk, peer, tag, comm, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&fullChunk);
            if (x < tile.height) {
                MPI_Datatype tail;
                MPI_Type_vector(tile.height - x, tile.width, n, MPI_DOUBLE, &tail);
                MPI_Type_commit(&tail);
                MPI_Recv(global.data() + static_cast<size_t>(tile.row + x) * n + tile.col,
                         1, tail, peer, tag, comm, MPI_STATUS_IGNORE);
                MPI_Type_free(&tail);
            }
        }
    } else {
        const int rows = std::max(1, chunkElements / local.width);
        for (int x = 0; x < local.height; ) {
            const int count = std::min(rows, local.height - x);
            MPI_Send(field.get() + static_cast<size_t>(x) * local.width,
                     count * local.width, MPI_DOUBLE, 0, tag, comm);
            x += count;
        }
    }
    return global;
}

bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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
    if (n < 1) {
        if (rank == 0) {
            fprintf(stderr, "Grid size must be positive\n");
        }
        return 1;
    }

    const auto elements = static_cast<uint64_t>(n) * n;
    int active = static_cast<int>(std::min<uint64_t>(elements, ranks));
    int dims[2];
    for (;; --active) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(active, 2, dims);
        if (dims[0] <= n && dims[1] <= n) {
            break;
        }
    }

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) {
        return 0;
    }
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device is visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % devices;
    checkCuda(cudaSetDevice(device));
    int multiprocessors;
    checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));

    const Tile tile = tileForRank(n, dims, rank);
    const int64_t localCells = int64_t{tile.height} * tile.width;
    if (localCells > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Rank %d: tile exceeds the local index range\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int cells = static_cast<int>(localCells);
    const Halo halo = makeHalo(tile, n, dims[1], rank, cells);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, elements);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    DeviceBuffer<int> links(size_t{directions} * cells);
    DeviceBuffer<val_t> first(cells + halo.size);
    DeviceBuffer<val_t> second(cells + halo.size);
    DeviceBuffer<val_t> flux(cells);
    DeviceBuffer<val_t> outgoing(halo.size);
    buildConnectivity(tile, halo, cells, links.data);
    checkCuda(cudaMemset(first.data, 0, static_cast<size_t>(cells + halo.size) * sizeof(val_t)));
    checkCuda(cudaMemset(flux.data, 0, static_cast<size_t>(cells) * sizeof(val_t)));
    checkCuda(cudaDeviceSynchronize());

    uint64_t localValues = 3 * (static_cast<uint64_t>(cells) + halo.size);
    uint64_t dynamicValues = 0;
    MPI_Reduce(&localValues, &dynamicValues, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    val_t* hostHalo = nullptr;
    cudaStream_t interiorStream, boundaryStream = nullptr;
    cudaEvent_t interiorDone = nullptr, boundaryDone = nullptr, sendReady = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&interiorStream, cudaStreamNonBlocking));
    if (halo.size != 0) {
        if (!cudaAwareMpi()) {
            checkCuda(cudaMallocHost(&hostHalo, 3 * static_cast<size_t>(halo.size) * sizeof(val_t)));
        }
        checkCuda(cudaStreamCreateWithFlags(&boundaryStream, cudaStreamNonBlocking));
        checkCuda(cudaEventCreateWithFlags(&interiorDone, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&boundaryDone, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&sendReady, cudaEventDisableTiming));
    }
    if (iterations > 0) {
        // Load the selected kernels before timing, including any driver PTX JIT.
        cudaFuncAttributes attributes;
        if (halo.size == 0) {
            checkCuda(cudaFuncGetAttributes(&attributes, updateEnergy<Region::All>));
        } else {
            checkCuda(cudaFuncGetAttributes(&attributes, updateEnergy<Region::Boundary>));
            if (tile.height > 2 && tile.width > 2) {
                checkCuda(cudaFuncGetAttributes(&attributes, updateEnergy<Region::Interior>));
            }
        }
    }

    if (rank == 0) {
        const double staticMem = elements * (directions * sizeof(int) / (1024.0 * 1024.0));
        const double dynamicMem = dynamicValues * (sizeof(val_t) / (1024.0 * 1024.0));
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               staticMem + dynamicMem, staticMem, dynamicMem);
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const val_t* energy = runSimulation(tile, n, cells, halo, iterations, first.data, second.data,
                                        flux.data, links.data, outgoing.data, hostHalo, comm,
                                        interiorStream, boundaryStream, interiorDone, boundaryDone,
                                        sendReady);
    double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    checkCuda(cudaStreamDestroy(interiorStream));
    if (halo.size != 0) {
        checkCuda(cudaStreamDestroy(boundaryStream));
        checkCuda(cudaEventDestroy(interiorDone));
        checkCuda(cudaEventDestroy(boundaryDone));
        checkCuda(cudaEventDestroy(sendReady));
        if (hostHalo) {
            checkCuda(cudaFreeHost(hostHalo));
        }
    }

    uint64_t localHash = computeHash(tile, n, cells, multiprocessors, energy, flux.data);
    uint64_t hash = 0;
    MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    if (rank == 0) {
        const int measuredIterations = std::max(iterations, 2) - 1;
        const double durationMs = elapsed * 1000.0;
        const double throughput = (static_cast<double>(elements) * measuredIterations) / elapsed / 1e9;
        printf("Computation time: %.3f ms\n", durationMs);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", durationMs / measuredIterations);
        printf("  Elements/sec: %.4f GigaElements/s\n", throughput);
        printf("  Performance: %.4f GFLOPS\n", throughput * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }

    int status = 0;
    if (printResults || validate) {
        const auto energyData = gatherField(energy, n, dims, rank, comm);
        if (rank == 0 && printResults) {
            print_results(energyData, "ElementEnergy");
        }
        if (validate) {
            const auto fluxData = gatherField(flux.data, n, dims, rank, comm);
            if (rank == 0) {
                status = validateResults(energyData, fluxData) ? 0 : 1;
            }
            MPI_Bcast(&status, 1, MPI_INT, 0, comm);
        }
    }
    MPI_Comm_free(&comm);
    return status;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
