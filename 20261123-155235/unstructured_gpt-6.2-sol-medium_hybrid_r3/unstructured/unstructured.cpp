#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <mpi.h>
#include <omp.h>

#ifdef OPEN_MPI
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

constexpr int BLOCK_SIZE = 128;
constexpr val_t TRANSFER_COEFF = 0.8;
enum Face { TOP, BOTTOM, LEFT, RIGHT, FACE_COUNT };

void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data != nullptr) {
            CUDA_CHECK(cudaFree(data));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data = nullptr;
};

struct Tile {
    int row, col;
    int height, width;
};

struct Halo {
    int neighbors[FACE_COUNT];
    int offsets[FACE_COUNT + 1] = {};
};

// Use as many ranks as fit on the grid, preferring a balanced tile layout.
void rankGrid(int size, int ranks, int (&dims)[2]) {
    dims[0] = dims[1] = 1;
    for (int rows = 1; rows <= size && rows <= ranks / rows; ++rows) {
        const int cols = std::min(size, ranks / rows);
        if (rows * cols >= dims[0] * dims[1]) {
            dims[0] = rows;
            dims[1] = cols;
        }
    }
}

Tile tileForRank(MPI_Comm cart, int rank, int size, const int (&dims)[2]) {
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    const auto split = [size](int position, int parts) {
        return static_cast<int>(static_cast<int64_t>(position) * size / parts);
    };
    const int row = split(coords[0], dims[0]);
    const int col = split(coords[1], dims[1]);
    return {row, col, split(coords[0] + 1, dims[0]) - row,
            split(coords[1] + 1, dims[1]) - col};
}

int hostThreads(size_t items, size_t itemsPerThread) {
    return static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), 1 + (items - 1) / itemsPerThread));
}

int blocksFor(int count, int blockSize = BLOCK_SIZE) {
    return 1 + (count - 1) / blockSize;
}

Halo makeHalo(MPI_Comm cart, const Tile& tile) {
    Halo halo;
    MPI_Cart_shift(cart, 0, 1, &halo.neighbors[TOP], &halo.neighbors[BOTTOM]);
    MPI_Cart_shift(cart, 1, 1, &halo.neighbors[LEFT], &halo.neighbors[RIGHT]);
    for (int face = 0; face < FACE_COUNT; ++face) {
        const int length = face < LEFT ? tile.width : tile.height;
        halo.offsets[face + 1] = halo.offsets[face] +
                                (halo.neighbors[face] == MPI_PROC_NULL ? 0 : length);
    }
    return halo;
}

int4 buildMesh(int4* deviceLinks, const Tile& tile, const Halo& halo, int size) {
    const int width = tile.width;
    const int height = tile.height;
    const int cells = height * width;
    std::unique_ptr<int4[]> links(new int4[cells]);

    // Neighbor indices address either the owned cells or the appended ghost faces.
    #pragma omp parallel for schedule(static) num_threads(hostThreads(cells, 32768))
    for (int row = 0; row < height; ++row) {
        const int rowStart = row * width;
        for (int col = 0; col < width; ++col) {
            const int cell = rowStart + col;
            int neighbors[4] = {-1, -1, -1, -1};
            int count = 0;
            // Preserve the serial mesh's flux addition order.
            if (row + 1 < height) {
                neighbors[count++] = cell + width;
            } else if (halo.neighbors[BOTTOM] != MPI_PROC_NULL) {
                neighbors[count++] = cells + halo.offsets[BOTTOM] + col;
            }
            if (row > 0) {
                neighbors[count++] = cell - width;
            } else if (halo.neighbors[TOP] != MPI_PROC_NULL) {
                neighbors[count++] = cells + halo.offsets[TOP] + col;
            }
            if (col + 1 < width) {
                neighbors[count++] = cell + 1;
            } else if (halo.neighbors[RIGHT] != MPI_PROC_NULL) {
                neighbors[count++] = cells + halo.offsets[RIGHT] + row;
            }
            if (col > 0) {
                neighbors[count++] = cell - 1;
            } else if (halo.neighbors[LEFT] != MPI_PROC_NULL) {
                neighbors[count++] = cells + halo.offsets[LEFT] + row;
            }
            links[cell] = make_int4(neighbors[0], neighbors[1], neighbors[2], neighbors[3]);
        }
    }
    CUDA_CHECK(cudaMemcpy(deviceLinks, links.get(), cells * sizeof(int4), cudaMemcpyHostToDevice));

    // The corner sources share the default material's transfer coefficient.
    const auto localCell = [&](int row, int col) {
        if (row < tile.row || row >= tile.row + height ||
            col < tile.col || col >= tile.col + width) {
            return -1;
        }
        return (row - tile.row) * width + col - tile.col;
    };
    const int last = size - 1;
    return make_int4(localCell(0, 0), localCell(last, last),
                     localCell(0, last), localCell(last, 0));
}

template <bool CheckMissing>
__device__ __forceinline__ void updateCell(int cell, const int4* __restrict__ links,
                                           const val_t* __restrict__ current,
                                           val_t* __restrict__ next,
                                           val_t* __restrict__ accumulatedFlux, int4 sources) {
    const int4 adjacent = links[cell];
    const val_t energy = current[cell];
    val_t flux = 0.0;
    if (CheckMissing && adjacent.w < 0) {
        // Inflow takes precedence when all four corners coincide (size == 1).
        if (cell == sources.x || cell == sources.y) {
            flux = 0.5;
        } else if (cell == sources.z || cell == sources.w) {
            flux = -0.5;
        }
    }

    const int neighbors[4] = {adjacent.x, adjacent.y, adjacent.z, adjacent.w};
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        if (CheckMissing && neighbors[step] < 0) {
            break;
        }
        // Each connection has weight 1.0. Match the serial build's contracted
        // scalar remainder for three-neighbor cells, including underflow.
        const val_t weighted = (current[neighbors[step]] - energy) * TRANSFER_COEFF;
        if (CheckMissing && step == 2 && adjacent.w < 0) {
            flux = __fma_rn(weighted, 0.25, flux);
        } else {
            flux += weighted * 0.25;
        }
    }
    next[cell] = energy + flux;
    accumulatedFlux[cell] += fabs(flux);
}

__global__ void updateAll(const int4* links, const val_t* current, val_t* next,
                          val_t* accumulatedFlux, int cells, int4 sources) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < cells) {
        updateCell<true>(cell, links, current, next, accumulatedFlux, sources);
    }
}

__global__ void updateInterior(const int4* links, const val_t* current, val_t* next,
                               val_t* accumulatedFlux, int width, int height) {
    const int col = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width - 1) {
        return;
    }
    for (int row = 1 + blockIdx.y * blockDim.y + threadIdx.y;
         row < height - 1; row += gridDim.y * blockDim.y) {
        updateCell<false>(row * width + col, links, current, next, accumulatedFlux, {});
    }
}

__global__ void updateBoundary(const int4* links, const val_t* current, val_t* next,
                               val_t* accumulatedFlux, int width, int height,
                               int boundaryCells, int4 sources) {
    const int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= boundaryCells) {
        return;
    }
    int cell = edge;
    if (width > 1 && height > 1) {
        if (edge >= 2 * width) {
            const int side = edge - 2 * width;
            cell = (1 + side / 2) * width + (side % 2) * (width - 1);
        } else if (edge >= width) {
            cell = (height - 1) * width + edge - width;
        }
    }
    updateCell<true>(cell, links, current, next, accumulatedFlux, sources);
}

__global__ void packHalo(const val_t* current, val_t* packed, int width, int height, Halo halo) {
    const int face = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = halo.offsets[face];
    if (i >= halo.offsets[face + 1] - offset) {
        return;
    }
    int cell;
    switch (face) {
        case TOP:    cell = i; break;
        case BOTTOM: cell = (height - 1) * width + i; break;
        case LEFT:   cell = i * width; break;
        default:     cell = i * width + width - 1; break;
    }
    packed[offset + i] = current[cell];
}

bool cudaAwareMpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

double runSimulation(const Tile& tile, const Halo& halo, const int4* links, int4 sources,
                     val_t*& current, val_t*& next, val_t* accumulatedFlux, int iterations,
                     MPI_Comm cart) {
    const int width = tile.width;
    const int height = tile.height;
    const int cells = width * height;
    const int ghosts = halo.offsets[FACE_COUNT];
    const bool exchange = ghosts != 0 && iterations > 1;
    const bool staged = exchange && !cudaAwareMpi();
    const size_t haloBytes = ghosts * sizeof(val_t);
    DeviceBuffer<val_t> packed(exchange ? ghosts : 0);
    val_t* hostSend = nullptr;
    val_t* hostReceive = nullptr;
    cudaEvent_t packedReady{};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (exchange) {
        CUDA_CHECK(cudaEventCreateWithFlags(&packedReady, cudaEventDisableTiming));
    }
    if (staged) {
        CUDA_CHECK(cudaMallocHost(&hostSend, haloBytes));
        // Alternate receive buffers so MPI can post receives while CUDA reads the previous halo.
        CUDA_CHECK(cudaMallocHost(&hostReceive, 2 * haloBytes));
    }

    CUDA_CHECK(cudaMemsetAsync(current, 0, (static_cast<size_t>(cells) + ghosts) * sizeof(val_t), stream));
    CUDA_CHECK(cudaMemsetAsync(accumulatedFlux, 0, cells * sizeof(val_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(cart);
    const double start = MPI_Wtime();

    // Initially all ghost energies are zero, so the first step needs no messages.
    if (iterations > 0) {
        updateAll<<<blocksFor(cells), BLOCK_SIZE, 0, stream>>>(
            links, current, next, accumulatedFlux, cells, sources);
        std::swap(current, next);
    }

    for (int step = 1; step < iterations; ++step) {
        if (!exchange) {
            updateAll<<<blocksFor(cells), BLOCK_SIZE, 0, stream>>>(
                links, current, next, accumulatedFlux, cells, sources);
            std::swap(current, next);
            continue;
        }

        constexpr int opposite[FACE_COUNT] = {BOTTOM, TOP, RIGHT, LEFT};
        MPI_Request requests[2 * FACE_COUNT];
        int requestCount = 0;
        val_t* receive = staged ? hostReceive + (step % 2) * ghosts : current + cells;
        const val_t* send = staged ? hostSend : packed.data;
        for (int face = 0; face < FACE_COUNT; ++face) {
            if (halo.neighbors[face] != MPI_PROC_NULL) {
                MPI_Irecv(receive + halo.offsets[face], halo.offsets[face + 1] - halo.offsets[face],
                          MPI_DOUBLE, halo.neighbors[face], opposite[face], cart,
                          &requests[requestCount++]);
            }
        }

        const dim3 packGrid(blocksFor(std::max(width, height)), FACE_COUNT);
        packHalo<<<packGrid, BLOCK_SIZE, 0, stream>>>(current, packed.data, width, height, halo);
        if (staged) {
            CUDA_CHECK(cudaMemcpyAsync(hostSend, packed.data, haloBytes, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaEventRecord(packedReady, stream));

        if (width > 2 && height > 2) {
            const dim3 block(32, 4);
            const dim3 grid(blocksFor(width - 2, block.x),
                            std::min(65535, blocksFor(height - 2, block.y)));
            updateInterior<<<grid, block, 0, stream>>>(links, current, next, accumulatedFlux, width, height);
        }

        CUDA_CHECK(cudaEventSynchronize(packedReady));
        for (int face = 0; face < FACE_COUNT; ++face) {
            if (halo.neighbors[face] != MPI_PROC_NULL) {
                MPI_Isend(send + halo.offsets[face], halo.offsets[face + 1] - halo.offsets[face],
                          MPI_DOUBLE, halo.neighbors[face], face, cart, &requests[requestCount++]);
            }
        }
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);

        if (staged) {
            CUDA_CHECK(cudaMemcpyAsync(current + cells, receive, haloBytes, cudaMemcpyHostToDevice, stream));
        }
        const int boundaryCells = (width == 1 || height == 1) ? cells : 2 * width + 2 * (height - 2);
        updateBoundary<<<blocksFor(boundaryCells), BLOCK_SIZE, 0, stream>>>(
            links, current, next, accumulatedFlux, width, height, boundaryCells, sources);
        std::swap(current, next);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double longest = 0.0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (staged) {
        CUDA_CHECK(cudaFreeHost(hostReceive));
        CUDA_CHECK(cudaFreeHost(hostSend));
    }
    if (exchange) {
        CUDA_CHECK(cudaEventDestroy(packedReady));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return longest;
}

struct Results {
    uint64_t hash = 0;
    val_t energySum = 0.0;
    val_t fluxSum = 0.0;
    val_t energyMin = std::numeric_limits<val_t>::max();
    val_t energyMax = std::numeric_limits<val_t>::lowest();
};

template <bool Validate>
struct CombineResults {
    __device__ Results operator()(Results a, const Results& b) const {
        a.hash ^= b.hash;
        if (Validate) {
            a.energySum += b.energySum;
            a.fluxSum += b.fluxSum;
            a.energyMin = fmin(a.energyMin, b.energyMin);
            a.energyMax = fmax(a.energyMax, b.energyMax);
        }
        return a;
    }
};

template <bool Validate>
__global__ void reduceResults(const val_t* energy, const val_t* flux, Tile tile, int size,
                              Results* partials) {
    using BlockReduce = cub::BlockReduce<Results, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage scratch;
    Results local;
    const int cells = tile.height * tile.width;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cells;
         i += gridDim.x * blockDim.x) {
        const int row = static_cast<int>(i) / tile.width;
        const int col = static_cast<int>(i) % tile.width;
        const uint64_t globalIndex = static_cast<uint64_t>(tile.row + row) * size + tile.col + col;
        const val_t e = energy[i];
        const val_t f = flux[i];
        local.hash ^= (static_cast<uint64_t>(__double_as_longlong(e)) + globalIndex) * 0x9e3779b97f4a7c15ULL;
        local.hash ^= (static_cast<uint64_t>(__double_as_longlong(f)) + globalIndex) * 0xbf58476d1ce4e5b9ULL;
        if (Validate) {
            local.energySum += e;
            local.fluxSum += f;
            local.energyMin = fmin(local.energyMin, e);
            local.energyMax = fmax(local.energyMax, e);
        }
    }
    const Results result = BlockReduce(scratch).Reduce(local, CombineResults<Validate>{});
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = result;
    }
}

Results collectResults(const val_t* energy, const val_t* flux, const Tile& tile,
                       int size, bool validate, MPI_Comm cart) {
    const int blocks = std::min(1024, blocksFor(tile.height * tile.width));
    DeviceBuffer<Results> devicePartials(blocks);
    if (validate) {
        reduceResults<true><<<blocks, BLOCK_SIZE>>>(energy, flux, tile, size, devicePartials.data);
    } else {
        reduceResults<false><<<blocks, BLOCK_SIZE>>>(energy, flux, tile, size, devicePartials.data);
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<Results> partials(blocks);
    CUDA_CHECK(cudaMemcpy(partials.data(), devicePartials.data, blocks * sizeof(Results), cudaMemcpyDeviceToHost));

    uint64_t hash = 0;
    val_t energySum = 0.0, fluxSum = 0.0;
    val_t energyMin = std::numeric_limits<val_t>::max();
    val_t energyMax = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for num_threads(hostThreads(blocks, 256)) reduction(^:hash) \
        reduction(+:energySum,fluxSum) reduction(min:energyMin) reduction(max:energyMax)
    for (int i = 0; i < blocks; ++i) {
        const Results& part = partials[i];
        hash ^= part.hash;
        energySum += part.energySum;
        fluxSum += part.fluxSum;
        energyMin = std::min(energyMin, part.energyMin);
        energyMax = std::max(energyMax, part.energyMax);
    }

    Results global;
    MPI_Reduce(&hash, &global.hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart);
    if (validate) {
        const val_t sums[] = {energySum, fluxSum};
        const val_t extrema[] = {-energyMin, energyMax};
        val_t globalSums[2] = {}, globalExtrema[2] = {};
        MPI_Reduce(sums, globalSums, 2, MPI_DOUBLE, MPI_SUM, 0, cart);
        MPI_Reduce(extrema, globalExtrema, 2, MPI_DOUBLE, MPI_MAX, 0, cart);
        global.energySum = globalSums[0];
        global.fluxSum = globalSums[1];
        global.energyMin = -globalExtrema[0];
        global.energyMax = globalExtrema[1];
    }
    return global;
}

// The external check uses an order-dependent hash and compensated sum.
void printGlobalEnergy(const val_t* energy, const Tile& tile, int size, const int (&dims)[2],
                       MPI_Comm cart, int rank) {
    constexpr int resultTag = FACE_COUNT;
    const int cells = tile.height * tile.width;
    std::unique_ptr<val_t[]> local(new val_t[cells]);
    CUDA_CHECK(cudaMemcpy(local.get(), energy, cells * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (rank != 0) {
        MPI_Send(local.get(), cells, MPI_DOUBLE, 0, resultTag, cart);
        return;
    }

    std::vector<val_t> global(static_cast<size_t>(size) * size);
    const int ranks = dims[0] * dims[1];
    std::vector<MPI_Request> requests(ranks - 1);
    for (int source = 1; source < ranks; ++source) {
        const Tile remote = tileForRank(cart, source, size, dims);
        MPI_Datatype layout;
        MPI_Type_vector(remote.height, remote.width, size, MPI_DOUBLE, &layout);
        MPI_Type_commit(&layout);
        MPI_Irecv(global.data() + static_cast<size_t>(remote.row) * size + remote.col,
                  1, layout, source, resultTag, cart, &requests[source - 1]);
        MPI_Type_free(&layout);
    }
    #pragma omp parallel for schedule(static) num_threads(hostThreads(cells, 32768))
    for (int row = 0; row < tile.height; ++row) {
        std::copy_n(local.get() + static_cast<size_t>(row) * tile.width, tile.width,
                    global.data() + static_cast<size_t>(tile.row + row) * size + tile.col);
    }
    MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE);
    print_results(global, "ElementEnergy");
}

bool validateResults(const Results& results) {
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", results.energySum);
    printf("  Flux sum: %.2f\n", results.fluxSum);
    printf("  Energy range: [%.6f, %.6f]\n", results.energyMin, results.energyMax);

    if (!std::isfinite(results.energySum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(results.energySum) > 1e-8) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(results.fluxSum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(results.energyMax) || !std::isfinite(results.energyMin)) {
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

bool runBenchmark(int size, int iterations, bool validate, bool printResults,
                  MPI_Comm cart, const int (&dims)[2]) {
    int rank;
    MPI_Comm_rank(cart, &rank);
    const Tile tile = tileForRank(cart, rank, size, dims);
    // Check the local index space before constructing the ghost offsets.
    const int64_t localCells = static_cast<int64_t>(tile.width) * tile.height;
    if (localCells + 2LL * (tile.width + static_cast<int64_t>(tile.height)) >
        std::numeric_limits<int>::max()) {
        fprintf(stderr, "Rank %d: mesh tile is too large for local CUDA indices\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return false;
    }
    const int cells = static_cast<int>(localCells);
    const Halo halo = makeHalo(cart, tile);
    const int ghosts = halo.offsets[FACE_COUNT];
    const uint64_t totalCells = static_cast<uint64_t>(size) * size;

    MPI_Comm node;
    MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device is visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return false;
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", size, size, totalCells);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    DeviceBuffer<int4> links(cells);
    DeviceBuffer<val_t> energyA(static_cast<size_t>(cells) + ghosts);
    DeviceBuffer<val_t> energyB(static_cast<size_t>(cells) + ghosts);
    DeviceBuffer<val_t> flux(cells);
    val_t* current = energyA.data;
    val_t* next = energyB.data;
    const int4 sources = buildMesh(links.data, tile, halo, size);

    const uint64_t localGhosts = ghosts;
    uint64_t globalGhosts = 0;
    MPI_Reduce(&localGhosts, &globalGhosts, 1, MPI_UINT64_T, MPI_SUM, 0, cart);
    if (rank == 0) {
        constexpr double mb = 1024.0 * 1024.0;
        const double staticMb = static_cast<double>(totalCells) * sizeof(int4) / mb;
        const double dynamicMb = (3.0 * totalCells + 2.0 * globalGhosts) * sizeof(val_t) / mb;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               staticMb + dynamicMb, staticMb, dynamicMb);
        printf("Running simulation...\n");
    }

    const double seconds = runSimulation(tile, halo, links.data, sources, current, next,
                                         flux.data, iterations, cart);
    if (rank == 0) {
        const int measuredIterations = iterations > 1 ? iterations - 1 : 1;
        const double rate = (static_cast<double>(measuredIterations) * totalCells) / seconds / 1e9;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", seconds * 1000.0 / measuredIterations);
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
    }

    const Results results = collectResults(current, flux.data, tile, size, validate, cart);
    if (rank == 0) {
        printf("  Result hash: %016" PRIX64 "\n\n", results.hash);
    }
    if (printResults) {
        printGlobalEnergy(current, tile, size, dims, cart, rank);
    }
    return rank != 0 || !validate || validateResults(results);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int size = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (size <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Grid size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dims[2];
    rankGrid(size, ranks, dims);
    const int periods[] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
    int status = 0;
    if (cart != MPI_COMM_NULL) {
        status = runBenchmark(size, iterations, validate, printResults, cart, dims) ? 0 : 1;
        MPI_Comm_free(&cart);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
