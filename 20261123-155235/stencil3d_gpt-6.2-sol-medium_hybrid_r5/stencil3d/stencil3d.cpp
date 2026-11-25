#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

namespace {

using Real = double;

void checkCuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, call, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

struct Extent {
    size_t x, y, z;

    size_t cells() const { return x * y * z; }
};

struct Tile {
    Extent offset;
    Extent size;
};

// Rank order is X fastest, matching the order of the cells within each tile.
Tile tileForRank(Extent global, const std::array<int, 3>& parts, int rank) {
    const auto split = [](size_t length, int count, int index) {
        const size_t extra = length % count;
        return std::array<size_t, 2>{
            (length / count) * index + std::min(extra, size_t(index)),
            length / count + (size_t(index) < extra)
        };
    };
    const auto x = split(global.x, parts[0], rank % parts[0]);
    const auto y = split(global.y, parts[1], (rank / parts[0]) % parts[1]);
    const auto z = split(global.z, parts[2], rank / parts[0] / parts[1]);
    return {{x[0], y[0], z[0]}, {x[1], y[1], z[1]}};
}

std::array<int, 3> chooseDecomposition(Extent global, int ranks) {
    // Balance the largest tile and minimize the area of the internal faces.
    // A contiguous Z face is cheaper to exchange than a packed X or Y face.
    for (int active = std::min<size_t>(ranks, global.cells()); active > 0; --active) {
        std::array<int, 3> best{};
        long double bestCost = std::numeric_limits<long double>::infinity();
        for (int x = 1; x <= active && size_t(x) <= global.x; ++x) {
            if (active % x != 0) continue;
            const int yz = active / x;
            for (int y = 1; y <= yz && size_t(y) <= global.y; ++y) {
                if (yz % y != 0) continue;
                const int z = yz / y;
                if (size_t(z) > global.z) continue;
                const long double maxTile =
                    static_cast<long double>(1 + (global.x - 1) / x)
                    * (1 + (global.y - 1) / y) * (1 + (global.z - 1) / z);
                const long double halo =
                    1.15L * (x - 1) / global.x
                    + 1.05L * (y - 1) / global.y
                    + 1.00L * (z - 1) / global.z;
                const long double cost = maxTile / (global.cells() / static_cast<long double>(active))
                                       + 4 * halo;
                if (cost < bestCost) {
                    bestCost = cost;
                    best = {x, y, z};
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

struct Layout {
    // Owned coordinates start at (1,1,1), leaving one ghost cell on each side.
    Extent owned;
    size_t row, plane;

    explicit Layout(Extent size) : owned(size), row(size.x + 2), plane(row * (size.y + 2)) {}

    __host__ __device__ size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * row + x;
    }

    size_t bytes() const { return plane * (owned.z + 2) * sizeof(Real); }
};

struct Box {
    Extent begin, end;

    Extent size() const { return {end.x - begin.x, end.y - begin.y, end.z - begin.z}; }
};

__device__ Real average(Real center, Real left, Real right, Real front,
                        Real back, Real bottom, Real top) {
    const Real sum = center + left + right + front + back + bottom + top;
    if (sum < 0x1p-1018) return sum / 7.0;

    // Correct the rounded reciprocal product with its exact residual. For
    // these bounded normal sums, sum / 7 is at least 1/14 ulp from a
    // rounding midpoint, so reciprocal error in the correction cannot
    // change the rounded result.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = sum * reciprocal;
    return __fma_rn(__fma_rn(-7.0, quotient, sum), reciprocal, quotient);
}

constexpr unsigned blockX = 32;
constexpr unsigned blockY = 4;
constexpr unsigned zChunk = 8;
constexpr unsigned linearBlock = 256;

// Consecutive lanes cover X. Reuse the three center-column values as Z advances.
__global__ void stencilBulk(const Real* __restrict__ input, Real* __restrict__ output,
                            Layout layout, Box box) {
    const size_t x = box.begin.x + size_t(blockIdx.x) * blockX + threadIdx.x;
    if (x >= box.end.x) return;

    for (size_t y = box.begin.y + blockIdx.y * blockY + threadIdx.y;
         y < box.end.y; y += gridDim.y * blockY) {
        for (size_t first = box.begin.z + blockIdx.z * zChunk;
             first < box.end.z; first += gridDim.z * zChunk) {
            size_t cell = layout.index(x, y, first);
            Real bottom = input[cell - layout.plane];
            Real center = input[cell];
            const size_t last = first + zChunk < box.end.z ? first + zChunk : box.end.z;
            for (size_t z = first; z < last; ++z, cell += layout.plane) {
                const Real top = input[cell + layout.plane];
                output[cell] = average(center, input[cell - 1], input[cell + 1],
                                       input[cell - layout.row], input[cell + layout.row],
                                       bottom, top);
                bottom = center;
                center = top;
            }
        }
    }
}

struct Shell {
    Box boxes[6];
    size_t endBlock[6];
    unsigned count = 0;

    void add(Box box) {
        const size_t cells = box.size().cells();
        if (cells == 0) return;
        const size_t previous = count == 0 ? 0 : endBlock[count - 1];
        boxes[count] = box;
        endBlock[count++] = previous + 1 + (cells - 1) / linearBlock;
    }

    size_t blocks() const { return count == 0 ? 0 : endBlock[count - 1]; }
};

// Each block belongs to one disjoint face of the shell, including any edges
// or corners assigned to that face when the bulk box is carved out.
template <typename Index>
__global__ void stencilShell(const Real* __restrict__ input, Real* __restrict__ output,
                             Layout layout, Shell shell) {
    size_t first = 0;
    unsigned face = 0;
    while (blockIdx.x >= shell.endBlock[face]) {
        first = shell.endBlock[face++];
    }
    const Box box = shell.boxes[face];
    const Index width = box.end.x - box.begin.x;
    const Index height = box.end.y - box.begin.y;
    const Index lane = (blockIdx.x - first) * linearBlock + threadIdx.x;
    const Index layer = lane / (width * height);
    if (layer >= box.end.z - box.begin.z) return;
    const size_t cell = layout.index(box.begin.x + lane % width,
                                     box.begin.y + (lane / width) % height,
                                     box.begin.z + layer);
    output[cell] = average(input[cell], input[cell - 1], input[cell + 1],
                           input[cell - layout.row], input[cell + layout.row],
                           input[cell - layout.plane], input[cell + layout.plane]);
}

template <bool pack>
__global__ void copyFace(Real* grid, Real* buffer, Layout layout, int axis,
                         size_t coordinate, size_t width) {
    const size_t lane = size_t(blockIdx.x) * blockX + threadIdx.x;
    if (lane >= width) return;
    for (size_t z = blockIdx.y * blockY + threadIdx.y; z < layout.owned.z; z += gridDim.y * blockY) {
        const size_t cell = axis == 0 ? layout.index(coordinate, lane + 1, z + 1)
                                      : layout.index(lane + 1, coordinate, z + 1);
        if constexpr (pack) buffer[z * width + lane] = grid[cell];
        else grid[cell] = buffer[z * width + lane];
    }
}

bool cudaAwareMPI() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

struct Face {
    int side, peer;
    size_t count, owned, ghost;
    Real* sendDevice = nullptr;
    Real* recvDevice = nullptr;
    Real* sendHost = nullptr;
    Real* recvHost = nullptr;

    Face(int side, int peer, Layout layout, bool direct)
        : side(side), peer(peer) {
        const size_t lengths[] = {layout.owned.x, layout.owned.y, layout.owned.z};
        const size_t areas[] = {layout.owned.y * layout.owned.z,
                                layout.owned.x * layout.owned.z, layout.plane};
        const int axis = side / 2;
        count = areas[axis];
        owned = side % 2 ? lengths[axis] : 1;
        ghost = side % 2 ? lengths[axis] + 1 : 0;
        if (axis != 2) {
            CUDA_CHECK(cudaMalloc(&sendDevice, count * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&recvDevice, count * sizeof(Real)));
        }
        if (!direct) {
            CUDA_CHECK(cudaMallocHost(&sendHost, count * sizeof(Real)));
            CUDA_CHECK(cudaMallocHost(&recvHost, count * sizeof(Real)));
        }
    }

    void release() const {
        if (sendDevice) CUDA_CHECK(cudaFree(sendDevice));
        if (recvDevice) CUDA_CHECK(cudaFree(recvDevice));
        if (sendHost) CUDA_CHECK(cudaFreeHost(sendHost));
        if (recvHost) CUDA_CHECK(cudaFreeHost(recvHost));
    }

    Real* outgoing(Real* grid, Layout layout) const {
        return side / 2 == 2 ? grid + owned * layout.plane : sendDevice;
    }

    Real* incoming(Real* grid, Layout layout) const {
        return side / 2 == 2 ? grid + ghost * layout.plane : recvDevice;
    }

    template <bool pack>
    void transfer(Real* grid, Real* buffer, Layout layout, size_t coordinate, cudaStream_t stream) const {
        const int axis = side / 2;
        const size_t width = axis == 0 ? layout.owned.y : layout.owned.x;
        const dim3 blocks(1 + (width - 1) / blockX, std::min<size_t>(65535, 1 + (layout.owned.z - 1) / blockY));
        const dim3 threads(blockX, blockY);
        copyFace<pack><<<blocks, threads, 0, stream>>>(grid, buffer, layout, axis, coordinate, width);
        CUDA_CHECK(cudaGetLastError());
    }

    void prepare(Real* grid, Layout layout, cudaStream_t stream) const {
        Real* buffer = outgoing(grid, layout);
        if (sendDevice) transfer<true>(grid, buffer, layout, owned, stream);
        if (sendHost) {
            CUDA_CHECK(cudaMemcpyAsync(sendHost, buffer, count * sizeof(Real),
                                       cudaMemcpyDeviceToHost, stream));
        }
    }

    void finish(Real* grid, Layout layout, cudaStream_t stream) const {
        Real* buffer = incoming(grid, layout);
        if (recvHost) {
            CUDA_CHECK(cudaMemcpyAsync(buffer, recvHost, count * sizeof(Real),
                                       cudaMemcpyHostToDevice, stream));
        }
        if (recvDevice) transfer<false>(grid, buffer, layout, ghost, stream);
    }

    void post(bool receive, Real* grid, Layout layout, MPI_Comm comm,
              std::vector<MPI_Request>& requests) const {
        Real* buffer = receive ? recvHost : sendHost;
        if (!buffer) buffer = receive ? incoming(grid, layout) : outgoing(grid, layout);
        for (size_t offset = 0; offset < count;) {
            const int chunk = std::min<size_t>(INT_MAX, count - offset);
            requests.emplace_back();
            if (receive) {
                MPI_Irecv(buffer + offset, chunk, MPI_DOUBLE, peer, side ^ 1, comm, &requests.back());
            } else {
                MPI_Isend(buffer + offset, chunk, MPI_DOUBLE, peer, side, comm, &requests.back());
            }
            offset += chunk;
        }
    }
};

void splitWork(Tile tile, Extent global, Box& bulk, Shell& shell) {
    const bool low[] = {tile.offset.x != 0, tile.offset.y != 0, tile.offset.z != 0};
    const bool high[] = {tile.offset.x + tile.size.x != global.x,
                         tile.offset.y + tile.size.y != global.y,
                         tile.offset.z + tile.size.z != global.z};
    bulk = {{low[0] ? 1u : 2u, low[1] ? 1u : 2u, low[2] ? 1u : 2u},
            {tile.size.x + high[0], tile.size.y + high[1], tile.size.z + high[2]}};
    if (bulk.begin.x >= bulk.end.x || bulk.begin.y >= bulk.end.y || bulk.begin.z >= bulk.end.z) {
        bulk = {};
        return;
    }

    const auto axes = std::array{&Extent::x, &Extent::y, &Extent::z};
    for (int axis = 0; axis < 3; ++axis) {
        const auto coordinate = axes[axis];
        if (low[axis]) {
            Box face = bulk;
            face.end.*coordinate = ++(bulk.begin.*coordinate);
            shell.add(face);
        }
        // A tile one cell thick contributes the cell only once.
        if (high[axis] && bulk.end.*coordinate > bulk.begin.*coordinate) {
            Box face = bulk;
            face.begin.*coordinate = --(bulk.end.*coordinate);
            shell.add(face);
        }
    }
}

[[maybe_unused]] int hostThreads(size_t cells) {
    constexpr size_t cellsPerThread = 65536;
    return std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, cells / cellsPerThread));
}

void initialize(Real* host, Tile tile, Extent global) {
    const Extent local = tile.size;
    #pragma omp parallel for collapse(2) schedule(static) num_threads(hostThreads(local.cells()))
    for (size_t z = 0; z < local.z; ++z) {
        for (size_t y = 0; y < local.y; ++y) {
            const size_t globalStart =
                ((tile.offset.z + z) * global.y + tile.offset.y + y) * global.x + tile.offset.x;
            const size_t localStart = (z * local.y + y) * local.x;
            #pragma omp simd
            for (size_t x = 0; x < local.x; ++x) {
                host[localStart + x] = (globalStart + x) % 19;
            }
        }
    }
}

void copyOwned(Real* device, Real* host, Layout layout, cudaMemcpyKind direction) {
    const Extent size = layout.owned;
    const size_t width = size.x * sizeof(Real);
    auto compact = make_cudaPitchedPtr(host, width, width, size.y);
    auto padded = make_cudaPitchedPtr(device, layout.row * sizeof(Real),
                                      layout.row * sizeof(Real), size.y + 2);
    const auto origin = make_cudaPos(sizeof(Real), 1, 1);
    cudaMemcpy3DParms copy{};
    copy.extent = make_cudaExtent(width, size.y, size.z);
    copy.kind = direction;
    if (direction == cudaMemcpyHostToDevice) {
        copy.srcPtr = compact;
        copy.dstPtr = padded;
        copy.dstPos = origin;
    } else {
        copy.srcPtr = padded;
        copy.srcPos = origin;
        copy.dstPtr = compact;
    }
    CUDA_CHECK(cudaMemcpy3D(&copy));
}

double runStencil(Real*& input, Real*& output, int iterations, Layout layout, Box bulk,
                  Shell shell, const std::vector<Face>& faces, MPI_Comm comm) {
    cudaStream_t compute, exchange = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    cudaEvent_t bulkDone[2]{}, shellDone{};
    if (!faces.empty()) {
        int highPriority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(nullptr, &highPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&exchange, cudaStreamNonBlocking, highPriority));
        for (auto& event : bulkDone) CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&shellDone, cudaEventDisableTiming));
    }
    std::vector<MPI_Request> requests;
    requests.reserve(2 * faces.size());

    const Extent extent = bulk.size();
    const size_t tilesX = (extent.x + blockX - 1) / blockX;
    const size_t tilesY = (extent.y + blockY - 1) / blockY;
    const size_t tilesZ = (extent.z + zChunk - 1) / zChunk;
    const dim3 blocks(tilesX, std::min<size_t>(tilesY, 65535), std::min<size_t>(tilesZ, 65535));
    const dim3 threads(blockX, blockY);
    const size_t shellBlocks = shell.blocks();

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        const int current = iter % 2;
        // The previous bulk kernel never writes cells that are sent.
        for (const Face& face : faces) face.prepare(input, layout, exchange);

        if (iter != 0 && shellBlocks != 0) {
            // Bulk can read last step's shell, but never the incoming ghosts.
            CUDA_CHECK(cudaStreamWaitEvent(compute, shellDone, 0));
        }
        if (extent.cells() != 0) {
            stencilBulk<<<blocks, threads, 0, compute>>>(input, output, layout, bulk);
            CUDA_CHECK(cudaGetLastError());
        }

        if (exchange) {
            if (shellBlocks != 0) CUDA_CHECK(cudaEventRecord(bulkDone[current], compute));
            CUDA_CHECK(cudaStreamSynchronize(exchange));

            requests.clear();
            for (const Face& face : faces) face.post(true, input, layout, comm, requests);
            for (const Face& face : faces) face.post(false, input, layout, comm, requests);
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            for (const Face& face : faces) face.finish(input, layout, exchange);
            if (shellBlocks != 0) {
                if (iter != 0) CUDA_CHECK(cudaStreamWaitEvent(exchange, bulkDone[current ^ 1], 0));
                if (layout.owned.cells() <= UINT_MAX - linearBlock) {
                    stencilShell<unsigned><<<shellBlocks, linearBlock, 0, exchange>>>(input, output, layout, shell);
                } else {
                    stencilShell<size_t><<<shellBlocks, linearBlock, 0, exchange>>>(input, output, layout, shell);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(shellDone, exchange));
            }
        }
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    if (exchange) CUDA_CHECK(cudaStreamSynchronize(exchange));
    const double elapsed = MPI_Wtime() - start;
    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (exchange) {
        CUDA_CHECK(cudaEventDestroy(shellDone));
        for (auto event : bulkDone) CUDA_CHECK(cudaEventDestroy(event));
        CUDA_CHECK(cudaStreamDestroy(exchange));
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    return slowest;
}

void placeTile(const Real* local, Real* grid, Tile tile, Extent global) {
    #pragma omp parallel for collapse(2) schedule(static) num_threads(hostThreads(tile.size.cells()))
    for (size_t z = 0; z < tile.size.z; ++z) {
        for (size_t y = 0; y < tile.size.y; ++y) {
            const size_t start =
                ((tile.offset.z + z) * global.y + tile.offset.y + y) * global.x + tile.offset.x;
            std::copy_n(local + (z * tile.size.y + y) * tile.size.x, tile.size.x, grid + start);
        }
    }
}

void printGlobalResult(Real* local, Extent global, const std::array<int, 3>& parts,
                       int rank, int ranks, MPI_Comm comm) {
    constexpr int resultTag = 6;
    if (rank != 0) {
        const size_t cells = tileForRank(global, parts, rank).size.cells();
        for (size_t offset = 0; offset < cells;) {
            const int count = std::min<size_t>(INT_MAX, cells - offset);
            MPI_Send(local + offset, count, MPI_DOUBLE, 0, resultTag, comm);
            offset += count;
        }
        return;
    }

    std::vector<Real> grid(global.cells());
    placeTile(local, grid.data(), tileForRank(global, parts, 0), global);
    std::vector<Real> buffer;
    const bool slabs = parts[0] == 1 && parts[1] == 1;
    for (int peer = 1; peer < ranks; ++peer) {
        const Tile tile = tileForRank(global, parts, peer);
        const size_t cells = tile.size.cells();
        Real* target = grid.data() + tile.offset.z * global.x * global.y;
        if (!slabs) {
            buffer.resize(cells);
            target = buffer.data();
        }
        for (size_t offset = 0; offset < cells;) {
            const int count = std::min<size_t>(INT_MAX, cells - offset);
            MPI_Recv(target + offset, count, MPI_DOUBLE, peer, resultTag, comm, MPI_STATUS_IGNORE);
            offset += count;
        }
        if (!slabs) placeTile(target, grid.data(), tile, global);
    }
    print_results(grid, "Grid");
}

bool validateResult(const Real* local, size_t cells, int rank, MPI_Comm comm) {
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -minValue;
    int nonfinite = 0;
    #pragma omp parallel for schedule(static) num_threads(hostThreads(cells)) \
        reduction(min:minValue) reduction(max:maxValue) reduction(|:nonfinite)
    for (size_t i = 0; i < cells; ++i) {
        const Real value = local[i];
        if (!std::isfinite(value)) {
            nonfinite = 1;
        } else {
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
    }
    // One reduction suffices for the nonfinite flag and both bounds.
    const Real localBounds[] = {-minValue, maxValue, Real(nonfinite)};
    Real bounds[3]{};
    MPI_Allreduce(localBounds, bounds, 3, MPI_DOUBLE, MPI_MAX, comm);
    const bool valid = bounds[2] == 0.0 && bounds[0] <= 1e6 && bounds[1] <= 1e6;
    if (rank == 0) {
        if (bounds[2] != 0.0) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Value range: [%.6f, %.6f]\n", -bounds[0], bounds[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseNumber(const char* value, size_t& result) {
    if (*value < '0' || *value > '9') return false;
    char* end = nullptr;
    errno = 0;
    const auto number = std::strtoull(value, &end, 10);
    if (errno != 0 || *end != '\0' || number > std::numeric_limits<size_t>::max()) return false;
    result = number;
    return true;
}

bool fitsAllocation(Extent global, Extent local) {
    constexpr size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (global.x == 0 || global.y == 0 || global.z == 0 ||
        global.x > maxCells / global.y / global.z) return false;
    // The CUDA layout includes one ghost cell on both ends of every axis.
    return local.x + 2 <= maxCells / (local.y + 2) / (local.z + 2);
}

int benchmark(int argc, char** argv, int rank, int worldSize) {
    Extent global{128, 0, 0};
    size_t iterations = 10;
    bool validate = false;
    bool printResult = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResult = true;
        } else {
            const char* option = argv[i];
            size_t* value = nullptr;
            if (strcmp(option, "-x") == 0) value = &global.x;
            if (strcmp(option, "-y") == 0) value = &global.y;
            if (strcmp(option, "-z") == 0) value = &global.z;
            if (strcmp(option, "-i") == 0) value = &iterations;
            if (!value || i + 1 == argc || !parseNumber(argv[++i], *value)) {
                if (rank == 0) {
                    printf("Invalid option or value: %s\n", option);
                    printUsage(argv[0]);
                }
                return 1;
            }
        }
    }
    if (global.y == 0) global.y = global.x;
    if (global.z == 0) global.z = global.x;
    if (!fitsAllocation(global, {1, 1, 1}) || iterations > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Grid size or iteration count is out of range\n");
        return 1;
    }

    const auto parts = chooseDecomposition(global, worldSize);
    const int active = parts[0] * parts[1] * parts[2];
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) return 0;

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank = 0;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_free(&node);
    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    if (gpuCount == 0) {
        if (nodeRank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % gpuCount));

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", global.x, global.y, global.z);
        printf("Iterations: %zu\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const Tile tile = tileForRank(global, parts, rank);
    if (!fitsAllocation(global, tile.size)) {
        fprintf(stderr, "Rank %d: local grid size is out of range\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    const Layout layout(tile.size);
    const size_t cells = tile.size.cells();
    const bool hasInterior = global.x > 2 && global.y > 2 && global.z > 2;
    Real* input = nullptr;
    Real* output = nullptr;
    Real* host = nullptr;
    CUDA_CHECK(cudaMalloc(&input, layout.bytes()));
    CUDA_CHECK(cudaMalloc(&output, layout.bytes()));
    CUDA_CHECK(cudaMallocHost(&host, cells * sizeof(Real)));
    initialize(host, tile, global);
    CUDA_CHECK(cudaMemset(input, 0, layout.bytes()));
    copyOwned(input, host, layout, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMemcpy(output, input, layout.bytes(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    if (!printResult && !validate) {
        CUDA_CHECK(cudaFreeHost(host));
        host = nullptr;
    }

    std::vector<Face> faces;
    faces.reserve(6);
    Box bulk{};
    Shell shell{};
    if (hasInterior && iterations != 0) {
        splitWork(tile, global, bulk, shell);
        const int strides[] = {1, parts[0], parts[0] * parts[1]};
        const bool direct = cudaAwareMPI();
        for (int axis = 0; axis < 3; ++axis) {
            const int coordinate = (rank / strides[axis]) % parts[axis];
            if (coordinate != 0) faces.emplace_back(2 * axis, rank - strides[axis], layout, direct);
            if (coordinate + 1 < parts[axis]) {
                faces.emplace_back(2 * axis + 1, rank + strides[axis], layout, direct);
            }
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    const double elapsed =
        runStencil(input, output, hasInterior ? int(iterations) : 0, layout, bulk, shell, faces, comm);

    if (rank == 0) {
        const double updates = hasInterior
            ? double(global.x - 2) * (global.y - 2) * (global.z - 2) * iterations : 0.0;
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
    }

    if (host) copyOwned(input, host, layout, cudaMemcpyDeviceToHost);
    for (const Face& face : faces) face.release();
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));

    if (printResult) printGlobalResult(host, global, parts, rank, active, comm);
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(host, cells, rank, comm);
    }
    if (host) CUDA_CHECK(cudaFreeHost(host));
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const int result = benchmark(argc, argv, rank, ranks);
    int status = 0;
    MPI_Allreduce(&result, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
