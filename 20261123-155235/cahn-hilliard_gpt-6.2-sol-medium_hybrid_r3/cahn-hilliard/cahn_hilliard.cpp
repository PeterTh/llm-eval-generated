#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

using Dimensions = std::array<int, 3>;
constexpr int faceCount = 6;
constexpr int linearThreads = 128;
constexpr auto maxUnsignedIndex = std::numeric_limits<unsigned>::max();

size_t volume(const Dimensions& size) {
    return size_t(size[0]) * size[1] * size[2];
}

Dimensions processGrid(const Dimensions& size, int ranks) {
    Dimensions best{1, 1, 1};
    int active = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    // Prefer a balanced volume and low total interface area. Z faces are
    // contiguous and Y faces retain contiguous rows when packing.
    for (int px = 1; px <= std::min(size[0], ranks); ++px) {
        for (int py = 1; py <= std::min(size[1], ranks / px); ++py) {
            const int pz = std::min(size[2], ranks / px / py);
            const int count = px * py * pz;
            const Dimensions largest{1 + (size[0] - 1) / px,
                                      1 + (size[1] - 1) / py,
                                      1 + (size[2] - 1) / pz};
            const double area = 1.5 * (px - 1) * size[1] * size[2]
                              + 1.2 * (py - 1) * size[0] * size[2]
                              + 1.0 * (pz - 1) * size[0] * size[1];
            const double cost = (1.0 + area) * volume(largest);
            if (count > active || (count == active && cost < bestCost)) {
                best = {px, py, pz};
                active = count;
                bestCost = cost;
            }
        }
    }
    return best;
}

struct Box {
    Dimensions origin;
    Dimensions size;
};

Box rankBox(const Dimensions& global, const Dimensions& parts, const Dimensions& coords) {
    Box box{};
    for (int axis = 0; axis < 3; ++axis) {
        const int base = global[axis] / parts[axis];
        const int extra = global[axis] % parts[axis];
        box.origin[axis] = coords[axis] * base + std::min(coords[axis], extra);
        box.size[axis] = base + (coords[axis] < extra);
    }
    return box;
}

struct Region {
    int x, y, z;
    int nx, ny, nz;

    size_t cells() const { return size_t(nx) * ny * nz; }
};

struct Boundaries {
    Region regions[faceCount]{};
    int count = 0;
    size_t maxCells = 0;

    void split(Region& interior, int face) {
        int* extent[] = {&interior.nx, &interior.ny, &interior.nz};
        int* origin[] = {&interior.x, &interior.y, &interior.z};
        const int axis = face / 2;
        if (interior.cells() == 0) return;
        const bool high = face % 2;
        const int coordinate = *origin[axis] + (high ? *extent[axis] - 1 : 0);
        Region region = interior;
        switch (axis) {
            case 0: region.x = coordinate; region.nx = 1; break;
            case 1: region.y = coordinate; region.ny = 1; break;
            case 2: region.z = coordinate; region.nz = 1; break;
        }
        --*extent[axis];
        if (!high) ++*origin[axis];
        regions[count++] = region;
        maxCells = std::max(maxCells, region.cells());
    }
};

struct Mesh {
    int nx, ny, nz;
    size_t plane;
};

struct HaloPointers {
    const double* face[faceCount]{};
};

struct PackPointers {
    double* face[4]{};
};

__device__ __forceinline__ size_t cellIndex(Mesh mesh, int x, int y, int z) {
    return size_t(z) * mesh.plane + size_t(y) * mesh.nx + x;
}

template <bool HasHalos>
__device__ __forceinline__ double laplacian(const double* field, HaloPointers halos,
                                           Mesh mesh, int x, int y, int z, size_t i) {
    const double center = field[i];
    const bool west = x > 0;
    const bool east = x + 1 < mesh.nx;
    const bool south = y > 0;
    const bool north = y + 1 < mesh.ny;
    const bool bottom = z > 0;
    const bool top = z + 1 < mesh.nz;
    double xm = field[west ? i - 1 : i];
    double xp = field[east ? i + 1 : i];
    double ym = field[south ? i - mesh.nx : i];
    double yp = field[north ? i + mesh.nx : i];
    double zm = field[bottom ? i - mesh.plane : i];
    double zp = field[top ? i + mesh.plane : i];

    if constexpr (HasHalos) {
        const size_t xFace = size_t(z) * mesh.ny + y;
        const size_t yFace = size_t(z) * mesh.nx + x;
        const size_t zFace = size_t(y) * mesh.nx + x;
        if (!west && halos.face[0]) xm = halos.face[0][xFace];
        if (!east && halos.face[1]) xp = halos.face[1][xFace];
        if (!south && halos.face[2]) ym = halos.face[2][yFace];
        if (!north && halos.face[3]) yp = halos.face[3][yFace];
        if (!bottom && halos.face[4]) zm = halos.face[4][zFace];
        if (!top && halos.face[5]) zp = halos.face[5][zFace];
    }
    // All three grid spacings are one.
    const double twice = 2.0 * center;
    return ((xp + xm - twice) + (yp + ym - twice)) + (zp + zm - twice);
}

template <bool Chemical, bool HasHalos>
__device__ __forceinline__ void applyStencil(const double* input, const double* cold,
                                              double* output, HaloPointers halos, Mesh mesh,
                                              int x, int y, int z) {
    const size_t i = cellIndex(mesh, x, y, z);
    const double lap = laplacian<HasHalos>(input, halos, mesh, x, y, z, i);
    if constexpr (Chemical) {
        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = 2.0 / 9.0;
        const double c = input[i];
        // Keep the original arithmetic and use explicit FP64 contractions so
        // interior and halo paths produce identical values.
        const double pairEnergy = fma(c + 1.0, eAA, (c - 1.0) * eBB);
        const double energy = fma(-2.0 * c, eAB, pairEnergy);
        const double linear = fma(4.5, energy, 3.0 * c);
        output[i] = fma(-0.5, lap, fma(c * c, c, linear));
    } else {
        constexpr double dt = 0.01;
        constexpr double mobility = 1.0;
        output[i] = fma(dt * mobility, lap, cold[i]);
    }
}

template <bool Chemical>
__global__ void stencilInterior(const double* __restrict__ input, const double* __restrict__ cold,
                                double* __restrict__ output,
                                Mesh mesh, Region region) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= region.nx) return;
    for (size_t z = blockIdx.z; z < size_t(region.nz); z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < size_t(region.ny);
             y += gridDim.y * blockDim.y) {
            applyStencil<Chemical, false>(input, cold, output, {}, mesh,
                                         region.x + x, region.y + int(y), region.z + int(z));
        }
    }
}

template <typename Index>
__device__ __forceinline__ void regionCoordinates(Index i, Region region, int& x, int& y, int& z) {
    const Index row = i / Index(region.nx);
    x = region.x + int(i % Index(region.nx));
    y = region.y + int(row % Index(region.ny));
    z = region.z + int(row / Index(region.ny));
}

template <bool Chemical>
__global__ void stencilBoundary(const double* __restrict__ input, const double* __restrict__ cold,
                                double* __restrict__ output,
                                Mesh mesh, HaloPointers halos, Boundaries boundaries) {
    const Region region = boundaries.regions[blockIdx.y];
    const size_t cells = size_t(region.nx) * region.ny * region.nz;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
         i += size_t(gridDim.x) * blockDim.x) {
        int x, y, z;
        if (cells <= maxUnsignedIndex) {
            regionCoordinates(unsigned(i), region, x, y, z);
        } else {
            regionCoordinates(i, region, x, y, z);
        }
        applyStencil<Chemical, true>(input, cold, output, halos, mesh, x, y, z);
    }
}

__global__ void packFaces(const double* field, Mesh mesh, PackPointers packed) {
    const int face = blockIdx.y;
    if (!packed.face[face]) return;
    const bool xFace = face < 2;
    const int width = xFace ? mesh.ny : mesh.nx;
    const size_t cells = size_t(width) * mesh.nz;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
         i += size_t(gridDim.x) * blockDim.x) {
        const int along = cells <= maxUnsignedIndex
                          ? unsigned(i) % unsigned(width) : i % width;
        const int z = cells <= maxUnsignedIndex
                      ? unsigned(i) / unsigned(width) : i / width;
        const int x = xFace ? (face % 2 ? mesh.nx - 1 : 0) : along;
        const int y = xFace ? along : (face % 2 ? mesh.ny - 1 : 0);
        packed.face[face][i] = field[cellIndex(mesh, x, y, z)];
    }
}

unsigned linearBlocks(size_t cells) {
    return unsigned(std::min<size_t>(1 + (cells - 1) / linearThreads, 2147483647));
}

bool cudaAwareMpi() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

double* allocateDevice(size_t cells) {
    if (!cells) return nullptr;
    double* buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer, cells * sizeof(double)));
    return buffer;
}

double* allocatePinned(size_t cells) {
    if (!cells) return nullptr;
    double* buffer = nullptr;
    CUDA_CHECK(cudaMallocHost(&buffer, cells * sizeof(double)));
    return buffer;
}

struct Face {
    int neighbor = MPI_PROC_NULL;
    size_t cells = 0;
    size_t offset = 0;
};

class Evolution {
public:
    Evolution(MPI_Comm comm, const Dimensions& size)
        : comm_(comm), mesh_{size[0], size[1], size[2], size_t(size[0]) * size[1]},
          interior_{0, 0, 0, size[0], size[1], size[2]} {
        size_t haloCells = 0;
        size_t packCells = 0;
        size_t maxPack = 0;
        for (int axis = 0; axis < 3; ++axis) {
            MPI_Cart_shift(comm_, axis, 1, &faces_[2 * axis].neighbor, &faces_[2 * axis + 1].neighbor);
        }
        // Peel Z, Y, then X faces so edge and corner cells are updated once.
        // Physical faces stay in the interior region and clamp locally.
        for (int i : {4, 5, 2, 3, 0, 1}) {
            Face& face = faces_[i];
            if (face.neighbor == MPI_PROC_NULL) continue;
            face.cells = volume(size) / size[i / 2];
            face.offset = haloCells;
            haloCells += face.cells;
            if (i < 4) {
                packCells += face.cells;
                maxPack = std::max(maxPack, face.cells);
            }
            boundaries_.split(interior_, i);
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_, cudaStreamNonBlocking));
        if (interior_.cells()) {
            block_ = dim3(size[0] < 32 ? 8 : 32, size[0] < 32 ? 16 : 4);
            // Smaller tiles preserve occupancy on very narrow grids.
            if (size[0] < 8) {
                const unsigned width = size[0] >= 4 ? 4 : (size[0] >= 2 ? 2 : 1);
                block_ = dim3(width, 128 / width);
            }
            grid_ = dim3(1 + (interior_.nx - 1) / block_.x,
                         std::min(65535u, 1 + (interior_.ny - 1) / block_.y),
                         std::min(65535, interior_.nz));
        }
        if (!haloCells) return;
        haloBytes_ = haloCells * sizeof(double);

        direct_ = cudaAwareMpi();
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&interiorDone_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&boundaryDone_, cudaEventDisableTiming));
        boundaryGrid_ = dim3(linearBlocks(boundaries_.maxCells), boundaries_.count);
        deviceSend_ = allocateDevice(packCells);
        if (maxPack) packGrid_ = dim3(linearBlocks(maxPack), 4);
        size_t packOffset = 0;
        for (int i = 0; i < 4; ++i) {
            if (!faces_[i].cells) continue;
            pack_.face[i] = deviceSend_ + packOffset;
            packOffset += faces_[i].cells;
        }
        // Alternate receive buffers so MPI can post the next exchange while
        // the previous boundary kernel or host-to-device copy reads its halo.
        for (int slot = 0; slot < 2; ++slot) {
            deviceRecv_[slot] = allocateDevice(haloCells);
            if (!direct_) hostRecv_[slot] = allocatePinned(haloCells);
            for (int i = 0; i < faceCount; ++i) {
                if (faces_[i].cells) halos_[slot].face[i] = deviceRecv_[slot] + faces_[i].offset;
            }
        }
        if (!direct_) hostSend_ = allocatePinned(haloCells);

        size_t messages = 0;
        for (const Face& face : faces_) {
            if (face.cells) messages += 1 + (face.cells - 1) / maxMessage;
        }
        requests_.reserve(2 * messages);
    }

    Evolution(const Evolution&) = delete;
    Evolution& operator=(const Evolution&) = delete;

    ~Evolution() {
        if (exchange_) {
            CUDA_CHECK(cudaEventDestroy(interiorDone_));
            CUDA_CHECK(cudaEventDestroy(boundaryDone_));
            CUDA_CHECK(cudaStreamDestroy(exchange_));
            if (deviceSend_) CUDA_CHECK(cudaFree(deviceSend_));
            for (int slot = 0; slot < 2; ++slot) {
                CUDA_CHECK(cudaFree(deviceRecv_[slot]));
                if (hostRecv_[slot]) CUDA_CHECK(cudaFreeHost(hostRecv_[slot]));
            }
            if (hostSend_) CUDA_CHECK(cudaFreeHost(hostSend_));
        }
        CUDA_CHECK(cudaStreamDestroy(compute_));
    }

    void run(double*& cold, double*& next, double* mu, int steps) {
        for (int step = 0; step < steps; ++step) {
            stencil<true>(cold, nullptr, mu, 0);
            stencil<false>(mu, cold, next, 1);
            std::swap(cold, next);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_));
        if (exchange_) CUDA_CHECK(cudaStreamSynchronize(exchange_));
    }

private:
    static constexpr size_t maxMessage = std::numeric_limits<int>::max();

    template <bool Receive, typename Pointer>
    void post(Pointer buffer, const Face& face, int tag) {
        for (size_t offset = 0; offset < face.cells; offset += maxMessage) {
            const int count = int(std::min(maxMessage, face.cells - offset));
            requests_.push_back(MPI_REQUEST_NULL);
            if constexpr (Receive) {
                MPI_Irecv(buffer + offset, count, MPI_DOUBLE, face.neighbor, tag, comm_, &requests_.back());
            } else {
                MPI_Isend(buffer + offset, count, MPI_DOUBLE, face.neighbor, tag, comm_, &requests_.back());
            }
        }
    }

    template <bool Chemical>
    void stencil(const double* input, const double* cold, double* output, int slot) {
        double* recv = direct_ ? deviceRecv_[slot] : hostRecv_[slot];
        if (exchange_) {
            requests_.clear();
            for (int i = 0; i < faceCount; ++i) {
                if (faces_[i].cells) post<true>(recv + faces_[i].offset, faces_[i], i ^ 1);
            }
        }
        if (interior_.cells()) {
            stencilInterior<Chemical><<<grid_, block_, 0, compute_>>>(input, cold, output, mesh_, interior_);
            CUDA_CHECK(cudaGetLastError());
        }
        if (!exchange_) return;

        if (deviceSend_) {
            packFaces<<<packGrid_, linearThreads, 0, exchange_>>>(input, mesh_, pack_);
            CUDA_CHECK(cudaGetLastError());
        }
        const double* send[faceCount]{pack_.face[0], pack_.face[1], pack_.face[2], pack_.face[3],
                                      input, input + size_t(mesh_.nz - 1) * mesh_.plane};
        if (!direct_) {
            for (int i = 0; i < faceCount; ++i) {
                const Face& face = faces_[i];
                if (!face.cells) continue;
                double* host = hostSend_ + face.offset;
                CUDA_CHECK(cudaMemcpyAsync(host, send[i], face.cells * sizeof(double),
                                           cudaMemcpyDeviceToHost, exchange_));
                send[i] = host;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(exchange_));
        for (int i = 0; i < faceCount; ++i) {
            if (faces_[i].cells) post<false>(send[i], faces_[i], i);
        }
        MPI_Waitall(int(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
        if (!direct_) {
            CUDA_CHECK(cudaMemcpyAsync(deviceRecv_[slot], hostRecv_[slot], haloBytes_,
                                       cudaMemcpyHostToDevice, exchange_));
        }

        // Boundary cells depend on the previous pass's interior, and the next
        // interior depends on this pass's boundary. Packing needs only faces.
        CUDA_CHECK(cudaStreamWaitEvent(exchange_, interiorDone_, 0));
        CUDA_CHECK(cudaEventRecord(interiorDone_, compute_));
        stencilBoundary<Chemical><<<boundaryGrid_, linearThreads, 0, exchange_>>>(
            input, cold, output, mesh_, halos_[slot], boundaries_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(boundaryDone_, exchange_));
        CUDA_CHECK(cudaStreamWaitEvent(compute_, boundaryDone_, 0));
    }

    MPI_Comm comm_;
    Mesh mesh_;
    Region interior_;
    Boundaries boundaries_;
    Face faces_[faceCount];
    PackPointers pack_;
    HaloPointers halos_[2];
    size_t haloBytes_ = 0;
    bool direct_ = false;
    double* deviceSend_ = nullptr;
    double* deviceRecv_[2]{};
    double* hostSend_ = nullptr;
    double* hostRecv_[2]{};
    cudaStream_t compute_ = nullptr;
    cudaStream_t exchange_ = nullptr;
    cudaEvent_t interiorDone_ = nullptr;
    cudaEvent_t boundaryDone_ = nullptr;
    dim3 block_, grid_, boundaryGrid_, packGrid_;
    std::vector<MPI_Request> requests_;
};

int hostThreads(size_t cells) {
    const size_t useful = std::min<size_t>(16, 1 + (cells - 1) / 16384);
    return std::min(int(useful), omp_get_max_threads());
}

void initialize(double* field, const Dimensions& global, const Box& box, size_t cells) {
    const size_t nx = global[0];
    const size_t plane = nx * global[1];
    const size_t globalCells = volume(global);
    const int localX = box.size[0];
    const size_t rows = cells / localX;
    [[maybe_unused]] const int threads = hostThreads(cells);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t row = 0; row < rows; ++row) {
        const size_t z = box.origin[2] + row / box.size[1];
        const size_t y = box.origin[1] + row % box.size[1];
        const size_t start = z * plane + y * nx + box.origin[0];
        for (int x = 0; x < localX; ++x) {
            const size_t pseudo = ((start + x + 1) * 1299709) % globalCells;
            field[row * localX + x] = -1.0 + 2.0 * (pseudo / double(globalCells));
        }
    }
}

MPI_Datatype subarray(const Dimensions& global, const Box& box) {
    const int sizes[]{global[2], global[1], global[0]};
    const int subsizes[]{box.size[2], box.size[1], box.size[0]};
    const int starts[]{box.origin[2], box.origin[1], box.origin[0]};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

void reportResults(MPI_Comm comm, int rank, int ranks, const Dimensions& global,
                   const Dimensions& parts, const Box& box, const std::vector<double>& local) {
    if (ranks == 1) {
        print_results(local, "Concentration");
        return;
    }
    if (rank != 0) {
        if (local.size() <= size_t(std::numeric_limits<int>::max())) {
            MPI_Send(local.data(), int(local.size()), MPI_DOUBLE, 0, 0, comm);
        } else {
            MPI_Datatype type = subarray(box.size, {{0, 0, 0}, box.size});
            MPI_Send(local.data(), 1, type, 0, 0, comm);
            MPI_Type_free(&type);
        }
        return;
    }

    std::vector<double> result(volume(global));
    std::vector<MPI_Request> requests(ranks - 1);
    for (int source = 1; source < ranks; ++source) {
        Dimensions coords;
        MPI_Cart_coords(comm, source, 3, coords.data());
        MPI_Datatype type = subarray(global, rankBox(global, parts, coords));
        MPI_Irecv(result.data(), 1, type, source, 0, comm, &requests[source - 1]);
        MPI_Type_free(&type);
    }
    const size_t rows = local.size() / box.size[0];
    [[maybe_unused]] const int threads = hostThreads(local.size());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t row = 0; row < rows; ++row) {
        const size_t z = box.origin[2] + row / box.size[1];
        const size_t y = box.origin[1] + row % box.size[1];
        const size_t offset = (z * global[1] + y) * global[0] + box.origin[0];
        std::copy_n(local.data() + row * box.size[0], box.size[0], result.data() + offset);
    }
    MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE);
    print_results(result, "Concentration");
}

bool validateResult(MPI_Comm comm, int rank, const std::vector<double>& local) {
    int invalid = 0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    [[maybe_unused]] const int threads = hostThreads(local.size());
    #pragma omp parallel for simd schedule(static) num_threads(threads) \
        reduction(|:invalid) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < local.size(); ++i) {
        const double value = local[i];
        invalid |= !std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }

    int anyInvalid;
    MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_BOR, comm);
    if (anyInvalid) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    const double limits[]{minVal, -maxVal};
    double globalLimits[2];
    MPI_Allreduce(limits, globalLimits, 2, MPI_DOUBLE, MPI_MIN, comm);
    const double globalMin = globalLimits[0];
    const double globalMax = -globalLimits[1];
    const bool valid = globalMin >= -10.0 && globalMax <= 10.0;
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (!valid) printf("Validation failed: values out of expected range\n");
    }
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    Dimensions global{64, 0, 0};
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            global[0] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            global[1] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            global[2] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    size_t gridSize = 1;
    for (int extent : global) {
        if (extent < 1 || size_t(extent) > std::numeric_limits<size_t>::max() / sizeof(double) / gridSize) {
            if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
            return 1;
        }
        gridSize *= extent;
    }

    const Dimensions parts = processGrid(global, ranks);
    const int activeRanks = int(volume(parts));
    const int periods[]{0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods, 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;

    MPI_Comm node;
    MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, deviceCount;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    Dimensions coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    const Box box = rankBox(global, parts, coords);
    const size_t cells = volume(box.size);
    const size_t bytes = cells * sizeof(double);
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    double* cold = allocateDevice(cells);
    double* next = allocateDevice(cells);
    double* mu = allocateDevice(cells);
    {
        std::unique_ptr<double[]> initial(new double[cells]);
        initialize(initial.get(), global, box, cells);
        CUDA_CHECK(cudaMemcpy(cold, initial.get(), bytes, cudaMemcpyHostToDevice));
    }

    {
        Evolution evolution(cart, box.size);
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        evolution.run(cold, next, mu, iterations);
        const double elapsed = MPI_Wtime() - start;
        double duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", long(duration * 1000.0));
            const double updates = double(gridSize) * std::max(iterations, 0);
            printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0.0);
        }
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<double> local(cells);
        CUDA_CHECK(cudaMemcpy(local.data(), cold, bytes, cudaMemcpyDeviceToHost));
        if (printResults) reportResults(cart, rank, activeRanks, global, parts, box, local);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const bool valid = validateResult(cart, rank, local);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(cold));
    MPI_Comm_free(&cart);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int status = 1;
    try {
        status = benchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
