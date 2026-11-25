#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
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
#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using Real = double;
using Shape = std::array<int, 3>;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA(call) checkCuda((call), #call)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) {
        if (size != 0) CUDA(cudaMalloc(&data, size * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data) CUDA(cudaFree(data));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

struct Box {
    Shape start;
    Shape size;

    size_t count() const {
        return size_t(size[0]) * size[1] * size[2];
    }
};

Box partition(const Shape& extent, const Shape& parts, const Shape& coord, int border) {
    Box box{};
    for (int axis = 0; axis < 3; ++axis) {
        const int quotient = extent[axis] / parts[axis];
        const int remainder = extent[axis] % parts[axis];
        box.start[axis] = border + coord[axis] * quotient + std::min(coord[axis], remainder);
        box.size[axis] = quotient + (coord[axis] < remainder);
    }
    return box;
}

Box includeBoundary(Box box, const Shape& parts, const Shape& coord, int border) {
    for (int axis = 0; axis < 3; ++axis) {
        if (coord[axis] == 0) {
            box.start[axis] -= border;
            box.size[axis] += border;
        }
        if (coord[axis] == parts[axis] - 1) box.size[axis] += border;
    }
    return box;
}

Shape chooseDecomposition(const Shape& extent, int ranks) {
    Shape best{1, 1, 1};
    int used = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= std::min(extent[0], ranks); ++x) {
        for (int y = 1; y <= std::min(extent[1], ranks / x); ++y) {
            const int z = std::min(extent[2], ranks / x / y);
            const int count = x * y * z;
            if (count < used) continue;

            // Balance the largest local volume and its communication surface.
            // Faces perpendicular to X have strided memory access.
            const auto a = double(extent[0] / x + (extent[0] % x != 0));
            const auto b = double(extent[1] / y + (extent[1] % y != 0));
            const auto c = double(extent[2] / z + (extent[2] % z != 0));
            const double surface = 1.5 * (x > 1) * b * c + (y > 1) * a * c + (z > 1) * a * b;
            const double cost = a * b * c + 4.0 * surface;
            if (count > used || cost < bestCost) {
                best = {x, y, z};
                used = count;
                bestCost = cost;
            }
        }
    }
    return best;
}

int hostThreads(size_t elements) {
    constexpr size_t elementsPerThread = 65536;
    return int(std::min(size_t(omp_get_max_threads()), 1 + elements / elementsPerThread));
}

void initializeGrid(Real* grid, const Box& box, const Shape& global, int border,
                    size_t pitch, size_t slice, size_t count) {
    // Interior blocks include a one-cell halo. Its initial values are known even
    // across rank boundaries, so the first iteration needs no input exchange.
    const size_t origin = (size_t(box.start[2] - border) * global[1] +
                           (box.start[1] - border)) * global[0] + (box.start[0] - border);
    const size_t rows = count / pitch;
    #pragma omp parallel for schedule(static) num_threads(hostThreads(count))
    for (size_t row = 0; row < rows; ++row) {
        const size_t z = row / (slice / pitch);
        const size_t y = row % (slice / pitch);
        const size_t globalIndex = origin + (z * global[1] + y) * global[0];
        Real* destination = grid + row * pitch;
        for (size_t x = 0; x < pitch; ++x) destination[x] = Real((globalIndex + x) % 19);
    }
}

__device__ __noinline__ Real divideSmall(Real sum) {
    return sum / 7.0;
}

__device__ __forceinline__ Real average(Real center, const Real* input, size_t i,
                                       size_t pitch, Real below, Real above) {
    const Real sum = center + input[i - 1] + input[i + 1] + input[i - pitch] +
                     input[i + pitch] + below + above;
    if (sum < 0x1p-970 && sum != 0.0) return divideSmall(sum);

    // Refine the reciprocal product with its exact remainder. The quotient
    // is at least 1/14 ulp from a rounding midpoint, so the error in the
    // reciprocal used for the correction cannot change the rounded result.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real q = sum * reciprocal;
    const Real remainder = __fma_rn(-7.0, q, sum);
    return __fma_rn(remainder, reciprocal, q);
}

constexpr int tileX = 32;
constexpr int tileY = 4;
constexpr int tileZ = 4;

__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t pitch, size_t slice, int x0, int y0, int z0,
                        unsigned nx, unsigned ny, unsigned nz) {
    const unsigned x = blockIdx.x * tileX + threadIdx.x;
    if (x >= nx) return;
    for (unsigned z = blockIdx.z * tileZ; z < nz; z += gridDim.z * tileZ) {
        const unsigned depth = min(unsigned(tileZ), nz - z);
        for (unsigned y = blockIdx.y * tileY + threadIdx.y; y < ny; y += gridDim.y * tileY) {
            size_t i = size_t(z + z0) * slice + size_t(y + y0) * pitch + x + x0;
            Real below = input[i - slice];
            Real center = input[i];
            #pragma unroll
            for (unsigned step = 0; step < tileZ; ++step) {
                if (step >= depth) break;
                const Real above = input[i + slice];
                output[i] = average(center, input, i, pitch, below, above);
                below = center;
                center = above;
                i += slice;
            }
        }
    }
}

void launchStencil(const Real* input, Real* output, size_t pitch, size_t slice,
                   const Shape& first, const Shape& last, cudaStream_t stream) {
    const int nx = last[0] - first[0];
    const int ny = last[1] - first[1];
    const int nz = last[2] - first[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const dim3 blocks((unsigned(nx) + tileX - 1) / tileX,
                      std::min((unsigned(ny) + tileY - 1) / tileY, 65535u),
                      std::min((unsigned(nz) + tileZ - 1) / tileZ, 65535u));
    stencil<<<blocks, dim3(tileX, tileY), 0, stream>>>(input, output, pitch, slice,
        first[0], first[1], first[2], nx, ny, nz);
    CUDA(cudaGetLastError());
}

struct Face {
    size_t count;
    size_t offset;
    size_t start;
    size_t halo;
    size_t strideU;
    size_t strideV;
    int width;
    unsigned endBlock;
};

struct Faces {
    Face face[6];
    int nx;
    int ny;
};

constexpr int faceThreads = 128;

template <bool compute>
__global__ void transferFaces(const Real* __restrict__ input, Real* __restrict__ output,
                              Real* __restrict__ packed, Faces faces, size_t pitch, size_t slice) {
    int side = 0;
    while (blockIdx.x >= faces.face[side].endBlock) ++side;
    const Face face = faces.face[side];
    const int axis = side / 2;
    const unsigned startBlock = side == 0 ? 0 : faces.face[side - 1].endBlock;
    const size_t step = size_t(face.endBlock - startBlock) * blockDim.x;
    for (size_t j = size_t(blockIdx.x - startBlock) * blockDim.x + threadIdx.x;
         j < face.count; j += step) {
        const size_t v = j <= UINT_MAX ? unsigned(j) / unsigned(face.width) : j / face.width;
        const size_t u = j - v * face.width;
        const size_t offset = u * face.strideU + v * face.strideV;
        if constexpr (compute) {
            const size_t i = face.start + offset;
            const Real value = average(input[i], input, i, pitch, input[i - slice], input[i + slice]);
            packed[face.offset + j] = value;

            // A cell on multiple faces is packed for every neighbor. Only
            // its first face writes to the output grid.
            const bool onX = axis > 0 && ((u == 0 && faces.face[0].count != 0) ||
                             (u == size_t(faces.nx - 1) && faces.face[1].count != 0));
            const bool onY = axis > 1 && ((v == 0 && faces.face[2].count != 0) ||
                             (v == size_t(faces.ny - 1) && faces.face[3].count != 0));
            const bool duplicateSide = (side % 2 != 0) && faces.face[side - 1].count != 0 &&
                                       face.start == faces.face[side - 1].start;
            if (!onX && !onY && !duplicateSide) output[i] = value;
        } else {
            output[face.halo + offset] = input[face.offset + j];
        }
    }
}

class HaloExchange {
public:
    HaloExchange(const Box& box, size_t pitch, size_t slice, MPI_Comm comm, bool enabled)
        : comm(comm), faces(enabled ? makeFaces(box, pitch, slice) : Faces{}),
          sendDevice(total), receiveDevice(2 * total) {
        if (total == 0) return;
        int direct = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        direct = MPIX_Query_cuda_support();
#endif
        int allDirect = 0;
        MPI_Allreduce(&direct, &allDirect, 1, MPI_INT, MPI_MIN, comm);
        if (!allDirect) {
            CUDA(cudaMallocHost(&sendHost, total * sizeof(Real)));
            CUDA(cudaMallocHost(&receiveHost, 2 * total * sizeof(Real)));
        }
        int highestPriority = 0;
        CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &highestPriority));
        CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, highestPriority));
        CUDA(cudaEventCreateWithFlags(&bulkReady, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
        cudaFuncAttributes attributes{};
        CUDA(cudaFuncGetAttributes(&attributes, transferFaces<true>));
        CUDA(cudaFuncGetAttributes(&attributes, transferFaces<false>));

        size_t requestCount = 0;
        for (const Face& face : faces.face) {
            requestCount += 2 * ((face.count + INT_MAX - 1) / INT_MAX);
        }
        requests.reserve(requestCount);
    }

    ~HaloExchange() {
        if (stream) {
            CUDA(cudaStreamDestroy(stream));
            CUDA(cudaEventDestroy(bulkReady));
            CUDA(cudaEventDestroy(haloReady));
        }
        if (sendHost) CUDA(cudaFreeHost(sendHost));
        if (receiveHost) CUDA(cudaFreeHost(receiveHost));
    }

    void start(const Real* input, Real* output, size_t pitch, size_t slice,
               cudaStream_t compute, bool previousExchange) {
        if (previousExchange) {
            CUDA(cudaStreamWaitEvent(stream, bulkReady, 0));
            waitForHalo(compute);
            receiveOffset = total - receiveOffset;
        }
        requests.clear();
        // MPI may write the next receive buffer while CUDA consumes the previous one.
        postMessages(false, (receiveHost ? receiveHost : receiveDevice.data) + receiveOffset);
        transferFaces<true><<<blocks, faceThreads, 0, stream>>>(
            input, output, sendDevice.data, faces, pitch, slice);
        CUDA(cudaGetLastError());
        if (sendHost) {
            CUDA(cudaMemcpyAsync(sendHost, sendDevice.data, total * sizeof(Real),
                                 cudaMemcpyDeviceToHost, stream));
        }
    }

    void finish(Real* output, cudaStream_t compute) {
        CUDA(cudaEventRecord(bulkReady, compute));
        CUDA(cudaStreamSynchronize(stream));

        postMessages(true, sendHost ? sendHost : sendDevice.data);
        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        if (receiveHost) {
            CUDA(cudaMemcpyAsync(receiveDevice.data + receiveOffset, receiveHost + receiveOffset,
                                 total * sizeof(Real),
                                 cudaMemcpyHostToDevice, stream));
        }
        transferFaces<false><<<blocks, faceThreads, 0, stream>>>(
            receiveDevice.data + receiveOffset, output, nullptr, faces, 0, 0);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(haloReady, stream));
    }

    bool active() const {
        return total != 0;
    }

    bool hasNeighbor(int side) const {
        return faces.face[side].count != 0;
    }

    void waitForHalo(cudaStream_t compute) const {
        CUDA(cudaStreamWaitEvent(compute, haloReady, 0));
    }

private:
    MPI_Comm comm;
    size_t total = 0;
    int neighbors[6];
    unsigned blocks = 0;
    Faces faces;
    cudaStream_t stream = nullptr;
    cudaEvent_t bulkReady = nullptr;
    cudaEvent_t haloReady = nullptr;
    DeviceBuffer<Real> sendDevice;
    DeviceBuffer<Real> receiveDevice;
    Real* sendHost = nullptr;
    Real* receiveHost = nullptr;
    size_t receiveOffset = 0;
    std::vector<MPI_Request> requests;

    Faces makeFaces(const Box& box, size_t pitch, size_t slice) {
        const size_t stride[] = {1, pitch, slice};
        Faces result{};
        result.nx = box.size[0];
        result.ny = box.size[1];
        for (int axis = 0; axis < 3; ++axis) {
            MPI_Cart_shift(comm, axis, 1, &neighbors[2 * axis], &neighbors[2 * axis + 1]);
            const int u = axis == 0 ? 1 : 0;
            const int v = axis == 2 ? 1 : 2;
            for (int side = 0; side < 2; ++side) {
                const int id = 2 * axis + side;
                Face& face = result.face[id];
                face.endBlock = blocks;
                if (neighbors[id] == MPI_PROC_NULL) continue;
                const size_t first = slice + pitch + 1;
                face.start = first + (side ? size_t(box.size[axis] - 1) * stride[axis] : 0);
                face.halo = side ? face.start + stride[axis] : face.start - stride[axis];
                face.width = box.size[u];
                face.count = size_t(face.width) * box.size[v];
                face.strideU = stride[u];
                face.strideV = stride[v];
                face.offset = total;
                total += face.count;
                blocks += std::min((face.count + faceThreads - 1) / faceThreads, size_t(65535));
                face.endBlock = blocks;
            }
        }
        return result;
    }

    void postMessages(bool send, Real* buffer) {
        for (int side = 0; side < 6; ++side) {
            const Face& face = faces.face[side];
            for (size_t offset = 0; offset < face.count;) {
                const int count = int(std::min(face.count - offset, size_t(INT_MAX)));
                Real* message = buffer + face.offset + offset;
                requests.emplace_back();
                if (send) {
                    MPI_Isend(message, count, MPI_DOUBLE, neighbors[side], side, comm, &requests.back());
                } else {
                    MPI_Irecv(message, count, MPI_DOUBLE, neighbors[side], side ^ 1, comm, &requests.back());
                }
                offset += count;
            }
        }
    }
};

void copyOwnedGrid(Real* host, const Real* device, const Box& owned, const Box& box,
                   int border, size_t pitch) {
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(const_cast<Real*>(device), pitch * sizeof(Real),
                                     pitch * sizeof(Real), size_t(box.size[1]) + 2 * border);
    copy.srcPos = make_cudaPos(size_t(owned.start[0] - box.start[0] + border) * sizeof(Real),
                               owned.start[1] - box.start[1] + border,
                               owned.start[2] - box.start[2] + border);
    const size_t rowBytes = size_t(owned.size[0]) * sizeof(Real);
    copy.dstPtr = make_cudaPitchedPtr(host, rowBytes, rowBytes, owned.size[1]);
    copy.extent = make_cudaExtent(rowBytes, owned.size[1], owned.size[2]);
    copy.kind = cudaMemcpyDeviceToHost;
    CUDA(cudaMemcpy3D(&copy));
}

void collectResults(const Real* local, const Box& owned, const Shape& extent,
                    const Shape& global, const Shape& parts, int border, int rank,
                    MPI_Comm comm) {
    constexpr int tag = 6;
    if (rank != 0) {
        for (size_t offset = 0; offset < owned.count();) {
            const int count = int(std::min(owned.count() - offset, size_t(INT_MAX)));
            MPI_Send(local + offset, count, MPI_DOUBLE, 0, tag, comm);
            offset += count;
        }
        return;
    }

    std::vector<Real> grid(Box{{}, global}.count());
    const int ranks = parts[0] * parts[1] * parts[2];
    // The first block is at least as large as all others along every axis.
    std::unique_ptr<Real[]> received(ranks > 1 ? new Real[owned.count()] : nullptr);
    for (int source = 0; source < ranks; ++source) {
        Shape coord{};
        MPI_Cart_coords(comm, source, 3, coord.data());
        const Box chunk = includeBoundary(partition(extent, parts, coord, border),
                                          parts, coord, border);
        const Real* data = local;
        const size_t elements = chunk.count();
        if (source != 0) {
            for (size_t offset = 0; offset < elements;) {
                const int count = int(std::min(elements - offset, size_t(INT_MAX)));
                MPI_Recv(received.get() + offset, count, MPI_DOUBLE, source, tag, comm,
                         MPI_STATUS_IGNORE);
                offset += count;
            }
            data = received.get();
        }

        const size_t rows = size_t(chunk.size[1]) * chunk.size[2];
        #pragma omp parallel for schedule(static) num_threads(hostThreads(elements))
        for (size_t row = 0; row < rows; ++row) {
            const size_t z = chunk.start[2] + row / chunk.size[1];
            const size_t y = chunk.start[1] + row % chunk.size[1];
            const size_t destination = (z * global[1] + y) * global[0] + chunk.start[0];
            std::copy_n(data + row * chunk.size[0], chunk.size[0], grid.data() + destination);
        }
    }
    print_results(grid, "Grid");
}

bool validateResult(const Real* local, size_t count, int rank, MPI_Comm comm) {
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -minValue;
    int nonfinite = 0;
    #pragma omp parallel for schedule(static) num_threads(hostThreads(count)) \
        reduction(min:minValue) reduction(max:maxValue) reduction(|:nonfinite)
    for (size_t i = 0; i < count; ++i) {
        const Real value = local[i];
        nonfinite |= !std::isfinite(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    int anyNonfinite = 0;
    Real globalMin = 0;
    Real globalMax = 0;
    MPI_Reduce(&nonfinite, &anyNonfinite, 1, MPI_INT, MPI_BOR, 0, comm);
    MPI_Reduce(&minValue, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&maxValue, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) return true;

    if (anyNonfinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 1e6 || globalMin < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
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

struct Options {
    Shape global{128, 0, 0};
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

int parseOptions(int argc, char** argv, Options& options, int rank) {
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if (strcmp(option, "-v") == 0) {
            options.validate = true;
        } else if (strcmp(option, "-r") == 0) {
            options.printResults = true;
        } else if (strcmp(option, "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else if (strlen(option) == 2 && option[0] == '-' &&
                   strchr("xyzi", option[1]) && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const long n = strtol(value, &end, 10);
            if (errno || end == value || *end || n < 0 || n > INT_MAX ||
                (option[1] == 'x' && n == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid value for %s: %s\n", option, value);
                return 1;
            }
            if (option[1] == 'i') {
                options.iterations = int(n);
            } else {
                options.global[option[1] - 'x'] = int(n);
            }
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (options.global[1] == 0) options.global[1] = options.global[0];
    if (options.global[2] == 0) options.global[2] = options.global[0];

    size_t capacity = std::numeric_limits<size_t>::max() / sizeof(Real);
    for (int dim : options.global) {
        if (size_t(dim) > capacity) {
            if (rank == 0) fprintf(stderr, "Grid dimensions are too large\n");
            return 1;
        }
        capacity /= dim;
    }
    return -1;
}

int run(const Options& options, MPI_Comm comm, const Shape& parts,
        const Shape& extent, int border) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    Shape coord{};
    MPI_Cart_coords(comm, rank, 3, coord.data());
    const Box box = partition(extent, parts, coord, border);

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int deviceCount = 0;
    CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) CUDA(cudaErrorNoDevice);
    CUDA(cudaSetDevice(localRank % deviceCount));

    const size_t pitch = size_t(box.size[0]) + 2 * border;
    const size_t slice = pitch * (size_t(box.size[1]) + 2 * border);
    const size_t count = slice * (size_t(box.size[2]) + 2 * border);
    DeviceBuffer<Real> grid1(count);
    DeviceBuffer<Real> grid2(border && options.iterations > 0 ? count : 0);
    const auto bytes = count * sizeof(Real);
    {
        std::unique_ptr<Real[]> initial(new Real[count]);
        initializeGrid(initial.get(), box, options.global, border, pitch, slice, count);
        CUDA(cudaMemcpy(grid1.data, initial.get(), bytes, cudaMemcpyHostToDevice));
        if (grid2.data) CUDA(cudaMemcpy(grid2.data, grid1.data, bytes, cudaMemcpyDeviceToDevice));
        CUDA(cudaDeviceSynchronize());
    }

    Real* input = grid1.data;
    Real* output = grid2.data;
    double elapsed = 0.0;
    if (rank == 0) printf("Running stencil computation...\n");

    if (grid2.data) {
        cudaStream_t compute;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        cudaFuncAttributes attributes{};
        CUDA(cudaFuncGetAttributes(&attributes, stencil));
        HaloExchange halo(box, pitch, slice, comm, options.iterations > 1);
        const Shape last{box.size[0] + 1, box.size[1] + 1, box.size[2] + 1};
        Shape bulkFirst{1, 1, 1};
        Shape bulkLast = last;
        for (int axis = 0; axis < 3; ++axis) {
            bulkFirst[axis] += halo.hasNeighbor(2 * axis);
            bulkLast[axis] -= halo.hasNeighbor(2 * axis + 1);
        }

        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int step = 0; step < options.iterations; ++step) {
            const bool previousExchange = halo.active() && step != 0;
            if (halo.active() && step < options.iterations - 1) {
                halo.start(input, output, pitch, slice, compute, previousExchange);
                launchStencil(input, output, pitch, slice, bulkFirst, bulkLast, compute);
                halo.finish(output, compute);
            } else {
                if (previousExchange) halo.waitForHalo(compute);
                launchStencil(input, output, pitch, slice, {1, 1, 1}, last, compute);
            }
            std::swap(input, output);
        }
        CUDA(cudaStreamSynchronize(compute));
        elapsed = MPI_Wtime() - start;
        CUDA(cudaStreamDestroy(compute));
    }

    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", 1000.0 * maxElapsed);
        const double updates = border ? double(Box{{}, extent}.count()) * options.iterations : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }

    if (!options.printResults && !options.validate) return 0;
    const Box owned = includeBoundary(box, parts, coord, border);
    std::unique_ptr<Real[]> local(new Real[owned.count()]);
    copyOwnedGrid(local.get(), input, owned, box, border, pitch);
    if (options.printResults) {
        collectResults(local.get(), owned, extent, options.global, parts, border, rank, comm);
    }
    if (!options.validate) return 0;

    if (rank == 0) printf("Validating result...\n");
    const bool valid = validateResult(local.get(), owned.count(), rank, comm);
    if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    Options options;
    int status = parseOptions(argc, argv, options, rank);
    if (status < 0) {
        const int border = *std::min_element(options.global.begin(), options.global.end()) > 2;
        Shape extent = options.global;
        for (int& dim : extent) dim -= 2 * border;
        const Shape parts = chooseDecomposition(extent, ranks);
        const Shape periods{};
        MPI_Comm comm;
        MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods.data(), 0, &comm);
        status = 0;
        if (comm != MPI_COMM_NULL) {
            if (rank == 0) {
                printf("3D Stencil Benchmark\n");
                printf("Grid size: %d x %d x %d\n", options.global[0], options.global[1], options.global[2]);
                printf("Iterations: %d\n", options.iterations);
                printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
                printf("Initializing grid...\n");
            }
            try {
                status = run(options, comm, parts, extent, border);
            } catch (const std::exception& error) {
                fprintf(stderr, "Rank %d: %s\n", rank, error.what());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Comm_free(&comm);
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
