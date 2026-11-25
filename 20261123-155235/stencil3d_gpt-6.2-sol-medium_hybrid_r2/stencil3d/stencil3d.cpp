#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>

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

#include "../common/results_output.hpp"

using Real = double;
using Shape = std::array<int, 3>;

[[noreturn]] void fail(const char* operation, const char* detail) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

#define CUDA(operation) checkCuda((operation), #operation)

struct Box {
    int begin[3];
    int end[3];

    int length(int axis) const { return end[axis] - begin[axis]; }
};

struct Domain {
    Box owned;
    Box interior;
    Shape origin;
    Shape extent;
    std::array<size_t, 3> stride;

    size_t elements() const { return stride[2] * extent[2]; }
};

Box ownedBox(const Shape& global, const Shape& parts, const Shape& coords) {
    Box box{};
    for (int axis = 0; axis < 3; ++axis) {
        const int cells = std::max(global[axis] - 2, 0);
        const int count = cells / parts[axis];
        const int remainder = cells % parts[axis];
        const int coord = coords[axis];
        const int start = 1 + coord * count + std::min(coord, remainder);
        box.begin[axis] = coord == 0 ? 0 : start;
        box.end[axis] = coord == parts[axis] - 1
            ? global[axis] : start + count + (coord < remainder);
    }
    return box;
}

Domain localDomain(const Shape& global, const Shape& parts, const Shape& coords) {
    Domain domain{};
    const Box globalOwned = ownedBox(global, parts, coords);
    for (int axis = 0; axis < 3; ++axis) {
        const bool lowHalo = coords[axis] > 0;
        const bool highHalo = coords[axis] + 1 < parts[axis];
        domain.origin[axis] = globalOwned.begin[axis] - lowHalo;
        domain.owned.begin[axis] = lowHalo;
        domain.owned.end[axis] = lowHalo + globalOwned.length(axis);
        domain.extent[axis] = domain.owned.end[axis] + highHalo;
        domain.interior.begin[axis] = domain.owned.begin[axis] + !lowHalo;
        domain.interior.end[axis] = domain.owned.end[axis] - !highHalo;
    }
    domain.stride = {1, size_t(domain.extent[0]),
                     size_t(domain.extent[0]) * domain.extent[1]};
    return domain;
}

Shape processGrid(const Shape& global, int ranks) {
    const Shape limit{std::max(global[0] - 2, 1), std::max(global[1] - 2, 1),
                      std::max(global[2] - 2, 1)};
    const int capacity = int(std::min(size_t(ranks), size_t(limit[0]) * limit[1] * limit[2]));
    const double weight[] = {3.0, 2.0, 1.0};
    for (int used = capacity; used > 0; --used) {
        Shape best{};
        double bestCost = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(limit[0], used); ++x) {
            if (used % x != 0) continue;
            const int yz = used / x;
            for (int y = 1; y <= std::min(limit[1], yz); ++y) {
                if (yz % y != 0 || yz / y > limit[2]) continue;
                const Shape candidate{x, y, yz / y};
                // A Z face is contiguous in a slab. X faces have strided GPU loads.
                double cost = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    cost += weight[axis] * (candidate[axis] - 1) / global[axis];
                }
                if (cost < bestCost) {
                    best = candidate;
                    bestCost = cost;
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

__device__ __noinline__ Real divideSmall(Real sum) {
    return sum / 7.0;
}

__device__ Real average(Real sum) {
    if (sum < 0x1p-900) return divideSmall(sum);
    // The FMA correction gives the same rounding as division by seven.
    // The lower bound keeps the quotient and correction out of the subnormal range.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = sum * reciprocal;
    const Real remainder = fma(-7.0, quotient, sum);
    return fma(remainder, reciprocal, quotient);
}

__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        Box box, size_t row, size_t plane, int alignedX) {
    constexpr int planesPerThread = 4;
    const int x = alignedX + blockIdx.x * blockDim.x + threadIdx.x;
    if (x < box.begin[0] || x >= box.end[0]) return;

    for (size_t y = size_t(box.begin[1]) + blockIdx.y * blockDim.y + threadIdx.y;
         y < size_t(box.end[1]); y += gridDim.y * blockDim.y) {
        for (size_t z = size_t(box.begin[2]) + blockIdx.z * planesPerThread;
             z < size_t(box.end[2]); z += gridDim.z * planesPerThread) {
            const size_t first = x + y * row + z * plane;
            Real bottom = input[first - plane];
            Real center = input[first];
            size_t index = first;
            const int steps = min(planesPerThread, int(size_t(box.end[2]) - z));
            #pragma unroll
            for (int step = 0; step < steps; ++step) {
                const Real top = input[index + plane];
                output[index] = average(center + input[index - 1] + input[index + 1]
                                        + input[index - row] + input[index + row] + bottom + top);
                bottom = center;
                center = top;
                index += plane;
            }
        }
    }
}

void launchStencil(const Real* input, Real* output, const Domain& domain, Box box,
                   cudaStream_t stream) {
    if (box.length(0) <= 0 || box.length(1) <= 0 || box.length(2) <= 0) return;
    const unsigned width = box.length(0) == 1 ? 1 : (box.length(0) <= 16 ? 8 : 32);
    const dim3 block(width, std::min(128 / width, 32u));
    const int alignedX = box.begin[0] / width * width;
    const dim3 blocks((unsigned(box.end[0] - alignedX) + block.x - 1) / block.x,
                      std::min((unsigned(box.length(1)) + block.y - 1) / block.y, 65535u),
                      std::min((unsigned(box.length(2)) + 3) / 4, 65535u));
    stencil<<<blocks, block, 0, stream>>>(input, output, box, domain.stride[1],
                                         domain.stride[2], alignedX);
    CUDA(cudaGetLastError());
}

struct Face {
    int axis;
    int side;
    int peer;
    size_t sendOffset;
    size_t recvOffset;
    int width;
    int height;
    size_t colStride;
    size_t rowStride;
    Real* packedSend = nullptr;
    Real* packedRecv = nullptr;
    Real* hostSend = nullptr;
    Real* hostRecv = nullptr;

    size_t count() const { return size_t(width) * height; }
    bool contiguous() const { return colStride == 1 && (height == 1 || rowStride == size_t(width)); }
};

template<bool pack>
__global__ void transferFace(Real* grid, Real* packed, int width, int height,
                             size_t colStride, size_t rowStride) {
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= size_t(width)) return;
    for (size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         row < size_t(height); row += gridDim.y * blockDim.y) {
        const size_t index = row * rowStride + col * colStride;
        if constexpr (pack) packed[row * width + col] = grid[index];
        else grid[index] = packed[row * width + col];
    }
}

template<bool pack>
void launchFace(Real* grid, Real* packed, const Face& face, cudaStream_t stream) {
    const dim3 block(32, 4);
    const dim3 blocks((unsigned(face.width) + block.x - 1) / block.x,
                      std::min((unsigned(face.height) + block.y - 1) / block.y, 65535u));
    transferFace<pack><<<blocks, block, 0, stream>>>(grid, packed, face.width, face.height,
                                                    face.colStride, face.rowStride);
    CUDA(cudaGetLastError());
}

class HaloExchange {
public:
    HaloExchange(MPI_Comm comm, const Domain& domain, bool needed)
        : comm(comm), core(domain.interior) {
        if (!needed) return;
        #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        deviceAware = MPIX_Query_cuda_support() != 0;
        #endif

        const Box& owned = domain.owned;
        const size_t start = owned.begin[0] + owned.begin[1] * domain.stride[1]
                                            + owned.begin[2] * domain.stride[2];
        for (int axis = 0; axis < 3; ++axis) {
            int peers[2];
            MPI_Cart_shift(comm, axis, 1, &peers[0], &peers[1]);
            const int col = axis == 0 ? 1 : 0;
            const int row = axis == 2 ? 1 : 2;
            for (int side = 0; side < 2; ++side) {
                if (peers[side] == MPI_PROC_NULL) continue;
                const size_t stride = domain.stride[axis];
                const size_t offset = start + (side == 0 ? 0 : owned.length(axis) - 1) * stride;
                Face face{axis, side, peers[side], offset,
                          side == 0 ? offset - stride : offset + stride,
                          owned.length(col), owned.length(row),
                          domain.stride[col], domain.stride[row]};
                const size_t bytes = face.count() * sizeof(Real);
                if (!face.contiguous()) {
                    CUDA(cudaMalloc(&face.packedSend, bytes));
                    CUDA(cudaMalloc(&face.packedRecv, bytes));
                }
                if (!deviceAware) {
                    CUDA(cudaMallocHost(&face.hostSend, bytes));
                    CUDA(cudaMallocHost(&face.hostRecv, bytes));
                }
                faces.push_back(face);

                // Peel each skin box from the core so edges and corners are updated once.
                if (core.length(axis) > 0) {
                    Box skin = core;
                    if (side == 0) skin.end[axis] = ++core.begin[axis];
                    else skin.begin[axis] = --core.end[axis];
                    skins.push_back(skin);
                }
            }
        }
        if (!faces.empty()) {
            size_t requestCount = 0;
            for (const Face& face : faces) requestCount += 2 * (1 + (face.count() - 1) / INT_MAX);
            requests.reserve(requestCount);
            int highestPriority;
            CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &highestPriority));
            CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, highestPriority));
            CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        }
    }

    ~HaloExchange() {
        if (faces.empty()) return;
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaStreamDestroy(stream));
        for (const Face& face : faces) {
            if (face.packedSend) {
                CUDA(cudaFree(face.packedSend));
                CUDA(cudaFree(face.packedRecv));
            }
            if (face.hostSend) {
                CUDA(cudaFreeHost(face.hostSend));
                CUDA(cudaFreeHost(face.hostRecv));
            }
        }
    }

    void step(Real* input, Real* output, const Domain& domain, cudaStream_t compute) {
        if (faces.empty()) {
            launchStencil(input, output, domain, domain.interior, compute);
            return;
        }

        // The previous step must have produced the owned cells used for outgoing halos.
        CUDA(cudaStreamSynchronize(compute));
        requests.clear();
        for (const Face& face : faces) {
            Real* receive = face.contiguous() ? input + face.recvOffset : face.packedRecv;
            post(face, deviceAware ? receive : face.hostRecv, false);
        }
        launchStencil(input, output, domain, core, compute);

        bool preparing = false;
        for (const Face& face : faces) {
            Real* send = input + face.sendOffset;
            if (!face.contiguous()) {
                launchFace<true>(send, face.packedSend, face, stream);
                send = face.packedSend;
                preparing = true;
            }
            if (!deviceAware) {
                CUDA(cudaMemcpyAsync(face.hostSend, send, face.count() * sizeof(Real),
                                     cudaMemcpyDeviceToHost, stream));
                preparing = true;
            }
        }
        if (preparing) CUDA(cudaStreamSynchronize(stream));

        for (const Face& face : faces) {
            Real* send = face.contiguous() ? input + face.sendOffset : face.packedSend;
            post(face, deviceAware ? send : face.hostSend, true);
        }
        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        for (const Face& face : faces) {
            Real* receive = input + face.recvOffset;
            Real* destination = face.contiguous() ? receive : face.packedRecv;
            if (!deviceAware) {
                CUDA(cudaMemcpyAsync(destination, face.hostRecv, face.count() * sizeof(Real),
                                     cudaMemcpyHostToDevice, stream));
            }
            if (!face.contiguous()) launchFace<false>(receive, destination, face, stream);
        }
        for (Box skin : skins) launchStencil(input, output, domain, skin, stream);
        CUDA(cudaEventRecord(ready, stream));
        CUDA(cudaStreamWaitEvent(compute, ready, 0));
    }

private:
    void post(const Face& face, Real* buffer, bool send) {
        const int tag = 2 * face.axis + (send ? face.side : 1 - face.side);
        for (size_t done = 0; done < face.count();) {
            const int count = int(std::min(face.count() - done, size_t(INT_MAX)));
            MPI_Request request;
            if (send) MPI_Isend(buffer + done, count, MPI_DOUBLE, face.peer, tag, comm, &request);
            else MPI_Irecv(buffer + done, count, MPI_DOUBLE, face.peer, tag, comm, &request);
            requests.push_back(request);
            done += count;
        }
    }

    MPI_Comm comm;
    Box core;
    bool deviceAware = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    std::vector<Face> faces;
    std::vector<Box> skins;
    std::vector<MPI_Request> requests;
};

void initialize(Real* input, Real* output, const Shape& global, const Domain& domain,
                cudaStream_t stream) {
    const size_t count = domain.elements();
    std::unique_ptr<Real[]> host(new Real[count]);
    const size_t globalPlane = size_t(global[0]) * global[1];
    #pragma omp parallel for collapse(2) schedule(static) if(count >= 32768)
    for (size_t z = 0; z < size_t(domain.extent[2]); ++z) {
        for (size_t y = 0; y < size_t(domain.extent[1]); ++y) {
            const size_t base = (z + domain.origin[2]) * globalPlane
                                + (y + domain.origin[1]) * global[0] + domain.origin[0];
            Real* dest = host.get() + z * domain.stride[2] + y * domain.stride[1];
            for (int x = 0; x < domain.extent[0]; ++x) dest[x] = Real((base + x) % 19);
        }
    }
    const size_t bytes = count * sizeof(Real);
    CUDA(cudaMemcpyAsync(input, host.get(), bytes, cudaMemcpyHostToDevice, stream));
    // Both buffers keep their initialized physical boundary cells throughout the run.
    CUDA(cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA(cudaStreamSynchronize(stream));
}

size_t volume(const Box& box) {
    return size_t(box.length(0)) * box.length(1) * box.length(2);
}

void download(Real* host, Real* device, const Domain& domain) {
    const Box& owned = domain.owned;
    if (owned.length(0) == domain.extent[0] && owned.length(1) == domain.extent[1]) {
        CUDA(cudaMemcpy(host, device + owned.begin[2] * domain.stride[2],
                        volume(owned) * sizeof(Real), cudaMemcpyDeviceToHost));
        return;
    }
    const size_t rowBytes = owned.length(0) * sizeof(Real);
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(device, domain.stride[1] * sizeof(Real),
                                    domain.extent[0] * sizeof(Real), domain.extent[1]);
    copy.srcPos = make_cudaPos(owned.begin[0] * sizeof(Real), owned.begin[1], owned.begin[2]);
    copy.dstPtr = make_cudaPitchedPtr(host, rowBytes, rowBytes, owned.length(1));
    copy.extent = make_cudaExtent(rowBytes, owned.length(1), owned.length(2));
    copy.kind = cudaMemcpyDeviceToHost;
    CUDA(cudaMemcpy3D(&copy));
}

std::array<Real, 3> resultRange(const Real* data, size_t count, MPI_Comm comm) {
    Real smallest = std::numeric_limits<Real>::infinity();
    Real largest = -smallest;
    int invalid = 0;
    #pragma omp parallel for schedule(static) reduction(min:smallest) reduction(max:largest) reduction(|:invalid) if(count >= 32768)
    for (size_t i = 0; i < count; ++i) {
        const Real value = data[i];
        if (std::isfinite(value)) {
            smallest = std::min(smallest, value);
            largest = std::max(largest, value);
        } else {
            invalid = 1;
        }
    }
    const Real local[] = {Real(invalid), -smallest, largest};
    std::array<Real, 3> global{};
    MPI_Allreduce(local, global.data(), 3, MPI_DOUBLE, MPI_MAX, comm);
    return global;
}

void reportResults(const Real* local, size_t count, const Shape& global, const Shape& parts,
                   MPI_Comm comm, int rank) {
    // The shared result hash and compensated sum require the original global order.
    const size_t plane = size_t(global[0]) * global[1];
    constexpr int tag = 6;
    if (rank != 0) {
        for (size_t done = 0; done < count;) {
            const int chunk = int(std::min(count - done, size_t(INT_MAX)));
            MPI_Send(local + done, chunk, MPI_DOUBLE, 0, tag, comm);
            done += chunk;
        }
        return;
    }

    std::vector<Real> result(plane * global[2]);
    std::unique_ptr<Real[]> scratch;
    size_t scratchCapacity = 0;
    const bool slab = parts[0] == 1 && parts[1] == 1;
    int ranks;
    MPI_Comm_size(comm, &ranks);
    for (int peer = 0; peer < ranks; ++peer) {
        Shape coords;
        MPI_Cart_coords(comm, peer, 3, coords.data());
        const Box box = ownedBox(global, parts, coords);
        const size_t cells = volume(box);
        Real* destination = result.data() + box.begin[2] * plane;
        const Real* data = local;
        if (peer != 0) {
            if (!slab) {
                if (scratchCapacity < cells) {
                    scratch.reset(new Real[cells]);
                    scratchCapacity = cells;
                }
                destination = scratch.get();
            }
            for (size_t done = 0; done < cells;) {
                const int chunk = int(std::min(cells - done, size_t(INT_MAX)));
                MPI_Recv(destination + done, chunk, MPI_DOUBLE, peer, tag, comm, MPI_STATUS_IGNORE);
                done += chunk;
            }
            if (slab) continue;
            data = destination;
        }
        const size_t width = box.length(0);
        const size_t height = box.length(1);
        const size_t depth = box.length(2);
        const size_t localPlane = width * height;
        #pragma omp parallel for collapse(2) schedule(static) if(cells >= 32768)
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                const size_t offset = (z + box.begin[2]) * plane
                                      + (y + box.begin[1]) * global[0] + box.begin[0];
                std::copy_n(data + z * localPlane + y * width, width, result.data() + offset);
            }
        }
    }
    print_results(result, "Grid");
}

bool benchmark(const Shape& global, int iterations, bool validate, bool printResults,
               const Shape& parts, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    Shape coords;
    MPI_Cart_coords(comm, rank, 3, coords.data());
    const Domain domain = localDomain(global, parts, coords);

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("CUDA device selection", "no visible CUDA devices");
    CUDA(cudaSetDevice(localRank % devices));
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(omp_get_max_threads(), 8));

    if (rank == 0) printf("Initializing grid...\n");
    const size_t bytes = domain.elements() * sizeof(Real);
    Real* input;
    Real* output;
    CUDA(cudaMalloc(&input, bytes));
    CUDA(cudaMalloc(&output, bytes));
    cudaStream_t compute;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    initialize(input, output, global, domain, compute);
    const bool updates = global[0] > 2 && global[1] > 2 && global[2] > 2;
    HaloExchange halos(comm, domain, updates && iterations > 1);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; updates && iter < iterations; ++iter) {
        // The first stencil can use the halos initialized from global indices.
        if (iter == 0) launchStencil(input, output, domain, domain.interior, compute);
        else halos.step(input, output, domain, compute);
        std::swap(input, output);
    }
    CUDA(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double cells = updates ? double(global[0] - 2) * (global[1] - 2) * (global[2] - 2) : 0.0;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n",
               seconds > 0 ? cells * iterations / (seconds * 1e6) : 0.0);
    }

    const size_t count = volume(domain.owned);
    std::unique_ptr<Real[]> local;
    if (validate || printResults) {
        local.reset(new Real[count]);
        download(local.get(), input, domain);
    }
    CUDA(cudaFree(input));
    CUDA(cudaFree(output));
    CUDA(cudaStreamDestroy(compute));

    if (printResults) reportResults(local.get(), count, global, parts, comm, rank);
    if (!validate) return true;

    if (rank == 0) printf("Validating result...\n");
    const auto range = resultRange(local.get(), count, comm);
    const bool finite = range[0] == 0.0;
    const bool valid = finite && -range[1] >= -1e6 && range[2] <= 1e6;
    if (rank == 0) {
        if (finite) {
            printf("Value range: [%.6f, %.6f]\n", -range[1], range[2]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        } else {
            printf("Validation failed: found NaN or Inf value\n");
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

int run(int argc, char** argv, int rank) {
    Shape global{128, 0, 0};
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        int* number = nullptr;
        if (strcmp(option, "-x") == 0) number = &global[0];
        else if (strcmp(option, "-y") == 0) number = &global[1];
        else if (strcmp(option, "-z") == 0) number = &global[2];
        else if (strcmp(option, "-i") == 0) number = &iterations;
        else if (strcmp(option, "-v") == 0) validate = true;
        else if (strcmp(option, "-r") == 0) printResults = true;
        else if (strcmp(option, "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
        if (number) {
            char* end = nullptr;
            errno = 0;
            const long value = i + 1 < argc ? std::strtol(argv[++i], &end, 10) : -1;
            if (value < 0 || value > INT_MAX || errno != 0 || end == argv[i] || *end != '\0') {
                if (rank == 0) fprintf(stderr, "Invalid value for %s\n", option);
                return 1;
            }
            *number = int(value);
        }
    }
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    if (global[0] == 0 || std::numeric_limits<size_t>::max() / sizeof(Real) / global[0]
                                           / global[1] / global[2] == 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid size\n");
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    Shape parts;
    if (rank == 0) parts = processGrid(global, ranks);
    MPI_Bcast(parts.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    int periods[3] = {0, 0, 0};
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods, 0, &comm);
    int status = 0;
    if (comm != MPI_COMM_NULL) {
        status = benchmark(global, iterations, validate, printResults, parts, comm) ? 0 : 1;
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) fail("MPI initialization", "MPI_THREAD_FUNNELED is unavailable");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status;
    try {
        status = run(argc, argv, rank);
    } catch (const std::exception& error) {
        fail("stencil3d", error.what());
    }
    MPI_Finalize();
    return status;
}
