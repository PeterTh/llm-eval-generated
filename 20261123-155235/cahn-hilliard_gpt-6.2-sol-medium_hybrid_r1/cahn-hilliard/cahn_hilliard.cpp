#include <cuda_runtime.h>
#include <mpi.h>
#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void fail(const char* message, const char* detail = "") {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s%s\n", rank, message, detail);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) fail("CUDA: ", cudaGetErrorString(status));
}

struct Extent {
    int x, y, z;

    __host__ __device__ size_t plane() const { return size_t(x) * y; }
    __host__ __device__ size_t volume() const { return plane() * z; }
    __host__ __device__ size_t index(int i, int j, int k) const {
        return size_t(k) * plane() + size_t(j) * x + i;
    }
};

struct Box {
    Extent begin, size;
};

Box rankBox(Extent grid, Extent ranks, Extent coords) {
    // Distribute remainders across the first coordinates of each axis.
    const auto split = [](int cells, int parts, int coord, int& begin, int& size) {
        begin = coord * (cells / parts) + std::min(coord, cells % parts);
        size = cells / parts + (coord < cells % parts);
    };
    Box box;
    split(grid.x, ranks.x, coords.x, box.begin.x, box.size.x);
    split(grid.y, ranks.y, coords.y, box.begin.y, box.size.y);
    split(grid.z, ranks.z, coords.z, box.begin.z, box.size.z);
    return box;
}

Extent processGrid(Extent grid, int available) {
    // Use the largest feasible rank count, then minimize the worst tile's
    // work and face traffic. In a tie keep the contiguous x axis intact.
    for (int count = int(std::min(grid.volume(), size_t(available))); count > 0; --count) {
        Extent best{};
        double bestCost = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= grid.x && x <= count; ++x) {
            if (count % x) continue;
            const int yz = count / x;
            for (int y = 1; y <= grid.y && y <= yz; ++y) {
                if (yz % y) continue;
                const int z = yz / y;
                if (z > grid.z) continue;
                const Extent tile{
                    1 + (grid.x - 1) / x, 1 + (grid.y - 1) / y, 1 + (grid.z - 1) / z
                };
                const double faces = std::min(x - 1, 2) * double(tile.y) * tile.z
                                   + std::min(y - 1, 2) * double(tile.x) * tile.z
                                   + std::min(z - 1, 2) * double(tile.plane());
                const double cost = double(tile.volume()) + 4.0 * faces;
                if (cost < bestCost) {
                    best = {x, y, z};
                    bestCost = cost;
                }
            }
        }
        if (best.x) return best;
    }
    return {1, 1, 1};
}

void initializeConcentration(double* field, Extent grid, Box box) {
    const size_t volume = grid.volume();
    #pragma omp parallel for collapse(2) schedule(static) if(box.size.volume() >= 16384)
    for (int z = 0; z < box.size.z; ++z) {
        for (int y = 0; y < box.size.y; ++y) {
            const size_t local = box.size.index(0, y, z);
            const size_t global = grid.index(box.begin.x, y + box.begin.y, z + box.begin.z);
            for (int x = 0; x < box.size.x; ++x) {
                const double pseudo = ((global + x + 1) * 1299709) % volume / double(volume);
                field[local + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Faces are ordered low/high x, low/high y, low/high z. The buffers store
// each face in the same x-fastest order as the local concentration field.
constexpr int faceCount = 6;
constexpr size_t noFace = size_t(-1);

struct Faces {
    size_t offset[faceCount];
};

struct Regions {
    Box boxes[faceCount]{};
    int count = 0;
    size_t largest = 0;

    void add(Box box) {
        const size_t cells = box.size.volume();
        if (cells != 0) {
            boxes[count++] = box;
            largest = std::max(largest, cells);
        }
    }
};

void takeFace(Box& interior, Regions& boundary, int face) {
    int* begin[] = {&interior.begin.x, &interior.begin.y, &interior.begin.z};
    int* size[] = {&interior.size.x, &interior.size.y, &interior.size.z};
    const int axis = face / 2;
    if (*size[axis] == 0) return;

    Box edge = interior;
    int* edgeBegin[] = {&edge.begin.x, &edge.begin.y, &edge.begin.z};
    int* edgeSize[] = {&edge.size.x, &edge.size.y, &edge.size.z};
    *edgeSize[axis] = 1;
    --*size[axis];
    if (face % 2) {
        *edgeBegin[axis] += *size[axis];
    } else {
        ++*begin[axis];
    }
    boundary.add(edge);
}

__device__ Extent regionPoint(size_t linear, Box box) {
    const size_t row = linear / box.size.x;
    const size_t z = row / box.size.y;
    return {box.begin.x + int(linear - row * box.size.x),
            box.begin.y + int(row - z * box.size.y), box.begin.z + int(z)};
}

__global__ void packFaces(const double* field, double* packed, Extent size,
                          Regions regions, Faces starts) {
    const Box box = regions.boxes[blockIdx.y];
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < box.size.volume(); i += size_t(gridDim.x) * blockDim.x) {
        const Extent p = regionPoint(i, box);
        packed[starts.offset[blockIdx.y] + i] = field[size.index(p.x, p.y, p.z)];
    }
}

template <bool chemical, bool useHalo>
__device__ double stencilPoint(const double* field, const double* next, const double* halo,
                               Extent size, Faces faces, int x, int y, int z) {
    const size_t row = size.x;
    const size_t plane = size.plane();
    const size_t i = size.index(x, y, z);
    const double center = field[i];

    const size_t fx = size_t(z) * size.y + y;
    const size_t fy = size_t(z) * size.x + x;
    const size_t fz = size_t(y) * size.x + x;
    const double xm = x > 0 ? field[i - 1] :
        (useHalo && faces.offset[0] != noFace ? halo[faces.offset[0] + fx] : center);
    const double xp = x < size.x - 1 ? field[i + 1] :
        (useHalo && faces.offset[1] != noFace ? halo[faces.offset[1] + fx] : center);
    const double ym = y > 0 ? field[i - row] :
        (useHalo && faces.offset[2] != noFace ? halo[faces.offset[2] + fy] : center);
    const double yp = y < size.y - 1 ? field[i + row] :
        (useHalo && faces.offset[3] != noFace ? halo[faces.offset[3] + fy] : center);
    const double zm = z > 0 ? field[i - plane] :
        (useHalo && faces.offset[4] != noFace ? halo[faces.offset[4] + fz] : center);
    const double zp = z < size.z - 1 ? field[i + plane] :
        (useHalo && faces.offset[5] != noFace ? halo[faces.offset[5] + fz] : center);
    const double twice = 2.0 * center;
    // Grid spacings and diffusivity are all 1 in this benchmark.
    const double lap = ((xp + xm - twice) + (yp + ym - twice)) + (zp + zm - twice);
    if constexpr (chemical) {
        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = 2.0 / 9.0;
        constexpr double gamma = 0.5;
        const double energy = fma(-2.0 * center, eAB,
                                  fma(center + 1.0, eAA, (center - 1.0) * eBB));
        const double bulk = fma(center * center, center, fma(4.5, energy, 3.0 * center));
        return fma(-gamma, lap, bulk);
    } else {
        constexpr double dt = 0.01;
        return fma(dt, lap, next[i]);
    }
}

template <bool chemical>
__global__ void interiorStencil(const double* __restrict__ field, double* __restrict__ next,
                                Extent size, Box box) {
    for (size_t z = size_t(blockIdx.z) * blockDim.z + threadIdx.z;
         z < size_t(box.size.z); z += size_t(gridDim.z) * blockDim.z) {
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < size_t(box.size.y); y += size_t(gridDim.y) * blockDim.y) {
            for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < size_t(box.size.x); x += size_t(gridDim.x) * blockDim.x) {
                const int i = box.begin.x + int(x);
                const int j = box.begin.y + int(y);
                const int k = box.begin.z + int(z);
                next[size.index(i, j, k)] =
                    stencilPoint<chemical, false>(field, next, nullptr, size, {}, i, j, k);
            }
        }
    }
}

template <bool chemical>
__global__ void boundaryStencil(const double* __restrict__ field, double* __restrict__ next,
                                const double* __restrict__ halo, double* __restrict__ outgoing,
                                Extent size, Faces faces, Regions regions) {
    const Box box = regions.boxes[blockIdx.y];
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < box.size.volume(); i += size_t(gridDim.x) * blockDim.x) {
        const Extent p = regionPoint(i, box);
        const double value = stencilPoint<chemical, true>(field, next, halo, size, faces, p.x, p.y, p.z);
        next[size.index(p.x, p.y, p.z)] = value;
        // Export the output faces here so the next stencil can send them
        // without reading the field again. Edge cells can belong to two faces.
        const size_t fx = size_t(p.z) * size.y + p.y;
        const size_t fy = size_t(p.z) * size.x + p.x;
        const size_t fz = size_t(p.y) * size.x + p.x;
        if (p.x == 0 && faces.offset[0] != noFace) outgoing[faces.offset[0] + fx] = value;
        if (p.x == size.x - 1 && faces.offset[1] != noFace) outgoing[faces.offset[1] + fx] = value;
        if (p.y == 0 && faces.offset[2] != noFace) outgoing[faces.offset[2] + fy] = value;
        if (p.y == size.y - 1 && faces.offset[3] != noFace) outgoing[faces.offset[3] + fy] = value;
        if (p.z == 0 && faces.offset[4] != noFace) outgoing[faces.offset[4] + fz] = value;
        if (p.z == size.z - 1 && faces.offset[5] != noFace) outgoing[faces.offset[5] + fz] = value;
    }
}

constexpr int threads = 256;

unsigned blocks(size_t cells, int width) {
    return unsigned(std::min((cells + width - 1) / width, size_t(65535)));
}

dim3 stencilBlock(Extent size) {
    // Preserve coalesced x accesses, while filling warps on narrow grids.
    unsigned x = 32;
    while (x > 1 && x / 2 >= unsigned(size.x)) x /= 2;
    unsigned y = threads / x;
    while (y > 1 && y / 2 >= unsigned(size.y)) y /= 2;
    unsigned z = std::min(64u, threads / (x * y));
    while (z > 1 && z / 2 >= unsigned(size.z)) z /= 2;
    return {x, y, z};
}

bool cudaAwareMPI() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

class Stencils {
public:
    Stencils(Extent size, MPI_Comm cart) : size(size), interior{{0, 0, 0}, size} {
        checkCuda(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking));
        int peers[faceCount];
        // MPI Cartesian axes are z, y, x so ranks vary fastest in x.
        for (int axis = 0; axis < 3; ++axis) {
            MPI_Cart_shift(cart, 2 - axis, 1, &peers[2 * axis], &peers[2 * axis + 1]);
        }

        size_t cells = 0;
        for (int face = 0; face < faceCount; ++face) {
            faces.offset[face] = noFace;
            if (peers[face] == MPI_PROC_NULL) continue;
            takeFace(interior, boundary, face);
            Box plane = {{0, 0, 0}, size};
            if (face / 2 == 0) {
                plane.size.x = 1;
                if (face % 2) plane.begin.x = size.x - 1;
            } else if (face / 2 == 1) {
                plane.size.y = 1;
                if (face % 2) plane.begin.y = size.y - 1;
            } else {
                plane.size.z = 1;
                if (face % 2) plane.begin.z = size.z - 1;
            }
            faces.offset[face] = cells;
            packStarts.offset[packing.count] = cells;
            packing.add(plane);
            cells += plane.size.volume();
        }
        if (interior.size.volume()) {
            interiorBlock = stencilBlock(interior.size);
            interiorGrid = {blocks(interior.size.x, interiorBlock.x),
                            blocks(interior.size.y, interiorBlock.y),
                            blocks(interior.size.z, interiorBlock.z)};
        }
        if (cells == 0) return;

        boundaryGrid = {blocks(boundary.largest, threads), unsigned(boundary.count)};
        haloBytes = cells * sizeof(double);
        checkCuda(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        checkCuda(cudaMalloc(&deviceSend, haloBytes));
        checkCuda(cudaMalloc(&deviceRecv, haloBytes));
        double* sendBuffer = deviceSend;
        double* recvBuffer = deviceRecv;
        if (!cudaAwareMPI()) {
            checkCuda(cudaMallocHost(&hostSend, haloBytes));
            checkCuda(cudaMallocHost(&hostRecv, haloBytes));
            sendBuffer = hostSend;
            recvBuffer = hostRecv;
        }

        for (int box = 0, face = 0; face < faceCount; ++face) {
            if (peers[face] == MPI_PROC_NULL) continue;
            const size_t end = faces.offset[face] + packing.boxes[box++].size.volume();
            for (size_t i = faces.offset[face]; i < end;) {
                const int count = int(std::min(end - i, size_t(INT_MAX)));
                MPI_Request send, recv;
                MPI_Send_init(sendBuffer + i, count, MPI_DOUBLE, peers[face], face, cart, &send);
                MPI_Recv_init(recvBuffer + i, count, MPI_DOUBLE, peers[face], face ^ 1, cart, &recv);
                sends.push_back(send);
                receives.push_back(recv);
                i += count;
            }
        }
    }

    Stencils(const Stencils&) = delete;
    Stencils& operator=(const Stencils&) = delete;

    ~Stencils() {
        if (stepGraph) checkCuda(cudaGraphExecDestroy(stepGraph));
        for (auto& request : sends) MPI_Request_free(&request);
        for (auto& request : receives) MPI_Request_free(&request);
        if (transfer) {
            if (hostSend) {
                checkCuda(cudaFreeHost(hostSend));
                checkCuda(cudaFreeHost(hostRecv));
            }
            checkCuda(cudaFree(deviceSend));
            checkCuda(cudaFree(deviceRecv));
            checkCuda(cudaEventDestroy(ready));
            checkCuda(cudaStreamDestroy(transfer));
        }
        checkCuda(cudaStreamDestroy(work));
    }

    void prepare(double* concentration, double* mu, int iterations) {
        if (iterations <= 0) return;
        if (transfer) {
            packFaces<<<dim3(blocks(packing.largest, threads), packing.count), threads, 0, work>>>(
                concentration, deviceSend, size, packing, packStarts);
            checkCuda(cudaGetLastError());
        } else {
            // With no MPI neighbors the two kernels can be replayed together.
            cudaGraph_t graph;
            checkCuda(cudaStreamBeginCapture(work, cudaStreamCaptureModeThreadLocal));
            apply<true>(concentration, mu);
            apply<false>(mu, concentration);
            checkCuda(cudaStreamEndCapture(work, &graph));
            checkCuda(cudaGraphInstantiate(&stepGraph, graph, nullptr, nullptr, 0));
            checkCuda(cudaGraphDestroy(graph));
            checkCuda(cudaGraphUpload(stepGraph, work));
        }
        synchronize();
    }

    void step(double* concentration, double* mu) {
        if (stepGraph) {
            checkCuda(cudaGraphLaunch(stepGraph, work));
        } else {
            apply<true>(concentration, mu);
            // The update reads neighboring mu values but only its own
            // concentration, so updating concentration in place is safe.
            apply<false>(mu, concentration);
        }
    }

    void synchronize() const { checkCuda(cudaStreamSynchronize(work)); }

private:
    template <bool chemical>
    void apply(const double* field, double* next) {
        if (transfer) checkCuda(cudaEventRecord(ready, work));
        if (hostSend) {
            checkCuda(cudaStreamWaitEvent(transfer, ready, 0));
            checkCuda(cudaMemcpyAsync(hostSend, deviceSend, haloBytes, cudaMemcpyDeviceToHost, transfer));
        }

        if (interior.size.volume()) {
            interiorStencil<chemical><<<interiorGrid, interiorBlock, 0, work>>>(field, next, size, interior);
        }

        if (transfer) {
            // The prior stencil has also finished consuming the old halos
            // before either MPI or a host-to-device copy overwrites them.
            if (hostSend) {
                checkCuda(cudaStreamSynchronize(transfer));
            } else {
                checkCuda(cudaEventSynchronize(ready));
            }
            MPI_Startall(int(receives.size()), receives.data());
            MPI_Startall(int(sends.size()), sends.data());
            MPI_Waitall(int(receives.size()), receives.data(), MPI_STATUSES_IGNORE);
            if (hostRecv) {
                checkCuda(cudaMemcpyAsync(deviceRecv, hostRecv, haloBytes, cudaMemcpyHostToDevice, transfer));
                checkCuda(cudaEventRecord(ready, transfer));
                checkCuda(cudaStreamWaitEvent(work, ready, 0));
            }
            MPI_Waitall(int(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
            boundaryStencil<chemical><<<boundaryGrid, threads, 0, work>>>(
                field, next, deviceRecv, deviceSend, size, faces, boundary);
        }
        checkCuda(cudaGetLastError());
    }

    Extent size;
    Box interior;
    Regions boundary, packing;
    Faces faces{}, packStarts{};
    dim3 interiorGrid, interiorBlock, boundaryGrid;
    size_t haloBytes = 0;
    cudaStream_t work = nullptr, transfer = nullptr;
    cudaEvent_t ready = nullptr;
    cudaGraphExec_t stepGraph = nullptr;
    double *deviceSend = nullptr, *deviceRecv = nullptr;
    double *hostSend = nullptr, *hostRecv = nullptr;
    std::vector<MPI_Request> sends, receives;
};

MPI_Datatype brickType(Extent storage, Extent brick) {
    // Use a count of one even when the local brick exceeds INT_MAX elements.
    MPI_Datatype row, plane, volume;
    MPI_Type_contiguous(brick.x, MPI_DOUBLE, &row);
    MPI_Type_create_hvector(brick.y, 1, MPI_Aint(storage.x) * sizeof(double), row, &plane);
    MPI_Type_create_hvector(brick.z, 1, MPI_Aint(storage.plane() * sizeof(double)), plane, &volume);
    MPI_Type_commit(&volume);
    MPI_Type_free(&plane);
    MPI_Type_free(&row);
    return volume;
}

void printGlobalResult(const double* local, Extent grid, Extent ranks, Box own, int rank, MPI_Comm cart) {
    constexpr int resultsTag = faceCount;
    if (rank != 0) {
        MPI_Datatype type = brickType(own.size, own.size);
        MPI_Send(local, 1, type, 0, resultsTag, cart);
        MPI_Type_free(&type);
        return;
    }

    std::vector<double> result(grid.volume());
    #pragma omp parallel for collapse(2) schedule(static) if(own.size.volume() >= 16384)
    for (int z = 0; z < own.size.z; ++z) {
        for (int y = 0; y < own.size.y; ++y) {
            const size_t dst = grid.index(own.begin.x, y + own.begin.y, z + own.begin.z);
            std::copy_n(local + own.size.index(0, y, z), own.size.x, result.data() + dst);
        }
    }

    const int active = int(ranks.volume());
    for (int source = 1; source < active; ++source) {
        int coords[3];
        MPI_Cart_coords(cart, source, 3, coords);
        const Box box = rankBox(grid, ranks, {coords[2], coords[1], coords[0]});
        MPI_Datatype type = brickType(grid, box.size);
        double* dst = result.data() + grid.index(box.begin.x, box.begin.y, box.begin.z);
        MPI_Recv(dst, 1, type, source, resultsTag, cart, MPI_STATUS_IGNORE);
        MPI_Type_free(&type);
    }
    print_results(result, "Concentration");
}

bool validateResult(const double* field, size_t cells, int rank, MPI_Comm cart) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    int nonfinite = 0;
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal) \
        reduction(|:nonfinite) if(cells >= 16384)
    for (size_t i = 0; i < cells; ++i) {
        const double val = field[i];
        if (std::isfinite(val)) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        } else {
            nonfinite = 1;
        }
    }

    int anyNonfinite;
    double globalMin, globalMax;
    MPI_Reduce(&nonfinite, &anyNonfinite, 1, MPI_INT, MPI_MAX, 0, cart);
    MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
    MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank != 0) return true;
    if (anyNonfinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 10.0 || globalMin < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
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

int simulate(Extent grid, Extent ranks, int iterations, bool validate, bool printResults,
             int rank, MPI_Comm cart) {
    MPI_Comm node;
    MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("No CUDA device available");
    checkCuda(cudaSetDevice(localRank % devices));

    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    const Box box = rankBox(grid, ranks, {coords[2], coords[1], coords[0]});
    const size_t cells = box.size.volume();
    const size_t bytes = cells * sizeof(double);

    double *concentration, *mu;
    checkCuda(cudaMalloc(&concentration, bytes));
    checkCuda(cudaMalloc(&mu, bytes));
    Stencils stencils(box.size, cart);

    if (rank == 0) printf("Initializing concentration field...\n");
    std::unique_ptr<double[]> host(new double[cells]);
    initializeConcentration(host.get(), grid, box);
    checkCuda(cudaMemcpy(concentration, host.get(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());
    if (!validate && !printResults) host.reset();
    stencils.prepare(concentration, mu, iterations);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        stencils.step(concentration, mu);
    }
    stencils.synchronize();
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double updates = double(grid.volume()) * std::max(iterations, 0);
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
    }

    if (host) checkCuda(cudaMemcpy(host.get(), concentration, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(mu));
    checkCuda(cudaFree(concentration));
    if (printResults) printGlobalResult(host.get(), grid, ranks, box, rank, cart);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(host.get(), cells, rank, cart);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int run(int argc, char** argv) {
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    Extent grid{64, 0, 0};
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            grid.x = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            grid.y = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            grid.z = atoi(argv[++i]);
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

    if (grid.y == 0) grid.y = grid.x;
    if (grid.z == 0) grid.z = grid.x;
    constexpr size_t maxCells = std::numeric_limits<ptrdiff_t>::max() / sizeof(double);
    if (grid.x <= 0 || grid.y <= 0 || grid.z <= 0 ||
        size_t(grid.x) > maxCells / grid.y / grid.z) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", grid.x, grid.y, grid.z);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int dims[3];
    if (rank == 0) {
        const Extent ranks = processGrid(grid, worldSize);
        dims[0] = ranks.z;
        dims[1] = ranks.y;
        dims[2] = ranks.x;
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    const int periods[3]{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    int status = 0;
    if (cart != MPI_COMM_NULL) {
        status = simulate(grid, {dims[2], dims[1], dims[0]}, iterations, validate, printResults, rank, cart);
        MPI_Comm_free(&cart);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not support MPI_THREAD_FUNNELED");
    int status;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        fail("Host error: ", error.what());
    }
    MPI_Finalize();
    return status;
}
