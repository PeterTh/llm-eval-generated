#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

static void fail(const char* message) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

struct Slab {
    size_t begin, depth;
};

static Slab partition(size_t nz, int ranks, int rank) {
    size_t base = nz / ranks, extra = nz % ranks;
    return {rank * base + std::min<size_t>(rank, extra), base + (size_t(rank) < extra)};
}

// A warp spans contiguous x coordinates. Walk several planes per thread to
// reuse the two vertical neighbors in registers across successive updates.
constexpr unsigned tileDepth = 8;
__global__ void stencil(const Real* __restrict__ src, Real* __restrict__ dst,
                        size_t nx, size_t ny, size_t begin, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t z = begin + blockIdx.z * tileDepth; z < end;
             z += size_t(gridDim.z) * tileDepth) {
            size_t i = z * plane + y * nx + x;
            Real bottom = src[i - plane], center = src[i];
            for (unsigned k = 0; k < tileDepth && z + k < end; ++k, i += plane) {
                Real top = src[i + plane];
                dst[i] = (center + src[i-1] + src[i+1] + src[i-nx] +
                          src[i+nx] + bottom + top) / 7.0;
                bottom = center;
                center = top;
            }
        }
    }
}

static void update(const Real* src, Real* dst, size_t nx, size_t ny,
                   size_t begin, size_t end, cudaStream_t stream) {
    if (begin >= end) return;
    dim3 block(32, 4);
    dim3 grid((nx - 3) / block.x + 1,
              std::min<size_t>((ny - 3) / block.y + 1, 65535),
              std::min<size_t>((end - begin - 1) / tileDepth + 1, 65535));
    stencil<<<grid, block, 0, stream>>>(src, dst, nx, ny, begin, end);
    check(cudaGetLastError());
}

// Planes and result slabs can exceed the MPI int count limit.
static void postPlane(Real* data, size_t count, int peer, bool send,
                      std::vector<MPI_Request>& requests) {
    for (size_t i = 0; i < count;) {
        int n = std::min<size_t>(count - i, INT_MAX);
        requests.emplace_back();
        if (send) MPI_Isend(data + i, n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, &requests.back());
        else MPI_Irecv(data + i, n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, &requests.back());
        i += n;
    }
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

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED required");
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    if (!nx || !ny || !nz || ny > SIZE_MAX / nx || nz > SIZE_MAX / (nx * ny))
        fail("Invalid grid size");
    const size_t plane = nx * ny;
    const Slab slab = partition(nz, ranks, rank);
    if (slab.depth > SIZE_MAX - 2 || slab.depth + 2 > SIZE_MAX / sizeof(Real) / plane)
        fail("Local grid exceeds the addressable size");
    const int active = std::min<size_t>(ranks, nz);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    Real *input = nullptr, *output = nullptr, *staging = nullptr;
    cudaStream_t bulk{}, edge{};
    const size_t planeBytes = plane * sizeof(Real);
    const bool compute = slab.depth && nx > 2 && ny > 2 && nz > 2 && iterations > 0;
    const bool exchange = compute && active > 1 && iterations > 1;
    bool direct = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    direct = MPIX_Query_cuda_support() != 0;
#endif
    std::vector<Real> host;
    if (slab.depth) {
        int devices;
        check(cudaGetDeviceCount(&devices));
        if (devices == 0) fail("A CUDA device is required");
        check(cudaSetDevice(localRank % devices));
        check(cudaStreamCreateWithFlags(&bulk, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&edge, cudaStreamNonBlocking));
        size_t cells = (slab.depth + 2) * plane;
        host.resize(cells);
        // Initialize the first iteration's halos directly from global indices.
        size_t start = slab.begin ? 0 : plane;
        size_t stop = cells - (rank == active - 1 ? plane : 0);
        size_t global = (slab.begin ? slab.begin - 1 : 0) * plane;
        #pragma omp parallel for schedule(static)
        for (size_t i = start; i < stop; ++i) host[i] = (global + (i - start)) % 19;
        check(cudaMalloc(&input, cells * sizeof(Real)));
        check(cudaMalloc(&output, cells * sizeof(Real)));
        check(cudaMemcpyAsync(input, host.data(), cells * sizeof(Real), cudaMemcpyHostToDevice, bulk));
        check(cudaMemcpyAsync(output, input, cells * sizeof(Real), cudaMemcpyDeviceToDevice, bulk));
        // With a negative odd count the original selects its unused, zero-filled buffer.
        if (iterations < 0 && iterations % 2)
            check(cudaMemsetAsync(input, 0, cells * sizeof(Real), bulk));
        check(cudaStreamSynchronize(bulk));
        if (exchange && !direct) check(cudaMallocHost(&staging, 4 * planeBytes));
        if (!validate && !printResults) std::vector<Real>().swap(host);
    }

    if (rank == 0) printf("Running stencil computation...\n");
    const size_t begin = rank == 0 ? 2 : 1;
    const size_t end = slab.depth + 1 - (rank == active - 1);
    const int peers[2] = {rank - 1, rank + 1};
    const size_t faces[2] = {1, slab.depth};
    const size_t ghosts[2] = {0, slab.depth + 1};
    std::vector<MPI_Request> requests;
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();
    for (int it = 0; compute && it < iterations; ++it) {
        if (!exchange || it + 1 == iterations) {
            update(input, output, nx, ny, begin, end, bulk);
        } else {
            requests.clear();
            for (int side = 0; side < 2; ++side) {
                if (peers[side] < 0 || peers[side] >= active) continue;
                Real* recv = direct ? output + ghosts[side] * plane : staging + side * plane;
                postPlane(recv, plane, peers[side], false, requests);
            }
            if (begin <= 1 && 1 < end) update(input, output, nx, ny, 1, 2, edge);
            if (slab.depth > 1 && slab.depth >= begin && slab.depth < end)
                update(input, output, nx, ny, slab.depth, slab.depth + 1, edge);
            if (!direct) {
                for (int side = 0; side < 2; ++side)
                    if (peers[side] >= 0 && peers[side] < active)
                        check(cudaMemcpyAsync(staging + (2 + side) * plane, output + faces[side] * plane,
                                              planeBytes, cudaMemcpyDeviceToHost, edge));
            }
            // Launch the faces first so their transfers can start while the
            // longer bulk kernel is still occupying the GPU.
            update(input, output, nx, ny, std::max(begin, size_t(2)),
                   std::min(end, slab.depth), bulk);
            check(cudaStreamSynchronize(edge));
            for (int side = 0; side < 2; ++side) {
                if (peers[side] < 0 || peers[side] >= active) continue;
                Real* send = direct ? output + faces[side] * plane : staging + (2 + side) * plane;
                postPlane(send, plane, peers[side], true, requests);
            }
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (!direct) {
                for (int side = 0; side < 2; ++side)
                    if (peers[side] >= 0 && peers[side] < active)
                        check(cudaMemcpyAsync(output + ghosts[side] * plane, staging + side * plane,
                                              planeBytes, cudaMemcpyHostToDevice, edge));
                // MPI may reuse these receive buffers at the next iteration.
                check(cudaStreamSynchronize(edge));
            }
            check(cudaStreamSynchronize(bulk));
        }
        std::swap(input, output);
    }
    if (slab.depth) check(cudaStreamSynchronize(bulk));
    double elapsed = MPI_Wtime() - startTime, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double updates = nx > 2 && ny > 2 && nz > 2 ?
            double(nx - 2) * (ny - 2) * (nz - 2) * std::max(iterations, 0) : 0;
        printf("Computation time: %.0f ms\n", duration * 1000);
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }

    size_t owned = slab.depth * plane;
    if (owned && (printResults || validate))
        check(cudaMemcpy(host.data(), input + plane, owned * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Reassemble only for the order-dependent reference hash and Kahan sum.
        if (rank == 0) {
            host.resize(nz * plane);
            for (int peer = 1; peer < active; ++peer) {
                Slab other = partition(nz, ranks, peer);
                for (size_t i = 0; i < other.depth * plane;) {
                    int n = std::min<size_t>(other.depth * plane - i, INT_MAX);
                    MPI_Recv(host.data() + other.begin * plane + i, n, MPI_DOUBLE, peer, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    i += n;
                }
            }
            print_results(host, "Grid");
        } else {
            for (size_t i = 0; i < owned;) {
                int n = std::min<size_t>(owned - i, INT_MAX);
                MPI_Send(host.data() + i, n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                i += n;
            }
        }
    }
    int valid = 1;
    if (validate) {
        Real low = std::numeric_limits<Real>::infinity(), high = -low;
        int finite = 1;
        #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(&:finite)
        for (size_t i = 0; i < owned; ++i) {
            low = std::min(low, host[i]);
            high = std::max(high, host[i]);
            finite &= std::isfinite(host[i]);
        }
        Real minVal, maxVal;
        MPI_Reduce(&low, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&high, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&finite, &valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                valid = minVal >= -1e6 && maxVal <= 1e6;
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (slab.depth) {
        if (staging) check(cudaFreeHost(staging));
        check(cudaFree(output));
        check(cudaFree(input));
        check(cudaStreamDestroy(edge));
        check(cudaStreamDestroy(bulk));
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
