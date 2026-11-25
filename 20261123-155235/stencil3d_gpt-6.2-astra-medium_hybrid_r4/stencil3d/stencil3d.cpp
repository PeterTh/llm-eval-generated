#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

static void fail(const char* message) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t error) {
    if (error != cudaSuccess) fail(cudaGetErrorString(error));
}

// Each warp covers adjacent x coordinates. Reuse the two preceding z-plane
// loads in registers as the thread advances along its column.
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t first, size_t end) {
    const size_t plane = nx * ny;
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (!x || x >= nx - 1) return;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t z = first + size_t(blockIdx.z) * 8; z < end; z += size_t(gridDim.z) * 8) {
            size_t i = z * plane + y * nx + x;
            Real bottom = input[i - plane], center = input[i];
            for (int k = 0; k < 8 && z + k < end; ++k, i += plane) {
                Real top = input[i + plane];
                output[i] = (center + input[i-1] + input[i+1] + input[i-nx] +
                             input[i+nx] + bottom + top) / 7.0;
                bottom = center;
                center = top;
            }
        }
    }
}

static void update(const Real* in, Real* out, size_t nx, size_t ny,
                   size_t first, size_t end, cudaStream_t stream) {
    if (first >= end || nx < 3 || ny < 3) return;
    dim3 grid((nx + 30) / 32, std::min<size_t>((ny + 1) / 4, 65535),
              std::min<size_t>((end - first + 7) / 8, 65535));
    stencil<<<grid, dim3(32, 4), 0, stream>>>(in, out, nx, ny, first, end);
    check(cudaGetLastError());
}

struct Halo {
    int peer;
    size_t face, ghost;
    Real *send = nullptr, *recv = nullptr;
};

// Chunk messages so plane and slab sizes are independent of MPI's int count.
static void post(Real* ptr, size_t count, int peer, bool send,
                 std::vector<MPI_Request>& requests) {
    while (count) {
        int n = static_cast<int>(std::min<size_t>(count, INT_MAX));
        requests.emplace_back();
        if (send) MPI_Isend(ptr, n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, &requests.back());
        else MPI_Irecv(ptr, n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, &requests.back());
        count -= n;
        ptr += n;
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
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    int rank, ranks, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > limit / ny || nx * ny > limit / nz)
        fail("Invalid grid dimensions");
    const size_t plane = nx * ny;
    const int active = static_cast<int>(std::min<size_t>(nz, ranks));
    const size_t base = nz / active, extra = nz % active;
    auto offset = [&](int r) { return base * r + std::min<size_t>(r, extra); };
    const size_t depth = rank < active ? offset(rank + 1) - offset(rank) : 0;

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\nInitializing grid...\n",
               iterations, validate ? "enabled" : "disabled");
    }

    Real *in = nullptr, *out = nullptr;
    cudaStream_t compute{}, transfer{};
    cudaEvent_t ready{};
    std::vector<Halo> halos;
    std::vector<MPI_Request> requests;
    bool direct = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    direct = MPIX_Query_cuda_support() != 0;
#endif
    // A short run can generate its entire dependency region at startup.
    // Limit the extra depth per face to half a slab to bound redundant work.
    const bool independent = active > 1 && iterations > 1 && nx > 2 && ny > 2 &&
                             static_cast<size_t>(iterations) <= base / 2;
    const size_t startZ = depth ? offset(rank) : 0;
    const size_t lower = independent ? std::min<size_t>(startZ, iterations) : 1;
    const size_t upper = independent ? std::min<size_t>(nz - startZ - depth, iterations) : 1;
    const bool exchange = !independent && active > 1 && nx > 2 && ny > 2 && nz > 2;
    if (depth) {
        int devices;
        check(cudaGetDeviceCount(&devices));
        if (!devices) fail("A CUDA device is required");
        check(cudaSetDevice(localRank % devices));
        if (depth + lower + upper > limit / plane) fail("Local slab is too large");
        size_t cells = (depth + lower + upper) * plane;
        check(cudaMalloc(&in, cells * sizeof(Real)));
        check(cudaMalloc(&out, cells * sizeof(Real)));
        std::vector<Real> initial(cells);
        const size_t localBegin = !independent && startZ == 0 ? plane : 0;
        const size_t localEnd = cells - (!independent && rank + 1 == active ? plane : 0);
        const size_t globalBegin = startZ ? (startZ - lower) * plane : 0;
        #pragma omp parallel for schedule(static)
        for (size_t i = localBegin; i < localEnd; ++i)
            initial[i] = (globalBegin + (i - localBegin)) % 19;
        check(cudaMemcpy(in, initial.data(), cells * sizeof(Real), cudaMemcpyHostToDevice));
        check(cudaMemcpy(out, in, cells * sizeof(Real), cudaMemcpyDeviceToDevice));
        check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        if (exchange && rank > 0) halos.push_back({rank - 1, 1, 0});
        if (exchange && rank + 1 < active) halos.push_back({rank + 1, depth, depth + 1});
        for (auto& h : halos) {
            if (!direct && iterations > 1) {
                check(cudaMallocHost(&h.send, plane * sizeof(Real)));
                check(cudaMallocHost(&h.recv, plane * sizeof(Real)));
            }
        }
        check(cudaDeviceSynchronize());
    }

    if (!rank) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    const size_t first = rank == 0 ? 2 : 1;
    const size_t end = depth + (rank + 1 != active);
    for (int iter = 0; depth && iter < iterations; ++iter) {
        if (independent) {
            // At step t only planes within (iterations - t) of the owned
            // slab can affect its final values. Keep physical faces fixed.
            size_t remaining = static_cast<size_t>(iterations - iter - 1);
            size_t beginZ = startZ - std::min(startZ, remaining);
            size_t endZ = startZ + depth + std::min(nz - startZ - depth, remaining);
            size_t origin = startZ - lower;
            update(in, out, nx, ny, std::max<size_t>(beginZ, 1) - origin,
                   std::min(endZ, nz - 1) - origin, compute);
        } else if (halos.empty() || iter + 1 == iterations) {
            update(in, out, nx, ny, first, end, compute);
        } else {
            // Compute outgoing faces first. The remaining planes can run while
            // MPI progresses and while the transfer stream stages halo data.
            size_t innerFirst = first, innerEnd = end;
            for (auto& h : halos) {
                if (h.face >= innerFirst && h.face < innerEnd) {
                    update(in, out, nx, ny, h.face, h.face + 1, compute);
                    if (h.face == innerFirst) ++innerFirst;
                    else --innerEnd;
                }
            }
            check(cudaEventRecord(ready, compute));
            check(cudaStreamWaitEvent(transfer, ready));
            update(in, out, nx, ny, innerFirst, innerEnd, compute);
            requests.clear();
            for (auto& h : halos) {
                if (!direct) check(cudaMemcpyAsync(h.send, out + h.face * plane,
                    plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            check(cudaStreamSynchronize(transfer));
            // Previous reads of the receive buffers and current face writes
            // are complete before MPI takes ownership of these buffers.
            for (auto& h : halos)
                post(direct ? out + h.ghost * plane : h.recv, plane, h.peer, false, requests);
            for (auto& h : halos)
                post(direct ? out + h.face * plane : h.send, plane, h.peer, true, requests);
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (!direct) {
                for (auto& h : halos)
                    check(cudaMemcpyAsync(out + h.ghost * plane, h.recv, plane * sizeof(Real),
                                          cudaMemcpyHostToDevice, transfer));
                check(cudaEventRecord(ready, transfer));
                check(cudaStreamWaitEvent(compute, ready));
            }
        }
        std::swap(in, out);
    }
    if (depth) check(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        double updates = nx > 2 && ny > 2 && nz > 2 ?
            double(nx - 2) * (ny - 2) * (nz - 2) * std::max(iterations, 0) : 0;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds ? updates / seconds / 1e6 : 0);
    }

    std::vector<Real> host;
    const size_t owned = depth * plane;
    if (depth && (printResults || validate)) {
        host.resize(!rank && printResults ? nz * plane : owned);
        check(cudaMemcpy(host.data(), in + lower * plane, owned * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    if (depth) {
        for (auto& h : halos) {
            if (h.send) check(cudaFreeHost(h.send));
            if (h.recv) check(cudaFreeHost(h.recv));
        }
        check(cudaEventDestroy(ready));
        check(cudaStreamDestroy(transfer));
        check(cudaStreamDestroy(compute));
        check(cudaFree(out));
        check(cudaFree(in));
    }
    if (printResults && depth) {
        requests.clear();
        if (!rank) {
            for (int r = 1; r < active; ++r)
                post(host.data() + offset(r) * plane, (offset(r+1) - offset(r)) * plane,
                     r, false, requests);
        } else {
            post(host.data(), owned, 0, true, requests);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        if (!rank) print_results(host, "Grid");
    }
    int valid = 1;
    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
        int finite = 1;
        #pragma omp parallel for reduction(min:lo,finite) reduction(max:hi) schedule(static)
        for (size_t i = 0; i < owned; ++i) {
            finite = std::min(finite, int(std::isfinite(host[i])));
            lo = std::min(lo, host[i]);
            hi = std::max(hi, host[i]);
        }
        Real minVal, maxVal;
        MPI_Reduce(&lo, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&hi, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&finite, &valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (!rank) {
            printf("Validating result...\n");
            if (valid) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
            else printf("Validation failed: found NaN or Inf value\n");
            if (minVal < -1e6 || maxVal > 1e6) {
                printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
