#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Each thread walks a short Z column, reusing center and Z neighbors in registers.
// Both buffers already contain the fixed physical boundaries.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t first, size_t end) {
    const size_t plane = nx * ny;
    for (size_t y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
         y + 1 < ny; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t x = 1 + blockIdx.x * blockDim.x + threadIdx.x;
             x + 1 < nx; x += size_t(gridDim.x) * blockDim.x) {
            for (size_t z0 = first + size_t(blockIdx.z) * 8;
                 z0 < end; z0 += size_t(gridDim.z) * 8) {
                size_t i = z0 * plane + y * nx + x;
                Real bottom = in[i - plane], center = in[i];
                const size_t stop = z0 + 8 < end ? z0 + 8 : end;
                for (size_t z = z0; z < stop; ++z, i += plane) {
                    const Real top = in[i + plane];
                    Real sum = center + in[i - 1];
                    sum = sum + in[i + 1];
                    sum = sum + in[i - nx];
                    sum = sum + in[i + nx];
                    sum = sum + bottom;
                    sum = sum + top;
                    out[i] = __ddiv_rn(sum, 7.0);
                    bottom = center;
                    center = top;
                }
            }
        }
    }
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny,
                   size_t first, size_t end, cudaStream_t stream) {
    if (first >= end || nx < 3 || ny < 3) return;
    dim3 block(32, 4);
    dim3 grid(static_cast<unsigned>(std::min(size_t(65535), (nx - 2 + 31) / 32)),
              static_cast<unsigned>(std::min(size_t(65535), (ny - 2 + 3) / 4)),
              static_cast<unsigned>(std::min(size_t(65535), (end - first + 7) / 8)));
    stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, first, end);
    CUDA(cudaGetLastError());
}

// Chunk messages to avoid the MPI int-count limit, including for result output.
static void transfer(Real* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 20, comm);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 20, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                    !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long n = strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end || n > size_t(INT_MAX) ||
                (option == 'x' && n == 0)) {
                if (!worldRank) fprintf(stderr, "Invalid numeric argument: %s\n", value);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (option == 'x') nx = n;
            else if (option == 'y') ny = n;
            else if (option == 'z') nz = n;
            else iterations = static_cast<int>(n);
        } else {
            if (!worldRank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / sizeof(Real) / (nz + 2)) {
        if (!worldRank) fprintf(stderr, "Grid size overflow\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Ranks beyond the number of planes have no work, but still finalize collectively.
    const int ranks = static_cast<int>(std::min(nz, size_t(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA GPU is required on every participating node\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    const size_t plane = nx * ny;
    const size_t depth = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t offset = size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
    const size_t cells = (depth + 2) * plane;
    std::vector<Real> host(cells);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells; ++i) {
        const size_t z = i / plane;
        if ((z != 0 || offset != 0) && offset + z <= nz)
            host[i] = Real(((offset + z - 1) * plane + i % plane) % 19);
    }
    Real *a, *b, *staging = nullptr;
    CUDA(cudaMalloc(&a, cells * sizeof(Real)));
    CUDA(cudaMalloc(&b, cells * sizeof(Real)));
    CUDA(cudaMemcpy(a, host.data(), cells * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b, a, cells * sizeof(Real), cudaMemcpyDeviceToDevice));
    cudaStream_t bulk, halo;
    CUDA(cudaStreamCreateWithFlags(&bulk, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&halo, cudaStreamNonBlocking));
    if (ranks > 1) CUDA(cudaMallocHost(&staging, 4 * plane * sizeof(Real)));
    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    const size_t first = offset == 0 ? 2 : 1;
    const size_t end = offset + depth == nz ? depth : depth + 1;
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        printf("Validation: %s\nInitializing grid...\nRunning stencil computation...\n", validate ? "enabled" : "disabled");
    }
    std::vector<MPI_Request> requests;
    requests.reserve(4 * ((plane - 1) / INT_MAX + 1));
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launch(a, b, nx, ny, first, end, bulk);
        } else {
            // Post receives before packing, and compute planes independent of incoming halos.
            requests.clear();
            auto post = [&](Real* data, int peer, int tag, bool send) {
                if (peer == MPI_PROC_NULL) return;
                for (size_t p = 0; p < plane;) {
                    int count = static_cast<int>(std::min(plane - p, size_t(INT_MAX)));
                    MPI_Request req;
                    if (send) MPI_Isend(data + p, count, MPI_DOUBLE, peer, tag, comm, &req);
                    else MPI_Irecv(data + p, count, MPI_DOUBLE, peer, tag, comm, &req);
                    requests.push_back(req);
                    p += count;
                }
            };
            post(staging + 2 * plane, prev, 11, false);
            post(staging + 3 * plane, next, 10, false);
            launch(a, b, nx, ny, std::max(first, size_t(2)), std::min(end, depth), bulk);
            if (prev != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(staging, a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, halo));
            if (next != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(staging + plane, a + depth * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, halo));
            CUDA(cudaStreamSynchronize(halo));
            post(staging, prev, 10, true);
            post(staging + plane, next, 11, true);
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (prev != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(a, staging + 2 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, halo));
            if (next != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(a + (depth + 1) * plane, staging + 3 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, halo));
            if (first <= 1 && end > 1) launch(a, b, nx, ny, 1, 2, halo);
            if (depth > 1 && first <= depth && end > depth) launch(a, b, nx, ny, depth, depth + 1, halo);
            CUDA(cudaStreamSynchronize(halo));
        }
        // Single-rank launches are naturally ordered in the bulk stream.
        if (ranks > 1) CUDA(cudaStreamSynchronize(bulk));
        std::swap(a, b);
    }
    CUDA(cudaStreamSynchronize(bulk));
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    if (validate || printResults)
        CUDA(cudaMemcpy(host.data(), a + plane, depth * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (!rank) {
            std::vector<Real> global(nx * ny * nz);
            std::copy_n(host.data(), depth * plane, global.data());
            for (int p = 1; p < ranks; ++p) {
                size_t d = nz / ranks + (size_t(p) < nz % ranks);
                size_t off = size_t(p) * (nz / ranks) + std::min(size_t(p), nz % ranks);
                transfer(global.data() + off * plane, d * plane, p, false, comm);
            }
            print_results(global, "Grid");
        } else transfer(host.data(), depth * plane, 0, true, comm);
    }
    int valid = 1;
    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
        int bad = 0;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad) schedule(static)
        for (size_t i = 0; i < depth * plane; ++i) {
            const Real v = host[i];
            bad |= !std::isfinite(v);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        Real globalLo, globalHi;
        int globalBad;
        MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, comm);
        valid = !globalBad && globalLo >= -1e6 && globalHi <= 1e6;
        if (!rank) printf("Validating result...\nValue range: [%.6f, %.6f]\nValidation: %s\n", globalLo, globalHi, valid ? "PASSED" : "FAILED");
    }
    if (staging) CUDA(cudaFreeHost(staging));
    CUDA(cudaFree(a));
    CUDA(cudaFree(b));
    CUDA(cudaStreamDestroy(bulk));
    CUDA(cudaStreamDestroy(halo));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
