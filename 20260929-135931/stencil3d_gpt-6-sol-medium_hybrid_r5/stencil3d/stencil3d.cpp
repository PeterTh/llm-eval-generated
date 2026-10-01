#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

using Real = double;

static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void mpi(int e, const char* what) {
    if (e != MPI_SUCCESS) {
        char msg[MPI_MAX_ERROR_STRING]; int len = 0;
        MPI_Error_string(e, msg, &len);
        std::fprintf(stderr, "%s: %.*s\n", what, len, msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static size_t slabStart(int rank, size_t nz, int ranks) {
    return size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
}
static size_t slabDepth(int rank, size_t nz, int ranks) {
    return nz / ranks + (size_t(rank) < nz % ranks);
}

// Local planes start at 1; plane 0 and plane depth+1 hold neighboring halos.
__global__ static void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                               size_t nx, size_t ny, size_t nz, size_t firstZ,
                               size_t plane, size_t beginZ, size_t endZ) {
    size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = beginZ + blockIdx.z;
    if (x >= nx || y >= ny || z >= endZ) return;
    size_t i = z * plane + y * nx + x;
    size_t globalZ = firstZ + z - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        globalZ == 0 || globalZ + 1 == nz) {
        out[i] = in[i];
    } else {
        out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] +
                  in[i + nx] + in[i - plane] + in[i + plane]) / 7.0;
    }
}
static void launch(const Real* in, Real* out, size_t nx, size_t ny, size_t nz,
                   size_t firstZ, size_t plane, size_t beginZ, size_t endZ,
                   cudaStream_t stream) {
    if (beginZ >= endZ) return;
    dim3 block(32, 8);
    for (size_t z = beginZ; z < endZ;) {
        size_t next = z + std::min(endZ - z, size_t(65535));
        dim3 blocks(static_cast<unsigned>((nx + 31) / 32),
                    static_cast<unsigned>((ny + 7) / 8),
                    static_cast<unsigned>(next - z));
        stencil<<<blocks, block, 0, stream>>>(
            in, out, nx, ny, nz, firstZ, plane, z, next);
        gpu(cudaGetLastError(), "stencil kernel launch");
        z = next;
    }
}
static bool validateResult(const std::vector<Real>& grid) {
    Real lo = grid[0], hi = grid[0];
    for (Real v : grid) {
        if (!std::isfinite(v)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    std::printf("Value range: [%.6f, %.6f]\n", lo, hi);
    if (hi > 1e6 || lo < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}
static void usage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}
int main(int argc, char** argv) {
    mpi(MPI_Init(&argc, &argv), "MPI_Init");
    int worldRank, worldSize;
    mpi(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
    mpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, results = false;
    int argStatus = 0;
    for (int a = 1; a < argc; ++a) {
        if (!std::strcmp(argv[a], "-x") && a + 1 < argc) nx = std::strtoull(argv[++a], nullptr, 10);
        else if (!std::strcmp(argv[a], "-y") && a + 1 < argc) ny = std::strtoull(argv[++a], nullptr, 10);
        else if (!std::strcmp(argv[a], "-z") && a + 1 < argc) nz = std::strtoull(argv[++a], nullptr, 10);
        else if (!std::strcmp(argv[a], "-i") && a + 1 < argc) iterations = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "-v")) validate = true;
        else if (!std::strcmp(argv[a], "-r")) results = true;
        else if (!std::strcmp(argv[a], "-h")) { if (!worldRank) usage(argv[0]); argStatus = 2; }
        else { if (!worldRank) { std::printf("Unknown option: %s\n", argv[a]); usage(argv[0]); } argStatus = 1; }
    }
    if (argStatus) { MPI_Finalize(); return argStatus == 2 ? 0 : 1; }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > size_t(INT_MAX)) {
        if (!worldRank) std::fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize(); return 1;
    }
    int ranks = static_cast<int>(std::min(nz, size_t(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    mpi(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                       worldRank, &comm), "MPI_Comm_split");
    if (comm == MPI_COMM_NULL) { MPI_Finalize(); return 0; }
    int rank; mpi(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank(active)");
    MPI_Comm shared;
    mpi(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared), "MPI_Comm_split_type");
    int localRank, deviceCount = 0;
    mpi(MPI_Comm_rank(shared, &localRank), "MPI_Comm_rank(shared)");
    gpu(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (!deviceCount) { std::fprintf(stderr, "No CUDA GPU is available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    gpu(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    mpi(MPI_Comm_free(&shared), "MPI_Comm_free(shared)");

    size_t plane = nx * ny, depth = slabDepth(rank, nz, ranks);
    size_t firstZ = slabStart(rank, nz, ranks);
    if (depth > SIZE_MAX / plane - 2 || (depth + 2) * plane > SIZE_MAX / sizeof(Real)) {
        std::fprintf(stderr, "Local grid is too large\n"); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t planeBytes = plane * sizeof(Real);
    size_t localBytes = (depth + 2) * planeBytes;
    if (!rank) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\nInitializing grid...\n",
                    iterations, validate ? "enabled" : "disabled");
    }
    std::vector<Real> initial(depth * plane);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = Real((firstZ * plane + i) % 19);
    Real *in = nullptr, *out = nullptr;
    gpu(cudaMalloc(&in, localBytes), "cudaMalloc(input)");
    gpu(cudaMalloc(&out, localBytes), "cudaMalloc(output)");
    gpu(cudaMemcpy(in + plane, initial.data(), depth * planeBytes, cudaMemcpyHostToDevice), "upload initial grid");
    initial.clear(); initial.shrink_to_fit();
    Real *sendLo = nullptr, *sendHi = nullptr, *recvLo = nullptr, *recvHi = nullptr;
    if (ranks > 1) {
        gpu(cudaMallocHost(&sendLo, planeBytes), "cudaMallocHost(sendLo)");
        gpu(cudaMallocHost(&sendHi, planeBytes), "cudaMallocHost(sendHi)");
        gpu(cudaMallocHost(&recvLo, planeBytes), "cudaMallocHost(recvLo)");
        gpu(cudaMallocHost(&recvHi, planeBytes), "cudaMallocHost(recvHi)");
    }
    cudaStream_t compute, transfer;
    gpu(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "create compute stream");
    gpu(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking), "create transfer stream");
    mpi(MPI_Barrier(comm), "MPI_Barrier");
    if (!rank) std::printf("Running stencil computation...\n");
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launch(in, out, nx, ny, nz, firstZ, plane, 1, depth + 1, compute);
            gpu(cudaStreamSynchronize(compute), "synchronize stencil");
        } else {
            MPI_Request requests[4]; int n = 0;
            if (rank > 0) {
                mpi(MPI_Irecv(recvLo, int(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[n++]), "receive lower halo");
                gpu(cudaMemcpyAsync(sendLo, in + plane, planeBytes, cudaMemcpyDeviceToHost, transfer), "download lower face");
            }
            if (rank + 1 < ranks) {
                mpi(MPI_Irecv(recvHi, int(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[n++]), "receive upper halo");
                gpu(cudaMemcpyAsync(sendHi, in + depth * plane, planeBytes, cudaMemcpyDeviceToHost, transfer), "download upper face");
            }
            launch(in, out, nx, ny, nz, firstZ, plane, 2, depth, compute);
            gpu(cudaStreamSynchronize(transfer), "synchronize outgoing faces");
            if (rank > 0)
                mpi(MPI_Isend(sendLo, int(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[n++]), "send lower face");
            if (rank + 1 < ranks)
                mpi(MPI_Isend(sendHi, int(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[n++]), "send upper face");
            mpi(MPI_Waitall(n, requests, MPI_STATUSES_IGNORE), "MPI_Waitall(halos)");
            if (rank > 0)
                gpu(cudaMemcpyAsync(in, recvLo, planeBytes, cudaMemcpyHostToDevice, transfer), "upload lower halo");
            if (rank + 1 < ranks)
                gpu(cudaMemcpyAsync(in + (depth + 1) * plane, recvHi, planeBytes, cudaMemcpyHostToDevice, transfer), "upload upper halo");
            launch(in, out, nx, ny, nz, firstZ, plane, 1, 2, transfer);
            if (depth > 1) launch(in, out, nx, ny, nz, firstZ, plane, depth, depth + 1, transfer);
            gpu(cudaStreamSynchronize(transfer), "synchronize boundary stencil");
            gpu(cudaStreamSynchronize(compute), "synchronize interior stencil");
        }
        std::swap(in, out);
    }
    double localElapsed = MPI_Wtime() - start, elapsed = 0;
    mpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm), "MPI_Reduce(time)");
    if (!rank) {
        std::printf("Computation time: %ld ms\n", long(elapsed * 1000));
        double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) *
                         double(nz > 2 ? nz - 2 : 0) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", updates / elapsed / 1e6);
    }
    int status = 0;
    if (results || validate) {
        std::vector<Real> local(depth * plane);
        gpu(cudaMemcpy(local.data(), in + plane, depth * planeBytes, cudaMemcpyDeviceToHost), "download result");
        if (!rank) {
            std::vector<Real> global(nx * ny * nz);
            std::copy(local.begin(), local.end(), global.begin());
            for (int source = 1; source < ranks; ++source) {
                size_t offset = slabStart(source, nz, ranks) * plane;
                size_t count = slabDepth(source, nz, ranks) * plane;
                for (size_t done = 0; done < count;) {
                    int chunk = int(std::min(count - done, size_t(INT_MAX)));
                    mpi(MPI_Recv(global.data() + offset + done, chunk, MPI_DOUBLE, source, 2, comm,
                                 MPI_STATUS_IGNORE), "receive result");
                    done += chunk;
                }
            }
            if (results) print_results(global, "Grid");
            if (validate) {
                std::printf("Validating result...\n");
                bool valid = validateResult(global);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        } else {
            for (size_t done = 0; done < local.size();) {
                int chunk = int(std::min(local.size() - done, size_t(INT_MAX)));
                mpi(MPI_Send(local.data() + done, chunk, MPI_DOUBLE, 0, 2, comm), "send result");
                done += chunk;
            }
        }
    }
    mpi(MPI_Bcast(&status, 1, MPI_INT, 0, comm), "MPI_Bcast(status)");
    gpu(cudaStreamDestroy(compute), "destroy compute stream");
    gpu(cudaStreamDestroy(transfer), "destroy transfer stream");
    if (ranks > 1) {
        gpu(cudaFreeHost(sendLo), "free sendLo"); gpu(cudaFreeHost(sendHi), "free sendHi");
        gpu(cudaFreeHost(recvLo), "free recvLo"); gpu(cudaFreeHost(recvHi), "free recvHi");
    }
    gpu(cudaFree(in), "free input"); gpu(cudaFree(out), "free output");
    mpi(MPI_Comm_free(&comm), "MPI_Comm_free(active)");
    mpi(MPI_Finalize(), "MPI_Finalize");
    return status;
}
