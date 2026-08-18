#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(e_));                                   \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

inline constexpr size_t idx3(size_t x, size_t y, size_t z,
                             size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

__global__ void stencilKernel(const Real* __restrict__ in,
                              Real* __restrict__ out,
                              size_t nx, size_t ny, size_t localNz,
                              size_t globalZ0, size_t globalNz,
                              size_t firstZ, size_t lastZ) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = size_t(blockIdx.z) * blockDim.z + threadIdx.z + firstZ;
    if (x >= nx - 1 || y >= ny - 1 || z > lastZ || z > localNz) return;
    const size_t globalZ = globalZ0 + z - 1;
    if (globalZ == 0 || globalZ + 1 == globalNz) return;
    const size_t i = (z * ny + y) * nx + x;
    const size_t plane = nx * ny;
    out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx]
              + in[i-plane] + in[i+plane]) * (Real(1) / Real(7));
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny,
                          size_t localNz, size_t globalZ0, size_t globalNz,
                          size_t firstZ, size_t lastZ, cudaStream_t stream) {
    if (firstZ > lastZ || nx < 3 || ny < 3 || globalNz < 3) return;
    const dim3 block(32, 4, 2);
    const dim3 grid((unsigned)((nx - 2 + block.x - 1) / block.x),
                    (unsigned)((ny - 2 + block.y - 1) / block.y),
                    (unsigned)((lastZ - firstZ + 1 + block.z - 1) / block.z));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, localNz,
                                              globalZ0, globalNz, firstZ, lastZ);
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n", p);
    std::printf("  -x <num>  Grid size in X (default: 128)\n"
                "  -y <num>  Grid size in Y (default: X)\n"
                "  -z <num>  Grid size in Z (default: X)\n"
                "  -i <num>  Iterations (default: 10)\n"
                "  -v        Validate result\n"
                "  -r        Print results for external validation\n"
                "  -h        Show help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    bool argsOk = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else argsOk = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 || size_t(ranks) > nz) argsOk = false;
    if (rank == 0 && (help || !argsOk)) printUsage(argv[0]);
    if (help || !argsOk) { MPI_Finalize(); return help ? 0 : 1; }
    if (nx > std::numeric_limits<size_t>::max() / ny || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions overflow size_t\n");
        MPI_Finalize(); return 1;
    }

    // Block distribution; the first remainder ranks own one extra Z plane.
    const size_t base = nz / size_t(ranks), rem = nz % size_t(ranks);
    const size_t localNz = base + (size_t(rank) < rem);
    const size_t globalZ0 = size_t(rank) * base + std::min(size_t(rank), rem);
    const size_t plane = nx * ny, localElems = (localNz + 2) * plane;

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (rank == 0) std::fprintf(stderr, "No CUDA devices found\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<Real> host(localElems, 0.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 1; lz <= localNz; ++lz)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = ((globalZ0 + lz - 1) * ny + y) * nx + x;
                host[idx3(x, y, lz, nx, ny)] = Real(globalIdx % 19);
            }

    Real *dA = nullptr, *dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dB, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dA, host.data(), localElems * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, host.data(), localElems * sizeof(Real), cudaMemcpyHostToDevice));
    Real *sendLo = nullptr, *sendHi = nullptr, *recvLo = nullptr, *recvHi = nullptr;
    CUDA_CHECK(cudaMallocHost(&sendLo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendHi, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvHi, plane * sizeof(Real)));
    cudaStream_t computeStream, copyStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&copyStream));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", ranks, omp_get_max_threads());
        std::printf("Validation: %s\nRunning stencil computation...\n", validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* in = dA; Real* out = dB;
    for (int iter = 0; iter < iterations; ++iter) {
        const int lo = rank - 1, hi = rank + 1;
        // Stage contiguous boundary planes while the GPU computes planes that
        // cannot depend on incoming halos.
        if (lo >= 0) CUDA_CHECK(cudaMemcpyAsync(sendLo, in + plane, plane*sizeof(Real), cudaMemcpyDeviceToHost, copyStream));
        if (hi < ranks) CUDA_CHECK(cudaMemcpyAsync(sendHi, in + localNz*plane, plane*sizeof(Real), cudaMemcpyDeviceToHost, copyStream));
        launchStencil(in, out, nx, ny, localNz, globalZ0, nz, 2,
                      localNz > 1 ? localNz - 1 : 0, computeStream);
        CUDA_CHECK(cudaStreamSynchronize(copyStream));
        MPI_Request req[4]; int nr = 0;
        if (lo >= 0) {
            MPI_Irecv(recvLo, int(plane), MPI_DOUBLE, lo, 101, MPI_COMM_WORLD, &req[nr++]);
            MPI_Isend(sendLo, int(plane), MPI_DOUBLE, lo, 102, MPI_COMM_WORLD, &req[nr++]);
        }
        if (hi < ranks) {
            MPI_Irecv(recvHi, int(plane), MPI_DOUBLE, hi, 102, MPI_COMM_WORLD, &req[nr++]);
            MPI_Isend(sendHi, int(plane), MPI_DOUBLE, hi, 101, MPI_COMM_WORLD, &req[nr++]);
        }
        if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        if (lo >= 0) CUDA_CHECK(cudaMemcpyAsync(in, recvLo, plane*sizeof(Real), cudaMemcpyHostToDevice, copyStream));
        if (hi < ranks) CUDA_CHECK(cudaMemcpyAsync(in + (localNz+1)*plane, recvHi, plane*sizeof(Real), cudaMemcpyHostToDevice, copyStream));
        CUDA_CHECK(cudaStreamSynchronize(copyStream));
        launchStencil(in, out, nx, ny, localNz, globalZ0, nz, 1, 1, computeStream);
        if (localNz > 1) launchStencil(in, out, nx, ny, localNz, globalZ0, nz,
                                       localNz, localNz, computeStream);
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        std::swap(in, out);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(host.data() + plane, in + plane, localNz*plane*sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<Real> finalGrid;
    std::vector<int> counts, displs;
    if (rank == 0) {
        finalGrid.resize(nx*ny*nz); counts.resize(ranks); displs.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t rn = base + (size_t(r) < rem);
            const size_t rz = size_t(r)*base + std::min(size_t(r), rem);
            counts[r] = int(rn*plane); displs[r] = int(rz*plane);
        }
    }
    MPI_Gatherv(host.data()+plane, int(localNz*plane), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        const double updates = double((nx-2)*(ny-2)*(nz-2))*iterations;
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",
                    elapsed*1000.0, elapsed > 0 ? updates/elapsed/1e6 : 0.0);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            bool ok = true; Real minVal = finalGrid[0], maxVal = finalGrid[0];
#pragma omp parallel for reduction(&&:ok) reduction(min:minVal) reduction(max:maxVal)
            for (size_t i = 0; i < finalGrid.size(); ++i) {
                ok = ok && std::isfinite(finalGrid[i]);
                minVal = std::min(minVal, finalGrid[i]); maxVal = std::max(maxVal, finalGrid[i]);
            }
            ok = ok && minVal >= -1e6 && maxVal <= 1e6;
            std::printf("Value range: [%.6f, %.6f]\nValidation: %s\n",
                        minVal, maxVal, ok ? "PASSED" : "FAILED");
            rc = ok ? 0 : 1;
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaStreamDestroy(computeStream)); CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaFreeHost(sendLo)); CUDA_CHECK(cudaFreeHost(sendHi));
    CUDA_CHECK(cudaFreeHost(recvLo)); CUDA_CHECK(cudaFreeHost(recvHi));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    MPI_Finalize();
    return rc;
}
