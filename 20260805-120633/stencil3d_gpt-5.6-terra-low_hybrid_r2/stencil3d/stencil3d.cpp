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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } \
} while (0)
#define MPI_CHECK(call) do { const int e = (call); if (e != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e); } \
} while (0)

// z is a local plane number.  Planes 0 and localNz+1 are read-only halos.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              size_t nx, size_t ny, int zFirst, int zLast) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = zFirst + static_cast<int>(blockIdx.z);
    if (x >= nx || y >= ny || z > zLast) return;
    const size_t plane = nx * ny;
    const size_t i = static_cast<size_t>(z) * plane + y * nx + x;
    if (x == 0 || y == 0 || x + 1 == nx || y + 1 == ny) out[i] = in[i];
    else out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx]
                   + in[i - plane] + in[i + plane]) * (1.0 / 7.0);
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny,
                          int first, int last, cudaStream_t stream) {
    if (first > last) return;
    const dim3 block(32, 8, 1);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    static_cast<unsigned>(last - first + 1));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, first, last);
    CUDA_CHECK(cudaGetLastError());
}

static void initializeGlobal(std::vector<Real>& grid, size_t nx, size_t ny, size_t nz) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                grid[i] = static_cast<Real>(i % 19);
            }
}

static bool validateResult(const std::vector<Real>& grid) {
    Real lo = grid.front(), hi = grid.front();
    for (Real v : grid) {
        if (!std::isfinite(v)) { printf("Validation failed: found NaN or Inf value\n"); return false; }
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    printf("Value range: [%.6f, %.6f]\n", lo, hi);
    if (hi > 1e6 || lo < -1e6) { printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

static void usage(const char* p) {
    printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 128, ny = 0, nz = 0; int iterations = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || static_cast<size_t>(ranks) > nz - 2) {
        if (!rank) fprintf(stderr, "Grid dimensions must be >= 3, iterations >= 0, and MPI ranks <= z-2.\n");
        MPI_Finalize(); return 1;
    }
    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    MPI_Comm nodeComm; MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm));
    int localRank; MPI_Comm_rank(nodeComm, &localRank); MPI_Comm_free(&nodeComm);
    if (!deviceCount) { if (!rank) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t interiorZ = nz - 2, base = interiorZ / ranks, extra = interiorZ % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < extra);
    const size_t firstGlobalZ = 1 + static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny, localElements = (localNz + 2) * plane;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max()) || localNz * plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) fprintf(stderr, "Grid is too large for this MPI count interface.\n"); MPI_Abort(MPI_COMM_WORLD, 4);
    }
    std::vector<Real> initial(localElements);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < localNz + 2; ++lz)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = firstGlobalZ + lz - 1;
                initial[idx3(x, y, lz, nx, ny)] = static_cast<Real>(idx3(x, y, gz, nx, ny) % 19);
            }
    Real *a, *b; CUDA_CHECK(cudaMalloc(&a, localElements * sizeof(Real))); CUDA_CHECK(cudaMalloc(&b, localElements * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(a, initial.data(), localElements * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, initial.data(), localElements * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Host staging makes the MPI path correct on both CUDA-aware and ordinary MPI builds.
    Real *sendLow, *sendHigh, *recvLow, *recvHigh;
    CUDA_CHECK(cudaMallocHost(&sendLow, plane * sizeof(Real))); CUDA_CHECK(cudaMallocHost(&sendHigh, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLow, plane * sizeof(Real))); CUDA_CHECK(cudaMallocHost(&recvHigh, plane * sizeof(Real)));

    if (!rank) { printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations); }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        Real *in = (iter & 1) ? b : a, *out = (iter & 1) ? a : b;
        CUDA_CHECK(cudaMemcpyAsync(sendLow, in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sendHigh, in + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Request req[4]; int nreq = 0;
        if (rank) { MPI_CHECK(MPI_Irecv(recvLow, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 17, MPI_COMM_WORLD, &req[nreq++]));
                    MPI_CHECK(MPI_Isend(sendLow, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 18, MPI_COMM_WORLD, &req[nreq++])); }
        if (rank + 1 < ranks) { MPI_CHECK(MPI_Irecv(recvHigh, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 18, MPI_COMM_WORLD, &req[nreq++]));
                                MPI_CHECK(MPI_Isend(sendHigh, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 17, MPI_COMM_WORLD, &req[nreq++])); }
        launchStencil(in, out, nx, ny, 2, static_cast<int>(localNz) - 1, stream);
        MPI_CHECK(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE));
        if (rank) CUDA_CHECK(cudaMemcpyAsync(in, recvLow, plane * sizeof(Real), cudaMemcpyHostToDevice, stream));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(in + (localNz + 1) * plane, recvHigh, plane * sizeof(Real), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        launchStencil(in, out, nx, ny, 1, 1, stream);
        if (localNz > 1) launchStencil(in, out, nx, ny, static_cast<int>(localNz), static_cast<int>(localNz), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto stop = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(stop - start).count(), maxElapsed = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    Real* finalDevice = (iterations & 1) ? b : a;
    std::vector<Real> local(localNz * plane);
    CUDA_CHECK(cudaMemcpy(local.data(), finalDevice + plane, local.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<int> counts, displs; std::vector<Real> finalGrid;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r = 0; r < ranks; ++r) { const size_t n = base + (static_cast<size_t>(r) < extra); counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((1 + static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * plane); } finalGrid.resize(nx * ny * nz); initializeGlobal(finalGrid, nx, ny, nz); }
    MPI_CHECK(MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank ? nullptr : finalGrid.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (!rank) { printf("Computation time: %.3f ms\n", maxElapsed * 1000.0); const double updates = static_cast<double>((nx-2)*(ny-2)*(nz-2))*iterations; printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6); if (printResults) print_results(finalGrid, "Grid"); if (validate) printf("Validation: %s\n", validateResult(finalGrid) ? "PASSED" : "FAILED"); }
    CUDA_CHECK(cudaFreeHost(sendLow)); CUDA_CHECK(cudaFreeHost(sendHigh)); CUDA_CHECK(cudaFreeHost(recvLow)); CUDA_CHECK(cudaFreeHost(recvHigh));
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    MPI_Finalize(); return 0;
}
