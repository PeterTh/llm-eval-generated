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

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void initialize(double* c, size_t plane, size_t localNz, size_t globalZ0, size_t volume) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = plane * localNz;
    if (i < n) {
        const size_t globalId = (globalZ0 + i / plane) * plane + i % plane;
        c[plane + i] = -1.0 + 2.0 * (((globalId + 1) * 1299709ULL) % volume) / static_cast<double>(volume);
    }
}

__global__ void chemicalPotential(const double* c, double* mu, size_t nx, size_t ny, size_t localNz,
                                  double gamma, double eAA, double eBB, double eAB) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = nx * ny, n = plane * localNz;
    if (i >= n) return;
    const size_t p = plane + i, x = i % nx, y = (i / nx) % ny;
    const size_t xp = x + (x + 1 < nx), xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), ym = y - (y > 0);
    const double v = c[p];
    const double lap = c[p - x + xp] + c[p - x + xm] + c[p - y * nx + yp * nx] + c[p - y * nx + ym * nx]
                     + c[p + plane] + c[p - plane] - 6.0 * v;
    mu[p] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB) + 3.0 * v + v * v * v - gamma * lap;
}

__global__ void update(const double* cold, const double* mu, double* cnew, size_t nx, size_t ny, size_t localNz,
                       double dtD) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = nx * ny, n = plane * localNz;
    if (i >= n) return;
    const size_t p = plane + i, x = i % nx, y = (i / nx) % ny;
    const size_t xp = x + (x + 1 < nx), xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), ym = y - (y > 0);
    const double v = mu[p];
    const double lap = mu[p - x + xp] + mu[p - x + xm] + mu[p - y * nx + yp * nx] + mu[p - y * nx + ym * nx]
                     + mu[p + plane] + mu[p - plane] - 6.0 * v;
    cnew[p] = cold[p] + dtD * lap;
}

static void exchangeHalos(double* d, size_t plane, size_t localNz, int rank, int ranks, std::vector<double>& sendLo,
                          std::vector<double>& sendHi, std::vector<double>& recvLo, std::vector<double>& recvHi) {
    CUDA_CHECK(cudaMemcpy(sendLo.data(), d + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sendHi.data(), d + localNz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Sendrecv(sendLo.data(), static_cast<int>(plane), MPI_DOUBLE, rank ? rank - 1 : MPI_PROC_NULL, 17,
                 recvHi.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL, 17,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendHi.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL, 18,
                 recvLo.data(), static_cast<int>(plane), MPI_DOUBLE, rank ? rank - 1 : MPI_PROC_NULL, 18,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Physical boundaries retain the nearest interior value, exactly matching the original clamped stencil.
    CUDA_CHECK(cudaMemcpy(d, rank ? recvLo.data() : sendLo.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d + (localNz + 1) * plane, rank + 1 < ranks ? recvHi.data() : sendHi.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
}

static void usage(const char* p) { std::printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || ranks > static_cast<int>(nz) || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Invalid grid or more MPI ranks than z planes\\n"); MPI_Finalize(); return 1;
    }
    // Map ranks by node, rather than global rank, so every node uses its local GPUs.
    MPI_Comm localComm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank; MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\\n"); MPI_Finalize(); return 1; }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < extra);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny, localN = plane * localNz, globalN = plane * nz;
    if (!rank) std::printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\\nGrid size: %zu x %zu x %zu, ranks: %d, OpenMP threads/rank: %d\\nTime steps: %d\\n", nx, ny, nz, ranks, omp_get_max_threads(), iterations);
    double *cold, *cnew, *mu; CUDA_CHECK(cudaMalloc(&cold, (localN + 2 * plane) * sizeof(double))); CUDA_CHECK(cudaMalloc(&cnew, (localN + 2 * plane) * sizeof(double))); CUDA_CHECK(cudaMalloc(&mu, (localN + 2 * plane) * sizeof(double)));
    const int threads = 256, blocks = static_cast<int>((localN + threads - 1) / threads);
    initialize<<<blocks, threads>>>(cold, plane, localNz, z0, globalN); CUDA_CHECK(cudaGetLastError());
    std::vector<double> sendLo(plane), sendHi(plane), recvLo(plane), recvHi(plane), local(localN);
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, localNz, rank, ranks, sendLo, sendHi, recvLo, recvHi);
        chemicalPotential<<<blocks, threads>>>(cold, mu, nx, ny, localNz, .5, -(2.0/9.0), -(2.0/9.0), 2.0/9.0); CUDA_CHECK(cudaGetLastError());
        exchangeHalos(mu, plane, localNz, rank, ranks, sendLo, sendHi, recvLo, recvHi);
        update<<<blocks, threads>>>(cold, mu, cnew, nx, ny, localNz, .01); CUDA_CHECK(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); const double elapsed = MPI_Wtime() - start, elapsedMax = [&] { double v; MPI_Reduce(&elapsed, &v, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); return v; }();
    CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, localN * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<int> counts, offsets; std::vector<double> all;
    if (!rank) { counts.resize(ranks); offsets.resize(ranks); for (int r=0; r<ranks; ++r) { const size_t lz=base+(static_cast<size_t>(r)<extra); counts[r]=static_cast<int>(lz*plane); offsets[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),extra))*plane); } if (validate || printResults) all.resize(globalN); }
    if (validate || printResults)
        MPI_Gatherv(local.data(), static_cast<int>(localN), MPI_DOUBLE, rank ? nullptr : all.data(), rank ? nullptr : counts.data(), rank ? nullptr : offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int ok = 1;
    if (!rank) {
        std::printf("Computation time: %.3f ms\\nPerformance: %.3f MCellUpdates/s\\n", elapsedMax * 1000.0, static_cast<double>(globalN) * iterations / elapsedMax / 1e6);
        if (printResults) print_results(all, "Concentration");
        if (validate) { double lo=std::numeric_limits<double>::infinity(), hi=-lo; int finite=1;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:finite)
            for (long long i=0;i<static_cast<long long>(all.size());++i) { finite &= std::isfinite(all[i]); lo=std::min(lo,all[i]); hi=std::max(hi,all[i]); }
            std::printf("Concentration range: [%.6f, %.6f]\\n", lo, hi); ok = finite && hi <= 10.0 && lo >= -10.0; std::printf("Validation: %s\\n", ok ? "PASSED" : "FAILED"); }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD); CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu)); MPI_Comm_free(&localComm); MPI_Finalize(); return ok ? 0 : 1;
}
