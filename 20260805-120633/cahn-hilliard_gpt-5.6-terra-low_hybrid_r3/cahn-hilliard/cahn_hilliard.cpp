#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } \
} while (0)

__global__ void chemical_potential(const double* __restrict__ c, double* __restrict__ mu,
                                   size_t nx, size_t ny, size_t localNz, double gamma,
                                   double eAA, double eBB, double eAB) {
    const size_t plane = nx * ny, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = plane * localNz;
    if (i >= n) return;
    const size_t z = i / plane + 1, p = i % plane, y = p / nx, x = p % nx, q = z * plane + p;
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const double v = c[q];
    const double lap = c[z * plane + y * nx + xp] + c[z * plane + y * nx + xn] - 2.0 * v
                     + c[z * plane + yp * nx + x] + c[z * plane + yn * nx + x] - 2.0 * v
                     + c[(z + 1) * plane + p] + c[(z - 1) * plane + p] - 2.0 * v;
    mu[q] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
          + 3.0 * v + v * v * v - gamma * lap;
}

__global__ void update_concentration(const double* __restrict__ cold, const double* __restrict__ mu,
                                     double* __restrict__ cnew, size_t nx, size_t ny, size_t localNz,
                                     double dtD) {
    const size_t plane = nx * ny, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = plane * localNz;
    if (i >= n) return;
    const size_t z = i / plane + 1, p = i % plane, y = p / nx, x = p % nx, q = z * plane + p;
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const double v = mu[q];
    const double lap = mu[z * plane + y * nx + xp] + mu[z * plane + y * nx + xn] - 2.0 * v
                     + mu[z * plane + yp * nx + x] + mu[z * plane + yn * nx + x] - 2.0 * v
                     + mu[(z + 1) * plane + p] + mu[(z - 1) * plane + p] - 2.0 * v;
    cnew[q] = cold[q] + dtD * lap;
}

static void exchange_halos(double* d, size_t plane, size_t localNz, int rank, int ranks,
                           double* sendLo, double* sendHi, double* recvLo, double* recvHi) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaMemcpy(sendLo, d + plane, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sendHi, d + localNz * plane, bytes, cudaMemcpyDeviceToHost));
    const int lo = rank ? rank - 1 : MPI_PROC_NULL, hi = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(sendLo, (int)plane, MPI_DOUBLE, lo, 11, recvHi, (int)plane, MPI_DOUBLE, hi, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendHi, (int)plane, MPI_DOUBLE, hi, 12, recvLo, (int)plane, MPI_DOUBLE, lo, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (lo == MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(d, d + plane, bytes, cudaMemcpyDeviceToDevice));
    else CUDA_CHECK(cudaMemcpy(d, recvLo, bytes, cudaMemcpyHostToDevice));
    if (hi == MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(d + (localNz + 1) * plane, d + localNz * plane, bytes, cudaMemcpyDeviceToDevice));
    else CUDA_CHECK(cudaMemcpy(d + (localNz + 1) * plane, recvHi, bytes, cudaMemcpyHostToDevice));
}

static void usage(const char* p) {
    printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, printResults = false;
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
    if (!nx || !ny || !nz || iterations < 0 || (size_t)ranks > nz) {
        if (!rank) fprintf(stderr, "Grid dimensions must be positive and z must be at least the MPI rank count.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t plane = nx * ny, localNz = nz / ranks + ((size_t)rank < nz % ranks),
                 zStart = (nz / ranks) * rank + std::min((size_t)rank, nz % ranks), localN = plane * localNz;
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu, MPI ranks: %d\nTime steps: %d\n", nx, ny, nz, ranks, iterations); }

    std::vector<double> hInitial(localN);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < localNz; ++z) for (size_t p = 0; p < plane; ++p) {
        const size_t id = (zStart + z) * plane + p, vol = nx * ny * nz;
        hInitial[z * plane + p] = -1.0 + 2.0 * ((((id + 1) * 1299709) % vol) / (double)vol);
    }
    double *cold, *cnew, *mu, *sendLo, *sendHi, *recvLo, *recvHi;
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "No CUDA device is available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));
    CUDA_CHECK(cudaMalloc(&cold, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold + plane, hInitial.data(), localN * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaHostAlloc(&sendLo, plane * sizeof(double), cudaHostAllocDefault)); CUDA_CHECK(cudaHostAlloc(&sendHi, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recvLo, plane * sizeof(double), cudaHostAllocDefault)); CUDA_CHECK(cudaHostAlloc(&recvHi, plane * sizeof(double), cudaHostAllocDefault));
    const int threads = 256, blocks = (int)((localN + threads - 1) / threads);
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchange_halos(cold, plane, localNz, rank, ranks, sendLo, sendHi, recvLo, recvHi);
        chemical_potential<<<blocks, threads>>>(cold, mu, nx, ny, localNz, .5, -(2.0/9.0), -(2.0/9.0), 2.0/9.0); CUDA_CHECK(cudaGetLastError());
        exchange_halos(mu, plane, localNz, rank, ranks, sendLo, sendHi, recvLo, recvHi);
        update_concentration<<<blocks, threads>>>(cold, mu, cnew, nx, ny, localNz, .01); CUDA_CHECK(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> local(localN); CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, localN * sizeof(double), cudaMemcpyDeviceToHost));
    int bad = 0; double minv = std::numeric_limits<double>::infinity(), maxv = -minv;
    #pragma omp parallel for reduction(+:bad) reduction(min:minv) reduction(max:maxv)
    for (size_t i = 0; i < localN; ++i) { bad += !std::isfinite(local[i]); minv = std::min(minv, local[i]); maxv = std::max(maxv, local[i]); }
    int globalBad; double globalMin, globalMax; MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); MPI_Reduce(&minv, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); MPI_Reduce(&maxv, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<int> counts, displs; std::vector<double> result;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r=0; r<ranks; ++r) { size_t lz=nz/ranks+((size_t)r<nz%ranks); counts[r]=(int)(lz*plane); displs[r]=(int)(((nz/ranks)*r+std::min((size_t)r,nz%ranks))*plane); } if (printResults) result.resize(nx*ny*nz); }
    if (printResults) MPI_Gatherv(local.data(), (int)localN, MPI_DOUBLE, rank ? nullptr : result.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int valid = !globalBad && globalMin >= -10. && globalMax <= 10.; int globalValid = valid;
    MPI_Bcast(&globalValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) { printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", maxElapsed*1000., (double)(nx*ny*nz)*iterations/maxElapsed/1e6); if (printResults) print_results(result, "Concentration"); if (validate) printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n", globalMin, globalMax, globalValid ? "PASSED" : "FAILED"); }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu)); CUDA_CHECK(cudaFreeHost(sendLo)); CUDA_CHECK(cudaFreeHost(sendHi)); CUDA_CHECK(cudaFreeHost(recvLo)); CUDA_CHECK(cudaFreeHost(recvHi));
    MPI_Finalize(); return (validate && !globalValid) ? 1 : 0;
}
