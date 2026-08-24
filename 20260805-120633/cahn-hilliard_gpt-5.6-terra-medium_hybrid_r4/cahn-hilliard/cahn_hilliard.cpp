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

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t error_ = (call);                                            \
    if (error_ != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(error_));                                      \
        MPI_Abort(MPI_COMM_WORLD, 2);                                             \
    }                                                                             \
} while (0)

__device__ __forceinline__ size_t index3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

__global__ void initialize(double* c, size_t nx, size_t ny, size_t localNz, size_t globalZ, size_t volume) {
    const size_t n = nx * ny * localNz;
    for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += blockDim.x * gridDim.x) {
        const size_t z = p / (nx * ny);
        const size_t linear = (globalZ + z) * nx * ny + p % (nx * ny);
        const double pseudo = ((linear + 1) * static_cast<size_t>(1299709) % volume) / static_cast<double>(volume);
        c[p + nx * ny] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void chemicalPotential(const double* c, double* mu, size_t nx, size_t ny, size_t localNz,
                                  double gamma, double eAA, double eBB, double eAB) {
    const size_t plane = nx * ny, n = plane * localNz;
    for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += blockDim.x * gridDim.x) {
        const size_t z = p / plane + 1, rem = p % plane, y = rem / nx, x = rem % nx, i = p + plane;
        const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
        const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
        const double cv = c[i];
        const double lap = c[index3(xp,y,z,nx,ny)] + c[index3(xn,y,z,nx,ny)]
                         + c[index3(x,yp,z,nx,ny)] + c[index3(x,yn,z,nx,ny)]
                         + c[i + plane] + c[i - plane] - 6.0 * cv;
        mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
              + 3.0 * cv + cv * cv * cv - gamma * lap;
    }
}

__global__ void update(const double* cold, const double* mu, double* cnew, size_t nx, size_t ny, size_t localNz,
                       double dt) {
    const size_t plane = nx * ny, n = plane * localNz;
    for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += blockDim.x * gridDim.x) {
        const size_t z = p / plane + 1, rem = p % plane, y = rem / nx, x = rem % nx, i = p + plane;
        const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
        const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
        const double lap = mu[index3(xp,y,z,nx,ny)] + mu[index3(xn,y,z,nx,ny)]
                         + mu[index3(x,yp,z,nx,ny)] + mu[index3(x,yn,z,nx,ny)]
                         + mu[i + plane] + mu[i - plane] - 6.0 * mu[i];
        cnew[i] = cold[i] + dt * lap;
    }
}

static void exchangeHalos(double* field, size_t plane, size_t localNz, int rank, int ranks,
                          double* lowerSend, double* upperSend, double* lowerRecv, double* upperRecv) {
    CUDA_CHECK(cudaMemcpy(lowerSend, field + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upperSend, field + localNz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(lowerSend, static_cast<int>(plane), MPI_DOUBLE, lower, 0,
                 upperRecv, static_cast<int>(plane), MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(upperSend, static_cast<int>(plane), MPI_DOUBLE, upper, 1,
                 lowerRecv, static_cast<int>(plane), MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (lower == MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(field, field + plane, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    else
        CUDA_CHECK(cudaMemcpy(field, lowerRecv, plane * sizeof(double), cudaMemcpyHostToDevice));
    if (upper == MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(field + (localNz + 1) * plane, field + localNz * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    else
        CUDA_CHECK(cudaMemcpy(field + (localNz + 1) * plane, upperRecv, plane * sizeof(double), cudaMemcpyHostToDevice));
}

static bool validateResult(const std::vector<double>& c) {
    double minVal = std::numeric_limits<double>::infinity(), maxVal = -minVal;
    int invalid = 0;
#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (size_t i = 0; i < c.size(); ++i) {
        const double v = c[i];
        if (!std::isfinite(v)) invalid = 1;
        minVal = std::min(minVal, v); maxVal = std::max(maxVal, v);
    }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    return !invalid && maxVal <= 10.0 && minVal >= -10.0;
}

static void printUsage(const char* p) {
    printf("Usage: %s [options]\n  -x <num>  Grid X (default: 64)\n  -y <num>  Grid Y (default: X)\n"
           "  -z <num>  Grid Z (default: X)\n  -i <num>  Time steps (default: 20)\n"
           "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nz < static_cast<size_t>(ranks) || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) fprintf(stderr, "Invalid dimensions or more MPI ranks than Z planes.\n"); MPI_Finalize(); return 1;
    }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank; MPI_Comm_rank(local, &localRank); MPI_Comm_free(&local);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    const size_t localNz = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t startZ = static_cast<size_t>(rank) * (nz / ranks) + std::min(static_cast<size_t>(rank), nz % ranks);
    const size_t plane = nx * ny, localCells = plane * localNz, volume = plane * nz;
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                        nx, ny, nz, iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads()); }
    double *cold, *cnew, *mu; CUDA_CHECK(cudaMalloc(&cold, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, (localNz + 2) * plane * sizeof(double))); CUDA_CHECK(cudaMalloc(&mu, (localNz + 2) * plane * sizeof(double)));
    double *lowerSend, *upperSend, *lowerRecv, *upperRecv;
    CUDA_CHECK(cudaMallocHost(&lowerSend, plane * sizeof(double))); CUDA_CHECK(cudaMallocHost(&upperSend, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&lowerRecv, plane * sizeof(double))); CUDA_CHECK(cudaMallocHost(&upperRecv, plane * sizeof(double)));
    const int blocks = static_cast<int>(std::min<size_t>((localCells + 255) / 256, 65535));
    initialize<<<blocks, 256>>>(cold, nx, ny, localNz, startZ, volume); CUDA_CHECK(cudaGetLastError());
    exchangeHalos(cold, plane, localNz, rank, ranks, lowerSend, upperSend, lowerRecv, upperRecv);
    MPI_Barrier(MPI_COMM_WORLD); const auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotential<<<blocks, 256>>>(cold, mu, nx, ny, localNz, .5, -(2.0/9.0), -(2.0/9.0), 2.0/9.0); CUDA_CHECK(cudaGetLastError());
        exchangeHalos(mu, plane, localNz, rank, ranks, lowerSend, upperSend, lowerRecv, upperRecv);
        update<<<blocks, 256>>>(cold, mu, cnew, nx, ny, localNz, .01); CUDA_CHECK(cudaGetLastError());
        std::swap(cold, cnew); exchangeHalos(cold, plane, localNz, rank, ranks, lowerSend, upperSend, lowerRecv, upperRecv);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD); const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end-start).count(); double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) { printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", seconds*1e3, volume*iterations/seconds/1e6); }
    std::vector<double> localResult(localCells); CUDA_CHECK(cudaMemcpy(localResult.data(), cold + plane, localCells*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<int> counts, offsets; std::vector<double> result;
    if (!rank) { counts.resize(ranks); offsets.resize(ranks); for (int r=0; r<ranks; ++r) { const size_t n = nz/ranks + (static_cast<size_t>(r)<nz%ranks); counts[r]=static_cast<int>(n*plane); offsets[r]=r ? offsets[r-1]+counts[r-1] : 0; } result.resize(volume); }
    MPI_Gatherv(localResult.data(), static_cast<int>(localCells), MPI_DOUBLE, rank ? nullptr : result.data(), rank ? nullptr : counts.data(), rank ? nullptr : offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int status = 0;
    if (!rank && printResults) print_results(result, "Concentration");
    if (!rank && validate) { printf("Validating result...\n"); status = validateResult(result) ? 0 : 1; printf("Validation: %s\n", status ? "FAILED" : "PASSED"); }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFreeHost(lowerSend); cudaFreeHost(upperSend); cudaFreeHost(lowerRecv); cudaFreeHost(upperRecv); cudaFree(cold); cudaFree(cnew); cudaFree(mu);
    MPI_Finalize(); return status;
}
