#include <algorithm>
#include <chrono>
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

#define CUDA_CHECK(call) do { \
    const cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error__)); \
    } \
} while (0)

struct DeviceArray {
    double* data = nullptr;
    explicit DeviceArray(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(double))); }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
};

// The local z range is [1, localNz].  Planes 0 and localNz+1 are MPI halos.
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t localNz,
                                        double gamma, double eAA, double eBB, double eAB) {
    const size_t plane = nx * ny;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = plane * localNz;
    if (i >= cells) return;
    const size_t z = i / plane + 1;
    const size_t inPlane = i % plane;
    const size_t y = inPlane / nx;
    const size_t x = inPlane % nx;
    const size_t center = z * plane + inPlane;
    const size_t xp = x + (x + 1 < nx);
    const size_t xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t ym = y - (y > 0);
    const double cv = c[center];
    const double lap = c[z * plane + y * nx + xp] + c[z * plane + y * nx + xm]
                     + c[z * plane + yp * nx + x] + c[z * plane + ym * nx + x]
                     + c[center + plane] + c[center - plane] - 6.0 * cv;
    mu[center] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
               + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t nx, size_t ny, size_t localNz,
                             double dtD) {
    const size_t plane = nx * ny;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = plane * localNz;
    if (i >= cells) return;
    const size_t z = i / plane + 1;
    const size_t inPlane = i % plane;
    const size_t y = inPlane / nx;
    const size_t x = inPlane % nx;
    const size_t center = z * plane + inPlane;
    const size_t xp = x + (x + 1 < nx);
    const size_t xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t ym = y - (y > 0);
    const double lap = mu[z * plane + y * nx + xp] + mu[z * plane + y * nx + xm]
                     + mu[z * plane + yp * nx + x] + mu[z * plane + ym * nx + x]
                     + mu[center + plane] + mu[center - plane] - 6.0 * mu[center];
    cnew[center] = cold[center] + dtD * lap;
}

// Staged transfers work with every MPI implementation, including non CUDA-aware ones.
// They can be replaced by CUDA-aware MPI transparently at deployment time without changing semantics.
static void exchangeHalos(double* device, size_t plane, size_t localNz, int lower, int upper,
                          std::vector<double>& sendLower, std::vector<double>& sendUpper,
                          std::vector<double>& recvLower, std::vector<double>& recvUpper) {
    CUDA_CHECK(cudaMemcpy(sendLower.data(), device + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sendUpper.data(), device + localNz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Request requests[4];
    int nreq = 0;
    if (lower != MPI_PROC_NULL) {
        MPI_Irecv(recvLower.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 11, MPI_COMM_WORLD, &requests[nreq++]);
        MPI_Isend(sendLower.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 12, MPI_COMM_WORLD, &requests[nreq++]);
    } else {
        std::copy(sendLower.begin(), sendLower.end(), recvLower.begin());
    }
    if (upper != MPI_PROC_NULL) {
        MPI_Irecv(recvUpper.data(), static_cast<int>(plane), MPI_DOUBLE, upper, 12, MPI_COMM_WORLD, &requests[nreq++]);
        MPI_Isend(sendUpper.data(), static_cast<int>(plane), MPI_DOUBLE, upper, 11, MPI_COMM_WORLD, &requests[nreq++]);
    } else {
        std::copy(sendUpper.begin(), sendUpper.end(), recvUpper.begin());
    }
    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    CUDA_CHECK(cudaMemcpy(device, recvLower.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device + (localNz + 1) * plane, recvUpper.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
}

static void initialize(std::vector<double>& c, size_t nx, size_t ny, size_t localNz, size_t globalZ0, size_t globalNz) {
    const size_t plane = nx * ny, volume = plane * globalNz;
    #pragma omp parallel for schedule(static)
    for (long long z = 0; z < static_cast<long long>(localNz); ++z) {
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            const size_t globalId = (globalZ0 + static_cast<size_t>(z)) * plane + y * nx + x;
            const double pseudo = (((globalId + 1) * 1299709 % volume) / static_cast<double>(volume));
            c[(static_cast<size_t>(z) + 1) * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
        }
    }
}

static bool validateResult(const std::vector<double>& c) {
    double minValue = c[0], maxValue = c[0];
    bool finite = true;
    #pragma omp parallel for reduction(min:minValue) reduction(max:maxValue) reduction(&:finite)
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
        finite &= std::isfinite(c[static_cast<size_t>(i)]);
        minValue = std::min(minValue, c[static_cast<size_t>(i)]);
        maxValue = std::max(maxValue, c[static_cast<size_t>(i)]);
    }
    if (!finite) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
    std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) { std::printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 64)\n  -y <num>  Grid Y (default X)\n"
                "  -z <num>  Grid Z (default X)\n  -i <num>  Time steps (default 20)\n"
                "  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show help\n", p);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || static_cast<size_t>(ranks) > nz || nx * ny > static_cast<size_t>(INT_MAX)) {
        if (!rank) std::fprintf(stderr, "Invalid grid/iteration count (need at least one z plane per MPI rank).\n");
        MPI_Finalize(); return 1;
    }
    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Finalize(); return 1; }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const size_t base = nz / ranks, remainder = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny, localCells = plane * localNz;
    const int lower = rank ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }

    std::vector<double> hostC((localNz + 2) * plane, 0.0), sendLower(plane), sendUpper(plane), recvLower(plane), recvUpper(plane);
    initialize(hostC, nx, ny, localNz, z0, nz);
    DeviceArray cold((localNz + 2) * plane), cnew((localNz + 2) * plane), mu((localNz + 2) * plane);
    CUDA_CHECK(cudaMemcpy(cold.data, hostC.data(), hostC.size() * sizeof(double), cudaMemcpyHostToDevice));
    exchangeHalos(cold.data, plane, localNz, lower, upper, sendLower, sendUpper, recvLower, recvUpper);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((localCells + threads - 1) / threads);
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threads>>>(cold.data, mu.data, nx, ny, localNz, .5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        CUDA_CHECK(cudaGetLastError());
        exchangeHalos(mu.data, plane, localNz, lower, upper, sendLower, sendUpper, recvLower, recvUpper);
        updateKernel<<<blocks, threads>>>(cnew.data, cold.data, mu.data, nx, ny, localNz, .01);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cold.data, cnew.data);
        // The final field is copied out directly; its halo is only needed by the next step.
        if (t + 1 < iterations)
            exchangeHalos(cold.data, plane, localNz, lower, upper, sendLower, sendUpper, recvLower, recvUpper);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(hostC.data() + plane, cold.data + plane, localCells * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> globalC;
    if (validate || printResults) {
        std::vector<int> counts, offsets;
        if (!rank) { counts.resize(ranks); offsets.resize(ranks); }
        const int localCount = static_cast<int>(localCells);
        MPI_Gather(&localCount, 1, MPI_INT, rank ? nullptr : counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!rank) {
            int offset = 0; for (int r = 0; r < ranks; ++r) { offsets[r] = offset; offset += counts[r]; }
            globalC.resize(nx * ny * nz);
        }
        MPI_Gatherv(hostC.data() + plane, localCount, MPI_DOUBLE, rank ? nullptr : globalC.data(),
                    rank ? nullptr : counts.data(), rank ? nullptr : offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", maxElapsed * 1000.0,
                    static_cast<double>(nx * ny * nz) * iterations / maxElapsed / 1e6);
        if (printResults) print_results(globalC, "Concentration");
        if (validate && !validateResult(globalC)) { MPI_Finalize(); return 1; }
        if (validate) std::printf("Validation: PASSED\n");
    }
    MPI_Finalize();
    return 0;
}
