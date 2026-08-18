#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(e_));                                   \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

__device__ __forceinline__ size_t didx(size_t x, size_t y, size_t z,
                                        size_t nx, size_t plane) {
    return z * plane + y * nx + x;
}

// The local allocation has one halo plane on either side in z.  Those halos
// also implement the clamped global boundary condition at the end ranks.
__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                         double* __restrict__ mu,
                                         size_t nx, size_t ny, size_t localNz,
                                         double gamma, double eAA,
                                         double eBB, double eAB) {
    const size_t n = nx * ny * localNz;
    const size_t plane = nx * ny;
    for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         q < n; q += (size_t)blockDim.x * gridDim.x) {
        const size_t z = q / plane + 1;
        const size_t rem = q % plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const size_t xm = x ? x - 1 : x;
        const size_t xp = x + 1 < nx ? x + 1 : x;
        const size_t ym = y ? y - 1 : y;
        const size_t yp = y + 1 < ny ? y + 1 : y;
        const size_t i = didx(x, y, z, nx, plane);
        const double v = c[i];
        const double lap = c[didx(xm,y,z,nx,plane)] + c[didx(xp,y,z,nx,plane)]
                         + c[didx(x,ym,z,nx,plane)] + c[didx(x,yp,z,nx,plane)]
                         + c[i-plane] + c[i+plane] - 6.0 * v;
        mu[i] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
              + 3.0 * v + v * v * v - gamma * lap;
    }
}

__global__ void updateKernel(double* __restrict__ out,
                             const double* __restrict__ in,
                             const double* __restrict__ mu,
                             size_t nx, size_t ny, size_t localNz,
                             double dtD) {
    const size_t n = nx * ny * localNz;
    const size_t plane = nx * ny;
    for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         q < n; q += (size_t)blockDim.x * gridDim.x) {
        const size_t z = q / plane + 1;
        const size_t rem = q % plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const size_t xm = x ? x - 1 : x;
        const size_t xp = x + 1 < nx ? x + 1 : x;
        const size_t ym = y ? y - 1 : y;
        const size_t yp = y + 1 < ny ? y + 1 : y;
        const size_t i = didx(x, y, z, nx, plane);
        const double v = mu[i];
        const double lap = mu[didx(xm,y,z,nx,plane)] + mu[didx(xp,y,z,nx,plane)]
                         + mu[didx(x,ym,z,nx,plane)] + mu[didx(x,yp,z,nx,plane)]
                         + mu[i-plane] + mu[i+plane] - 6.0 * v;
        out[i] = in[i] + dtD * lap;
    }
}

static void exchangeHalos(double* d, size_t plane, size_t localNz, int rank,
                          int ranks, double* sendLo, double* sendHi,
                          double* recvLo, double* recvHi, int tag) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(sendLo, d + plane, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(sendHi, d + localNz * plane, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(sendLo, (int)plane, MPI_DOUBLE, prev, tag,
                 recvHi, (int)plane, MPI_DOUBLE, next, tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendHi, (int)plane, MPI_DOUBLE, next, tag + 1,
                 recvLo, (int)plane, MPI_DOUBLE, prev, tag + 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (prev == MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(d, d + plane, bytes, cudaMemcpyDeviceToDevice));
    else
        CUDA_CHECK(cudaMemcpyAsync(d, recvLo, bytes, cudaMemcpyHostToDevice));
    if (next == MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(d + (localNz + 1) * plane,
                                   d + localNz * plane, bytes, cudaMemcpyDeviceToDevice));
    else
        CUDA_CHECK(cudaMemcpyAsync(d + (localNz + 1) * plane,
                                   recvHi, bytes, cudaMemcpyHostToDevice));
}

static bool validateResult(const std::vector<double>& c) {
    int bad = 0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
#pragma omp parallel for reduction(+:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < (long long)c.size(); ++i) {
        bad += !std::isfinite(c[(size_t)i]);
        minVal = std::min(minVal, c[(size_t)i]);
        maxVal = std::max(maxVal, c[(size_t)i]);
    }
    if (bad) std::printf("Validation failed: found NaN or Inf value\n");
    std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0)
        std::printf("Validation failed: values out of expected range\n");
    return bad == 0 && maxVal <= 10.0 && minVal >= -10.0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -x <num> Grid X (default: 64)\n"
                "  -y <num> Grid Y (default: X)\n  -z <num> Grid Z (default: X)\n"
                "  -i <num> Time steps (default: 20)\n  -v Validate\n"
                "  -r Print results for external validation\n  -h Help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else parseError = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || parseError) {
        if (!rank) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    if (!nx || !ny || !nz || iterations < 0 || (size_t)ranks > nz || nx * ny > (size_t)std::numeric_limits<int>::max()) {
        if (!rank) std::fprintf(stderr, "Invalid grid/iteration count, too many ranks, or MPI plane too large\n");
        MPI_Finalize();
        return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        if (!rank) std::fprintf(stderr, "CUDA device required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t base = nz / (size_t)ranks, extra = nz % (size_t)ranks;
    const size_t localNz = base + ((size_t)rank < extra);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, extra);
    const size_t plane = nx * ny, localElements = (localNz + 2) * plane;
    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs/node: %d\n",
                    iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads(), deviceCount);
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> host(localNz * plane);
#pragma omp parallel for schedule(static)
    for (long long q = 0; q < (long long)host.size(); ++q) {
        const size_t globalId = z0 * plane + (size_t)q;
        const double pseudo = (((globalId + 1) * (size_t)1299709) % (nx * ny * nz))
                            / static_cast<double>(nx * ny * nz);
        host[(size_t)q] = -1.0 + 2.0 * pseudo;
    }

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, localElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, localElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, localElements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    double *sendLo, *sendHi, *recvLo, *recvHi;
    CUDA_CHECK(cudaMallocHost(&sendLo, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&sendHi, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recvLo, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recvHi, plane * sizeof(double)));

    if (!rank) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int threads = 256;
    const int blocks = (int)std::min<size_t>((localNz * plane + threads - 1) / threads, 65535);
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, localNz, rank, ranks, sendLo, sendHi, recvLo, recvHi, 100);
        chemicalPotentialKernel<<<blocks, threads>>>(cold, mu, nx, ny, localNz, 0.5,
                                                     -(2.0/9.0), -(2.0/9.0), 2.0/9.0);
        CUDA_CHECK(cudaGetLastError());
        exchangeHalos(mu, plane, localNz, rank, ranks, sendLo, sendHi, recvLo, recvHi, 200);
        updateKernel<<<blocks, threads>>>(cnew, cold, mu, nx, ny, localNz, 0.01);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsedLocal = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&elapsedLocal, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displs;
    std::vector<double> global;
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (!rank && (validate || printResults)) {
        counts.resize(ranks); displs.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t rnz = base + ((size_t)r < extra);
            const size_t rz0 = (size_t)r * base + std::min((size_t)r, extra);
            counts[r] = (int)(rnz * plane);
            displs[r] = (int)(rz0 * plane);
        }
        global.resize(nx * ny * nz);
    }
    if (validate || printResults)
        MPI_Gatherv(host.data(), (int)host.size(), MPI_DOUBLE, global.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (!rank) {
        const double ms = elapsed * 1000.0;
        std::printf("Computation time: %.0f ms\n", ms);
        const double updates = (double)nx * ny * nz * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
        if (printResults) print_results(global, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            rc = validateResult(global) ? 0 : 1;
            std::printf("Validation: %s\n", rc ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFreeHost(sendLo); cudaFreeHost(sendHi); cudaFreeHost(recvLo); cudaFreeHost(recvHi);
    cudaFree(cold); cudaFree(cnew); cudaFree(mu);
    MPI_Finalize();
    return rc;
}
