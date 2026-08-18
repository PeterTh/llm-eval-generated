#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t cuda_status_ = (call);                                    \
    if (cuda_status_ != cudaSuccess) {                                          \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(cuda_status_));                         \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (false)

#define MPI_CHECK(call) do {                                                    \
    const int mpi_status_ = (call);                                             \
    if (mpi_status_ != MPI_SUCCESS) {                                           \
        char mpi_error_[MPI_MAX_ERROR_STRING];                                  \
        int mpi_error_length_ = 0;                                              \
        MPI_Error_string(mpi_status_, mpi_error_, &mpi_error_length_);           \
        std::fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, \
                     mpi_error_length_, mpi_error_);                            \
        MPI_Abort(MPI_COMM_WORLD, 3);                                           \
    }                                                                           \
} while (false)

struct Slab {
    size_t begin;
    size_t size;
};

static Slab decompose(const size_t nz, const int rank, const int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, extra), base + (r < extra ? 1U : 0U)};
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             const size_t x, const size_t y,
                                             const size_t z, const size_t nx,
                                             const size_t ny) {
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = (x == 0) ? i : i - 1;
    const size_t xp = (x + 1 == nx) ? i : i + 1;
    const size_t ym = (y == 0) ? i : i - nx;
    const size_t yp = (y + 1 == ny) ? i : i + nx;
    // dx == dy == dz == 1.0 in this benchmark.
    return field[xm] + field[xp] + field[ym] + field[yp]
         + field[i - plane] + field[i + plane] - 6.0 * field[i];
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                         double* __restrict__ mu,
                                         const size_t nx, const size_t ny,
                                         const double gamma,
                                         const double eAA,
                                         const double eBB,
                                         const double eAB,
                                         const size_t zBegin,
                                         const size_t zEnd) {
    const size_t plane = nx * ny;
    for (size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(blockDim.y) * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x;
                 x < nx; x += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const size_t i = z * plane + y * nx + x;
                const double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                      + 3.0 * cv + cv * cv * cv
                      - gamma * laplacian(c, x, y, z, nx, ny);
            }
        }
    }
}

__global__ void updateKernel(const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             double* __restrict__ cnew,
                             const size_t nx, const size_t ny,
                             const double dtD,
                             const size_t zBegin, const size_t zEnd) {
    const size_t plane = nx * ny;
    for (size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(blockDim.y) * gridDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x;
                 x < nx; x += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const size_t i = z * plane + y * nx + x;
                cnew[i] = cold[i] + dtD * laplacian(mu, x, y, z, nx, ny);
            }
        }
    }
}

static dim3 launchGrid(const size_t nx, const size_t ny, const size_t nz) {
    return dim3(static_cast<unsigned>(std::min<size_t>((nx + 31) / 32, 2147483647)),
                static_cast<unsigned>(std::min<size_t>((ny + 3) / 4, 65535)),
                static_cast<unsigned>(std::min<size_t>(nz, 65535)));
}

struct HaloBuffers {
    bool cudaAware = false;
    double* storage = nullptr;
    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* recvLower = nullptr;
    double* recvUpper = nullptr;
};

static void beginHaloExchange(double* field, HaloBuffers& buffers, const size_t plane,
                              const size_t localNz, const int rank,
                              const int ranks, MPI_Request requests[4]) {
    const int previous = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int next = (rank + 1 == ranks) ? MPI_PROC_NULL : rank + 1;
    const int count = static_cast<int>(plane);

    double* sendLower = field + plane;
    double* sendUpper = field + localNz * plane;
    double* recvLower = field;
    double* recvUpper = field + (localNz + 1) * plane;
    if (!buffers.cudaAware) {
        // Pinned staging works with MPI stacks that cannot access device memory.
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendLower, sendLower,
                                   plane * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendUpper, sendUpper,
                                   plane * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        sendLower = buffers.sendLower;
        sendUpper = buffers.sendUpper;
        recvLower = buffers.recvLower;
        recvUpper = buffers.recvUpper;
    } else {
        // Ensure the preceding kernel has published its boundary planes before
        // CUDA-aware MPI accesses them directly.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_CHECK(MPI_Irecv(recvLower, count, MPI_DOUBLE, previous, 101,
                        MPI_COMM_WORLD, &requests[0]));
    MPI_CHECK(MPI_Irecv(recvUpper, count, MPI_DOUBLE,
                        next, 100, MPI_COMM_WORLD, &requests[1]));
    MPI_CHECK(MPI_Isend(sendLower, count, MPI_DOUBLE, previous, 100,
                        MPI_COMM_WORLD, &requests[2]));
    MPI_CHECK(MPI_Isend(sendUpper, count, MPI_DOUBLE, next, 101,
                        MPI_COMM_WORLD, &requests[3]));
}

static void finishHaloExchange(double* field, HaloBuffers& buffers, const size_t plane,
                               const size_t localNz, const int rank,
                               const int ranks, MPI_Request requests[4]) {
    MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
    if (rank == 0) {
        CUDA_CHECK(cudaMemcpyAsync(field, field + plane, plane * sizeof(double),
                                   cudaMemcpyDeviceToDevice));
    } else if (!buffers.cudaAware) {
        CUDA_CHECK(cudaMemcpyAsync(field, buffers.recvLower, plane * sizeof(double),
                                   cudaMemcpyHostToDevice));
    }
    if (rank + 1 == ranks) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane,
                                   field + localNz * plane,
                                   plane * sizeof(double), cudaMemcpyDeviceToDevice));
    } else if (!buffers.cudaAware) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, buffers.recvUpper,
                                   plane * sizeof(double), cudaMemcpyHostToDevice));
    }
}

template <typename Kernel, typename... Args>
static void stencilStage(double* haloField, HaloBuffers& buffers, const size_t plane,
                         const size_t nx, const size_t ny, const size_t localNz,
                         const int rank, const int ranks,
                         Kernel kernel, Args... args) {
    MPI_Request requests[4];
    beginHaloExchange(haloField, buffers, plane, localNz, rank, ranks, requests);

    const dim3 threads(32, 4, 1);
    if (localNz > 2) {
        kernel<<<launchGrid(nx, ny, localNz - 2), threads>>>(args..., size_t{2}, localNz);
        CUDA_CHECK(cudaGetLastError());
    }

    finishHaloExchange(haloField, buffers, plane, localNz, rank, ranks, requests);
    kernel<<<launchGrid(nx, ny, 1), threads>>>(args..., size_t{1}, size_t{2});
    CUDA_CHECK(cudaGetLastError());
    if (localNz > 1) {
        kernel<<<launchGrid(nx, ny, 1), threads>>>(args..., localNz, localNz + 1);
        CUDA_CHECK(cudaGetLastError());
    }
}

static bool validateResult(const std::vector<double>& c) {
    int bad = 0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
    // OpenMP handles the host-side diagnostic pass; the numerical timestep is
    // always performed by CUDA.
#pragma omp parallel for reduction(|:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
        const double value = c[static_cast<size_t>(i)];
        bad |= !std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    if (bad) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const bool invalid = nx == 0 || ny == 0 || nz == 0 || iterations < 0
        || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz
        || nx * ny > static_cast<size_t>(INT_MAX)
        || static_cast<size_t>(ranks) > nz;
    if (invalid) {
        if (rank == 0) std::fprintf(stderr, "Invalid grid/iteration count, MPI ranks exceed nz, or MPI message is too large\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&localComm));

    const Slab slab = decompose(nz, rank, ranks);
    const size_t plane = nx * ny;
    const size_t localElements = slab.size * plane;
    const size_t allocatedElements = (slab.size + 2) * plane;
    const size_t globalElements = plane * nz;

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Parallelism: %d MPI ranks, up to %d OpenMP threads/rank, CUDA\n",
                    ranks, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> initial(localElements);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(localElements); ++i) {
        const size_t globalId = slab.begin * plane + static_cast<size_t>(i);
        const double pseudo = (((globalId + 1) * size_t{1299709}) % globalElements)
                            / static_cast<double>(globalElements);
        initial[static_cast<size_t>(i)] = -1.0 + 2.0 * pseudo;
    }

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    HaloBuffers halos;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    halos.cudaAware = MPIX_Query_cuda_support() != 0;
#endif
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cold), allocatedElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cnew), allocatedElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mu), allocatedElements * sizeof(double)));
    if (!halos.cudaAware) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&halos.storage),
                                 4 * plane * sizeof(double), cudaHostAllocDefault));
        halos.sendLower = halos.storage;
        halos.sendUpper = halos.storage + plane;
        halos.recvLower = halos.storage + 2 * plane;
        halos.recvUpper = halos.storage + 3 * plane;
    }
    CUDA_CHECK(cudaMemcpy(cold + plane, initial.data(), localElements * sizeof(double),
                          cudaMemcpyHostToDevice));
    initial.clear();
    initial.shrink_to_fit();

    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB =  (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double dtD = 0.01;

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < iterations; ++step) {
        stencilStage(cold, halos, plane, nx, ny, slab.size, rank, ranks,
                     chemicalPotentialKernel, cold, mu, nx, ny,
                     gamma, eAA, eBB, eAB);
        stencilStage(mu, halos, plane, nx, ny, slab.size, rank, ranks,
                     updateKernel, cold, mu, cnew, nx, ny, dtD);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double updates = static_cast<double>(globalElements) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    seconds > 0.0 ? updates / seconds / 1.0e6 : 0.0);
    }

    int resultCode = 0;
    if (printResults || validate) {
        if (localElements > static_cast<size_t>(INT_MAX)
            || globalElements > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI_Gatherv limits\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
        std::vector<double> localResult(localElements);
        CUDA_CHECK(cudaMemcpy(localResult.data(), cold + plane,
                              localElements * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> counts, displacements;
        std::vector<double> result;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const Slab part = decompose(nz, r, ranks);
                counts[r] = static_cast<int>(part.size * plane);
                displacements[r] = static_cast<int>(part.begin * plane);
            }
            result.resize(globalElements);
        }
        MPI_CHECK(MPI_Gatherv(localResult.data(), static_cast<int>(localElements), MPI_DOUBLE,
                              rank == 0 ? result.data() : nullptr,
                              rank == 0 ? counts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr,
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            if (printResults) print_results(result, "Concentration");
            if (validate) {
                std::printf("Validating result...\n");
                const bool valid = validateResult(result);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                resultCode = valid ? 0 : 1;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    if (halos.storage) CUDA_CHECK(cudaFreeHost(halos.storage));
    MPI_CHECK(MPI_Finalize());
    return resultCode;
}
