#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int kLowerHaloTag = 101;
constexpr int kUpperHaloTag = 102;
constexpr int kGatherTag = 201;

[[noreturn]] void abortRun(const char* message, const int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* expression, const int rank,
               const char* file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d in %s: %s\n", rank,
                     file, line, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

void checkMpi(const int error, const char* expression, const int rank,
              const char* file, const int line) {
    if (error != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(error, errorString, &length);
        std::fprintf(stderr, "Rank %d: MPI failure at %s:%d in %s: %.*s\n", rank,
                     file, line, expression, length, errorString);
        MPI_Abort(MPI_COMM_WORLD, error);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, rank, __FILE__, __LINE__)

struct Options {
    std::size_t nx = 64;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool help = false;
};

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, std::size_t& value) {
    if (text[0] == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parseOptions(const int argc, char** argv, Options& options, int rank) {
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0 ||
             std::strcmp(argv[i], "-z") == 0) && i + 1 < argc) {
            std::size_t parsed = 0;
            if (!parseSize(argv[++i], parsed)) {
                if (rank == 0) std::fprintf(stderr, "Invalid positive grid size: %s\n", argv[i]);
                return false;
            }
            if (std::strcmp(argv[i - 1], "-x") == 0) options.nx = parsed;
            if (std::strcmp(argv[i - 1], "-y") == 0) options.ny = parsed;
            if (std::strcmp(argv[i - 1], "-z") == 0) options.nz = parsed;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], options.iterations)) {
                if (rank == 0) std::fprintf(stderr, "Invalid iteration count: %s\n", argv[i]);
                return false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return false;
        }
    }
    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    return true;
}

bool checkedMultiply(const std::size_t a, const std::size_t b, std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    result = a * b;
    return true;
}

std::pair<std::size_t, std::size_t> slabForRank(const std::size_t nz,
                                                const int worldSize,
                                                const int rank) {
    const std::size_t base = nz / static_cast<std::size_t>(worldSize);
    const std::size_t remainder = nz % static_cast<std::size_t>(worldSize);
    const std::size_t localNz = base + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
    const std::size_t zOffset = static_cast<std::size_t>(rank) * base +
                                std::min(static_cast<std::size_t>(rank), remainder);
    return {zOffset, localNz};
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            const std::size_t index,
                                            const std::size_t x,
                                            const std::size_t y,
                                            const std::size_t nx,
                                            const std::size_t ny,
                                            const std::size_t plane,
                                            const double invDx2,
                                            const double invDy2,
                                            const double invDz2) {
    const std::size_t xm = x == 0 ? index : index - 1;
    const std::size_t xp = x + 1 == nx ? index : index + 1;
    const std::size_t ym = y == 0 ? index : index - nx;
    const std::size_t yp = y + 1 == ny ? index : index + nx;
    const double center = field[index];
    return (field[xp] + field[xm] - 2.0 * center) * invDx2 +
           (field[yp] + field[ym] - 2.0 * center) * invDy2 +
           (field[index + plane] + field[index - plane] - 2.0 * center) * invDz2;
}

__global__ void chemicalPotentialRange(const double* __restrict__ c,
                                       double* __restrict__ mu,
                                       const std::size_t nx,
                                       const std::size_t ny,
                                       const std::size_t plane,
                                       const std::size_t zBegin,
                                       const std::size_t zEnd,
                                       const double invDx2,
                                       const double invDy2,
                                       const double invDz2,
                                       const double gamma,
                                       const double eAA,
                                       const double eBB,
                                       const double eAB) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    for (std::size_t z = zBegin + blockIdx.z; z <= zEnd; z += gridDim.z) {
        const std::size_t index = z * plane + y * nx + x;
        const double cv = c[index];
        mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                    3.0 * cv + cv * cv * cv -
                    gamma * laplacian(c, index, x, y, nx, ny, plane,
                                      invDx2, invDy2, invDz2);
    }
}

__global__ void chemicalPotentialBoundary(const double* __restrict__ c,
                                          double* __restrict__ mu,
                                          const std::size_t nx,
                                          const std::size_t ny,
                                          const std::size_t plane,
                                          const std::size_t localNz,
                                          const double invDx2,
                                          const double invDy2,
                                          const double invDz2,
                                          const double gamma,
                                          const double eAA,
                                          const double eBB,
                                          const double eAB) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t z = blockIdx.z == 0 ? 1 : localNz;
    const std::size_t index = z * plane + y * nx + x;
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                3.0 * cv + cv * cv * cv -
                gamma * laplacian(c, index, x, y, nx, ny, plane,
                                  invDx2, invDy2, invDz2);
}

__global__ void updateRange(double* __restrict__ cNew,
                            const double* __restrict__ cOld,
                            const double* __restrict__ mu,
                            const std::size_t nx,
                            const std::size_t ny,
                            const std::size_t plane,
                            const std::size_t zBegin,
                            const std::size_t zEnd,
                            const double dtD,
                            const double invDx2,
                            const double invDy2,
                            const double invDz2) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    for (std::size_t z = zBegin + blockIdx.z; z <= zEnd; z += gridDim.z) {
        const std::size_t index = z * plane + y * nx + x;
        cNew[index] = cOld[index] + dtD * laplacian(mu, index, x, y, nx, ny,
                                                    plane, invDx2, invDy2, invDz2);
    }
}

__global__ void updateBoundary(double* __restrict__ cNew,
                               const double* __restrict__ cOld,
                               const double* __restrict__ mu,
                               const std::size_t nx,
                               const std::size_t ny,
                               const std::size_t plane,
                               const std::size_t localNz,
                               const double dtD,
                               const double invDx2,
                               const double invDy2,
                               const double invDz2) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t z = blockIdx.z == 0 ? 1 : localNz;
    const std::size_t index = z * plane + y * nx + x;
    cNew[index] = cOld[index] + dtD * laplacian(mu, index, x, y, nx, ny,
                                                plane, invDx2, invDy2, invDz2);
}

dim3 makeGrid(const std::size_t nx, const std::size_t ny, const std::size_t nz) {
    constexpr unsigned int maxGridZ = 65535;
    return dim3(static_cast<unsigned int>((nx + 31) / 32),
                static_cast<unsigned int>((ny + 3) / 4),
                static_cast<unsigned int>(std::min<std::size_t>(nz, maxGridZ)));
}

class HaloExchange {
public:
    HaloExchange(const std::size_t plane, const std::size_t localNz,
                 const int rank, const int worldSize, const bool cudaAware)
        : plane_(plane), localNz_(localNz), rank_(rank),
          lowerRank_(rank == 0 ? MPI_PROC_NULL : rank - 1),
          upperRank_(rank + 1 == worldSize ? MPI_PROC_NULL : rank + 1),
          cudaAware_(cudaAware) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&readyEvent_, cudaEventDisableTiming));
        if (!cudaAware_) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostStorage_),
                                      4 * plane_ * sizeof(double)));
        }
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        if (hostStorage_ != nullptr) cudaFreeHost(hostStorage_);
        if (readyEvent_ != nullptr) cudaEventDestroy(readyEvent_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    void begin(double* field, cudaStream_t computeStream) {
        const int rank = rank_;
        requestCount_ = 0;
        double* const lowerGhost = field;
        double* const lowerPlane = field + plane_;
        double* const upperPlane = field + localNz_ * plane_;
        double* const upperGhost = field + (localNz_ + 1) * plane_;
        const std::size_t bytes = plane_ * sizeof(double);

        if (cudaAware_) {
            // Device-buffer MPI has no portable stream dependency mechanism.
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            if (lowerRank_ == MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(lowerGhost, lowerPlane, bytes,
                                           cudaMemcpyDeviceToDevice, stream_));
            } else {
                MPI_CHECK(MPI_Irecv(lowerGhost, static_cast<int>(plane_), MPI_DOUBLE,
                                    lowerRank_, kLowerHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
                MPI_CHECK(MPI_Isend(lowerPlane, static_cast<int>(plane_), MPI_DOUBLE,
                                    lowerRank_, kUpperHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
            }
            if (upperRank_ == MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(upperGhost, upperPlane, bytes,
                                           cudaMemcpyDeviceToDevice, stream_));
            } else {
                MPI_CHECK(MPI_Irecv(upperGhost, static_cast<int>(plane_), MPI_DOUBLE,
                                    upperRank_, kUpperHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
                MPI_CHECK(MPI_Isend(upperPlane, static_cast<int>(plane_), MPI_DOUBLE,
                                    upperRank_, kLowerHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
            }
        } else {
            // Only the outgoing boundary planes depend on prior stencil work.
            // The event lets their staging copies overlap the next interior kernel.
            CUDA_CHECK(cudaEventRecord(readyEvent_, computeStream));
            CUDA_CHECK(cudaStreamWaitEvent(stream_, readyEvent_, 0));
            double* const sendLower = hostStorage_;
            double* const sendUpper = hostStorage_ + plane_;
            if (lowerRank_ == MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(lowerGhost, lowerPlane, bytes,
                                           cudaMemcpyDeviceToDevice, stream_));
            } else {
                MPI_CHECK(MPI_Irecv(hostStorage_ + 2 * plane_, static_cast<int>(plane_),
                                    MPI_DOUBLE, lowerRank_, kLowerHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
                CUDA_CHECK(cudaMemcpyAsync(sendLower, lowerPlane, bytes,
                                           cudaMemcpyDeviceToHost, stream_));
            }
            if (upperRank_ == MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(upperGhost, upperPlane, bytes,
                                           cudaMemcpyDeviceToDevice, stream_));
            } else {
                MPI_CHECK(MPI_Irecv(hostStorage_ + 3 * plane_, static_cast<int>(plane_),
                                    MPI_DOUBLE, upperRank_, kUpperHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
                CUDA_CHECK(cudaMemcpyAsync(sendUpper, upperPlane, bytes,
                                           cudaMemcpyDeviceToHost, stream_));
            }
        }
    }

    void finish(double* field) {
        const int rank = rank_;
        if (!cudaAware_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (lowerRank_ != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(hostStorage_, static_cast<int>(plane_), MPI_DOUBLE,
                                    lowerRank_, kUpperHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
            }
            if (upperRank_ != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Isend(hostStorage_ + plane_, static_cast<int>(plane_), MPI_DOUBLE,
                                    upperRank_, kLowerHaloTag, MPI_COMM_WORLD,
                                    &requests_[requestCount_++]));
            }
        }
        if (requestCount_ != 0) {
            MPI_CHECK(MPI_Waitall(requestCount_, requests_, MPI_STATUSES_IGNORE));
        }
        if (!cudaAware_) {
            const std::size_t bytes = plane_ * sizeof(double);
            if (lowerRank_ != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(field, hostStorage_ + 2 * plane_, bytes,
                                           cudaMemcpyHostToDevice, stream_));
            }
            if (upperRank_ != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(field + (localNz_ + 1) * plane_,
                                           hostStorage_ + 3 * plane_, bytes,
                                           cudaMemcpyHostToDevice, stream_));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    std::size_t plane_;
    std::size_t localNz_;
    int rank_;
    int lowerRank_;
    int upperRank_;
    bool cudaAware_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t readyEvent_ = nullptr;
    double* hostStorage_ = nullptr;
    MPI_Request requests_[4] = {};
    int requestCount_ = 0;
};

void launchChemicalInterior(const double* c, double* mu, const Options& options,
                            const std::size_t plane, const std::size_t localNz,
                            const double invDx2, const double invDy2, const double invDz2,
                            const double gamma, const double eAA, const double eBB,
                            const double eAB, cudaStream_t stream, const int rank) {
    if (localNz <= 2) return;
    const dim3 block(32, 4, 1);
    chemicalPotentialRange<<<makeGrid(options.nx, options.ny, localNz - 2), block, 0, stream>>>(
        c, mu, options.nx, options.ny, plane, 2, localNz - 1, invDx2, invDy2,
        invDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchChemicalAll(const double* c, double* mu, const Options& options,
                       const std::size_t plane, const std::size_t localNz,
                       const double invDx2, const double invDy2, const double invDz2,
                       const double gamma, const double eAA, const double eBB,
                       const double eAB, cudaStream_t stream, const int rank) {
    const dim3 block(32, 4, 1);
    chemicalPotentialRange<<<makeGrid(options.nx, options.ny, localNz), block, 0, stream>>>(
        c, mu, options.nx, options.ny, plane, 1, localNz, invDx2, invDy2,
        invDz2, gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchChemicalBoundary(const double* c, double* mu, const Options& options,
                            const std::size_t plane, const std::size_t localNz,
                            const double invDx2, const double invDy2, const double invDz2,
                            const double gamma, const double eAA, const double eBB,
                            const double eAB, cudaStream_t stream, const int rank) {
    const dim3 block(32, 4, 1);
    const std::size_t boundaryCount = localNz == 1 ? 1 : 2;
    chemicalPotentialBoundary<<<makeGrid(options.nx, options.ny, boundaryCount), block, 0, stream>>>(
        c, mu, options.nx, options.ny, plane, localNz, invDx2, invDy2, invDz2,
        gamma, eAA, eBB, eAB);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateInterior(double* cNew, const double* cOld, const double* mu,
                          const Options& options, const std::size_t plane,
                          const std::size_t localNz, const double dtD,
                          const double invDx2, const double invDy2, const double invDz2,
                          cudaStream_t stream, const int rank) {
    if (localNz <= 2) return;
    const dim3 block(32, 4, 1);
    updateRange<<<makeGrid(options.nx, options.ny, localNz - 2), block, 0, stream>>>(
        cNew, cOld, mu, options.nx, options.ny, plane, 2, localNz - 1,
        dtD, invDx2, invDy2, invDz2);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateAll(double* cNew, const double* cOld, const double* mu,
                     const Options& options, const std::size_t plane,
                     const std::size_t localNz, const double dtD,
                     const double invDx2, const double invDy2, const double invDz2,
                     cudaStream_t stream, const int rank) {
    const dim3 block(32, 4, 1);
    updateRange<<<makeGrid(options.nx, options.ny, localNz), block, 0, stream>>>(
        cNew, cOld, mu, options.nx, options.ny, plane, 1, localNz,
        dtD, invDx2, invDy2, invDz2);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdateBoundary(double* cNew, const double* cOld, const double* mu,
                          const Options& options, const std::size_t plane,
                          const std::size_t localNz, const double dtD,
                          const double invDx2, const double invDy2, const double invDz2,
                          cudaStream_t stream, const int rank) {
    const dim3 block(32, 4, 1);
    const std::size_t boundaryCount = localNz == 1 ? 1 : 2;
    updateBoundary<<<makeGrid(options.nx, options.ny, boundaryCount), block, 0, stream>>>(
        cNew, cOld, mu, options.nx, options.ny, plane, localNz, dtD,
        invDx2, invDy2, invDz2);
    CUDA_CHECK(cudaPeekAtLastError());
}

bool queryCudaAwareMpi(const int rank) {
    int localSupport = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    localSupport = MPIX_Query_cuda_support() != 0 ? 1 : 0;
#endif
    int globalSupport = 0;
    MPI_CHECK(MPI_Allreduce(&localSupport, &globalSupport, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    return globalSupport != 0;
}

void gatherField(const std::vector<double>& local, std::vector<double>& global,
                 const Options& options, const std::size_t plane,
                 const int rank, const int worldSize) {
    if (rank == 0) {
        std::copy(local.begin(), local.end(), global.begin());
        for (int source = 1; source < worldSize; ++source) {
            const auto [zOffset, sourceNz] = slabForRank(options.nz, worldSize, source);
            std::size_t remaining = sourceNz * plane;
            double* destination = global.data() + zOffset * plane;
            while (remaining != 0) {
                const int chunk = static_cast<int>(std::min<std::size_t>(remaining, INT_MAX));
                MPI_CHECK(MPI_Recv(destination, chunk, MPI_DOUBLE, source, kGatherTag,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                destination += chunk;
                remaining -= static_cast<std::size_t>(chunk);
            }
        }
    } else {
        std::size_t remaining = local.size();
        const double* source = local.data();
        while (remaining != 0) {
            const int chunk = static_cast<int>(std::min<std::size_t>(remaining, INT_MAX));
            MPI_CHECK(MPI_Send(source, chunk, MPI_DOUBLE, 0, kGatherTag, MPI_COMM_WORLD));
            source += chunk;
            remaining -= static_cast<std::size_t>(chunk);
        }
    }
}

bool validateDistributed(const std::vector<double>& local, const int rank) {
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    int localFinite = 1;
    const long long count = static_cast<long long>(local.size());

#pragma omp parallel for schedule(static) reduction(min : localMin) reduction(max : localMax) reduction(& : localFinite)
    for (long long i = 0; i < count; ++i) {
        const double value = local[static_cast<std::size_t>(i)];
        localFinite &= std::isfinite(value) ? 1 : 0;
        if (std::isfinite(value)) {
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    int globalFinite = 0;
    MPI_CHECK(MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD));

    if (rank == 0) {
        if (!globalFinite) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return globalFinite != 0 && globalMax <= 10.0 && globalMin >= -10.0;
}

int runSimulation(const Options& options, const int rank, const int worldSize,
                  const bool cudaAware) {
    std::size_t plane = 0;
    std::size_t gridSize = 0;
    if (!checkedMultiply(options.nx, options.ny, plane) ||
        !checkedMultiply(plane, options.nz, gridSize)) {
        abortRun("Grid dimensions overflow size_t", rank);
    }
    if (plane > static_cast<std::size_t>(INT_MAX)) {
        abortRun("An X-Y halo plane exceeds the MPI count limit", rank);
    }

    const auto [zOffset, localNz] = slabForRank(options.nz, worldSize, rank);
    std::size_t localElements = 0;
    std::size_t paddedElements = 0;
    if (!checkedMultiply(plane, localNz, localElements) ||
        !checkedMultiply(plane, localNz + 2, paddedElements) ||
        paddedElements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        abortRun("Local grid allocation overflows size_t", rank);
    }
    if (localElements > static_cast<std::size_t>(LLONG_MAX)) {
        abortRun("Local grid is too large for OpenMP loop indexing", rank);
    }

    std::vector<double> initial(localElements);
    const long long initialCount = static_cast<long long>(localElements);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < initialCount; ++i) {
        const std::size_t globalId = zOffset * plane + static_cast<std::size_t>(i);
        const double pseudo = (((globalId + 1) * static_cast<std::size_t>(1299709)) % gridSize) /
                              static_cast<double>(gridSize);
        initial[static_cast<std::size_t>(i)] = -1.0 + 2.0 * pseudo;
    }

    double* dCold = nullptr;
    double* dNew = nullptr;
    double* dMu = nullptr;
    cudaStream_t computeStream = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dCold), paddedElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNew), paddedElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMu), paddedElements * sizeof(double)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(dCold + plane, initial.data(), localElements * sizeof(double),
                               cudaMemcpyHostToDevice, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    initial.clear();
    initial.shrink_to_fit();

    // Load all kernels and favor the L1 cache before entering the timed region.
    cudaFuncAttributes functionAttributes = {};
    CUDA_CHECK(cudaFuncGetAttributes(&functionAttributes, chemicalPotentialRange));
    CUDA_CHECK(cudaFuncGetAttributes(&functionAttributes, chemicalPotentialBoundary));
    CUDA_CHECK(cudaFuncGetAttributes(&functionAttributes, updateRange));
    CUDA_CHECK(cudaFuncGetAttributes(&functionAttributes, updateBoundary));
    CUDA_CHECK(cudaFuncSetCacheConfig(chemicalPotentialRange, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(chemicalPotentialBoundary, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(updateRange, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(updateBoundary, cudaFuncCachePreferL1));

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double invDx2 = 1.0 / (dx * dx);
    constexpr double invDy2 = 1.0 / (dy * dy);
    constexpr double invDz2 = 1.0 / (dz * dz);

    double elapsed = 0.0;
    {
        HaloExchange halo(plane, localNz, rank, worldSize, cudaAware);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();

        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            if (worldSize == 1) {
                halo.begin(dCold, computeStream);
                halo.finish(dCold);
                launchChemicalAll(dCold, dMu, options, plane, localNz,
                                  invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB,
                                  computeStream, rank);
                halo.begin(dMu, computeStream);
                halo.finish(dMu);
                launchUpdateAll(dNew, dCold, dMu, options, plane, localNz,
                                dt * diffusion, invDx2, invDy2, invDz2,
                                computeStream, rank);
            } else {
                halo.begin(dCold, computeStream);
                launchChemicalInterior(dCold, dMu, options, plane, localNz,
                                       invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB,
                                       computeStream, rank);
                halo.finish(dCold);
                launchChemicalBoundary(dCold, dMu, options, plane, localNz,
                                       invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB,
                                       computeStream, rank);

                halo.begin(dMu, computeStream);
                launchUpdateInterior(dNew, dCold, dMu, options, plane, localNz,
                                     dt * diffusion, invDx2, invDy2, invDz2,
                                     computeStream, rank);
                halo.finish(dMu);
                launchUpdateBoundary(dNew, dCold, dMu, options, plane, localNz,
                                     dt * diffusion, invDx2, invDy2, invDz2,
                                     computeStream, rank);
            }
            std::swap(dCold, dNew);
        }

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        const double localElapsed = MPI_Wtime() - start;
        MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD));
    }

    std::vector<double> localResult;
    if (options.validate || options.printResults) {
        localResult.resize(localElements);
        CUDA_CHECK(cudaMemcpy(localResult.data(), dCold + plane,
                              localElements * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        const double cellUpdates = static_cast<double>(gridSize) * options.iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (options.printResults) {
        std::vector<double> globalResult;
        if (rank == 0) globalResult.resize(gridSize);
        gatherField(localResult, globalResult, options, plane, rank, worldSize);
        if (rank == 0) print_results(globalResult, "Concentration");
    }

    bool valid = true;
    if (options.validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localResult, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(dMu));
    CUDA_CHECK(cudaFree(dNew));
    CUDA_CHECK(cudaFree(dCold));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) abortRun("MPI does not provide MPI_THREAD_FUNNELED", rank);
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    Options options;
    const bool parsed = parseOptions(argc, argv, options, rank);
    if (!parsed || options.help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parsed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (options.nz < static_cast<std::size_t>(worldSize)) {
        if (rank == 0) {
            std::fprintf(stderr, "The Z dimension (%zu) must be at least the MPI rank count (%d)\n",
                         options.nz, worldSize);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localSize));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) abortRun("No CUDA-capable GPU is available", rank);
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    const int locallyOversubscribed = localSize > deviceCount ? 1 : 0;
    int anyOversubscribed = 0;
    MPI_CHECK(MPI_Allreduce(&locallyOversubscribed, &anyOversubscribed, 1, MPI_INT,
                            MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    const bool cudaAware = queryCudaAwareMpi(rank);
    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Time steps: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA GPUs\n",
                    worldSize, omp_get_max_threads());
        std::printf("MPI GPU transport: %s\n", cudaAware ? "direct CUDA buffers" : "pinned host staging");
        if (anyOversubscribed) {
            std::fprintf(stderr, "Warning: at least one node has more MPI ranks than CUDA devices\n");
        }
        std::printf("Initializing concentration field...\n");
        std::printf("Running Cahn-Hilliard simulation...\n");
    }

    const int result = runSimulation(options, rank, worldSize, cudaAware);
    MPI_Finalize();
    return result;
}
