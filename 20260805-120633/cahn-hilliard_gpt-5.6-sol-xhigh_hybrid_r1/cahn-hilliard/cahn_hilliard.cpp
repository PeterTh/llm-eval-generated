#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int kConcentrationTagLower = 100;
constexpr int kConcentrationTagUpper = 101;
constexpr int kPotentialTagLower = 200;
constexpr int kPotentialTagUpper = 201;
constexpr int kGatherTag = 300;

struct Configuration {
    std::uint64_t nx = 64;
    std::uint64_t ny = 0;
    std::uint64_t nz = 0;
    int iterations = 20;
    int validate = 0;
    int printResults = 0;
};

struct Decomposition {
    std::size_t localNz;
    std::size_t zOffset;
};

void printUsage(const char* program) {
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

bool parseUnsigned(const char* text, std::uint64_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

// Returns 0 to run, 1 for help, and 2 for an invalid command line.
int parseArguments(const int argc, char** argv, Configuration& config) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseUnsigned(argv[++i], config.nx)) return 2;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseUnsigned(argv[++i], config.ny)) return 2;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseUnsigned(argv[++i], config.nz)) return 2;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], config.iterations)) return 2;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            config.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            config.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return 1;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return 2;
        }
    }
    if (config.ny == 0) config.ny = config.nx;
    if (config.nz == 0) config.nz = config.nx;
    return 0;
}

void mpiCheck(const int result, const char* operation, const int rank) {
    if (result == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(result, message, &length);
    std::fprintf(stderr, "Rank %d: %s failed: %.*s\n", rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, result);
    std::abort();
}

void cudaCheck(const cudaError_t result, const char* operation, const int rank) {
    if (result == cudaSuccess) return;
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, cudaGetErrorString(result));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(result));
    std::abort();
}

#define MPI_CHECK(call, rank) mpiCheck((call), #call, (rank))
#define CUDA_CHECK(call, rank) cudaCheck((call), #call, (rank))

Decomposition decompose(const std::size_t nz, const int size, const int rank) {
    const std::size_t base = nz / static_cast<std::size_t>(size);
    const std::size_t remainder = nz % static_cast<std::size_t>(size);
    const std::size_t rankValue = static_cast<std::size_t>(rank);
    return {base + (rankValue < remainder ? 1U : 0U),
            rankValue * base + std::min(rankValue, remainder)};
}

class DeviceFields {
  public:
    DeviceFields(const std::size_t elements, const int rank) : rank_(rank) {
        const std::size_t bytes = elements * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cold), bytes), rank_);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cnew), bytes), rank_);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mu), bytes), rank_);
    }

    DeviceFields(const DeviceFields&) = delete;
    DeviceFields& operator=(const DeviceFields&) = delete;

    ~DeviceFields() {
        if (cold != nullptr) cudaFree(cold);
        if (cnew != nullptr) cudaFree(cnew);
        if (mu != nullptr) cudaFree(mu);
    }

    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;

  private:
    int rank_;
};

class HaloBuffers {
  public:
    HaloBuffers(const std::size_t planeElements, const int rank) : rank_(rank) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&storage_), 4 * planeElements * sizeof(double),
                                 cudaHostAllocDefault),
                   rank_);
        sendLower = storage_;
        sendUpper = sendLower + planeElements;
        recvLower = sendUpper + planeElements;
        recvUpper = recvLower + planeElements;
    }

    HaloBuffers(const HaloBuffers&) = delete;
    HaloBuffers& operator=(const HaloBuffers&) = delete;

    ~HaloBuffers() {
        if (storage_ != nullptr) cudaFreeHost(storage_);
    }

    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* recvLower = nullptr;
    double* recvUpper = nullptr;

  private:
    int rank_;
    double* storage_ = nullptr;
};

__device__ __forceinline__ double laplacian(const double* __restrict__ values,
                                             const std::size_t nx,
                                             const std::size_t ny,
                                             const std::size_t plane,
                                             const std::size_t x,
                                             const std::size_t y,
                                             const std::size_t z,
                                             const double inverseDx2,
                                             const double inverseDy2,
                                             const double inverseDz2) {
    const std::size_t centerIndex = z * plane + y * nx + x;
    const double center = values[centerIndex];
    const double left = values[x == 0 ? centerIndex : centerIndex - 1];
    const double right = values[x + 1 == nx ? centerIndex : centerIndex + 1];
    const double down = values[y == 0 ? centerIndex : centerIndex - nx];
    const double up = values[y + 1 == ny ? centerIndex : centerIndex + nx];
    return (left + right - 2.0 * center) * inverseDx2 +
           (down + up - 2.0 * center) * inverseDy2 +
           (values[centerIndex - plane] + values[centerIndex + plane] - 2.0 * center) * inverseDz2;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ potential,
                                        const std::size_t nx,
                                        const std::size_t ny,
                                        const std::size_t plane,
                                        const std::size_t firstZ,
                                        const std::size_t zCount,
                                        const double inverseDx2,
                                        const double inverseDy2,
                                        const double inverseDz2,
                                        const double gamma) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t z = firstZ + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= firstZ + zCount) return;

    const std::size_t index = z * plane + y * nx + x;
    const double value = concentration[index];
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    potential[index] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB) +
                       3.0 * value + value * value * value -
                       gamma * laplacian(concentration, nx, ny, plane, x, y, z,
                                         inverseDx2, inverseDy2, inverseDz2);
}

__global__ void chemicalPotentialBoundaryKernel(const double* __restrict__ concentration,
                                                double* __restrict__ potential,
                                                const std::size_t nx,
                                                const std::size_t ny,
                                                const std::size_t plane,
                                                const std::size_t localNz,
                                                const double inverseDx2,
                                                const double inverseDy2,
                                                const double inverseDz2,
                                                const double gamma) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t z = blockIdx.z == 0 ? 1 : localNz;
    const std::size_t index = z * plane + y * nx + x;
    const double value = concentration[index];
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    potential[index] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB) +
                       3.0 * value + value * value * value -
                       gamma * laplacian(concentration, nx, ny, plane, x, y, z,
                                         inverseDx2, inverseDy2, inverseDz2);
}

__global__ void updateKernel(const double* __restrict__ concentration,
                             const double* __restrict__ potential,
                             double* __restrict__ result,
                             const std::size_t nx,
                             const std::size_t ny,
                             const std::size_t plane,
                             const std::size_t firstZ,
                             const std::size_t zCount,
                             const double inverseDx2,
                             const double inverseDy2,
                             const double inverseDz2,
                             const double dtTimesD) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t z = firstZ + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= firstZ + zCount) return;

    const std::size_t index = z * plane + y * nx + x;
    result[index] = concentration[index] +
                    dtTimesD * laplacian(potential, nx, ny, plane, x, y, z,
                                         inverseDx2, inverseDy2, inverseDz2);
}

__global__ void updateBoundaryKernel(const double* __restrict__ concentration,
                                     const double* __restrict__ potential,
                                     double* __restrict__ result,
                                     const std::size_t nx,
                                     const std::size_t ny,
                                     const std::size_t plane,
                                     const std::size_t localNz,
                                     const double inverseDx2,
                                     const double inverseDy2,
                                     const double inverseDz2,
                                     const double dtTimesD) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t z = blockIdx.z == 0 ? 1 : localNz;
    const std::size_t index = z * plane + y * nx + x;
    result[index] = concentration[index] +
                    dtTimesD * laplacian(potential, nx, ny, plane, x, y, z,
                                         inverseDx2, inverseDy2, inverseDz2);
}

void beginHaloCopies(double* field,
                     const std::size_t localNz,
                     const std::size_t plane,
                     const int previous,
                     const int next,
                     HaloBuffers& buffers,
                     const bool cudaAwareMpi,
                     const cudaStream_t stream,
                     const int rank) {
    const std::size_t bytes = plane * sizeof(double);
    if (previous == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field, field + plane, bytes, cudaMemcpyDeviceToDevice, stream), rank);
    } else if (!cudaAwareMpi) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendLower, field + plane, bytes, cudaMemcpyDeviceToHost, stream), rank);
    }
    if (next == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, field + localNz * plane, bytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   rank);
    } else if (!cudaAwareMpi) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.sendUpper, field + localNz * plane, bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   rank);
    }
}

void finishHaloExchange(double* field,
                        const std::size_t localNz,
                        const std::size_t plane,
                        const int previous,
                        const int next,
                        const int lowerTag,
                        const int upperTag,
                        HaloBuffers& buffers,
                        const bool cudaAwareMpi,
                        const cudaStream_t stream,
                        const int rank) {
    CUDA_CHECK(cudaStreamSynchronize(stream), rank);

    MPI_Request requests[4];
    int requestCount = 0;
    const int count = static_cast<int>(plane);
    if (previous != MPI_PROC_NULL) {
        double* const receive = cudaAwareMpi ? field : buffers.recvLower;
        double* const send = cudaAwareMpi ? field + plane : buffers.sendLower;
        MPI_CHECK(MPI_Irecv(receive, count, MPI_DOUBLE, previous, upperTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
        MPI_CHECK(MPI_Isend(send, count, MPI_DOUBLE, previous, lowerTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
    }
    if (next != MPI_PROC_NULL) {
        double* const receive = cudaAwareMpi ? field + (localNz + 1) * plane : buffers.recvUpper;
        double* const send = cudaAwareMpi ? field + localNz * plane : buffers.sendUpper;
        MPI_CHECK(MPI_Irecv(receive, count, MPI_DOUBLE, next, lowerTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
        MPI_CHECK(MPI_Isend(send, count, MPI_DOUBLE, next, upperTag,
                            MPI_COMM_WORLD, &requests[requestCount++]),
                  rank);
    }
    if (requestCount != 0) {
        MPI_CHECK(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE), rank);
    }

    const std::size_t bytes = plane * sizeof(double);
    if (!cudaAwareMpi && previous != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field, buffers.recvLower, bytes, cudaMemcpyHostToDevice, stream), rank);
    }
    if (!cudaAwareMpi && next != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, buffers.recvUpper, bytes,
                                   cudaMemcpyHostToDevice, stream),
                   rank);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream), rank);
}

void launchChemicalInterior(const double* concentration,
                            double* potential,
                            const std::size_t nx,
                            const std::size_t ny,
                            const std::size_t plane,
                            const std::size_t localNz,
                            const cudaStream_t stream,
                            const int rank) {
    if (localNz <= 2) return;
    constexpr dim3 block(32, 4, 2);
    const std::size_t zCount = localNz - 2;
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((zCount + block.z - 1) / block.z));
    chemicalPotentialKernel<<<grid, block, 0, stream>>>(concentration, potential, nx, ny, plane, 2, zCount,
                                                         1.0, 1.0, 1.0, 0.5);
    CUDA_CHECK(cudaGetLastError(), rank);
}

void launchChemicalBoundary(const double* concentration,
                            double* potential,
                            const std::size_t nx,
                            const std::size_t ny,
                            const std::size_t plane,
                            const std::size_t localNz,
                            const cudaStream_t stream,
                            const int rank) {
    constexpr dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    localNz == 1 ? 1U : 2U);
    chemicalPotentialBoundaryKernel<<<grid, block, 0, stream>>>(concentration, potential, nx, ny, plane,
                                                                 localNz, 1.0, 1.0, 1.0, 0.5);
    CUDA_CHECK(cudaGetLastError(), rank);
}

void launchUpdateInterior(const double* concentration,
                          const double* potential,
                          double* result,
                          const std::size_t nx,
                          const std::size_t ny,
                          const std::size_t plane,
                          const std::size_t localNz,
                          const cudaStream_t stream,
                          const int rank) {
    if (localNz <= 2) return;
    constexpr dim3 block(32, 4, 2);
    const std::size_t zCount = localNz - 2;
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((zCount + block.z - 1) / block.z));
    updateKernel<<<grid, block, 0, stream>>>(concentration, potential, result, nx, ny, plane, 2, zCount,
                                             1.0, 1.0, 1.0, 0.01);
    CUDA_CHECK(cudaGetLastError(), rank);
}

void launchUpdateBoundary(const double* concentration,
                          const double* potential,
                          double* result,
                          const std::size_t nx,
                          const std::size_t ny,
                          const std::size_t plane,
                          const std::size_t localNz,
                          const cudaStream_t stream,
                          const int rank) {
    constexpr dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    localNz == 1 ? 1U : 2U);
    updateBoundaryKernel<<<grid, block, 0, stream>>>(concentration, potential, result, nx, ny, plane,
                                                     localNz, 1.0, 1.0, 1.0, 0.01);
    CUDA_CHECK(cudaGetLastError(), rank);
}

std::vector<double> copyOwnedResult(const double* device,
                                    const std::size_t localCells,
                                    const std::size_t plane,
                                    const int rank) {
    std::vector<double> result(localCells);
    CUDA_CHECK(cudaMemcpy(result.data(), device + plane, localCells * sizeof(double), cudaMemcpyDeviceToHost), rank);
    return result;
}

std::vector<double> gatherResult(const std::vector<double>& local,
                                 const std::size_t nx,
                                 const std::size_t ny,
                                 const std::size_t nz,
                                 const int rank,
                                 const int size) {
    const std::size_t plane = nx * ny;
    std::vector<double> global;
    if (rank == 0) {
        global.resize(plane * nz);
        std::copy(local.begin(), local.end(), global.begin());
        for (int source = 1; source < size; ++source) {
            const Decomposition sourceDomain = decompose(nz, size, source);
            std::size_t remaining = sourceDomain.localNz * plane;
            std::size_t offset = sourceDomain.zOffset * plane;
            while (remaining != 0) {
                const int chunk = static_cast<int>(std::min<std::size_t>(
                    remaining, static_cast<std::size_t>(std::numeric_limits<int>::max())));
                MPI_CHECK(MPI_Recv(global.data() + offset, chunk, MPI_DOUBLE, source, kGatherTag,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                          rank);
                offset += static_cast<std::size_t>(chunk);
                remaining -= static_cast<std::size_t>(chunk);
            }
        }
    } else {
        std::size_t remaining = local.size();
        std::size_t offset = 0;
        while (remaining != 0) {
            const int chunk = static_cast<int>(std::min<std::size_t>(
                remaining, static_cast<std::size_t>(std::numeric_limits<int>::max())));
            MPI_CHECK(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, kGatherTag, MPI_COMM_WORLD), rank);
            offset += static_cast<std::size_t>(chunk);
            remaining -= static_cast<std::size_t>(chunk);
        }
    }
    return global;
}

bool validateDistributed(const std::vector<double>& local, const int rank) {
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    int localInvalid = 0;

#pragma omp parallel for schedule(static) reduction(min : localMin) reduction(max : localMax) reduction(| : localInvalid)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(local.size()); ++index) {
        const double value = local[static_cast<std::size_t>(index)];
        if (!std::isfinite(value)) {
            localInvalid = 1;
        } else {
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    int globalInvalid = 0;
    MPI_CHECK(MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD), rank);
    MPI_CHECK(MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), rank);
    MPI_CHECK(MPI_Reduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD), rank);

    int valid = 1;
    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        }
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD), rank);
    return valid != 0;
}

int runBenchmark(const Configuration& config,
                 const int rank,
                 const int size,
                 const int localRank,
                 const int localSize,
                 const int device,
                 const bool cudaAwareMpi) {
    const std::size_t nx = static_cast<std::size_t>(config.nx);
    const std::size_t ny = static_cast<std::size_t>(config.ny);
    const std::size_t nz = static_cast<std::size_t>(config.nz);
    const std::size_t plane = nx * ny;
    const std::size_t globalCells = plane * nz;
    const Decomposition domain = decompose(nz, size, rank);
    const std::size_t localCells = domain.localNz * plane;
    const std::size_t allocatedCells = (domain.localNz + 2) * plane;

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device), rank);
    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", config.iterations);
        std::printf("Validation: %s\n", config.validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "%d rank(s) on rank 0's node\n",
                    size, omp_get_max_threads(), localSize);
        std::printf("Rank 0 GPU: %s (device %d, local rank %d)\n", properties.name, device, localRank);
        std::printf("MPI GPU transport: %s\n", cudaAwareMpi ? "direct device buffers" : "pinned host staging");
        std::printf("Initializing concentration field...\n");
    }

    DeviceFields fields(allocatedCells, rank);
    HaloBuffers halo(plane, rank);
    cudaStream_t computeStream = nullptr;
    cudaStream_t communicationStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking), rank);
    CUDA_CHECK(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking), rank);
    cudaEvent_t potentialReady = nullptr;
    cudaEvent_t concentrationReady = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&potentialReady, cudaEventDisableTiming), rank);
    CUDA_CHECK(cudaEventCreateWithFlags(&concentrationReady, cudaEventDisableTiming), rank);

    std::vector<double> initial(localCells);
#pragma omp parallel for schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(localCells); ++index) {
        const std::size_t globalIndex = domain.zOffset * plane + static_cast<std::size_t>(index);
        const double pseudo = static_cast<double>(((globalIndex + 1) * std::size_t{1299709}) % globalCells) /
                              static_cast<double>(globalCells);
        initial[static_cast<std::size_t>(index)] = -1.0 + 2.0 * pseudo;
    }
    CUDA_CHECK(cudaMemcpy(fields.cold + plane, initial.data(), localCells * sizeof(double), cudaMemcpyHostToDevice), rank);
    initial.clear();
    initial.shrink_to_fit();

    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == size ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD), rank);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < config.iterations; ++iteration) {
        if (iteration != 0) {
            CUDA_CHECK(cudaStreamWaitEvent(communicationStream, concentrationReady, 0), rank);
        }
        beginHaloCopies(fields.cold, domain.localNz, plane, previous, next, halo, cudaAwareMpi,
                        communicationStream, rank);
        launchChemicalInterior(fields.cold, fields.mu, nx, ny, plane, domain.localNz, computeStream, rank);
        finishHaloExchange(fields.cold, domain.localNz, plane, previous, next,
                           kConcentrationTagLower, kConcentrationTagUpper, halo, cudaAwareMpi,
                           communicationStream, rank);
        launchChemicalBoundary(fields.cold, fields.mu, nx, ny, plane, domain.localNz, computeStream, rank);

        // Boundary potential values depend on the concentration halo. Recording this event lets
        // the transfer stream wait without serializing unrelated host work or the whole device.
        CUDA_CHECK(cudaEventRecord(potentialReady, computeStream), rank);
        CUDA_CHECK(cudaStreamWaitEvent(communicationStream, potentialReady, 0), rank);
        beginHaloCopies(fields.mu, domain.localNz, plane, previous, next, halo, cudaAwareMpi,
                        communicationStream, rank);
        launchUpdateInterior(fields.cold, fields.mu, fields.cnew, nx, ny, plane, domain.localNz,
                             computeStream, rank);
        finishHaloExchange(fields.mu, domain.localNz, plane, previous, next,
                           kPotentialTagLower, kPotentialTagUpper, halo, cudaAwareMpi,
                           communicationStream, rank);
        launchUpdateBoundary(fields.cold, fields.mu, fields.cnew, nx, ny, plane, domain.localNz,
                             computeStream, rank);
        CUDA_CHECK(cudaEventRecord(concentrationReady, computeStream), rank);
        std::swap(fields.cold, fields.cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream), rank);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD), rank);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), rank);

    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        const double updates = static_cast<double>(globalCells) * static_cast<double>(config.iterations);
        const double mcups = seconds > 0.0 ? updates / seconds / 1.0e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> localResult;
    if (config.printResults || config.validate) {
        localResult = copyOwnedResult(fields.cold, localCells, plane, rank);
    }
    if (config.printResults) {
        std::vector<double> globalResult = gatherResult(localResult, nx, ny, nz, rank, size);
        if (rank == 0) print_results(globalResult, "Concentration");
    }

    bool valid = true;
    if (config.validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localResult, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaEventDestroy(potentialReady), rank);
    CUDA_CHECK(cudaEventDestroy(concentrationReady), rank);
    CUDA_CHECK(cudaStreamDestroy(computeStream), rank);
    CUDA_CHECK(cudaStreamDestroy(communicationStream), rank);
    return valid ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initResult = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initResult != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }

    int rank = 0;
    int size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), rank);
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), rank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required MPI_THREAD_FUNNELED support\n");
        MPI_Finalize();
        return 1;
    }

    Configuration config;
    int parseResult = 0;
    if (rank == 0) {
        parseResult = parseArguments(argc, argv, config);
        if (parseResult != 0) printUsage(argv[0]);
    }
    MPI_CHECK(MPI_Bcast(&parseResult, 1, MPI_INT, 0, MPI_COMM_WORLD), rank);
    if (parseResult != 0) {
        MPI_Finalize();
        return parseResult == 1 ? 0 : 1;
    }
    MPI_CHECK(MPI_Bcast(&config, static_cast<int>(sizeof(config)), MPI_BYTE, 0, MPI_COMM_WORLD), rank);

    int inputValid = 1;
    if (config.nx == 0 || config.ny == 0 || config.nz == 0 ||
        config.nx > std::numeric_limits<std::size_t>::max() ||
        config.ny > std::numeric_limits<std::size_t>::max() ||
        config.nz > std::numeric_limits<std::size_t>::max()) {
        inputValid = 0;
    }
    if (inputValid) {
        const std::size_t nx = static_cast<std::size_t>(config.nx);
        const std::size_t ny = static_cast<std::size_t>(config.ny);
        const std::size_t nz = static_cast<std::size_t>(config.nz);
        if (nx > std::numeric_limits<std::size_t>::max() / ny) {
            inputValid = 0;
        } else {
            const std::size_t plane = nx * ny;
            if (plane > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                plane > std::numeric_limits<std::size_t>::max() / nz ||
                static_cast<std::size_t>(size) > nz) {
                inputValid = 0;
            } else {
                const std::size_t maxDoubleElements =
                    std::numeric_limits<std::size_t>::max() / sizeof(double);
                const std::size_t globalCells = plane * nz;
                const std::size_t largestLocalNz = decompose(nz, size, 0).localNz;
                const std::size_t maximumLayers = maxDoubleElements / plane;
                if (globalCells > maxDoubleElements ||
                    globalCells > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
                    maximumLayers < 3 || largestLocalNz > maximumLayers - 2) {
                    inputValid = 0;
                }
            }
        }
    }
    if (!inputValid) {
        if (rank == 0) {
            std::fprintf(stderr, "Invalid grid: dimensions must be positive, fit in memory, each Z slab must "
                                 "fit an MPI message, and nz must be at least the MPI rank count\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localCommunicator),
              rank);
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank), rank);
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localSize), rank);

    int deviceCount = 0;
    const cudaError_t countResult = cudaGetDeviceCount(&deviceCount);
    const int localGpuFailure = countResult != cudaSuccess || deviceCount == 0;
    int anyGpuFailure = 0;
    MPI_CHECK(MPI_Allreduce(&localGpuFailure, &anyGpuFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), rank);
    if (anyGpuFailure != 0) {
        if (rank == 0) std::fprintf(stderr, "Every MPI rank requires access to a CUDA-capable GPU\n");
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), rank);
    CUDA_CHECK(cudaFree(nullptr), rank);  // Create the context before allocating or timing.

    int localCudaAware = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    localCudaAware = MPIX_Query_cuda_support();
#endif
    int cudaAwareMpi = 0;
    MPI_CHECK(MPI_Allreduce(&localCudaAware, &cudaAwareMpi, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), rank);

    const int result = runBenchmark(config, rank, size, localRank, localSize, device, cudaAwareMpi != 0);
    MPI_CHECK(MPI_Comm_free(&localCommunicator), rank);
    MPI_CHECK(MPI_Finalize(), rank);
    return result;
}
