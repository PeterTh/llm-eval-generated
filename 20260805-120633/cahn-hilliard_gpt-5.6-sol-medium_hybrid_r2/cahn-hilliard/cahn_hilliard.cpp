#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

int worldRank = 0;

[[noreturn]] void fatal(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA failure in %s: %s", operation,
                      cudaGetErrorString(error));
        fatal(message);
    }
}

void checkMpi(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char mpiMessage[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, mpiMessage, &length);
        char message[768];
        std::snprintf(message, sizeof(message), "MPI failure in %s: %.*s", operation,
                      length, mpiMessage);
        fatal(message);
    }
}

bool parseSize(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
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

struct Slab {
    size_t firstZ;
    size_t nz;
};

Slab rankSlab(size_t globalNz, int rank, int ranks) {
    const size_t base = globalNz / static_cast<size_t>(ranks);
    const size_t remainder = globalNz % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, remainder), base + (r < remainder ? 1U : 0U)};
}

bool mpiSupportsDeviceBuffers() {
    // Open MPI/HPC-X exposes a reliable compile- and run-time query. Other MPI
    // implementations take the fully portable pinned-host path below.
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            size_t index, size_t x, size_t y,
                                            size_t nx, size_t ny,
                                            double invDx2, double invDy2,
                                            double invDz2) {
    const size_t plane = nx * ny;
    const size_t left = (x == 0) ? index : index - 1;
    const size_t right = (x + 1 == nx) ? index : index + 1;
    const size_t down = (y == 0) ? index : index - nx;
    const size_t up = (y + 1 == ny) ? index : index + nx;
    const double center = field[index];
    return (field[left] + field[right] - 2.0 * center) * invDx2 +
           (field[down] + field[up] - 2.0 * center) * invDy2 +
           (field[index - plane] + field[index + plane] - 2.0 * center) * invDz2;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ chemicalPotential,
                                        size_t nx, size_t ny, double invDx2,
                                        double invDy2, double invDz2, double gamma,
                                        double eAA, double eBB, double eAB,
                                        size_t firstLocalZ, size_t planeCount) {
    const size_t plane = nx * ny;
    const size_t cell = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = plane * planeCount;
    if (cell >= cells) return;
    const size_t z = firstLocalZ + cell / plane;
    const size_t inPlane = cell % plane;
    const size_t x = inPlane % nx;
    const size_t y = inPlane / nx;
    const size_t index = z * plane + inPlane;
    const double c = concentration[index];
    chemicalPotential[index] =
        4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB) +
        3.0 * c + c * c * c -
        gamma * laplacian(concentration, index, x, y, nx, ny,
                          invDx2, invDy2, invDz2);
}

__global__ void updateKernel(double* __restrict__ updated,
                             const double* __restrict__ concentration,
                             const double* __restrict__ chemicalPotential,
                             size_t nx, size_t ny, double scale, double invDx2,
                             double invDy2, double invDz2,
                             size_t firstLocalZ, size_t planeCount) {
    const size_t plane = nx * ny;
    const size_t cell = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = plane * planeCount;
    if (cell >= cells) return;
    const size_t z = firstLocalZ + cell / plane;
    const size_t inPlane = cell % plane;
    const size_t x = inPlane % nx;
    const size_t y = inPlane / nx;
    const size_t index = z * plane + inPlane;
    updated[index] = concentration[index] +
                     scale * laplacian(chemicalPotential, index, x, y, nx, ny,
                                       invDx2, invDy2, invDz2);
}

template <typename Kernel, typename... Arguments>
void launchPlanes(Kernel kernel, cudaStream_t stream, size_t plane,
                  size_t firstZ, size_t count, Arguments... arguments) {
    if (count == 0) return;
    constexpr unsigned threads = 256;
    const size_t cells = plane * count;
    const size_t blocks = (cells + threads - 1) / threads;
    if (blocks > static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
        fatal("local CUDA launch grid is too large");
    }
    kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(arguments..., firstZ, count);
    checkCuda(cudaGetLastError(), "kernel launch");
}

class HaloExchange {
public:
    HaloExchange(size_t planeElements, int previous, int next, bool deviceAware)
        : plane_(planeElements), previous_(previous), next_(next), deviceAware_(deviceAware) {
        if (plane_ > static_cast<size_t>(INT_MAX)) {
            fatal("an MPI halo plane exceeds MPI's count limit");
        }
        if (!deviceAware_) {
            checkCuda(cudaHostAlloc(&sendLower_, plane_ * sizeof(double), cudaHostAllocDefault),
                      "cudaHostAlloc(send lower)");
            checkCuda(cudaHostAlloc(&sendUpper_, plane_ * sizeof(double), cudaHostAllocDefault),
                      "cudaHostAlloc(send upper)");
            checkCuda(cudaHostAlloc(&recvLower_, plane_ * sizeof(double), cudaHostAllocDefault),
                      "cudaHostAlloc(receive lower)");
            checkCuda(cudaHostAlloc(&recvUpper_, plane_ * sizeof(double), cudaHostAllocDefault),
                      "cudaHostAlloc(receive upper)");
        }
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "cudaStreamCreate(communication)");
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        cudaStreamDestroy(stream_);
        if (!deviceAware_) {
            cudaFreeHost(sendLower_);
            cudaFreeHost(sendUpper_);
            cudaFreeHost(recvLower_);
            cudaFreeHost(recvUpper_);
        }
    }

    void begin(double* deviceField, size_t localNz, int tagBase) {
        field_ = deviceField;
        const size_t bytes = plane_ * sizeof(double);
        void* receiveLower = nullptr;
        void* receiveUpper = nullptr;
        const void* sendLower = nullptr;
        const void* sendUpper = nullptr;
        if (deviceAware_) {
            receiveLower = deviceField;
            receiveUpper = deviceField + (localNz + 1) * plane_;
            sendLower = deviceField + plane_;
            sendUpper = deviceField + localNz * plane_;
            if (previous_ == MPI_PROC_NULL) {
                checkCuda(cudaMemcpyAsync(receiveLower, sendLower, bytes,
                                          cudaMemcpyDeviceToDevice, stream_),
                          "clamp lower device halo");
            }
            if (next_ == MPI_PROC_NULL) {
                checkCuda(cudaMemcpyAsync(receiveUpper, sendUpper, bytes,
                                          cudaMemcpyDeviceToDevice, stream_),
                          "clamp upper device halo");
            }
            checkCuda(cudaStreamSynchronize(stream_), "prepare device MPI halos");
        } else {
            checkCuda(cudaMemcpyAsync(sendLower_, deviceField + plane_, bytes,
                                      cudaMemcpyDeviceToHost, stream_), "copy lower send halo");
            checkCuda(cudaMemcpyAsync(sendUpper_, deviceField + localNz * plane_, bytes,
                                      cudaMemcpyDeviceToHost, stream_), "copy upper send halo");
            checkCuda(cudaStreamSynchronize(stream_), "prepare MPI halos");
            if (previous_ == MPI_PROC_NULL) std::memcpy(recvLower_, sendLower_, bytes);
            if (next_ == MPI_PROC_NULL) std::memcpy(recvUpper_, sendUpper_, bytes);
            receiveLower = recvLower_;
            receiveUpper = recvUpper_;
            sendLower = sendLower_;
            sendUpper = sendUpper_;
        }

        checkMpi(MPI_Irecv(receiveLower, static_cast<int>(plane_), MPI_DOUBLE, previous_,
                           tagBase + 1, MPI_COMM_WORLD, &requests_[0]), "MPI_Irecv(lower)");
        checkMpi(MPI_Irecv(receiveUpper, static_cast<int>(plane_), MPI_DOUBLE, next_,
                           tagBase, MPI_COMM_WORLD, &requests_[1]), "MPI_Irecv(upper)");
        checkMpi(MPI_Isend(sendLower, static_cast<int>(plane_), MPI_DOUBLE, previous_,
                           tagBase, MPI_COMM_WORLD, &requests_[2]), "MPI_Isend(lower)");
        checkMpi(MPI_Isend(sendUpper, static_cast<int>(plane_), MPI_DOUBLE, next_,
                           tagBase + 1, MPI_COMM_WORLD, &requests_[3]), "MPI_Isend(upper)");
    }

    void finish(size_t localNz, cudaStream_t computeStream) {
        checkMpi(MPI_Waitall(4, requests_, MPI_STATUSES_IGNORE), "MPI_Waitall(halos)");
        if (deviceAware_) return;
        const size_t bytes = plane_ * sizeof(double);
        checkCuda(cudaMemcpyAsync(field_, recvLower_, bytes, cudaMemcpyHostToDevice, stream_),
                  "copy lower receive halo");
        checkCuda(cudaMemcpyAsync(field_ + (localNz + 1) * plane_, recvUpper_, bytes,
                                  cudaMemcpyHostToDevice, stream_), "copy upper receive halo");
        checkCuda(cudaEventRecord(ready_, stream_), "record halo event");
        checkCuda(cudaStreamWaitEvent(computeStream, ready_, 0), "wait for halo event");
    }

    void createEvent() {
        if (deviceAware_) return;
        checkCuda(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming),
                  "cudaEventCreate(halo)");
    }

    void destroyEvent() {
        if (!deviceAware_) cudaEventDestroy(ready_);
    }

private:
    size_t plane_;
    int previous_;
    int next_;
    bool deviceAware_;
    double* sendLower_ = nullptr;
    double* sendUpper_ = nullptr;
    double* recvLower_ = nullptr;
    double* recvUpper_ = nullptr;
    double* field_ = nullptr;
    cudaStream_t stream_{};
    cudaEvent_t ready_{};
    MPI_Request requests_[4]{};
};

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
    int worldSize = 1;
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");
    if (provided < MPI_THREAD_FUNNELED) fatal("MPI does not provide required thread support");

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            argumentsValid &= parseSize(argv[++i], nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            argumentsValid &= parseSize(argv[++i], ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            argumentsValid &= parseSize(argv[++i], nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid &= parseIterations(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            argumentsValid = false;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || !argumentsValid) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }
    if (nz < static_cast<size_t>(worldSize)) fatal("grid Z dimension must be at least the MPI rank count");
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        fatal("grid dimensions overflow size_t");
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                 MPI_INFO_NULL, &localComm), "MPI_Comm_split_type");
    int localRank = 0;
    checkMpi(MPI_Comm_rank(localComm, &localRank), "MPI local rank");
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) fatal("no CUDA device is available");
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
    checkMpi(MPI_Comm_free(&localComm), "MPI_Comm_free(local)");

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const Slab slab = rankSlab(nz, worldRank, worldSize);
    const size_t localCells = slab.nz * plane;
    if (localCells > static_cast<size_t>(INT_MAX)) {
        fatal("local slab exceeds MPI_Gatherv's count limit");
    }

    if (worldRank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize, omp_get_max_threads());
        std::printf("CUDA devices per node: %d\n", deviceCount);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> localHost(localCells);
    const double inverseVolume = 1.0 / static_cast<double>(gridSize);
    #pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(localCells); ++local) {
        const size_t global = slab.firstZ * plane + static_cast<size_t>(local);
        const double pseudo = static_cast<double>(((global + 1) * size_t{1299709}) % gridSize) * inverseVolume;
        localHost[static_cast<size_t>(local)] = -1.0 + 2.0 * pseudo;
    }

    const size_t allocatedCells = (slab.nz + 2) * plane;
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    checkCuda(cudaMalloc(&cold, allocatedCells * sizeof(double)), "cudaMalloc(concentration)");
    checkCuda(cudaMalloc(&cnew, allocatedCells * sizeof(double)), "cudaMalloc(updated)");
    checkCuda(cudaMalloc(&mu, allocatedCells * sizeof(double)), "cudaMalloc(chemical potential)");
    checkCuda(cudaMemcpy(cold + plane, localHost.data(), localCells * sizeof(double),
                         cudaMemcpyHostToDevice), "copy initial concentration");
    localHost.clear();
    localHost.shrink_to_fit();

    cudaStream_t computeStream{};
    checkCuda(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking),
              "cudaStreamCreate(compute)");
    const int previous = worldRank == 0 ? MPI_PROC_NULL : worldRank - 1;
    const int next = worldRank + 1 == worldSize ? MPI_PROC_NULL : worldRank + 1;
    const bool deviceAwareMpi = mpiSupportsDeviceBuffers();
    HaloExchange halos(plane, previous, next, deviceAwareMpi);
    halos.createEvent();
    if (worldRank == 0) {
        std::printf("MPI halo transport: %s\n",
                    deviceAwareMpi ? "CUDA-aware direct" : "pinned-host staged");
    }

    constexpr double dx = 1.0, dy = 1.0, dz = 1.0;
    constexpr double dt = 0.01, diffusion = 1.0, gamma = 0.5;
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    constexpr double invDx2 = 1.0 / (dx * dx);
    constexpr double invDy2 = 1.0 / (dy * dy);
    constexpr double invDz2 = 1.0 / (dz * dz);

    if (worldRank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    checkCuda(cudaDeviceSynchronize(), "synchronize before timing");
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
    const double start = MPI_Wtime();

    for (int step = 0; step < iterations; ++step) {
        halos.begin(cold, slab.nz, 100);
        if (slab.nz > 2) {
            launchPlanes(chemicalPotentialKernel, computeStream, plane, 2, slab.nz - 2,
                         cold, mu, nx, ny, invDx2, invDy2, invDz2,
                         gamma, eAA, eBB, eAB);
        }
        halos.finish(slab.nz, computeStream);
        launchPlanes(chemicalPotentialKernel, computeStream, plane, 1, 1,
                     cold, mu, nx, ny, invDx2, invDy2, invDz2,
                     gamma, eAA, eBB, eAB);
        if (slab.nz > 1) {
            launchPlanes(chemicalPotentialKernel, computeStream, plane, slab.nz, 1,
                         cold, mu, nx, ny, invDx2, invDy2, invDz2,
                         gamma, eAA, eBB, eAB);
        }

        // The communication stream must not copy a boundary plane until its
        // chemical-potential kernel has produced it.
        checkCuda(cudaStreamSynchronize(computeStream), "chemical potential boundary");
        halos.begin(mu, slab.nz, 200);
        if (slab.nz > 2) {
            launchPlanes(updateKernel, computeStream, plane, 2, slab.nz - 2,
                         cnew, cold, mu, nx, ny, dt * diffusion,
                         invDx2, invDy2, invDz2);
        }
        halos.finish(slab.nz, computeStream);
        launchPlanes(updateKernel, computeStream, plane, 1, 1,
                     cnew, cold, mu, nx, ny, dt * diffusion,
                     invDx2, invDy2, invDz2);
        if (slab.nz > 1) {
            launchPlanes(updateKernel, computeStream, plane, slab.nz, 1,
                         cnew, cold, mu, nx, ny, dt * diffusion,
                         invDx2, invDy2, invDz2);
        }
        // The next iteration stages cnew's end planes on a different CUDA
        // stream, so publish the complete new field before swapping buffers.
        checkCuda(cudaStreamSynchronize(computeStream), "complete update step");
        std::swap(cold, cnew);
    }

    checkCuda(cudaStreamSynchronize(computeStream), "finish simulation");
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    checkMpi(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "MPI_Reduce(time)");
    if (worldRank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double updates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = seconds > 0.0 ? updates / seconds / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    bool valid = true;
    if (validate || printResults) {
        localHost.resize(localCells);
        checkCuda(cudaMemcpy(localHost.data(), cold + plane, localCells * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy final concentration");
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> global;
        if (worldRank == 0) {
            counts.resize(worldSize);
            displacements.resize(worldSize);
            global.resize(gridSize);
            for (int rank = 0; rank < worldSize; ++rank) {
                const Slab part = rankSlab(nz, rank, worldSize);
                const size_t count = part.nz * plane;
                const size_t displacement = part.firstZ * plane;
                if (count > static_cast<size_t>(INT_MAX) || displacement > static_cast<size_t>(INT_MAX)) {
                    fatal("global grid exceeds MPI_Gatherv's integer limits");
                }
                counts[rank] = static_cast<int>(count);
                displacements[rank] = static_cast<int>(displacement);
            }
        }
        checkMpi(MPI_Gatherv(localHost.data(), static_cast<int>(localCells), MPI_DOUBLE,
                             worldRank == 0 ? global.data() : nullptr,
                             worldRank == 0 ? counts.data() : nullptr,
                             worldRank == 0 ? displacements.data() : nullptr,
                             MPI_DOUBLE, 0, MPI_COMM_WORLD), "MPI_Gatherv(results)");
        if (worldRank == 0) print_results(global, "Concentration");
    }

    if (validate) {
        int localFinite = 1;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        #pragma omp parallel for reduction(&:localFinite) reduction(min:localMin) reduction(max:localMax)
        for (long long i = 0; i < static_cast<long long>(localCells); ++i) {
            const double value = localHost[static_cast<size_t>(i)];
            localFinite &= std::isfinite(value) ? 1 : 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int allFinite = 0;
        double globalMin = 0.0, globalMax = 0.0;
        checkMpi(MPI_Allreduce(&localFinite, &allFinite, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                 "MPI_Allreduce(finite)");
        checkMpi(MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
                 "MPI_Allreduce(minimum)");
        checkMpi(MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                 "MPI_Allreduce(maximum)");
        valid = allFinite != 0 && globalMax <= 10.0 && globalMin >= -10.0;
        if (worldRank == 0) {
            std::printf("Validating result...\n");
            if (!allFinite) std::printf("Validation failed: found NaN or Inf value\n");
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (allFinite && !valid) std::printf("Validation failed: values out of expected range\n");
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    halos.destroyEvent();
    checkCuda(cudaStreamDestroy(computeStream), "cudaStreamDestroy(compute)");
    checkCuda(cudaFree(cold), "cudaFree(concentration)");
    checkCuda(cudaFree(cnew), "cudaFree(updated)");
    checkCuda(cudaFree(mu), "cudaFree(chemical potential)");
    checkMpi(MPI_Finalize(), "MPI_Finalize");
    return valid ? 0 : 1;
}
