#include <algorithm>
#include <chrono>
#include <climits>
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

static void cudaCheck(cudaError_t error, const char* action) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s failed: %s\n", action, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

static void mpiCheck(int error, const char* action) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        std::fprintf(stderr, "MPI %s failed: %.*s\n", action, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define MPI_CHECK(call) mpiCheck((call), #call)

// Each rank owns z=1..localNz; planes 0 and localNz+1 are MPI ghosts.
// Global domain faces are clamped exactly as in the original stencil.
__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t localNz, size_t zOffset,
                                  size_t globalNz, double gamma, double eAA, double eBB,
                                  double eAB, size_t firstZ, size_t planeCount,
                                  bool boundaryOnly) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t p = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || p >= planeCount) return;
    const size_t z = boundaryOnly ? (p == 0 ? 1 : localNz) : firstZ + p;
    if (boundaryOnly && p == 1 && localNz == 1) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t zm = zOffset + z == 1 ? i : i - plane;
    const size_t zp = zOffset + z == globalNz ? i : i + plane;
    const double v = c[i];
    const double lap = ((c[xp] + c[xm] - 2.0 * v) / 1.0 +
                        (c[yp] + c[ym] - 2.0 * v) / 1.0) +
                       (c[zp] + c[zm] - 2.0 * v) / 1.0;
    mu[i] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
          + 3.0 * v + v * v * v - gamma * lap;
}

__global__ void updateConcentration(const double* __restrict__ cold,
                                    const double* __restrict__ mu,
                                    double* __restrict__ cnew,
                                    size_t nx, size_t ny, size_t localNz,
                                    size_t zOffset, size_t globalNz,
                                    double dt, double diffusion,
                                    size_t firstZ, size_t planeCount,
                                    bool boundaryOnly) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t p = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || p >= planeCount) return;
    const size_t z = boundaryOnly ? (p == 0 ? 1 : localNz) : firstZ + p;
    if (boundaryOnly && p == 1 && localNz == 1) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t zm = zOffset + z == 1 ? i : i - plane;
    const size_t zp = zOffset + z == globalNz ? i : i + plane;
    const double v = mu[i];
    const double lap = ((mu[xp] + mu[xm] - 2.0 * v) / 1.0 +
                        (mu[yp] + mu[ym] - 2.0 * v) / 1.0) +
                       (mu[zp] + mu[zm] - 2.0 * v) / 1.0;
    cnew[i] = cold[i] + dt * diffusion * lap;
}

struct HaloExchange {
    double* sendLow = nullptr;
    double* sendHigh = nullptr;
    double* recvLow = nullptr;
    double* recvHigh = nullptr;
    cudaStream_t compute = nullptr;
    cudaStream_t transfer = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t received = nullptr;
    size_t plane;
    size_t localNz;
    int rank;
    int ranks;
    MPI_Comm comm;

    HaloExchange(size_t planeSize, size_t ownedZ, int myRank, int numRanks, MPI_Comm communicator)
        : plane(planeSize), localNz(ownedZ), rank(myRank), ranks(numRanks), comm(communicator) {
        const size_t bytes = plane * sizeof(double);
        CUDA_CHECK(cudaHostAlloc(&sendLow, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&sendHigh, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&recvLow, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&recvHigh, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
    }

    ~HaloExchange() {
        cudaEventDestroy(ready);
        cudaEventDestroy(received);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(transfer);
        cudaFreeHost(sendLow);
        cudaFreeHost(sendHigh);
        cudaFreeHost(recvLow);
        cudaFreeHost(recvHigh);
    }

    // Transfers are staged through pinned memory, so CUDA-aware MPI is optional.
    // The interior kernel runs while neighboring ranks exchange surface planes.
    template <typename LaunchInterior>
    void exchange(double* field, int tag, LaunchInterior launchInterior) {
        if (ranks == 1) {
            launchInterior();
            return;
        }
        const size_t bytes = plane * sizeof(double);
        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready));
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(sendLow, field + plane, bytes,
                                       cudaMemcpyDeviceToHost, transfer));
        if (rank + 1 < ranks)
            CUDA_CHECK(cudaMemcpyAsync(sendHigh, field + localNz * plane, bytes,
                                       cudaMemcpyDeviceToHost, transfer));
        launchInterior();
        CUDA_CHECK(cudaStreamSynchronize(transfer));

        MPI_Request requests[4];
        int count = 0;
        // MPI counts are int; send long planes in bounded chunks.
        for (size_t offset = 0; offset < plane; offset += INT_MAX) {
            const int chunk = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
            count = 0;
            if (rank > 0) {
                MPI_CHECK(MPI_Irecv(recvLow + offset, chunk, MPI_DOUBLE, rank - 1, tag, comm, &requests[count++]));
                MPI_CHECK(MPI_Isend(sendLow + offset, chunk, MPI_DOUBLE, rank - 1, tag + 1, comm, &requests[count++]));
            }
            if (rank + 1 < ranks) {
                MPI_CHECK(MPI_Irecv(recvHigh + offset, chunk, MPI_DOUBLE, rank + 1, tag + 1, comm, &requests[count++]));
                MPI_CHECK(MPI_Isend(sendHigh + offset, chunk, MPI_DOUBLE, rank + 1, tag, comm, &requests[count++]));
            }
            MPI_CHECK(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE));
        }
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(field, recvLow, bytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks)
            CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, recvHigh, bytes,
                                       cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(received, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, received));
    }
};

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  Grid size in X (default: 64)\n");
    std::printf("  -y <num>  Grid size in Y (default: same as X)\n");
    std::printf("  -z <num>  Grid size in Z (default: same as X)\n");
    std::printf("  -i <num>  Number of time steps (default: 20)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int worldRank = 0, worldRanks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldRanks));

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    bool validArgs = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0 ||
             std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char* option = argv[i];
            char* end = nullptr;
            const long long value = std::strtoll(argv[++i], &end, 10);
            const bool isIterations = std::strcmp(option, "-i") == 0;
            if (*end != '\0' || value < (isIterations ? 0 : 1) ||
                (isIterations && value > INT_MAX)) {
                validArgs = false;
                continue;
            }
            if (std::strcmp(option, "-x") == 0) nx = static_cast<size_t>(value);
            else if (std::strcmp(option, "-y") == 0) ny = static_cast<size_t>(value);
            else if (std::strcmp(option, "-z") == 0) nz = static_cast<size_t>(value);
            else iterations = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else validArgs = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || !validArgs) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return validArgs ? 0 : 1;
    }
    if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(double)) {
        if (worldRank == 0) std::fprintf(stderr, "Grid dimensions overflow address space\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldRanks)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                             worldRank, &comm));
    if (worldRank >= activeRanks) {
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    const int rank = worldRank;
    const size_t base = nz / activeRanks;
    const size_t extra = nz % activeRanks;
    const size_t localNz = base + (static_cast<size_t>(rank) < extra);
    const size_t zOffset = static_cast<size_t>(rank) * base +
                           std::min(static_cast<size_t>(rank), extra);
    if (plane > SIZE_MAX / (localNz + 2) / sizeof(double)) {
        if (rank == 0) std::fprintf(stderr, "Local slab dimensions overflow address space\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm shared = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    int localRank = 0, deviceCount = 0;
    MPI_CHECK(MPI_Comm_rank(shared, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d has no CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&shared));

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", activeRanks);
        std::printf("Initializing concentration field...\n");
    }
    const size_t localCells = plane * localNz;
    std::vector<double> host(localCells);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localCells; ++i) {
        const size_t globalId = zOffset * plane + i;
        const double pseudo = (((globalId + 1) * size_t{1299709}) % gridSize) /
                              static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    const size_t bytes = (localNz + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), localCells * sizeof(double),
                          cudaMemcpyHostToDevice));
    HaloExchange halo(plane, localNz, rank, activeRanks, comm);
    const dim3 block(32, 4, 1);
    auto shape = [&](size_t count) {
        return dim3(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(count));
    };
    // CUDA grid.z is limited to 65535. Larger slabs are launched in chunks.
    auto launchMu = [&](size_t first, size_t count, bool boundary) {
        for (size_t offset = 0; offset < count; offset += 65535) {
            const size_t batch = std::min(count - offset, size_t{65535});
            chemicalPotential<<<shape(batch), block, 0, halo.compute>>>(
                cold, mu, nx, ny, localNz, zOffset, nz, 0.5,
                -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0,
                first + offset, batch, boundary);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    auto launchUpdate = [&](size_t first, size_t count, bool boundary) {
        for (size_t offset = 0; offset < count; offset += 65535) {
            const size_t batch = std::min(count - offset, size_t{65535});
            updateConcentration<<<shape(batch), block, 0, halo.compute>>>(
                cold, mu, cnew, nx, ny, localNz, zOffset, nz,
                0.01, 1.0, first + offset, batch, boundary);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        if (activeRanks == 1) {
            launchMu(1, localNz, false);
            launchUpdate(1, localNz, false);
        } else {
            halo.exchange(cold, 10, [&] {
                if (localNz > 2) launchMu(2, localNz - 2, false);
            });
            launchMu(0, localNz == 1 ? 1 : 2, true);
            halo.exchange(mu, 20, [&] {
                if (localNz > 2) launchUpdate(2, localNz - 2, false);
            });
            launchUpdate(0, localNz == 1 ? 1 : 2, true);
        }
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(halo.compute));
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double mcups = static_cast<double>(gridSize) * iterations / maxElapsed / 1e6;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, localCells * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        std::vector<double> global;
        if (rank == 0) {
            global.resize(gridSize);
            std::copy(host.begin(), host.end(), global.begin());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t sourceNz = base + (static_cast<size_t>(source) < extra);
                const size_t sourceOffset = static_cast<size_t>(source) * base +
                    std::min(static_cast<size_t>(source), extra);
                const size_t cells = sourceNz * plane;
                for (size_t offset = 0; offset < cells; offset += INT_MAX) {
                    const int chunk = static_cast<int>(std::min(cells - offset, static_cast<size_t>(INT_MAX)));
                    MPI_CHECK(MPI_Recv(global.data() + sourceOffset * plane + offset,
                                       chunk, MPI_DOUBLE, source, 30, comm, MPI_STATUS_IGNORE));
                }
            }
            print_results(global, "Concentration");
        } else {
            for (size_t offset = 0; offset < localCells; offset += INT_MAX) {
                const int chunk = static_cast<int>(std::min(localCells - offset, static_cast<size_t>(INT_MAX)));
                MPI_CHECK(MPI_Send(host.data() + offset, chunk, MPI_DOUBLE, 0, 30, comm));
            }
        }
    }
    int result = 0;
    if (validate) {
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        int localFinite = 1;
        #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(&:localFinite)
        for (size_t i = 0; i < localCells; ++i) {
            const double value = host[i];
            localFinite &= std::isfinite(value) ? 1 : 0;
            if (std::isfinite(value)) {
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
        double minVal = 0.0, maxVal = 0.0;
        int allFinite = 0;
        MPI_CHECK(MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
        MPI_CHECK(MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        MPI_CHECK(MPI_Reduce(&localFinite, &allFinite, 1, MPI_INT, MPI_MIN, 0, comm));
        if (rank == 0) {
            std::printf("Validating result...\n");
            if (!allFinite) std::printf("Validation failed: found NaN or Inf value\n");
            std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0)
                std::printf("Validation failed: values out of expected range\n");
            result = allFinite && maxVal <= 10.0 && minVal >= -10.0 ? 0 : 1;
            std::printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
        MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, comm));
    }
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return result;
}
