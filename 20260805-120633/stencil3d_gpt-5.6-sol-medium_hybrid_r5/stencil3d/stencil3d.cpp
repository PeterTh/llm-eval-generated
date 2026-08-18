#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

[[noreturn]] void abortRun(const int rank, const char* message) {
    if (rank == 0) {
        std::fprintf(stderr, "Error: %s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const int rank, const char* operation) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s: %s", operation,
                      cudaGetErrorString(status));
        abortRun(rank, message);
    }
}

bool mpiSupportsDevicePointers() {
#if defined(OPEN_MPI) && defined(OMPI_HAVE_MPI_EXT_CUDA)
    return MPIX_Query_cuda_support() != 0;
#else
    // No portable MPI query exists. Unknown implementations use the safe,
    // asynchronously staged path rather than risking invalid device access.
    return false;
#endif
}

size_t checkedProduct(const size_t a, const size_t b, const int rank) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        abortRun(rank, "grid dimensions overflow addressable memory");
    }
    return a * b;
}

size_t parseSize(const char* text, const char* option, const int rank) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || text[0] == '-' ||
        value > std::numeric_limits<size_t>::max()) {
        char message[128];
        std::snprintf(message, sizeof(message), "invalid value for %s", option);
        abortRun(rank, message);
    }
    return static_cast<size_t>(value);
}

int parseIterations(const char* text, const int rank) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > INT_MAX) {
        abortRun(rank, "invalid value for -i");
    }
    return static_cast<int>(value);
}

void initializeLocal(std::vector<Real>& grid, const size_t nx, const size_t ny,
                     const size_t localNz, const size_t globalFirstZ) {
    const size_t plane = nx * ny;
    // Local plane 1 is globalFirstZ, hence local plane 0 is its lower halo.
#pragma omp parallel for schedule(static)
    for (long long localZ = 0; localZ < static_cast<long long>(localNz); ++localZ) {
        const size_t globalZ = globalFirstZ + static_cast<size_t>(localZ) - 1;
        const size_t localBase = static_cast<size_t>(localZ) * plane;
        const size_t globalBase = globalZ * plane;
        for (size_t xy = 0; xy < plane; ++xy) {
            grid[localBase + xy] = static_cast<Real>((globalBase + xy) % 19);
        }
    }
}

void initializeGlobal(std::vector<Real>& grid) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        grid[static_cast<size_t>(i)] = static_cast<Real>(static_cast<size_t>(i) % 19);
    }
}

__global__ void stencilRange(const Real* __restrict__ input,
                             Real* __restrict__ output, const size_t nx,
                             const size_t ny, const size_t zBegin,
                             const size_t zEnd) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    for (size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
        const size_t index = z * plane + y * nx + x;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny) {
            output[index] = input[index];
        } else {
            output[index] = (input[index] + input[index - 1] + input[index + 1] +
                             input[index - nx] + input[index + nx] +
                             input[index - plane] + input[index + plane]) /
                            Real{7.0};
        }
    }
}

void launchRange(const Real* input, Real* output, const size_t nx, const size_t ny,
                 const size_t zBegin, const size_t zEnd, const int rank) {
    if (zBegin >= zEnd) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(std::min<size_t>(zEnd - zBegin, 65535)));
    stencilRange<<<grid, block>>>(input, output, nx, ny, zBegin, zEnd);
    checkCuda(cudaGetLastError(), rank, "launching stencil kernel");
}

bool validateDistributed(const std::vector<Real>& owned, const size_t nx,
                         const size_t ny, const size_t nz, MPI_Comm comm,
                         const int rank) {
    int invalid = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
#pragma omp parallel for reduction(| : invalid) reduction(min : localMin) reduction(max : localMax) schedule(static)
    for (long long i = 0; i < static_cast<long long>(owned.size()); ++i) {
        const Real value = owned[static_cast<size_t>(i)];
        invalid |= !std::isfinite(value);
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }
    // The distributed owned regions exclude only the two immutable global Z
    // boundary planes; rank zero accounts for them before the reductions.
    if (rank == 0) {
        const size_t plane = nx * ny;
        for (size_t xy = 0; xy < plane; ++xy) {
            const Real lower = static_cast<Real>(xy % 19);
            const Real upper = static_cast<Real>(((nz - 1) * plane + xy) % 19);
            localMin = std::min(localMin, std::min(lower, upper));
            localMax = std::max(localMax, std::max(lower, upper));
        }
    }

    int globalInvalid = 0;
    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_LOR, comm);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        if (globalInvalid) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6)
            std::printf("Validation failed: values out of expected range\n");
    }
    return !globalInvalid && globalMax <= 1e6 && globalMin >= -1e6;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED)
        abortRun(rank, "MPI does not provide MPI_THREAD_FUNNELED");

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = parseSize(argv[++i], "-x", rank);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = parseSize(argv[++i], "-y", rank);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = parseSize(argv[++i], "-z", rank);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIterations(argv[++i], rank);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3)
        abortRun(rank, "all grid dimensions must be at least 3");
    if (static_cast<size_t>(ranks) > nz - 2)
        abortRun(rank, "number of MPI ranks cannot exceed the number of interior Z planes");

    const size_t plane = checkedProduct(nx, ny, rank);
    const size_t globalSize = checkedProduct(plane, nz, rank);
    const size_t interiorZ = nz - 2;
    const size_t base = interiorZ / static_cast<size_t>(ranks);
    const size_t remainder = interiorZ % static_cast<size_t>(ranks);
    const size_t localZ = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalFirstZ = 1 + static_cast<size_t>(rank) * base +
                                std::min(static_cast<size_t>(rank), remainder);
    const size_t localElements = checkedProduct(localZ + 2, plane, rank);
    const size_t ownedElements = checkedProduct(localZ, plane, rank);
    if (plane > static_cast<size_t>(INT_MAX) || ownedElements > static_cast<size_t>(INT_MAX))
        abortRun(rank, "per-rank MPI message exceeds the implementation's count limit");

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "querying CUDA devices");
    if (deviceCount == 0) abortRun(rank, "no CUDA device is available");
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), rank, "selecting CUDA device");
    checkCuda(cudaFree(nullptr), rank, "initializing CUDA context");
    const bool cudaAwareMpi = mpiSupportsDevicePointers();

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), %d OpenMP thread(s)/rank, CUDA GPUs\n",
                    ranks, omp_get_max_threads());
        std::printf("MPI halo transport: %s\n",
                    cudaAwareMpi ? "direct CUDA device buffers" : "asynchronous pinned staging");
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> localInitial(localElements);
    initializeLocal(localInitial, nx, ny, localZ + 2, globalFirstZ);
    Real* deviceA = nullptr;
    Real* deviceB = nullptr;
    Real* staging = nullptr;
    cudaStream_t communicationStream = nullptr;
    const size_t localBytes = checkedProduct(localElements, sizeof(Real), rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), localBytes), rank,
              "allocating first device grid");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), localBytes), rank,
              "allocating second device grid");
    checkCuda(cudaMemcpy(deviceA, localInitial.data(), localBytes, cudaMemcpyHostToDevice),
              rank, "copying initial grid to first device buffer");
    checkCuda(cudaMemcpy(deviceB, localInitial.data(), localBytes, cudaMemcpyHostToDevice),
              rank, "copying initial grid to second device buffer");
    if (!cudaAwareMpi) {
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&staging),
                                checkedProduct(checkedProduct(4, plane, rank),
                                               sizeof(Real), rank),
                                cudaHostAllocDefault),
                  rank, "allocating pinned halo staging buffers");
        checkCuda(cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking),
                  rank, "creating halo communication stream");
    }
    localInitial.clear();
    localInitial.shrink_to_fit();

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = deviceA;
    Real* output = deviceB;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    for (int iter = 0; iter < iterations; ++iter) {
        // Ensure the boundary planes produced in the prior iteration are visible to MPI.
        checkCuda(cudaDeviceSynchronize(), rank, "synchronizing before halo exchange");
        MPI_Request requests[4];
        if (cudaAwareMpi) {
            MPI_Irecv(input, static_cast<int>(plane), MPI_DOUBLE, lower, 1,
                      MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(input + (localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                      upper, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(input + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 0,
                      MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(input + localZ * plane, static_cast<int>(plane), MPI_DOUBLE, upper,
                      1, MPI_COMM_WORLD, &requests[3]);
            // Planes 2..localZ-1 do not depend on incoming halo data.
            launchRange(input, output, nx, ny, 2, localZ, rank);
        } else {
            Real* lowerReceive = staging;
            Real* upperReceive = staging + plane;
            Real* lowerSend = staging + 2 * plane;
            Real* upperSend = staging + 3 * plane;
            if (lower != MPI_PROC_NULL)
                checkCuda(cudaMemcpyAsync(lowerSend, input + plane, plane * sizeof(Real),
                                          cudaMemcpyDeviceToHost, communicationStream),
                          rank, "staging lower outgoing halo");
            if (upper != MPI_PROC_NULL)
                checkCuda(cudaMemcpyAsync(upperSend, input + localZ * plane,
                                          plane * sizeof(Real), cudaMemcpyDeviceToHost,
                                          communicationStream),
                          rank, "staging upper outgoing halo");
            launchRange(input, output, nx, ny, 2, localZ, rank);
            checkCuda(cudaStreamSynchronize(communicationStream), rank,
                      "completing outgoing halo staging");
            MPI_Irecv(lowerReceive, static_cast<int>(plane), MPI_DOUBLE, lower, 1,
                      MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(upperReceive, static_cast<int>(plane), MPI_DOUBLE, upper, 0,
                      MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(lowerSend, static_cast<int>(plane), MPI_DOUBLE, lower, 0,
                      MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(upperSend, static_cast<int>(plane), MPI_DOUBLE, upper, 1,
                      MPI_COMM_WORLD, &requests[3]);
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (!cudaAwareMpi) {
            if (lower != MPI_PROC_NULL)
                checkCuda(cudaMemcpyAsync(input, staging, plane * sizeof(Real),
                                          cudaMemcpyHostToDevice, communicationStream),
                          rank, "copying lower incoming halo to device");
            if (upper != MPI_PROC_NULL)
                checkCuda(cudaMemcpyAsync(input + (localZ + 1) * plane, staging + plane,
                                          plane * sizeof(Real), cudaMemcpyHostToDevice,
                                          communicationStream),
                          rank, "copying upper incoming halo to device");
            checkCuda(cudaStreamSynchronize(communicationStream), rank,
                      "completing incoming halo staging");
        }
        launchRange(input, output, nx, ny, 1, std::min<size_t>(2, localZ + 1), rank);
        if (localZ > 1)
            launchRange(input, output, nx, ny, localZ, localZ + 1, rank);
        std::swap(input, output);
    }
    checkCuda(cudaDeviceSynchronize(), rank, "completing stencil computation");
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double cellUpdates = static_cast<double>(nx - 2) *
                                   static_cast<double>(ny - 2) *
                                   static_cast<double>(nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0);
    }

    std::vector<Real> owned;
    if (validate || printResults) {
        owned.resize(ownedElements);
        checkCuda(cudaMemcpy(owned.data(), input + plane, ownedElements * sizeof(Real),
                             cudaMemcpyDeviceToHost),
                  rank, "copying final grid to host");
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<Real> global;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t rz = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t first = 1 + static_cast<size_t>(r) * base +
                                     std::min(static_cast<size_t>(r), remainder);
                const size_t count = rz * plane;
                const size_t displacement = first * plane;
                if (count > static_cast<size_t>(INT_MAX) ||
                    displacement > static_cast<size_t>(INT_MAX))
                    abortRun(rank, "result gathering exceeds the MPI count limit");
                counts[r] = static_cast<int>(count);
                displacements[r] = static_cast<int>(displacement);
            }
            global.resize(globalSize);
            initializeGlobal(global);  // Supplies the two unchanged global Z boundaries.
        }
        MPI_Gatherv(owned.data(), static_cast<int>(ownedElements), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Grid");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(owned, nx, ny, nz, MPI_COMM_WORLD, rank) ? 1 : 0;
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    checkCuda(cudaFree(deviceA), rank, "freeing first device grid");
    checkCuda(cudaFree(deviceB), rank, "freeing second device grid");
    if (!cudaAwareMpi) {
        checkCuda(cudaStreamDestroy(communicationStream), rank,
                  "destroying halo communication stream");
        checkCuda(cudaFreeHost(staging), rank, "freeing pinned halo staging buffers");
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
