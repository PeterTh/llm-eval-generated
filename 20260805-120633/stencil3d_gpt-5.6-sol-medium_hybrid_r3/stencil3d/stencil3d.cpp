#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

void cudaCheck(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        int initialized = 0;
        int rank = -1;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: CUDA error at %s:%d (%s): %s\n", rank, file, line,
                     expression, cudaGetErrorString(error));
        if (initialized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::exit(EXIT_FAILURE);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t localNz,
                              size_t globalZ0, size_t globalNz,
                              size_t firstLocalZ, size_t lastLocalZ) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t lz = firstLocalZ + static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lz > lastLocalZ || lz > localNz) return;

    const size_t plane = nx * ny;
    const size_t index = lz * plane + y * nx + x;
    const size_t globalZ = globalZ0 + lz - 1;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        globalZ == 0 || globalZ + 1 == globalNz) {
        output[index] = input[index];
    } else {
        output[index] = (input[index] + input[index - 1] + input[index + 1] +
                         input[index - nx] + input[index + nx] +
                         input[index - plane] + input[index + plane]) / 7.0;
    }
}

void launchPlanes(const Real* input, Real* output, size_t nx, size_t ny,
                  size_t localNz, size_t globalZ0, size_t globalNz,
                  size_t first, size_t last, cudaStream_t stream) {
    if (first > last || first == 0 || last > localNz) return;
    constexpr dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>((last - first + 1 + block.z - 1) / block.z));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, localNz,
                                              globalZ0, globalNz, first, last);
    CUDA_CHECK(cudaGetLastError());
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

bool validateResult(const std::vector<Real>& grid) {
    int invalid = 0;
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -std::numeric_limits<Real>::infinity();

#pragma omp parallel for reduction(| : invalid) reduction(min : minValue) reduction(max : maxValue) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        invalid |= !std::isfinite(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    if (invalid) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 1e6 || minValue < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (worldRank == 0) std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (help) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        nz > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        argumentsValid = false;
    }
    if (!argumentsValid) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid dimensions must be at least 2, iterations nonnegative, and each XY plane must fit an MPI count\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Empty MPI subdomains are avoided: at most one active rank per global Z plane.
    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t quotient = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = quotient + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * quotient +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t planeSize = nx * ny;
    const size_t localElementsWithHalos = (localNz + 2) * planeSize;
    const size_t bytes = localElementsWithHalos * sizeof(Real);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs/node: %d\n",
                    ranks, omp_get_max_threads(), deviceCount);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }

    // Pinned host initialization lets OpenMP fill global-indexed values efficiently
    // and allows the one-time host-to-device transfer to use full PCIe bandwidth.
    Real* hostInitial = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostInitial, bytes));
    std::fill_n(hostInitial, localElementsWithHalos, Real{0});
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 1; lz <= localNz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t globalBase = (globalZ0 + lz - 1) * planeSize + y * nx;
            const size_t localBase = lz * planeSize + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                hostInitial[localBase + x] = static_cast<Real>((globalBase + x) % 19);
            }
        }
    }

    Real* deviceA = nullptr;
    Real* deviceB = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMemcpy(deviceA, hostInitial, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceB, 0, bytes));
    CUDA_CHECK(cudaFreeHost(hostInitial));

    cudaStream_t computeStream;
    cudaStream_t transferStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    Real* haloHost = nullptr;
    CUDA_CHECK(cudaMallocHost(&haloHost, 4 * planeSize * sizeof(Real)));
    Real* const sendPrevious = haloHost;
    Real* const sendNext = haloHost + planeSize;
    Real* const receivePrevious = haloHost + 2 * planeSize;
    Real* const receiveNext = haloHost + 3 * planeSize;
    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int mpiPlaneCount = static_cast<int>(planeSize);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Stage only the boundary planes through reusable pinned storage. This
        // works with every MPI implementation, including builds without CUDA support.
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        if (previous != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendPrevious, deviceA + planeSize,
                                       planeSize * sizeof(Real), cudaMemcpyDeviceToHost,
                                       transferStream));
        }
        if (next != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(sendNext, deviceA + localNz * planeSize,
                                       planeSize * sizeof(Real), cudaMemcpyDeviceToHost,
                                       transferStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(transferStream));
        MPI_Request requests[4];
        MPI_Irecv(receivePrevious, mpiPlaneCount, MPI_DOUBLE, previous, 101, comm, &requests[0]);
        MPI_Irecv(receiveNext, mpiPlaneCount, MPI_DOUBLE, next, 100, comm, &requests[1]);
        MPI_Isend(sendPrevious, mpiPlaneCount, MPI_DOUBLE, previous, 100, comm, &requests[2]);
        MPI_Isend(sendNext, mpiPlaneCount, MPI_DOUBLE, next, 101, comm, &requests[3]);

        // Planes independent of incoming halos execute concurrently with MPI traffic.
        if (localNz > 2) {
            launchPlanes(deviceA, deviceB, nx, ny, localNz, globalZ0, nz,
                         2, localNz - 1, computeStream);
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        // These copies queue behind the independent interior kernel, so the
        // edge kernels see completed halo data without a device-wide barrier.
        if (previous != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(deviceA, receivePrevious, planeSize * sizeof(Real),
                                       cudaMemcpyHostToDevice, computeStream));
        }
        if (next != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(deviceA + (localNz + 1) * planeSize, receiveNext,
                                       planeSize * sizeof(Real), cudaMemcpyHostToDevice,
                                       computeStream));
        }
        launchPlanes(deviceA, deviceB, nx, ny, localNz, globalZ0, nz,
                     1, 1, computeStream);
        if (localNz > 1) {
            launchPlanes(deviceA, deviceB, nx, ny, localNz, globalZ0, nz,
                         localNz, localNz, computeStream);
        }
        std::swap(deviceA, deviceB);
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    MPI_Barrier(comm);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int valid = 1;
    if (printResults || validate) {
        std::vector<Real> localGrid(localNz * planeSize);
        CUDA_CHECK(cudaMemcpy(localGrid.data(), deviceA + planeSize,
                              localGrid.size() * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<Real> globalGrid;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t rNz = quotient + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t rZ0 = static_cast<size_t>(r) * quotient +
                                    std::min(static_cast<size_t>(r), remainder);
                const size_t count = rNz * planeSize;
                const size_t displacement = rZ0 * planeSize;
                if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    std::fprintf(stderr, "Result gathering exceeds MPI_Gatherv integer limits\n");
                    MPI_Abort(comm, EXIT_FAILURE);
                }
                counts[r] = static_cast<int>(count);
                displacements[r] = static_cast<int>(displacement);
            }
            globalGrid.resize(nx * ny * nz);
        }
        if (localGrid.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "Local result exceeds MPI_Gatherv integer limits\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
        MPI_Gatherv(localGrid.data(), static_cast<int>(localGrid.size()), MPI_DOUBLE,
                    rank == 0 ? globalGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);

        if (rank == 0) {
            if (printResults) print_results(globalGrid, "Grid");
            if (validate) {
                std::printf("Validating result...\n");
                valid = validateResult(globalGrid) ? 1 : 0;
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }

    CUDA_CHECK(cudaFreeHost(haloHost));
    CUDA_CHECK(cudaStreamDestroy(transferStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
