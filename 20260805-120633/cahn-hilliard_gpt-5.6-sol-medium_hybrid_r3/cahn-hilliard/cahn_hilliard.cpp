#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void cudaCheck(cudaError_t error, const char* operation, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             size_t index, size_t x, size_t y,
                                             size_t z, size_t nx, size_t ny,
                                             size_t localNz, double idx2,
                                             double idy2, double idz2) {
    const size_t slice = nx * ny;
    const size_t xm = (x == 0) ? index : index - 1;
    const size_t xp = (x + 1 == nx) ? index : index + 1;
    const size_t ym = (y == 0) ? index : index - nx;
    const size_t yp = (y + 1 == ny) ? index : index + nx;
    // Z halos have already been filled. z is in [1, localNz].
    const size_t zm = index - slice;
    const size_t zp = index + slice;
    const double center = field[index];
    return (field[xp] + field[xm] - 2.0 * center) * idx2
         + (field[yp] + field[ym] - 2.0 * center) * idy2
         + (field[zp] + field[zm] - 2.0 * center) * idz2;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                         double* __restrict__ mu, size_t nx,
                                         size_t ny, size_t localNz, double idx2,
                                         double idy2, double idz2, double gamma,
                                         double eAA, double eBB, double eAB) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = nx * ny * localNz;
    if (linear >= cells) return;

    const size_t slice = nx * ny;
    const size_t z = linear / slice + 1;
    const size_t rem = linear % slice;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const size_t index = z * slice + rem;
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
              + 3.0 * cv + cv * cv * cv
              - gamma * laplacian(c, index, x, y, z, nx, ny, localNz,
                                   idx2, idy2, idz2);
}

__global__ void updateKernel(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t nx,
                             size_t ny, size_t localNz, double dtD,
                             double idx2, double idy2, double idz2) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t cells = nx * ny * localNz;
    if (linear >= cells) return;

    const size_t slice = nx * ny;
    const size_t z = linear / slice + 1;
    const size_t rem = linear % slice;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const size_t index = z * slice + rem;
    cnew[index] = cold[index] + dtD * laplacian(mu, index, x, y, z, nx, ny,
                                                localNz, idx2, idy2, idz2);
}

struct HaloBuffers {
    double* sendLower = nullptr;
    double* sendUpper = nullptr;
    double* recvLower = nullptr;
    double* recvUpper = nullptr;
};

void exchangeHalos(double* deviceField, size_t slice, size_t localNz,
                   int lower, int upper, HaloBuffers& halo,
                   cudaStream_t stream, bool cudaAwareMpi, int rank) {
    const size_t bytes = slice * sizeof(double);
    if (cudaAwareMpi) {
        // A CUDA-aware MPI can move the two contiguous planes directly between
        // GPUs (and use GPUDirect RDMA when the MPI runtime and fabric support it).
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Request requests[4];
        MPI_Irecv(deviceField, static_cast<int>(slice), MPI_DOUBLE, lower, 101,
                  MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(deviceField + (localNz + 1) * slice, static_cast<int>(slice),
                  MPI_DOUBLE, upper, 100, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(deviceField + slice, static_cast<int>(slice), MPI_DOUBLE, lower,
                  100, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(deviceField + localNz * slice, static_cast<int>(slice),
                  MPI_DOUBLE, upper, 101, MPI_COMM_WORLD, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (lower == MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(deviceField, deviceField + slice, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        if (upper == MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(deviceField + (localNz + 1) * slice,
                                       deviceField + localNz * slice, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(halo.sendLower, deviceField + slice, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(halo.sendUpper, deviceField + localNz * slice,
                               bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Request requests[4];
    MPI_Irecv(halo.recvLower, static_cast<int>(slice), MPI_DOUBLE, lower, 101,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(halo.recvUpper, static_cast<int>(slice), MPI_DOUBLE, upper, 100,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(halo.sendLower, static_cast<int>(slice), MPI_DOUBLE, lower, 100,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(halo.sendUpper, static_cast<int>(slice), MPI_DOUBLE, upper, 101,
              MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // Clamped physical boundaries are represented by repeating the edge plane.
    if (lower == MPI_PROC_NULL) std::memcpy(halo.recvLower, halo.sendLower, bytes);
    if (upper == MPI_PROC_NULL) std::memcpy(halo.recvUpper, halo.sendUpper, bytes);
    CUDA_CHECK(cudaMemcpyAsync(deviceField, halo.recvLower, bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceField + (localNz + 1) * slice,
                               halo.recvUpper, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
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

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            argumentsValid = false;
            if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
        }
    }
    if (help || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const bool overflow = nx == 0 || ny == 0 || nz == 0 || iterations < 0
        || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz;
    if (overflow || nz < static_cast<size_t>(ranks) || nx * ny > INT_MAX) {
        if (rank == 0) {
            std::fprintf(stderr, "Invalid grid/iteration count, too many MPI ranks, or halo plane exceeds MPI count limit\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const size_t baseNz = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * baseNz
                          + std::min(static_cast<size_t>(rank), remainder);
    const size_t slice = nx * ny;
    const size_t localCells = localNz * slice;
    const size_t globalCells = nx * ny * nz;
    if (localCells > INT_MAX || globalCells > INT_MAX) {
        if (rank == 0) std::fprintf(stderr, "Grid exceeds MPI_Gatherv count limit\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    int cudaAwareMpi = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA)
    cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif
    int allCudaAwareMpi = 0;
    MPI_Allreduce(&cudaAwareMpi, &allCudaAwareMpi, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Parallel configuration: %d MPI ranks, up to %d OpenMP threads/rank, CUDA\n",
                    ranks, omp_get_max_threads());
        std::printf("MPI GPU transport: %s\n",
                    allCudaAwareMpi ? "direct CUDA-aware" : "pinned host staging");
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> hostInitial(localCells);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(localCells); ++i) {
        const size_t globalLinear = globalZ0 * slice + static_cast<size_t>(i);
        const double pseudo = (((globalLinear + 1) * static_cast<size_t>(1299709)) % globalCells)
                            / static_cast<double>(globalCells);
        hostInitial[static_cast<size_t>(i)] = -1.0 + 2.0 * pseudo;
    }

    const size_t allocatedCells = (localNz + 2) * slice;
    const size_t allocatedBytes = allocatedCells * sizeof(double);
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, allocatedBytes));
    CUDA_CHECK(cudaMalloc(&cnew, allocatedBytes));
    CUDA_CHECK(cudaMalloc(&mu, allocatedBytes));
    CUDA_CHECK(cudaMemcpy(cold + slice, hostInitial.data(), localCells * sizeof(double),
                          cudaMemcpyHostToDevice));
    hostInitial.clear();
    hostInitial.shrink_to_fit();

    HaloBuffers halo;
    const size_t haloBytes = slice * sizeof(double);
    CUDA_CHECK(cudaMallocHost(&halo.sendLower, haloBytes));
    CUDA_CHECK(cudaMallocHost(&halo.sendUpper, haloBytes));
    CUDA_CHECK(cudaMallocHost(&halo.recvLower, haloBytes));
    CUDA_CHECK(cudaMallocHost(&halo.recvUpper, haloBytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    exchangeHalos(cold, slice, localNz, lower, upper, halo, stream,
                  allCudaAwareMpi != 0, rank);

    constexpr int threads = 256;
    const int blocks = static_cast<int>((localCells + threads - 1) / threads);
    const double idx2 = 1.0;
    const double idy2 = 1.0;
    const double idz2 = 1.0;
    const double dtD = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = 2.0 / 9.0;
    const double gamma = 0.5;

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        chemicalPotentialKernel<<<blocks, threads, 0, stream>>>(
            cold, mu, nx, ny, localNz, idx2, idy2, idz2, gamma, eAA, eBB, eAB);
        CUDA_CHECK(cudaGetLastError());
        exchangeHalos(mu, slice, localNz, lower, upper, halo, stream,
                      allCudaAwareMpi != 0, rank);

        updateKernel<<<blocks, threads, 0, stream>>>(
            cnew, cold, mu, nx, ny, localNz, dtD, idx2, idy2, idz2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::swap(cold, cnew);
        if (step + 1 < iterations) {
            exchangeHalos(cold, slice, localNz, lower, upper, halo, stream,
                          allCudaAwareMpi != 0, rank);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double mcups = elapsed > 0.0
            ? static_cast<double>(globalCells) * iterations / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> localResult(localCells);
    CUDA_CHECK(cudaMemcpy(localResult.data(), cold + slice, localCells * sizeof(double),
                          cudaMemcpyDeviceToHost));

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> globalResult;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t rankNz = baseNz + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t rankZ0 = static_cast<size_t>(r) * baseNz
                                    + std::min(static_cast<size_t>(r), remainder);
                counts[r] = static_cast<int>(rankNz * slice);
                displacements[r] = static_cast<int>(rankZ0 * slice);
            }
            globalResult.resize(globalCells);
        }
        MPI_Gatherv(localResult.data(), static_cast<int>(localCells), MPI_DOUBLE,
                    rank == 0 ? globalResult.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) print_results(globalResult, "Concentration");
    }

    int localFinite = 1;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    #pragma omp parallel for reduction(&:localFinite) reduction(min:localMin) reduction(max:localMax) schedule(static)
    for (long long i = 0; i < static_cast<long long>(localCells); ++i) {
        const double value = localResult[static_cast<size_t>(i)];
        localFinite &= std::isfinite(value) ? 1 : 0;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }
    int globalFinite = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Reduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        if (!globalFinite) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        valid = globalFinite && globalMax <= 10.0 && globalMin >= -10.0;
        if (globalFinite && !valid) {
            std::printf("Validation failed: values out of expected range\n");
        }
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(halo.sendLower));
    CUDA_CHECK(cudaFreeHost(halo.sendUpper));
    CUDA_CHECK(cudaFreeHost(halo.recvLower));
    CUDA_CHECK(cudaFreeHost(halo.recvUpper));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
