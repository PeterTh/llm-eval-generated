#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#ifdef OPEN_MPI
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

[[noreturn]] void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        fail("CUDA operation failed");
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Each warp accesses contiguous x values. A register window reuses z neighbors
// across eight planes; the remaining spatial reuse is handled by the GPU cache.
template<bool Edges>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output, size_t nx, size_t ny,
                              size_t nz, size_t zOffset, size_t localNz,
                              size_t begin, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    const size_t plane = nx * ny;
    size_t z = Edges ? (blockIdx.z == 0 ? 1 : localNz)
                     : begin + size_t(blockIdx.z) * 8;
    const size_t stop = Edges ? z + 1 : (z + 8 < end ? z + 8 : end);
    if (z >= stop) return;
    // Physical boundary planes never change in either ping-pong buffer.
    if (zOffset + z - 1 == 0) ++z;
    if (z >= stop || zOffset + z - 1 >= nz - 1) return;
    size_t index = z * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    for (; z < stop && zOffset + z - 1 < nz - 1; ++z, index += plane) {
        const Real top = input[index + plane];
        // Preserve the original left-to-right double-precision evaluation.
        output[index] = __ddiv_rn(center + input[index - 1] + input[index + 1]
                                  + input[index - nx] + input[index + nx]
                                  + bottom + top, 7.0);
        bottom = center;
        center = top;
    }
}

template<bool Edges>
void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t nz, size_t zOffset, size_t localNz,
                   size_t begin, size_t end, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || nz < 3 || (!Edges && begin >= end)) return;
    const dim3 block(32, 4);
    const dim3 grid((nx - 2 + 31) / 32, (ny - 2 + 3) / 4,
                    Edges ? (localNz > 1 ? 2 : 1) : (end - begin + 7) / 8);
    stencilKernel<Edges><<<grid, block, 0, stream>>>(
        input, output, nx, ny, nz, zOffset, localNz, begin, end);
    CUDA_CHECK(cudaGetLastError());
}

// Slabs include physical boundary planes; excess ranks are left inactive.
size_t slabSize(size_t nz, int rank, int ranks) {
    return nz / ranks + (size_t(rank) < nz % ranks);
}

size_t slabOffset(size_t nz, int rank, int ranks) {
    return size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
}

bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny,
                    size_t nz, size_t zOffset, MPI_Comm comm, int rank) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
        if (!std::isfinite(value)) invalid = 1;
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        const size_t z = zOffset + i / plane;
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
            if (value != Real((zOffset * plane + i) % 19)) invalid = 1;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm);
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    return !invalid && minVal >= -1e6 && maxVal <= 1e6;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

size_t parseNumber(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || errno == ERANGE ||
        value > std::numeric_limits<size_t>::max()) fail("Invalid numeric argument");
    return size_t(value);
}

int runBenchmark(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = parseNumber(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = parseNumber(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = parseNumber(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            const size_t count = parseNumber(argv[++i]);
            if (count > INT_MAX) fail("Iteration count exceeds INT_MAX");
            iterations = int(count);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit / ny || nx * ny > limit / nz)
        fail("Invalid or overflowing grid dimensions");
    const size_t plane = nx * ny;
    if (plane > INT_MAX) fail("A halo plane exceeds the MPI count limit");

    const int ranks = int(std::min(nz, size_t(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    const int rank = worldRank;
    const size_t localNz = slabSize(nz, rank, ranks);
    const size_t zOffset = slabOffset(nz, rank, ranks);
    if (localNz > limit / plane - 2) fail("Local grid with halos is too large");
    const size_t owned = localNz * plane;
    const size_t bytes = (localNz + 2) * plane * sizeof(Real);
    const size_t faceBytes = plane * sizeof(Real);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("A CUDA GPU is required on every participating rank");
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&nodeComm);

    int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support();
#endif
    // Other MPI implementations can explicitly enable their GPU-aware path.
    // Setting this to 0 also forces the portable pinned-host staging path.
    if (const char* setting = std::getenv("STENCIL3D_CUDA_AWARE_MPI"))
        cudaAware = std::strcmp(setting, "0") != 0;
    MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, comm);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA halo transport: %s\n",
               ranks, omp_get_max_threads(), cudaAware ? "device-direct" : "pinned-host");
        printf("Initializing grid...\n");
    }
    std::vector<Real> localGrid(owned);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < owned; ++i)
        localGrid[i] = Real((zOffset * plane + i) % 19);

    Real *input = nullptr, *output = nullptr, *staging = nullptr;
    CUDA_CHECK(cudaMalloc(&input, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));
    // Initialize both buffers once: physical boundaries then need no copying.
    CUDA_CHECK(cudaMemcpy(input + plane, localGrid.data(), owned * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output + plane, localGrid.data(), owned * sizeof(Real), cudaMemcpyHostToDevice));
    if (ranks > 1 && !cudaAware)
        CUDA_CHECK(cudaMallocHost(&staging, 4 * faceBytes));
    cudaStream_t computeStream, haloStream;
    cudaEvent_t bulkDone, haloDone;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&bulkDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloDone, cudaEventDisableTiming));
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    if (rank == 0) printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launchStencil<false>(input, output, nx, ny, nz, zOffset, localNz,
                                 1, localNz + 1, computeStream);
        } else {
            if (iter > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloDone, 0));
                CUDA_CHECK(cudaStreamWaitEvent(haloStream, bulkDone, 0));
            }
            // Only the first and last owned planes depend on incoming halos.
            launchStencil<false>(input, output, nx, ny, nz, zOffset, localNz,
                                 2, localNz, computeStream);
            CUDA_CHECK(cudaEventRecord(bulkDone, computeStream));
            if (!cudaAware) {
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(staging, input + plane, faceBytes,
                                               cudaMemcpyDeviceToHost, haloStream));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(staging + plane, input + localNz * plane,
                                               faceBytes, cudaMemcpyDeviceToHost, haloStream));
            }
            // MPI is called only by the main thread (MPI_THREAD_FUNNELED).
            // This also protects pinned receive buffers from reuse during H2D.
            CUDA_CHECK(cudaStreamSynchronize(haloStream));
            Real* recvLower = cudaAware ? input : staging + 2 * plane;
            Real* recvUpper = cudaAware ? input + (localNz + 1) * plane : staging + 3 * plane;
            Real* sendLower = cudaAware ? input + plane : staging;
            Real* sendUpper = cudaAware ? input + localNz * plane : staging + plane;
            MPI_Request requests[4];
            MPI_Irecv(recvLower, int(plane), MPI_DOUBLE, lower, 1, comm, &requests[0]);
            MPI_Irecv(recvUpper, int(plane), MPI_DOUBLE, upper, 0, comm, &requests[1]);
            MPI_Isend(sendLower, int(plane), MPI_DOUBLE, lower, 0, comm, &requests[2]);
            MPI_Isend(sendUpper, int(plane), MPI_DOUBLE, upper, 1, comm, &requests[3]);
            // The MPI progress engine runs while the bulk CUDA kernel executes.
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            if (!cudaAware) {
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(input, recvLower, faceBytes,
                                               cudaMemcpyHostToDevice, haloStream));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * plane, recvUpper,
                                               faceBytes, cudaMemcpyHostToDevice, haloStream));
            }
            launchStencil<true>(input, output, nx, ny, nz, zOffset, localNz,
                                0, 0, haloStream);
            CUDA_CHECK(cudaEventRecord(haloDone, haloStream));
        }
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(haloStream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = double(nx > 2 ? nx - 2 : 0) *
                               double(ny > 2 ? ny - 2 : 0) *
                               double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }

    if (printResults || validate)
        CUDA_CHECK(cudaMemcpy(localGrid.data(), input + plane, owned * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Preserve the original global order and exact statistics/hash. Gather
        // only on request, in chunks so total grid size may exceed INT_MAX.
        std::vector<Real> finalGrid;
        if (rank == 0) {
            finalGrid.resize(plane * nz);
            std::copy(localGrid.begin(), localGrid.end(), finalGrid.begin());
        }
        for (int source = 1; source < ranks; ++source) {
            const size_t count = slabSize(nz, source, ranks) * plane;
            const size_t offset = slabOffset(nz, source, ranks) * plane;
            for (size_t pos = 0; pos < count;) {
                const int chunk = int(std::min(count - pos, size_t(INT_MAX)));
                if (rank == 0)
                    MPI_Recv(finalGrid.data() + offset + pos, chunk, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                else if (rank == source)
                    MPI_Send(localGrid.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                pos += size_t(chunk);
            }
        }
        if (rank == 0) print_results(finalGrid, "Grid");
    }
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(localGrid, nx, ny, nz, zOffset, comm, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaEventDestroy(bulkDone));
    CUDA_CHECK(cudaEventDestroy(haloDone));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 1;
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    MPI_Finalize();
    return result;
}
