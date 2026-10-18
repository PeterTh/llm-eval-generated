#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <exception>

#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

// A warp traverses X contiguously. Each thread rolls three Z values through
// registers, reusing the center and Z neighbors for an eight-plane tile.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t first, size_t last) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t z = first + size_t(blockIdx.z) * 8;
    if (x >= nx - 1 || z >= last) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        size_t index = z * plane + y * nx + x;
        Real bottom = input[index - plane];
        Real center = input[index];
        for (size_t k = z; k < last && k < z + 8; ++k, index += plane) {
            const Real top = input[index + plane];
            output[index] = (center + input[index - 1] + input[index + 1]
                             + input[index - nx] + input[index + nx] + bottom + top) / 7.0;
            bottom = center;
            center = top;
        }
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t first, size_t last, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || first >= last) return;
    // Split the launch in Z to support slabs larger than the CUDA grid limit.
    for (size_t z = first; z < last; z += 8 * size_t(65535)) {
        const size_t end = std::min(last, z + 8 * size_t(65535));
        dim3 block(32, 4);
        dim3 grid((nx - 2 + 31) / 32, std::min((ny - 2 + 3) / 4, size_t(65535)),
                  (end - z + 7) / 8);
        stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, z, end);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Balanced contiguous slabs, including the fixed global boundary planes.
static size_t slabSize(size_t nz, int ranks, int rank) {
    return nz / ranks + (size_t(rank) < nz % ranks);
}
static size_t slabStart(size_t nz, int ranks, int rank) {
    return size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > size_t(INT_MAX) || ny > size_t(INT_MAX) || nz > size_t(INT_MAX) ||
        nx > size_t(INT_MAX) / ny ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny) / sizeof(Real) - 2) {
        if (worldRank == 0) fprintf(stderr, "Invalid or oversized grid/iteration count\n");
        MPI_Finalize();
        return 1;
    }
    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Excess ranks remain in the final world broadcast, with no empty GPU slabs.
    const int activeRanks = std::min(size_t(worldSize), nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    int status = 0;
    if (comm != MPI_COMM_NULL) {
      try {
        int rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm local;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "No CUDA device available on rank %d\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
        const size_t plane = nx * ny;
        const size_t depth = slabSize(nz, activeRanks, rank);
        const size_t startZ = slabStart(nz, activeRanks, rank);
        const size_t count = (depth + 2) * plane;
        const size_t bytes = count * sizeof(Real);
        Real *host, *input, *output, *halos = nullptr;
        CUDA_CHECK(cudaMallocHost(&host, bytes));
        // Initialize halos analytically, so the first step needs no exchange.
        #pragma omp parallel for schedule(static)
        for (size_t z = 0; z < depth + 2; ++z) {
            const size_t globalZ = z == 0 ? (startZ ? startZ - 1 : 0)
                                        : std::min(startZ + z - 1, nz - 1);
            for (size_t i = 0; i < plane; ++i)
                host[z * plane + i] = Real((globalZ * plane + i) % 19);
        }
        CUDA_CHECK(cudaMalloc(&input, bytes));
        CUDA_CHECK(cudaMalloc(&output, bytes));
        CUDA_CHECK(cudaMemcpy(input, host, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(output, host, bytes, cudaMemcpyHostToDevice));
        if (!printResults && !validate) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
        }
        cudaStream_t bulk, edge;
        CUDA_CHECK(cudaStreamCreateWithFlags(&bulk, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&edge, cudaStreamNonBlocking));
        // Open MPI exposes a runtime capability query. Other MPI implementations
        // use the portable pinned-host transport, without assuming device support.
        bool deviceMPI = false;
        #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        deviceMPI = MPIX_Query_cuda_support() != 0;
        #endif
        if (activeRanks > 1 && !deviceMPI)
            CUDA_CHECK(cudaMallocHost(&halos, 4 * plane * sizeof(Real)));
        const int lower = rank ? rank - 1 : MPI_PROC_NULL;
        const int upper = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
        const size_t first = startZ == 0 ? 2 : 1;
        const size_t last = startZ + depth == nz ? depth : depth + 1;
        if (rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(comm);
        const double begin = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            if (activeRanks == 1 || iter + 1 == iterations) {
                launchStencil(input, output, nx, ny, first, last, bulk);
            } else {
                // Edge planes are ready to send while the bulk kernel runs.
                launchStencil(input, output, nx, ny, first, std::min(last, size_t(2)), edge);
                if (depth > 1)
                    launchStencil(input, output, nx, ny, std::max(first, depth), last, edge);
                launchStencil(input, output, nx, ny, std::max(first, size_t(2)),
                              std::min(last, depth), bulk);
                MPI_Request requests[4];
                Real* receiveLower = deviceMPI ? output : halos + 2 * plane;
                Real* receiveUpper = deviceMPI ? output + (depth + 1) * plane : halos + 3 * plane;
                Real* sendLower = deviceMPI ? output + plane : halos;
                Real* sendUpper = deviceMPI ? output + depth * plane : halos + plane;
                MPI_Irecv(receiveLower, int(plane), MPI_DOUBLE, lower, 11, comm, &requests[0]);
                MPI_Irecv(receiveUpper, int(plane), MPI_DOUBLE, upper, 10, comm, &requests[1]);
                if (!deviceMPI && lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halos, output + plane, plane * sizeof(Real),
                                               cudaMemcpyDeviceToHost, edge));
                if (!deviceMPI && upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halos + plane, output + depth * plane,
                                               plane * sizeof(Real), cudaMemcpyDeviceToHost, edge));
                CUDA_CHECK(cudaStreamSynchronize(edge));
                // Pinned staging also works with MPI libraries lacking CUDA support.
                MPI_Isend(sendLower, int(plane), MPI_DOUBLE, lower, 10, comm, &requests[2]);
                MPI_Isend(sendUpper, int(plane), MPI_DOUBLE, upper, 11, comm, &requests[3]);
                MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
                if (!deviceMPI && lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(output, halos + 2 * plane, plane * sizeof(Real),
                                               cudaMemcpyHostToDevice, edge));
                if (!deviceMPI && upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(output + (depth + 1) * plane, halos + 3 * plane,
                                               plane * sizeof(Real), cudaMemcpyHostToDevice, edge));
                CUDA_CHECK(cudaStreamSynchronize(edge));
            }
            // A single rank queues all steps on one stream without a host round trip.
            if (activeRanks > 1) CUDA_CHECK(cudaStreamSynchronize(bulk));
            std::swap(input, output);
        }
        CUDA_CHECK(cudaStreamSynchronize(bulk));
        double elapsed = MPI_Wtime() - begin, seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", long(seconds * 1000));
            const double updates = double(nx > 2 ? nx - 2 : 0) *
                double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
            printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
        }
        if (printResults || validate)
            CUDA_CHECK(cudaMemcpy(host + plane, input + plane, depth * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        if (printResults) {
            // Gather only on request, in bounded chunks (no MPI int displacement limit).
            constexpr size_t chunk = 1 << 26;
            std::vector<Real> finalGrid;
            if (rank == 0) {
                finalGrid.resize(nx * ny * nz);
                std::copy(host + plane, host + (depth + 1) * plane, finalGrid.begin());
            }
            for (int source = 1; source < activeRanks; ++source) {
                const size_t n = slabSize(nz, activeRanks, source) * plane;
                const size_t offset = slabStart(nz, activeRanks, source) * plane;
                for (size_t i = 0; i < n; i += chunk) {
                    const int length = int(std::min(chunk, n - i));
                    if (rank == source) MPI_Send(host + plane + i, length, MPI_DOUBLE, 0, 20, comm);
                    if (rank == 0) MPI_Recv(finalGrid.data() + offset + i, length, MPI_DOUBLE,
                                          source, 20, comm, MPI_STATUS_IGNORE);
                }
            }
            if (rank == 0) print_results(finalGrid, "Grid");
        }
        if (validate) {
            Real minimum = std::numeric_limits<Real>::infinity();
            Real maximum = -minimum;
            int invalid = 0;
            #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
            for (size_t i = 0; i < depth * plane; ++i) {
                const Real value = host[plane + i];
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                invalid |= !std::isfinite(value);
                const size_t z = startZ + i / plane, y = (i % plane) / nx, x = i % nx;
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
                    invalid |= value != Real((startZ * plane + i) % 19);
            }
            MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_BOR, comm);
            status = invalid || maximum > 1e6 || minimum < -1e6;
            if (rank == 0) {
                printf("Validating result...\nValue range: [%.6f, %.6f]\n", minimum, maximum);
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        CUDA_CHECK(cudaStreamDestroy(bulk));
        CUDA_CHECK(cudaStreamDestroy(edge));
        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
        if (host) CUDA_CHECK(cudaFreeHost(host));
        if (halos) CUDA_CHECK(cudaFreeHost(halos));
      } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
