#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// A thread walks eight consecutive Z cells, reusing the Z neighbors in registers.
// X is contiguous within each warp; the read-only cache reuses X/Y neighbors.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t first, size_t last) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    const size_t plane = nx * ny;
    for (size_t z0 = first + size_t(blockIdx.z) * 8; z0 <= last;
         z0 += size_t(gridDim.z) * 8) {
        size_t p = z0 * plane + y * nx + x;
        Real bottom = input[p - plane];
        Real center = input[p];
        const size_t end = min(z0 + 7, last);
        for (size_t z = z0; z <= end; ++z, p += plane) {
            const Real top = input[p + plane];
            // Preserve the original sum order and correctly rounded division.
            const Real sum = center + input[p - 1] + input[p + 1]
                           + input[p - nx] + input[p + nx] + bottom + top;
            output[p] = __ddiv_rn(sum, 7.0);
            bottom = center;
            center = top;
        }
    }
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t nz, size_t startZ, size_t first, size_t last,
                          cudaStream_t stream) {
    // Local plane 1 corresponds to global plane startZ. Physical boundaries
    // are initialized in both buffers and never overwritten.
    first = std::max(first, startZ == 0 ? size_t(2) : size_t(1));
    last = std::min(last, nz - 1 - startZ);
    if (nx < 3 || ny < 3 || first > last) return;
    dim3 block(32, 4);
    dim3 grid((nx - 2 + 31) / 32, (ny - 2 + 3) / 4,
              std::min(size_t(65535), (last - first + 8) / 8));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, first, last);
    CUDA_CHECK(cudaGetLastError());
}

// MPI counts are int-sized. Ordered chunks also support grids larger than 2 GiB.
static void postHalo(Real* data, size_t count, int peer, int tag, bool receive,
                     MPI_Comm comm, std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        requests.push_back(MPI_REQUEST_NULL);
        if (receive)
            MPI_Irecv(data + offset, chunk, MPI_DOUBLE, peer, tag, comm, &requests.back());
        else
            MPI_Isend(data + offset, chunk, MPI_DOUBLE, peer, tag, comm, &requests.back());
        offset += chunk;
    }
}

static size_t parseSize(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end || errno ||
        value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Invalid numeric argument");
    }
    return size_t(value);
}

static bool validateResult(const std::vector<Real>& grid, MPI_Comm comm, int rank) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        invalid |= !std::isfinite(grid[i]);
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_BOR, comm);
    if (rank == 0) {
        if (invalid) printf("Validation failed: found NaN or Inf value\n");
        printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
        if (maxVal > 1e6 || minVal < -1e6)
            printf("Validation failed: values out of expected range\n");
    }
    return !invalid && maxVal <= 1e6 && minVal >= -1e6;
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

static int run(int argc, char** argv, int rank, int worldSize) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = parseSize(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = parseSize(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = parseSize(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            const size_t count = parseSize(argv[++i]);
            if (count > INT_MAX) throw std::runtime_error("Too many iterations");
            iterations = int(count);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > maxElements / ny || nx * ny > maxElements / nz)
        throw std::runtime_error("Grid dimensions are zero or too large");
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    // Avoid zero-sized slabs when there are more ranks than Z planes.
    const int ranks = int(std::min(nz, size_t(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= ranks) return 0;
    const size_t localZ = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t startZ = size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
    if (maxElements / plane < 2 || localZ > maxElements / plane - 2 || plane > maxElements / 4)
        throw std::runtime_error("Local grid including halos is too large");
    const size_t localCells = (localZ + 2) * plane;
    const size_t planeBytes = plane * sizeof(Real);
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    // Bind ranks to node-local accelerators, respecting CUDA_VISIBLE_DEVICES.
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("A CUDA accelerator is required");
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, localRank % devices));
    if ((nx > 2 && (nx - 2 + 31) / 32 > size_t(properties.maxGridSize[0])) ||
        (ny > 2 && (ny - 2 + 3) / 4 > size_t(properties.maxGridSize[1])))
        throw std::runtime_error("Grid dimensions exceed CUDA launch limits");

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI ranks, OpenMP host work, CUDA stencil\n", ranks);
        printf("Initializing grid...\n");
    }
    Real *input, *output;
    CUDA_CHECK(cudaMalloc(&input, localCells * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&output, localCells * sizeof(Real)));
    {
        std::vector<Real> initial(localCells);
        #pragma omp parallel for schedule(static)
        for (size_t z = 0; z < localZ + 2; ++z) {
            const size_t globalZ = z == 0 ? (startZ ? startZ - 1 : 0)
                                                  : std::min(startZ + z - 1, nz - 1);
            for (size_t i = 0; i < plane; ++i)
                initial[z * plane + i] = Real((globalZ * plane + i) % 19);
        }
        CUDA_CHECK(cudaMemcpy(input, initial.data(), localCells * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(output, input, localCells * sizeof(Real), cudaMemcpyDeviceToDevice));
    }
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    // Use direct device communication when MPI advertises CUDA support. The
    // pinned staging path remains portable to MPI libraries without GPU support.
    bool directMPI = false;
    #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    directMPI = MPIX_Query_cuda_support() != 0;
    #endif
    int allDirect = directMPI ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &allDirect, 1, MPI_INT, MPI_MIN, comm);
    directMPI = allDirect != 0;
    Real* halos = nullptr;
    if (ranks > 1 && !directMPI) CUDA_CHECK(cudaMallocHost(&halos, 4 * planeBytes));
    std::vector<MPI_Request> requests;
    requests.reserve(4 * (plane / INT_MAX + 1));
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1 || iter + 1 == iterations) {
            launchStencil(input, output, nx, ny, nz, startZ, 1, localZ, compute);
        } else {
            // Compute the outgoing planes first on an independent stream.
            // While MPI progresses, the GPU processes the rest of the slab.
            requests.clear();
            if (lower != MPI_PROC_NULL)
                postHalo(directMPI ? output : halos + 2 * plane,
                         plane, lower, 11, true, comm, requests);
            if (upper != MPI_PROC_NULL)
                postHalo(directMPI ? output + (localZ + 1) * plane : halos + 3 * plane,
                         plane, upper, 10, true, comm, requests);
            launchStencil(input, output, nx, ny, nz, startZ, 1, 1, transfer);
            if (localZ > 1)
                launchStencil(input, output, nx, ny, nz, startZ, localZ, localZ, transfer);
            if (!directMPI && lower != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(halos, output + plane, planeBytes,
                                           cudaMemcpyDeviceToHost, transfer));
            if (!directMPI && upper != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(halos + plane, output + localZ * plane,
                                           planeBytes, cudaMemcpyDeviceToHost, transfer));
            if (localZ > 2)
                launchStencil(input, output, nx, ny, nz, startZ, 2, localZ - 1, compute);
            CUDA_CHECK(cudaStreamSynchronize(transfer));
            if (lower != MPI_PROC_NULL)
                postHalo(directMPI ? output + plane : halos,
                         plane, lower, 10, false, comm, requests);
            if (upper != MPI_PROC_NULL)
                postHalo(directMPI ? output + localZ * plane : halos + plane,
                         plane, upper, 11, false, comm, requests);
            MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (!directMPI && lower != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(output, halos + 2 * plane, planeBytes,
                                           cudaMemcpyHostToDevice, transfer));
            if (!directMPI && upper != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(output + (localZ + 1) * plane, halos + 3 * plane,
                                           planeBytes, cudaMemcpyHostToDevice, transfer));
            CUDA_CHECK(cudaStreamSynchronize(transfer));
        }
        // Multiple ranks must complete both writers before reusing a buffer.
        // A single rank can queue all iterations in the same ordered stream.
        if (ranks > 1) CUDA_CHECK(cudaStreamSynchronize(compute));
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0)
                             * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(localZ * plane);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input + plane,
                              finalGrid.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        // Gather only when required for the original ordered hash and samples.
        std::vector<Real> global;
        if (rank == 0) {
            global.resize(gridSize);
            std::copy(finalGrid.begin(), finalGrid.end(), global.begin());
        }
        for (int source = 1; source < ranks; ++source) {
            const size_t count = (nz / ranks + (size_t(source) < nz % ranks)) * plane;
            const size_t offset = (size_t(source) * (nz / ranks)
                                 + std::min(size_t(source), nz % ranks)) * plane;
            for (size_t pos = 0; pos < count;) {
                const int chunk = int(std::min(count - pos, size_t(INT_MAX)));
                if (rank == 0)
                    MPI_Recv(global.data() + offset + pos, chunk, MPI_DOUBLE, source, 20, comm, MPI_STATUS_IGNORE);
                else if (rank == source)
                    MPI_Send(finalGrid.data() + pos, chunk, MPI_DOUBLE, 0, 20, comm);
                pos += chunk;
            }
        }
        if (rank == 0) print_results(global, "Grid");
    }
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(finalGrid, comm, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (halos) CUDA_CHECK(cudaFreeHost(halos));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
