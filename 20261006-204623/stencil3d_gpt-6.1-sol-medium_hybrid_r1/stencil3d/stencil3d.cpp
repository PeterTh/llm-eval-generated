#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#ifdef OPEN_MPI
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
#define CUDA(call) cudaCheck((call), #call)

// X-contiguous warps and a rolling Z window reuse three values in registers.
// The addition order matches the serial benchmark; floating-point contraction
// is disabled in the build so external result hashes remain reproducible.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t firstY,
                              size_t first, size_t last) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = firstY + size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z0 = first + size_t(blockIdx.z) * 8;
    if (x >= nx - 1 || y >= ny - 1 || z0 >= last) return;
    const size_t plane = nx * ny;
    size_t index = z0 * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    const size_t end = z0 + 8 < last ? z0 + 8 : last;
    for (size_t z = z0; z < end; ++z, index += plane) {
        const Real top = input[index + plane];
        output[index] = (center + input[index - 1] + input[index + 1]
                         + input[index - nx] + input[index + nx]
                         + bottom + top) / 7.0;
        bottom = center;
        center = top;
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t first, size_t last, cudaStream_t stream) {
    if (first >= last || nx < 3 || ny < 3) return;
    // Split tall/deep slabs to respect CUDA's grid Y and Z limits.
    for (size_t z = first; z < last;) {
        const size_t end = std::min(last, z + size_t(65535) * 8);
        for (size_t y = 1; y < ny - 1;) {
            const size_t endY = std::min(ny - 1, y + size_t(65535) * 4);
            dim3 block(32, 4);
            dim3 grid((nx - 2 + 31) / 32, (endY - y + 3) / 4, (end - z + 7) / 8);
            stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, y, z, end);
            CUDA(cudaGetLastError());
            y = endY;
        }
        z = end;
    }
}

// MPI's traditional interfaces use int counts; chunk large planes and grids.
static void postTransfer(Real* data, size_t count, int peer, int tag,
                         bool receive, MPI_Comm comm,
                         std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        MPI_Request request;
        if (receive) MPI_Irecv(data + offset, chunk, MPI_DOUBLE, peer, tag, comm, &request);
        else MPI_Isend(data + offset, chunk, MPI_DOUBLE, peer, tag, comm, &request);
        requests.push_back(request);
        offset += chunk;
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || iterations < 0 || nx > limit / ny || nx * ny > limit / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Ranks beyond the number of planes have no work. They still participate
    // in the final world-wide exit, without allocating a grid or a GPU.
    int active = int(std::min(nz, size_t(world)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (rank < active) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank, devices;
        MPI_Comm_rank(node, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&node);

        const size_t plane = nx * ny;
        const size_t base = nz / active, remainder = nz % active;
        const size_t owned = base + (size_t(rank) < remainder);
        const size_t startZ = size_t(rank) * base + std::min(size_t(rank), remainder);
        if (limit / plane < 2 || owned > limit / plane - 2)
            MPI_Abort(MPI_COMM_WORLD, 1);
        const size_t cells = (owned + 2) * plane;
        std::vector<Real> host(cells);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells; ++i) {
            const size_t localZ = i / plane;
            // Only the outermost, unused halo can fall outside the grid.
            if ((localZ == 0 && startZ == 0) || startZ + localZ > nz) host[i] = 0;
            else host[i] = Real(((startZ + localZ - 1) * plane + i % plane) % 19);
        }
        Real *input, *output;
        CUDA(cudaMalloc(&input, cells * sizeof(Real)));
        CUDA(cudaMalloc(&output, cells * sizeof(Real)));
        // Both buffers retain the fixed physical boundary values forever.
        CUDA(cudaMemcpy(input, host.data(), cells * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(output, host.data(), cells * sizeof(Real), cudaMemcpyHostToDevice));
        cudaStream_t compute, exchange;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        int lower = rank ? rank - 1 : MPI_PROC_NULL;
        int upper = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
        // Open MPI exposes a runtime capability query. Other MPI libraries
        // use the portable pinned-host path rather than assuming GPU support.
        bool cudaAware = false;
        #if defined(OPEN_MPI) && defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        cudaAware = MPIX_Query_cuda_support() != 0;
        #endif
        Real* staging = nullptr;
        if (active > 1 && !cudaAware) {
            if (plane > limit / 4) MPI_Abort(MPI_COMM_WORLD, 1);
            CUDA(cudaMallocHost(&staging, 4 * plane * sizeof(Real)));
        }
        const size_t first = startZ == 0 ? 2 : 1;
        const size_t last = owned + 1 - (startZ + owned == nz);
        std::vector<MPI_Request> requests;
        requests.reserve(4 * (plane / INT_MAX + 1));
        // Load/JIT the kernel before timing. This writes only the scratch grid.
        launch(input, output, nx, ny, first, last, compute);
        CUDA(cudaStreamSynchronize(compute));
        if (rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(comm);
        double begin = MPI_Wtime();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            bool communicate = active > 1 && iteration + 1 < iterations && nx >= 3 && ny >= 3 && nz >= 3;
            if (communicate) {
                // Compute the outgoing faces first, then overlap their copies
                // and the network transfer with the independent interior work.
                if (first <= 1 && 1 < last) launch(input, output, nx, ny, 1, 2, exchange);
                if (owned != 1 && first <= owned && owned < last)
                    launch(input, output, nx, ny, owned, owned + 1, exchange);
                launch(input, output, nx, ny, std::max(first, size_t(2)),
                       std::min(last, owned), compute);
                if (!cudaAware && lower != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(staging, output + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, exchange));
                if (!cudaAware && upper != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(staging + plane, output + owned * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, exchange));
                requests.clear();
                postTransfer(cudaAware ? output : staging + 2 * plane,
                             plane, lower, 1, true, comm, requests);
                postTransfer(cudaAware ? output + (owned + 1) * plane : staging + 3 * plane,
                             plane, upper, 0, true, comm, requests);
                CUDA(cudaStreamSynchronize(exchange));
                postTransfer(cudaAware ? output + plane : staging,
                             plane, lower, 0, false, comm, requests);
                postTransfer(cudaAware ? output + owned * plane : staging + plane,
                             plane, upper, 1, false, comm, requests);
                MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
                if (!cudaAware && lower != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(output, staging + 2 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, exchange));
                if (!cudaAware && upper != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(output + (owned + 1) * plane, staging + 3 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, exchange));
                CUDA(cudaStreamSynchronize(exchange));
            } else launch(input, output, nx, ny, first, last, compute);
            // With no communication, stream ordering permits all iterations
            // to stay queued on the device without a host round trip.
            if (communicate) CUDA(cudaStreamSynchronize(compute));
            std::swap(input, output);
        }
        CUDA(cudaStreamSynchronize(compute));
        double elapsed = MPI_Wtime() - begin, totalTime;
        MPI_Reduce(&elapsed, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0)
                             * double(nz > 2 ? nz - 2 : 0) * iterations;
            printf("Computation time: %ld ms\n", long(totalTime * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", totalTime > 0 ? updates / totalTime / 1e6 : 0);
        }
        if (printResults || validate)
            CUDA(cudaMemcpy(host.data(), input + plane, owned * plane * sizeof(Real), cudaMemcpyDeviceToHost));
        if (printResults) {
            std::vector<Real> grid;
            if (rank == 0) {
                grid.resize(plane * nz);
                std::copy_n(host.data(), owned * plane, grid.data());
                for (int peer = 1; peer < active; ++peer) {
                    size_t peerStart = size_t(peer) * base + std::min(size_t(peer), remainder);
                    size_t count = (base + (size_t(peer) < remainder)) * plane;
                    for (size_t offset = 0; offset < count;) {
                        int chunk = int(std::min(count - offset, size_t(INT_MAX)));
                        MPI_Recv(grid.data() + peerStart * plane + offset, chunk, MPI_DOUBLE, peer, 2, comm, MPI_STATUS_IGNORE);
                        offset += chunk;
                    }
                }
                print_results(grid, "Grid");
            } else {
                size_t count = owned * plane;
                for (size_t offset = 0; offset < count;) {
                    int chunk = int(std::min(count - offset, size_t(INT_MAX)));
                    MPI_Send(host.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm);
                    offset += chunk;
                }
            }
        }
        if (validate) {
            Real minimum = std::numeric_limits<Real>::infinity();
            Real maximum = -minimum;
            int invalid = 0;
            #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < owned * plane; ++i) {
                Real value = host[i];
                invalid |= !std::isfinite(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                size_t z = startZ + i / plane, y = (i / nx) % ny, x = i % nx;
                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz)
                    invalid |= value != Real((z * plane + y * nx + x) % 19);
            }
            Real globalMin, globalMax;
            int globalInvalid;
            MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, comm);
            status = globalInvalid || globalMax > 1e6 || globalMin < -1e6;
            if (rank == 0) {
                printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalMin, globalMax);
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        if (staging) CUDA(cudaFreeHost(staging));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(exchange));
        CUDA(cudaFree(input));
        CUDA(cudaFree(output));
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
