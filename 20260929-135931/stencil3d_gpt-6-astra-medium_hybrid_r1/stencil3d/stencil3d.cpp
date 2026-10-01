#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using Real = double;

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) checkCuda((call), #call)

// Each thread sweeps a short Z column, reusing the three axial values.
// X is contiguous across a warp; no floating-point reassociation is enabled.
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t first, size_t last) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = first + size_t(blockIdx.z) * 8;
    if (x >= nx - 1 || y >= ny - 1 || z >= last) return;
    const size_t plane = nx * ny;
    size_t i = z * plane + y * nx + x;
    Real bottom = input[i - plane], center = input[i];
    for (size_t k = z; k < last && k < z + 8; ++k, i += plane) {
        const Real top = input[i + plane];
        output[i] = (center + input[i-1] + input[i+1] + input[i-nx]
                     + input[i+nx] + bottom + top) / 7.0;
        bottom = center;
        center = top;
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t first, size_t last, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || first >= last) return;
    // Keep the Z grid within CUDA's grid dimension limit.
    for (size_t z = first; z < last; z += 8 * 65535) {
        const size_t end = std::min(last, z + 8 * 65535);
        stencil<<<dim3((nx-2+31)/32, (ny-2+3)/4, (end-z+7)/8),
                  dim3(32,4), 0, stream>>>(input, output, nx, ny, z, end);
    }
    CUDA(cudaGetLastError());
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
    int provided, rank, world;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    

    if (!nx || !ny || !nz || iterations < 0 ||
        nx > size_t(INT_MAX) / ny ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny) / sizeof(Real)) {
        if (!rank) fprintf(stderr, "Invalid or unsupported grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    // Ranks beyond the number of planes participate only in finalization.
    const int ranks = int(std::min(nz, size_t(world)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= ranks) { MPI_Finalize(); return 0; }
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    const size_t plane = nx * ny;
    const size_t depth = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t offset = size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
    const size_t count = (depth + 2) * plane;
    std::vector<Real> host(count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const size_t localZ = i / plane;
        // Exterior ghosts are never read by the stencil.
        if ((localZ == 0 && offset == 0) || (localZ == depth + 1 && offset + depth == nz))
            host[i] = 0;
        else
            host[i] = Real(((offset + localZ - 1) * plane + i % plane) % 19);
    }
    Real *a, *b, *halo = nullptr;
    CUDA(cudaMalloc(&a, count * sizeof(Real)));
    CUDA(cudaMalloc(&b, count * sizeof(Real)));
    CUDA(cudaMemcpy(a, host.data(), count * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b, a, count * sizeof(Real), cudaMemcpyDeviceToDevice));
    cudaStream_t compute, exchange;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
    if (ranks > 1) CUDA(cudaMallocHost(&halo, 4 * plane * sizeof(Real)));
    const int lower = rank ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    const size_t first = offset == 0 ? 2 : 1;
    const size_t last = depth + 1 - (offset + depth == nz);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1 || iter == iterations - 1) {
            launch(a, b, nx, ny, first, last, compute);
            CUDA(cudaStreamSynchronize(compute));
        } else {
            MPI_Request requests[4];
            MPI_Irecv(halo + 2 * plane, int(plane), MPI_DOUBLE, lower, 1, comm, &requests[0]);
            MPI_Irecv(halo + 3 * plane, int(plane), MPI_DOUBLE, upper, 0, comm, &requests[1]);
            // Produce interface planes first. Fixed faces are already in both buffers.
            launch(a, b, nx, ny, first, std::min(last, size_t(2)), exchange);
            if (depth > 1) launch(a, b, nx, ny, std::max(first, depth), last, exchange);
            if (lower != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(halo, b + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, exchange));
            if (upper != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(halo + plane, b + depth * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, exchange));
            launch(a, b, nx, ny, std::max(first, size_t(2)), std::min(last, depth), compute);
            CUDA(cudaStreamSynchronize(exchange));
            MPI_Isend(halo, int(plane), MPI_DOUBLE, lower, 0, comm, &requests[2]);
            MPI_Isend(halo + plane, int(plane), MPI_DOUBLE, upper, 1, comm, &requests[3]);
            // MPI makes progress here while the bulk CUDA kernel is running.
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            if (lower != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(b, halo + 2 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, exchange));
            if (upper != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(b + (depth + 1) * plane, halo + 3 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, exchange));
            CUDA(cudaStreamSynchronize(exchange));
            CUDA(cudaStreamSynchronize(compute));
        }
        std::swap(a, b);
    }
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        const double updates = double(nx > 2 ? nx-2 : 0) * double(ny > 2 ? ny-2 : 0)
                             * double(nz > 2 ? nz-2 : 0) * iterations;
        printf("Computation time: %ld ms\n", long(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    if (printResults || validate)
        CUDA(cudaMemcpy(host.data(), a + plane, depth * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Chunked ordered gathering also supports total grids larger than INT_MAX.
        std::vector<Real> global;
        if (!rank) global.resize(nx * ny * nz);
        for (int owner = 0; owner < ranks; ++owner) {
            const size_t n = (nz / ranks + (size_t(owner) < nz % ranks)) * plane;
            const size_t at = (size_t(owner) * (nz / ranks) + std::min(size_t(owner), nz % ranks)) * plane;
            for (size_t pos = 0; pos < n;) {
                const int chunk = int(std::min(n - pos, size_t(INT_MAX)));
                if (!rank && !owner) std::copy_n(host.data() + pos, chunk, global.data() + at + pos);
                else if (rank == owner) MPI_Send(host.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                else if (!rank) MPI_Recv(global.data() + at + pos, chunk, MPI_DOUBLE, owner, 2, comm, MPI_STATUS_IGNORE);
                pos += chunk;
            }
        }
        if (!rank) print_results(global, "Grid");
    }
    int valid = 1;
    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
        int finite = 1;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:finite) schedule(static)
        for (size_t i = 0; i < depth * plane; ++i) {
            finite &= std::isfinite(host[i]);
            lo = std::min(lo, host[i]); hi = std::max(hi, host[i]);
        }
        Real globalLo, globalHi;
        MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
        valid = valid && globalLo >= -1e6 && globalHi <= 1e6;
        if (!rank) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalLo, globalHi);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (halo) CUDA(cudaFreeHost(halo));
    CUDA(cudaFree(a)); CUDA(cudaFree(b));
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(exchange));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
