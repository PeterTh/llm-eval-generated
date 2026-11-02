#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Blocks cover contiguous rows for coalesced loads. Each block handles one
// 32-by-8 tile; the second grid dimension strides over tiles and planes.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t plane,
                        size_t first_z, size_t planes, size_t tiles_y) {
    const size_t total_tiles = tiles_y * planes;
    for (size_t tile = blockIdx.y; tile < total_tiles; tile += gridDim.y) {
        const size_t z = first_z + tile / tiles_y;
        const size_t y = (tile % tiles_y) * blockDim.y + threadIdx.y;
        if (y == 0 || y + 1 >= ny) continue;
        for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             x < nx; x += static_cast<size_t>(gridDim.x) * blockDim.x) {
            if (x == 0 || x + 1 >= nx) continue;
            const size_t i = z * plane + y * nx + x;
            out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx]
                      + in[i-plane] + in[i+plane]) / 7.0;
        }
    }
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny,
                   size_t plane, size_t first, size_t count, cudaStream_t stream) {
    if (!count || nx < 3 || ny < 3) return;
    const size_t tiles_y = (ny + 7) / 8;
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned>(std::min<size_t>((nx + 31) / 32, 65535)),
              static_cast<unsigned>(std::min<size_t>(tiles_y * count, 65535)));
    stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, plane, first, count, tiles_y);
    CUDA_CHECK(cudaGetLastError());
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

// Split large result messages into portable MPI counts.
static void send_grid(const Real* p, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int n = static_cast<int>(std::min<size_t>(count, INT_MAX));
        if (send) MPI_Send(p, n, MPI_DOUBLE, peer, 42, comm);
        else MPI_Recv(const_cast<Real*>(p), n, MPI_DOUBLE, peer, 42, comm, MPI_STATUS_IGNORE);
        p += n;
        count -= n;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* value = argv[++i];
            long long n = strtoll(value, &end, 10);
            if (*end || end == value || n < 0 || (option == 'i' && n > INT_MAX)) {
                if (!world_rank) fprintf(stderr, "Invalid value for -%c: %s\n", option, value);
                MPI_Finalize(); return 1;
            }
            if (option == 'x') nx = static_cast<size_t>(n);
            else if (option == 'y') ny = static_cast<size_t>(n);
            else if (option == 'z') nz = static_cast<size_t>(n);
            else iterations = static_cast<int>(n);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > INT_MAX || nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!world_rank) fprintf(stderr, "Invalid or unsupported grid size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const int active = static_cast<int>(std::min<size_t>(world_size, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices;
    MPI_Comm_rank(local, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA devices available on rank %d\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local);

    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t local_nz = base + (static_cast<size_t>(rank) < extra);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min<size_t>(rank, extra);
    if (local_nz > SIZE_MAX / plane - 2 || (local_nz + 2) * plane > SIZE_MAX / sizeof(Real)) {
        fprintf(stderr, "Local grid size overflow\n"); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t local_count = local_nz * plane, bytes = (local_nz + 2) * plane * sizeof(Real);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(local_count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_count; ++i) host[i] = static_cast<Real>((z0 * plane + i) % 19);
    Real *a, *b;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemcpy(a + plane, host.data(), local_count * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b + plane, host.data(), local_count * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    Real *send_low = nullptr, *send_high = nullptr, *recv_low = nullptr, *recv_high = nullptr;
    if (ranks > 1) {
        CUDA_CHECK(cudaMallocHost(&send_low, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&send_high, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recv_low, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recv_high, plane * sizeof(Real)));
    }
    if (!rank) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    for (int it = 0; it < iterations; ++it) {
        if (ranks == 1) {
            if (nz > 2) launch(a, b, nx, ny, plane, 2, nz - 2, compute);
        } else {
            MPI_Request requests[4];
            int nreq = 0;
            if (rank > 0) {
                MPI_Irecv(recv_low, static_cast<int>(plane), MPI_DOUBLE, rank-1, 1, comm, &requests[nreq++]);
                CUDA_CHECK(cudaMemcpyAsync(send_low, a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            if (rank + 1 < ranks) {
                MPI_Irecv(recv_high, static_cast<int>(plane), MPI_DOUBLE, rank+1, 0, comm, &requests[nreq++]);
                CUDA_CHECK(cudaMemcpyAsync(send_high, a + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            // Planes strictly inside the slab do not depend on incoming halos.
            if (local_nz > 2) launch(a, b, nx, ny, plane, 2, local_nz - 2, compute);
            CUDA_CHECK(cudaStreamSynchronize(transfer));
            if (rank > 0) MPI_Isend(send_low, static_cast<int>(plane), MPI_DOUBLE, rank-1, 0, comm, &requests[nreq++]);
            if (rank + 1 < ranks) MPI_Isend(send_high, static_cast<int>(plane), MPI_DOUBLE, rank+1, 1, comm, &requests[nreq++]);
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
            if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(a, recv_low, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
            if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(a + (local_nz+1)*plane, recv_high, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
            CUDA_CHECK(cudaStreamSynchronize(transfer));
            if (z0 > 0 && z0 + 1 < nz) launch(a, b, nx, ny, plane, 1, 1, compute);
            if (local_nz > 1 && z0 + local_nz - 1 > 0 && z0 + local_nz < nz)
                launch(a, b, nx, ny, plane, local_nz, 1, compute);
            CUDA_CHECK(cudaStreamSynchronize(compute));
        }
        std::swap(a, b);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        double updates = static_cast<double>(nx > 2 ? nx-2 : 0) * (ny > 2 ? ny-2 : 0) * (nz > 2 ? nz-2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / max_elapsed / 1e6);
    }
    if (validate || printResults) CUDA_CHECK(cudaMemcpy(host.data(), a + plane, local_count * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (!rank) {
            std::vector<Real> grid(nx * ny * nz);
            std::copy(host.begin(), host.end(), grid.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t count = base + (static_cast<size_t>(r) < extra);
                size_t offset = static_cast<size_t>(r) * base + std::min<size_t>(r, extra);
                send_grid(grid.data() + offset * plane, count * plane, r, false, comm);
            }
            print_results(grid, "Grid");
        } else send_grid(host.data(), local_count, 0, true, comm);
    }
    int valid = 1;
    if (validate) {
        Real minval = std::numeric_limits<Real>::infinity(), maxval = -minval;
        int bad = 0;
        #pragma omp parallel for reduction(min:minval) reduction(max:maxval) reduction(|:bad) schedule(static)
        for (size_t i = 0; i < local_count; ++i) {
            Real v = host[i];
            bad |= !std::isfinite(v);
            minval = std::min(minval, v); maxval = std::max(maxval, v);
        }
        Real global_min, global_max;
        int global_bad;
        MPI_Reduce(&minval, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maxval, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&bad, &global_bad, 1, MPI_INT, MPI_BOR, 0, comm);
        if (!rank) {
            printf("Validating result...\n");
            if (global_bad) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 1e6 || global_min < -1e6) { printf("Validation failed: values out of expected range\n"); valid = 0; }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    if (ranks > 1) {
        CUDA_CHECK(cudaFreeHost(send_low)); CUDA_CHECK(cudaFreeHost(send_high));
        CUDA_CHECK(cudaFreeHost(recv_low)); CUDA_CHECK(cudaFreeHost(recv_high));
    }
    CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
