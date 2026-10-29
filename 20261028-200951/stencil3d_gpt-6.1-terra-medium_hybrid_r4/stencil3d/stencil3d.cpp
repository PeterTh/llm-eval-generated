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

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

// z is a local plane index; planes 0 and count+1 are ghost planes.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t plane, size_t global_z,
                        size_t nz, size_t first, size_t last) {
    size_t x = 1 + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = 1 + size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = first + blockIdx.z;
    if (x >= nx - 1 || y >= ny - 1 || z >= last ||
        global_z + z - 1 == 0 || global_z + z - 1 == nz - 1) return;
    size_t p = z * plane + y * nx + x;
    out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] + in[p-plane] + in[p+plane]) / 7.0;
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny, size_t plane,
                   size_t start, size_t nz, size_t first, size_t last, cudaStream_t stream) {
    if (first >= last || nx < 3 || ny < 3 || nz < 3) return;
    dim3 block(32, 4);
    // Chunk the z dimension to respect CUDA's grid dimension limit.
    for (size_t z = first; z < last;) {
        size_t end = std::min(last, z + size_t(65535));
        dim3 grid((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y, end - z);
        stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, plane, start, nz, z, end);
        CUDA(cudaGetLastError());
        z = end;
    }
}

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n", p);
    printf("  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* option = argv[i++]; char* end = nullptr;
            long long value = strtoll(argv[i], &end, 10);
            if (*end || end == argv[i] || value < 0 || (!strcmp(option, "-i") && value > INT_MAX)) {
                if (!rank) fprintf(stderr, "Invalid value for %s\n", option); MPI_Finalize(); return 1;
            }
            if (!strcmp(option, "-x")) nx = size_t(value);
            else if (!strcmp(option, "-y")) ny = size_t(value);
            else if (!strcmp(option, "-z")) nz = size_t(value);
            else iterations = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > INT_MAX || nz < size_t(ranks) || nx * ny > SIZE_MAX / sizeof(Real) ||
        nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!rank) fprintf(stderr, "Invalid grid size, plane too large for MPI, or more ranks than Z planes\n");
        MPI_Finalize(); return 1;
    }
    size_t plane = nx * ny, count = nz / ranks + (size_t(rank) < nz % ranks);
    size_t start = size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int local_rank, devices;
    MPI_CHECK(MPI_Comm_rank(local, &local_rank));
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA devices\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local));
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    std::vector<Real> host(count * plane);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count * plane; ++i) host[i] = Real((start * plane + i) % 19);
    Real *a, *b, *send_low, *send_high, *recv_low, *recv_high;
    size_t bytes = (count + 2) * plane * sizeof(Real), plane_bytes = plane * sizeof(Real);
    CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes));
    CUDA(cudaMemcpy(a + plane, host.data(), count * plane_bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b + plane, host.data(), count * plane_bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMallocHost(&send_low, plane_bytes)); CUDA(cudaMallocHost(&send_high, plane_bytes));
    CUDA(cudaMallocHost(&recv_low, plane_bytes)); CUDA(cudaMallocHost(&recv_high, plane_bytes));
    cudaStream_t compute, transfer;
    CUDA(cudaStreamCreate(&compute)); CUDA(cudaStreamCreate(&transfer));
    int low = rank ? rank - 1 : MPI_PROC_NULL, high = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    if (!rank) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double t0 = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[4]; int nreq = 0;
        if (low != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_low, int(plane), MPI_DOUBLE, low, 1, MPI_COMM_WORLD, &req[nreq++]));
            CUDA(cudaMemcpyAsync(send_low, a + plane, plane_bytes, cudaMemcpyDeviceToHost, transfer));
        }
        if (high != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_high, int(plane), MPI_DOUBLE, high, 0, MPI_COMM_WORLD, &req[nreq++]));
            CUDA(cudaMemcpyAsync(send_high, a + count * plane, plane_bytes, cudaMemcpyDeviceToHost, transfer));
        }
        // Planes away from the slab edges do not depend on received halos.
        launch(a, b, nx, ny, plane, start, nz, 2, count, compute);
        CUDA(cudaStreamSynchronize(transfer));
        if (low != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(send_low, int(plane), MPI_DOUBLE, low, 0, MPI_COMM_WORLD, &req[nreq++]));
        if (high != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(send_high, int(plane), MPI_DOUBLE, high, 1, MPI_COMM_WORLD, &req[nreq++]));
        MPI_CHECK(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE));
        if (low != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(a, recv_low, plane_bytes, cudaMemcpyHostToDevice, transfer));
        if (high != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(a + (count + 1) * plane, recv_high, plane_bytes, cudaMemcpyHostToDevice, transfer));
        CUDA(cudaStreamSynchronize(transfer));
        launch(a, b, nx, ny, plane, start, nz, 1, 2, compute);
        if (count > 1) launch(a, b, nx, ny, plane, start, nz, count, count + 1, compute);
        CUDA(cudaStreamSynchronize(compute));
        std::swap(a, b);
    }
    double elapsed = MPI_Wtime() - t0, max_elapsed;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank) {
        printf("Computation time: %ld ms\n", long(max_elapsed * 1000));
        double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / max_elapsed / 1e6);
    }
    int valid = 1;
    if (results || validate) {
        CUDA(cudaMemcpy(host.data(), a + plane, count * plane_bytes, cudaMemcpyDeviceToHost));
        std::vector<Real> full;
        if (!rank) full.resize(nx * ny * nz);
        // Chunk transfers so the MPI count remains representable for large slabs.
        for (int r = 0; r < ranks; ++r) {
            size_t rc = nz / ranks + (size_t(r) < nz % ranks);
            size_t rs = size_t(r) * (nz / ranks) + std::min(size_t(r), nz % ranks);
            size_t total = rc * plane;
            for (size_t off = 0; off < total;) {
                int n = int(std::min(total - off, size_t(INT_MAX)));
                if (r == 0 && rank == 0) std::copy_n(host.data() + off, n, full.data() + rs * plane + off);
                else if (rank == r) MPI_CHECK(MPI_Send(host.data() + off, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
                else if (rank == 0) MPI_CHECK(MPI_Recv(full.data() + rs * plane + off, n, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                off += n;
            }
        }
        if (!rank) {
            if (results) print_results(full, "Grid");
            if (validate) {
                printf("Validating result...\n");
                Real lo = full[0], hi = full[0];
                for (Real v : full) { if (!std::isfinite(v)) valid = 0; lo = std::min(lo, v); hi = std::max(hi, v); }
                if (!valid) printf("Validation failed: found NaN or Inf value\n");
                else { printf("Value range: [%.6f, %.6f]\n", lo, hi); if (lo < -1e6 || hi > 1e6) { printf("Validation failed: values out of expected range\n"); valid = 0; } }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFreeHost(send_low)); CUDA(cudaFreeHost(send_high)); CUDA(cudaFreeHost(recv_low)); CUDA(cudaFreeHost(recv_high));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
