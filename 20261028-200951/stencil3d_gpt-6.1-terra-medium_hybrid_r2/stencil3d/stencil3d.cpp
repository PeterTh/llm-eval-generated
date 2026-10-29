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
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { fprintf(stderr, "Rank %d: MPI error at %s:%d\n", rank, __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Each rank owns complete Z planes, plus a ghost plane at either end.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t first_global_z, size_t nz,
                        size_t z_begin, size_t z_end) {
    size_t plane = nx * ny;
    size_t total = (z_end - z_begin) * plane;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)blockDim.x * gridDim.x) {
        size_t z = z_begin + t / plane, p = t % plane;
        size_t x = p % nx, y = p / nx, gz = first_global_z + z - 1;
        if (x && x + 1 < nx && y && y + 1 < ny && gz && gz + 1 < nz) {
            size_t i = z * plane + p;
            out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] + in[i-plane] + in[i+plane]) / 7.0;
        }
    }
}

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10; bool validate = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* opt = argv[i++]; char* end = nullptr;
            long long v = strtoll(argv[i], &end, 10);
            if (*end || end == argv[i] || v < 0 || (!strcmp(opt, "-i") && v > INT_MAX)) { if (!rank) fprintf(stderr, "Invalid value for %s\n", opt); MPI_Finalize(); return 1; }
            if (!strcmp(opt, "-x")) nx = (size_t)v;
            else if (!strcmp(opt, "-y")) ny = (size_t)v;
            else if (!strcmp(opt, "-z")) nz = (size_t)v;
            else iterations = (int)v;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || nz < (size_t)ranks || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!rank) fprintf(stderr, "Invalid grid size or too many MPI ranks (one rank per Z plane maximum)\n"); MPI_Finalize(); return 1;
    }
    size_t plane = nx * ny, count = nz / ranks + ((size_t)rank < nz % ranks);
    size_t start_z = (size_t)rank * (nz / ranks) + std::min((size_t)rank, nz % ranks);
    if (count > SIZE_MAX / plane - 2 || (count + 2) * plane > SIZE_MAX / sizeof(Real)) { if (!rank) fprintf(stderr, "Local grid too large\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm local; MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    size_t bytes = (count + 2) * plane * sizeof(Real), plane_bytes = plane * sizeof(Real);
    std::vector<Real> host((count + 2) * plane, 0);
    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < count; ++z)
        for (size_t p = 0; p < plane; ++p) host[(z+1)*plane+p] = (Real)(((start_z+z)*plane+p) % 19);
    Real *a, *b, *send_low, *send_high, *recv_low, *recv_high;
    CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes));
    CUDA(cudaMemcpy(a, host.data(), bytes, cudaMemcpyHostToDevice)); CUDA(cudaMemcpy(b, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMallocHost(&send_low, plane_bytes)); CUDA(cudaMallocHost(&send_high, plane_bytes));
    CUDA(cudaMallocHost(&recv_low, plane_bytes)); CUDA(cudaMallocHost(&recv_high, plane_bytes));
    if (!rank) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); double start = MPI_Wtime();
    auto launch = [&](size_t lo, size_t hi) {
        if (hi <= lo) return;
        size_t n = (hi-lo)*plane; unsigned blocks = (unsigned)std::min((n + 255)/256, (size_t)65535);
        stencil<<<blocks, 256>>>(a, b, nx, ny, start_z, nz, lo, hi);
        CUDA(cudaGetLastError());
    };
    for (int iter = 0; iter < iterations; ++iter) {
        if (rank > 0) CUDA(cudaMemcpyAsync(send_low, a + plane, plane_bytes, cudaMemcpyDeviceToHost));
        if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(send_high, a + count*plane, plane_bytes, cudaMemcpyDeviceToHost));
        if (ranks > 1) CUDA(cudaStreamSynchronize(0));
        MPI_Request req[4]; int nr = 0;
        if (rank > 0) { MPI_CHECK(MPI_Irecv(recv_low, (int)plane, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[nr++])); MPI_CHECK(MPI_Isend(send_low, (int)plane, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[nr++])); }
        if (rank + 1 < ranks) { MPI_CHECK(MPI_Irecv(recv_high, (int)plane, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[nr++])); MPI_CHECK(MPI_Isend(send_high, (int)plane, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[nr++])); }
        launch(2, count); // Planes independent of remote halos.
        MPI_CHECK(MPI_Waitall(nr, req, MPI_STATUSES_IGNORE));
        if (rank > 0) CUDA(cudaMemcpyAsync(a, recv_low, plane_bytes, cudaMemcpyHostToDevice));
        if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(a + (count+1)*plane, recv_high, plane_bytes, cudaMemcpyHostToDevice));
        launch(1, 2); if (count > 1) launch(count, count+1);
        std::swap(a, b);
    }
    CUDA(cudaDeviceSynchronize()); double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank) {
        printf("Computation time: %ld ms\n", (long)(max_elapsed * 1000));
        double updates = (double)(nx > 2 ? nx-2 : 0) * (ny > 2 ? ny-2 : 0) * (nz > 2 ? nz-2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / max_elapsed / 1e6);
    }
    if (print || validate) CUDA(cudaMemcpy(host.data()+plane, a+plane, count*plane_bytes, cudaMemcpyDeviceToHost));
    if (print) {
        // Send in chunks to avoid MPI's int count limit.
        std::vector<Real> all; if (!rank) { all.resize(nx*ny*nz); std::copy_n(host.data()+plane, count*plane, all.data()); }
        for (int r = 1; r < ranks; ++r) {
            size_t rc = nz / ranks + ((size_t)r < nz % ranks);
            size_t rz = (size_t)r * (nz / ranks) + std::min((size_t)r, nz % ranks);
            for (size_t off = 0; off < rc*plane;) {
                int n = (int)std::min(rc*plane-off, (size_t)INT_MAX);
                if (rank == r) MPI_CHECK(MPI_Send(host.data()+plane+off, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
                if (!rank) MPI_CHECK(MPI_Recv(all.data()+rz*plane+off, n, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                off += n;
            }
        }
        if (!rank) print_results(all, "Grid");
    }
    int valid = 1;
    if (validate) {
        double minval = std::numeric_limits<double>::infinity(), maxval = -minval; int finite = 1;
        #pragma omp parallel for reduction(min:minval) reduction(max:maxval) reduction(&:finite) schedule(static)
        for (size_t i = plane; i < (count+1)*plane; ++i) { double v = host[i]; finite &= std::isfinite(v); minval = std::min(minval, v); maxval = std::max(maxval, v); }
        double global_min, global_max; int global_finite;
        MPI_CHECK(MPI_Reduce(&minval, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&maxval, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&finite, &global_finite, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD));
        if (!rank) { printf("Validating result...\n"); if (!global_finite) printf("Validation failed: found NaN or Inf value\n"); else { printf("Value range: [%.6f, %.6f]\n", global_min, global_max); if (global_max > 1e6 || global_min < -1e6) printf("Validation failed: values out of expected range\n"); } valid = global_finite && global_max <= 1e6 && global_min >= -1e6; printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFreeHost(send_low)); CUDA(cudaFreeHost(send_high)); CUDA(cudaFreeHost(recv_low)); CUDA(cudaFreeHost(recv_high));
    MPI_Finalize(); return valid ? 0 : 1;
}
