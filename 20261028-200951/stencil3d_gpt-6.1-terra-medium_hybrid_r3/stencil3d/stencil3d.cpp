#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

using Real = double;

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

// z is a local index; planes 0 and count+1 are halos.
__global__ void stencil(const Real* in, Real* out, size_t nx, size_t ny,
                        size_t global_z, size_t nz, size_t first, size_t last) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    size_t plane = nx * ny;
    for (size_t z = blockIdx.z + first; z <= last; z += gridDim.z) {
        if (global_z + z - 1 == 0 || global_z + z - 1 == nz - 1) continue;
        size_t p = z * plane + y * nx + x;
        out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] + in[p-plane] + in[p+plane]) / 7.0;
    }
}

static void usage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided; MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t nx = 128, ny = 0, nz = 0; int iterations = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* opt = argv[i]; char* end; long long n = strtoll(argv[++i], &end, 10);
            if (*end || end == argv[i] || n < 0 || (strcmp(opt, "-i") == 0 && n > INT_MAX)) { if (!rank) fprintf(stderr, "Invalid value for %s\n", opt); MPI_Finalize(); return 1; }
            if (!strcmp(opt, "-x")) nx = n; else if (!strcmp(opt, "-y")) ny = n; else if (!strcmp(opt, "-z")) nz = n; else iterations = n;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nz > SIZE_MAX / (nx * ny) || nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!rank) fprintf(stderr, "Invalid or unsupported grid size\n"); MPI_Finalize(); return 1;
    }
    int active = std::min<size_t>(size, nz); MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm));
    if (rank >= active) { MPI_Finalize(); return 0; }
    MPI_Comm local; MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    size_t plane = nx * ny, count = nz / active + (size_t(rank) < nz % active), start = rank * (nz / active) + std::min<size_t>(rank, nz % active);
    if (count > SIZE_MAX / plane - 2 || (count + 2) * plane > SIZE_MAX / sizeof(Real)) { fprintf(stderr, "Local grid too large\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t bytes = (count + 2) * plane * sizeof(Real), plane_bytes = plane * sizeof(Real);
    std::vector<Real> host((count + 2) * plane, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < count; ++z)
        for (size_t p = 0; p < plane; ++p) host[(z + 1) * plane + p] = ((start + z) * plane + p) % 19;
    Real *a, *b, *send_lo, *send_hi, *recv_lo, *recv_hi;
    CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes));
    CUDA(cudaMemcpy(a, host.data(), bytes, cudaMemcpyHostToDevice)); CUDA(cudaMemcpy(b, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMallocHost(&send_lo, plane_bytes)); CUDA(cudaMallocHost(&send_hi, plane_bytes));
    CUDA(cudaMallocHost(&recv_lo, plane_bytes)); CUDA(cudaMallocHost(&recv_hi, plane_bytes));
    cudaStream_t compute, transfer; CUDA(cudaStreamCreate(&compute)); CUDA(cudaStreamCreate(&transfer));
    if (!rank) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); }
    int prev = rank ? rank - 1 : MPI_PROC_NULL, next = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
    dim3 block(32, 4); dim3 grid((nx > 2 ? nx - 2 + 31 : 0) / 32, (ny > 2 ? ny - 2 + 3 : 0) / 4);
    MPI_CHECK(MPI_Barrier(comm)); double begin = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (prev != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send_lo, a + plane, plane_bytes, cudaMemcpyDeviceToHost, transfer));
        if (next != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send_hi, a + count * plane, plane_bytes, cudaMemcpyDeviceToHost, transfer));
        if (nx > 2 && ny > 2 && count > 2) {
            grid.z = std::min<size_t>(count - 2, 65535); stencil<<<grid, block, 0, compute>>>(a, b, nx, ny, start, nz, 2, count - 1); CUDA(cudaGetLastError());
        }
        CUDA(cudaStreamSynchronize(transfer));
        MPI_Request requests[4]; int nreq = 0;
        if (prev != MPI_PROC_NULL) { MPI_CHECK(MPI_Irecv(recv_lo, plane, MPI_DOUBLE, prev, 1, comm, &requests[nreq++])); MPI_CHECK(MPI_Isend(send_lo, plane, MPI_DOUBLE, prev, 0, comm, &requests[nreq++])); }
        if (next != MPI_PROC_NULL) { MPI_CHECK(MPI_Irecv(recv_hi, plane, MPI_DOUBLE, next, 0, comm, &requests[nreq++])); MPI_CHECK(MPI_Isend(send_hi, plane, MPI_DOUBLE, next, 1, comm, &requests[nreq++])); }
        MPI_CHECK(MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE));
        if (prev != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(a, recv_lo, plane_bytes, cudaMemcpyHostToDevice, transfer));
        if (next != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(a + (count + 1) * plane, recv_hi, plane_bytes, cudaMemcpyHostToDevice, transfer));
        CUDA(cudaStreamSynchronize(transfer));
        if (nx > 2 && ny > 2) {
            grid.z = 1; stencil<<<grid, block, 0, compute>>>(a, b, nx, ny, start, nz, 1, 1);
            if (count > 1) stencil<<<grid, block, 0, compute>>>(a, b, nx, ny, start, nz, count, count);
            CUDA(cudaGetLastError());
        }
        CUDA(cudaStreamSynchronize(compute)); std::swap(a, b);
    }
    double elapsed = MPI_Wtime() - begin, max_elapsed; MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) { long ms = long(max_elapsed * 1000); double updates = double(nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations; printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms, updates / max_elapsed / 1e6); }
    if (results || validate) {
        CUDA(cudaMemcpy(host.data() + plane, a + plane, count * plane_bytes, cudaMemcpyDeviceToHost));
        std::vector<Real> full; if (!rank) full.resize(nx * ny * nz);
        // Chunk transfers to support grids larger than MPI's int count limit.
        for (int r = 0; r < active; ++r) {
            size_t cnt = (nz / active + (size_t(r) < nz % active)) * plane;
            size_t off = (r * (nz / active) + std::min<size_t>(r, nz % active)) * plane;
            for (size_t pos = 0; pos < cnt;) {
                int chunk = int(std::min<size_t>(cnt - pos, INT_MAX));
                if (r == 0 && rank == 0) std::copy_n(host.data() + plane + pos, chunk, full.data() + off + pos);
                else if (rank == r) MPI_CHECK(MPI_Send(host.data() + plane + pos, chunk, MPI_DOUBLE, 0, 2, comm));
                else if (!rank) MPI_CHECK(MPI_Recv(full.data() + off + pos, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                pos += chunk;
            }
        }
        if (!rank && results) print_results(full, "Grid");
        if (!rank && validate) {
            printf("Validating result...\n"); int bad = 0; Real lo = full[0], hi = full[0];
            #pragma omp parallel for reduction(|:bad) reduction(min:lo) reduction(max:hi)
            for (size_t i = 0; i < full.size(); ++i) { Real v = full[i]; bad |= !std::isfinite(v); lo = std::min(lo, v); hi = std::max(hi, v); }
            if (bad) printf("Validation failed: found NaN or Inf value\n");
            else { printf("Value range: [%.6f, %.6f]\n", lo, hi); if (hi > 1e6 || lo < -1e6) { printf("Validation failed: values out of expected range\n"); bad = 1; } }
            printf("Validation: %s\n", bad ? "FAILED" : "PASSED"); if (bad) MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer)); CUDA(cudaFree(a)); CUDA(cudaFree(b));
    CUDA(cudaFreeHost(send_lo)); CUDA(cudaFreeHost(send_hi)); CUDA(cudaFreeHost(recv_lo)); CUDA(cudaFreeHost(recv_hi));
    MPI_Comm_free(&comm); MPI_Finalize(); return 0;
}
