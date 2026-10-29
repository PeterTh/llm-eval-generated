#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include "../common/results_output.hpp"

using Real = double;

static void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cuda_check((call), __FILE__, __LINE__)

// The two ghost planes contain neighboring ranks' current values. Global
// boundaries are initialized in both buffers and never overwritten.
__global__ void stencil(const Real* in, Real* out, size_t nx, size_t ny,
                        size_t nz, size_t zstart, size_t first, size_t last) {
    size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    size_t z = size_t(blockIdx.z) + first;
    if (x >= nx - 1 || y >= ny - 1 || z >= last || zstart + z - 1 == 0 || zstart + z - 1 == nz - 1) return;
    size_t plane = nx * ny, p = z * plane + y * nx + x;
    out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] + in[p-plane] + in[p+plane]) / 7.0;
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny, size_t nz,
                   size_t zstart, size_t first, size_t last, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || first >= last) return;
    dim3 block(32, 4);
    // Split Z launches to respect the CUDA grid dimension limit.
    while (first < last) {
        size_t end = std::min(last, first + size_t(65535));
        dim3 grid((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y, end - first);
        stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, zstart, first, end);
        CUDA(cudaGetLastError());
        first = end;
    }
}

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided, worldRank, worldSize;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) { if (!worldRank) fprintf(stderr, "MPI thread support unavailable\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* opt = argv[i++]; char* end = nullptr;
            long long v = strtoll(argv[i], &end, 10);
            if (*end || v < 0 || v > INT_MAX || (!strcmp(opt, "-x") && v == 0)) { if (!worldRank) fprintf(stderr, "Invalid value for %s\n", opt); MPI_Finalize(); return 1; }
            if (!strcmp(opt, "-x")) nx = v;
            else if (!strcmp(opt, "-y")) ny = v;
            else if (!strcmp(opt, "-z")) nz = v;
            else iterations = int(v);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!worldRank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!worldRank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz || nx * ny > INT_MAX || nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!worldRank) fprintf(stderr, "Grid size exceeds supported limits\n"); MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    // Ranks without a plane take no part in the calculation.
    int active = int(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= active) { MPI_Finalize(); return 0; }
    int rank; MPI_Comm_rank(comm, &rank);
    size_t count = nz / active + (size_t(rank) < nz % active);
    size_t start = size_t(rank) * (nz / active) + std::min<size_t>(rank, nz % active);
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices; MPI_Comm_rank(local, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA devices available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    std::vector<Real> host(count * plane);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count * plane; ++i) host[i] = Real((start * plane + i) % 19);
    Real *a, *b, *sendLo, *sendHi, *recvLo, *recvHi;
    CUDA(cudaMalloc(&a, (count + 2) * plane * sizeof(Real)));
    CUDA(cudaMalloc(&b, (count + 2) * plane * sizeof(Real)));
    CUDA(cudaMemcpy(a + plane, host.data(), count * plane * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b + plane, host.data(), count * plane * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMallocHost(&sendLo, plane * sizeof(Real))); CUDA(cudaMallocHost(&sendHi, plane * sizeof(Real)));
    CUDA(cudaMallocHost(&recvLo, plane * sizeof(Real))); CUDA(cudaMallocHost(&recvHi, plane * sizeof(Real)));
    cudaStream_t compute, transfer; CUDA(cudaStreamCreate(&compute)); CUDA(cudaStreamCreate(&transfer));
    int lo = rank ? rank - 1 : MPI_PROC_NULL, hi = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
    if (!rank) printf("Running stencil computation...\n");
    MPI_Barrier(comm); double begin = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        // Interior computation overlaps staged halo transfers and MPI traffic.
        launch(a, b, nx, ny, nz, start, 2, count, compute);
        if (lo != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(sendLo, a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
        if (hi != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(sendHi, a + count * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
        MPI_Request req[4]; int n = 0;
        if (lo != MPI_PROC_NULL) MPI_Irecv(recvLo, int(plane), MPI_DOUBLE, lo, 1, comm, &req[n++]);
        if (hi != MPI_PROC_NULL) MPI_Irecv(recvHi, int(plane), MPI_DOUBLE, hi, 0, comm, &req[n++]);
        CUDA(cudaStreamSynchronize(transfer));
        if (lo != MPI_PROC_NULL) MPI_Isend(sendLo, int(plane), MPI_DOUBLE, lo, 0, comm, &req[n++]);
        if (hi != MPI_PROC_NULL) MPI_Isend(sendHi, int(plane), MPI_DOUBLE, hi, 1, comm, &req[n++]);
        MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
        if (lo != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(a, recvLo, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
        if (hi != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(a + (count + 1) * plane, recvHi, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
        CUDA(cudaStreamSynchronize(transfer));
        launch(a, b, nx, ny, nz, start, 1, 2, compute);
        if (count > 1) launch(a, b, nx, ny, nz, start, count, count + 1, compute);
        CUDA(cudaStreamSynchronize(compute));
        std::swap(a, b);
    }
    double elapsed = MPI_Wtime() - begin, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        long ms = long(maxElapsed * 1000);
        double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms, updates / maxElapsed / 1e6);
    }
    if (results || validate) CUDA(cudaMemcpy(host.data(), a + plane, count * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    if (results) {
        std::vector<Real> full;
        if (!rank) { full.resize(nx * ny * nz); std::copy(host.begin(), host.end(), full.begin()); }
        // Chunked transfers avoid MPI's int count limit for a slab.
        for (int r = 1; r < active; ++r) {
            size_t rc = nz / active + (size_t(r) < nz % active);
            size_t rs = size_t(r) * (nz / active) + std::min<size_t>(r, nz % active);
            size_t total = rc * plane;
            for (size_t off = 0; off < total;) {
                int chunk = int(std::min<size_t>(total - off, INT_MAX));
                if (rank == r) MPI_Send(host.data() + off, chunk, MPI_DOUBLE, 0, 2, comm);
                if (!rank) MPI_Recv(full.data() + rs * plane + off, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                off += chunk;
            }
        }
        if (!rank) print_results(full, "Grid");
    }
    int valid = 1;
    if (validate) {
        Real minval = std::numeric_limits<Real>::infinity(), maxval = -minval; int bad = 0;
        #pragma omp parallel for reduction(min:minval) reduction(max:maxval) reduction(|:bad) schedule(static)
        for (size_t i = 0; i < host.size(); ++i) { Real v = host[i]; if (!std::isfinite(v)) bad = 1; else { minval = std::min(minval, v); maxval = std::max(maxval, v); } }
        Real globalMin, globalMax; int globalBad;
        MPI_Reduce(&minval, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maxval, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Validating result...\n");
            if (globalBad) printf("Validation failed: found NaN or Inf value\n");
            else { printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax); if (globalMax > 1e6 || globalMin < -1e6) printf("Validation failed: values out of expected range\n"); }
            valid = !globalBad && globalMax <= 1e6 && globalMin >= -1e6;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFreeHost(sendLo)); CUDA(cudaFreeHost(sendHi)); CUDA(cudaFreeHost(recvLo)); CUDA(cudaFreeHost(recvHi));
    MPI_Comm_free(&comm); MPI_Finalize(); return valid ? 0 : 1;
}
