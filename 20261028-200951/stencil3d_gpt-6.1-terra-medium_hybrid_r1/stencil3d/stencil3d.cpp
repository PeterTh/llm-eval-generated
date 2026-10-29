#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
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

static void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cuda_check((call), __FILE__, __LINE__)

// Each block covers a row segment; the third launch dimension selects a plane.
__global__ void stencil(const Real* in, Real* out, size_t nx, size_t ny,
                        size_t first, size_t count, size_t global_first, size_t nz) {
    size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    size_t z = first + blockIdx.z;
    if (x >= nx - 1 || y >= ny - 1 || blockIdx.z >= count ||
        global_first + z - 1 == 0 || global_first + z - 1 == nz - 1) return;
    size_t plane = nx * ny, p = z * plane + y * nx + x;
    out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] + in[p-plane] + in[p+plane]) / 7.0;
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny, size_t first,
                   size_t count, size_t global_first, size_t nz, cudaStream_t stream) {
    if (!count || nx < 3 || ny < 3) return;
    dim3 block(32, 4);
    // CUDA grid z is limited to 65535 planes.
    while (count) {
        unsigned n = static_cast<unsigned>(std::min<size_t>(count, 65535));
        dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                  static_cast<unsigned>((ny - 2 + block.y - 1) / block.y), n);
        stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, first, n, global_first, nz);
        CUDA(cudaGetLastError());
        first += n; count -= n;
    }
}

static void usage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

// MPI counts are int, so transfer large planes and result slabs in chunks.
static void post_chunks(Real* data, size_t n, int peer, int tag, bool send,
                        std::vector<MPI_Request>& requests) {
    while (n) {
        int chunk = static_cast<int>(std::min<size_t>(n, INT_MAX));
        MPI_Request request;
        if (send) MPI_Isend(data, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else MPI_Irecv(data, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        data += chunk; n -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10; bool validate = false, results = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* opt = argv[i++]; char* end = nullptr;
            long long v = strtoll(argv[i], &end, 10);
            if (*end || end == argv[i] || v < 0 || (!strcmp(opt, "-i") && v > INT_MAX)) { bad = true; break; }
            if (!strcmp(opt, "-x")) nx = static_cast<size_t>(v);
            else if (!strcmp(opt, "-y")) ny = static_cast<size_t>(v);
            else if (!strcmp(opt, "-z")) nz = static_cast<size_t>(v);
            else iterations = static_cast<int>(v);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { bad = true; break; }
    }
    if (help || bad) { if (!rank) usage(argv[0]); MPI_Finalize(); return bad ? 1 : 0; }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || static_cast<size_t>(ranks) > nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!rank) fprintf(stderr, "Invalid grid dimensions or more MPI ranks than Z planes\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices));
    const size_t plane = nx * ny, count = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t start = (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
    const size_t slab = plane * count, bytes = plane * sizeof(Real);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(slab);
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < slab; ++p) host[p] = static_cast<Real>((start * plane + p) % 19);
    Real *a, *b, *send_lo, *send_hi, *recv_lo, *recv_hi;
    CUDA(cudaMalloc(&a, (count + 2) * bytes)); CUDA(cudaMalloc(&b, (count + 2) * bytes));
    CUDA(cudaMemcpy(a + plane, host.data(), slab * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b + plane, host.data(), slab * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA(cudaMallocHost(&send_lo, bytes)); CUDA(cudaMallocHost(&send_hi, bytes));
    CUDA(cudaMallocHost(&recv_lo, bytes)); CUDA(cudaMallocHost(&recv_hi, bytes));
    cudaStream_t compute, transfer; CUDA(cudaStreamCreate(&compute)); CUDA(cudaStreamCreate(&transfer));
    if (!rank) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD); double begin = MPI_Wtime();
    for (int it = 0; it < iterations; ++it) {
        std::vector<MPI_Request> requests;
        if (rank > 0) {
            post_chunks(recv_lo, plane, rank - 1, 1, false, requests);
            CUDA(cudaMemcpyAsync(send_lo, a + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        }
        if (rank + 1 < ranks) {
            post_chunks(recv_hi, plane, rank + 1, 0, false, requests);
            CUDA(cudaMemcpyAsync(send_hi, a + count * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        }
        if (count > 2) launch(a, b, nx, ny, 2, count - 2, start, nz, compute);
        CUDA(cudaStreamSynchronize(transfer));
        if (rank > 0) post_chunks(send_lo, plane, rank - 1, 0, true, requests);
        if (rank + 1 < ranks) post_chunks(send_hi, plane, rank + 1, 1, true, requests);
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        if (rank > 0) CUDA(cudaMemcpyAsync(a, recv_lo, bytes, cudaMemcpyHostToDevice, compute));
        if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(a + (count + 1) * plane, recv_hi, bytes, cudaMemcpyHostToDevice, compute));
        launch(a, b, nx, ny, 1, 1, start, nz, compute);
        if (count > 1) launch(a, b, nx, ny, count, 1, start, nz, compute);
        CUDA(cudaStreamSynchronize(compute));
        std::swap(a, b);
    }
    double elapsed = MPI_Wtime() - begin, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", max_elapsed > 0 ? updates / max_elapsed / 1e6 : 0.0);
    }
    if (results || validate) CUDA(cudaMemcpy(host.data(), a + plane, slab * sizeof(Real), cudaMemcpyDeviceToHost));
    if (results) {
        if (!rank) {
            std::vector<Real> all(plane * nz);
            std::copy(host.begin(), host.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t rcount = nz / ranks + (static_cast<size_t>(r) < nz % ranks);
                size_t offset = ((nz / ranks) * r + std::min<size_t>(r, nz % ranks)) * plane;
                size_t remaining = rcount * plane;
                while (remaining) { int n = static_cast<int>(std::min<size_t>(remaining, INT_MAX)); MPI_Recv(all.data() + offset, n, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); offset += n; remaining -= n; }
            }
            print_results(all, "Grid");
        } else {
            size_t offset = 0;
            while (offset < slab) { int n = static_cast<int>(std::min<size_t>(slab - offset, INT_MAX)); MPI_Send(host.data() + offset, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD); offset += n; }
        }
    }
    int valid = 1;
    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -lo; int invalid = 0;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
        for (size_t p = 0; p < slab; ++p) { Real v = host[p]; invalid |= !std::isfinite(v); lo = std::min(lo, v); hi = std::max(hi, v); }
        Real global_lo, global_hi; int global_invalid;
        MPI_Reduce(&lo, &global_lo, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&hi, &global_hi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&invalid, &global_invalid, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) {
            printf("Validating result...\n");
            if (global_invalid) printf("Validation failed: found NaN or Inf value\n");
            else { printf("Value range: [%.6f, %.6f]\n", global_lo, global_hi); if (global_hi > 1e6 || global_lo < -1e6) printf("Validation failed: values out of expected range\n"); }
            valid = !global_invalid && global_hi <= 1e6 && global_lo >= -1e6;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFreeHost(send_lo)); CUDA(cudaFreeHost(send_hi)); CUDA(cudaFreeHost(recv_lo)); CUDA(cudaFreeHost(recv_hi));
    MPI_Finalize(); return valid ? 0 : 1;
}
