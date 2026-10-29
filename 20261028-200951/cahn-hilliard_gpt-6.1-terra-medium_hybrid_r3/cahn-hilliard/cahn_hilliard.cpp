#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include "../common/results_output.hpp"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

__global__ void stencil(const double* in, const double* old, double* out, size_t nx, size_t ny, size_t n, bool chemical) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;
    if (i >= n * plane) return;
    size_t x = i % nx, y = (i / nx) % ny, p = i + plane;
    double v = in[p];
    double xx = in[p + (x + 1 < nx ? 1 : 0)] + in[p - (x > 0 ? 1 : 0)] - 2.0 * v;
    double yy = in[p + (y + 1 < ny ? nx : 0)] + in[p - (y > 0 ? nx : 0)] - 2.0 * v;
    double zz = in[p + plane] + in[p - plane] - 2.0 * v;
    double lap = xx + yy + zz;
    if (chemical) {
        const double aa = -(2.0 / 9.0), bb = -(2.0 / 9.0), ab = 2.0 / 9.0;
        out[p] = 4.5 * ((v + 1.0) * aa + (v - 1.0) * bb - 2.0 * v * ab) + 3.0 * v + v * v * v - 0.5 * lap;
    } else out[p] = old[p] + 0.01 * lap;
}

// Pinned staging permits MPI implementations without GPU-aware support.
void exchange(double* field, size_t plane, size_t n, int rank, int ranks, double* send, double* recv, MPI_Comm comm) {
    size_t bytes = plane * sizeof(double);
    if (ranks == 1) {
        CUDA(cudaMemcpy(field, field + plane, bytes, cudaMemcpyDeviceToDevice));
        CUDA(cudaMemcpy(field + (n + 1) * plane, field + n * plane, bytes, cudaMemcpyDeviceToDevice));
        return;
    }
    CUDA(cudaMemcpy(send, field + plane, bytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(send + plane, field + n * plane, bytes, cudaMemcpyDeviceToHost));
    int left = rank ? rank - 1 : MPI_PROC_NULL;
    int right = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_CHECK(MPI_Sendrecv(send, (int)plane, MPI_DOUBLE, left, 0, recv + plane, (int)plane, MPI_DOUBLE, right, 0, comm, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(send + plane, (int)plane, MPI_DOUBLE, right, 1, recv, (int)plane, MPI_DOUBLE, left, 1, comm, MPI_STATUS_IGNORE));
    CUDA(cudaMemcpy(field, left == MPI_PROC_NULL ? send : recv, bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(field + (n + 1) * plane, right == MPI_PROC_NULL ? send + plane : recv + plane, bytes, cudaMemcpyHostToDevice));
}

void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!world_rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!world_rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nz > SIZE_MAX / (nx * ny) || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Invalid or unsupported grid size\n"); MPI_Finalize(); return 1;
    }
    int active = (int)std::min(nz, (size_t)world_size);
    MPI_Comm comm; MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, ranks; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    MPI_Comm local; MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    size_t plane = nx * ny, vol = plane * nz;
    size_t n = nz / ranks + ((size_t)rank < nz % ranks), z0 = (nz / ranks) * rank + std::min((size_t)rank, nz % ranks);
    std::vector<double> initial(n * plane);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n * plane; ++i) {
        size_t id = z0 * plane + i;
        double pseudo = (((id + 1) * (size_t)1299709) % vol) / (double)vol;
        initial[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu, *send, *recv;
    CUDA(cudaMalloc(&cold, (n + 2) * plane * sizeof(double))); CUDA(cudaMalloc(&cnew, (n + 2) * plane * sizeof(double))); CUDA(cudaMalloc(&mu, (n + 2) * plane * sizeof(double)));
    CUDA(cudaMallocHost(&send, 2 * plane * sizeof(double))); CUDA(cudaMallocHost(&recv, 2 * plane * sizeof(double)));
    CUDA(cudaMemcpy(cold + plane, initial.data(), n * plane * sizeof(double), cudaMemcpyHostToDevice));
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); }
    MPI_Barrier(comm); double start = MPI_Wtime();
    unsigned blocks = (unsigned)((n * plane + 255) / 256);
    for (int t = 0; t < iterations; ++t) {
        exchange(cold, plane, n, rank, ranks, send, recv, comm);
        stencil<<<blocks, 256>>>(cold, cold, mu, nx, ny, n, true); CUDA(cudaGetLastError());
        exchange(mu, plane, n, rank, ranks, send, recv, comm);
        stencil<<<blocks, 256>>>(mu, cold, cnew, nx, ny, n, false); CUDA(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA(cudaDeviceSynchronize()); double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) { long ms = (long)(max_elapsed * 1000.0); printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms, (double)vol * iterations / max_elapsed / 1e6); }
    int status = 0;
    if (results || validate) {
        CUDA(cudaMemcpy(initial.data(), cold + plane, n * plane * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> all;
        if (!rank) {
            all.resize(vol); std::copy(initial.begin(), initial.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t rn = nz / ranks + ((size_t)r < nz % ranks), rz = (nz / ranks) * r + std::min((size_t)r, nz % ranks), count = rn * plane;
                for (size_t off = 0; off < count;) { int chunk = (int)std::min(count - off, (size_t)INT_MAX); MPI_CHECK(MPI_Recv(all.data() + rz * plane + off, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE)); off += chunk; }
            }
            if (results) print_results(all, "Concentration");
            if (validate) {
                printf("Validating result...\n"); bool finite = true; double lo = all[0], hi = all[0];
                for (double v : all) { if (!std::isfinite(v)) finite = false; lo = std::min(lo, v); hi = std::max(hi, v); }
                if (!finite) printf("Validation failed: found NaN or Inf value\n");
                else { printf("Concentration range: [%.6f, %.6f]\n", lo, hi); if (hi > 10.0 || lo < -10.0) { printf("Validation failed: values out of expected range\n"); finite = false; } }
                status = finite ? 0 : 1; printf("Validation: %s\n", finite ? "PASSED" : "FAILED");
            }
        } else for (size_t off = 0; off < initial.size();) { int chunk = (int)std::min(initial.size() - off, (size_t)INT_MAX); MPI_CHECK(MPI_Send(initial.data() + off, chunk, MPI_DOUBLE, 0, 2, comm)); off += chunk; }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    CUDA(cudaFree(cold)); CUDA(cudaFree(cnew)); CUDA(cudaFree(mu)); CUDA(cudaFreeHost(send)); CUDA(cudaFreeHost(recv)); MPI_Comm_free(&comm); MPI_Finalize(); return status;
}
