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

#define CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Each slab has one ghost plane on either side. Physical boundaries clamp to
// the edge plane, just as in the original stencil.
__global__ void stencil(const double* in, const double* cold, double* out,
                        size_t nx, size_t ny, size_t local_z, bool chemical) {
    size_t plane = nx * ny, n = plane * local_z;
    for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n;
         j += (size_t)blockDim.x * gridDim.x) {
        size_t x = j % nx, y = (j / nx) % ny, i = j + plane;
        double v = in[i];
        double xx = in[i + (x + 1 < nx ? 1 : 0)] + in[i - (x > 0 ? 1 : 0)] - 2.0 * v;
        double yy = in[i + (y + 1 < ny ? nx : 0)] + in[i - (y > 0 ? nx : 0)] - 2.0 * v;
        double zz = in[i + plane] + in[i - plane] - 2.0 * v;
        double lap = xx + yy + zz;
        if (chemical) {
            const double eaa = -(2.0 / 9.0), ebb = -(2.0 / 9.0), eab = 2.0 / 9.0;
            out[i] = 4.5 * ((v + 1.0) * eaa + (v - 1.0) * ebb - 2.0 * v * eab)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else {
            out[i] = cold[i] + 0.01 * lap;
        }
    }
}

// Host staging works with MPI installations that are not CUDA-aware.
void exchange(double* field, size_t plane, size_t local_z, int rank, int ranks,
              double* send, double* recv) {
    size_t bytes = plane * sizeof(double);
    if (rank > 0) CUDA(cudaMemcpy(send, field + plane, bytes, cudaMemcpyDeviceToHost));
    if (rank + 1 < ranks) CUDA(cudaMemcpy(send + plane, field + local_z * plane, bytes, cudaMemcpyDeviceToHost));
    int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(send, (int)plane, MPI_DOUBLE, lower, 0, recv + plane, (int)plane, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send + plane, (int)plane, MPI_DOUBLE, upper, 1, recv, (int)plane, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (lower != MPI_PROC_NULL) CUDA(cudaMemcpy(field, recv, bytes, cudaMemcpyHostToDevice));
    else CUDA(cudaMemcpy(field, field + plane, bytes, cudaMemcpyDeviceToDevice));
    if (upper != MPI_PROC_NULL) CUDA(cudaMemcpy(field + (local_z + 1) * plane, recv + plane, bytes, cudaMemcpyHostToDevice));
    else CUDA(cudaMemcpy(field + (local_z + 1) * plane, field + local_z * plane, bytes, cudaMemcpyDeviceToDevice));
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n", p);
    printf("  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nz > SIZE_MAX / (nx * ny) || nz < (size_t)ranks) {
        if (!rank) fprintf(stderr, "Invalid grid dimensions or more MPI ranks than Z planes.\n");
        MPI_Finalize(); return 1;
    }
    size_t plane = nx * ny, volume = plane * nz;
    size_t local_z = nz / ranks + (size_t(rank) < nz % ranks);
    size_t z0 = size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
    if (local_z > SIZE_MAX / plane - 2 || (local_z + 2) * plane > SIZE_MAX / sizeof(double)) {
        if (!rank) fprintf(stderr, "Grid is too large.\n"); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int local_rank, devices; MPI_Comm_rank(node, &local_rank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&node);
    if (!rank) printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    size_t n = plane * local_z, bytes = (local_z + 2) * plane * sizeof(double);
    std::vector<double> host(n);
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; ++j) {
        size_t id = z0 * plane + j;
        double pseudo = (((id + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
        host[j] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu, *send, *recv;
    CUDA(cudaMalloc(&cold, bytes)); CUDA(cudaMalloc(&cnew, bytes)); CUDA(cudaMalloc(&mu, bytes));
    CUDA(cudaMallocHost(&send, 2 * plane * sizeof(double))); CUDA(cudaMallocHost(&recv, 2 * plane * sizeof(double)));
    CUDA(cudaMemcpy(cold + plane, host.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    int blocks = (int)std::min((n + 255) / 256, size_t(65535));
    for (int t = 0; t < iterations; ++t) {
        exchange(cold, plane, local_z, rank, ranks, send, recv);
        stencil<<<blocks, 256>>>(cold, nullptr, mu, nx, ny, local_z, true); CUDA(cudaGetLastError());
        exchange(mu, plane, local_z, rank, ranks, send, recv);
        stencil<<<blocks, 256>>>(mu, cold, cnew, nx, ny, local_z, false); CUDA(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA(cudaDeviceSynchronize()); double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) { long ms = (long)(max_elapsed * 1000); printf("Computation time: %ld ms\n", ms); printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / max_elapsed / 1e6); }
    if (results || validate) CUDA(cudaMemcpy(host.data(), cold + plane, n * sizeof(double), cudaMemcpyDeviceToHost));
    if (results) {
        std::vector<double> all;
        if (!rank) {
            all.resize(volume); std::copy(host.begin(), host.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t rz = nz / ranks + (size_t(r) < nz % ranks);
                size_t offset = (size_t(r) * (nz / ranks) + std::min(size_t(r), nz % ranks)) * plane;
                size_t remaining = rz * plane;
                while (remaining) { int count = (int)std::min(remaining, size_t(INT_MAX)); MPI_Recv(all.data() + offset, count, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); offset += count; remaining -= count; }
            }
            print_results(all, "Concentration");
        } else {
            size_t offset = 0; while (offset < n) { int count = (int)std::min(n - offset, size_t(INT_MAX)); MPI_Send(host.data() + offset, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD); offset += count; }
        }
    }
    int valid = 1;
    if (validate) {
        double minval = std::numeric_limits<double>::infinity(), maxval = -minval; int finite = 1;
        #pragma omp parallel for reduction(min:minval) reduction(max:maxval) reduction(&:finite) schedule(static)
        for (size_t j = 0; j < n; ++j) { double v = host[j]; finite &= std::isfinite(v); minval = std::min(minval, v); maxval = std::max(maxval, v); }
        double gmin, gmax; int gfinite;
        MPI_Allreduce(&minval, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&maxval, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&finite, &gfinite, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
        valid = gfinite && gmin >= -10.0 && gmax <= 10.0;
        if (!rank) { printf("Validating result...\n"); if (!gfinite) printf("Validation failed: found NaN or Inf value\n"); else { printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax); if (!valid) printf("Validation failed: values out of expected range\n"); } printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
    }
    CUDA(cudaFree(cold)); CUDA(cudaFree(cnew)); CUDA(cudaFree(mu)); CUDA(cudaFreeHost(send)); CUDA(cudaFreeHost(recv));
    MPI_Finalize(); return valid ? 0 : 1;
}
