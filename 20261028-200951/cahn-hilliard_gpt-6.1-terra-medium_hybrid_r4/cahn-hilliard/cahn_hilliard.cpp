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

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__global__ void stencil(const double* in, const double* old, double* out,
                        size_t nx, size_t ny, size_t planes, bool chemical) {
    size_t plane = nx * ny, n = plane * planes;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x) {
        size_t x = i % nx, y = (i / nx) % ny, p = i + plane;
        double v = in[p];
        double xx = in[p + (x + 1 < nx ? 1 : 0)] + in[p - (x > 0 ? 1 : 0)] - 2.0 * v;
        double yy = in[p + (y + 1 < ny ? nx : 0)] + in[p - (y > 0 ? nx : 0)] - 2.0 * v;
        double zz = in[p + plane] + in[p - plane] - 2.0 * v;
        double lap = xx + yy + zz;
        if (chemical) {
            const double aa = -(2.0 / 9.0), bb = -(2.0 / 9.0), ab = 2.0 / 9.0;
            out[p] = 4.5 * ((v + 1.0) * aa + (v - 1.0) * bb - 2.0 * v * ab)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else out[p] = old[p] + 0.01 * lap;
    }
}

// Host staging works with both CUDA-aware and ordinary MPI implementations.
static void exchange(double* d, size_t plane, size_t nz, int rank, int ranks, double* h) {
    size_t bytes = plane * sizeof(double);
    if (ranks == 1) {
        CUDA(cudaMemcpyAsync(d, d + plane, bytes, cudaMemcpyDeviceToDevice));
        CUDA(cudaMemcpyAsync(d + (nz + 1) * plane, d + nz * plane, bytes, cudaMemcpyDeviceToDevice));
        return;
    }
    CUDA(cudaMemcpy(h, d + plane, bytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h + plane, d + nz * plane, bytes, cudaMemcpyDeviceToHost));
    int prev = rank ? rank - 1 : MPI_PROC_NULL, next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(h, (int)plane, MPI_DOUBLE, prev, 0, h + 3 * plane, (int)plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h + plane, (int)plane, MPI_DOUBLE, next, 1, h + 2 * plane, (int)plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    CUDA(cudaMemcpy(d, prev == MPI_PROC_NULL ? h : h + 2 * plane, bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d + (nz + 1) * plane, next == MPI_PROC_NULL ? h + plane : h + 3 * plane, bytes, cudaMemcpyHostToDevice));
}

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
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
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || nz < (size_t)ranks || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nx * ny > SIZE_MAX / nz) {
        if (!rank) fprintf(stderr, "Invalid grid size or too many MPI ranks for Z dimension\n"); MPI_Finalize(); return 1;
    }
    size_t plane = nx * ny, volume = plane * nz;
    size_t local = nz / ranks + ((size_t)rank < nz % ranks);
    size_t startz = (size_t)rank * (nz / ranks) + std::min((size_t)rank, nz % ranks);
    MPI_Comm localcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localcomm);
    int localrank, devices; MPI_Comm_rank(localcomm, &localrank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA devices available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localrank % devices)); MPI_Comm_free(&localcomm);
    if (!rank) printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    std::vector<double> host(local * plane);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < host.size(); ++i) {
        size_t id = startz * plane + i;
        double pseudo = (((id + 1) * (size_t)1299709) % volume) / (double)volume;
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *a, *b, *mu, *halo;
    size_t bytes = (local + 2) * plane * sizeof(double);
    CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes)); CUDA(cudaMalloc(&mu, bytes));
    CUDA(cudaMallocHost(&halo, 4 * plane * sizeof(double)));
    CUDA(cudaMemcpy(a + plane, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    int blocks = (int)std::min((host.size() + 255) / 256, (size_t)65535);
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD); double begin = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        exchange(a, plane, local, rank, ranks, halo);
        stencil<<<blocks, 256>>>(a, nullptr, mu, nx, ny, local, true); CUDA(cudaGetLastError());
        exchange(mu, plane, local, rank, ranks, halo);
        stencil<<<blocks, 256>>>(mu, a, b, nx, ny, local, false); CUDA(cudaGetLastError());
        std::swap(a, b);
    }
    CUDA(cudaDeviceSynchronize()); double elapsed = MPI_Wtime() - begin, maxelapsed;
    MPI_Reduce(&elapsed, &maxelapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", (long)(maxelapsed * 1000), (double)volume * iterations / maxelapsed / 1e6);
    int valid = 1;
    if (results || validate) {
        CUDA(cudaMemcpy(host.data(), a + plane, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
        // Gather in global order, in chunks to avoid MPI's int count limit.
        std::vector<double> all; if (!rank) all.resize(volume);
        for (int r = 0; r < ranks; ++r) {
            size_t n = (nz / ranks + ((size_t)r < nz % ranks)) * plane;
            size_t offset = ((size_t)r * (nz / ranks) + std::min((size_t)r, nz % ranks)) * plane;
            for (size_t p = 0; p < n;) {
                int count = (int)std::min(n - p, (size_t)INT_MAX);
                if (r == 0 && rank == 0) std::copy_n(host.data() + p, count, all.data() + offset + p);
                else if (rank == r) MPI_Send(host.data() + p, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                else if (rank == 0) MPI_Recv(all.data() + offset + p, count, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                p += count;
            }
        }
        if (!rank) {
            if (results) print_results(all, "Concentration");
            if (validate) {
                printf("Validating result...\n");
                for (double v : all) if (!std::isfinite(v)) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; break; }
                if (valid) {
                    auto range = std::minmax_element(all.begin(), all.end());
                    printf("Concentration range: [%.6f, %.6f]\n", *range.first, *range.second);
                    if (*range.second > 10 || *range.first < -10) { printf("Validation failed: values out of expected range\n"); valid = 0; }
                }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFree(mu)); CUDA(cudaFreeHost(halo));
    MPI_Finalize(); return valid ? 0 : 1;
}
