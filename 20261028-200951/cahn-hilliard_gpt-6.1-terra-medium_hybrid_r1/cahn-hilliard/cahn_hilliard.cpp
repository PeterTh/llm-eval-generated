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

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Planes 1..nz hold the subdomain; planes 0 and nz+1 are neighbor halos.
__global__ void stencil(const double* in, const double* cold, double* out,
                        size_t nx, size_t ny, size_t nz, bool lower, bool upper,
                        bool boundary, bool chemical) {
    size_t plane = nx * ny;
    size_t n = plane * (boundary ? (nz == 1 ? 1 : 2) : nz - 2);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        size_t slab_z = i / plane, rem = i % plane, y = rem / nx, x = rem % nx;
        size_t z = boundary ? (slab_z == 0 ? 0 : nz - 1) : slab_z + 1;
        size_t p = (z + 1) * plane + rem;
        double v = in[p];
        double xx = in[p + (x + 1 < nx)] + in[p - (x > 0)] - 2.0 * v;
        double yy = in[p + (y + 1 < ny ? nx : 0)] + in[p - (y > 0 ? nx : 0)] - 2.0 * v;
        double zz = in[p + (z + 1 < nz || upper ? plane : 0)] +
                    in[p - (z > 0 || lower ? plane : 0)] - 2.0 * v;
        double lap = xx + yy + zz;
        if (chemical) {
            const double aa = -(2.0 / 9.0), bb = -(2.0 / 9.0), ab = 2.0 / 9.0;
            out[p] = 4.5 * ((v + 1.0) * aa + (v - 1.0) * bb - 2.0 * v * ab)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else out[p] = cold[p] + 0.01 * lap;
    }
}

struct Exchange {
    double *send_lo, *send_hi, *recv_lo, *recv_hi;
    size_t plane, nz;
    int rank, ranks;
    MPI_Comm comm;
    Exchange(size_t p, size_t n, int r, int nr, MPI_Comm c): plane(p), nz(n), rank(r), ranks(nr), comm(c) {
        CUDA(cudaMallocHost(&send_lo, p * sizeof(double)));
        CUDA(cudaMallocHost(&send_hi, p * sizeof(double)));
        CUDA(cudaMallocHost(&recv_lo, p * sizeof(double)));
        CUDA(cudaMallocHost(&recv_hi, p * sizeof(double)));
    }
    ~Exchange() { cudaFreeHost(send_lo); cudaFreeHost(send_hi); cudaFreeHost(recv_lo); cudaFreeHost(recv_hi); }
    void run(double* input, const double* cold, double* output, size_t nx, size_t ny, bool chemical) {
        int lo = rank ? rank - 1 : MPI_PROC_NULL, hi = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        if (lo != MPI_PROC_NULL) CUDA(cudaMemcpy(send_lo, input + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
        if (hi != MPI_PROC_NULL) CUDA(cudaMemcpy(send_hi, input + nz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Request req[4];
        MPI_Irecv(recv_lo, int(plane), MPI_DOUBLE, lo, 1, comm, &req[0]);
        MPI_Irecv(recv_hi, int(plane), MPI_DOUBLE, hi, 0, comm, &req[1]);
        MPI_Isend(send_lo, int(plane), MPI_DOUBLE, lo, 0, comm, &req[2]);
        MPI_Isend(send_hi, int(plane), MPI_DOUBLE, hi, 1, comm, &req[3]);
        const int threads = 256;
        int blocks = int(std::min<size_t>((plane * nz + threads - 1) / threads, 65535));
        int edge_blocks = int(std::min<size_t>((plane * (nz == 1 ? 1 : 2) + threads - 1) / threads, 65535));
        if (nz > 2) stencil<<<blocks, threads>>>(input, cold, output, nx, ny, nz, lo != MPI_PROC_NULL, hi != MPI_PROC_NULL, false, chemical);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (lo != MPI_PROC_NULL) CUDA(cudaMemcpy(input, recv_lo, plane * sizeof(double), cudaMemcpyHostToDevice));
        if (hi != MPI_PROC_NULL) CUDA(cudaMemcpy(input + (nz + 1) * plane, recv_hi, plane * sizeof(double), cudaMemcpyHostToDevice));
        stencil<<<edge_blocks, threads>>>(input, cold, output, nx, ny, nz, lo != MPI_PROC_NULL, hi != MPI_PROC_NULL, true, chemical);
        CUDA(cudaGetLastError());
    }
};

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}
int main(int argc, char** argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) { if (!world_rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h")) { if (!world_rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* opt = argv[i++]; char* end; long long v = strtoll(argv[i], &end, 10);
            if (*end || end == argv[i] || v < 0 || v > INT_MAX || (!strcmp(opt, "-x") && v == 0)) { if (!world_rank) fprintf(stderr, "Invalid value for %s\n", opt); MPI_Finalize(); return 1; }
            if (!strcmp(opt, "-x")) nx = size_t(v); else if (!strcmp(opt, "-y")) ny = size_t(v); else if (!strcmp(opt, "-z")) nz = size_t(v); else iterations = int(v);
        } else { if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (nx > SIZE_MAX / ny || nx * ny > INT_MAX || nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Grid is too large\n"); MPI_Finalize(); return 1;
    }
    int active = int(std::min<size_t>(world_size, nz));
    MPI_Comm comm; MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, ranks; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    MPI_Comm local; MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices));
    size_t plane = nx * ny, volume = plane * nz;
    size_t local_nz = nz / ranks + (size_t(rank) < nz % ranks);
    size_t z0 = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    size_t count = plane * local_nz;
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        size_t id = z0 * plane + i;
        double pseudo = (((id + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    size_t bytes = (local_nz + 2) * plane * sizeof(double);
    CUDA(cudaMalloc(&cold, bytes)); CUDA(cudaMalloc(&cnew, bytes)); CUDA(cudaMalloc(&mu, bytes));
    CUDA(cudaMemcpy(cold + plane, host.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    {
        Exchange exchange(plane, local_nz, rank, ranks, comm);
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm); double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            exchange.run(cold, cold, mu, nx, ny, true);
            exchange.run(mu, cold, cnew, nx, ny, false);
            std::swap(cold, cnew);
        }
        CUDA(cudaDeviceSynchronize());
        double elapsed = MPI_Wtime() - start, max_elapsed;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Computation time: %ld ms\n", long(max_elapsed * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / max_elapsed / 1e6);
        }
    }
    if (validate || results) CUDA(cudaMemcpy(host.data(), cold + plane, count * sizeof(double), cudaMemcpyDeviceToHost));
    if (results) {
        std::vector<double> all;
        if (!rank) {
            all.resize(volume);
            std::copy(host.begin(), host.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t rn = nz / ranks + (size_t(r) < nz % ranks);
                size_t rz = size_t(r) * (nz / ranks) + std::min<size_t>(r, nz % ranks);
                size_t remaining = rn * plane, offset = rz * plane;
                while (remaining) { int chunk = int(std::min<size_t>(remaining, INT_MAX)); MPI_Recv(all.data() + offset, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE); offset += chunk; remaining -= chunk; }
            }
            print_results(all, "Concentration");
        } else {
            size_t remaining = count, offset = 0;
            while (remaining) { int chunk = int(std::min<size_t>(remaining, INT_MAX)); MPI_Send(host.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm); offset += chunk; remaining -= chunk; }
        }
    }
    int failed = 0;
    if (validate) {
        double local_min = std::numeric_limits<double>::infinity(), local_max = -local_min;
        int nonfinite = 0;
        #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(|:nonfinite) schedule(static)
        for (size_t i = 0; i < count; ++i) {
            double v = host[i]; nonfinite |= !std::isfinite(v);
            if (v < local_min) local_min = v; if (v > local_max) local_max = v;
        }
        double minval, maxval; int bad;
        MPI_Reduce(&local_min, &minval, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&local_max, &maxval, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&nonfinite, &bad, 1, MPI_INT, MPI_BOR, 0, comm);
        if (!rank) {
            printf("Validating result...\n");
            if (bad) { printf("Validation failed: found NaN or Inf value\n"); failed = 1; }
            else { printf("Concentration range: [%.6f, %.6f]\n", minval, maxval); if (maxval > 10 || minval < -10) { printf("Validation failed: values out of expected range\n"); failed = 1; } }
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
    }
    CUDA(cudaFree(cold)); CUDA(cudaFree(cnew)); CUDA(cudaFree(mu));
    MPI_Comm_free(&comm); MPI_Finalize(); return failed;
}
