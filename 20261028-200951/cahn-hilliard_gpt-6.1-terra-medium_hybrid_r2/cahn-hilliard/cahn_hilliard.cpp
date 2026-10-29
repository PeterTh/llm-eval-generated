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

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Local planes are numbered 1..nz, with a halo on either side.
__global__ void stencil(const double* in, const double* cold, double* out,
                        size_t nx, size_t ny, size_t first, size_t planes, bool chemical) {
    const size_t plane = nx * ny, count = plane * planes;
    for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < count; j += (size_t)blockDim.x * gridDim.x) {
        size_t x = j % nx, y = (j / nx) % ny, z = first + j / plane;
        size_t i = z * plane + y * nx + x;
        double v = in[i];
        double xx = (in[i + (x + 1 < nx ? 1 : 0)] + in[i - (x > 0 ? 1 : 0)] - 2.0 * v);
        double yy = (in[i + (y + 1 < ny ? nx : 0)] + in[i - (y > 0 ? nx : 0)] - 2.0 * v);
        double zz = (in[i + plane] + in[i - plane] - 2.0 * v);
        double lap = xx + yy + zz;
        if (chemical) {
            const double aa = -(2.0 / 9.0), bb = -(2.0 / 9.0), ab = 2.0 / 9.0;
            out[i] = 4.5 * ((v + 1.0) * aa + (v - 1.0) * bb - 2.0 * v * ab)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else out[i] = cold[i] + 0.01 * lap;
    }
}

static void launch(const double* in, const double* cold, double* out, size_t nx, size_t ny,
                   size_t first, size_t planes, bool chemical, cudaStream_t stream) {
    if (!planes) return;
    size_t blocks = (nx * ny * planes + 255) / 256;
    stencil<<<(unsigned)std::min<size_t>(blocks, 65535), 256, 0, stream>>>(in, cold, out, nx, ny, first, planes, chemical);
    CUDA(cudaGetLastError());
}

// Stage boundary planes through pinned host memory, allowing ordinary MPI implementations
// to work without CUDA-aware support. Interior computation runs while MPI transfers halos.
static void pass(double* in, const double* cold, double* out, size_t nx, size_t ny, size_t nz,
                 bool chemical, int rank, int ranks, double* host, cudaStream_t compute, cudaStream_t transfer) {
    size_t plane = nx * ny, bytes = plane * sizeof(double);
    if (ranks == 1) {
        CUDA(cudaMemcpyAsync(in, in + plane, bytes, cudaMemcpyDeviceToDevice, compute));
        CUDA(cudaMemcpyAsync(in + (nz + 1) * plane, in + nz * plane, bytes, cudaMemcpyDeviceToDevice, compute));
        launch(in, cold, out, nx, ny, 1, nz, chemical, compute);
        return;
    }
    launch(in, cold, out, nx, ny, 2, nz > 2 ? nz - 2 : 0, chemical, compute);
    CUDA(cudaMemcpyAsync(host, in + plane, bytes, cudaMemcpyDeviceToHost, transfer));
    CUDA(cudaMemcpyAsync(host + plane, in + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
    CUDA(cudaStreamSynchronize(transfer));
    int lower = rank ? rank - 1 : MPI_PROC_NULL, upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Request req[4];
    MPI_Irecv(host + 2 * plane, (int)plane, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(host + 3 * plane, (int)plane, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(host, (int)plane, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(host + plane, (int)plane, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (lower != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(in, host + 2 * plane, bytes, cudaMemcpyHostToDevice, transfer));
    else CUDA(cudaMemcpyAsync(in, in + plane, bytes, cudaMemcpyDeviceToDevice, transfer));
    if (upper != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(in + (nz + 1) * plane, host + 3 * plane, bytes, cudaMemcpyHostToDevice, transfer));
    else CUDA(cudaMemcpyAsync(in + (nz + 1) * plane, in + nz * plane, bytes, cudaMemcpyDeviceToDevice, transfer));
    CUDA(cudaStreamSynchronize(transfer));
    launch(in, cold, out, nx, ny, 1, 1, chemical, compute);
    if (nz > 1) launch(in, cold, out, nx, ny, nz, 1, chemical, compute);
    CUDA(cudaStreamSynchronize(compute));
}

static void usage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}
int main(int argc, char** argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
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
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nz > SIZE_MAX / (nx * ny) || (size_t)ranks > nz || provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "Invalid grid size, MPI thread support, or more MPI ranks than Z planes.\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    size_t plane = nx * ny, vol = plane * nz, nlocal = nz / ranks + ((size_t)rank < nz % ranks),
           z0 = (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks), count = nlocal * plane;
    if (count > SIZE_MAX / sizeof(double) - 2 * plane) { fprintf(stderr, "Grid allocation too large\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (!rank) printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nInitializing concentration field...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    std::vector<double> data(count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        size_t id = z0 * plane + i;
        double pseudo = (((id + 1) * (size_t)1299709) % vol) / (double)vol;
        data[i] = -1.0 + 2.0 * pseudo;
    }
    double *a, *b, *mu, *host; size_t bytes = (nlocal + 2) * plane * sizeof(double);
    CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes)); CUDA(cudaMalloc(&mu, bytes)); CUDA(cudaMallocHost(&host, 4 * plane * sizeof(double)));
    cudaStream_t compute, transfer; CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking)); CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    CUDA(cudaMemcpy(a + plane, data.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        pass(a, a, mu, nx, ny, nlocal, true, rank, ranks, host, compute, transfer);
        pass(mu, a, b, nx, ny, nlocal, false, rank, ranks, host, compute, transfer);
        std::swap(a, b);
    }
    CUDA(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed; MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", (long)(max_elapsed * 1000), (double)vol * iterations / max_elapsed / 1e6);
    int valid = 1;
    if (validate || results) {
        CUDA(cudaMemcpy(data.data(), a + plane, count * sizeof(double), cudaMemcpyDeviceToHost));
        if (results) {
            std::vector<double> all; if (!rank) { all.resize(vol); std::copy(data.begin(), data.end(), all.begin()); }
            for (int r = 1; r < ranks; ++r) {
                size_t rn = nz / ranks + ((size_t)r < nz % ranks), rz = (nz / ranks) * r + std::min<size_t>(r, nz % ranks), n = rn * plane;
                for (size_t offset = 0; offset < n; ) { int chunk = (int)std::min<size_t>(n - offset, INT_MAX);
                    if (rank == r) MPI_Send(data.data() + offset, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                    if (!rank) MPI_Recv(all.data() + rz * plane + offset, chunk, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk;
                }
            }
            if (!rank) print_results(all, "Concentration");
        }
        if (validate) {
            double mn = INFINITY, mx = -INFINITY; int bad = 0;
            #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(|:bad) schedule(static)
            for (size_t i = 0; i < count; ++i) { double v = data[i]; if (!std::isfinite(v)) bad = 1; else { mn = std::min(mn, v); mx = std::max(mx, v); } }
            double gmin, gmax; int gbad;
            MPI_Reduce(&mn, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); MPI_Reduce(&mx, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); MPI_Reduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!rank) { printf("Validating result...\n"); if (gbad) printf("Validation failed: found NaN or Inf value\n"); else { printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax); if (gmax > 10 || gmin < -10) printf("Validation failed: values out of expected range\n"); } valid = !gbad && gmax <= 10 && gmin >= -10; printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer)); CUDA(cudaFreeHost(host)); CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFree(mu)); MPI_Finalize(); return valid ? 0 : 1;
}
