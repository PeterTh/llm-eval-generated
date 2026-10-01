#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each device field has a ghost Z plane before and after its owned planes.
template<bool Chemical>
__global__ void stencil(const double* __restrict__ c, const double* __restrict__ mu,
                        double* __restrict__ out, size_t nx, size_t ny,
                        size_t first, size_t count) {
    const size_t plane = nx * ny, work = plane * count;
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < work; t += size_t(blockDim.x) * gridDim.x) {
        const size_t z = first + t / plane, j = t % plane;
        const size_t x = j % nx, y = j / nx, i = z * plane + j;
        const double* a = Chemical ? c : mu;
        const double v = a[i];
        const double xx = a[i + (x + 1 < nx)] + a[i - (x > 0)] - 2.0 * v;
        const double yy = a[i + (y + 1 < ny ? nx : 0)] +
                          a[i - (y > 0 ? nx : 0)] - 2.0 * v;
        const double zz = a[i + plane] + a[i - plane] - 2.0 * v;
        if constexpr (Chemical) {
            constexpr double aa = -(2.0 / 9.0), bb = -(2.0 / 9.0), ab = 2.0 / 9.0;
            out[i] = 4.5 * ((v + 1.0) * aa + (v - 1.0) * bb - 2.0 * v * ab)
                   + 3.0 * v + v * v * v - 0.5 * (xx + yy + zz);
        } else {
            out[i] = c[i] + 0.01 * (xx + yy + zz);
        }
    }
}

template<bool Chemical>
static void launch(const double* c, const double* mu, double* out,
                   size_t nx, size_t ny, size_t first, size_t count,
                   cudaStream_t stream) {
    if (!count) return;
    const size_t blocks = std::min<size_t>((count * nx * ny + 255) / 256, 65535);
    stencil<Chemical><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        c, mu, out, nx, ny, first, count);
    cudaCheck(cudaGetLastError(), "stencil launch");
}

struct Halo {
    double *sendLow, *sendHigh, *recvLow, *recvHigh;
};

static Halo makeHalo(size_t bytes) {
    Halo h{};
    cudaCheck(cudaMallocHost(&h.sendLow, bytes), "allocate send halo");
    cudaCheck(cudaMallocHost(&h.sendHigh, bytes), "allocate send halo");
    cudaCheck(cudaMallocHost(&h.recvLow, bytes), "allocate receive halo");
    cudaCheck(cudaMallocHost(&h.recvHigh, bytes), "allocate receive halo");
    return h;
}

static void exchange(double* a, size_t localZ, size_t plane, int rank, int ranks,
                     Halo& h, cudaStream_t stream) {
    const size_t bytes = plane * sizeof(double);
    const int low = rank ? rank - 1 : MPI_PROC_NULL;
    const int high = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    if (low != MPI_PROC_NULL)
        cudaCheck(cudaMemcpyAsync(h.sendLow, a + plane, bytes, cudaMemcpyDeviceToHost, stream), "stage low halo");
    else
        cudaCheck(cudaMemcpyAsync(a, a + plane, bytes, cudaMemcpyDeviceToDevice, stream), "clamp low face");
    if (high != MPI_PROC_NULL)
        cudaCheck(cudaMemcpyAsync(h.sendHigh, a + localZ * plane, bytes, cudaMemcpyDeviceToHost, stream), "stage high halo");
    else
        cudaCheck(cudaMemcpyAsync(a + (localZ + 1) * plane, a + localZ * plane, bytes,
                                  cudaMemcpyDeviceToDevice, stream), "clamp high face");
    cudaCheck(cudaStreamSynchronize(stream), "stage halo");

    MPI_Request req[4];
    int n = 0;
    if (low != MPI_PROC_NULL)
        MPI_Irecv(h.recvLow, static_cast<int>(plane), MPI_DOUBLE, low, 1, MPI_COMM_WORLD, &req[n++]);
    if (high != MPI_PROC_NULL)
        MPI_Irecv(h.recvHigh, static_cast<int>(plane), MPI_DOUBLE, high, 0, MPI_COMM_WORLD, &req[n++]);
    if (low != MPI_PROC_NULL)
        MPI_Isend(h.sendLow, static_cast<int>(plane), MPI_DOUBLE, low, 0, MPI_COMM_WORLD, &req[n++]);
    if (high != MPI_PROC_NULL)
        MPI_Isend(h.sendHigh, static_cast<int>(plane), MPI_DOUBLE, high, 1, MPI_COMM_WORLD, &req[n++]);
    if (n) MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
    if (low != MPI_PROC_NULL)
        cudaCheck(cudaMemcpyAsync(a, h.recvLow, bytes, cudaMemcpyHostToDevice, stream), "receive low halo");
    if (high != MPI_PROC_NULL)
        cudaCheck(cudaMemcpyAsync(a + (localZ + 1) * plane, h.recvHigh, bytes,
                                  cudaMemcpyHostToDevice, stream), "receive high halo");
    cudaCheck(cudaStreamSynchronize(stream), "install halo");
}

static void usage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || nz < size_t(ranks) || iterations < 0 ||
        nx > SIZE_MAX / ny || nx * ny > INT_MAX || nx * ny > SIZE_MAX / nz ||
        nz > SIZE_MAX / (nx * ny) - 2) {
        if (!rank) fprintf(stderr, "Invalid grid, iteration count, or too many MPI ranks\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, cells = plane * nz;
    const size_t localZ = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t firstZ = (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
    if (localZ * plane > INT_MAX) {
        if (!rank) fprintf(stderr, "Local slab exceeds MPI count limit\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank, gpuCount = 0;
    MPI_Comm_rank(shared, &localRank);
    cudaCheck(cudaGetDeviceCount(&gpuCount), "find CUDA devices");
    if (!gpuCount) { fprintf(stderr, "Rank %d has no CUDA device\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % gpuCount), "select CUDA device");
    MPI_Comm_free(&shared);

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> initial(localZ * plane);
#pragma omp parallel for schedule(static)
    for (size_t z = 0; z < localZ; ++z)
        for (size_t j = 0; j < plane; ++j) {
            const size_t id = (firstZ + z) * plane + j;
            const double pseudo = (((id + 1) * size_t(1299709)) % cells) / double(cells);
            initial[z * plane + j] = -1.0 + 2.0 * pseudo;
        }
    const size_t bytes = (localZ + 2) * plane * sizeof(double);
    double *cold, *cnew, *mu;
    cudaCheck(cudaMalloc(&cold, bytes), "allocate concentration");
    cudaCheck(cudaMalloc(&cnew, bytes), "allocate next concentration");
    cudaCheck(cudaMalloc(&mu, bytes), "allocate chemical potential");
    cudaCheck(cudaMemcpy(cold + plane, initial.data(), initial.size() * sizeof(double),
                         cudaMemcpyHostToDevice), "initialize device field");
    std::vector<double>().swap(initial);
    Halo halo = makeHalo(plane * sizeof(double));
    cudaStream_t compute, transfer;
    cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "create compute stream");
    cudaCheck(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking), "create transfer stream");

    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        launch<true>(cold, nullptr, mu, nx, ny, 2, localZ > 2 ? localZ - 2 : 0, compute);
        exchange(cold, localZ, plane, rank, ranks, halo, transfer);
        launch<true>(cold, nullptr, mu, nx, ny, 1, 1, compute);
        if (localZ > 1) launch<true>(cold, nullptr, mu, nx, ny, localZ, 1, compute);
        cudaCheck(cudaStreamSynchronize(compute), "finish chemical potential");
        launch<false>(cold, mu, cnew, nx, ny, 2, localZ > 2 ? localZ - 2 : 0, compute);
        exchange(mu, localZ, plane, rank, ranks, halo, transfer);
        launch<false>(cold, mu, cnew, nx, ny, 1, 1, compute);
        if (localZ > 1) launch<false>(cold, mu, cnew, nx, ny, localZ, 1, compute);
        cudaCheck(cudaStreamSynchronize(compute), "finish concentration update");
        std::swap(cold, cnew);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n",
               seconds > 0 ? double(cells) * iterations / seconds / 1e6 : 0.0);
    }

    std::vector<double> local;
    if (results || validate) {
        local.resize(localZ * plane);
        cudaCheck(cudaMemcpy(local.data(), cold + plane, local.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy result");
    }
    if (results) {
        if (!rank) {
            std::vector<double> all(cells);
            std::copy(local.begin(), local.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                const size_t z = nz / ranks + (size_t(r) < nz % ranks);
                const size_t first = (nz / ranks) * r + std::min<size_t>(r, nz % ranks);
                MPI_Recv(all.data() + first * plane, static_cast<int>(z * plane),
                         MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            print_results(all, "Concentration");
        } else {
            MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                     0, 2, MPI_COMM_WORLD);
        }
    }
    int status = 0;
    if (validate) {
        double lo = INFINITY, hi = -INFINITY;
        int bad = 0;
#pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(+:bad)
        for (size_t i = 0; i < local.size(); ++i) {
            const double v = local[i];
            if (!std::isfinite(v)) ++bad;
            else { lo = std::min(lo, v); hi = std::max(hi, v); }
        }
        double globalLo, globalHi;
        int globalBad;
        MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        status = globalBad || globalLo < -10.0 || globalHi > 10.0;
        if (!rank) {
            printf("Validating result...\n");
            if (globalBad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                if (status) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    cudaCheck(cudaStreamDestroy(compute), "destroy compute stream");
    cudaCheck(cudaStreamDestroy(transfer), "destroy transfer stream");
    cudaCheck(cudaFree(cold), "free concentration");
    cudaCheck(cudaFree(cnew), "free next concentration");
    cudaCheck(cudaFree(mu), "free chemical potential");
    cudaCheck(cudaFreeHost(halo.sendLow), "free send halo");
    cudaCheck(cudaFreeHost(halo.sendHigh), "free send halo");
    cudaCheck(cudaFreeHost(halo.recvLow), "free receive halo");
    cudaCheck(cudaFreeHost(halo.recvHigh), "free receive halo");
    MPI_Finalize();
    return status;
}
