#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        int rank_ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_, __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// The local arrays contain a one-plane halo on each side of the z slab.
// Each thread handles one x/y location in one or more planes.
template<bool chemical, bool clamp_z, bool boundary>
__global__ void stencil(const double* __restrict__ in, const double* __restrict__ cold,
                        double* __restrict__ out, size_t nx, size_t ny,
                        size_t first, size_t planes, size_t last) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t plane = nx * ny;
    for (size_t zi = first + blockIdx.z; zi < first + planes; zi += gridDim.z) {
        const size_t z = boundary && zi == 2 ? last : zi;
        const size_t i = z * plane + y * nx + x;
        const double v = in[i];
        const double xx = in[i + (x + 1 < nx ? 1 : 0)] + in[i - (x > 0 ? 1 : 0)] - 2.0 * v;
        const double yy = in[i + (y + 1 < ny ? nx : 0)] + in[i - (y > 0 ? nx : 0)] - 2.0 * v;
        const double zz = in[i + (clamp_z && z == planes ? 0 : plane)]
                        + in[i - (clamp_z && z == 1 ? 0 : plane)] - 2.0 * v;
        const double lap = xx + yy + zz;
        if (chemical) {
            const double e_AA = -(2.0 / 9.0);
            const double e_BB = -(2.0 / 9.0);
            const double e_AB = 2.0 / 9.0;
            out[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else {
            out[i] = cold[i] + 0.01 * lap;
        }
    }
}

template<bool chemical, bool clamp_z = false, bool boundary = false>
void launch_stencil(const double* in, const double* cold, double* out,
                    size_t nx, size_t ny, size_t first, size_t planes, cudaStream_t stream, size_t last = 0) {
    if (!planes) return;
    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(planes, 65535)));
    stencil<chemical, clamp_z, boundary><<<grid, block, 0, stream>>>(in, cold, out, nx, ny, first, planes, last);
    CUDA_CHECK(cudaGetLastError());
}

struct HaloExchange {
    size_t plane, n;
    int lower, upper;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    double *send_lower = nullptr, *send_upper = nullptr, *recv_lower = nullptr, *recv_upper = nullptr;

    HaloExchange(size_t p, size_t local_n, int rank, int ranks, MPI_Comm c)
        : plane(p), n(local_n), lower(rank ? rank - 1 : MPI_PROC_NULL),
          upper(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL), comm(c) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        if (ranks > 1) {
            CUDA_CHECK(cudaMallocHost(&send_lower, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&send_upper, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_lower, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_upper, plane * sizeof(double)));
        }
    }
    ~HaloExchange() {
        if (send_lower) cudaFreeHost(send_lower);
        if (send_upper) cudaFreeHost(send_upper);
        if (recv_lower) cudaFreeHost(recv_lower);
        if (recv_upper) cudaFreeHost(recv_upper);
        cudaStreamDestroy(transfer); cudaStreamDestroy(compute);
    }

    template<bool chemical>
    void apply(double* in, const double* cold, double* out, size_t nx, size_t ny) {
        const size_t bytes = plane * sizeof(double);
        // The previous kernel must finish before its boundary planes can be sent.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        if (lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(send_lower, in + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        else
            CUDA_CHECK(cudaMemcpyAsync(in, in + plane, bytes, cudaMemcpyDeviceToDevice, transfer));
        if (upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(send_upper, in + n * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        else
            CUDA_CHECK(cudaMemcpyAsync(in + (n + 1) * plane, in + n * plane, bytes, cudaMemcpyDeviceToDevice, transfer));

        // The interior does not depend on halo planes and overlaps staging and MPI.
        if (n > 2) launch_stencil<chemical>(in, cold, out, nx, ny, 2, n - 2, compute);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        // MPI counts are int; chunk very large planes without changing their ordering.
        for (size_t offset = 0; offset < plane;) {
            int count = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
            MPI_Request requests[4];
            MPI_Irecv(recv_lower + offset, count, MPI_DOUBLE, lower, 11, comm, &requests[0]);
            MPI_Irecv(recv_upper + offset, count, MPI_DOUBLE, upper, 10, comm, &requests[1]);
            MPI_Isend(send_lower + offset, count, MPI_DOUBLE, lower, 10, comm, &requests[2]);
            MPI_Isend(send_upper + offset, count, MPI_DOUBLE, upper, 11, comm, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            offset += static_cast<size_t>(count);
        }
        if (lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(in, recv_lower, bytes, cudaMemcpyHostToDevice, transfer));
        if (upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(in + (n + 1) * plane, recv_upper, bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        launch_stencil<chemical, false, true>(in, cold, out, nx, ny, 1, n > 1 ? 2 : 1, compute, n);
    }
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!nx || !ny || !nz || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny > std::numeric_limits<size_t>::max() / sizeof(double) ||
        (nx + 31) / 32 > static_cast<size_t>(INT_MAX) || (ny + 3) / 4 > 65535) {
        if (!world_rank) fprintf(stderr, "Invalid or unsupported grid dimensions\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active_ranks = static_cast<int>(std::min<size_t>(nz, world_size));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_ranks) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    const size_t local_n = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t z_start = static_cast<size_t>(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    if (local_n > std::numeric_limits<size_t>::max() / plane - 2 ||
        (local_n + 2) * plane > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank, devices;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(local_n * plane);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_n * plane; ++i) {
        const size_t linear_id = z_start * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    const size_t bytes = (local_n + 2) * plane * sizeof(double);
    double *cold, *cnew, *mu;
    CUDA_CHECK(cudaMalloc(&cold, bytes)); CUDA_CHECK(cudaMalloc(&cnew, bytes)); CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    {
        HaloExchange halo(plane, local_n, rank, ranks, comm);
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            if (ranks == 1) {
                launch_stencil<true, true>(cold, nullptr, mu, nx, ny, 1, local_n, halo.compute);
                launch_stencil<false, true>(mu, cold, cnew, nx, ny, 1, local_n, halo.compute);
            } else {
                halo.apply<true>(cold, nullptr, mu, nx, ny);
                halo.apply<false>(mu, cold, cnew, nx, ny);
            }
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(halo.compute));
        const double local_time = MPI_Wtime() - start;
        double elapsed;
        MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
            printf("Computation time: %lld ms\n", milliseconds);
            const double mcups = static_cast<double>(gridSize) * iterations / elapsed / 1.0e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }
    }
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<double> result;
        if (!rank) { result.resize(gridSize); std::copy(host.begin(), host.end(), result.begin()); }
        // Receive slabs in rank order so the result matches global z-major order.
        for (int source = 1; source < ranks; ++source) {
            const size_t count = (nz / ranks + (static_cast<size_t>(source) < nz % ranks)) * plane;
            const size_t start = (static_cast<size_t>(source) * (nz / ranks) + std::min<size_t>(source, nz % ranks)) * plane;
            if (rank == source || rank == 0) {
                for (size_t offset = 0; offset < count;) {
                    int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
                    if (rank == source) MPI_Send(host.data() + offset, chunk, MPI_DOUBLE, 0, 20, comm);
                    else MPI_Recv(result.data() + start + offset, chunk, MPI_DOUBLE, source, 20, comm, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(chunk);
                }
            }
        }
        if (!rank) print_results(result, "Concentration");
    }
    int success = 1;
    if (validate) {
        if (!rank) printf("Validating result...\n");
        int bad = 0;
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();
        #pragma omp parallel for reduction(|:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
        for (size_t i = 0; i < host.size(); ++i) {
            const double value = host[i];
            if (!std::isfinite(value)) bad = 1;
            else { minVal = std::min(minVal, value); maxVal = std::max(maxVal, value); }
        }
        int global_bad;
        double global_min, global_max;
        MPI_Reduce(&bad, &global_bad, 1, MPI_INT, MPI_BOR, 0, comm);
        MPI_Reduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            if (global_bad) { printf("Validation failed: found NaN or Inf value\n"); success = 0; }
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 10.0 || global_min < -10.0) {
                    printf("Validation failed: values out of expected range\n"); success = 0;
                }
            }
            printf("Validation: %s\n", success ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&success, 1, MPI_INT, 0, comm);
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return success ? 0 : 1;
}
