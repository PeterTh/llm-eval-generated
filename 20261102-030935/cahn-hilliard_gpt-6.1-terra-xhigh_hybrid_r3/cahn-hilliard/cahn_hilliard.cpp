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

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        int rank_ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_, __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

#define MPI_CHECK(call) do { \
    int error_ = (call); \
    if (error_ != MPI_SUCCESS) { \
        char message_[MPI_MAX_ERROR_STRING]; int length_ = 0; \
        MPI_Error_string(error_, message_, &length_); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length_, message_); \
        MPI_Abort(MPI_COMM_WORLD, error_); \
    } \
} while (0)

// Local planes are numbered 1..local_z, with one halo plane on either side.
__device__ inline double laplacian(const double* a, size_t i, size_t x, size_t y,
                                   size_t z, size_t nx, size_t ny, size_t local_z,
                                   bool lower, bool upper) {
    const size_t plane = nx * ny;
    const double center = a[i];
    const double cxx = a[i + (x + 1 < nx ? 1 : 0)] + a[i - (x > 0 ? 1 : 0)] - 2.0 * center;
    const double cyy = a[i + (y + 1 < ny ? nx : 0)] + a[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = a[i + (z < local_z || upper ? plane : 0)] + a[i - (z > 1 || lower ? plane : 0)] - 2.0 * center;
    return cxx + cyy + czz;
}

template <bool Chemical>
__global__ void stencil(const double* __restrict__ in, const double* __restrict__ old,
                        double* __restrict__ out, size_t nx, size_t ny, size_t local_z,
                        size_t first_z, size_t count_z, bool lower, bool upper) {
    const size_t plane = nx * ny;
    const size_t count = plane * count_z;
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t z = first_z + j / plane;
        const size_t p = j % plane;
        const size_t y = p / nx;
        const size_t x = p - y * nx;
        const size_t i = z * plane + p;
        const double lap = laplacian(in, i, x, y, z, nx, ny, local_z, lower, upper);
        if (Chemical) {
            const double cv = in[i];
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            out[i] = old[i] + 0.01 * lap;
        }
    }
}

struct Slab {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, exchange;
    double *send_lo, *send_hi, *recv_lo, *recv_hi;

    Slab(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x * y), rank(r), ranks(n), comm(c),
          send_lo(nullptr), send_hi(nullptr), recv_lo(nullptr), recv_hi(nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        if (ranks > 1) {
            CUDA_CHECK(cudaMallocHost(&send_lo, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&send_hi, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_lo, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_hi, plane * sizeof(double)));
        }
    }
    ~Slab() {
        if (ranks > 1) {
            CUDA_CHECK(cudaFreeHost(send_lo)); CUDA_CHECK(cudaFreeHost(send_hi));
            CUDA_CHECK(cudaFreeHost(recv_lo)); CUDA_CHECK(cudaFreeHost(recv_hi));
        }
        CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(exchange));
    }

    template <bool Chemical>
    void launch(const double* in, const double* old, double* out, size_t first, size_t count, cudaStream_t stream) {
        if (!count) return;
        const size_t blocks = std::min<size_t>((plane * count + 255) / 256, 65535);
        stencil<Chemical><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
            in, old, out, nx, ny, nz, first, count, rank > 0, rank + 1 < ranks);
        CUDA_CHECK(cudaGetLastError());
    }

    template <bool Chemical>
    void stage(double* in, const double* old, double* out) {
        if (ranks == 1) {
            launch<Chemical>(in, old, out, 1, nz, compute);
            return;
        }
        const size_t bytes = plane * sizeof(double);
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(send_lo, in + plane, bytes, cudaMemcpyDeviceToHost, exchange));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(send_hi, in + nz * plane, bytes, cudaMemcpyDeviceToHost, exchange));
        // The interior does not depend on incoming halos and runs while MPI progresses.
        if (nz > 2) launch<Chemical>(in, old, out, 2, nz - 2, compute);
        CUDA_CHECK(cudaStreamSynchronize(exchange));
        MPI_Request requests[4]; int nreq = 0;
        if (rank > 0) {
            MPI_CHECK(MPI_Irecv(recv_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[nreq++]));
            MPI_CHECK(MPI_Isend(send_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[nreq++]));
        }
        if (rank + 1 < ranks) {
            MPI_CHECK(MPI_Irecv(recv_hi, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[nreq++]));
            MPI_CHECK(MPI_Isend(send_hi, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[nreq++]));
        }
        MPI_CHECK(MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE));
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(in, recv_lo, bytes, cudaMemcpyHostToDevice, exchange));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(in + (nz + 1) * plane, recv_hi, bytes, cudaMemcpyHostToDevice, exchange));
        launch<Chemical>(in, old, out, 1, 1, exchange);
        if (nz > 1) launch<Chemical>(in, old, out, nz, 1, exchange);
        // Both streams must complete before the next stencil reads their output.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamSynchronize(exchange));
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int world_rank = 0, world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (!nx || nx > limit / ny || nx * ny > limit / nz || nx * ny > INT_MAX ||
        nx * ny * nz > limit / sizeof(double) || nz > limit - 2 ||
        nx * ny > limit / (nz + 2) / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Invalid grid size or MPI halo plane too large\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active = static_cast<int>(std::min<size_t>(nz, static_cast<size_t>(world_size)));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm));
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank = 0, ranks = 1;
    MPI_CHECK(MPI_Comm_rank(comm, &rank)); MPI_CHECK(MPI_Comm_size(comm, &ranks));
    const size_t local_z = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t z_offset = static_cast<size_t>(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    const size_t local_count = plane * local_z;

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> local(local_count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_count; ++i) {
        const size_t linear_id = z_offset * plane + i;
        const double pseudo = (((linear_id + 1) * static_cast<size_t>(1299709)) % gridSize) / static_cast<double>(gridSize);
        local[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    const size_t alloc_bytes = (local_z + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, alloc_bytes)); CUDA_CHECK(cudaMalloc(&cnew, alloc_bytes)); CUDA_CHECK(cudaMalloc(&mu, alloc_bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, local.data(), local_count * sizeof(double), cudaMemcpyHostToDevice));
    int result = 0;
    {
        Slab slab(nx, ny, local_z, rank, ranks, comm);
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        MPI_CHECK(MPI_Barrier(comm));
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            slab.stage<true>(cold, nullptr, mu);
            slab.stage<false>(mu, cold, cnew);
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(slab.compute));
        const double elapsed = MPI_Wtime() - start;
        double max_elapsed = 0.0;
        MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!rank) {
            const long milliseconds = static_cast<long>(max_elapsed * 1000.0);
            printf("Computation time: %ld ms\n", milliseconds);
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
        }
        if (printResults || validate) CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, local_count * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            // Gather in global plane order; chunk messages to stay within MPI's int count limit.
            if (!rank) {
                std::vector<double> all(gridSize);
                std::copy(local.begin(), local.end(), all.begin());
                size_t offset = local_count;
                for (int r = 1; r < ranks; ++r) {
                    size_t remaining = plane * (nz / ranks + (static_cast<size_t>(r) < nz % ranks));
                    while (remaining) {
                        int chunk = static_cast<int>(std::min<size_t>(remaining, INT_MAX));
                        MPI_CHECK(MPI_Recv(all.data() + offset, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                        offset += chunk; remaining -= chunk;
                    }
                }
                print_results(all, "Concentration");
            } else {
                for (size_t offset = 0; offset < local_count;) {
                    int chunk = static_cast<int>(std::min<size_t>(local_count - offset, INT_MAX));
                    MPI_CHECK(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm));
                    offset += chunk;
                }
            }
        }
        if (validate) {
            if (!rank) printf("Validating result...\n");
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            int invalid = 0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < local_count; ++i) {
                double value = local[i];
                if (!std::isfinite(value)) invalid = 1;
                else { lo = std::min(lo, value); hi = std::max(hi, value); }
            }
            double global_lo = 0, global_hi = 0; int global_invalid = 0;
            MPI_CHECK(MPI_Allreduce(&lo, &global_lo, 1, MPI_DOUBLE, MPI_MIN, comm));
            MPI_CHECK(MPI_Allreduce(&hi, &global_hi, 1, MPI_DOUBLE, MPI_MAX, comm));
            MPI_CHECK(MPI_Allreduce(&invalid, &global_invalid, 1, MPI_INT, MPI_MAX, comm));
            if (global_invalid || global_hi > 10.0 || global_lo < -10.0) result = 1;
            if (!rank) {
                if (global_invalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", global_lo, global_hi);
                    if (result) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return result;
}
