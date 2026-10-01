#include <algorithm>
#include <chrono>
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

#define CUDA_CHECK(call) do { cudaError_t error_ = (call); if (error_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int error_ = (call); if (error_ != MPI_SUCCESS) { \
    char message_[MPI_MAX_ERROR_STRING]; int length_; MPI_Error_string(error_, message_, &length_); \
    fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length_, message_); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Arrays have one ghost plane at either end. The kernel processes only owned cells.
__device__ __forceinline__ double laplacian(const double* __restrict__ a,
                                              size_t p, size_t x, size_t y,
                                              size_t nx, size_t ny, size_t plane) {
    const size_t xp = x + 1 < nx ? p + 1 : p;
    const size_t xm = x ? p - 1 : p;
    const size_t yp = y + 1 < ny ? p + nx : p;
    const size_t ym = y ? p - nx : p;
    const double center = a[p];
    const double xx = (a[xp] + a[xm] - 2.0 * center) / 1.0;
    const double yy = (a[yp] + a[ym] - 2.0 * center) / 1.0;
    const double zz = (a[p + plane] + a[p - plane] - 2.0 * center) / 1.0;
    return xx + yy + zz;
}

__global__ void chemical_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                 size_t nx, size_t ny, size_t plane, size_t offset, size_t count) {
    for (size_t i = offset + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < offset + count; i += (size_t)blockDim.x * gridDim.x) {
        const size_t p = i + plane;
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        const double cv = c[p];
        const double e_aa = -(2.0 / 9.0);
        const double e_bb = -(2.0 / 9.0);
        const double e_ab = 2.0 / 9.0;
        mu[p] = 4.5 * ((cv + 1.0) * e_aa + (cv - 1.0) * e_bb - 2.0 * cv * e_ab)
                + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c, p, x, y, nx, ny, plane);
    }
}

__global__ void update_kernel(const double* __restrict__ c, const double* __restrict__ mu,
                               double* __restrict__ next, size_t nx, size_t ny,
                               size_t plane, size_t offset, size_t count) {
    for (size_t i = offset + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < offset + count; i += (size_t)blockDim.x * gridDim.x) {
        const size_t p = i + plane;
        next[p] = c[p] + 0.01 * 1.0 * laplacian(mu, p, i % nx, (i / nx) % ny, nx, ny, plane);
    }
}

struct HaloBuffers {
    double *send_low, *send_high, *recv_low, *recv_high;
    explicit HaloBuffers(size_t bytes) {
        CUDA_CHECK(cudaMallocHost(&send_low, bytes));
        CUDA_CHECK(cudaMallocHost(&send_high, bytes));
        CUDA_CHECK(cudaMallocHost(&recv_low, bytes));
        CUDA_CHECK(cudaMallocHost(&recv_high, bytes));
    }
    ~HaloBuffers() {
        cudaFreeHost(send_low); cudaFreeHost(send_high);
        cudaFreeHost(recv_low); cudaFreeHost(recv_high);
    }
};

static void exchange_halos(double* field, size_t depth, size_t plane, int rank, int ranks,
                           HaloBuffers& buffers, cudaStream_t stream) {
    const size_t bytes = plane * sizeof(double);
    const int low = rank ? rank - 1 : MPI_PROC_NULL;
    const int high = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    if (low != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(buffers.send_low, field + plane, bytes, cudaMemcpyDeviceToHost, stream));
    else
        CUDA_CHECK(cudaMemcpyAsync(field, field + plane, bytes, cudaMemcpyDeviceToDevice, stream));
    if (high != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(buffers.send_high, field + depth * plane, bytes, cudaMemcpyDeviceToHost, stream));
    else
        CUDA_CHECK(cudaMemcpyAsync(field + (depth + 1) * plane, field + depth * plane,
                                   bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Request requests[4];
    int n = 0;
    if (low != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(buffers.recv_low, (int)plane, MPI_DOUBLE, low, 1, MPI_COMM_WORLD, &requests[n++]));
        MPI_CHECK(MPI_Isend(buffers.send_low, (int)plane, MPI_DOUBLE, low, 2, MPI_COMM_WORLD, &requests[n++]));
    }
    if (high != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Irecv(buffers.recv_high, (int)plane, MPI_DOUBLE, high, 2, MPI_COMM_WORLD, &requests[n++]));
        MPI_CHECK(MPI_Isend(buffers.send_high, (int)plane, MPI_DOUBLE, high, 1, MPI_COMM_WORLD, &requests[n++]));
    }
    if (n) MPI_CHECK(MPI_Waitall(n, requests, MPI_STATUSES_IGNORE));
    if (low != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(field, buffers.recv_low, bytes, cudaMemcpyHostToDevice, stream));
    if (high != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(field + (depth + 1) * plane, buffers.recv_high,
                                   bytes, cudaMemcpyHostToDevice, stream));
}

static void print_usage(const char* name) {
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
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_CHECK(MPI_Finalize()); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); print_usage(argv[0]); }
            MPI_CHECK(MPI_Finalize()); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || (size_t)ranks > nz ||
        nx > SIZE_MAX / ny || nx * ny > (size_t)std::numeric_limits<int>::max() ||
        nz > SIZE_MAX / (nx * ny) || nx * ny * nz > (size_t)std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid grid, step count, or MPI rank count\n");
        MPI_CHECK(MPI_Finalize()); return 1;
    }
    const size_t plane = nx * ny;
    const size_t total = plane * nz;
    const size_t base = nz / ranks;
    const size_t extra = nz % ranks;
    const size_t depth = base + ((size_t)rank < extra);
    const size_t start_z = (size_t)rank * base + std::min((size_t)rank, extra);
    const size_t cells = depth * plane;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    MPI_Comm shared;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(shared, &local_rank));
    int gpu_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
    if (!gpu_count) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % gpu_count));
    MPI_CHECK(MPI_Comm_free(&shared));

    std::vector<double> initial(cells);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)cells; ++i) {
        const size_t global_id = start_z * plane + (size_t)i;
        const double pseudo = (((global_id + 1) * 1299709) % total) / (double)total;
        initial[(size_t)i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    const size_t allocation = (depth + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, allocation));
    CUDA_CHECK(cudaMalloc(&cnew, allocation));
    CUDA_CHECK(cudaMalloc(&mu, allocation));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(cold + plane, initial.data(), cells * sizeof(double), cudaMemcpyHostToDevice, stream));
    HaloBuffers halos(plane * sizeof(double));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    initial.clear(); initial.shrink_to_fit();

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    auto launch_chemical = [&](size_t offset, size_t count) {
        if (!count) return;
        const int blocks = (int)std::min<size_t>((count + 255) / 256, 65535);
        chemical_kernel<<<blocks, 256, 0, compute_stream>>>(cold, mu, nx, ny, plane, offset, count);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launch_update = [&](size_t offset, size_t count) {
        if (!count) return;
        const int blocks = (int)std::min<size_t>((count + 255) / 256, 65535);
        update_kernel<<<blocks, 256, 0, compute_stream>>>(cold, mu, cnew, nx, ny, plane, offset, count);
        CUDA_CHECK(cudaGetLastError());
    };
    for (int t = 0; t < iterations; ++t) {
        // Previous update must finish before boundary copies of its output.
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        if (ranks == 1) {
            exchange_halos(cold, depth, plane, rank, ranks, halos, stream);
            launch_chemical(0, cells);
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            exchange_halos(mu, depth, plane, rank, ranks, halos, stream);
            launch_update(0, cells);
        } else {
            // Interior planes do not depend on the incoming ghost planes.
            if (depth > 2) launch_chemical(plane, (depth - 2) * plane);
            exchange_halos(cold, depth, plane, rank, ranks, halos, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            launch_chemical(0, plane);
            if (depth > 1) launch_chemical((depth - 1) * plane, plane);
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));

            if (depth > 2) launch_update(plane, (depth - 2) * plane);
            exchange_halos(mu, depth, plane, rank, ranks, halos, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            launch_update(0, plane);
            if (depth > 1) launch_update((depth - 1) * plane, plane);
        }
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_CHECK(MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)(seconds * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? (double)total * iterations / seconds / 1e6 : 0.0);
    }

    int result = 0;
    if (validate || print_results_flag) {
        std::vector<double> local(cells);
        CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, cells * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> counts, offsets;
        std::vector<double> all;
        if (rank == 0) {
            counts.resize(ranks); offsets.resize(ranks); all.resize(total);
            for (int r = 0; r < ranks; ++r) {
                const size_t d = base + ((size_t)r < extra);
                const size_t z = (size_t)r * base + std::min((size_t)r, extra);
                counts[r] = (int)(d * plane); offsets[r] = (int)(z * plane);
            }
        }
        MPI_CHECK(MPI_Gatherv(local.data(), (int)cells, MPI_DOUBLE, rank == 0 ? all.data() : nullptr,
                              rank == 0 ? counts.data() : nullptr, rank == 0 ? offsets.data() : nullptr,
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            if (print_results_flag) print_results(all, "Concentration");
            if (validate) {
                printf("Validating result...\n");
                int bad = 0;
                double min_value = std::numeric_limits<double>::infinity();
                double max_value = -std::numeric_limits<double>::infinity();
                #pragma omp parallel for reduction(+:bad) reduction(min:min_value) reduction(max:max_value)
                for (long long i = 0; i < (long long)total; ++i) {
                    const double value = all[(size_t)i];
                    bad += !std::isfinite(value);
                    if (std::isfinite(value)) {
                        min_value = std::min(min_value, value);
                        max_value = std::max(max_value, value);
                    }
                }
                if (bad) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
                    if (max_value > 10.0 || min_value < -10.0)
                        printf("Validation failed: values out of expected range\n");
                }
                result = bad || max_value > 10.0 || min_value < -10.0;
                printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Finalize());
    return result;
}
