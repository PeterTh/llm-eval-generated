#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        int rank_ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each slab has one ghost plane at either end. Physical boundaries are clamped.
__global__ void stencil(const double* __restrict__ in, const double* __restrict__ cold,
                        double* __restrict__ out, size_t nx, size_t ny, size_t local_z,
                        size_t first_z, size_t count_z, bool has_lower, bool has_upper,
                        bool chemical) {
    const size_t plane = nx * ny;
    const size_t count = plane * count_z;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t z = first_z + i / plane;
        const size_t rem = i % plane;
        const size_t y = rem / nx;
        const size_t x = rem % nx;
        const size_t p = z * plane + rem;
        const double v = in[p];
        const size_t xp = p + (x + 1 < nx ? 1 : 0);
        const size_t xm = p - (x > 0 ? 1 : 0);
        const size_t yp = p + (y + 1 < ny ? nx : 0);
        const size_t ym = p - (y > 0 ? nx : 0);
        const size_t zp = p + ((z < local_z || has_upper) ? plane : 0);
        const size_t zm = p - ((z > 1 || has_lower) ? plane : 0);
        const double cxx = (in[xp] + in[xm] - 2.0 * v);
        const double cyy = (in[yp] + in[ym] - 2.0 * v);
        const double czz = (in[zp] + in[zm] - 2.0 * v);
        const double lap = cxx + cyy + czz;
        if (chemical) {
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            out[p] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                   + 3.0 * v + v * v * v - 0.5 * lap;
        } else {
            out[p] = cold[p] + 0.01 * lap;
        }
    }
}

static void launch_stencil(const double* in, const double* cold, double* out,
                           size_t nx, size_t ny, size_t local_z, size_t first_z,
                           size_t count_z, bool has_lower, bool has_upper, bool chemical,
                           cudaStream_t stream) {
    if (!count_z) return;
    const size_t n = nx * ny * count_z;
    const int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((n + threads - 1) / threads, 65535));
    stencil<<<blocks, threads, 0, stream>>>(in, cold, out, nx, ny, local_z, first_z,
                                           count_z, has_lower, has_upper, chemical);
    CUDA_CHECK(cudaGetLastError());
}

// Host staging makes communication work with MPI implementations that are not CUDA-aware.
// Each chunk uses the same tag: MPI preserves message order between each rank pair.
static void post_chunks(std::vector<MPI_Request>& requests, double* buffer, size_t count,
                        int peer, int tag, bool send, MPI_Comm comm) {
    while (count) {
        int n = static_cast<int>(std::min<size_t>(count, INT_MAX));
        MPI_Request request;
        if (send) MPI_Isend(buffer, n, MPI_DOUBLE, peer, tag, comm, &request);
        else MPI_Irecv(buffer, n, MPI_DOUBLE, peer, tag, comm, &request);
        requests.push_back(request);
        buffer += n;
        count -= n;
    }
}

static void run_stage(double* in, const double* cold, double* out, size_t nx, size_t ny,
                      size_t local_z, int rank, int ranks, bool chemical,
                      double* staging, cudaStream_t compute, cudaStream_t transfer,
                      MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t bytes = plane * sizeof(double);
    const bool lower = rank > 0, upper = rank + 1 < ranks;
    if (ranks == 1) {
        launch_stencil(in, cold, out, nx, ny, local_z, 1, local_z, false, false, chemical, compute);
        return;
    }

    // The previous stage must be complete before its boundary can be copied.
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double* send_lower = staging;
    double* send_upper = staging + plane;
    double* recv_lower = staging + 2 * plane;
    double* recv_upper = staging + 3 * plane;
    if (lower) CUDA_CHECK(cudaMemcpyAsync(send_lower, in + plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (upper) CUDA_CHECK(cudaMemcpyAsync(send_upper, in + local_z * plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (local_z > 2)
        launch_stencil(in, cold, out, nx, ny, local_z, 2, local_z - 2, lower, upper, chemical, compute);
    CUDA_CHECK(cudaStreamSynchronize(transfer));

    std::vector<MPI_Request> requests;
    if (lower) post_chunks(requests, recv_lower, plane, rank - 1, 1, false, comm);
    if (upper) post_chunks(requests, recv_upper, plane, rank + 1, 0, false, comm);
    if (lower) post_chunks(requests, send_lower, plane, rank - 1, 0, true, comm);
    if (upper) post_chunks(requests, send_upper, plane, rank + 1, 1, true, comm);
    if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    if (lower) CUDA_CHECK(cudaMemcpyAsync(in, recv_lower, bytes, cudaMemcpyHostToDevice, transfer));
    if (upper) CUDA_CHECK(cudaMemcpyAsync(in + (local_z + 1) * plane, recv_upper, bytes, cudaMemcpyHostToDevice, transfer));
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    launch_stencil(in, cold, out, nx, ny, local_z, 1, 1, lower, upper, chemical, compute);
    if (local_z > 1)
        launch_stencil(in, cold, out, nx, ny, local_z, local_z, 1, lower, upper, chemical, compute);
}

static void printUsage(const char* progName) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
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
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t max = std::numeric_limits<size_t>::max();
    if (!nx || !ny || !nz || nx > max / ny || nx * ny > max / nz ||
        nx * ny > max / sizeof(double) / 4 || nz > max / sizeof(double) / (nx * ny) - 2) {
        if (!world_rank) fprintf(stderr, "Invalid or oversized grid\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active_size = static_cast<int>(std::min<size_t>(world_size, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_size) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    const size_t local_z = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t first_z = static_cast<size_t>(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    const size_t local_count = plane * local_z;
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { fprintf(stderr, "Rank %d: no CUDA devices available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    std::vector<double> host(local_count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_count; ++i) {
        const size_t linear_id = first_z * plane + i;
        const double pseudo = (((linear_id + 1) * static_cast<size_t>(1299709)) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu, *staging = nullptr;
    const size_t allocation_bytes = (local_z + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, allocation_bytes));
    CUDA_CHECK(cudaMalloc(&cnew, allocation_bytes));
    CUDA_CHECK(cudaMalloc(&mu, allocation_bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), local_count * sizeof(double), cudaMemcpyHostToDevice));
    if (ranks > 1) CUDA_CHECK(cudaMallocHost(&staging, 4 * plane * sizeof(double)));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        run_stage(cold, cold, mu, nx, ny, local_z, rank, ranks, true, staging, compute, transfer, comm);
        run_stage(mu, cold, cnew, nx, ny, local_z, rank, ranks, false, staging, compute, transfer, comm);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
    }
    if (printResults || validate)
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_count * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<double> global;
        if (!rank) {
            global.resize(gridSize);
            std::copy(host.begin(), host.end(), global.begin());
            size_t offset = local_count;
            for (int r = 1; r < ranks; ++r) {
                size_t remaining = plane * (nz / ranks + (static_cast<size_t>(r) < nz % ranks));
                while (remaining) {
                    int n = static_cast<int>(std::min<size_t>(remaining, INT_MAX));
                    MPI_Recv(global.data() + offset, n, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    remaining -= n; offset += n;
                }
            }
            print_results(global, "Concentration");
        } else {
            size_t offset = 0;
            while (offset < local_count) {
                int n = static_cast<int>(std::min<size_t>(local_count - offset, INT_MAX));
                MPI_Send(host.data() + offset, n, MPI_DOUBLE, 0, 2, comm);
                offset += n;
            }
        }
    }
    int valid = 1;
    if (validate) {
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        int bad = 0;
        #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(|:bad) schedule(static)
        for (size_t i = 0; i < local_count; ++i) {
            const double v = host[i];
            if (!std::isfinite(v)) bad = 1;
            else { local_min = std::min(local_min, v); local_max = std::max(local_max, v); }
        }
        double global_min, global_max;
        int global_bad;
        MPI_Allreduce(&bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        valid = !global_bad && global_min >= -10.0 && global_max <= 10.0;
        if (!rank) {
            printf("Validating result...\n");
            if (global_bad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(transfer));
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
