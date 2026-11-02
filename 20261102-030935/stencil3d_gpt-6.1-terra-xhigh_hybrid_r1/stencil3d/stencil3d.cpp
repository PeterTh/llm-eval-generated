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

using Real = double;

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// The two device grids are initialized identically, so the fixed global
// boundary never needs to be copied during the iterations.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t plane,
                        size_t z_first, size_t z_end, size_t global_z_first,
                        size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    for (size_t z = z_first + blockIdx.z; z < z_end; z += gridDim.z) {
        const size_t global_z = global_z_first + z - 1;
        if (global_z == 0 || global_z + 1 == nz) continue;
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
             y < ny - 1; y += gridDim.y * blockDim.y) {
            const size_t i = z * plane + y * nx + x;
            out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] +
                      in[i + nx] + in[i - plane] + in[i + plane]) / 7.0;
        }
    }
}

static void launch_stencil(const Real* in, Real* out, size_t nx, size_t ny,
                           size_t plane, size_t first, size_t end,
                           size_t global_first, size_t nz, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || nz < 3 || first >= end) return;
    const dim3 threads(32, 4);
    const dim3 blocks(static_cast<unsigned>((nx - 2 + threads.x - 1) / threads.x),
                      static_cast<unsigned>(std::min<size_t>((ny - 2 + threads.y - 1) / threads.y, 65535)),
                      static_cast<unsigned>(std::min<size_t>(end - first, 65535)));
    stencil<<<blocks, threads, 0, stream>>>(in, out, nx, ny, plane, first, end, global_first, nz);
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
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
        if (!world_rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
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
            MPI_Finalize();
            return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > INT_MAX ||
        nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        if (!world_rank) fprintf(stderr, "Invalid grid size or iteration count (a plane must fit in an MPI count)\n");
        MPI_Finalize();
        return 1;
    }

    // A rank with no Z planes does not participate in the stencil communicator.
    const int active_count = static_cast<int>(std::min<size_t>(nz, world_size));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_count ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_count) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local_comm);

    const size_t plane = nx * ny;
    const size_t local_nz = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t z_start = static_cast<size_t>(rank) * (nz / ranks) +
                           std::min<size_t>(rank, nz % ranks);
    const size_t local_size = plane * local_nz;
    if (local_nz > SIZE_MAX - 2 || plane > SIZE_MAX / sizeof(Real) / (local_nz + 2)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t device_size = plane * (local_nz + 2);
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> host(local_size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_size; ++i) host[i] = static_cast<Real>((z_start * plane + i) % 19);
    Real *in, *out;
    CUDA_CHECK(cudaMalloc(&in, device_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&out, device_size * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(in + plane, host.data(), local_size * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out + plane, host.data(), local_size * sizeof(Real), cudaMemcpyHostToDevice));
    const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    Real *send_lower = nullptr, *send_upper = nullptr, *recv_lower = nullptr, *recv_upper = nullptr;
    if (lower != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMallocHost(&send_lower, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recv_lower, plane * sizeof(Real)));
    }
    if (upper != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMallocHost(&send_upper, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recv_upper, plane * sizeof(Real)));
    }
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    MPI_Barrier(comm);
    if (!rank) printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launch_stencil(in, out, nx, ny, plane, 1, local_nz + 1, z_start, nz, compute);
        } else {
            // Planes that do not depend on incoming halos run during MPI exchange.
            const size_t first = lower == MPI_PROC_NULL ? 1 : 2;
            const size_t end = local_nz + 1 - (upper != MPI_PROC_NULL ? 1 : 0);
            launch_stencil(in, out, nx, ny, plane, first, end, z_start, nz, compute);
            MPI_Request req[4];
            int nreq = 0;
            if (lower != MPI_PROC_NULL) {
                MPI_Irecv(recv_lower, static_cast<int>(plane), MPI_DOUBLE, lower, 1, comm, &req[nreq++]);
                CUDA_CHECK(cudaMemcpyAsync(send_lower, in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            if (upper != MPI_PROC_NULL) {
                MPI_Irecv(recv_upper, static_cast<int>(plane), MPI_DOUBLE, upper, 0, comm, &req[nreq++]);
                CUDA_CHECK(cudaMemcpyAsync(send_upper, in + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            CUDA_CHECK(cudaStreamSynchronize(transfer));
            if (lower != MPI_PROC_NULL) MPI_Isend(send_lower, static_cast<int>(plane), MPI_DOUBLE, lower, 0, comm, &req[nreq++]);
            if (upper != MPI_PROC_NULL) MPI_Isend(send_upper, static_cast<int>(plane), MPI_DOUBLE, upper, 1, comm, &req[nreq++]);
            MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
            if (lower != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(in, recv_lower, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
            if (upper != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(in + (local_nz + 1) * plane, recv_upper, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
            if (lower != MPI_PROC_NULL) launch_stencil(in, out, nx, ny, plane, 1, 2, z_start, nz, transfer);
            if (upper != MPI_PROC_NULL && (local_nz != 1 || lower == MPI_PROC_NULL))
                launch_stencil(in, out, nx, ny, plane, local_nz, local_nz + 1, z_start, nz, transfer);
            CUDA_CHECK(cudaStreamSynchronize(transfer));
            CUDA_CHECK(cudaStreamSynchronize(compute));
        }
        std::swap(in, out);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / max_elapsed / 1e6);
    }
    if (validate || printResults) CUDA_CHECK(cudaMemcpy(host.data(), in + plane, local_size * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<Real> full;
        if (!rank) full.resize(plane * nz);
        // Chunked point-to-point gathering avoids MPI's int displacement limit.
        for (int r = 0; r < ranks; ++r) {
            const size_t count = plane * (nz / ranks + (static_cast<size_t>(r) < nz % ranks));
            const size_t offset = plane * (static_cast<size_t>(r) * (nz / ranks) + std::min<size_t>(r, nz % ranks));
            for (size_t pos = 0; pos < count; ) {
                const int chunk = static_cast<int>(std::min<size_t>(count - pos, INT_MAX));
                if (!rank && !r) std::copy_n(host.data() + pos, chunk, full.data() + offset + pos);
                else if (rank == r) MPI_Send(host.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                else if (!rank) MPI_Recv(full.data() + offset + pos, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                pos += chunk;
            }
        }
        if (!rank) print_results(full, "Grid");
    }
    int valid = 1;
    if (validate) {
        double min_value = std::numeric_limits<double>::infinity();
        double max_value = -std::numeric_limits<double>::infinity();
        int bad = 0;
        #pragma omp parallel for reduction(min:min_value) reduction(max:max_value) reduction(|:bad) schedule(static)
        for (size_t i = 0; i < local_size; ++i) {
            const double value = host[i];
            bad |= !std::isfinite(value);
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
        double global_min, global_max;
        int global_bad;
        MPI_Reduce(&min_value, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&max_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&bad, &global_bad, 1, MPI_INT, MPI_BOR, 0, comm);
        if (!rank) {
            printf("Validating result...\n");
            if (global_bad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 1e6 || global_min < -1e6) printf("Validation failed: values out of expected range\n");
            }
            valid = !global_bad && global_max <= 1e6 && global_min >= -1e6;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(transfer));
    if (send_lower) CUDA_CHECK(cudaFreeHost(send_lower));
    if (recv_lower) CUDA_CHECK(cudaFreeHost(recv_lower));
    if (send_upper) CUDA_CHECK(cudaFreeHost(send_upper));
    if (recv_upper) CUDA_CHECK(cudaFreeHost(recv_upper));
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
