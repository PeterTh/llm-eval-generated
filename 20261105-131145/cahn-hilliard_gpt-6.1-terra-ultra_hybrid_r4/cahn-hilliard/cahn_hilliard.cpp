#include <algorithm>
#include <chrono>
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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each rank owns a contiguous set of z planes and one ghost plane on either
// side. x is contiguous, so each exchanged halo is one contiguous plane.
__device__ inline double laplacian(const double* a, size_t x, size_t y, size_t z,
                                   size_t nx, size_t ny, size_t local_nz,
                                   bool has_lower, bool has_upper) {
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t zm = (z > 1 || has_lower) ? i - plane : i;
    const size_t zp = (z < local_nz || has_upper) ? i + plane : i;
    const double cxx = (a[xp] + a[xm] - 2.0 * a[i]) / (1.0 * 1.0);
    const double cyy = (a[yp] + a[ym] - 2.0 * a[i]) / (1.0 * 1.0);
    const double czz = (a[zp] + a[zm] - 2.0 * a[i]) / (1.0 * 1.0);
    return cxx + cyy + czz;
}

template<bool chemical>
__global__ void step_kernel(const double* input, const double* cold, double* output,
                            size_t nx, size_t ny, size_t local_nz,
                            size_t first_z, size_t num_z, bool has_lower, bool has_upper) {
    const size_t plane = nx * ny;
    for (size_t z = first_z + blockIdx.z; z < first_z + num_z; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(blockDim.y) * gridDim.y) {
            for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 x < nx; x += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const size_t i = z * plane + y * nx + x;
                const double lap = laplacian(input, x, y, z, nx, ny, local_nz, has_lower, has_upper);
                if (chemical) {
                    const double cv = input[i];
                    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                } else {
                    output[i] = cold[i] + 0.01 * 1.0 * lap;
                }
            }
        }
    }
}

template<bool chemical>
void launch(const double* input, const double* cold, double* output,
            size_t nx, size_t ny, size_t local_nz, size_t first_z, size_t num_z,
            bool has_lower, bool has_upper, cudaStream_t stream) {
    if (!num_z) return;
    const dim3 threads(32, 4);
    const dim3 blocks(static_cast<unsigned>(std::min<size_t>((nx - 1) / 32 + 1, 65535)),
                      static_cast<unsigned>(std::min<size_t>((ny - 1) / 4 + 1, 65535)),
                      static_cast<unsigned>(std::min<size_t>(num_z, 65535)));
    step_kernel<chemical><<<blocks, threads, 0, stream>>>(input, cold, output, nx, ny, local_nz,
                                                         first_z, num_z, has_lower, has_upper);
    CUDA_CHECK(cudaGetLastError());
}

struct HaloBuffers {
    double *send_lower, *send_upper, *recv_lower, *recv_upper;
};

// Stage halos through pinned host memory so this also works with MPI libraries
// that are not CUDA-aware. Computation on interior planes overlaps the exchange.
template<bool chemical>
void distributed_step(double* input, const double* cold, double* output,
                      size_t nx, size_t ny, size_t local_nz, int rank, int ranks,
                      MPI_Comm comm, HaloBuffers& h, cudaStream_t compute, cudaStream_t transfer) {
    const size_t plane = nx * ny;
    const size_t bytes = plane * sizeof(double);
    const bool lower = rank > 0, upper = rank + 1 < ranks;
    if (ranks == 1) {
        launch<chemical>(input, cold, output, nx, ny, local_nz, 1, local_nz, false, false, compute);
        return;
    }

    if (lower) CUDA_CHECK(cudaMemcpyAsync(h.send_lower, input + plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (upper) CUDA_CHECK(cudaMemcpyAsync(h.send_upper, input + local_nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (local_nz > 2)
        launch<chemical>(input, cold, output, nx, ny, local_nz, 2, local_nz - 2, lower, upper, compute);
    CUDA_CHECK(cudaStreamSynchronize(transfer));

    MPI_Request requests[4];
    int nreq = 0;
    if (lower) {
        MPI_Irecv(h.recv_lower, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[nreq++]);
        MPI_Isend(h.send_lower, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[nreq++]);
    }
    if (upper) {
        MPI_Irecv(h.recv_upper, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[nreq++]);
        MPI_Isend(h.send_upper, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[nreq++]);
    }
    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    if (lower) CUDA_CHECK(cudaMemcpyAsync(input, h.recv_lower, bytes, cudaMemcpyHostToDevice, transfer));
    if (upper) CUDA_CHECK(cudaMemcpyAsync(input + (local_nz + 1) * plane, h.recv_upper, bytes, cudaMemcpyHostToDevice, transfer));
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    launch<chemical>(input, cold, output, nx, ny, local_nz, 1, 1, lower, upper, compute);
    if (local_nz > 1)
        launch<chemical>(input, cold, output, nx, ny, local_nz, local_nz, 1, lower, upper, compute);
    // The next exchange sends the output produced by this step.
    CUDA_CHECK(cudaStreamSynchronize(compute));
}

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
    int provided = 0;
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
            MPI_Finalize();
            return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > static_cast<size_t>(INT_MAX) || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Invalid grid size or halo exceeds MPI count limit\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active_ranks = static_cast<int>(std::min<size_t>(world_size, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_ranks) {
        MPI_Finalize();
        return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, device_count;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    const size_t local_nz = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t z_start = static_cast<size_t>(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    const size_t local_count = local_nz * plane;
    if (local_nz > SIZE_MAX - 2 || local_nz + 2 > SIZE_MAX / sizeof(double) / plane) {
        fprintf(stderr, "Rank %d: local grid allocation is too large\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(local_count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_count; ++i) {
        const size_t linear_id = z_start * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }

    double *cold, *cnew, *mu;
    const size_t device_bytes = (local_nz + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, device_bytes));
    CUDA_CHECK(cudaMalloc(&cnew, device_bytes));
    CUDA_CHECK(cudaMalloc(&mu, device_bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), local_count * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    HaloBuffers h{};
    if (ranks > 1) {
        CUDA_CHECK(cudaMallocHost(&h.send_lower, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h.send_upper, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h.recv_lower, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h.recv_upper, plane * sizeof(double)));
    }
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        distributed_step<true>(cold, cold, mu, nx, ny, local_nz, rank, ranks, comm, h, compute, transfer);
        distributed_step<false>(mu, cold, cnew, nx, ny, local_nz, rank, ranks, comm, h, compute, transfer);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
    }
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_count * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        // MPI point-to-point chunks avoid the int displacement limit of MPI_Gatherv.
        std::vector<double> all;
        if (!rank) {
            all.resize(gridSize);
            std::copy(host.begin(), host.end(), all.begin());
            size_t offset = local_count;
            for (int r = 1; r < ranks; ++r) {
                size_t count = (nz / ranks + (static_cast<size_t>(r) < nz % ranks)) * plane;
                while (count) {
                    const int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
                    MPI_Recv(all.data() + offset, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    offset += chunk;
                    count -= chunk;
                }
            }
            print_results(all, "Concentration");
        } else {
            size_t offset = 0;
            while (offset < local_count) {
                const int chunk = static_cast<int>(std::min<size_t>(local_count - offset, INT_MAX));
                MPI_Send(host.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm);
                offset += chunk;
            }
        }
    }
    int valid = 1;
    if (validate) {
        int bad = 0;
        double minval = std::numeric_limits<double>::infinity();
        double maxval = -std::numeric_limits<double>::infinity();
        #pragma omp parallel for reduction(|:bad) reduction(min:minval) reduction(max:maxval) schedule(static)
        for (size_t i = 0; i < local_count; ++i) {
            const double value = host[i];
            if (!std::isfinite(value)) bad = 1;
            minval = std::min(minval, value);
            maxval = std::max(maxval, value);
        }
        int global_bad;
        double global_min, global_max;
        MPI_Allreduce(&bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&minval, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&maxval, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
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
    if (ranks > 1) {
        CUDA_CHECK(cudaFreeHost(h.send_lower));
        CUDA_CHECK(cudaFreeHost(h.send_upper));
        CUDA_CHECK(cudaFreeHost(h.recv_lower));
        CUDA_CHECK(cudaFreeHost(h.recv_upper));
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
