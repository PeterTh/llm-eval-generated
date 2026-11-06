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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

#define MPI_CHECK(call) do { \
    int err_ = (call); \
    if (err_ != MPI_SUCCESS) { \
        char msg_[MPI_MAX_ERROR_STRING]; int len_; \
        MPI_Error_string(err_, msg_, &len_); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len_, msg_); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each rank owns a contiguous set of z planes, with one ghost plane on each side.
// x is the contiguous dimension, so neighboring CUDA threads access adjacent cells.
__global__ void chemical_potential(const double* __restrict__ c, double* __restrict__ mu,
                                   size_t nx, size_t ny, size_t nz, size_t first, size_t count, bool clamp_z) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t plane = nx * ny;
    for (size_t z = first + blockIdx.z; z < first + count; z += gridDim.z) {
        const size_t p = z * plane + y * nx + x;
        const double v = c[p];
        const double cxx = c[p + (x + 1 < nx ? 1 : 0)] + c[p - (x > 0 ? 1 : 0)] - 2.0 * v;
        const double cyy = c[p + (y + 1 < ny ? nx : 0)] + c[p - (y > 0 ? nx : 0)] - 2.0 * v;
        const double czz = c[p + ((clamp_z && z == nz) ? 0 : plane)]
                         + c[p - ((clamp_z && z == 1) ? 0 : plane)] - 2.0 * v;
        const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
        mu[p] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                + 3.0 * v + v * v * v - 0.5 * (cxx + cyy + czz);
    }
}

__global__ void update_concentration(double* __restrict__ next, const double* __restrict__ c,
                                     const double* __restrict__ mu, size_t nx, size_t ny,
                                     size_t nz, size_t first, size_t count, bool clamp_z) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t plane = nx * ny;
    for (size_t z = first + blockIdx.z; z < first + count; z += gridDim.z) {
        const size_t p = z * plane + y * nx + x;
        const double v = mu[p];
        const double cxx = mu[p + (x + 1 < nx ? 1 : 0)] + mu[p - (x > 0 ? 1 : 0)] - 2.0 * v;
        const double cyy = mu[p + (y + 1 < ny ? nx : 0)] + mu[p - (y > 0 ? nx : 0)] - 2.0 * v;
        const double czz = mu[p + ((clamp_z && z == nz) ? 0 : plane)]
                         + mu[p - ((clamp_z && z == 1) ? 0 : plane)] - 2.0 * v;
        next[p] = c[p] + 0.01 * (cxx + cyy + czz);
    }
}

struct Solver {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready;
    double *send_lo, *send_hi, *recv_lo, *recv_hi;

    Solver(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x * y), rank(r), ranks(n), comm(c) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        if (ranks > 1) {
            CUDA_CHECK(cudaMallocHost(&send_lo, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&send_hi, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_lo, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_hi, plane * sizeof(double)));
        }
    }
    ~Solver() {
        if (ranks > 1) {
            CUDA_CHECK(cudaFreeHost(send_lo)); CUDA_CHECK(cudaFreeHost(send_hi));
            CUDA_CHECK(cudaFreeHost(recv_lo)); CUDA_CHECK(cudaFreeHost(recv_hi));
        }
        CUDA_CHECK(cudaEventDestroy(ready));
        CUDA_CHECK(cudaStreamDestroy(transfer)); CUDA_CHECK(cudaStreamDestroy(compute));
    }

    void launch(bool chemical, double* out, const double* in, const double* mu,
                size_t first, size_t count) {
        if (!count) return;
        dim3 block(32, 4);
        dim3 grid(static_cast<unsigned>((nx + 31) / 32), static_cast<unsigned>((ny + 3) / 4),
                  static_cast<unsigned>(std::min<size_t>(count, 65535)));
        if (chemical) chemical_potential<<<grid, block, 0, compute>>>(in, out, nx, ny, nz, first, count, ranks == 1);
        else update_concentration<<<grid, block, 0, compute>>>(out, in, mu, nx, ny, nz, first, count, ranks == 1);
        CUDA_CHECK(cudaGetLastError());
    }

    // Exchange the input field's boundary planes while the GPU processes the interior.
    void stage(bool chemical, double* out, double* field, const double* c) {
        if (ranks == 1) {
            launch(chemical, out, c, field, 1, nz);
            return;
        }
        const int low = rank ? rank - 1 : MPI_PROC_NULL;
        const int high = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        // The previous stage may still be copying these receive buffers to the GPU.
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        MPI_Request req[4]; int nr = 0;
        if (low != MPI_PROC_NULL) MPI_CHECK(MPI_Irecv(recv_lo, static_cast<int>(plane), MPI_DOUBLE, low, 1, comm, &req[nr++]));
        if (high != MPI_PROC_NULL) MPI_CHECK(MPI_Irecv(recv_hi, static_cast<int>(plane), MPI_DOUBLE, high, 0, comm, &req[nr++]));

        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        const size_t bytes = plane * sizeof(double);
        if (low != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(send_lo, field + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        else
            CUDA_CHECK(cudaMemcpyAsync(field, field + plane, bytes, cudaMemcpyDeviceToDevice, transfer));
        if (high != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(send_hi, field + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        else
            CUDA_CHECK(cudaMemcpyAsync(field + (nz + 1) * plane, field + nz * plane, bytes, cudaMemcpyDeviceToDevice, transfer));

        // The interior does not depend on incoming ghost planes.
        if (nz > 2) launch(chemical, out, c, field, 2, nz - 2);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        if (low != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(send_lo, static_cast<int>(plane), MPI_DOUBLE, low, 0, comm, &req[nr++]));
        if (high != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(send_hi, static_cast<int>(plane), MPI_DOUBLE, high, 1, comm, &req[nr++]));
        if (nr) MPI_CHECK(MPI_Waitall(nr, req, MPI_STATUSES_IGNORE));
        if (low != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field, recv_lo, bytes, cudaMemcpyHostToDevice, transfer));
        if (high != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field + (nz + 1) * plane, recv_hi, bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(ready, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready, 0));
        launch(chemical, out, c, field, 1, 1);
        if (nz > 1) launch(chemical, out, c, field, nz, 1);
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int world_rank, world_size;
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
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > INT_MAX || nx > static_cast<size_t>(INT_MAX) || ny > 262140 ||
        nx * ny > SIZE_MAX / sizeof(double) || nx * ny * nz > SIZE_MAX / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Invalid or unsupported grid dimensions\n");
        MPI_Finalize(); return 1;
    }

    // Excess ranks are excluded, so every active rank owns at least one plane.
    int active = static_cast<int>(std::min<size_t>(nz, world_size));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm));
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    const size_t local_nz = nz / active + (static_cast<size_t>(rank) < nz % active);
    const size_t z_start = static_cast<size_t>(rank) * (nz / active) + std::min<size_t>(rank, nz % active);
    const size_t plane = nx * ny, local_size = plane * local_nz, gridSize = plane * nz;
    if (local_nz > SIZE_MAX / plane - 2 || (local_nz + 2) * plane > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Rank %d: grid allocation size overflows\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(local_size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_size; ++i) {
        const size_t linear_id = z_start * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    size_t bytes = (local_nz + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, bytes)); CUDA_CHECK(cudaMalloc(&cnew, bytes)); CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), local_size * sizeof(double), cudaMemcpyHostToDevice));
    int status = 0;
    {
        Solver solver(nx, ny, local_nz, rank, active, comm);
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        MPI_CHECK(MPI_Barrier(comm));
        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < iterations; ++t) {
            solver.stage(true, mu, cold, cold);
            solver.stage(false, cnew, mu, cold);
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(solver.compute));
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count(), max_elapsed;
        MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!rank) {
            auto ms = static_cast<long>(max_elapsed * 1000.0);
            printf("Computation time: %ld ms\n", ms);
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
        }
        if (validate || printResults)
            CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            std::vector<double> all;
            if (!rank) {
                all.resize(gridSize);
                std::copy(host.begin(), host.end(), all.begin());
                for (int r = 1; r < active; ++r) {
                    size_t n = plane * (nz / active + (static_cast<size_t>(r) < nz % active));
                    size_t offset = plane * (static_cast<size_t>(r) * (nz / active) + std::min<size_t>(r, nz % active));
                    while (n) {
                        int count = static_cast<int>(std::min<size_t>(n, INT_MAX));
                        MPI_CHECK(MPI_Recv(all.data() + offset, count, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                        offset += count; n -= count;
                    }
                }
                print_results(all, "Concentration");
            } else {
                for (size_t offset = 0; offset < local_size;) {
                    int count = static_cast<int>(std::min<size_t>(local_size - offset, INT_MAX));
                    MPI_CHECK(MPI_Send(host.data() + offset, count, MPI_DOUBLE, 0, 2, comm));
                    offset += count;
                }
            }
        }
        if (validate) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            int invalid = 0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < local_size; ++i) {
                double v = host[i];
                if (!std::isfinite(v)) invalid = 1;
                else { lo = std::min(lo, v); hi = std::max(hi, v); }
            }
            double global_lo, global_hi; int global_invalid;
            MPI_CHECK(MPI_Reduce(&lo, &global_lo, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
            MPI_CHECK(MPI_Reduce(&hi, &global_hi, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            MPI_CHECK(MPI_Reduce(&invalid, &global_invalid, 1, MPI_INT, MPI_BOR, 0, comm));
            if (!rank) {
                printf("Validating result...\n");
                if (global_invalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", global_lo, global_hi);
                    if (global_hi > 10.0 || global_lo < -10.0) printf("Validation failed: values out of expected range\n");
                }
                status = global_invalid || global_hi > 10.0 || global_lo < -10.0;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
            MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, comm));
        }
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return status;
}
