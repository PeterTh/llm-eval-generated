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

#define CUDA_CHECK(call) do { cudaError_t err_ = (call); if (err_ != cudaSuccess) { \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(err_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int err_ = (call); if (err_ != MPI_SUCCESS) { \
    char msg_[MPI_MAX_ERROR_STRING]; int len_; MPI_Error_string(err_, msg_, &len_); \
    fprintf(stderr, "Rank %d: MPI error: %.*s\n", rank, len_, msg_); MPI_Abort(MPI_COMM_WORLD, err_); } } while (0)

// The arrays contain one ghost plane at each end. Each warp covers a row,
// avoiding per-cell division while keeping X accesses contiguous.
template <bool Chemical>
__global__ void stencil(const double* __restrict__ input, const double* __restrict__ old,
                        double* __restrict__ output, size_t nx, size_t ny,
                        size_t local_z, size_t first_z, size_t global_z,
                        size_t begin_z, size_t end_z) {
    const size_t plane = nx * ny;
    for (size_t z = begin_z + blockIdx.z; z < end_z; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
          for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
               x < nx; x += static_cast<size_t>(gridDim.x) * blockDim.x) {
            const size_t i = z * plane + y * nx + x;
            const double v = input[i];
            const size_t xp = x + 1 < nx ? i + 1 : i;
            const size_t xn = x > 0 ? i - 1 : i;
            const size_t yp = y + 1 < ny ? i + nx : i;
            const size_t yn = y > 0 ? i - nx : i;
            const size_t zp = (z < local_z || first_z + local_z < global_z) ? i + plane : i;
            const size_t zn = (z > 1 || first_z > 0) ? i - plane : i;
            const double cxx = input[xp] + input[xn] - 2.0 * v;
            const double cyy = input[yp] + input[yn] - 2.0 * v;
            const double czz = input[zp] + input[zn] - 2.0 * v;
            const double lap = cxx + cyy + czz;
            if constexpr (Chemical) {
                const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
                output[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                            + 3.0 * v + v * v * v - 0.5 * lap;
            } else {
                output[i] = old[i] + 0.01 * lap;
            }
          }
        }
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of time steps (default: 20)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* value = argv[++i];
            long long n = strtoll(value, &end, 10);
            if (!*value || *end || n < 0 || (option == 'i' && n > INT_MAX)) { bad = true; break; }
            if (option == 'x') nx = static_cast<size_t>(n);
            else if (option == 'y') ny = static_cast<size_t>(n);
            else if (option == 'z') nz = static_cast<size_t>(n);
            else iterations = static_cast<int>(n);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { bad = true; break; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > INT_MAX ||
        nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(double)) bad = true;
    if (help || bad) {
        if (!rank) { if (bad) fprintf(stderr, "Invalid arguments or grid size\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active_count = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active_count ? 0 : MPI_UNDEFINED, rank, &comm));
    if (rank >= active_count) { MPI_Finalize(); return 0; }
    const size_t local_z = nz / active_count + (static_cast<size_t>(rank) < nz % active_count);
    const size_t first_z = static_cast<size_t>(rank) * (nz / active_count) + std::min(static_cast<size_t>(rank), nz % active_count);
    if (local_z > SIZE_MAX / plane - 2 || (local_z + 2) * plane > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Rank %d: local grid too large\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // A node-local rank chooses a GPU independently of the global MPI rank.
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, device_count = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(plane * local_z);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < host.size(); ++i) {
        const size_t linear_id = first_z * plane + i;
        const double pseudo = (((linear_id + 1) * size_t(1299709)) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    const size_t bytes = (local_z + 2) * plane * sizeof(double), plane_bytes = plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, bytes)); CUDA_CHECK(cudaMalloc(&cnew, bytes)); CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t compute, transfer;
    cudaEvent_t ready;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    double *send_low = nullptr, *send_high = nullptr, *recv_low = nullptr, *recv_high = nullptr;
    if (active_count > 1) {
        CUDA_CHECK(cudaMallocHost(&send_low, plane_bytes)); CUDA_CHECK(cudaMallocHost(&send_high, plane_bytes));
        CUDA_CHECK(cudaMallocHost(&recv_low, plane_bytes)); CUDA_CHECK(cudaMallocHost(&recv_high, plane_bytes));
    }
    const int low = rank ? rank - 1 : MPI_PROC_NULL, high = rank + 1 < active_count ? rank + 1 : MPI_PROC_NULL;
    const dim3 block(32, 4);
    auto launch = [&](bool chemical, const double* input, const double* old, double* output, size_t begin, size_t end) {
        if (begin >= end) return;
        dim3 grid(static_cast<unsigned>(std::min((nx + 31) / 32, size_t(65535))),
                  static_cast<unsigned>(std::min((ny + 3) / 4, size_t(65535))),
                  static_cast<unsigned>(std::min(end - begin, size_t(65535))));
        if (chemical) stencil<true><<<grid, block, 0, compute>>>(input, old, output, nx, ny, local_z, first_z, nz, begin, end);
        else stencil<false><<<grid, block, 0, compute>>>(input, old, output, nx, ny, local_z, first_z, nz, begin, end);
        CUDA_CHECK(cudaGetLastError());
    };
    // Start transfer after the producer, overlap the interior stencil with MPI,
    // then make only the boundary stencil wait for the incoming ghost planes.
    auto step = [&](bool chemical, double* input, const double* old, double* output) {
        if (active_count == 1) { launch(chemical, input, old, output, 1, local_z + 1); return; }
        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        if (low != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(send_low, input + plane, plane_bytes, cudaMemcpyDeviceToHost, transfer));
        if (high != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(send_high, input + local_z * plane, plane_bytes, cudaMemcpyDeviceToHost, transfer));
        if (local_z > 2) launch(chemical, input, old, output, 2, local_z);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        MPI_Request req[4]; int count = 0;
        if (low != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_low, static_cast<int>(plane), MPI_DOUBLE, low, 1, comm, &req[count++]));
            MPI_CHECK(MPI_Isend(send_low, static_cast<int>(plane), MPI_DOUBLE, low, 0, comm, &req[count++]));
        }
        if (high != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_high, static_cast<int>(plane), MPI_DOUBLE, high, 0, comm, &req[count++]));
            MPI_CHECK(MPI_Isend(send_high, static_cast<int>(plane), MPI_DOUBLE, high, 1, comm, &req[count++]));
        }
        MPI_CHECK(MPI_Waitall(count, req, MPI_STATUSES_IGNORE));
        if (low != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(input, recv_low, plane_bytes, cudaMemcpyHostToDevice, transfer));
        if (high != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(input + (local_z + 1) * plane, recv_high, plane_bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(ready, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, ready, 0));
        launch(chemical, input, old, output, 1, 2);
        if (local_z > 1) launch(chemical, input, old, output, local_z, local_z + 1);
    };
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        step(true, cold, nullptr, mu);
        step(false, mu, cold, cnew);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
    }
    if (printResults || validate) CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<double> result;
        if (!rank) {
            result.resize(gridSize);
            std::copy(host.begin(), host.end(), result.begin());
            for (int r = 1; r < active_count; ++r) {
                size_t zcount = nz / active_count + (static_cast<size_t>(r) < nz % active_count);
                size_t zstart = static_cast<size_t>(r) * (nz / active_count) + std::min(static_cast<size_t>(r), nz % active_count);
                size_t remaining = zcount * plane, offset = zstart * plane;
                while (remaining) {
                    int n = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_CHECK(MPI_Recv(result.data() + offset, n, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                    offset += n; remaining -= n;
                }
            }
            print_results(result, "Concentration");
        } else {
            size_t remaining = host.size(), offset = 0;
            while (remaining) {
                int n = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                MPI_CHECK(MPI_Send(host.data() + offset, n, MPI_DOUBLE, 0, 2, comm));
                offset += n; remaining -= n;
            }
        }
    }
    int failed = 0;
    if (validate) {
        double minval = std::numeric_limits<double>::infinity(), maxval = -minval;
        int nonfinite = 0;
        #pragma omp parallel for reduction(min:minval) reduction(max:maxval) reduction(|:nonfinite) schedule(static)
        for (size_t i = 0; i < host.size(); ++i) {
            const double v = host[i];
            nonfinite |= !std::isfinite(v);
            minval = std::min(minval, v); maxval = std::max(maxval, v);
        }
        double global_min, global_max; int global_nonfinite;
        MPI_CHECK(MPI_Allreduce(&minval, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm));
        MPI_CHECK(MPI_Allreduce(&maxval, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm));
        MPI_CHECK(MPI_Allreduce(&nonfinite, &global_nonfinite, 1, MPI_INT, MPI_BOR, comm));
        failed = global_nonfinite || global_min < -10.0 || global_max > 10.0;
        if (!rank) {
            printf("Validating result...\n");
            if (global_nonfinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (failed) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
    }
    if (active_count > 1) {
        CUDA_CHECK(cudaFreeHost(send_low)); CUDA_CHECK(cudaFreeHost(send_high));
        CUDA_CHECK(cudaFreeHost(recv_low)); CUDA_CHECK(cudaFreeHost(recv_high));
    }
    CUDA_CHECK(cudaEventDestroy(ready)); CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_Finalize();
    return failed;
}
