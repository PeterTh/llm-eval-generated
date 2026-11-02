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
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int error = (call); \
    if (error != MPI_SUCCESS) { \
        char message[MPI_MAX_ERROR_STRING]; int length = 0; \
        MPI_Error_string(error, message, &length); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length, message); \
        MPI_Abort(MPI_COMM_WORLD, error); \
    } \
} while (0)

void printUsage(const char* name) {
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

// The local arrays contain one ghost plane on each side. Physical boundaries
// use the edge cell itself, as in the original clamped Laplacian.
template<bool Chemical, bool Boundary>
__global__ void stencil(const double* __restrict__ field, const double* __restrict__ cold,
                        double* __restrict__ out, size_t nx, size_t ny, size_t local_z,
                        bool low_edge, bool high_edge, size_t first_z, size_t planes) {
    const size_t plane = nx * ny;
    const size_t n = plane * planes;
    for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         q < n; q += size_t(blockDim.x) * gridDim.x) {
        const size_t p = q % plane;
        const size_t x = p % nx;
        const size_t y = p / nx;
        const size_t z = Boundary ? ((q / plane == 0) ? 1 : local_z) : first_z + q / plane;
        const size_t i = z * plane + p;
        const double v = field[i];
        const double xx = field[i + (x + 1 < nx)] + field[i - (x > 0)] - 2.0 * v;
        const double yy = field[i + (y + 1 < ny ? nx : 0)] + field[i - (y > 0 ? nx : 0)] - 2.0 * v;
        const double zz = field[i + ((z == local_z && high_edge) ? 0 : plane)]
                        + field[i - ((z == 1 && low_edge) ? 0 : plane)] - 2.0 * v;
        const double lap = xx + yy + zz;
        if (Chemical) {
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
            out[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else {
            out[i] = cold[i] + 0.01 * lap;
        }
    }
}

struct Slab {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready;
    double *send_low = nullptr, *send_high = nullptr, *recv_low = nullptr, *recv_high = nullptr;
    std::vector<MPI_Request> requests;

    Slab(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x*y), rank(r), ranks(n), comm(c) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        if (ranks > 1) {
            requests.resize(4 * ((plane - 1) / INT_MAX + 1));
            CUDA_CHECK(cudaMallocHost(&send_low, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&send_high, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_low, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_high, plane * sizeof(double)));
        }
    }
    ~Slab() {
        if (ranks > 1) {
            CUDA_CHECK(cudaFreeHost(send_low)); CUDA_CHECK(cudaFreeHost(send_high));
            CUDA_CHECK(cudaFreeHost(recv_low)); CUDA_CHECK(cudaFreeHost(recv_high));
        }
        CUDA_CHECK(cudaEventDestroy(ready));
        CUDA_CHECK(cudaStreamDestroy(transfer));
        CUDA_CHECK(cudaStreamDestroy(compute));
    }
    template<bool Chemical, bool Boundary>
    void launch(const double* field, const double* cold, double* out, size_t first, size_t count) {
        if (!count) return;
        const size_t blocks = std::min<size_t>((plane * count + 255) / 256, 65535);
        stencil<Chemical, Boundary><<<static_cast<unsigned>(blocks), 256, 0, compute>>>(
            field, cold, out, nx, ny, nz, rank == 0, rank == ranks - 1, first, count);
        CUDA_CHECK(cudaGetLastError());
    }
    template<bool Chemical>
    void apply(const double* field, const double* cold, double* out) {
        if (ranks == 1) {
            launch<Chemical, false>(field, cold, out, 1, nz);
            return;
        }
        // The transfer stream waits for the previous producer, while the
        // interior stencil can run concurrently with the MPI halo exchange.
        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        const size_t bytes = plane * sizeof(double);
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(send_low, field + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(send_high, field + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        launch<Chemical, false>(field, cold, out, 2, nz > 2 ? nz - 2 : 0);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        // Chunk planes larger than MPI's int count limit. Messages with the
        // same source and tag are matched in order.
        int nreq = 0;
        for (size_t offset = 0; offset < plane;) {
            const int count = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
            if (rank > 0) {
                MPI_CHECK(MPI_Irecv(recv_low + offset, count, MPI_DOUBLE, rank - 1, 1, comm, &requests[nreq++]));
                MPI_CHECK(MPI_Isend(send_low + offset, count, MPI_DOUBLE, rank - 1, 0, comm, &requests[nreq++]));
            }
            if (rank + 1 < ranks) {
                MPI_CHECK(MPI_Irecv(recv_high + offset, count, MPI_DOUBLE, rank + 1, 0, comm, &requests[nreq++]));
                MPI_CHECK(MPI_Isend(send_high + offset, count, MPI_DOUBLE, rank + 1, 1, comm, &requests[nreq++]));
            }
            offset += count;
        }
        MPI_CHECK(MPI_Waitall(nreq, requests.data(), MPI_STATUSES_IGNORE));
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(const_cast<double*>(field), recv_low, bytes, cudaMemcpyHostToDevice, compute));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(const_cast<double*>(field) + (nz + 1) * plane, recv_high, bytes, cudaMemcpyHostToDevice, compute));
        launch<Chemical, true>(field, cold, out, 0, nz == 1 ? 1 : 2);
    }
};

// MPI's traditional count is an int; transfer large result slabs in chunks.
void transfer_result(double* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
        if (send) MPI_CHECK(MPI_Send(data, chunk, MPI_DOUBLE, peer, 2, comm));
        else MPI_CHECK(MPI_Recv(data, chunk, MPI_DOUBLE, peer, 2, comm, MPI_STATUS_IGNORE));
        data += chunk; count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
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
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { if (!world_rank) printf("Unknown option: %s\n", argv[i]); bad = true; break; }
    }
    if (help || bad) {
        if (!world_rank) printUsage(argv[0]);
        MPI_Finalize(); return bad ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(double) ||
        nx * ny > SIZE_MAX / sizeof(double) / 3 || iterations < 0) {
        if (!world_rank) fprintf(stderr, "Invalid grid dimensions or time step count\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    int active = static_cast<int>(std::min<size_t>(world_size, nz));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm));
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(comm, &rank)); MPI_CHECK(MPI_Comm_size(comm, &ranks));
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank, devices = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    const size_t local_z = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t start_z = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    const size_t local_size = plane * local_z;
    if (local_z > SIZE_MAX / plane - 2 || (local_z + 2) * plane > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(local_size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_size; ++i) {
        const size_t id = start_z * plane + i;
        const double pseudo = (((id + 1) * size_t(1299709)) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    int status = 0;
    {
        Slab slab(nx, ny, local_z, rank, ranks, comm);
        double *cold, *cnew, *mu;
        const size_t bytes = (local_z + 2) * plane * sizeof(double);
        CUDA_CHECK(cudaMalloc(&cold, bytes)); CUDA_CHECK(cudaMalloc(&cnew, bytes)); CUDA_CHECK(cudaMalloc(&mu, bytes));
        CUDA_CHECK(cudaMemcpyAsync(cold + plane, host.data(), local_size * sizeof(double), cudaMemcpyHostToDevice, slab.compute));
        CUDA_CHECK(cudaStreamSynchronize(slab.compute));
        MPI_CHECK(MPI_Barrier(comm));
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            slab.apply<true>(cold, cold, mu);
            slab.apply<false>(mu, cold, cnew);
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(slab.compute));
        double elapsed = MPI_Wtime() - start, max_elapsed = 0;
        MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!rank) {
            printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(gridSize) * iterations / max_elapsed / 1e6);
        }
        if (validate || printResults) CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            if (!rank) {
                std::vector<double> result(gridSize);
                std::copy(host.begin(), host.end(), result.begin());
                size_t offset = local_size;
                for (int r = 1; r < ranks; ++r) {
                    size_t count = plane * (nz / ranks + (size_t(r) < nz % ranks));
                    transfer_result(result.data() + offset, count, r, false, comm);
                    offset += count;
                }
                print_results(result, "Concentration");
            } else transfer_result(host.data(), local_size, 0, true, comm);
        }
        if (validate) {
            int invalid = 0;
            double minval = std::numeric_limits<double>::infinity(), maxval = -minval;
            #pragma omp parallel for reduction(|:invalid) reduction(min:minval) reduction(max:maxval) schedule(static)
            for (size_t i = 0; i < local_size; ++i) {
                double value = host[i];
                invalid |= !std::isfinite(value);
                minval = std::min(minval, value); maxval = std::max(maxval, value);
            }
            int any_invalid = 0; double global_min = 0, global_max = 0;
            MPI_CHECK(MPI_Reduce(&invalid, &any_invalid, 1, MPI_INT, MPI_LOR, 0, comm));
            MPI_CHECK(MPI_Reduce(&minval, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
            MPI_CHECK(MPI_Reduce(&maxval, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            if (!rank) {
                printf("Validating result...\n");
                if (any_invalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                    if (global_max > 10.0 || global_min < -10.0) printf("Validation failed: values out of expected range\n");
                }
                status = any_invalid || global_max > 10.0 || global_min < -10.0;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
            MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, comm));
        }
        CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    }
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return status;
}
