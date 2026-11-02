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
    int rank_ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Each slab has a ghost plane at either end. The two physical end planes
// use their own value in place of a missing neighbor (clamped boundaries).
__global__ void stencil(const double* in, const double* cold, double* out,
                        size_t nx, size_t ny, size_t nz, size_t first, size_t count,
                        bool lower_edge, bool upper_edge, bool chemical) {
    const size_t plane = nx * ny;
    for (size_t z = first + blockIdx.z; z < first + count; z += gridDim.z)
    for (size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y; y < ny; y += (size_t)gridDim.y * blockDim.y)
    for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < nx; x += (size_t)gridDim.x * blockDim.x) {
        const size_t rem = y * nx + x;
        const size_t i = z * plane + rem;
        const double v = in[i];
        const double xx = (in[i + (x + 1 < nx ? 1 : 0)] + in[i - (x > 0 ? 1 : 0)] - 2.0 * v);
        const double yy = (in[i + (y + 1 < ny ? nx : 0)] + in[i - (y > 0 ? nx : 0)] - 2.0 * v);
        const double zz = (in[i + ((z == nz && upper_edge) ? 0 : plane)] +
                           in[i - ((z == 1 && lower_edge) ? 0 : plane)] - 2.0 * v);
        const double lap = xx + yy + zz;
        if (chemical) {
            const double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
            out[i] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
                     + 3.0 * v + v * v * v - 0.5 * lap;
        } else {
            out[i] = cold[i] + 0.01 * lap;
        }
    }
}

static void launch(const double* in, const double* cold, double* out, size_t nx, size_t ny,
                   size_t nz, size_t first, size_t count, bool low, bool high, bool chemical,
                   cudaStream_t stream) {
    if (!count) return;
    dim3 threads(32, 4);
    dim3 blocks(static_cast<unsigned>(std::min<size_t>((nx + 31) / 32, 2147483647)),
                static_cast<unsigned>(std::min<size_t>((ny + 3) / 4, 65535)),
                static_cast<unsigned>(std::min<size_t>(count, 65535)));
    stencil<<<blocks, threads, 0, stream>>>(in, cold, out, nx, ny, nz, first, count, low, high, chemical);
    CUDA_CHECK(cudaGetLastError());
}

// Host staging works with any MPI implementation, including ones without CUDA-aware MPI.
// Compute interior planes while MPI transfers the two boundary planes.
static void phase(double* in, const double* cold, double* out, size_t nx, size_t ny,
                  size_t nz, int rank, int ranks, bool chemical, double* staging,
                  cudaStream_t compute, cudaStream_t transfer, MPI_Comm comm) {
    const size_t plane = nx * ny, bytes = plane * sizeof(double);
    const bool low = rank == 0, high = rank == ranks - 1;
    if (ranks == 1) {
        launch(in, cold, out, nx, ny, nz, 1, nz, true, true, chemical, compute);
        return;
    }
    // For one-plane slabs the same plane is sent to both neighbors.
    if (!low) CUDA_CHECK(cudaMemcpyAsync(staging, in + plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (!high) CUDA_CHECK(cudaMemcpyAsync(staging + plane, in + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (nz > 2) launch(in, cold, out, nx, ny, nz, 2, nz - 2, low, high, chemical, compute);
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    std::vector<MPI_Request> requests;
    for (size_t offset = 0; offset < plane;) {
        int count = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
        MPI_Request req;
        if (!low) {
            MPI_Irecv(staging + 2 * plane + offset, count, MPI_DOUBLE, rank - 1, 1, comm, &req); requests.push_back(req);
            MPI_Isend(staging + offset, count, MPI_DOUBLE, rank - 1, 0, comm, &req); requests.push_back(req);
        }
        if (!high) {
            MPI_Irecv(staging + 3 * plane + offset, count, MPI_DOUBLE, rank + 1, 0, comm, &req); requests.push_back(req);
            MPI_Isend(staging + plane + offset, count, MPI_DOUBLE, rank + 1, 1, comm, &req); requests.push_back(req);
        }
        offset += count;
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    if (!low) CUDA_CHECK(cudaMemcpyAsync(in, staging + 2 * plane, bytes, cudaMemcpyHostToDevice, transfer));
    if (!high) CUDA_CHECK(cudaMemcpyAsync(in + (nz + 1) * plane, staging + 3 * plane, bytes, cudaMemcpyHostToDevice, transfer));
    launch(in, cold, out, nx, ny, nz, 1, 1, low, high, chemical, transfer);
    if (nz > 1) launch(in, cold, out, nx, ny, nz, nz, 1, low, high, chemical, transfer);
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    CUDA_CHECK(cudaStreamSynchronize(compute));
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20, parse_status = 0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char* opt = argv[i++];
            char* end = nullptr;
            long long value = strtoll(argv[i], &end, 10);
            if (*end || end == argv[i] || value < 0 || (!strcmp(opt, "-i") && value > INT_MAX)) { parse_status = 1; break; }
            if (!strcmp(opt, "-x")) nx = static_cast<size_t>(value);
            else if (!strcmp(opt, "-y")) ny = static_cast<size_t>(value);
            else if (!strcmp(opt, "-z")) nz = static_cast<size_t>(value);
            else iterations = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { parse_status = 2; break; }
        else { if (!world_rank) printf("Unknown option: %s\n", argv[i]); parse_status = 1; break; }
    }
    if (parse_status) {
        if (!world_rank) printUsage(argv[0]);
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (!nx || nx > limit / ny || nx * ny > limit / nz || nx * ny > limit / sizeof(double) / 4 ||
        nz > limit - 2 || nx * ny > limit / (nz + 2) / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, volume = plane * nz;
    int active = static_cast<int>(std::min<size_t>(world_size, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    const size_t local_nz = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t start_z = (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
    const size_t local_cells = plane * local_nz;

    MPI_Comm shared;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &shared);
    int local_rank, devices = 0;
    MPI_Comm_rank(shared, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&shared);

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(local_cells);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_cells; ++i) {
        size_t id = start_z * plane + i;
        double pseudo = (((id + 1) * size_t(1299709)) % volume) / static_cast<double>(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu, *staging = nullptr;
    size_t alloc_bytes = (local_nz + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, alloc_bytes)); CUDA_CHECK(cudaMalloc(&cnew, alloc_bytes)); CUDA_CHECK(cudaMalloc(&mu, alloc_bytes));
    if (ranks > 1) CUDA_CHECK(cudaMallocHost(&staging, 4 * plane * sizeof(double)));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), local_cells * sizeof(double), cudaMemcpyHostToDevice));
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        phase(cold, cold, mu, nx, ny, local_nz, rank, ranks, true, staging, compute, transfer, comm);
        phase(mu, cold, cnew, nx, ny, local_nz, rank, ranks, false, staging, compute, transfer, comm);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", (double)volume * iterations / max_elapsed / 1e6);
    }
    if (validate || printResults) CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_cells * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<double> all;
        if (!rank) {
            all.resize(volume);
            std::copy(host.begin(), host.end(), all.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t rstart = (nz / ranks) * r + std::min<size_t>(r, nz % ranks);
                size_t count = (nz / ranks + (static_cast<size_t>(r) < nz % ranks)) * plane;
                for (size_t off = 0; off < count;) {
                    int n = static_cast<int>(std::min<size_t>(count - off, INT_MAX));
                    MPI_Recv(all.data() + rstart * plane + off, n, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE); off += n;
                }
            }
            print_results(all, "Concentration");
        } else {
            for (size_t off = 0; off < local_cells;) {
                int n = static_cast<int>(std::min<size_t>(local_cells - off, INT_MAX));
                MPI_Send(host.data() + off, n, MPI_DOUBLE, 0, 2, comm); off += n;
            }
        }
    }
    int valid = 1;
    if (validate) {
        int invalid = 0;
        double minval = std::numeric_limits<double>::infinity(), maxval = -minval;
        #pragma omp parallel for reduction(|:invalid) reduction(min:minval) reduction(max:maxval) schedule(static)
        for (size_t i = 0; i < local_cells; ++i) {
            double v = host[i];
            if (!std::isfinite(v)) invalid = 1;
            minval = std::min(minval, v); maxval = std::max(maxval, v);
        }
        int any_invalid; double global_min, global_max;
        MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&minval, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&maxval, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        valid = !any_invalid && global_min >= -10.0 && global_max <= 10.0;
        if (!rank) {
            printf("Validating result...\n");
            if (any_invalid) printf("Validation failed: found NaN or Inf value\n");
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
