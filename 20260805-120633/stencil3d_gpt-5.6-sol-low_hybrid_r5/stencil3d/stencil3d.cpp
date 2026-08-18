#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e_));                                   \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

__global__ void stencil_kernel(const Real* __restrict__ in,
                               Real* __restrict__ out, size_t nx, size_t ny,
                               size_t local_nz, size_t global_z0, size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || lz > local_nz) return;
    const size_t i = (lz * ny + y) * nx + x;
    const size_t gz = global_z0 + lz - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        gz == 0 || gz + 1 == global_nz) {
        out[i] = in[i];
    } else {
        const size_t plane = nx * ny;
        out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] +
                  in[i + nx] + in[i - plane] + in[i + plane]) * (1.0 / 7.0);
    }
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n", p);
    std::printf("  -x <num>  X size (default 128)\n  -y <num>  Y size\n");
    std::printf("  -z <num>  Z size\n  -i <num>  iterations (default 10)\n");
    std::printf("  -v        validate\n  -r        print results\n  -h        help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    bool bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || bad) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || (size_t)nranks > nz ||
        nx > (size_t)std::numeric_limits<int>::max() / ny) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be >= 3, iterations >= 0, and ranks <= Z.\n");
        MPI_Finalize();
        return 1;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    const size_t base = nz / nranks, rem = nz % nranks;
    const size_t local_nz = base + ((size_t)rank < rem);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, rem);
    const size_t plane = nx * ny, local_cells = local_nz * plane;
    std::vector<Real> host(local_cells);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)local_cells; ++i)
        host[(size_t)i] = (Real)(((z0 * plane + (size_t)i) % 19));

    Real *a = nullptr, *b = nullptr;
    const size_t alloc_cells = (local_nz + 2) * plane;
    CUDA_CHECK(cudaMalloc(&a, alloc_cells * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&b, alloc_cells * sizeof(Real)));
    CUDA_CHECK(cudaMemset(a, 0, alloc_cells * sizeof(Real)));
    CUDA_CHECK(cudaMemset(b, 0, alloc_cells * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(a + plane, host.data(), local_cells * sizeof(Real), cudaMemcpyHostToDevice));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n",
                    iterations, validate ? "enabled" : "disabled", nranks, omp_get_max_threads());
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int plane_count = (int)plane;
    const dim3 block(32, 4, 2);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                    (unsigned)((ny + block.y - 1) / block.y),
                    (unsigned)((local_nz + block.z - 1) / block.z));
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[4]; int nreq = 0;
        // CUDA-aware MPI keeps halos on the accelerator and avoids host staging.
        if (rank > 0) {
            MPI_Irecv(a, plane_count, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Isend(a + plane, plane_count, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (rank + 1 < nranks) {
            MPI_Irecv(a + (local_nz + 1) * plane, plane_count, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Isend(a + local_nz * plane, plane_count, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        stencil_kernel<<<grid, block>>>(a, b, nx, ny, local_nz, z0, nz);
        CUDA_CHECK(cudaGetLastError());
        std::swap(a, b);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data(), a + plane, local_cells * sizeof(Real), cudaMemcpyDeviceToHost));

    std::vector<Real> global;
    if (printResults) {
        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(nranks); displs.resize(nranks);
            for (int r = 0; r < nranks; ++r) {
                size_t rnz = base + ((size_t)r < rem);
                size_t rz0 = (size_t)r * base + std::min((size_t)r, rem);
                if (rnz * plane > (size_t)std::numeric_limits<int>::max() || rz0 * plane > (size_t)std::numeric_limits<int>::max()) {
                    std::fprintf(stderr, "Grid is too large for MPI_Gatherv counts.\n"); MPI_Abort(MPI_COMM_WORLD, 3);
                }
                counts[r] = (int)(rnz * plane); displs[r] = (int)(rz0 * plane);
            }
            global.resize(nx * ny * nz);
        }
        MPI_Gatherv(host.data(), (int)local_cells, MPI_DOUBLE, global.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int rc = 0;
    Real local_lo = std::numeric_limits<Real>::infinity(), local_hi = -local_lo;
    int local_invalid = 0;
    if (validate) {
        #pragma omp parallel for reduction(min:local_lo) reduction(max:local_hi) reduction(|:local_invalid)
        for (long long i = 0; i < (long long)host.size(); ++i) {
            const Real v = host[(size_t)i];
            if (!std::isfinite(v)) local_invalid = 1;
            local_lo = std::min(local_lo, v); local_hi = std::max(local_hi, v);
        }
    }
    Real lo = 0.0, hi = 0.0;
    int invalid = 0;
    if (validate) {
        MPI_Reduce(&local_lo, &lo, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_hi, &hi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_invalid, &invalid, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double updates = (double)(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0.0);
        if (printResults) print_results(global, "Grid");
        if (validate) {
            std::printf("Validating result...\nValue range: [%.6f, %.6f]\n", lo, hi);
            rc = invalid || hi > 1e6 || lo < -1e6;
            std::printf("Validation: %s\n", rc ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(a); cudaFree(b);
    MPI_Finalize();
    return rc;
}
