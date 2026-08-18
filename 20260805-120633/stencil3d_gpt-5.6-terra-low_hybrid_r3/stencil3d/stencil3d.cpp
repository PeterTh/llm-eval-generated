#include <algorithm>
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

using Real = double;

#define CUDA_CHECK(call) do { \
    const cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

__global__ void initialize_kernel(Real* grid, size_t nx, size_t ny, size_t local_nz,
                                  size_t first_global_z) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z;
    if (x >= nx || y >= ny || lz >= local_nz) return;
    const size_t plane = nx * ny;
    const size_t global_index = (first_global_z + lz) * plane + y * nx + x;
    grid[(lz + 1) * plane + y * nx + x] = static_cast<Real>(global_index % 19);
}

// One MPI rank owns a contiguous z slab.  The two extra planes are MPI halos.
__global__ void stencil_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                               size_t nx, size_t ny, size_t local_nz,
                               size_t first_global_z, size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z + 1;
    if (x >= nx || y >= ny || lz > local_nz) return;

    const size_t plane = nx * ny;
    const size_t i = lz * plane + y * nx + x;
    const size_t gz = first_global_z + lz - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == global_nz) {
        out[i] = in[i];
    } else {
        out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx]
                  + in[i - plane] + in[i + plane]) / 7.0;
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>  Grid X (default: 128)\n  -y <num>  Grid Y (default: X)\n"
           "  -z <num>  Grid Z (default: X)\n  -i <num>  Iterations (default: 10)\n"
           "  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!world_rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        if (!world_rank) fprintf(stderr, "All dimensions must be at least 3 and iterations non-negative.\n");
        MPI_Finalize(); return 1;
    }

    // There cannot be more useful z slabs than z planes; excess ranks exit cleanly.
    const int active_size = std::min<int>(world_size, static_cast<int>(nz));
    MPI_Comm comm;
    // MPI_Comm_split is collective on MPI_COMM_WORLD, including ranks without a slab.
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active_size) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);

    const size_t base = nz / ranks, remainder = nz % ranks;
    const size_t local_nz = base + (static_cast<size_t>(rank) < remainder);
    const size_t first_z = static_cast<size_t>(rank) * base + std::min<size_t>(rank, remainder);
    const size_t plane = nx * ny, local_size = (local_nz + 2) * plane;
    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { if (!rank) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(comm, 1); }
    CUDA_CHECK(cudaSetDevice(world_rank % device_count));

    if (!rank) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
        printf("Initializing grid...\nRunning stencil computation...\n");
    }

    Real *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, local_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_b, local_size * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_a, 0, local_size * sizeof(Real)));
    dim3 block(32, 8, 1), grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, local_nz);
    initialize_kernel<<<grid, block>>>(d_a, nx, ny, local_nz, first_z);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Real> send_first(plane), send_last(plane), recv_lower(plane), recv_upper(plane);
    // Populate the initial input halos before the first stencil launch.
    if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_first.data(), d_a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
    if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_last.data(), d_a + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
    MPI_Sendrecv(send_first.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 10,
                 recv_upper.data(), static_cast<int>(plane), MPI_DOUBLE, next, 10, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_last.data(), static_cast<int>(plane), MPI_DOUBLE, next, 11,
                 recv_lower.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 11, comm, MPI_STATUS_IGNORE);
    if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(d_a, recv_lower.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
    if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(d_a + (local_nz + 1) * plane, recv_upper.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = (iter & 1) ? d_b : d_a;
        Real* out = (iter & 1) ? d_a : d_b;
        stencil_kernel<<<grid, block>>>(in, out, nx, ny, local_nz, first_z, nz);
        CUDA_CHECK(cudaGetLastError());
        // Portable staging keeps this correct with both CUDA-aware and ordinary MPI stacks.
        if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_first.data(), out + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_last.data(), out + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Sendrecv(send_first.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 10,
                     recv_upper.data(), static_cast<int>(plane), MPI_DOUBLE, next, 10, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_last.data(), static_cast<int>(plane), MPI_DOUBLE, next, 11,
                     recv_lower.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 11, comm, MPI_STATUS_IGNORE);
        if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(out, recv_lower.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(out + (local_nz + 1) * plane, recv_upper.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (!rank) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / elapsed / 1.0e6);
    }

    Real* final_device = (iterations & 1) ? d_b : d_a;
    std::vector<Real> local;
    if (validate || printResults) {
        local.resize(local_nz * plane);
        // OpenMP is used for the host-side contiguous slab packing before MPI collection/reduction.
        std::vector<Real> haloed(local_size);
        CUDA_CHECK(cudaMemcpy(haloed.data(), final_device, local_size * sizeof(Real), cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (size_t z = 0; z < local_nz; ++z)
            std::copy_n(haloed.data() + (z + 1) * plane, plane, local.data() + z * plane);
    }

    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -std::numeric_limits<Real>::infinity();
        int bad = 0;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad)
        for (size_t i = 0; i < local.size(); ++i) { const Real v = local[i]; lo = std::min(lo, v); hi = std::max(hi, v); bad |= !std::isfinite(v); }
        Real global_lo, global_hi; int global_bad;
        MPI_Reduce(&lo, &global_lo, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&hi, &global_hi, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&bad, &global_bad, 1, MPI_INT, MPI_BOR, 0, comm);
        if (!rank) { printf("Value range: [%.6f, %.6f]\n", global_lo, global_hi); printf("Validation: %s\n", (!global_bad && global_hi <= 1e6 && global_lo >= -1e6) ? "PASSED" : "FAILED"); }
    }
    if (printResults) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) { const size_t n = base + (static_cast<size_t>(r) < remainder); counts[r] = static_cast<int>(n * plane); offsets[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min<size_t>(r, remainder)) * plane); }
        std::vector<Real> global; if (!rank) global.resize(nx * ny * nz);
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, global.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        if (!rank) print_results(global, "Grid");
    }
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    MPI_Comm_free(&comm); MPI_Finalize();
    return 0;
}
