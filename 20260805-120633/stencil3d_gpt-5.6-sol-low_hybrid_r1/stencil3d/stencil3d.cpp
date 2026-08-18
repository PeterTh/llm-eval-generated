#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void stencil_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                               size_t nx, size_t ny, size_t local_nz, size_t global_z0,
                               size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || lz > local_nz) return;
    const size_t gz = global_z0 + lz - 1;
    if (gz == 0 || gz + 1 == global_nz) return; // fixed physical Z faces
    const size_t plane = nx * ny;
    const size_t p = lz * plane + y * nx + x;
    out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx]
              + in[p-plane] + in[p+plane]) * (1.0 / 7.0);
}

static void usage(const char* p) {
    printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, print_results_flag = false;
    bool args_ok = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else args_ok = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || !args_ok) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return args_ok ? 0 : 1;
    }
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 || (size_t)nranks > nz) {
        if (rank == 0) fprintf(stderr, "Dimensions must be >= 2, iterations >= 0, and ranks <= Z.\n");
        MPI_Finalize(); return 1;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { if (rank == 0) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    const size_t base = nz / nranks, rem = nz % nranks;
    const size_t local_nz = base + ((size_t)rank < rem);
    const size_t global_z0 = rank * base + std::min<size_t>(rank, rem);
    const size_t plane = nx * ny, local_elems = (local_nz + 2) * plane;
    if (plane > (size_t)std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "XY plane exceeds MPI count limit.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    std::vector<Real> host(local_nz * plane);
    #pragma omp parallel for schedule(static)
    for (long long q = 0; q < (long long)host.size(); ++q) {
        const size_t lz = (size_t)q / plane;
        const size_t global_index = (global_z0 + lz) * plane + (size_t)q % plane;
        host[(size_t)q] = (Real)(global_index % 19);
    }

    Real *a = nullptr, *b = nullptr;
    Real* halo = nullptr;
    CUDA_CHECK(cudaMalloc(&a, local_elems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&b, local_elems * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&halo, 4 * plane * sizeof(Real)));
    CUDA_CHECK(cudaMemset(a, 0, local_elems * sizeof(Real)));
    CUDA_CHECK(cudaMemset(b, 0, local_elems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(a + plane, host.data(), host.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b + plane, host.data(), host.size() * sizeof(Real), cudaMemcpyHostToDevice));

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI ranks, up to %d OpenMP threads/rank, CUDA devices\n", nranks, omp_get_max_threads());
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const dim3 block(32, 4, 2);
    const dim3 grid((unsigned)((nx - 2 + block.x - 1) / block.x),
                    (unsigned)((ny - 2 + block.y - 1) / block.y),
                    (unsigned)((local_nz + block.z - 1) / block.z));
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[4]; int nreq = 0;
        Real *send_lo = halo, *send_hi = halo + plane;
        Real *recv_lo = halo + 2 * plane, *recv_hi = halo + 3 * plane;
        // Pinned staging is portable across both CUDA-aware and conventional MPI stacks.
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(send_lo, a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        if (rank + 1 < nranks) CUDA_CHECK(cudaMemcpyAsync(send_hi, a + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (rank > 0) {
            MPI_Irecv(recv_lo, (int)plane, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Isend(send_lo, (int)plane, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (rank + 1 < nranks) {
            MPI_Irecv(recv_hi, (int)plane, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Isend(send_hi, (int)plane, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(a, recv_lo, plane * sizeof(Real), cudaMemcpyHostToDevice));
        if (rank + 1 < nranks) CUDA_CHECK(cudaMemcpyAsync(a + (local_nz + 1) * plane, recv_hi, plane * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (nx > 2 && ny > 2 && nz > 2) {
            stencil_kernel<<<grid, block>>>(a, b, nx, ny, local_nz, global_z0, nz);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize()); // output must be complete before next CUDA-aware exchange
        }
        std::swap(a, b);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const bool need_host = validate || print_results_flag;
    if (need_host) CUDA_CHECK(cudaMemcpy(host.data(), a + plane, host.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        size_t rn = base + ((size_t)r < rem);
        size_t rz = r * base + std::min<size_t>(r, rem);
        if (rn * plane > (size_t)INT_MAX || rz * plane > (size_t)INT_MAX) {
            if (need_host) { if (rank == 0) fprintf(stderr, "Result exceeds MPI_Gatherv count limit.\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
        }
        counts[r] = (int)(rn * plane); displs[r] = (int)(rz * plane);
    }
    std::vector<Real> global;
    if (rank == 0 && need_host) global.resize(nx * ny * nz);
    if (need_host) MPI_Gatherv(host.data(), (int)host.size(), MPI_DOUBLE, global.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1e3);
        const double updates = (double)(nx-2) * (ny-2) * (nz-2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0.0);
        if (print_results_flag) print_results(global, "Grid");
        if (validate) {
            bool ok = true; Real lo = global[0], hi = global[0];
            #pragma omp parallel for reduction(&&:ok) reduction(min:lo) reduction(max:hi)
            for (long long i = 0; i < (long long)global.size(); ++i) {
                const Real v = global[(size_t)i]; ok = ok && std::isfinite(v); lo = std::min(lo, v); hi = std::max(hi, v);
            }
            ok = ok && hi <= 1e6 && lo >= -1e6;
            printf("Value range: [%.6f, %.6f]\nValidation: %s\n", lo, hi, ok ? "PASSED" : "FAILED");
            exit_code = ok ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFreeHost(halo)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    MPI_Finalize();
    return exit_code;
}
