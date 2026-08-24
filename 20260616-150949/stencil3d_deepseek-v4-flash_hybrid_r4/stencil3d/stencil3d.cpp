#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error @ %s:%d  %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                  \
            int r;                                                             \
            MPI_Comm_rank(MPI_COMM_WORLD, &r);                                 \
            fprintf(stderr, "Rank %d aborting\n", r);                          \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// 3-D index helper – callable from host and device
// ---------------------------------------------------------------------------
inline constexpr __host__ __device__ size_t
idx3(const size_t x, const size_t y, const size_t z,
     const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA stencil kernel – processes interior points only
// Each thread handles one (lx, ly) column and loops over lz in [z0, z1].
// ---------------------------------------------------------------------------
__global__ void stencil_kernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny,
                               const size_t z0, const size_t z1) {
    // Interior X / Y in local coordinates  (ghosts are at 0 and nz_local+1)
    const size_t lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t ly = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (lx >= nx - 1 || ly >= ny - 1) return;

    // Stride through Z
    for (size_t lz = z0; lz <= z1; ++lz) {
        const size_t ci = idx3(lx, ly, lz, nx, ny);

        const Real c   = input[ci];
        const Real l   = input[idx3(lx - 1, ly,     lz,     nx, ny)];
        const Real r   = input[idx3(lx + 1, ly,     lz,     nx, ny)];
        const Real f   = input[idx3(lx,     ly - 1, lz,     nx, ny)];
        const Real b   = input[idx3(lx,     ly + 1, lz,     nx, ny)];
        const Real d   = input[idx3(lx,     ly,     lz - 1, nx, ny)];
        const Real u   = input[idx3(lx,     ly,     lz + 1, nx, ny)];

        output[ci] = (c + l + r + f + b + d + u) * static_cast<Real>(1.0 / 7.0);
    }
}

// ---------------------------------------------------------------------------
// Halo (ghost cell) exchange using non‑blocking MPI
// ---------------------------------------------------------------------------
static void exchange_halos(Real* d_grid,
                           Real* h_buf,
                           const size_t nx, const size_t ny,
                           const size_t nz_local,
                           const int rank, const int size,
                           MPI_Comm comm) {

    const size_t plane = nx * ny;          // elements per plane
    const MPI_Datatype mpi_real = sizeof(Real) == sizeof(double)
                                      ? MPI_DOUBLE
                                      : MPI_FLOAT;

    Real* h_bottom_send = h_buf;                // owned bottom plane
    Real* h_top_send    = h_buf + plane;        // owned top    plane
    Real* h_bottom_recv = h_buf + 2 * plane;    // bottom ghost
    Real* h_top_recv    = h_buf + 3 * plane;    // top    ghost

    // Device → host
    CUDA_CHECK(cudaMemcpy(h_bottom_send,
                          d_grid + plane,                // local_z = 1
                          plane * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_top_send,
                          d_grid + nz_local * plane,     // local_z = nz_local
                          plane * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    // Non‑blocking halo exchange (tag convention does not matter as long
    // as sends and receives between the same pair are matched)
    constexpr int TAG_BOTTOM = 0;   // bottom plane ↔ top ghost of neighbour below
    constexpr int TAG_TOP    = 1;   // top    plane ↔ bottom ghost of neighbour above

    MPI_Request req[4];
    int nreq = 0;

    // ---- Neighbour below (rank-1) ----
    if (rank > 0) {
        MPI_Irecv(h_bottom_recv, static_cast<int>(plane), mpi_real,
                  rank - 1, TAG_TOP, comm, &req[nreq++]);
        MPI_Isend(h_bottom_send, static_cast<int>(plane), mpi_real,
                  rank - 1, TAG_BOTTOM, comm, &req[nreq++]);
    }
    // ---- Neighbour above (rank+1) ----
    if (rank < size - 1) {
        MPI_Irecv(h_top_recv, static_cast<int>(plane), mpi_real,
                  rank + 1, TAG_BOTTOM, comm, &req[nreq++]);
        MPI_Isend(h_top_send, static_cast<int>(plane), mpi_real,
                  rank + 1, TAG_TOP, comm, &req[nreq++]);
    }

    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    // Host → device (ghost regions)
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(d_grid,                     // local_z = 0
                              h_bottom_recv,
                              plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
    if (rank < size - 1) {
        CUDA_CHECK(cudaMemcpy(d_grid + (nz_local + 1) * plane,   // local_z = nz_local+1
                              h_top_recv,
                              plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
}

// ---------------------------------------------------------------------------
// Initialise the owned portion of a local (padded) grid
// Uses OpenMP for parallelism on the host.
// ---------------------------------------------------------------------------
static void initialize_local_grid(Real* grid,
                                  const size_t nx, const size_t ny,
                                  const size_t nz_local,
                                  const size_t z_start) {
#pragma omp parallel for collapse(3)
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + (lz - 1);
                const size_t gidx = idx3(x, y, global_z, nx, ny);
                grid[idx3(x, y, lz, nx, ny)] = static_cast<Real>((gidx % 19) * 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Local grid validation  (no ghost cells included)
// ---------------------------------------------------------------------------
static bool validate_owned_grid(const Real* owned,     // nz_local consecutive planes
                                const size_t nx, const size_t ny,
                                const size_t nz_local) {

    Real min_val =  INFINITY;
    Real max_val = -INFINITY;
    bool ok = true;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val) \
    reduction(&& : ok) collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const Real v = owned[idx3(x, y, z, nx, ny)];
                if (std::isnan(v) || std::isinf(v)) ok = false;
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!ok) {
        if (rank == 0) printf("Validation failed: NaN or Inf found on rank %d\n", rank);
        return false;
    }

    // Global min / max
    Real gmin, gmax;
    MPI_Allreduce(&min_val, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_val, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
    }

    if (gmax > 1.0e6 || gmin < -1.0e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ===========================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- parse arguments (rank 0 only prints) ------------------------------
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int    iterations = 10;
    bool   validate   = false;
    bool   printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) { nx = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) { ny = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) { nz = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) { iterations = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-v") == 0) { validate = true; }
        else if (strcmp(argv[i], "-r") == 0) { printResults = true; }
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark (hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Processes: %d\n", size);
        printf("OMP threads per rank: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- CUDA device selection --------------------------------------------
    {
        int ndev = 0;
        cudaGetDeviceCount(&ndev);
        if (ndev == 0) {
            fprintf(stderr, "Rank %d: no CUDA-capable device found\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        CUDA_CHECK(cudaSetDevice(rank % ndev));

        struct cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, rank % ndev));
        if (rank == 0)
            printf("CUDA device: %s\n", prop.name);
    }

    // ---- Domain decomposition along Z ------------------------------------
    const size_t base_nz  = nz / static_cast<size_t>(size);
    const size_t rem      = nz % static_cast<size_t>(size);
    const size_t nz_local = base_nz + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // z_start = global Z index of first owned plane for this rank
    size_t z_start;
    if (static_cast<size_t>(rank) < rem) {
        z_start = static_cast<size_t>(rank) * nz_local;
    } else {
        z_start = rem * (base_nz + 1) +
                  (static_cast<size_t>(rank) - rem) * base_nz;
    }

    // Local Z range for interior stencil computation
    // (exclude global boundaries: z=0 and z=nz-1)
    const size_t z0 = (rank == 0)   ? 2 : 1;   // first local z for stencil
    const size_t z1 = (rank == size - 1) ? (nz_local - 1) : nz_local;

    const size_t plane     = nx * ny;                    // elements per plane
    const size_t local_sz  = plane * nz_local;           // owned element count
    const size_t padded_sz = plane * (nz_local + 2);     // +2 ghost planes

    if (rank == 0) {
        printf("Domain: %zu planes per rank (last rank may have fewer/more)\n",
               nz_local);
    }

    // ---- Allocate host (pinned) buffers and device grids ------------------
    Real* h_grid   = nullptr;   // host buffer for gather & I/O
    Real* h_buf    = nullptr;   // pinned MPI exchange buffer (4 planes)
    Real* d_grid1  = nullptr;   // device grid 1
    Real* d_grid2  = nullptr;   // device grid 2

    // Gather buffer only on rank 0
    if (rank == 0) {
        CUDA_CHECK(cudaMallocHost(&h_grid, plane * nz * sizeof(Real)));
    }
    CUDA_CHECK(cudaMallocHost(&h_buf, 4 * plane * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid1, padded_sz * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, padded_sz * sizeof(Real)));

    // ---- Initialise owned cells on host, copy to device ------------------
    // Use a temporary host buffer for initialization
    {
        std::vector<Real> host_local(padded_sz, Real{0});
        initialize_local_grid(host_local.data(), nx, ny, nz_local, z_start);

        CUDA_CHECK(cudaMemcpy(d_grid1, host_local.data(),
                              padded_sz * sizeof(Real),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid2, host_local.data(),
                              padded_sz * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Timed iteration loop --------------------------------------------
    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in  = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        // 1. Exchange ghost cells (MPI halo exchange)
        exchange_halos(d_in, h_buf, nx, ny, nz_local, rank, size,
                       MPI_COMM_WORLD);

        // 2. Launch CUDA stencil kernel
        if (z0 <= z1) {
            // 2-D block grid – each thread sweeps the Z range
            constexpr int BX = 16, BY = 16;
            dim3 blk(BX, BY);
            dim3 grd(static_cast<unsigned int>((nx - 2 + BX - 1) / BX),
                     static_cast<unsigned int>((ny - 2 + BY - 1) / BY));

            stencil_kernel<<<grd, blk>>>(d_in, d_out, nx, ny, z0, z1);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t_end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1,
               MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Gather results to rank 0 ----------------------------------------
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;

    // Copy owned portion to host
    std::vector<Real> local_owned(local_sz);
    CUDA_CHECK(cudaMemcpy(local_owned.data(),
                          d_final + plane,               // skip bottom ghost
                          local_sz * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    // Gather all owned slabs to rank 0
    {
        const MPI_Datatype mpi_real = sizeof(Real) == sizeof(double)
                                          ? MPI_DOUBLE : MPI_FLOAT;

        std::vector<int> recvcounts(static_cast<size_t>(size));
        std::vector<int> displs(static_cast<size_t>(size));

        int my_count = static_cast<int>(local_sz);
        MPI_Gather(&my_count, 1, MPI_INT,
                   recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (size_t i = 1; i < static_cast<size_t>(size); ++i)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Gatherv(local_owned.data(), my_count, mpi_real,
                    h_grid, recvcounts.data(), displs.data(), mpi_real,
                    0, MPI_COMM_WORLD);
    }

    // ---- Print timing & performance --------------------------------------
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        const double cell_updates =
            static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) *
            static_cast<double>(iterations);
        const double mcups = cell_updates /
                             (static_cast<double>(duration_ms) / 1000.0) /
                             1.0e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Print results for external validation (rank 0) ------------------
    if (printResults && rank == 0) {
        std::vector<Real> full_grid(h_grid, h_grid + plane * nz);
        print_results(full_grid, "Grid");
    }

    // ---- Validation ------------------------------------------------------
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        // Each rank validates its own portion (already on host in local_owned)
        bool local_ok = validate_owned_grid(local_owned.data(), nx, ny, nz_local);
        int  local_ok_i = local_ok ? 1 : 0;
        int  global_ok_i = 0;
        MPI_Reduce(&local_ok_i, &global_ok_i, 1,
                   MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n",
                   global_ok_i ? "PASSED" : "FAILED");
        }
    }

    // ---- Cleanup ---------------------------------------------------------
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFreeHost(h_buf));
    if (rank == 0) CUDA_CHECK(cudaFreeHost(h_grid));

    MPI_Finalize();
    return 0;
}
