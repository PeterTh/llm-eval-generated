#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;
#define TILE 16

// ---------------------------------------------------------------------------
// CUDA kernel – 7-point stencil with shared-memory tiling.
//   Each block covers a TILE×TILE region in one z-slab.  Shared memory
//   holds the tile plus a one-cell halo in x/y; z-neighbours come from
//   global memory.  Only interior cells are written (1 <= x,y < nx-1,ny-1
//   and 1 <= z < local_nz-1) so halo planes survive untouched.
// ---------------------------------------------------------------------------
__global__ void stencilKernel(
    const Real* __restrict__ inp,
    Real* __restrict__ out,
    int nx, int ny, int local_nz)
{
    __shared__ Real sTile[TILE + 2][TILE + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,    by = blockIdx.y,    bz = blockIdx.z;

    int gx = bx * TILE + tx;
    int gy = by * TILE + ty;
    int gz = bz + 1;                       // skip first halo plane

    int sx = tx + 1, sy = ty + 1;
    size_t xy = static_cast<size_t>(nx) * ny;

    // --- Load core cell -------------------------------------------------
    if (gx < nx && gy < ny)
        sTile[sy][sx] = inp[static_cast<size_t>(gz) * xy
                           + static_cast<size_t>(gy) * nx + gx];

    // --- Load one-cell halo around the tile -----------------------------
    if (tx == 0     && gx > 0     && gy < ny)
        sTile[sy][0]        = inp[static_cast<size_t>(gz) * xy
                                 + static_cast<size_t>(gy) * nx + (gx - 1)];
    if (tx == TILE-1 && gx + 1 < nx && gy < ny)
        sTile[sy][TILE + 1] = inp[static_cast<size_t>(gz) * xy
                                 + static_cast<size_t>(gy) * nx + (gx + 1)];
    if (ty == 0     && gx < nx     && gy > 0)
        sTile[0][sx]        = inp[static_cast<size_t>(gz) * xy
                                 + static_cast<size_t>(gy - 1) * nx + gx];
    if (ty == TILE-1 && gx < nx && gy + 1 < ny)
        sTile[TILE + 1][sx] = inp[static_cast<size_t>(gz) * xy
                                 + static_cast<size_t>(gy + 1) * nx + gx];

    __syncthreads();

    // --- Compute stencil ------------------------------------------------
    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1
        && gz > 0 && gz < local_nz - 1)
    {
        Real center = sTile[sy][sx];
        Real left   = sTile[sy][sx - 1];
        Real right  = sTile[sy][sx + 1];
        Real front  = sTile[sy - 1][sx];
        Real back   = sTile[sy + 1][sx];

        size_t base = static_cast<size_t>(gz) * xy
                    + static_cast<size_t>(gy) * nx + gx;
        Real bottom = inp[base - xy];
        Real top    = inp[base + xy];

        out[base] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// ---------------------------------------------------------------------------
// Non-blocking halo exchange in Z (1 halo plane on each side).
//   Sends interior planes adjacent to the halos and receives into the
//   halo planes from the neighbours.  Global z-boundaries (rank 0's z=0
//   and rank N-1's z=local_nz-1) are never touched.
// ---------------------------------------------------------------------------
void exchangeHalos(Real* grid, int nx, int ny, int local_nz,
                   int mpi_rank, int mpi_size)
{
    int plane = nx * ny;
    MPI_Request req[4];
    int n = 0;

    // receive bottom halo (z=0) from rank-1's top interior
    if (mpi_rank > 0)
        MPI_Irecv(grid, plane, MPI_DOUBLE, mpi_rank - 1, 42,
                  MPI_COMM_WORLD, &req[n++]);
    // receive top halo (z=local_nz-1) from rank+1's bottom interior
    if (mpi_rank < mpi_size - 1)
        MPI_Irecv(grid + static_cast<size_t>(local_nz - 1) * plane,
                  plane, MPI_DOUBLE, mpi_rank + 1, 42,
                  MPI_COMM_WORLD, &req[n++]);

    // send bottom interior (z=1) to rank-1
    if (mpi_rank > 0)
        MPI_Isend(grid + plane, plane, MPI_DOUBLE, mpi_rank - 1, 42,
                  MPI_COMM_WORLD, &req[n++]);
    // send top interior (z=local_nz-2) to rank+1
    if (mpi_rank < mpi_size - 1)
        MPI_Isend(grid + static_cast<size_t>(local_nz - 2) * plane,
                  plane, MPI_DOUBLE, mpi_rank + 1, 42,
                  MPI_COMM_WORLD, &req[n++]);

    MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
}

// ---------------------------------------------------------------------------
// Copy x/y boundary cells from input to output (OpenMP).
// ---------------------------------------------------------------------------
void copyXYBoundaries(const Real* inp, Real* out,
                      int nx, int ny, int local_nz)
{
    size_t xy = static_cast<size_t>(nx) * ny;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < local_nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
                    size_t i = static_cast<size_t>(z) * xy
                             + static_cast<size_t>(y) * nx + x;
                    out[i] = inp[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<Real>& grid,
                    [[maybe_unused]] size_t nx,
                    [[maybe_unused]] size_t ny,
                    [[maybe_unused]] size_t nz)
{
    for (const auto& v : grid)
        if (std::isnan(v) || std::isinf(v)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    Real lo = grid[0], hi = grid[0];
    for (const auto& v : grid) { lo = std::min(lo, v); hi = std::max(hi, v); }
    printf("Value range: [%.6f, %.6f]\n", lo, hi);
    if (hi > 1e6 || lo < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* prog)
{
    printf("Usage: mpirun -np <procs> %s [options]\n", prog);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // ---- parse CLI ----------------------------------------------------
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)
            nx = static_cast<size_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)
            ny = static_cast<size_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)
            nz = static_cast<size_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!mpi_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!mpi_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;

    // ---- domain decomposition (Z axis, 1 halo plane each side) --------
    int remaining   = static_cast<int>(nz) - 2;
    int base_int    = remaining / mpi_size;
    int leftover    = remaining % mpi_size;
    int my_interior = base_int + (mpi_rank < leftover ? 1 : 0);
    int local_nz    = my_interior + 2;   // interior + 2 halo planes

    // global z offset of first interior plane
    int z_start = 0;
    for (int r = 0; r < mpi_rank; ++r)
        z_start += base_int + (r < leftover ? 1 : 0);
    z_start += 1;   // skip global z=0 boundary

    size_t local_size = static_cast<size_t>(nx) * ny * local_nz;
    std::vector<Real> grid1(local_size), grid2(local_size);
    size_t xy = static_cast<size_t>(nx) * ny;

    if (!mpi_rank) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- initialise local grid (OpenMP) --------------------------------
    #pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t gidx = static_cast<size_t>(z_start + z - 1) * xy
                            + y * nx + x;
                grid1[static_cast<size_t>(z) * xy + y * nx + x]
                    = static_cast<Real>(gidx % 19);
            }
        }
    }

    // ---- CUDA setup ----------------------------------------------------
    cudaSetDevice(0);
    Real *d_g1 = nullptr, *d_g2 = nullptr;
    cudaMalloc(&d_g1, local_size * sizeof(Real));
    cudaMalloc(&d_g2, local_size * sizeof(Real));

    // ---- main stencil loop ---------------------------------------------
    if (!mpi_rank) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *h_in  = (iter & 1) ? grid2.data() : grid1.data();
        Real *h_out = (iter & 1) ? grid1.data() : grid2.data();
        Real *d_in  = (iter & 1) ? d_g2         : d_g1;
        Real *d_out = (iter & 1) ? d_g1         : d_g2;

        // 1. exchange halos of input on host
        exchangeHalos(h_in, static_cast<int>(nx), static_cast<int>(ny),
                      local_nz, mpi_rank, mpi_size);

        // 2. upload input to GPU
        cudaMemcpy(d_in, h_in, local_size * sizeof(Real),
                   cudaMemcpyHostToDevice);

        // 3. GPU stencil (interior only)
        int iz = local_nz - 2;
        if (iz > 0) {
            dim3 blk(TILE, TILE, 1);
            dim3 grd((static_cast<int>(nx) + TILE - 1) / TILE,
                     (static_cast<int>(ny) + TILE - 1) / TILE,
                     iz);
            stencilKernel<<<grd, blk>>>(d_in, d_out,
                                        static_cast<int>(nx),
                                        static_cast<int>(ny),
                                        local_nz);
        }

        // 4. download output
        cudaMemcpy(h_out, d_out, local_size * sizeof(Real),
                   cudaMemcpyDeviceToHost);

        // 5. copy x/y boundaries from input to output
        copyXYBoundaries(h_in, h_out,
                         static_cast<int>(nx),
                         static_cast<int>(ny),
                         local_nz);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // ---- gather interior planes to rank 0 ------------------------------
    const std::vector<Real>& final = (iterations % 2 == 0) ? grid1 : grid2;

    // contiguous interior buffer (skip halo planes)
    std::vector<Real> interior(my_interior * nx * ny);
    for (int z = 0; z < my_interior; ++z)
        std::memcpy(interior.data() + z * nx * ny,
                    final.data() + (z + 1) * xy,
                    nx * ny * sizeof(Real));

    std::vector<int> rcv(mpi_size);
    std::vector<int> disp(mpi_size);
    int total = 0;
    for (int r = 0; r < mpi_size; ++r) {
        int ri = base_int + (r < leftover ? 1 : 0);
        int c = ri * nx * ny;
        rcv[r]  = c;
        disp[r] = total;
        total  += c;
    }

    std::vector<Real> gathered(static_cast<size_t>(total));
    MPI_Gatherv(interior.data(), my_interior * nx * ny, MPI_DOUBLE,
                gathered.data(), rcv.data(), disp.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- rank 0: assemble global grid, report, validate ----------------
    if (!mpi_rank) {
        size_t gs = nx * ny * nz;
        std::vector<Real> global(gs);
        // initialise with original values (preserves global boundaries)
        for (size_t i = 0; i < gs; ++i)
            global[i] = static_cast<Real>(i % 19);

        // overwrite interior with gathered data
        int zo = 1;
        for (int r = 0; r < mpi_size; ++r) {
            int ri = base_int + (r < leftover ? 1 : 0);
            std::memcpy(global.data() + static_cast<size_t>(zo) * xy,
                        gathered.data() + disp[r],
                        static_cast<size_t>(ri) * nx * ny * sizeof(Real));
            zo += ri;
        }

        printf("Computation time: %ld ms\n", ms);
        double cu = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n",
               cu / (ms / 1000.0) / 1e6);

        if (printResults) print_results(global, "Grid");

        if (validate) {
            printf("Validating result...\n");
            printf("Validation: %s\n",
                   validateResult(global, nx, ny, nz) ? "PASSED" : "FAILED");
        }
    }

    cudaFree(d_g1);
    cudaFree(d_g2);
    MPI_Finalize();
    return 0;
}
