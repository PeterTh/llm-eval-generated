#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHK(call)                                                         \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

/*
 * CUDA kernel: 7-point 3D stencil.
 *
 * Layout: z=0 is bottom halo, z=1..local_nz are owned, z=local_nz+1 is top halo.
 * Only owned interior cells get stencil updates; halos and global boundaries
 * are copied verbatim.
 */
__global__ void stencilKernel(const Real * __restrict__ in,
                               Real * __restrict__ out,
                               size_t nx, size_t ny,
                               size_t local_nz,
                               size_t local_nz_h,
                               int z_start_g, int nz_g)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= local_nz_h)
        return;

    const size_t sl  = nx * ny;
    const size_t idx = z * sl + y * nx + x;
    const int    zg  = z_start_g + static_cast<int>(z) - 1;

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 &&
        z >= 1 && z <= local_nz &&
        zg > 0 && zg < nz_g - 1)
    {
        out[idx] = (in[idx]       + in[idx - 1] + in[idx + 1] +
                    in[idx - nx]  + in[idx + nx] +
                    in[idx - sl]  + in[idx + sl]) / 7.0;
    }
    else
    {
        out[idx] = in[idx];
    }
}

/*
 * Halo exchange via MPI + cudaMemcpy.
 *
 * Layout (local_nz_h = local_nz + 2):
 *   z = 0          : bottom halo   (global z = z_start - 1)
 *   z = 1..local_nz: owned cells   (global z = z_start .. z_start+local_nz-1)
 *   z = local_nz+1 : top halo      (global z = z_start + local_nz)
 *
 * rank 0's top halo needs global z = z_start(0) + local_nz(0) = z_start(1)
 *   This is rank 1's owned bottom (z=1 in rank 1's local grid).
 * rank 1's bottom halo needs global z = z_start(1) - 1 = z_start(0) + local_nz(0) - 1
 *   This is rank 0's owned top (z=local_nz in rank 0's local grid).
 *
 * Sends:
 *   rank sends owned top (st) to rank+1  -> fills rank+1's bottom halo
 *   rank sends owned bottom (sb) to rank-1  -> fills rank-1's top halo
 * Receives:
 *   rank receives into rb from rank+1 (rank+1's owned top -> my bottom halo)
 *   rank receives into rt from rank-1 (rank-1's owned bottom -> my top halo)
 *
 * Wait, rank+1's owned top is NOT my bottom halo. Let me think again.
 *
 * rank r's bottom halo = global z_start(r)-1 = rank (r-1)'s owned top.
 *   rank (r-1) sends its owned top to rank r.
 *   rank r receives from rank (r-1).
 *
 * rank r's top halo = global z_start(r)+local_nz(r) = z_start(r+1) = rank (r+1)'s owned bottom.
 *   rank (r+1) sends its owned bottom to rank r.
 *   rank r receives from rank (r+1).
 *
 * So:
 *   rank sends owned top to rank+1 (fills rank+1's bottom halo)
 *   rank sends owned bottom to rank-1 (fills rank-1's top halo)
 *   rank receives from rank+1 into rb (rank+1's owned bottom -> my bottom halo)
 *   rank receives from rank-1 into rt (rank-1's owned top -> my top halo)
 */
static void exchangeHalos(Real *d_grid, Real *sbuf, Real *rbuf,
                           size_t nx, size_t ny, size_t local_nz,
                           int rank, int nrank)
{
    const size_t sl = nx * ny;
    Real *sb = sbuf, *st = sbuf + sl;  // owned bottom, owned top
    Real *rb = rbuf, *rt = rbuf + sl;  // bottom halo, top halo

    if (local_nz == 0)
    {
        CUDA_CHK(cudaMemcpy(sb, d_grid, sl, cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(st, d_grid + sl, sl, cudaMemcpyDeviceToHost));

        MPI_Request req[4]; int nr = 0;
        if (rank > 0) {
            MPI_Isend(st, static_cast<int>(sl), MPI_DOUBLE, rank-1, 0,
                      MPI_COMM_WORLD, &req[nr++]);
            MPI_Irecv(rb, static_cast<int>(sl), MPI_DOUBLE, rank-1, 0,
                      MPI_COMM_WORLD, &req[nr++]);
        }
        if (rank < nrank - 1) {
            MPI_Isend(sb, static_cast<int>(sl), MPI_DOUBLE, rank+1, 0,
                      MPI_COMM_WORLD, &req[nr++]);
            MPI_Irecv(rt, static_cast<int>(sl), MPI_DOUBLE, rank+1, 0,
                      MPI_COMM_WORLD, &req[nr++]);
        }
        if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);

        if (rank > 0)
            CUDA_CHK(cudaMemcpy(d_grid, rb, sl, cudaMemcpyHostToDevice));
        if (rank < nrank - 1)
            CUDA_CHK(cudaMemcpy(d_grid + sl, rt, sl, cudaMemcpyHostToDevice));
        return;
    }

    /* Copy owned boundary slices to host */
    CUDA_CHK(cudaMemcpy(sb, d_grid + sl, sl, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(st, d_grid + local_nz * sl, sl, cudaMemcpyDeviceToHost));

    MPI_Request req[4]; int nr = 0;

    /* Send owned top to rank+1 (fills rank+1's bottom halo) */
    if (rank < nrank - 1) {
        MPI_Isend(st, static_cast<int>(sl), MPI_DOUBLE, rank+1, 0,
                  MPI_COMM_WORLD, &req[nr++]);
    }
    /* Send owned bottom to rank-1 (fills rank-1's top halo) */
    if (rank > 0) {
        MPI_Isend(sb, static_cast<int>(sl), MPI_DOUBLE, rank-1, 0,
                  MPI_COMM_WORLD, &req[nr++]);
    }
    /* Receive from rank-1 into rb (rank-1's owned top -> my bottom halo) */
    if (rank > 0) {
        MPI_Irecv(rb, static_cast<int>(sl), MPI_DOUBLE, rank-1, 0,
                  MPI_COMM_WORLD, &req[nr++]);
    }
    /* Receive from rank+1 into rt (rank+1's owned bottom -> my top halo) */
    if (rank < nrank - 1) {
        MPI_Irecv(rt, static_cast<int>(sl), MPI_DOUBLE, rank+1, 0,
                  MPI_COMM_WORLD, &req[nr++]);
    }
    if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);

    /* Copy received halos back to device */
    if (rank > 0)
        CUDA_CHK(cudaMemcpy(d_grid, rb, sl, cudaMemcpyHostToDevice));
    if (rank < nrank - 1)
        CUDA_CHK(cudaMemcpy(d_grid + (local_nz + 1) * sl, rt,
                             sl, cudaMemcpyHostToDevice));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    if (rank == 0)
    {
        for (int i = 1; i < argc; ++i)
        {
            if (strcmp(argv[i],"-x")==0 && i+1<argc)       nx   = atoi(argv[++i]);
            else if (strcmp(argv[i],"-y")==0 && i+1<argc)  ny   = atoi(argv[++i]);
            else if (strcmp(argv[i],"-z")==0 && i+1<argc)  nz   = atoi(argv[++i]);
            else if (strcmp(argv[i],"-i")==0 && i+1<argc)  iterations = atoi(argv[++i]);
            else if (strcmp(argv[i],"-v")==0)              validate = true;
            else if (strcmp(argv[i],"-r")==0)              printResults = true;
            else if (strcmp(argv[i],"-h")==0)
            {
                printf("Usage: mpirun -np <N> %s [options]\n", argv[0]);
                printf("  -x|-y|-z <num>  Grid dimensions (default 128)\n");
                printf("  -i <num>        Iterations (default 10)\n");
                printf("  -v              Validate\n");
                printf("  -r              Print results\n");
                printf("  -h              Help\n");
                MPI_Finalize();
                return 0;
            }
            else
            {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    /* Z-domain decomposition */
    const size_t base_nz  = nz / nrank;
    const size_t rem      = nz % nrank;
    const size_t local_nz = base_nz + (rank < static_cast<int>(rem) ? 1 : 0);

    size_t z_start = 0;
    for (int r = 0; r < rank; ++r)
        z_start += base_nz + (r < static_cast<int>(rem) ? 1 : 0);

    const size_t local_nz_h = local_nz + 2;
    const size_t sl         = nx * ny;
    const size_t local_sz   = sl * local_nz_h;

    if (rank == 0)
    {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d  MPI ranks: %d  OMP threads: %d\n",
               iterations, nrank, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Real *sbuf = new Real[2 * sl];
    Real *rbuf = new Real[2 * sl];

    Real *d_g1 = nullptr, *d_g2 = nullptr;
    if (local_sz > 0)
    {
        CUDA_CHK(cudaMalloc(&d_g1, local_sz * sizeof(Real)));
        CUDA_CHK(cudaMalloc(&d_g2, local_sz * sizeof(Real)));
    }

    /* Initialize grid with OpenMP */
    if (rank == 0) printf("Initializing grid...\n");

    Real *h_init = nullptr;
    if (local_sz > 0)
    {
        h_init = new Real[local_sz];

        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < local_nz_h; ++z)
            for (size_t y = 0; y < ny; ++y)
                for (size_t x = 0; x < nx; ++x)
                {
                    long gz = static_cast<long>(z_start) + static_cast<long>(z) - 1;
                    if (gz < 0) gz = 0;
                    if (gz >= static_cast<long>(nz)) gz = static_cast<long>(nz) - 1;
                    h_init[z * sl + y * nx + x] =
                        ((static_cast<size_t>(gz) * sl + y * nx + x) % 19) * 1.0;
                }

        CUDA_CHK(cudaMemcpy(d_g1, h_init, local_sz * sizeof(Real),
                             cudaMemcpyHostToDevice));
        delete[] h_init;
    }

    /* Initial halo exchange */
    if (local_sz > 0)
        exchangeHalos(d_g1, sbuf, rbuf, nx, ny, local_nz, rank, nrank);

    /* Stencil iterations */
    if (rank == 0) printf("Running stencil computation...\n");

    const dim3 blk(8, 8, 4);
    const dim3 grd((nx + 7) / 8, (ny + 7) / 8,
                   (local_nz_h + 3) / 4);

    Real **d_in  = &d_g1;
    Real **d_out = &d_g2;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int it = 0; it < iterations; ++it)
    {
        if (local_sz > 0)
        {
            stencilKernel<<<grd, blk>>>(
                *d_in, *d_out, nx, ny, local_nz, local_nz_h,
                static_cast<int>(z_start), static_cast<int>(nz));

            CUDA_CHK(cudaDeviceSynchronize());

            exchangeHalos(*d_out, sbuf, rbuf, nx, ny, local_nz, rank, nrank);
        }
        std::swap(*d_in, *d_out);
    }

    const double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    const double dur = t1 - t0;
    const double cellUpd =
        static_cast<double>((nx-2)*(ny-2)*(nz-2)) * iterations;
    const double mcups = cellUpd / dur / 1e6;

    if (rank == 0)
    {
        printf("Computation time: %.3f s\n", dur);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    /* Gather owned cells on rank 0 */
    const Real *d_final = (iterations % 2 == 0) ? d_g1 : d_g2;

    Real *h_owned = nullptr;
    if (local_nz > 0)
    {
        h_owned = new Real[local_nz * sl];
        CUDA_CHK(cudaMemcpy(h_owned, d_final + sl,
                             local_nz * sl * sizeof(Real),
                             cudaMemcpyDeviceToHost));
    }

    if (rank == 0)
    {
        std::vector<Real> fullGrid(nx * ny * nz);

        if (local_nz > 0)
            memcpy(fullGrid.data(), h_owned, local_nz * sl * sizeof(Real));

        for (int r = 1; r < nrank; ++r)
        {
            size_t r_ln = base_nz + (r < static_cast<int>(rem) ? 1 : 0);
            if (r_ln == 0) continue;
            size_t r_zs = 0;
            for (int rr = 0; rr < r; ++rr)
                r_zs += base_nz + (rr < static_cast<int>(rem) ? 1 : 0);
            MPI_Recv(fullGrid.data() + r_zs * sl,
                     static_cast<int>(r_ln * sl),
                     MPI_DOUBLE, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (printResults)
            print_results(fullGrid, "Grid");

        if (validate)
        {
            bool valid = true;
            Real gmin = fullGrid[0], gmax = fullGrid[0];
            for (const auto &v : fullGrid)
            {
                if (std::isnan(v) || std::isinf(v))
                { valid = false; break; }
                gmin = std::min(gmin, v);
                gmax = std::max(gmax, v);
            }
            printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
            if (gmax > 1e6 || gmin < -1e6)
            {
                printf("Validation failed: values out of range\n");
                valid = false;
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

            delete[] h_owned; delete[] sbuf; delete[] rbuf;
            if (d_g1) CUDA_CHK(cudaFree(d_g1));
            if (d_g2) CUDA_CHK(cudaFree(d_g2));
            MPI_Finalize();
            return valid ? 0 : 1;
        }

        delete[] h_owned;
    }
    else
    {
        if (local_nz > 0)
            MPI_Send(h_owned, static_cast<int>(local_nz * sl),
                     MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        delete[] h_owned;
    }

    delete[] sbuf;
    delete[] rbuf;
    if (d_g1) CUDA_CHK(cudaFree(d_g1));
    if (d_g2) CUDA_CHK(cudaFree(d_g2));

    MPI_Finalize();
    return 0;
}
