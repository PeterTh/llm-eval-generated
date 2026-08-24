#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

/* ------------------------------------------------------------------ */
/*  Global MPI / decomposition state                                  */
/* ------------------------------------------------------------------ */
static int MPI_RANK, MPI_SIZE;
static int RANK_X, RANK_Y, RANK_Z;
static int NX_RANKS, NY_RANKS, NZ_RANKS;
static size_t NX, NY, NZ;
static size_t LX, LY, LZ;          /* owned cells per rank            */
static size_t GX0, GY0, GZ0;       /* global offset of this rank      */
static MPI_Comm CART_COMM;
static int NB_XM, NB_XP, NB_YM, NB_YP, NB_ZM, NB_ZP;

/* Local index with 1-cell halo in every direction.
 * Valid range: x in [0, LX+1], y in [0, LY+1], z in [0, LZ+1].
 * Owned cells live at x in [1, LX], y in [1, LY], z in [1, LZ].     */
inline size_t lidx(size_t x, size_t y, size_t z) noexcept {
    return z * ((LX + 2) * (LY + 2)) + y * (LX + 2) + x;
}

/* ------------------------------------------------------------------ */
/*  Domain decomposition                                              */
/* ------------------------------------------------------------------ */
static void init_domain() {
    size_t bx = NX / NX_RANKS, rx = NX % NX_RANKS;
    LX  = bx + (RANK_X < (int)rx ? 1 : 0);
    GX0 = bx * RANK_X + std::min((size_t)RANK_X, rx);

    size_t by = NY / NY_RANKS, ry = NY % NY_RANKS;
    LY  = by + (RANK_Y < (int)ry ? 1 : 0);
    GY0 = by * RANK_Y + std::min((size_t)RANK_Y, ry);

    size_t bz = NZ / NZ_RANKS, rz = NZ % NZ_RANKS;
    LZ  = bz + (RANK_Z < (int)rz ? 1 : 0);
    GZ0 = bz * RANK_Z + std::min((size_t)RANK_Z, rz);
}

/* ------------------------------------------------------------------ */
/*  Grid initialisation – each rank fills its owned cells             */
/* ------------------------------------------------------------------ */
static void init_grid(std::vector<Real>& grid) {
    for (size_t lz = 0; lz < LZ; ++lz)
        for (size_t ly = 0; ly < LY; ++ly)
            for (size_t lx = 0; lx < LX; ++lx) {
                size_t gx = GX0 + lx, gy = GY0 + ly, gz = GZ0 + lz;
                size_t gidx = gz * (NX * NY) + gy * NX + gx;
                grid[lidx(lx + 1, ly + 1, lz + 1)] = (gidx % 19) * 1.0;
            }
}

/* ------------------------------------------------------------------ */
/*  Halo exchange (non-blocking, receives posted first to avoid dead) */
/* ------------------------------------------------------------------ */
static void exchange_halos(std::vector<Real>& grid) {
    size_t xh = (LY + 2) * (LZ + 2);
    size_t yh = (LX + 2) * (LZ + 2);
    size_t zh = (LX + 2) * (LY + 2);

    std::vector<Real> xms(xh), xps(xh), xmr(xh), xpr(xh);
    std::vector<Real> yms(yh), yps(yh), ymr(yh), ypr(yh);
    std::vector<Real> zms(zh), zps(zh), zmr(zh), zpr(zh);

    /* Pack send buffers – adjacent owned planes                    */
    for (size_t z = 0; z <= LZ + 1; ++z)
        for (size_t y = 0; y <= LY + 1; ++y) {
            xms[z * (LY + 2) + y] = grid[lidx(1, y, z)];
            xps[z * (LY + 2) + y] = grid[lidx(LX, y, z)];
        }
    for (size_t z = 0; z <= LZ + 1; ++z)
        for (size_t x = 0; x <= LX + 1; ++x) {
            yms[z * (LX + 2) + x] = grid[lidx(x, 1, z)];
            yps[z * (LX + 2) + x] = grid[lidx(x, LY, z)];
        }
    for (size_t y = 0; y <= LY + 1; ++y)
        for (size_t x = 0; x <= LX + 1; ++x) {
            zms[y * (LX + 2) + x] = grid[lidx(x, y, 1)];
            zps[y * (LX + 2) + x] = grid[lidx(x, y, LZ)];
        }

    /* Use MPI_Sendrecv (synchronous send + receive) for each
     * direction pair.  This is deadlock-free because each call
     * completes both the send and the matching receive.           */
    if (NB_XM != MPI_PROC_NULL) {
        MPI_Sendrecv(xms.data(), (int)xh, MPI_DOUBLE, NB_XM, 0,
                     xmr.data(), (int)xh, MPI_DOUBLE, NB_XM, 0,
                     CART_COMM, MPI_STATUS_IGNORE);
    }
    if (NB_XP != MPI_PROC_NULL) {
        MPI_Sendrecv(xps.data(), (int)xh, MPI_DOUBLE, NB_XP, 0,
                     xpr.data(), (int)xh, MPI_DOUBLE, NB_XP, 0,
                     CART_COMM, MPI_STATUS_IGNORE);
    }
    if (NB_YM != MPI_PROC_NULL) {
        MPI_Sendrecv(yms.data(), (int)yh, MPI_DOUBLE, NB_YM, 0,
                     ymr.data(), (int)yh, MPI_DOUBLE, NB_YM, 0,
                     CART_COMM, MPI_STATUS_IGNORE);
    }
    if (NB_YP != MPI_PROC_NULL) {
        MPI_Sendrecv(yps.data(), (int)yh, MPI_DOUBLE, NB_YP, 0,
                     ypr.data(), (int)yh, MPI_DOUBLE, NB_YP, 0,
                     CART_COMM, MPI_STATUS_IGNORE);
    }
    if (NB_ZM != MPI_PROC_NULL) {
        MPI_Sendrecv(zms.data(), (int)zh, MPI_DOUBLE, NB_ZM, 0,
                     zmr.data(), (int)zh, MPI_DOUBLE, NB_ZM, 0,
                     CART_COMM, MPI_STATUS_IGNORE);
    }
    if (NB_ZP != MPI_PROC_NULL) {
        MPI_Sendrecv(zps.data(), (int)zh, MPI_DOUBLE, NB_ZP, 0,
                     zpr.data(), (int)zh, MPI_DOUBLE, NB_ZP, 0,
                     CART_COMM, MPI_STATUS_IGNORE);
    }

    /* Unpack receive buffers into halo cells                       */
    for (size_t z = 0; z <= LZ + 1; ++z)
        for (size_t y = 0; y <= LY + 1; ++y) {
            grid[lidx(0, y, z)]     = xmr[z * (LY + 2) + y];
            grid[lidx(LX + 1, y, z)] = xpr[z * (LY + 2) + y];
        }
    for (size_t z = 0; z <= LZ + 1; ++z)
        for (size_t x = 0; x <= LX + 1; ++x) {
            grid[lidx(x, 0, z)]     = ymr[z * (LX + 2) + x];
            grid[lidx(x, LY + 1, z)] = ypr[z * (LX + 2) + x];
        }
    for (size_t y = 0; y <= LY + 1; ++y)
        for (size_t x = 0; x <= LX + 1; ++x) {
            grid[lidx(x, y, 0)]     = zmr[y * (LX + 2) + x];
            grid[lidx(x, y, LZ + 1)] = zpr[y * (LX + 2) + x];
        }
}

/* ------------------------------------------------------------------ */
/*  7-point stencil on owned cells                                    */
/* ------------------------------------------------------------------ */
static void stencil_iteration(const std::vector<Real>& input,
                              std::vector<Real>& output) {
    for (size_t lz = 1; lz <= LZ; ++lz) {
        size_t gz = GZ0 + (lz - 1);
        bool gz_bnd = (gz == 0 || gz == NZ - 1);
        for (size_t ly = 1; ly <= LY; ++ly) {
            size_t gy = GY0 + (ly - 1);
            bool gy_bnd = (gy == 0 || gy == NY - 1);
            for (size_t lx = 1; lx <= LX; ++lx) {
                size_t gx = GX0 + (lx - 1);
                size_t l  = lidx(lx, ly, lz);

                if (!gz_bnd && !gy_bnd && gx > 0 && gx < NX - 1) {
                    /* 7-point averaging stencil */
                    output[l] = (input[l] +
                                 input[lidx(lx - 1, ly, lz)] +
                                 input[lidx(lx + 1, ly, lz)] +
                                 input[lidx(lx, ly - 1, lz)] +
                                 input[lidx(lx, ly + 1, lz)] +
                                 input[lidx(lx, ly, lz - 1)] +
                                 input[lidx(lx, ly, lz + 1)]) / 7.0;
                } else {
                    /* Boundary cell – copy from input */
                    output[l] = input[l];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Validation (runs on rank 0 with the assembled full grid)          */
/* ------------------------------------------------------------------ */
static bool validate_result(const std::vector<Real>& grid,
                            [[maybe_unused]] size_t nx,
                            [[maybe_unused]] size_t ny,
                            [[maybe_unused]] size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0], maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);

    /* ---- parse arguments on rank 0, broadcast to all ranks ---- */
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    if (MPI_RANK == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0) validate = true;
            else if (strcmp(argv[i], "-r") == 0) printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]); MPI_Finalize(); return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                print_usage(argv[0]); MPI_Finalize(); return 1;
            }
        }
    }

    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    NX = nx; NY = ny; NZ = nz;

    /* ---- 3-D Cartesian topology ---- */
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(MPI_SIZE, 3, dims);
    NX_RANKS = dims[0]; NY_RANKS = dims[1]; NZ_RANKS = dims[2];

    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &CART_COMM);

    int coords[3];
    MPI_Cart_coords(CART_COMM, MPI_RANK, 3, coords);
    RANK_X = coords[0]; RANK_Y = coords[1]; RANK_Z = coords[2];

    MPI_Cart_shift(CART_COMM, 0, 1, &NB_XM, &NB_XP);
    MPI_Cart_shift(CART_COMM, 1, 1, &NB_YM, &NB_YP);
    MPI_Cart_shift(CART_COMM, 2, 1, &NB_ZM, &NB_ZP);

    init_domain();

    if (MPI_RANK == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", NX, NY, NZ);
        printf("MPI ranks: %d x %d x %d (total: %d)\n",
               NX_RANKS, NY_RANKS, NZ_RANKS, MPI_SIZE);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- allocate local grids (with halo) ---- */
    size_t localSize = (LX + 2) * (LY + 2) * (LZ + 2);
    std::vector<Real> grid1(localSize), grid2(localSize);

    if (MPI_RANK == 0) printf("Initializing grid...\n");
    init_grid(grid1);
    exchange_halos(grid1);          /* seed halo cells                */

    /* ---- timed stencil loop ---- */
    if (MPI_RANK == 0) printf("Running stencil computation...\n");

    MPI_Barrier(CART_COMM);
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchange_halos(grid1);
            stencil_iteration(grid1, grid2);
        } else {
            exchange_halos(grid2);
            stencil_iteration(grid2, grid1);
        }
    }

    double t1 = MPI_Wtime();
    MPI_Barrier(CART_COMM);
    double localDuration = t1 - t0;
    double duration;
    MPI_Allreduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, CART_COMM);

    if (MPI_RANK == 0) {
        printf("Computation time: %.3f s\n", duration);
        double cellUpdates = (double)((NX - 2) * (NY - 2) * (NZ - 2)) * iterations;
        double mcups = cellUpdates / duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    /* ---- gather owned cells to rank 0 ---- */
    const std::vector<Real>* finalGrid =
        (iterations % 2 == 0) ? &grid1 : &grid2;

    /* Compute per-rank owned dimensions and global offsets */
    std::vector<int> sendcounts(MPI_SIZE);
    std::vector<size_t> rank_gx0(MPI_SIZE), rank_gy0(MPI_SIZE), rank_gz0(MPI_SIZE);
    std::vector<size_t> rank_lx(MPI_SIZE), rank_ly(MPI_SIZE), rank_lz(MPI_SIZE);

    for (int rz = 0; rz < NZ_RANKS; ++rz) {
        size_t lz_v  = NZ / NZ_RANKS + (rz < (int)(NZ % NZ_RANKS) ? 1 : 0);
        size_t gz0_v = (NZ / NZ_RANKS) * rz + std::min((size_t)rz, NZ % NZ_RANKS);
        for (int ry = 0; ry < NY_RANKS; ++ry) {
            size_t ly_v  = NY / NY_RANKS + (ry < (int)(NY % NY_RANKS) ? 1 : 0);
            size_t gy0_v = (NY / NY_RANKS) * ry + std::min((size_t)ry, NY % NY_RANKS);
            for (int rx = 0; rx < NX_RANKS; ++rx) {
                size_t lx_v  = NX / NX_RANKS + (rx < (int)(NX % NX_RANKS) ? 1 : 0);
                size_t gx0_v = (NX / NX_RANKS) * rx + std::min((size_t)rx, NX % NX_RANKS);
                int rank;
                int rc[3] = {rx, ry, rz};
                MPI_Cart_rank(CART_COMM, rc, &rank);
                sendcounts[rank] = (int)(lx_v * ly_v * lz_v);
                rank_gx0[rank] = gx0_v; rank_gy0[rank] = gy0_v; rank_gz0[rank] = gz0_v;
                rank_lx[rank]  = lx_v;  rank_ly[rank]  = ly_v;  rank_lz[rank]  = lz_v;
            }
        }
    }

    /* Extract owned cells (no halo) into contiguous buffer */
    size_t ownedSize = LX * LY * LZ;
    std::vector<Real> owned(ownedSize);
    for (size_t lz = 0; lz < LZ; ++lz)
        for (size_t ly = 0; ly < LY; ++ly)
            for (size_t lx = 0; lx < LX; ++lx)
                owned[lz * (LX * LY) + ly * LX + lx] =
                    (*finalGrid)[lidx(lx + 1, ly + 1, lz + 1)];

    /* Gather into a temporary contiguous buffer, then scatter into
     * the correct global positions (owned cells are not contiguous
     * in the global array when multiple ranks exist).               */
    int totalOwned = 0;
    for (int i = 0; i < MPI_SIZE; ++i) totalOwned += sendcounts[i];
    std::vector<int> gatherDispls(MPI_SIZE);
    gatherDispls[0] = 0;
    for (int i = 1; i < MPI_SIZE; ++i)
        gatherDispls[i] = gatherDispls[i - 1] + sendcounts[i - 1];

    std::vector<Real> gathered;
    if (MPI_RANK == 0) gathered.resize(totalOwned);

    MPI_Gatherv(owned.data(), (int)ownedSize, MPI_DOUBLE,
                gathered.data(), sendcounts.data(), gatherDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    std::vector<Real> fullGrid;
    if (MPI_RANK == 0) {
        fullGrid.resize(NX * NY * NZ);
        for (int rank = 0; rank < MPI_SIZE; ++rank) {
            int offset = gatherDispls[rank];
            for (size_t lz = 0; lz < rank_lz[rank]; ++lz)
                for (size_t ly = 0; ly < rank_ly[rank]; ++ly)
                    for (size_t lx = 0; lx < rank_lx[rank]; ++lx) {
                        size_t gx = rank_gx0[rank] + lx;
                        size_t gy = rank_gy0[rank] + ly;
                        size_t gz = rank_gz0[rank] + lz;
                        fullGrid[gz * (NX * NY) + gy * NX + gx] = gathered[offset++];
                    }
        }
    }

    /* ---- results & validation on rank 0 ---- */
    if (MPI_RANK == 0) {
        if (printResults) print_results(fullGrid, "Grid");

        if (validate) {
            printf("Validating result...\n");
            if (validate_result(fullGrid, NX, NY, NZ))
                printf("Validation: PASSED\n");
            else
                printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
