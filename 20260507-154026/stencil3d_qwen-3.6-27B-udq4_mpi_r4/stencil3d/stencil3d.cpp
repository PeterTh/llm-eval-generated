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
/*  MPI global state                                                   */
/* ------------------------------------------------------------------ */
static int MPI_RANK, MPI_SIZE;
static int MPI_PX, MPI_PY, MPI_PZ;   // process-grid dimensions
static int MPI_PI, MPI_PJ, MPI_PK;   // this rank's coordinates

/* ------------------------------------------------------------------ */
/*  3-D index helpers                                                   */
/* ------------------------------------------------------------------ */
inline constexpr size_t idx3(size_t x, size_t y, size_t z,
                             size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

/* ------------------------------------------------------------------ */
/*  Domain decomposition                                                */
/* ------------------------------------------------------------------ */
struct Decomp {
    size_t gx0, gx1; // global x range [gx0, gx1)
    size_t gy0, gy1; // global y range [gy0, gy1)
    size_t gz0, gz1; // global z range [gz0, gz1)
    size_t lx, ly, lz;   // local dims including 1-cell halo
    size_t lix, liy, liz; // local interior dims (no halo)
};

static void compute_decomp(size_t nx, size_t ny, size_t nz,
                           int px, int py, int pz,
                           int pi, int pj, int pk,
                           Decomp& d) {
    d.gx0 = (nx * pi) / px;
    d.gx1 = (nx * (pi + 1)) / px;
    d.gy0 = (ny * pj) / py;
    d.gy1 = (ny * (pj + 1)) / py;
    d.gz0 = (nz * pk) / pz;
    d.gz1 = (nz * (pk + 1)) / pz;
    d.lix = d.gx1 - d.gx0;
    d.liy = d.gy1 - d.gy0;
    d.liz = d.gz1 - d.gz0;
    d.lx = d.lix + 2;
    d.ly = d.liy + 2;
    d.lz = d.liz + 2;
}

// map (pi,pj,pk) -> linear rank
static inline int rank_of(int pi, int pj, int pk, int px, int py) {
    return pk * (px * py) + pj * px + pi;
}

// find the 3-D process grid that gives the most balanced sub-domains
static void find_process_grid(int np, size_t nx, size_t ny, size_t nz,
                              int& px, int& py, int& pz) {
    if (np == 1) { px = py = pz = 1; return; }

    double best = -1.0;
    for (int a = 1; a <= np; ++a) {
        if (np % a != 0) continue;
        for (int b = 1; b <= np / a; ++b) {
            if ((np / a) % b != 0) continue;
            int c = np / (a * b);
            // try all 6 permutations of (a,b,c) -> (px,py,pz)
            int p[6][3] = {
                {a,b,c},{a,c,b},{b,a,c},{b,c,a},{c,a,b},{c,b,a}
            };
            for (int k = 0; k < 6; ++k) {
                double sx = nx / (double)p[k][0];
                double sy = ny / (double)p[k][1];
                double sz = nz / (double)p[k][2];
                double score = std::min({sx, sy, sz});
                if (score > best) {
                    best = score;
                    px = p[k][0]; py = p[k][1]; pz = p[k][2];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Initialise the local sub-domain                                     */
/* ------------------------------------------------------------------ */
static void init_local(std::vector<Real>& g, const Decomp& d,
                       size_t nx, size_t ny) {
    for (size_t iz = 0; iz < d.liz; ++iz)
        for (size_t iy = 0; iy < d.liy; ++iy)
            for (size_t ix = 0; ix < d.lix; ++ix) {
                size_t gi = idx3(d.gx0 + ix, d.gy0 + iy, d.gz0 + iz, nx, ny);
                g[idx3(ix + 1, iy + 1, iz + 1, d.lx, d.ly)] = (gi % 19) * 1.0;
            }
}

/* ------------------------------------------------------------------ */
/*  7-point stencil on the local sub-domain                             */
/* ------------------------------------------------------------------ */
static void stencil_local(const std::vector<Real>& in,
                          std::vector<Real>& out,
                          const Decomp& d,
                          size_t nx, size_t ny, size_t nz) {
    // tighten loop bounds so we never touch global boundaries
    size_t ix0 = 1, ix1 = d.lx - 1;
    if (d.gx0 == 0)      ix0 = 2;
    if (d.gx1 == nx)     ix1 = d.lx - 2;

    size_t iy0 = 1, iy1 = d.ly - 1;
    if (d.gy0 == 0)      iy0 = 2;
    if (d.gy1 == ny)     iy1 = d.ly - 2;

    size_t iz0 = 1, iz1 = d.lz - 1;
    if (d.gz0 == 0)      iz0 = 2;
    if (d.gz1 == nz)     iz1 = d.lz - 2;

    // interior stencil
    for (size_t iz = iz0; iz < iz1; ++iz)
        for (size_t iy = iy0; iy < iy1; ++iy)
            for (size_t ix = ix0; ix < ix1; ++ix) {
                size_t i = idx3(ix, iy, iz, d.lx, d.ly);
                out[i] = (in[i]
                        + in[idx3(ix-1, iy,   iz,   d.lx, d.ly)]
                        + in[idx3(ix+1, iy,   iz,   d.lx, d.ly)]
                        + in[idx3(ix,   iy-1, iz,   d.lx, d.ly)]
                        + in[idx3(ix,   iy+1, iz,   d.lx, d.ly)]
                        + in[idx3(ix,   iy,   iz-1, d.lx, d.ly)]
                        + in[idx3(ix,   iy,   iz+1, d.lx, d.ly)]) / 7.0;
            }

    // copy global-boundary values (only faces that belong to this rank)
    if (d.gx0 == 0)
        for (size_t iz = 0; iz < d.liz; ++iz)
            for (size_t iy = 0; iy < d.liy; ++iy) {
                size_t i = idx3(1, iy + 1, iz + 1, d.lx, d.ly);
                out[i] = in[i];
            }
    if (d.gx1 == nx)
        for (size_t iz = 0; iz < d.liz; ++iz)
            for (size_t iy = 0; iy < d.liy; ++iy) {
                size_t i = idx3(d.lx - 2, iy + 1, iz + 1, d.lx, d.ly);
                out[i] = in[i];
            }
    if (d.gy0 == 0)
        for (size_t iz = 0; iz < d.liz; ++iz)
            for (size_t ix = 0; ix < d.lix; ++ix) {
                size_t i = idx3(ix + 1, 1, iz + 1, d.lx, d.ly);
                out[i] = in[i];
            }
    if (d.gy1 == ny)
        for (size_t iz = 0; iz < d.liz; ++iz)
            for (size_t ix = 0; ix < d.lix; ++ix) {
                size_t i = idx3(ix + 1, d.ly - 2, iz + 1, d.lx, d.ly);
                out[i] = in[i];
            }
    if (d.gz0 == 0)
        for (size_t iy = 0; iy < d.liy; ++iy)
            for (size_t ix = 0; ix < d.lix; ++ix) {
                size_t i = idx3(ix + 1, iy + 1, 1, d.lx, d.ly);
                out[i] = in[i];
            }
    if (d.gz1 == nz)
        for (size_t iy = 0; iy < d.liy; ++iy)
            for (size_t ix = 0; ix < d.lix; ++ix) {
                size_t i = idx3(ix + 1, iy + 1, d.lz - 2, d.lx, d.ly);
                out[i] = in[i];
            }
}

/* ------------------------------------------------------------------ */
/*  Halo exchange with all 6 neighbours                                 */
/* ------------------------------------------------------------------ */
static void exchange_halos(std::vector<Real>& g, const Decomp& d) {
    size_t lx = d.lx, ly = d.ly, lz = d.lz;

    int left   = (MPI_PI > 0)        ? rank_of(MPI_PI-1, MPI_PJ, MPI_PK, MPI_PX, MPI_PY) : MPI_PROC_NULL;
    int right  = (MPI_PI < MPI_PX-1) ? rank_of(MPI_PI+1, MPI_PJ, MPI_PK, MPI_PX, MPI_PY) : MPI_PROC_NULL;
    int front  = (MPI_PJ > 0)        ? rank_of(MPI_PI, MPI_PJ-1, MPI_PK, MPI_PX, MPI_PY) : MPI_PROC_NULL;
    int back   = (MPI_PJ < MPI_PY-1) ? rank_of(MPI_PI, MPI_PJ+1, MPI_PK, MPI_PX, MPI_PY) : MPI_PROC_NULL;
    int bottom = (MPI_PK > 0)        ? rank_of(MPI_PI, MPI_PJ, MPI_PK-1, MPI_PX, MPI_PY) : MPI_PROC_NULL;
    int top    = (MPI_PK < MPI_PZ-1) ? rank_of(MPI_PI, MPI_PJ, MPI_PK+1, MPI_PX, MPI_PY) : MPI_PROC_NULL;

    size_t sz_x = ly * lz;
    size_t sz_y = lx * lz;
    size_t sz_z = lx * ly;

    // Separate buffers for each face
    std::vector<Real> snd_l(sz_x), snd_r(sz_x), rcv_l(sz_x), rcv_r(sz_x);
    std::vector<Real> snd_f(sz_y), snd_b(sz_y), rcv_f(sz_y), rcv_b(sz_y);
    std::vector<Real> snd_zb(sz_z), snd_zt(sz_z), rcv_zb(sz_z), rcv_zt(sz_z);

    // Pack X faces
    for (size_t iz = 0; iz < lz; ++iz)
        for (size_t iy = 0; iy < ly; ++iy) {
            snd_l[iz * ly + iy] = g[idx3(1,      iy, iz, lx, ly)];
            snd_r[iz * ly + iy] = g[idx3(lx - 2, iy, iz, lx, ly)];
        }
    // Pack Y faces
    for (size_t iz = 0; iz < lz; ++iz)
        for (size_t ix = 0; ix < lx; ++ix) {
            snd_f[iz * lx + ix] = g[idx3(ix, 1,      iz, lx, ly)];
            snd_b[iz * lx + ix] = g[idx3(ix, ly - 2, iz, lx, ly)];
        }
    // Pack Z faces
    for (size_t iy = 0; iy < ly; ++iy)
        for (size_t ix = 0; ix < lx; ++ix) {
            snd_zb[iy * lx + ix] = g[idx3(ix, iy, 1,      lx, ly)];
            snd_zt[iy * lx + ix] = g[idx3(ix, iy, lz - 2, lx, ly)];
        }

    // Post all 12 non-blocking operations
    int nreq = 0;
    MPI_Request reqs[12];

    if (left != MPI_PROC_NULL) {
        MPI_Isend(snd_l.data(), sz_x, MPI_DOUBLE, left,  11, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rcv_l.data(), sz_x, MPI_DOUBLE, left,  10, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (right != MPI_PROC_NULL) {
        MPI_Isend(snd_r.data(), sz_x, MPI_DOUBLE, right, 10, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rcv_r.data(), sz_x, MPI_DOUBLE, right, 11, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (front != MPI_PROC_NULL) {
        MPI_Isend(snd_f.data(), sz_y, MPI_DOUBLE, front, 21, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rcv_f.data(), sz_y, MPI_DOUBLE, front, 20, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (back != MPI_PROC_NULL) {
        MPI_Isend(snd_b.data(), sz_y, MPI_DOUBLE, back,  20, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rcv_b.data(), sz_y, MPI_DOUBLE, back,  21, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (bottom != MPI_PROC_NULL) {
        MPI_Isend(snd_zb.data(), sz_z, MPI_DOUBLE, bottom, 31, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rcv_zb.data(), sz_z, MPI_DOUBLE, bottom, 30, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (top != MPI_PROC_NULL) {
        MPI_Isend(snd_zt.data(), sz_z, MPI_DOUBLE, top,   30, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rcv_zt.data(), sz_z, MPI_DOUBLE, top,   31, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    if (nreq > 0)
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Unpack X
    for (size_t iz = 0; iz < lz; ++iz)
        for (size_t iy = 0; iy < ly; ++iy) {
            if (left  != MPI_PROC_NULL) g[idx3(0,      iy, iz, lx, ly)] = rcv_l[iz * ly + iy];
            if (right != MPI_PROC_NULL) g[idx3(lx - 1, iy, iz, lx, ly)] = rcv_r[iz * ly + iy];
        }
    // Unpack Y
    for (size_t iz = 0; iz < lz; ++iz)
        for (size_t ix = 0; ix < lx; ++ix) {
            if (front != MPI_PROC_NULL) g[idx3(ix, 0,      iz, lx, ly)] = rcv_f[iz * lx + ix];
            if (back  != MPI_PROC_NULL) g[idx3(ix, ly - 1, iz, lx, ly)] = rcv_b[iz * lx + ix];
        }
    // Unpack Z
    for (size_t iy = 0; iy < ly; ++iy)
        for (size_t ix = 0; ix < lx; ++ix) {
            if (bottom != MPI_PROC_NULL) g[idx3(ix, iy, 0,      lx, ly)] = rcv_zb[iy * lx + ix];
            if (top    != MPI_PROC_NULL) g[idx3(ix, iy, lz - 1, lx, ly)] = rcv_zt[iy * lx + ix];
        }
}

/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);

    /* ---- parse arguments (identical on every rank) ---- */
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)       nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)  ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)  nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)  iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)                  validate = true;
        else if (strcmp(argv[i], "-r") == 0)                  printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (MPI_RANK == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -x <num>  Grid size in X dimension (default: 128)\n");
                printf("  -y <num>  Grid size in Y dimension (default: same as X)\n");
                printf("  -z <num>  Grid size in Z dimension (default: same as X)\n");
                printf("  -i <num>  Number of iterations (default: 10)\n");
                printf("  -v        Enable validation\n");
                printf("  -r        Print results for external validation\n");
                printf("  -h        Show this help message\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (MPI_RANK == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printf("Use -h for help.\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (MPI_RANK == 0) {
        printf("3D Stencil Benchmark\n");
        printf("MPI processes: %d\n", MPI_SIZE);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    /* ---- domain decomposition ---- */
    find_process_grid(MPI_SIZE, nx, ny, nz, MPI_PX, MPI_PY, MPI_PZ);
    MPI_PI = MPI_RANK % MPI_PX;
    MPI_PJ = (MPI_RANK / MPI_PX) % MPI_PY;
    MPI_PK = MPI_RANK / (MPI_PX * MPI_PY);

    Decomp d;
    compute_decomp(nx, ny, nz, MPI_PX, MPI_PY, MPI_PZ,
                   MPI_PI, MPI_PJ, MPI_PK, d);

    if (MPI_RANK == 0)
        printf("Process grid: %d x %d x %d\n", MPI_PX, MPI_PY, MPI_PZ);

    /* ---- allocate local grids (with halo) ---- */
    size_t local_size = d.lx * d.ly * d.lz;
    std::vector<Real> g1(local_size, 0.0);
    std::vector<Real> g2(local_size, 0.0);

    /* ---- initialise ---- */
    if (MPI_RANK == 0) printf("Initializing grid...\n");
    init_local(g1, d, nx, ny);

    /* ---- timed stencil loop ---- */
    if (MPI_RANK == 0) printf("Running stencil computation...\n");
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchange_halos(g1, d);
            stencil_local(g1, g2, d, nx, ny, nz);
        } else {
            exchange_halos(g2, d);
            stencil_local(g2, g1, d, nx, ny, nz);
        }
    }

    double t1 = MPI_Wtime();
    double dur_ms = (t1 - t0) * 1000.0;
    double max_dur_ms;
    MPI_Allreduce(&dur_ms, &max_dur_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double cellUpdates =
        static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (max_dur_ms / 1000.0) / 1e6;

    if (MPI_RANK == 0) {
        printf("Computation time: %.0f ms\n", max_dur_ms);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? g1 : g2;

    /* ---- gather, validate, print ---- */
    if (printResults || validate) {
        // build Gatherv descriptors (flat buffer, not scattered)
        std::vector<int> recvcounts(MPI_SIZE);
        std::vector<int> displs(MPI_SIZE);
        size_t total_interior = 0;
        for (int r = 0; r < MPI_SIZE; ++r) {
            int rpi = r % MPI_PX;
            int rpj = (r / MPI_PX) % MPI_PY;
            int rpk = r / (MPI_PX * MPI_PY);
            Decomp rd;
            compute_decomp(nx, ny, nz, MPI_PX, MPI_PY, MPI_PZ,
                           rpi, rpj, rpk, rd);
            recvcounts[r] = static_cast<int>(rd.lix * rd.liy * rd.liz);
            displs[r] = static_cast<int>(total_interior);
            total_interior += rd.lix * rd.liy * rd.liz;
        }

        // pack local interior into contiguous buffer
        size_t local_count = d.lix * d.liy * d.liz;
        std::vector<Real> local_data(local_count);
        for (size_t iz = 0; iz < d.liz; ++iz)
            for (size_t iy = 0; iy < d.liy; ++iy)
                for (size_t ix = 0; ix < d.lix; ++ix) {
                    local_data[iz * d.liy * d.lix + iy * d.lix + ix] =
                        finalGrid[idx3(ix + 1, iy + 1, iz + 1, d.lx, d.ly)];
                }

        // gather to rank 0 (flat buffer)
        std::vector<Real> full_grid;
        std::vector<Real> flat_data;
        if (MPI_RANK == 0) {
            flat_data.resize(total_interior);
            full_grid.resize(nx * ny * nz);
        }

        MPI_Gatherv(local_data.data(),
                    static_cast<int>(local_count), MPI_DOUBLE,
                    flat_data.data(),
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // reconstruct global grid from flat buffer (rank 0 only)
        if (MPI_RANK == 0) {
            size_t offset = 0;
            for (int r = 0; r < MPI_SIZE; ++r) {
                int rpi = r % MPI_PX;
                int rpj = (r / MPI_PX) % MPI_PY;
                int rpk = r / (MPI_PX * MPI_PY);
                Decomp rd;
                compute_decomp(nx, ny, nz, MPI_PX, MPI_PY, MPI_PZ,
                               rpi, rpj, rpk, rd);
                size_t cnt = rd.lix * rd.liy * rd.liz;
                for (size_t iz = 0; iz < rd.liz; ++iz)
                    for (size_t iy = 0; iy < rd.liy; ++iy)
                        for (size_t ix = 0; ix < rd.lix; ++ix) {
                            size_t gi = idx3(rd.gx0 + ix, rd.gy0 + iy, rd.gz0 + iz, nx, ny);
                            full_grid[gi] = flat_data[offset + iz * rd.liy * rd.lix + iy * rd.lix + ix];
                        }
                offset += cnt;
            }
        }

        if (MPI_RANK == 0) {
            if (printResults) print_results(full_grid, "Grid");

            if (validate) {
                bool ok = true;
                Real minVal = full_grid[0], maxVal = full_grid[0];
                for (const auto& val : full_grid) {
                    if (std::isnan(val) || std::isinf(val)) {
                        printf("Validation failed: found NaN or Inf value\n");
                        ok = false;
                        break;
                    }
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
                if (ok) {
                    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                    if (maxVal > 1e6 || minVal < -1e6) {
                        printf("Validation failed: values out of expected range\n");
                        ok = false;
                    }
                }
                printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
