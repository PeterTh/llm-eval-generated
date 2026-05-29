#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Domain decomposition metadata
// ---------------------------------------------------------------------------
struct DomainInfo {
    int rank = 0, size = 1;
    size_t nx = 0, ny = 0, nz = 0;
    int nx_proc = 1, ny_proc = 1, nz_proc = 1;
    int my_rank_x = 0, my_rank_y = 0, my_rank_z = 0;
    size_t local_nx = 0, local_ny = 0, local_nz = 0;
    size_t global_start_x = 0, global_start_y = 0, global_start_z = 0;
    size_t lx = 0, ly = 0, lz = 0;          // local dims + 2 halo layers
    int left_x = MPI_PROC_NULL, right_x = MPI_PROC_NULL;
    int bottom_y = MPI_PROC_NULL, top_y = MPI_PROC_NULL;
    int back_z = MPI_PROC_NULL, front_z = MPI_PROC_NULL;
};

// ---------------------------------------------------------------------------
// 3D index (local layout with halos)
// ---------------------------------------------------------------------------
inline constexpr size_t idx3l(size_t x, size_t y, size_t z,
                              size_t lx, size_t ly) noexcept {
    return z * (ly * lx) + y * lx + x;
}

// ---------------------------------------------------------------------------
// Distribute `total` items among `n_procs` processes as evenly as possible
// ---------------------------------------------------------------------------
static void distribute(size_t total, int n_procs, int my_rank,
                       size_t& local_size, size_t& global_start) {
    size_t base = total / n_procs;
    size_t rem  = total % n_procs;
    if (my_rank < static_cast<int>(rem)) {
        local_size   = base + 1;
        global_start = my_rank * (base + 1);
    } else {
        local_size   = base;
        global_start = rem * (base + 1) +
                       static_cast<size_t>(my_rank - rem) * base;
    }
}

// ---------------------------------------------------------------------------
// Build the full DomainInfo for a given rank
// ---------------------------------------------------------------------------
static void computeDomainInfo(int rank, int size,
                              size_t nx, size_t ny, size_t nz,
                              DomainInfo& info) {
    info.rank = rank;
    info.size = size;
    info.nx   = nx;
    info.ny   = ny;
    info.nz   = nz;

    // 3-D decomposition  (nx_proc x ny_proc x nz_proc ≈ size)
    info.nz_proc = static_cast<int>(std::round(std::cbrt(static_cast<double>(size))));
    while (size % info.nz_proc != 0 && info.nz_proc > 1) --info.nz_proc;
    int remaining = size / info.nz_proc;
    info.ny_proc = static_cast<int>(std::round(std::sqrt(static_cast<double>(remaining))));
    while (remaining % info.ny_proc != 0 && info.ny_proc > 1) --info.ny_proc;
    info.nx_proc = remaining / info.ny_proc;

    info.my_rank_z = rank / (info.nx_proc * info.ny_proc);
    int rem2 = rank % (info.nx_proc * info.ny_proc);
    info.my_rank_y = rem2 / info.nx_proc;
    info.my_rank_x = rem2 % info.nx_proc;

    distribute(nx, info.nx_proc, info.my_rank_x, info.local_nx, info.global_start_x);
    distribute(ny, info.ny_proc, info.my_rank_y, info.local_ny, info.global_start_y);
    distribute(nz, info.nz_proc, info.my_rank_z, info.local_nz, info.global_start_z);

    info.lx = info.local_nx + 2;
    info.ly = info.local_ny + 2;
    info.lz = info.local_nz + 2;

    info.left_x   = (info.my_rank_x > 0)                   ? rank - 1 : MPI_PROC_NULL;
    info.right_x  = (info.my_rank_x < info.nx_proc - 1)    ? rank + 1 : MPI_PROC_NULL;
    info.bottom_y = (info.my_rank_y > 0)                   ? rank - info.nx_proc : MPI_PROC_NULL;
    info.top_y    = (info.my_rank_y < info.ny_proc - 1)    ? rank + info.nx_proc : MPI_PROC_NULL;
    info.back_z   = (info.my_rank_z > 0)                   ? rank - info.nx_proc * info.ny_proc : MPI_PROC_NULL;
    info.front_z  = (info.my_rank_z < info.nz_proc - 1)    ? rank + info.nx_proc * info.ny_proc : MPI_PROC_NULL;
}

// ---------------------------------------------------------------------------
// Non-blocking halo exchange for all 6 faces
// ---------------------------------------------------------------------------
static void exchangeHalos(std::vector<double>& data, const DomainInfo& info,
                          std::vector<double>& s_x, std::vector<double>& r_x,
                          std::vector<double>& s_xl, std::vector<double>& r_xr,
                          std::vector<double>& s_y, std::vector<double>& r_y,
                          std::vector<double>& s_yb, std::vector<double>& r_yt,
                          std::vector<double>& s_z, std::vector<double>& r_z,
                          std::vector<double>& s_zb, std::vector<double>& r_zf) {
    const size_t lx  = info.lx, ly  = info.ly, lz  = info.lz;
    const size_t lnx = info.local_nx, lny = info.local_ny, lnz = info.local_nz;

    const size_t xc = lny * lnz;
    const size_t yc = lnx * lnz;
    const size_t zc = lnx * lny;

    // --- pack send buffers ---
    for (size_t z = 1; z <= lnz; ++z)
        for (size_t y = 1; y <= lny; ++y) {
            size_t o = (z - 1) * lny + (y - 1);
            s_x[o]  = data[idx3l(lnx, y, z, lx, ly)];   // rightmost owned
            s_xl[o] = data[idx3l(1,   y, z, lx, ly)];   // leftmost owned
        }
    for (size_t z = 1; z <= lnz; ++z)
        for (size_t x = 1; x <= lnx; ++x) {
            size_t o = (z - 1) * lnx + (x - 1);
            s_y[o]  = data[idx3l(x, lny, z, lx, ly)];   // topmost owned
            s_yb[o] = data[idx3l(x, 1,   z, lx, ly)];   // bottommost owned
        }
    for (size_t y = 1; y <= lny; ++y)
        for (size_t x = 1; x <= lnx; ++x) {
            size_t o = (y - 1) * lnx + (x - 1);
            s_z[o]  = data[idx3l(x, y, lnz, lx, ly)];   // frontmost owned
            s_zb[o] = data[idx3l(x, y, 1,   lx, ly)];   // backmost owned
        }

    // --- post non-blocking communication ---
    std::vector<MPI_Request> reqs;
    MPI_Request r;
    auto is = [&](const double* b, size_t c, int to, int tag) {
        if (to != MPI_PROC_NULL) {
            MPI_Isend(b, c, MPI_DOUBLE, to, tag, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    };
    auto ir = [&](double* b, size_t c, int from, int tag) {
        if (from != MPI_PROC_NULL) {
            MPI_Irecv(b, c, MPI_DOUBLE, from, tag, MPI_COMM_WORLD, &r);
            reqs.push_back(r);
        }
    };

    // X
    is(s_x.data(),  xc, info.right_x,  10);
    ir(r_x.data(),  xc, info.left_x,   10);
    is(s_xl.data(), xc, info.left_x,   11);
    ir(r_xr.data(), xc, info.right_x,  11);
    // Y
    is(s_y.data(),  yc, info.top_y,    20);
    ir(r_y.data(),  yc, info.bottom_y, 20);
    is(s_yb.data(), yc, info.bottom_y, 21);
    ir(r_yt.data(), yc, info.top_y,    21);
    // Z
    is(s_z.data(),  zc, info.front_z,  30);
    ir(r_z.data(),  zc, info.back_z,   30);
    is(s_zb.data(), zc, info.back_z,   31);
    ir(r_zf.data(), zc, info.front_z,  31);

    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    // --- unpack receive buffers into halos ---
    for (size_t z = 1; z <= lnz; ++z)
        for (size_t y = 1; y <= lny; ++y) {
            size_t o = (z - 1) * lny + (y - 1);
            data[idx3l(0,     y, z, lx, ly)] = r_x[o];
            data[idx3l(lx - 1, y, z, lx, ly)] = r_xr[o];
        }
    for (size_t z = 1; z <= lnz; ++z)
        for (size_t x = 1; x <= lnx; ++x) {
            size_t o = (z - 1) * lnx + (x - 1);
            data[idx3l(x, 0,     z, lx, ly)] = r_y[o];
            data[idx3l(x, ly - 1, z, lx, ly)] = r_yt[o];
        }
    for (size_t y = 1; y <= lny; ++y)
        for (size_t x = 1; x <= lnx; ++x) {
            size_t o = (y - 1) * lnx + (x - 1);
            data[idx3l(x, y, 0,     lx, ly)] = r_z[o];
            data[idx3l(x, y, lz - 1, lx, ly)] = r_zf[o];
        }
}

// ---------------------------------------------------------------------------
// Chemical potential  (Laplacian inlined, clamped BC)
// ---------------------------------------------------------------------------
static void computeChemicalPotential(const std::vector<double>& c,
                                     std::vector<double>& mu,
                                     const DomainInfo& info,
                                     const double dx, const double dy, const double dz,
                                     const double gamma,
                                     const double e_AA, const double e_BB, const double e_AB) {
    const size_t lx  = info.lx, ly  = info.ly;
    const size_t lnx = info.local_nx, lny = info.local_ny, lnz = info.local_nz;
    const size_t nx  = info.nx, ny  = info.ny, nz  = info.nz;
    const size_t gsx = info.global_start_x, gsy = info.global_start_y, gsz = info.global_start_z;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    for (size_t z = 1; z <= lnz; ++z) {
        const size_t gz = gsz + z - 1;
        const size_t zp = (gz < nz - 1) ? z + 1 : z;
        const size_t zn = (gz > 0)      ? z - 1 : z;
        for (size_t y = 1; y <= lny; ++y) {
            const size_t gy = gsy + y - 1;
            const size_t yp = (gy < ny - 1) ? y + 1 : y;
            const size_t yn = (gy > 0)      ? y - 1 : y;
            for (size_t x = 1; x <= lnx; ++x) {
                const size_t gx = gsx + x - 1;
                const size_t xp = (gx < nx - 1) ? x + 1 : x;
                const size_t xn = (gx > 0)      ? x - 1 : x;

                const size_t idx = idx3l(x, y, z, lx, ly);
                const double cv  = c[idx];

                const double cxx = (c[idx3l(xp, y, z, lx, ly)] +
                                    c[idx3l(xn, y, z, lx, ly)] - 2.0 * cv) * inv_dx2;
                const double cyy = (c[idx3l(x, yp, z, lx, ly)] +
                                    c[idx3l(x, yn, z, lx, ly)] - 2.0 * cv) * inv_dy2;
                const double czz = (c[idx3l(x, y, zp, lx, ly)] +
                                    c[idx3l(x, y, zn, lx, ly)] - 2.0 * cv) * inv_dz2;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Concentration update  (Laplacian of mu, inlined, clamped BC)
// ---------------------------------------------------------------------------
static void cahnHilliardUpdate(std::vector<double>& cnew,
                               const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               const DomainInfo& info,
                               const double D, const double dt,
                               const double dx, const double dy, const double dz) {
    const size_t lx  = info.lx, ly  = info.ly;
    const size_t lnx = info.local_nx, lny = info.local_ny, lnz = info.local_nz;
    const size_t nx  = info.nx, ny  = info.ny, nz  = info.nz;
    const size_t gsx = info.global_start_x, gsy = info.global_start_y, gsz = info.global_start_z;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    for (size_t z = 1; z <= lnz; ++z) {
        const size_t gz = gsz + z - 1;
        const size_t zp = (gz < nz - 1) ? z + 1 : z;
        const size_t zn = (gz > 0)      ? z - 1 : z;
        for (size_t y = 1; y <= lny; ++y) {
            const size_t gy = gsy + y - 1;
            const size_t yp = (gy < ny - 1) ? y + 1 : y;
            const size_t yn = (gy > 0)      ? y - 1 : y;
            for (size_t x = 1; x <= lnx; ++x) {
                const size_t gx = gsx + x - 1;
                const size_t xp = (gx < nx - 1) ? x + 1 : x;
                const size_t xn = (gx > 0)      ? x - 1 : x;

                const size_t idx = idx3l(x, y, z, lx, ly);

                const double mu_xx = (mu[idx3l(xp, y, z, lx, ly)] +
                                      mu[idx3l(xn, y, z, lx, ly)] - 2.0 * mu[idx]) * inv_dx2;
                const double mu_yy = (mu[idx3l(x, yp, z, lx, ly)] +
                                      mu[idx3l(x, yn, z, lx, ly)] - 2.0 * mu[idx]) * inv_dy2;
                const double mu_zz = (mu[idx3l(x, y, zp, lx, ly)] +
                                      mu[idx3l(x, y, zn, lx, ly)] - 2.0 * mu[idx]) * inv_dz2;

                cnew[idx] = cold[idx] + dtD * (mu_xx + mu_yy + mu_zz);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Initialize concentration field (each rank initializes its subdomain)
// ---------------------------------------------------------------------------
static void initializeConcentration(std::vector<double>& c, const DomainInfo& info) {
    const size_t lx  = info.lx, ly  = info.ly;
    const size_t lnx = info.local_nx, lny = info.local_ny, lnz = info.local_nz;
    const size_t nx  = info.nx, ny  = info.ny, nz  = info.nz;
    const size_t gsx = info.global_start_x, gsy = info.global_start_y, gsz = info.global_start_z;
    const size_t vol = nx * ny * nz;

    for (size_t z = 1; z <= lnz; ++z) {
        const size_t gz = gsz + z - 1;
        for (size_t y = 1; y <= lny; ++y) {
            const size_t gy = gsy + y - 1;
            for (size_t x = 1; x <= lnx; ++x) {
                const size_t gx = gsx + x - 1;
                const size_t idx = idx3l(x, y, z, lx, ly);
                const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) /
                                       static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gather owned cells from all ranks to rank 0 in global order
// ---------------------------------------------------------------------------
static void gatherToGlobal(std::vector<double>& global_data,
                           const std::vector<double>& local_data,
                           const DomainInfo& info) {
    const size_t lx  = info.lx, ly  = info.ly;
    const size_t lnx = info.local_nx, lny = info.local_ny, lnz = info.local_nz;

    if (info.rank == 0) {
        size_t gridSize = info.nx * info.ny * info.nz;
        global_data.resize(gridSize);

        // place own data
        for (size_t z = 1; z <= lnz; ++z)
            for (size_t y = 1; y <= lny; ++y)
                for (size_t x = 1; x <= lnx; ++x) {
                    size_t gx = info.global_start_x + x - 1;
                    size_t gy = info.global_start_y + y - 1;
                    size_t gz = info.global_start_z + z - 1;
                    global_data[gz * info.nx * info.ny + gy * info.nx + gx] =
                        local_data[idx3l(x, y, z, lx, ly)];
                }

        // receive from other ranks
        for (int r = 1; r < info.size; ++r) {
            DomainInfo ri;
            computeDomainInfo(r, info.size, info.nx, info.ny, info.nz, ri);
            if (ri.local_nx == 0 || ri.local_ny == 0 || ri.local_nz == 0) continue;

            size_t ls = ri.local_nx * ri.local_ny * ri.local_nz;
            std::vector<double> buf(ls);
            MPI_Recv(buf.data(), ls, MPI_DOUBLE, r, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t z = 0; z < ri.local_nz; ++z)
                for (size_t y = 0; y < ri.local_ny; ++y)
                    for (size_t x = 0; x < ri.local_nx; ++x) {
                        size_t gx = ri.global_start_x + x;
                        size_t gy = ri.global_start_y + y;
                        size_t gz = ri.global_start_z + z;
                        global_data[gz * info.nx * info.ny + gy * info.nx + gx] =
                            buf[z * ri.local_nx * ri.local_ny +
                                y * ri.local_nx + x];
                    }
        }
    } else {
        if (lnx == 0 || lny == 0 || lnz == 0) return;
        size_t ls = lnx * lny * lnz;
        std::vector<double> buf(ls);
        for (size_t z = 1; z <= lnz; ++z)
            for (size_t y = 1; y <= lny; ++y)
                for (size_t x = 1; x <= lnx; ++x) {
                    buf[(z - 1) * lnx * lny + (y - 1) * lnx + (x - 1)] =
                        local_data[idx3l(x, y, z, lx, ly)];
                }
        MPI_Send(buf.data(), ls, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -x <num>     Grid size in X dimension (default: 64)\n");
                printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
                printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
                printf("  -i <num>     Number of time steps (default: 20)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printf("Usage: %s [options]\n", argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    DomainInfo info;
    computeDomainInfo(rank, size, nx, ny, nz, info);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d (decomposition: %d x %d x %d)\n",
               size, info.nx_proc, info.ny_proc, info.nz_proc);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    size_t localSize = info.lx * info.ly * info.lz;
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    // pre-allocate halo-exchange buffers
    const size_t xc = info.local_ny * info.local_nz;
    const size_t yc = info.local_nx * info.local_nz;
    const size_t zc = info.local_nx * info.local_ny;
    std::vector<double> s_x(xc), r_x(xc), s_xl(xc), r_xr(xc);
    std::vector<double> s_y(yc), r_y(yc), s_yb(yc), r_yt(yc);
    std::vector<double> s_z(zc), r_z(zc), s_zb(zc), r_zf(zc);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, info);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, info, s_x, r_x, s_xl, r_xr,
                      s_y, r_y, s_yb, r_yt, s_z, r_z, s_zb, r_zf);
        computeChemicalPotential(cold, mu, info,
                                 dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        exchangeHalos(mu, info, s_x, r_x, s_xl, r_xr,
                      s_y, r_y, s_yb, r_yt, s_z, r_z, s_zb, r_zf);
        cahnHilliardUpdate(cnew, cold, mu, info,
                           D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }

    double t_end = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    size_t gridSize = nx * ny * nz;
    double elapsed = t_end - t_start;
    double cellUpdates = static_cast<double>(gridSize) * iterations;
    double mcups = (elapsed > 0.0) ? cellUpdates / elapsed / 1e6 : 0.0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> global_data;
        gatherToGlobal(global_data, cold, info);
        if (rank == 0) print_results(global_data, "Concentration");
    }

    int result = 0;
    if (validate) {
        double local_min = 1e300, local_max = -1e300;
        int local_bad = 0;
        for (size_t z = 1; z <= info.local_nz; ++z)
            for (size_t y = 1; y <= info.local_ny; ++y)
                for (size_t x = 1; x <= info.local_nx; ++x) {
                    double v = cold[idx3l(x, y, z, info.lx, info.ly)];
                    if (std::isnan(v) || std::isinf(v)) local_bad = 1;
                    if (v < local_min) local_min = v;
                    if (v > local_max) local_max = v;
                }

        double global_min, global_max;
        int global_bad;
        MPI_Allreduce(&local_min,  &global_min,  1, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD);
        MPI_Allreduce(&local_max,  &global_max,  1, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);
        MPI_Allreduce(&local_bad,  &global_bad,  1, MPI_INT,    MPI_LOR, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);

            bool valid = !global_bad && global_max <= 10.0 && global_min >= -10.0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                if (global_bad)
                    printf("Validation failed: found NaN or Inf value\n");
                else
                    printf("Validation failed: values out of expected range\n");
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    MPI_Finalize();
    return result;
}
