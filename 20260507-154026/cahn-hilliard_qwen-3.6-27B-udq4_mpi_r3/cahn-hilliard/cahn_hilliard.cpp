#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------- domain decomposition ----------

struct DomainInfo {
    size_t global_nx, global_ny, global_nz;
    size_t lx, ly, lz;       // local interior size
    size_t lxg, lyg, lzg;    // local with ghost layer
    size_t gx0, gy0, gz0;    // global starting indices
    int rank, size;
    int px, py, pz;          // process grid dimensions
    int ix, iy, iz;          // position in process grid
    int nplus[3], nminus[3]; // neighbour ranks
};

static void computeProcessGrid(int nr, int& px, int& py, int& pz) {
    px = py = pz = 1;
    if (nr <= 1) return;
    int best = nr;
    for (int a = 1; a <= nr; ++a) {
        if (nr % a != 0) continue;
        for (int b = 1; b <= nr / a; ++b) {
            if ((nr / a) % b != 0) continue;
            int c = nr / (a * b);
            int mx = std::max({a, b, c});
            int mn = std::min({a, b, c});
            int imb = mx / std::max(1, mn);
            if (imb < best) { best = imb; px = a; py = b; pz = c; }
        }
    }
}

static DomainInfo computeDomainInfo(size_t nx, size_t ny, size_t nz,
                                    int rank, int size) {
    DomainInfo d;
    d.global_nx = nx; d.global_ny = ny; d.global_nz = nz;
    d.rank = rank; d.size = size;
    computeProcessGrid(size, d.px, d.py, d.pz);

    d.ix = rank % d.px;
    d.iy = (rank / d.px) % d.py;
    d.iz = rank / (d.px * d.py);

    d.lx = nx / d.px + (d.ix < (int)(nx % d.px) ? 1 : 0);
    d.ly = ny / d.py + (d.iy < (int)(ny % d.py) ? 1 : 0);
    d.lz = nz / d.pz + (d.iz < (int)(nz % d.pz) ? 1 : 0);

    d.gx0 = 0;
    for (int i = 0; i < d.ix; ++i)
        d.gx0 += nx / d.px + (i < (int)(nx % d.px) ? 1 : 0);
    d.gy0 = 0;
    for (int i = 0; i < d.iy; ++i)
        d.gy0 += ny / d.py + (i < (int)(ny % d.py) ? 1 : 0);
    d.gz0 = 0;
    for (int i = 0; i < d.iz; ++i)
        d.gz0 += nz / d.pz + (i < (int)(nz % d.pz) ? 1 : 0);

    d.lxg = d.lx + 2; d.lyg = d.ly + 2; d.lzg = d.lz + 2;

    d.nplus[0]  = (d.ix < d.px - 1) ? rank + 1 : -1;
    d.nminus[0] = (d.ix > 0)        ? rank - 1 : -1;
    d.nplus[1]  = (d.iy < d.py - 1) ? rank + d.px : -1;
    d.nminus[1] = (d.iy > 0)        ? rank - d.px : -1;
    d.nplus[2]  = (d.iz < d.pz - 1) ? rank + d.px * d.py : -1;
    d.nminus[2] = (d.iz > 0)        ? rank - d.px * d.py : -1;
    return d;
}

// ---------- halo communication ----------

struct ExchangeBuf {
    std::vector<double> sXp, sXn, rXp, rXn;
    std::vector<double> sYp, sYn, rYp, rYn;
    std::vector<double> sZp, sZn, rZp, rZn;
    MPI_Request req[12];

    void alloc(const DomainInfo& d) {
        sXp.resize(d.ly * d.lz); sXn.resize(d.ly * d.lz);
        rXp.resize(d.ly * d.lz); rXn.resize(d.ly * d.lz);
        sYp.resize(d.lx * d.lz); sYn.resize(d.lx * d.lz);
        rYp.resize(d.lx * d.lz); rYn.resize(d.lx * d.lz);
        sZp.resize(d.lx * d.ly); sZn.resize(d.lx * d.ly);
        rZp.resize(d.lx * d.ly); rZn.resize(d.lx * d.ly);
    }
};

static void setBoundaryGhosts(std::vector<double>& f, const DomainInfo& d) {
    const size_t lxg = d.lxg, lyg = d.lyg;
    if (d.ix == 0) {
        for (size_t z = 1; z <= d.lz; ++z)
            for (size_t y = 1; y <= d.ly; ++y)
                f[idx3(0, y, z, lxg, lyg)] = f[idx3(1, y, z, lxg, lyg)];
    }
    if (d.ix == d.px - 1) {
        for (size_t z = 1; z <= d.lz; ++z)
            for (size_t y = 1; y <= d.ly; ++y)
                f[idx3(d.lx + 1, y, z, lxg, lyg)] = f[idx3(d.lx, y, z, lxg, lyg)];
    }
    if (d.iy == 0) {
        for (size_t z = 1; z <= d.lz; ++z)
            for (size_t x = 1; x <= d.lx; ++x)
                f[idx3(x, 0, z, lxg, lyg)] = f[idx3(x, 1, z, lxg, lyg)];
    }
    if (d.iy == d.py - 1) {
        for (size_t z = 1; z <= d.lz; ++z)
            for (size_t x = 1; x <= d.lx; ++x)
                f[idx3(x, d.ly + 1, z, lxg, lyg)] = f[idx3(x, d.ly, z, lxg, lyg)];
    }
    if (d.iz == 0) {
        for (size_t y = 1; y <= d.ly; ++y)
            for (size_t x = 1; x <= d.lx; ++x)
                f[idx3(x, y, 0, lxg, lyg)] = f[idx3(x, y, 1, lxg, lyg)];
    }
    if (d.iz == d.pz - 1) {
        for (size_t y = 1; y <= d.ly; ++y)
            for (size_t x = 1; x <= d.lx; ++x)
                f[idx3(x, y, d.lz + 1, lxg, lyg)] = f[idx3(x, y, d.lz, lxg, lyg)];
    }
}

static void exchangeHalos(std::vector<double>& f, const DomainInfo& d, ExchangeBuf& b) {
    const size_t lxg = d.lxg, lyg = d.lyg;
    int nreq = 0;

    // X
    if (d.nplus[0] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t y = 0; y < d.ly; ++y)
                b.sXp[z * d.ly + y] = f[idx3(d.lx, y + 1, z + 1, lxg, lyg)];
        MPI_Isend(b.sXp.data(), d.ly * d.lz, MPI_DOUBLE, d.nplus[0], 0,
                  MPI_COMM_WORLD, &b.req[nreq++]);
        MPI_Irecv(b.rXp.data(), d.ly * d.lz, MPI_DOUBLE, d.nplus[0], 0,
                  MPI_COMM_WORLD, &b.req[nreq++]);
    }
    if (d.nminus[0] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t y = 0; y < d.ly; ++y)
                b.sXn[z * d.ly + y] = f[idx3(1, y + 1, z + 1, lxg, lyg)];
        MPI_Isend(b.sXn.data(), d.ly * d.lz, MPI_DOUBLE, d.nminus[0], 0,
                  MPI_COMM_WORLD, &b.req[nreq++]);
        MPI_Irecv(b.rXn.data(), d.ly * d.lz, MPI_DOUBLE, d.nminus[0], 0,
                  MPI_COMM_WORLD, &b.req[nreq++]);
    }
    // Y
    if (d.nplus[1] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t x = 0; x < d.lx; ++x)
                b.sYp[z * d.lx + x] = f[idx3(x + 1, d.ly, z + 1, lxg, lyg)];
        MPI_Isend(b.sYp.data(), d.lx * d.lz, MPI_DOUBLE, d.nplus[1], 1,
                  MPI_COMM_WORLD, &b.req[nreq++]);
        MPI_Irecv(b.rYp.data(), d.lx * d.lz, MPI_DOUBLE, d.nplus[1], 1,
                  MPI_COMM_WORLD, &b.req[nreq++]);
    }
    if (d.nminus[1] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t x = 0; x < d.lx; ++x)
                b.sYn[z * d.lx + x] = f[idx3(x + 1, 1, z + 1, lxg, lyg)];
        MPI_Isend(b.sYn.data(), d.lx * d.lz, MPI_DOUBLE, d.nminus[1], 1,
                  MPI_COMM_WORLD, &b.req[nreq++]);
        MPI_Irecv(b.rYn.data(), d.lx * d.lz, MPI_DOUBLE, d.nminus[1], 1,
                  MPI_COMM_WORLD, &b.req[nreq++]);
    }
    // Z
    if (d.nplus[2] >= 0) {
        for (size_t y = 0; y < d.ly; ++y)
            for (size_t x = 0; x < d.lx; ++x)
                b.sZp[y * d.lx + x] = f[idx3(x + 1, y + 1, d.lz, lxg, lyg)];
        MPI_Isend(b.sZp.data(), d.lx * d.ly, MPI_DOUBLE, d.nplus[2], 2,
                  MPI_COMM_WORLD, &b.req[nreq++]);
        MPI_Irecv(b.rZp.data(), d.lx * d.ly, MPI_DOUBLE, d.nplus[2], 2,
                  MPI_COMM_WORLD, &b.req[nreq++]);
    }
    if (d.nminus[2] >= 0) {
        for (size_t y = 0; y < d.ly; ++y)
            for (size_t x = 0; x < d.lx; ++x)
                b.sZn[y * d.lx + x] = f[idx3(x + 1, y + 1, 1, lxg, lyg)];
        MPI_Isend(b.sZn.data(), d.lx * d.ly, MPI_DOUBLE, d.nminus[2], 2,
                  MPI_COMM_WORLD, &b.req[nreq++]);
        MPI_Irecv(b.rZn.data(), d.lx * d.ly, MPI_DOUBLE, d.nminus[2], 2,
                  MPI_COMM_WORLD, &b.req[nreq++]);
    }

    if (nreq > 0) MPI_Waitall(nreq, b.req, MPI_STATUSES_IGNORE);

    // unpack
    if (d.nplus[0] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t y = 0; y < d.ly; ++y)
                f[idx3(d.lx + 1, y + 1, z + 1, lxg, lyg)] = b.rXp[z * d.ly + y];
    }
    if (d.nminus[0] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t y = 0; y < d.ly; ++y)
                f[idx3(0, y + 1, z + 1, lxg, lyg)] = b.rXn[z * d.ly + y];
    }
    if (d.nplus[1] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t x = 0; x < d.lx; ++x)
                f[idx3(x + 1, d.ly + 1, z + 1, lxg, lyg)] = b.rYp[z * d.lx + x];
    }
    if (d.nminus[1] >= 0) {
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t x = 0; x < d.lx; ++x)
                f[idx3(x + 1, 0, z + 1, lxg, lyg)] = b.rYn[z * d.lx + x];
    }
    if (d.nplus[2] >= 0) {
        for (size_t y = 0; y < d.ly; ++y)
            for (size_t x = 0; x < d.lx; ++x)
                f[idx3(x + 1, y + 1, d.lz + 1, lxg, lyg)] = b.rZp[y * d.lx + x];
    }
    if (d.nminus[2] >= 0) {
        for (size_t y = 0; y < d.ly; ++y)
            for (size_t x = 0; x < d.lx; ++x)
                f[idx3(x + 1, y + 1, 0, lxg, lyg)] = b.rZn[y * d.lx + x];
    }
}

// ---------- initialization ----------

static void initializeConcentration(std::vector<double>& c, const DomainInfo& d) {
    const size_t lxg = d.lxg, lyg = d.lyg;
    const size_t vol = d.global_nx * d.global_ny * d.global_nz;
    for (size_t z = 1; z <= d.lz; ++z)
        for (size_t y = 1; y <= d.ly; ++y)
            for (size_t x = 1; x <= d.lx; ++x) {
                const size_t idx = idx3(x, y, z, lxg, lyg);
                const size_t gx = d.gx0 + x - 1;
                const size_t gy = d.gy0 + y - 1;
                const size_t gz = d.gz0 + z - 1;
                const size_t lid = gz * (d.global_nx * d.global_ny) + gy * d.global_nx + gx;
                const double pseudo = ((((lid + 1) * 1299709) % vol) /
                                       static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
}

// ---------- validation ----------

static bool validateResult(const std::vector<double>& c) {
    for (const auto& v : c)
        if (std::isnan(v) || std::isinf(v)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    double mn = c[0], mx = c[0];
    for (const auto& v : c) { mn = std::min(mn, v); mx = std::max(mx, v); }
    printf("Concentration range: [%.6f, %.6f]\n", mn, mx);
    if (mx > 10.0 || mn < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

// ---------- main ----------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- argument parsing (rank 0) ----
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)
                nx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)
                ny = atoi(argv[++i]);
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)
                nz = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
                iterations = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate = true;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = true;
            else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -x <num>     Grid size in X dimension (default: 64)\n");
                printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
                printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
                printf("  -i <num>     Number of time steps (default: 20)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize(); return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize(); return 1;
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

    DomainInfo d = computeDomainInfo(nx, ny, nz, rank, size);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("Process grid: %d x %d x %d\n", d.px, d.py, d.pz);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- allocate local arrays (with ghost layer) ----
    const size_t lxg = d.lxg, lyg = d.lyg, lzg = d.lzg;
    std::vector<double> cold(lxg * lyg * lzg);
    std::vector<double> cnew(lxg * lyg * lzg);
    std::vector<double> mu(lxg * lyg * lzg);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, d);

    // ---- physical parameters ----
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    ExchangeBuf ebuf; ebuf.alloc(d);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // --- ghost exchange for cold ---
        setBoundaryGhosts(cold, d);
        exchangeHalos(cold, d, ebuf);

        // --- chemical potential (interior) ---
        for (size_t z = 1; z <= d.lz; ++z)
            for (size_t y = 1; y <= d.ly; ++y)
                for (size_t x = 1; x <= d.lx; ++x) {
                    const size_t i = idx3(x, y, z, lxg, lyg);
                    const double cv = cold[i];
                    const double lap =
                        (cold[idx3(x+1,y,z,lxg,lyg)] + cold[idx3(x-1,y,z,lxg,lyg)] - 2.0*cv)/(dx*dx) +
                        (cold[idx3(x,y+1,z,lxg,lyg)] + cold[idx3(x,y-1,z,lxg,lyg)] - 2.0*cv)/(dy*dy) +
                        (cold[idx3(x,y,z+1,lxg,lyg)] + cold[idx3(x,y,z-1,lxg,lyg)] - 2.0*cv)/(dz*dz);
                    mu[i] = 4.5*((cv+1.0)*e_AA + (cv-1.0)*e_BB - 2.0*cv*e_AB)
                           + 3.0*cv + cv*cv*cv - gamma*lap;
                }

        // --- ghost exchange for mu ---
        setBoundaryGhosts(mu, d);
        exchangeHalos(mu, d, ebuf);

        // --- concentration update (interior) ---
        for (size_t z = 1; z <= d.lz; ++z)
            for (size_t y = 1; y <= d.ly; ++y)
                for (size_t x = 1; x <= d.lx; ++x) {
                    const size_t i = idx3(x, y, z, lxg, lyg);
                    const double lap =
                        (mu[idx3(x+1,y,z,lxg,lyg)] + mu[idx3(x-1,y,z,lxg,lyg)] - 2.0*mu[i])/(dx*dx) +
                        (mu[idx3(x,y+1,z,lxg,lyg)] + mu[idx3(x,y-1,z,lxg,lyg)] - 2.0*mu[i])/(dy*dy) +
                        (mu[idx3(x,y,z+1,lxg,lyg)] + mu[idx3(x,y,z-1,lxg,lyg)] - 2.0*mu[i])/(dz*dz);
                    cnew[i] = cold[i] + dt * D * lap;
                }

        std::swap(cold, cnew);
    }

    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    double duration = (t1 - t0) * 1000.0;
    size_t gridSize = nx * ny * nz;
    double mcups = (double)gridSize * iterations / (duration / 1000.0) / 1e6;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- gather for print / validation ----
    if (printResults || validate) {
        size_t locInt = d.lx * d.ly * d.lz;
        std::vector<double> interior(locInt);
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t y = 0; y < d.ly; ++y)
                for (size_t x = 0; x < d.lx; ++x)
                    interior[z*d.ly*d.lx + y*d.lx + x] =
                        cold[idx3(x+1, y+1, z+1, lxg, lyg)];

        std::vector<int> counts(size), displs(size);
        std::vector<size_t> rlx(size), rly(size), rlz(size);
        std::vector<size_t> rgx(size), rgy(size), rgz(size);

        for (int iz = 0; iz < d.pz; ++iz)
            for (int iy = 0; iy < d.py; ++iy)
                for (int ix = 0; ix < d.px; ++ix) {
                    int r = ix + iy*d.px + iz*d.px*d.py;
                    rlx[r] = nx/d.px + (ix < (int)(nx%d.px) ? 1 : 0);
                    rly[r] = ny/d.py + (iy < (int)(ny%d.py) ? 1 : 0);
                    rlz[r] = nz/d.pz + (iz < (int)(nz%d.pz) ? 1 : 0);
                    counts[r] = (int)(rlx[r]*rly[r]*rlz[r]);
                    size_t gx=0; for(int i=0;i<ix;++i) gx+=nx/d.px+(i<(int)(nx%d.px)?1:0); rgx[r]=gx;
                    size_t gy=0; for(int i=0;i<iy;++i) gy+=ny/d.py+(i<(int)(ny%d.py)?1:0); rgy[r]=gy;
                    size_t gz=0; for(int i=0;i<iz;++i) gz+=nz/d.pz+(i<(int)(nz%d.pz)?1:0); rgz[r]=gz;
                }
        displs[0] = 0;
        for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];

        std::vector<double> gathered;
        if (rank == 0) gathered.resize(gridSize);

        MPI_Gatherv(interior.data(), (int)locInt, MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::vector<double> globalData(gridSize);
            for (int r = 0; r < size; ++r)
                for (size_t z = 0; z < rlz[r]; ++z)
                    for (size_t y = 0; y < rly[r]; ++y)
                        for (size_t x = 0; x < rlx[r]; ++x) {
                            size_t gi = (rgz[r]+z)*(nx*ny) + (rgy[r]+y)*nx + (rgx[r]+x);
                            size_t li = (size_t)displs[r] + z*(rly[r]*rlx[r]) + y*rlx[r] + x;
                            globalData[gi] = gathered[li];
                        }

            if (printResults) print_results(globalData, "Concentration");

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(globalData);
                MPI_Finalize();
                return valid ? 0 : 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
