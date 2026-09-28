#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Domain decomposition helpers
// ---------------------------------------------------------------------------

// Local (halo-padded) grid geometry of this rank.
struct LocalGrid {
    int lx = 0, ly = 0, lz = 0;    // interior extents owned by this rank
    int x0 = 0, y0 = 0, z0 = 0;    // global index of the first owned cell
    int nxp = 0, nyp = 0, nzp = 0; // padded extents (interior + 2 ghost layers)
    size_t stride_y = 0;           // nxp
    size_t stride_z = 0;           // nxp * nyp
    size_t padded_size = 0;
};

// Inclusive index range of a sub-box of the interior.
struct Box {
    int x0, x1, y0, y1, z0, z1;
};

static inline size_t lidx(const LocalGrid& g, const int x, const int y, const int z) noexcept {
    return static_cast<size_t>(z) * g.stride_z + static_cast<size_t>(y) * g.stride_y + static_cast<size_t>(x);
}

// Pick a 3D process grid minimizing the per-rank halo surface, subject to
// each dimension holding at least one cell per rank.
static void chooseDims(const int nranks, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    double best = -1.0;
    dims[0] = dims[1] = dims[2] = 1;

    for (int px = 1; px <= nranks; ++px) {
        if (nranks % px != 0 || static_cast<size_t>(px) > nx) continue;
        const int rem = nranks / px;
        for (int py = 1; py <= rem; ++py) {
            if (rem % py != 0 || static_cast<size_t>(py) > ny) continue;
            const int pz = rem / py;
            if (static_cast<size_t>(pz) > nz) continue;

            const double sx = static_cast<double>(nx) / px;
            const double sy = static_cast<double>(ny) / py;
            const double sz = static_cast<double>(nz) / pz;
            // Communication volume per rank (halo faces) + tiny bias towards
            // cubic blocks for cache friendliness.
            const double cost = sy * sz + sx * sz + sx * sy;
            if (best < 0.0 || cost < best - 1e-9) {
                best = cost;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compute kernels (operate on halo-padded local arrays; clamped boundary
// conditions are materialized in the ghost layers, so no branches are needed)
// ---------------------------------------------------------------------------

static void computeChemicalPotentialBox(const double* __restrict__ c, double* __restrict__ mu,
                                        const LocalGrid& g, const Box& b,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double ixx = 1.0 / (dx * dx);
    const double iyy = 1.0 / (dy * dy);
    const double izz = 1.0 / (dz * dz);
    const ptrdiff_t sy = static_cast<ptrdiff_t>(g.stride_y);
    const ptrdiff_t sz = static_cast<ptrdiff_t>(g.stride_z);

    for (int z = b.z0; z <= b.z1; ++z) {
        for (int y = b.y0; y <= b.y1; ++y) {
            const size_t base = lidx(g, 0, y, z);
            const double* __restrict__ cp = c + base;
            double* __restrict__ mp = mu + base;
            for (int x = b.x0; x <= b.x1; ++x) {
                const double cv = cp[x];
                const double cxx = (cp[x + 1] + cp[x - 1] - 2.0 * cv) * ixx;
                const double cyy = (cp[x + sy] + cp[x - sy] - 2.0 * cv) * iyy;
                const double czz = (cp[x + sz] + cp[x - sz] - 2.0 * cv) * izz;

                mp[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * (cxx + cyy + czz);
            }
        }
    }
}

static void cahnHilliardUpdateBox(double* __restrict__ cnew, const double* __restrict__ cold,
                                  const double* __restrict__ mu, const LocalGrid& g, const Box& b,
                                  const double D, const double dt, const double dx, const double dy, const double dz) {
    const double ixx = 1.0 / (dx * dx);
    const double iyy = 1.0 / (dy * dy);
    const double izz = 1.0 / (dz * dz);
    const double dtD = dt * D;
    const ptrdiff_t sy = static_cast<ptrdiff_t>(g.stride_y);
    const ptrdiff_t sz = static_cast<ptrdiff_t>(g.stride_z);

    for (int z = b.z0; z <= b.z1; ++z) {
        for (int y = b.y0; y <= b.y1; ++y) {
            const size_t base = lidx(g, 0, y, z);
            const double* __restrict__ mp = mu + base;
            const double* __restrict__ op = cold + base;
            double* __restrict__ np = cnew + base;
            for (int x = b.x0; x <= b.x1; ++x) {
                const double mv = mp[x];
                const double cxx = (mp[x + 1] + mp[x - 1] - 2.0 * mv) * ixx;
                const double cyy = (mp[x + sy] + mp[x - sy] - 2.0 * mv) * iyy;
                const double czz = (mp[x + sz] + mp[x - sz] - 2.0 * mv) * izz;
                np[x] = op[x] + dtD * (cxx + cyy + czz);
            }
        }
    }
}

// Initialize the local part of the concentration field from global indices.
static void initializeConcentration(std::vector<double>& c, const LocalGrid& g,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (int z = 1; z <= g.lz; ++z) {
        const size_t gz = static_cast<size_t>(g.z0 + z - 1);
        for (int y = 1; y <= g.ly; ++y) {
            const size_t gy = static_cast<size_t>(g.y0 + y - 1);
            const size_t row = gz * (nx * ny) + gy * nx;
            double* __restrict__ cp = c.data() + lidx(g, 0, y, z);
            for (int x = 1; x <= g.lx; ++x) {
                const size_t linear_id = row + static_cast<size_t>(g.x0 + x - 1);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cp[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Halo exchange
// ---------------------------------------------------------------------------

class HaloExchanger {
  public:
    HaloExchanger(MPI_Comm cart, const LocalGrid& g) : comm_(cart), g_(g) {
        MPI_Cart_shift(comm_, 0, 1, &nbr_[0], &nbr_[1]);
        MPI_Cart_shift(comm_, 1, 1, &nbr_[2], &nbr_[3]);
        MPI_Cart_shift(comm_, 2, 1, &nbr_[4], &nbr_[5]);

        const int cnt[6] = {g.ly * g.lz, g.ly * g.lz, g.lx * g.lz, g.lx * g.lz, g.lx * g.ly, g.lx * g.ly};
        for (int f = 0; f < 6; ++f) {
            count_[f] = cnt[f];
            sbuf_[f].resize(static_cast<size_t>(cnt[f]));
            rbuf_[f].resize(static_cast<size_t>(cnt[f]));
        }
    }

    // Fill ghost layers at physical (non-neighbored) boundaries such that the
    // 7-point stencil reproduces the original clamped boundary conditions.
    void fillPhysicalBoundaries(double* f) const {
        const LocalGrid& g = g_;
        if (nbr_[0] == MPI_PROC_NULL) copyPlaneX(f, 1, 0);
        if (nbr_[1] == MPI_PROC_NULL) copyPlaneX(f, g.lx, g.lx + 1);
        if (nbr_[2] == MPI_PROC_NULL) copyPlaneY(f, 1, 0);
        if (nbr_[3] == MPI_PROC_NULL) copyPlaneY(f, g.ly, g.ly + 1);
        if (nbr_[4] == MPI_PROC_NULL) copyPlaneZ(f, 1, 0);
        if (nbr_[5] == MPI_PROC_NULL) copyPlaneZ(f, g.lz, g.lz + 1);
    }

    void start(double* f) {
        field_ = f;
        int nreq = 0;
        for (int face = 0; face < 6; ++face) {
            if (nbr_[face] == MPI_PROC_NULL || count_[face] == 0) continue;
            MPI_Irecv(rbuf_[face].data(), count_[face], MPI_DOUBLE, nbr_[face], face, comm_, &req_[nreq++]);
        }
        for (int face = 0; face < 6; ++face) {
            if (nbr_[face] == MPI_PROC_NULL || count_[face] == 0) continue;
            pack(f, face);
            // Tag with the face index the receiver sees (opposite face).
            const int tag = face ^ 1;
            MPI_Isend(sbuf_[face].data(), count_[face], MPI_DOUBLE, nbr_[face], tag, comm_, &req_[nreq++]);
        }
        nreq_ = nreq;
    }

    void finish() {
        if (nreq_ > 0) MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE);
        for (int face = 0; face < 6; ++face) {
            if (nbr_[face] == MPI_PROC_NULL || count_[face] == 0) continue;
            unpack(field_, face);
        }
        nreq_ = 0;
    }

  private:
    void copyPlaneX(double* f, const int src, const int dst) const {
        for (int z = 1; z <= g_.lz; ++z)
            for (int y = 1; y <= g_.ly; ++y) f[lidx(g_, dst, y, z)] = f[lidx(g_, src, y, z)];
    }
    void copyPlaneY(double* f, const int src, const int dst) const {
        for (int z = 1; z <= g_.lz; ++z)
            memcpy(&f[lidx(g_, 1, dst, z)], &f[lidx(g_, 1, src, z)], sizeof(double) * static_cast<size_t>(g_.lx));
    }
    void copyPlaneZ(double* f, const int src, const int dst) const {
        for (int y = 1; y <= g_.ly; ++y)
            memcpy(&f[lidx(g_, 1, y, dst)], &f[lidx(g_, 1, y, src)], sizeof(double) * static_cast<size_t>(g_.lx));
    }

    void pack(const double* f, const int face) {
        double* b = sbuf_[face].data();
        const LocalGrid& g = g_;
        switch (face) {
        case 0:
        case 1: {
            const int x = (face == 0) ? 1 : g.lx;
            size_t k = 0;
            for (int z = 1; z <= g.lz; ++z)
                for (int y = 1; y <= g.ly; ++y) b[k++] = f[lidx(g, x, y, z)];
            break;
        }
        case 2:
        case 3: {
            const int y = (face == 2) ? 1 : g.ly;
            size_t k = 0;
            for (int z = 1; z <= g.lz; ++z, k += static_cast<size_t>(g.lx))
                memcpy(b + k, &f[lidx(g, 1, y, z)], sizeof(double) * static_cast<size_t>(g.lx));
            break;
        }
        default: {
            const int z = (face == 4) ? 1 : g.lz;
            size_t k = 0;
            for (int y = 1; y <= g.ly; ++y, k += static_cast<size_t>(g.lx))
                memcpy(b + k, &f[lidx(g, 1, y, z)], sizeof(double) * static_cast<size_t>(g.lx));
            break;
        }
        }
    }

    void unpack(double* f, const int face) {
        const double* b = rbuf_[face].data();
        const LocalGrid& g = g_;
        switch (face) {
        case 0:
        case 1: {
            const int x = (face == 0) ? 0 : g.lx + 1;
            size_t k = 0;
            for (int z = 1; z <= g.lz; ++z)
                for (int y = 1; y <= g.ly; ++y) f[lidx(g, x, y, z)] = b[k++];
            break;
        }
        case 2:
        case 3: {
            const int y = (face == 2) ? 0 : g.ly + 1;
            size_t k = 0;
            for (int z = 1; z <= g.lz; ++z, k += static_cast<size_t>(g.lx))
                memcpy(&f[lidx(g, 1, y, z)], b + k, sizeof(double) * static_cast<size_t>(g.lx));
            break;
        }
        default: {
            const int z = (face == 4) ? 0 : g.lz + 1;
            size_t k = 0;
            for (int y = 1; y <= g.ly; ++y, k += static_cast<size_t>(g.lx))
                memcpy(&f[lidx(g, 1, y, z)], b + k, sizeof(double) * static_cast<size_t>(g.lx));
            break;
        }
        }
    }

    MPI_Comm comm_;
    LocalGrid g_;
    int nbr_[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int count_[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> sbuf_[6];
    std::vector<double> rbuf_[6];
    MPI_Request req_[12];
    int nreq_ = 0;
    double* field_ = nullptr;
};

// Split the owned interior into an inner box (independent of ghost data) and
// the surrounding shell (needs completed halo exchange).
static void makeBoxes(const LocalGrid& g, Box& inner, Box shell[6], int& nshell) {
    inner = {2, g.lx - 1, 2, g.ly - 1, 2, g.lz - 1};
    nshell = 0;

    // z faces (full x/y extent)
    shell[nshell++] = {1, g.lx, 1, g.ly, 1, 1};
    if (g.lz > 1) shell[nshell++] = {1, g.lx, 1, g.ly, g.lz, g.lz};
    // y faces (full x extent, remaining z)
    if (g.lz > 2) {
        shell[nshell++] = {1, g.lx, 1, 1, 2, g.lz - 1};
        if (g.ly > 1) shell[nshell++] = {1, g.lx, g.ly, g.ly, 2, g.lz - 1};
        // x faces (remaining y and z)
        if (g.ly > 2) {
            shell[nshell++] = {1, 1, 2, g.ly - 1, 2, g.lz - 1};
            if (g.lx > 1) shell[nshell++] = {g.lx, g.lx, 2, g.ly - 1, 2, g.lz - 1};
        }
    }
}

static inline bool boxEmpty(const Box& b) {
    return b.x1 < b.x0 || b.y1 < b.y0 || b.z1 < b.z0;
}

// ---------------------------------------------------------------------------

static bool validateResult(const std::vector<double>& c, const LocalGrid& g, MPI_Comm comm, const int rank) {
    int localBad = 0;
    double minVal = 1e308, maxVal = -1e308;
    for (int z = 1; z <= g.lz; ++z) {
        for (int y = 1; y <= g.ly; ++y) {
            const double* cp = c.data() + lidx(g, 0, y, z);
            for (int x = 1; x <= g.lx; ++x) {
                const double v = cp[x];
                if (std::isnan(v) || std::isinf(v)) {
                    localBad = 1;
                } else {
                    minVal = std::min(minVal, v);
                    maxVal = std::max(maxVal, v);
                }
            }
        }
    }

    int bad = 0;
    MPI_Allreduce(&localBad, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);

    if (gmax > 10.0 || gmin < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

// Gather the distributed field into the global, x-fastest ordered array on rank 0.
static void gatherField(const std::vector<double>& c, const LocalGrid& g, MPI_Comm comm,
                        const int rank, const int nranks,
                        const size_t nx, const size_t ny, const size_t nz,
                        std::vector<double>& global) {
    const size_t localCount = static_cast<size_t>(g.lx) * g.ly * g.lz;
    std::vector<double> packed(localCount);
    size_t k = 0;
    for (int z = 1; z <= g.lz; ++z)
        for (int y = 1; y <= g.ly; ++y, k += static_cast<size_t>(g.lx))
            memcpy(packed.data() + k, &c[lidx(g, 1, y, z)], sizeof(double) * static_cast<size_t>(g.lx));

    const int info[6] = {g.x0, g.y0, g.z0, g.lx, g.ly, g.lz};
    std::vector<int> allInfo(rank == 0 ? static_cast<size_t>(6 * nranks) : size_t{0});
    MPI_Gather(info, 6, MPI_INT, allInfo.data(), 6, MPI_INT, 0, comm);

    std::vector<int> counts, displs;
    std::vector<double> recv;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(nranks));
        displs.resize(static_cast<size_t>(nranks));
        int off = 0;
        for (int r = 0; r < nranks; ++r) {
            counts[static_cast<size_t>(r)] = allInfo[static_cast<size_t>(6 * r + 3)] *
                                             allInfo[static_cast<size_t>(6 * r + 4)] *
                                             allInfo[static_cast<size_t>(6 * r + 5)];
            displs[static_cast<size_t>(r)] = off;
            off += counts[static_cast<size_t>(r)];
        }
        recv.resize(static_cast<size_t>(off));
    }

    MPI_Gatherv(packed.data(), static_cast<int>(localCount), MPI_DOUBLE,
                recv.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    if (rank != 0) return;

    global.resize(nx * ny * nz);
    for (int r = 0; r < nranks; ++r) {
        const int* in = &allInfo[static_cast<size_t>(6 * r)];
        const double* src = recv.data() + displs[static_cast<size_t>(r)];
        for (int z = 0; z < in[5]; ++z) {
            for (int y = 0; y < in[4]; ++y) {
                const size_t dst = (static_cast<size_t>(in[2] + z) * ny + static_cast<size_t>(in[1] + y)) * nx +
                                   static_cast<size_t>(in[0]);
                memcpy(&global[dst], src + (static_cast<size_t>(z) * in[4] + y) * in[3],
                       sizeof(double) * static_cast<size_t>(in[3]));
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;

    // --- Cartesian process grid -------------------------------------------
    int dims[3] = {1, 1, 1};
    chooseDims(nranks, nx, ny, nz, dims);

    // Ranks that cannot get any cells (more ranks than cells) are excluded.
    const int active = dims[0] * dims[1] * dims[2];
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm work = MPI_COMM_NULL;
    const int color = (rank < active) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &work);

    if (work == MPI_COMM_NULL) {
        // Idle rank: still participate in the final barrier-free finalize.
        MPI_Finalize();
        return 0;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(work, 3, dims, periods, /*reorder=*/1, &cart);

    int crank = 0, csize = 1;
    MPI_Comm_rank(cart, &crank);
    MPI_Comm_size(cart, &csize);
    int coords[3];
    MPI_Cart_coords(cart, crank, 3, coords);

    LocalGrid g;
    const size_t gn[3] = {nx, ny, nz};
    int lc[3], off[3];
    for (int d = 0; d < 3; ++d) {
        const int base = static_cast<int>(gn[d] / static_cast<size_t>(dims[d]));
        const int rem = static_cast<int>(gn[d] % static_cast<size_t>(dims[d]));
        lc[d] = base + (coords[d] < rem ? 1 : 0);
        off[d] = base * coords[d] + std::min(coords[d], rem);
    }
    g.lx = lc[0]; g.ly = lc[1]; g.lz = lc[2];
    g.x0 = off[0]; g.y0 = off[1]; g.z0 = off[2];
    g.nxp = g.lx + 2; g.nyp = g.ly + 2; g.nzp = g.lz + 2;
    g.stride_y = static_cast<size_t>(g.nxp);
    g.stride_z = static_cast<size_t>(g.nxp) * static_cast<size_t>(g.nyp);
    g.padded_size = g.stride_z * static_cast<size_t>(g.nzp);

    std::vector<double> cold(g.padded_size, 0.0);
    std::vector<double> cnew(g.padded_size, 0.0);
    std::vector<double> mu(g.padded_size, 0.0);

    if (crank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, g, nx, ny, nz);

    HaloExchanger halo(cart, g);
    Box inner, shell[6];
    int nshell = 0;
    makeBoxes(g, inner, shell, nshell);

    if (crank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(cart);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // --- chemical potential: needs halo of c ---
        halo.fillPhysicalBoundaries(cold.data());
        halo.start(cold.data());
        if (!boxEmpty(inner))
            computeChemicalPotentialBox(cold.data(), mu.data(), g, inner, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        halo.finish();
        for (int s = 0; s < nshell; ++s)
            computeChemicalPotentialBox(cold.data(), mu.data(), g, shell[s], dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // --- concentration update: needs halo of mu ---
        halo.fillPhysicalBoundaries(mu.data());
        halo.start(mu.data());
        if (!boxEmpty(inner))
            cahnHilliardUpdateBox(cnew.data(), cold.data(), mu.data(), g, inner, D, dt, dx, dy, dz);
        halo.finish();
        for (int s = 0; s < nshell; ++s)
            cahnHilliardUpdateBox(cnew.data(), cold.data(), mu.data(), g, shell[s], D, dt, dx, dy, dz);

        std::swap(cold, cnew);
    }

    const double end = MPI_Wtime();
    double elapsed = end - start;
    double maxElapsed = elapsed;
    MPI_Allreduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, cart);

    const long ms = static_cast<long>(maxElapsed * 1000.0);
    if (crank == 0) {
        printf("Computation time: %ld ms\n", ms);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> global;
        gatherField(cold, g, cart, crank, csize, nx, ny, nz, global);
        if (crank == 0) print_results(global, "Concentration");
    }

    int ret = 0;
    if (validate) {
        if (crank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold, g, cart, crank);
        if (valid) {
            if (crank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (crank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    MPI_Comm_free(&cart);
    MPI_Comm_free(&work);
    MPI_Finalize();
    return ret;
}
