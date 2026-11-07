#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Domain decomposition
//
// The global nx x ny x nz grid is split over a 3D Cartesian process grid.
// Each rank owns a block [gx0, gx0+lx) x [gy0, gy0+ly) x [gz0, gz0+lz) of the
// global domain and stores it in a local array padded by one halo cell on each
// side (dimensions (lx+2) x (ly+2) x (lz+2)).  Local index (i,j,k) with
// 1 <= i <= lx corresponds to global index gx0 + i - 1.
//
// Halo cells that lie on the global boundary are never used by the stencil
// (the global boundary itself is kept constant, exactly as in the original
// code which copies the boundary from input to output every iteration).
// ---------------------------------------------------------------------------
struct Domain {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1};    // process grid (x, y, z)
    int coords[3] = {0, 0, 0};
    int nbr_lo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbr_hi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    size_t gn[3] = {0, 0, 0};   // global sizes
    size_t g0[3] = {0, 0, 0};   // global start of owned block
    size_t ln[3] = {0, 0, 0};   // owned block sizes
    size_t pn[3] = {0, 0, 0};   // padded local sizes (ln + 2)

    // Range of *local* indices (inclusive start, exclusive end) that are
    // global-interior and therefore updated by the stencil.
    size_t ub[3] = {1, 1, 1};
    size_t ue[3] = {1, 1, 1};

    size_t padded_size() const { return pn[0] * pn[1] * pn[2]; }
    size_t lidx(size_t i, size_t j, size_t k) const { return idx3(i, j, k, pn[0], pn[1]); }
};

static void splitRange(size_t n, int parts, int p, size_t& start, size_t& count) {
    const size_t base = n / (size_t)parts;
    const size_t rem = n % (size_t)parts;
    const size_t up = (size_t)p;
    start = up * base + std::min(up, rem);
    count = base + (up < rem ? 1 : 0);
}

static Domain createDomain(size_t nx, size_t ny, size_t nz) {
    Domain d;
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    // Choose the process grid (px, py, pz) with px*py*pz == size that
    // minimises the per-rank halo surface (i.e. communication volume), while
    // never assigning more processes to a dimension than it has cells unless
    // unavoidable.  Ties favour splitting z, then y, keeping the contiguous x
    // extent long for vectorisation and contiguous face packing.
    {
        double best = -1.0;
        int bestDims[3] = {1, 1, d.size};
        const double n[3] = {(double)std::max<size_t>(nx, 1), (double)std::max<size_t>(ny, 1), (double)std::max<size_t>(nz, 1)};
        for (int px = 1; px <= d.size; ++px) {
            if (d.size % px) continue;
            for (int py = 1; py <= d.size / px; ++py) {
                if ((d.size / px) % py) continue;
                const int pz = d.size / px / py;
                const double lx = n[0] / px, ly = n[1] / py, lz = n[2] / pz;
                double cost = (px > 1 ? ly * lz : 0.0) + (py > 1 ? lx * lz : 0.0) + (pz > 1 ? lx * ly : 0.0);
                // Heavily penalise splits that leave ranks without cells.
                if (px > n[0]) cost += 1e30;
                if (py > n[1]) cost += 1e30;
                if (pz > n[2]) cost += 1e30;
                // Mild preference for fewer x splits, then fewer y splits.
                cost += px * 1e-6 + py * 1e-9;
                if (best < 0 || cost < best) { best = cost; bestDims[0] = px; bestDims[1] = py; bestDims[2] = pz; }
            }
        }
        d.dims[0] = bestDims[0]; d.dims[1] = bestDims[1]; d.dims[2] = bestDims[2];
    }

    int periods[3] = {0, 0, 0};
    int cdims[3] = {d.dims[2], d.dims[1], d.dims[0]};   // MPI orders z slowest; use (z,y,x)
    MPI_Cart_create(MPI_COMM_WORLD, 3, cdims, periods, 1, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    int ccoords[3];
    MPI_Cart_coords(d.cart, d.rank, 3, ccoords);
    d.coords[0] = ccoords[2];
    d.coords[1] = ccoords[1];
    d.coords[2] = ccoords[0];
    MPI_Cart_shift(d.cart, 2, 1, &d.nbr_lo[0], &d.nbr_hi[0]);  // x
    MPI_Cart_shift(d.cart, 1, 1, &d.nbr_lo[1], &d.nbr_hi[1]);  // y
    MPI_Cart_shift(d.cart, 0, 1, &d.nbr_lo[2], &d.nbr_hi[2]);  // z

    d.gn[0] = nx; d.gn[1] = ny; d.gn[2] = nz;
    for (int a = 0; a < 3; ++a) {
        splitRange(d.gn[a], d.dims[a], d.coords[a], d.g0[a], d.ln[a]);
        d.pn[a] = d.ln[a] + 2;
        // Local update range: global index must be in [1, gn-2].
        const size_t gbeg = std::max<size_t>(d.g0[a], 1);
        const size_t gend = (d.gn[a] >= 1) ? std::min(d.g0[a] + d.ln[a], d.gn[a] - 1) : 0;
        if (gend > gbeg) {
            d.ub[a] = gbeg - d.g0[a] + 1;
            d.ue[a] = gend - d.g0[a] + 1;
        } else {
            d.ub[a] = d.ue[a] = 1;
        }
    }
    return d;
}

// Initialise the owned block (plus any halo cells that lie on the global
// boundary, so boundary values are available without communication) using
// the same formula as the original code: value = (global_index % 19).
static void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    const size_t nx = d.gn[0], ny = d.gn[1], nz = d.gn[2];
    for (size_t k = 0; k < d.pn[2]; ++k) {
        const long gz = (long)d.g0[2] + (long)k - 1;
        if (gz < 0 || gz >= (long)nz) continue;
        for (size_t j = 0; j < d.pn[1]; ++j) {
            const long gy = (long)d.g0[1] + (long)j - 1;
            if (gy < 0 || gy >= (long)ny) continue;
            Real* row = &grid[d.lidx(0, j, k)];
            const size_t gbase = idx3(0, (size_t)gy, (size_t)gz, nx, ny);
            for (size_t i = 0; i < d.pn[0]; ++i) {
                const long gx = (long)d.g0[0] + (long)i - 1;
                if (gx < 0 || gx >= (long)nx) continue;
                row[i] = (Real)((gbase + (size_t)gx) % 19) * 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Halo exchange (non-blocking, manual packing for the strided x/y faces)
// ---------------------------------------------------------------------------
struct HaloExchange {
    const Domain* d = nullptr;
    std::vector<Real> sbuf[3][2], rbuf[3][2];  // [axis][lo/hi]
    MPI_Request reqs[12];
    int nreq = 0;

    void init(const Domain& dom) {
        d = &dom;
        const size_t face[3] = {dom.ln[1] * dom.ln[2], dom.ln[0] * dom.ln[2], dom.ln[0] * dom.ln[1]};
        for (int a = 0; a < 3; ++a) {
            for (int s = 0; s < 2; ++s) {
                sbuf[a][s].resize(face[a]);
                rbuf[a][s].resize(face[a]);
            }
        }
    }

    // Pack plane at local index `pos` along axis `a` into buf.
    void pack(const std::vector<Real>& g, int a, size_t pos, std::vector<Real>& buf) const {
        const Domain& D = *d;
        size_t n = 0;
        if (a == 0) {
            for (size_t k = 1; k <= D.ln[2]; ++k)
                for (size_t j = 1; j <= D.ln[1]; ++j)
                    buf[n++] = g[D.lidx(pos, j, k)];
        } else if (a == 1) {
            for (size_t k = 1; k <= D.ln[2]; ++k) {
                const Real* src = &g[D.lidx(1, pos, k)];
                std::memcpy(&buf[n], src, D.ln[0] * sizeof(Real));
                n += D.ln[0];
            }
        } else {
            for (size_t j = 1; j <= D.ln[1]; ++j) {
                const Real* src = &g[D.lidx(1, j, pos)];
                std::memcpy(&buf[n], src, D.ln[0] * sizeof(Real));
                n += D.ln[0];
            }
        }
    }

    void unpack(std::vector<Real>& g, int a, size_t pos, const std::vector<Real>& buf) const {
        const Domain& D = *d;
        size_t n = 0;
        if (a == 0) {
            for (size_t k = 1; k <= D.ln[2]; ++k)
                for (size_t j = 1; j <= D.ln[1]; ++j)
                    g[D.lidx(pos, j, k)] = buf[n++];
        } else if (a == 1) {
            for (size_t k = 1; k <= D.ln[2]; ++k) {
                std::memcpy(&g[D.lidx(1, pos, k)], &buf[n], D.ln[0] * sizeof(Real));
                n += D.ln[0];
            }
        } else {
            for (size_t j = 1; j <= D.ln[1]; ++j) {
                std::memcpy(&g[D.lidx(1, j, pos)], &buf[n], D.ln[0] * sizeof(Real));
                n += D.ln[0];
            }
        }
    }

    void start(const std::vector<Real>& g) {
        const Domain& D = *d;
        nreq = 0;
        for (int a = 0; a < 3; ++a) {
            const int cnt = (int)sbuf[a][0].size();
            if (D.nbr_lo[a] != MPI_PROC_NULL) {
                MPI_Irecv(rbuf[a][0].data(), cnt, MPI_DOUBLE, D.nbr_lo[a], 2 * a + 1, D.cart, &reqs[nreq++]);
            }
            if (D.nbr_hi[a] != MPI_PROC_NULL) {
                MPI_Irecv(rbuf[a][1].data(), cnt, MPI_DOUBLE, D.nbr_hi[a], 2 * a + 0, D.cart, &reqs[nreq++]);
            }
        }
        for (int a = 0; a < 3; ++a) {
            const int cnt = (int)sbuf[a][0].size();
            if (D.nbr_lo[a] != MPI_PROC_NULL) {
                pack(g, a, 1, sbuf[a][0]);
                MPI_Isend(sbuf[a][0].data(), cnt, MPI_DOUBLE, D.nbr_lo[a], 2 * a + 0, D.cart, &reqs[nreq++]);
            }
            if (D.nbr_hi[a] != MPI_PROC_NULL) {
                pack(g, a, D.ln[a], sbuf[a][1]);
                MPI_Isend(sbuf[a][1].data(), cnt, MPI_DOUBLE, D.nbr_hi[a], 2 * a + 1, D.cart, &reqs[nreq++]);
            }
        }
    }

    // Wait for all transfers and write received halos into g (const_cast-free:
    // the halo region of the *input* grid is written, which is what the
    // stencil reads).
    void finish(std::vector<Real>& g) {
        const Domain& D = *d;
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        for (int a = 0; a < 3; ++a) {
            if (D.nbr_lo[a] != MPI_PROC_NULL) unpack(g, a, 0, rbuf[a][0]);
            if (D.nbr_hi[a] != MPI_PROC_NULL) unpack(g, a, D.ln[a] + 1, rbuf[a][1]);
        }
    }
};

// ---------------------------------------------------------------------------
// 7-point stencil computation on a sub-box of the local grid
// ---------------------------------------------------------------------------
static inline void stencilBox(const Real* __restrict in, Real* __restrict out, const Domain& d,
                              size_t xb, size_t xe, size_t yb, size_t ye, size_t zb, size_t ze) {
    const size_t sx = 1, sy = d.pn[0], sz = d.pn[0] * d.pn[1];
    for (size_t k = zb; k < ze; ++k) {
        for (size_t j = yb; j < ye; ++j) {
            const size_t base = d.lidx(0, j, k);
            const Real* __restrict ic = in + base;
            Real* __restrict oc = out + base;
            for (size_t i = xb; i < xe; ++i) {
                const Real center = ic[i];
                const Real left = ic[i - sx];
                const Real right = ic[i + sx];
                const Real front = ic[i - sy];
                const Real back = ic[i + sy];
                const Real bottom = ic[i - sz];
                const Real top = ic[i + sz];
                // Same expression/order as the original for bit-identical results.
                oc[i] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// One stencil iteration: exchange halos of `input`, compute the part of the
// owned block that does not depend on halos while messages are in flight,
// then finish the halo-dependent shell.  Global boundary cells are never
// written: they are initialised identically in both buffers, which is
// equivalent to the original "copy boundary from input to output".
static void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                             const Domain& d, HaloExchange& hx) {
    hx.start(input);

    const size_t xb = d.ub[0], xe = d.ue[0];
    const size_t yb = d.ub[1], ye = d.ue[1];
    const size_t zb = d.ub[2], ze = d.ue[2];

    // Inner region: cells whose neighbours are all owned cells.
    const size_t ixb = std::max(xb, (size_t)2), ixe = std::min(xe, d.ln[0]);
    const size_t iyb = std::max(yb, (size_t)2), iye = std::min(ye, d.ln[1]);
    const size_t izb = std::max(zb, (size_t)2), ize = std::min(ze, d.ln[2]);
    const bool haveInner = ixb < ixe && iyb < iye && izb < ize;

    if (haveInner) stencilBox(input.data(), output.data(), d, ixb, ixe, iyb, iye, izb, ize);

    hx.finish(input);

    if (!haveInner) {
        stencilBox(input.data(), output.data(), d, xb, xe, yb, ye, zb, ze);
        return;
    }
    // Shell: z-slabs below/above the inner box (full x/y extent) ...
    if (zb < izb) stencilBox(input.data(), output.data(), d, xb, xe, yb, ye, zb, izb);
    if (ize < ze) stencilBox(input.data(), output.data(), d, xb, xe, yb, ye, ize, ze);
    // ... y-slabs within the inner z-range ...
    if (yb < iyb) stencilBox(input.data(), output.data(), d, xb, xe, yb, iyb, izb, ize);
    if (iye < ye) stencilBox(input.data(), output.data(), d, xb, xe, iye, ye, izb, ize);
    // ... x-slabs within the inner y/z-range.
    if (xb < ixb) stencilBox(input.data(), output.data(), d, xb, ixb, iyb, iye, izb, ize);
    if (ixe < xe) stencilBox(input.data(), output.data(), d, ixe, xe, iyb, iye, izb, ize);
}

// ---------------------------------------------------------------------------
// Gather the distributed grid into a full global array on rank 0 (used only
// for result printing).
// ---------------------------------------------------------------------------
static void gatherGrid(const std::vector<Real>& local, const Domain& d, std::vector<Real>& global) {
    const size_t nloc = d.ln[0] * d.ln[1] * d.ln[2];
    std::vector<Real> packed(nloc);
    {
        size_t n = 0;
        for (size_t k = 1; k <= d.ln[2]; ++k)
            for (size_t j = 1; j <= d.ln[1]; ++j) {
                std::memcpy(&packed[n], &local[d.lidx(1, j, k)], d.ln[0] * sizeof(Real));
                n += d.ln[0];
            }
    }
    // Describe every rank's block to rank 0.
    long long meta[6] = {(long long)d.g0[0], (long long)d.g0[1], (long long)d.g0[2],
                         (long long)d.ln[0], (long long)d.ln[1], (long long)d.ln[2]};
    std::vector<long long> allmeta(d.rank == 0 ? 6 * (size_t)d.size : 0);
    MPI_Gather(meta, 6, MPI_LONG_LONG, allmeta.data(), 6, MPI_LONG_LONG, 0, d.cart);

    std::vector<int> counts, displs;
    std::vector<Real> recv;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
        size_t off = 0;
        for (int r = 0; r < d.size; ++r) {
            const size_t c = (size_t)(allmeta[6 * r + 3] * allmeta[6 * r + 4] * allmeta[6 * r + 5]);
            counts[r] = (int)c;
            displs[r] = (int)off;
            off += c;
        }
        recv.resize(off);
    }
    MPI_Gatherv(packed.data(), (int)nloc, MPI_DOUBLE, recv.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, d.cart);

    if (d.rank == 0) {
        global.resize(d.gn[0] * d.gn[1] * d.gn[2]);
        for (int r = 0; r < d.size; ++r) {
            const size_t x0 = allmeta[6 * r + 0], y0 = allmeta[6 * r + 1], z0 = allmeta[6 * r + 2];
            const size_t lx = allmeta[6 * r + 3], ly = allmeta[6 * r + 4], lz = allmeta[6 * r + 5];
            const Real* src = recv.data() + displs[r];
            for (size_t k = 0; k < lz; ++k)
                for (size_t j = 0; j < ly; ++j) {
                    std::memcpy(&global[idx3(x0, y0 + j, z0 + k, d.gn[0], d.gn[1])], src, lx * sizeof(Real));
                    src += lx;
                }
        }
    }
}

// Validation over the distributed grid (same checks and output as the original).
static bool validateResult(const std::vector<Real>& local, const Domain& d) {
    int badLocal = 0;
    Real minLocal = INFINITY, maxLocal = -INFINITY;
    for (size_t k = 1; k <= d.ln[2]; ++k)
        for (size_t j = 1; j <= d.ln[1]; ++j) {
            const Real* row = &local[d.lidx(1, j, k)];
            for (size_t i = 0; i < d.ln[0]; ++i) {
                const Real val = row[i];
                if (std::isnan(val) || std::isinf(val)) badLocal = 1;
                minLocal = std::min(minLocal, val);
                maxLocal = std::max(maxLocal, val);
            }
        }
    int bad = 0;
    Real minVal = 0, maxVal = 0;
    MPI_Allreduce(&badLocal, &bad, 1, MPI_INT, MPI_MAX, d.cart);
    MPI_Allreduce(&minLocal, &minVal, 1, MPI_DOUBLE, MPI_MIN, d.cart);
    MPI_Allreduce(&maxLocal, &maxVal, 1, MPI_DOUBLE, MPI_MAX, d.cart);

    if (bad) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    Domain d = createDomain(nx, ny, nz);
    HaloExchange hx;
    hx.init(d);

    if (d.rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid %d x %d x %d)\n", d.size, d.dims[0], d.dims[1], d.dims[2]);
    }

    // Allocate local grids (double buffering), halo-padded.
    std::vector<Real> grid1(d.padded_size(), 0.0);
    std::vector<Real> grid2(d.padded_size(), 0.0);

    // Initialize.  Both buffers get the initial values so that global
    // boundary cells (never updated) are correct in whichever buffer ends up
    // holding the final result.
    if (d.rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, d);
    grid2 = grid1;

    // Run stencil iterations
    if (d.rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(d.cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, d, hx);
        } else {
            stencilIteration(grid2, grid1, d, hx);
        }
    }

    MPI_Barrier(d.cart);
    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, d.cart);
    auto duration = std::chrono::milliseconds(maxDuration);

    if (d.rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> global;
        gatherGrid(finalGrid, d, global);
        if (d.rank == 0) print_results(global, "Grid");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (d.rank == 0) printf("Validating result...\n");
        bool valid = validateResult(finalGrid, d);

        if (valid) {
            if (d.rank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (d.rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    MPI_Comm_free(&d.cart);
    MPI_Finalize();
    return ret;
}
