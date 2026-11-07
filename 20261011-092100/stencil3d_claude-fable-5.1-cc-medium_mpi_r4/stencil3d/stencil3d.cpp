#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// ---------------------------------------------------------------------------
// MPI parallelization: 3D Cartesian domain decomposition.
//
// The global nx x ny x nz grid is split into blocks, one per rank. Each rank
// stores its block plus a one-cell halo on every side. Global boundary cells
// (which the original code copies unchanged from input to output) are owned
// by the ranks at the edges of the process grid and are never recomputed.
//
// Per iteration: post non-blocking halo exchanges with the (up to) six face
// neighbors, compute the local cells that do not depend on halo data while
// the messages are in flight, then wait and compute the outer shell.
// ---------------------------------------------------------------------------

// 3D index calculation (global index, used for initialization semantics)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Domain {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[3] = {1, 1, 1};    // process grid: [x, y, z]
    int coords[3] = {0, 0, 0};
    int nbrLo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbrHi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    size_t nx = 0, ny = 0, nz = 0;  // global sizes
    size_t x0 = 0, y0 = 0, z0 = 0;  // global offset of local block
    size_t lx = 0, ly = 0, lz = 0;  // local block size (without halo)
    size_t sx = 0, sy = 0, sz = 0;  // local array size (with halo)

    // Local compute range (local coordinates, excludes global boundary cells)
    size_t ci0 = 0, ci1 = 0, cj0 = 0, cj1 = 0, ck0 = 0, ck1 = 0;

    size_t lidx(const size_t i, const size_t j, const size_t k) const noexcept {
        return (k + 1) * (sx * sy) + (j + 1) * sx + (i + 1);
    }
};

// Block partition of n cells over p processes: process c gets [start, start+len)
inline void blockRange(const size_t n, const int p, const int c, size_t& start, size_t& len) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    const size_t cc = static_cast<size_t>(c);
    start = cc * base + std::min(cc, rem);
    len = base + (cc < rem ? 1 : 0);
}

// Choose a process grid dx*dy*dz == p with dx<=nx, dy<=ny, dz<=nz that
// minimizes the halo surface area. Ties prefer splitting z (contiguous
// faces) over y over x (strided faces). Returns false if none exists.
static bool chooseDims(const int p, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    bool found = false;
    double bestCost = 0.0;
    for (int dx = 1; dx <= p; ++dx) {
        if (p % dx != 0 || static_cast<size_t>(dx) > nx) continue;
        const int rest = p / dx;
        for (int dy = 1; dy <= rest; ++dy) {
            if (rest % dy != 0 || static_cast<size_t>(dy) > ny) continue;
            const int dz = rest / dy;
            if (static_cast<size_t>(dz) > nz) continue;
            const double bx = static_cast<double>(nx) / dx;
            const double by = static_cast<double>(ny) / dy;
            const double bz = static_cast<double>(nz) / dz;
            // Surface area communicated; strided faces weighted slightly higher.
            const double cost = (dz > 1 ? 2.0 * bx * by : 0.0)
                              + (dy > 1 ? 2.2 * bx * bz : 0.0)
                              + (dx > 1 ? 2.5 * by * bz : 0.0);
            if (!found || cost < bestCost) {
                found = true;
                bestCost = cost;
                dims[0] = dx;
                dims[1] = dy;
                dims[2] = dz;
            }
        }
    }
    return found;
}

static bool setupDomain(Domain& d, const size_t nx, const size_t ny, const size_t nz) {
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;

    if (!chooseDims(d.size, nx, ny, nz, d.dims)) {
        return false;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims, periods, 1, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    MPI_Cart_coords(d.cart, d.rank, 3, d.coords);
    for (int a = 0; a < 3; ++a) {
        MPI_Cart_shift(d.cart, a, 1, &d.nbrLo[a], &d.nbrHi[a]);
    }

    blockRange(nx, d.dims[0], d.coords[0], d.x0, d.lx);
    blockRange(ny, d.dims[1], d.coords[1], d.y0, d.ly);
    blockRange(nz, d.dims[2], d.coords[2], d.z0, d.lz);
    d.sx = d.lx + 2;
    d.sy = d.ly + 2;
    d.sz = d.lz + 2;

    // Global interior is [1, n-1); clip to the local block.
    auto clip = [](const size_t n, const size_t o, const size_t l, size_t& c0, size_t& c1) {
        if (n < 3) { c0 = c1 = 0; return; }
        const size_t g0 = std::max<size_t>(o, 1);
        const size_t g1 = std::min<size_t>(o + l, n - 1);
        if (g1 <= g0) { c0 = c1 = 0; return; }
        c0 = g0 - o;
        c1 = g1 - o;
    };
    clip(nx, d.x0, d.lx, d.ci0, d.ci1);
    clip(ny, d.y0, d.ly, d.cj0, d.cj1);
    clip(nz, d.z0, d.lz, d.ck0, d.ck1);
    return true;
}

void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    for (size_t k = 0; k < d.lz; ++k) {
        for (size_t j = 0; j < d.ly; ++j) {
            Real* row = grid.data() + d.lidx(0, j, k);
            const size_t gbase = idx3(d.x0, d.y0 + j, d.z0 + k, d.nx, d.ny);
            for (size_t i = 0; i < d.lx; ++i) {
                row[i] = ((gbase + i) % 19) * 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Halo exchange
// ---------------------------------------------------------------------------
struct HaloBuffers {
    std::vector<Real> sendXlo, sendXhi, recvXlo, recvXhi;  // ly*lz
    std::vector<Real> sendYlo, sendYhi, recvYlo, recvYhi;  // lx*lz
    MPI_Request reqs[12];
    int nreq = 0;

    void allocate(const Domain& d) {
        const size_t nxf = d.ly * d.lz;
        const size_t nyf = d.lx * d.lz;
        sendXlo.resize(nxf); sendXhi.resize(nxf); recvXlo.resize(nxf); recvXhi.resize(nxf);
        sendYlo.resize(nyf); sendYhi.resize(nyf); recvYlo.resize(nyf); recvYhi.resize(nyf);
    }
};

static void packX(const std::vector<Real>& g, const Domain& d, const size_t i, std::vector<Real>& buf) {
    size_t n = 0;
    for (size_t k = 0; k < d.lz; ++k)
        for (size_t j = 0; j < d.ly; ++j)
            buf[n++] = g[d.lidx(i, j, k)];
}

static void unpackX(std::vector<Real>& g, const Domain& d, const size_t i, const std::vector<Real>& buf) {
    size_t n = 0;
    for (size_t k = 0; k < d.lz; ++k)
        for (size_t j = 0; j < d.ly; ++j)
            g[(k + 1) * (d.sx * d.sy) + (j + 1) * d.sx + i] = buf[n++];
}

static void packY(const std::vector<Real>& g, const Domain& d, const size_t j, std::vector<Real>& buf) {
    for (size_t k = 0; k < d.lz; ++k)
        std::memcpy(buf.data() + k * d.lx, g.data() + d.lidx(0, j, k), d.lx * sizeof(Real));
}

static void unpackY(std::vector<Real>& g, const Domain& d, const size_t j, const std::vector<Real>& buf) {
    for (size_t k = 0; k < d.lz; ++k)
        std::memcpy(g.data() + (k + 1) * (d.sx * d.sy) + j * d.sx + 1, buf.data() + k * d.lx, d.lx * sizeof(Real));
}

static void startHaloExchange(std::vector<Real>& g, const Domain& d, HaloBuffers& hb) {
    hb.nreq = 0;
    const int cntX = static_cast<int>(d.ly * d.lz);
    const int cntY = static_cast<int>(d.lx * d.lz);
    const int cntZ = static_cast<int>(d.sx * d.sy);  // whole plane incl. halo (contiguous)

    // Post receives first.
    if (d.nbrLo[0] != MPI_PROC_NULL)
        MPI_Irecv(hb.recvXlo.data(), cntX, MPI_DOUBLE, d.nbrLo[0], 0, d.cart, &hb.reqs[hb.nreq++]);
    if (d.nbrHi[0] != MPI_PROC_NULL)
        MPI_Irecv(hb.recvXhi.data(), cntX, MPI_DOUBLE, d.nbrHi[0], 1, d.cart, &hb.reqs[hb.nreq++]);
    if (d.nbrLo[1] != MPI_PROC_NULL)
        MPI_Irecv(hb.recvYlo.data(), cntY, MPI_DOUBLE, d.nbrLo[1], 2, d.cart, &hb.reqs[hb.nreq++]);
    if (d.nbrHi[1] != MPI_PROC_NULL)
        MPI_Irecv(hb.recvYhi.data(), cntY, MPI_DOUBLE, d.nbrHi[1], 3, d.cart, &hb.reqs[hb.nreq++]);
    if (d.nbrLo[2] != MPI_PROC_NULL)
        MPI_Irecv(g.data() + 0 * (d.sx * d.sy), cntZ, MPI_DOUBLE, d.nbrLo[2], 4, d.cart, &hb.reqs[hb.nreq++]);
    if (d.nbrHi[2] != MPI_PROC_NULL)
        MPI_Irecv(g.data() + (d.sz - 1) * (d.sx * d.sy), cntZ, MPI_DOUBLE, d.nbrHi[2], 5, d.cart, &hb.reqs[hb.nreq++]);

    // Sends: lo neighbor receives our first layer as its hi halo (tag 1), etc.
    if (d.nbrLo[0] != MPI_PROC_NULL) {
        packX(g, d, 0, hb.sendXlo);
        MPI_Isend(hb.sendXlo.data(), cntX, MPI_DOUBLE, d.nbrLo[0], 1, d.cart, &hb.reqs[hb.nreq++]);
    }
    if (d.nbrHi[0] != MPI_PROC_NULL) {
        packX(g, d, d.lx - 1, hb.sendXhi);
        MPI_Isend(hb.sendXhi.data(), cntX, MPI_DOUBLE, d.nbrHi[0], 0, d.cart, &hb.reqs[hb.nreq++]);
    }
    if (d.nbrLo[1] != MPI_PROC_NULL) {
        packY(g, d, 0, hb.sendYlo);
        MPI_Isend(hb.sendYlo.data(), cntY, MPI_DOUBLE, d.nbrLo[1], 3, d.cart, &hb.reqs[hb.nreq++]);
    }
    if (d.nbrHi[1] != MPI_PROC_NULL) {
        packY(g, d, d.ly - 1, hb.sendYhi);
        MPI_Isend(hb.sendYhi.data(), cntY, MPI_DOUBLE, d.nbrHi[1], 2, d.cart, &hb.reqs[hb.nreq++]);
    }
    if (d.nbrLo[2] != MPI_PROC_NULL)
        MPI_Isend(g.data() + 1 * (d.sx * d.sy), cntZ, MPI_DOUBLE, d.nbrLo[2], 5, d.cart, &hb.reqs[hb.nreq++]);
    if (d.nbrHi[2] != MPI_PROC_NULL)
        MPI_Isend(g.data() + (d.sz - 2) * (d.sx * d.sy), cntZ, MPI_DOUBLE, d.nbrHi[2], 4, d.cart, &hb.reqs[hb.nreq++]);
}

static void finishHaloExchange(std::vector<Real>& g, const Domain& d, HaloBuffers& hb) {
    MPI_Waitall(hb.nreq, hb.reqs, MPI_STATUSES_IGNORE);
    if (d.nbrLo[0] != MPI_PROC_NULL) unpackX(g, d, 0, hb.recvXlo);
    if (d.nbrHi[0] != MPI_PROC_NULL) unpackX(g, d, d.sx - 1, hb.recvXhi);
    if (d.nbrLo[1] != MPI_PROC_NULL) unpackY(g, d, 0, hb.recvYlo);
    if (d.nbrHi[1] != MPI_PROC_NULL) unpackY(g, d, d.sy - 1, hb.recvYhi);
}

// ---------------------------------------------------------------------------
// 7-point stencil computation on a local sub-box [i0,i1) x [j0,j1) x [k0,k1)
// (local coordinates), clipped to the rank's compute range.
// ---------------------------------------------------------------------------
static void stencilBox(const Real* __restrict in, Real* __restrict out, const Domain& d,
                       size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1) {
    i0 = std::max(i0, d.ci0); i1 = std::min(i1, d.ci1);
    j0 = std::max(j0, d.cj0); j1 = std::min(j1, d.cj1);
    k0 = std::max(k0, d.ck0); k1 = std::min(k1, d.ck1);
    if (i0 >= i1 || j0 >= j1 || k0 >= k1) return;

    const size_t sx = d.sx;
    const size_t sxy = d.sx * d.sy;
    for (size_t k = k0; k < k1; ++k) {
        for (size_t j = j0; j < j1; ++j) {
            const size_t base = (k + 1) * sxy + (j + 1) * sx + 1;
            const Real* __restrict c = in + base;
            Real* __restrict o = out + base;
            for (size_t i = i0; i < i1; ++i) {
                const Real center = c[i];
                const Real left = c[i - 1];
                const Real right = c[i + 1];
                const Real front = c[i - sx];
                const Real back = c[i + sx];
                const Real bottom = c[i - sxy];
                const Real top = c[i + sxy];
                // Simple averaging stencil (same operation order as the original)
                o[i] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// One stencil iteration: input -> output. Global boundary cells are not
// touched; they hold their initial values in both buffers (see main).
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output, const Domain& d, HaloBuffers& hb) {
    startHaloExchange(input, d, hb);

    const Real* in = input.data();
    Real* out = output.data();
    const size_t lx = d.lx, ly = d.ly, lz = d.lz;

    // Inner box: cells whose neighbors are all local.
    if (lx >= 3 && ly >= 3 && lz >= 3) {
        stencilBox(in, out, d, 1, lx - 1, 1, ly - 1, 1, lz - 1);
    }

    finishHaloExchange(input, d, hb);

    // Outer shell: cells touching at least one local face.
    // z faces (full x/y extent)
    stencilBox(in, out, d, 0, lx, 0, ly, 0, 1);
    if (lz > 1) stencilBox(in, out, d, 0, lx, 0, ly, lz - 1, lz);
    // y faces (interior z)
    stencilBox(in, out, d, 0, lx, 0, 1, 1, lz - 1);
    if (ly > 1) stencilBox(in, out, d, 0, lx, ly - 1, ly, 1, lz - 1);
    // x faces (interior y and z)
    stencilBox(in, out, d, 0, 1, 1, ly - 1, 1, lz - 1);
    if (lx > 1) stencilBox(in, out, d, lx - 1, lx, 1, ly - 1, 1, lz - 1);
}

// Gather the distributed grid into a global array on rank 0 (global index order).
static void gatherGrid(const std::vector<Real>& local, const Domain& d, std::vector<Real>& global) {
    std::vector<Real> packed(d.lx * d.ly * d.lz);
    size_t n = 0;
    for (size_t k = 0; k < d.lz; ++k)
        for (size_t j = 0; j < d.ly; ++j)
            for (size_t i = 0; i < d.lx; ++i)
                packed[n++] = local[d.lidx(i, j, k)];

    std::vector<int> counts, displs;
    std::vector<Real> recv;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
        size_t off = 0;
        for (int r = 0; r < d.size; ++r) {
            int c[3];
            MPI_Cart_coords(d.cart, r, 3, c);
            size_t o, lxr, lyr, lzr;
            blockRange(d.nx, d.dims[0], c[0], o, lxr);
            blockRange(d.ny, d.dims[1], c[1], o, lyr);
            blockRange(d.nz, d.dims[2], c[2], o, lzr);
            counts[r] = static_cast<int>(lxr * lyr * lzr);
            displs[r] = static_cast<int>(off);
            off += lxr * lyr * lzr;
        }
        recv.resize(off);
    }
    MPI_Gatherv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                recv.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, d.cart);

    if (d.rank == 0) {
        global.assign(d.nx * d.ny * d.nz, 0.0);
        for (int r = 0; r < d.size; ++r) {
            int c[3];
            MPI_Cart_coords(d.cart, r, 3, c);
            size_t xo, yo, zo, lxr, lyr, lzr;
            blockRange(d.nx, d.dims[0], c[0], xo, lxr);
            blockRange(d.ny, d.dims[1], c[1], yo, lyr);
            blockRange(d.nz, d.dims[2], c[2], zo, lzr);
            const Real* src = recv.data() + displs[r];
            for (size_t k = 0; k < lzr; ++k)
                for (size_t j = 0; j < lyr; ++j)
                    std::memcpy(global.data() + idx3(xo, yo + j, zo + k, d.nx, d.ny),
                                src + (k * lyr + j) * lxr, lxr * sizeof(Real));
        }
    }
}

// Distributed validation: same checks and messages as the original.
bool validateResult(const std::vector<Real>& grid, const Domain& d) {
    // 1. No NaN or Inf values; 2. value range (local pass, then reductions)
    int badLocal = 0;
    Real minVal = grid[d.lidx(0, 0, 0)];
    Real maxVal = minVal;
    for (size_t k = 0; k < d.lz; ++k) {
        for (size_t j = 0; j < d.ly; ++j) {
            const Real* row = grid.data() + d.lidx(0, j, k);
            for (size_t i = 0; i < d.lx; ++i) {
                const Real val = row[i];
                if (std::isnan(val) || std::isinf(val)) badLocal = 1;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }
    int bad = 0;
    MPI_Allreduce(&badLocal, &bad, 1, MPI_INT, MPI_LOR, d.cart);
    if (bad) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    Real gmin = 0, gmax = 0;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, d.cart);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, d.cart);

    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);

    // After averaging, values should be somewhat bounded
    if (gmax > 1e6 || gmin < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

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
    int exitCode = 0;

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

    Domain d;
    if (!setupDomain(d, nx, ny, nz)) {
        if (worldRank == 0) {
            printf("Error: cannot decompose a %zu x %zu x %zu grid over %d MPI ranks\n", nx, ny, nz, d.size);
        }
        MPI_Finalize();
        return 1;
    }
    const bool root = (d.rank == 0);

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid %d x %d x %d)\n", d.size, d.dims[0], d.dims[1], d.dims[2]);
    }

    const size_t localSize = d.sx * d.sy * d.sz;

    // Allocate grids (double buffering), including halo layers
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize, 0.0);

    // Initialize
    if (root) printf("Initializing grid...\n");
    initializeGrid(grid1, d);
    // Global boundary cells are never recomputed and are copied input->output
    // in the original; initializing both buffers identically makes them
    // correct in whichever buffer ends up holding the final result.
    initializeGrid(grid2, d);

    HaloBuffers hb;
    hb.allocate(d);

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(d.cart);
    const double tStart = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, d, hb);
        } else {
            stencilIteration(grid2, grid1, d, hb);
        }
    }

    const double tLocal = MPI_Wtime() - tStart;
    double tMax = 0.0;
    MPI_Reduce(&tLocal, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, d.cart);
    const long durationMs = static_cast<long>(tMax * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> global;
        gatherGrid(finalGrid, d, global);
        if (root) print_results(global, "Grid");
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateResult(finalGrid, d);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Comm_free(&d.cart);
    MPI_Finalize();
    return exitCode;
}
