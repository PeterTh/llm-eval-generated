#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Domain decomposition: 3D Cartesian grid of ranks. Each rank owns a block of
// lnx x lny x lnz cells and stores it in a local array with a one-cell halo on
// every side: (lnx+2) x (lny+2) x (lnz+2). Local index 1..lnx maps to global
// index gx0..gx0+lnx-1. Global boundary cells (which the original code copies
// unchanged every iteration) live in the local array of the owning rank and
// are simply never updated, which is semantically identical.
// ---------------------------------------------------------------------------
struct Domain {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1};    // [0]=z, [1]=y, [2]=x
    int coords[3] = {0, 0, 0};
    int nbrLo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbrHi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    size_t nx = 0, ny = 0, nz = 0;       // global sizes
    size_t lnx = 0, lny = 0, lnz = 0;    // owned block sizes
    size_t gx0 = 0, gy0 = 0, gz0 = 0;    // global offset of owned block
    size_t sx = 0, sy = 0, sz = 0;       // local (halo-padded) array sizes

    // Compute range in local coordinates [c0, c1) for interior points only.
    size_t cx0 = 0, cx1 = 0, cy0 = 0, cy1 = 0, cz0 = 0, cz1 = 0;

    // Pack buffers for the strided faces (x and y). z faces are contiguous.
    std::vector<Real> sendX[2], recvX[2], sendY[2], recvY[2];

    size_t lidx(const size_t x, const size_t y, const size_t z) const noexcept {
        return idx3(x, y, z, sx, sy);
    }
};

static void splitRange(const size_t n, const int parts, const int part, size_t& start, size_t& len) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t rem = n % static_cast<size_t>(parts);
    const size_t p = static_cast<size_t>(part);
    len = base + (p < rem ? 1 : 0);
    start = p * base + std::min(p, rem);
}

static void setupDomain(Domain& d, const size_t nx, const size_t ny, const size_t nz) {
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    d.nx = nx; d.ny = ny; d.nz = nz;

    // Pick a process grid. MPI_Dims_create returns non-increasing factors, so
    // the largest factor is applied to z (contiguous planes -> cheapest halo
    // exchange) and the smallest to x (keeps the unit-stride rows long for
    // vectorisation). Never split a dimension finer than its cell count.
    const size_t gsz[3] = {nz, ny, nx};
    int dims[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        if (gsz[i] <= 1) dims[i] = 1;
    }
    MPI_Dims_create(d.size, 3, dims);
    for (int i = 0; i < 3; ++i) {
        if (static_cast<size_t>(dims[i]) > gsz[i] && gsz[i] > 0) {
            // Fall back to a decomposition that does not over-split: fix this
            // dimension and let the others absorb the remaining ranks.
            int fixed[3] = {0, 0, 0};
            for (int j = 0; j < 3; ++j) fixed[j] = (static_cast<size_t>(dims[j]) > gsz[j]) ? static_cast<int>(std::max<size_t>(gsz[j], 1)) : 0;
            int prod = 1;
            for (int j = 0; j < 3; ++j) if (fixed[j]) prod *= fixed[j];
            if (d.size % prod == 0) {
                MPI_Dims_create(d.size, 3, fixed);
                for (int j = 0; j < 3; ++j) dims[j] = fixed[j];
            }
            break;
        }
    }
    for (int i = 0; i < 3; ++i) d.dims[i] = dims[i];

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims, periods, 1, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    MPI_Cart_coords(d.cart, d.rank, 3, d.coords);
    for (int i = 0; i < 3; ++i) {
        MPI_Cart_shift(d.cart, i, 1, &d.nbrLo[i], &d.nbrHi[i]);
    }

    splitRange(nz, d.dims[0], d.coords[0], d.gz0, d.lnz);
    splitRange(ny, d.dims[1], d.coords[1], d.gy0, d.lny);
    splitRange(nx, d.dims[2], d.coords[2], d.gx0, d.lnx);
    d.sx = d.lnx + 2; d.sy = d.lny + 2; d.sz = d.lnz + 2;

    // Interior (updated) range in local coordinates. Global boundary cells
    // are excluded exactly as in the original code.
    auto range = [](const size_t g0, const size_t ln, const size_t n, size_t& c0, size_t& c1) {
        c0 = 1; c1 = ln + 1;
        if (g0 == 0) c0 = 2;                       // global index 0 is local 1
        if (g0 + ln == n) c1 = (ln >= 1) ? ln : 0; // global index n-1 is local ln
        if (c1 < c0) c1 = c0;
    };
    range(d.gx0, d.lnx, nx, d.cx0, d.cx1);
    range(d.gy0, d.lny, ny, d.cy0, d.cy1);
    range(d.gz0, d.lnz, nz, d.cz0, d.cz1);

    for (int s = 0; s < 2; ++s) {
        d.sendX[s].assign(d.lny * d.lnz, 0.0);
        d.recvX[s].assign(d.lny * d.lnz, 0.0);
        d.sendY[s].assign(d.lnx * d.lnz, 0.0);
        d.recvY[s].assign(d.lnx * d.lnz, 0.0);
    }
}

void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    std::fill(grid.begin(), grid.end(), 0.0);
    for (size_t z = 0; z < d.lnz; ++z) {
        for (size_t y = 0; y < d.lny; ++y) {
            Real* row = &grid[d.lidx(1, y + 1, z + 1)];
            const size_t gbase = idx3(d.gx0, d.gy0 + y, d.gz0 + z, d.nx, d.ny);
            for (size_t x = 0; x < d.lnx; ++x) {
                row[x] = static_cast<Real>((gbase + x) % 19) * 1.0;
            }
        }
    }
}

// Apply the 7-point stencil on the local sub-box [x0,x1) x [y0,y1) x [z0,z1).
static inline void stencilRegion(const Real* __restrict__ in, Real* __restrict__ out,
                                 const size_t sx, const size_t sy,
                                 const size_t x0, const size_t x1,
                                 const size_t y0, const size_t y1,
                                 const size_t z0, const size_t z1) {
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    const size_t plane = sx * sy;
    for (size_t z = z0; z < z1; ++z) {
        for (size_t y = y0; y < y1; ++y) {
            const size_t base = z * plane + y * sx;
            const Real* __restrict__ c = in + base;
            const Real* __restrict__ ym = c - sx;
            const Real* __restrict__ yp = c + sx;
            const Real* __restrict__ zm = c - plane;
            const Real* __restrict__ zp = c + plane;
            Real* __restrict__ o = out + base;
            for (size_t x = x0; x < x1; ++x) {
                const Real center = c[x];
                const Real left = c[x - 1];
                const Real right = c[x + 1];
                const Real front = ym[x];
                const Real back = yp[x];
                const Real bottom = zm[x];
                const Real top = zp[x];
                // Simple averaging stencil (same operation order as original)
                o[x] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// Exchange halos for `input`, overlapping with computation of the inner part
// of the block; then finish the faces of the block.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output, Domain& d) {
    Real* in = input.data();
    Real* out = output.data();
    const size_t sx = d.sx, sy = d.sy;
    const size_t lnx = d.lnx, lny = d.lny, lnz = d.lnz;
    const size_t plane = sx * sy;

    MPI_Request reqs[12];
    int nreq = 0;

    // --- z faces: contiguous planes (include halo rows; harmless) ---
    const int zPlane = static_cast<int>(plane);
    if (d.nbrLo[0] != MPI_PROC_NULL) {
        MPI_Irecv(in + d.lidx(0, 0, 0), zPlane, MPI_DOUBLE, d.nbrLo[0], 0, d.cart, &reqs[nreq++]);
        MPI_Isend(in + d.lidx(0, 0, 1), zPlane, MPI_DOUBLE, d.nbrLo[0], 1, d.cart, &reqs[nreq++]);
    }
    if (d.nbrHi[0] != MPI_PROC_NULL) {
        MPI_Irecv(in + d.lidx(0, 0, lnz + 1), zPlane, MPI_DOUBLE, d.nbrHi[0], 1, d.cart, &reqs[nreq++]);
        MPI_Isend(in + d.lidx(0, 0, lnz), zPlane, MPI_DOUBLE, d.nbrHi[0], 0, d.cart, &reqs[nreq++]);
    }

    // --- y faces: rows of length lnx per z plane ---
    const int yCount = static_cast<int>(lnx * lnz);
    if (d.nbrLo[1] != MPI_PROC_NULL) {
        Real* sb = d.sendY[0].data();
        for (size_t z = 0; z < lnz; ++z)
            std::memcpy(sb + z * lnx, in + d.lidx(1, 1, z + 1), lnx * sizeof(Real));
        MPI_Irecv(d.recvY[0].data(), yCount, MPI_DOUBLE, d.nbrLo[1], 2, d.cart, &reqs[nreq++]);
        MPI_Isend(sb, yCount, MPI_DOUBLE, d.nbrLo[1], 3, d.cart, &reqs[nreq++]);
    }
    if (d.nbrHi[1] != MPI_PROC_NULL) {
        Real* sb = d.sendY[1].data();
        for (size_t z = 0; z < lnz; ++z)
            std::memcpy(sb + z * lnx, in + d.lidx(1, lny, z + 1), lnx * sizeof(Real));
        MPI_Irecv(d.recvY[1].data(), yCount, MPI_DOUBLE, d.nbrHi[1], 3, d.cart, &reqs[nreq++]);
        MPI_Isend(sb, yCount, MPI_DOUBLE, d.nbrHi[1], 2, d.cart, &reqs[nreq++]);
    }

    // --- x faces: fully strided ---
    const int xCount = static_cast<int>(lny * lnz);
    if (d.nbrLo[2] != MPI_PROC_NULL) {
        Real* sb = d.sendX[0].data();
        for (size_t z = 0; z < lnz; ++z)
            for (size_t y = 0; y < lny; ++y)
                sb[z * lny + y] = in[d.lidx(1, y + 1, z + 1)];
        MPI_Irecv(d.recvX[0].data(), xCount, MPI_DOUBLE, d.nbrLo[2], 4, d.cart, &reqs[nreq++]);
        MPI_Isend(sb, xCount, MPI_DOUBLE, d.nbrLo[2], 5, d.cart, &reqs[nreq++]);
    }
    if (d.nbrHi[2] != MPI_PROC_NULL) {
        Real* sb = d.sendX[1].data();
        for (size_t z = 0; z < lnz; ++z)
            for (size_t y = 0; y < lny; ++y)
                sb[z * lny + y] = in[d.lidx(lnx, y + 1, z + 1)];
        MPI_Irecv(d.recvX[1].data(), xCount, MPI_DOUBLE, d.nbrHi[2], 5, d.cart, &reqs[nreq++]);
        MPI_Isend(sb, xCount, MPI_DOUBLE, d.nbrHi[2], 4, d.cart, &reqs[nreq++]);
    }

    // --- compute inner region (does not touch halos) while messages fly ---
    const size_t ix0 = std::max(d.cx0, size_t{2}), ix1 = std::min(d.cx1, lnx);
    const size_t iy0 = std::max(d.cy0, size_t{2}), iy1 = std::min(d.cy1, lny);
    const size_t iz0 = std::max(d.cz0, size_t{2}), iz1 = std::min(d.cz1, lnz);
    const bool innerOk = ix0 < ix1 && iy0 < iy1 && iz0 < iz1;
    if (innerOk) stencilRegion(in, out, sx, sy, ix0, ix1, iy0, iy1, iz0, iz1);

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // --- unpack received halos ---
    if (d.nbrLo[1] != MPI_PROC_NULL) {
        const Real* rb = d.recvY[0].data();
        for (size_t z = 0; z < lnz; ++z)
            std::memcpy(in + d.lidx(1, 0, z + 1), rb + z * lnx, lnx * sizeof(Real));
    }
    if (d.nbrHi[1] != MPI_PROC_NULL) {
        const Real* rb = d.recvY[1].data();
        for (size_t z = 0; z < lnz; ++z)
            std::memcpy(in + d.lidx(1, lny + 1, z + 1), rb + z * lnx, lnx * sizeof(Real));
    }
    if (d.nbrLo[2] != MPI_PROC_NULL) {
        const Real* rb = d.recvX[0].data();
        for (size_t z = 0; z < lnz; ++z)
            for (size_t y = 0; y < lny; ++y)
                in[d.lidx(0, y + 1, z + 1)] = rb[z * lny + y];
    }
    if (d.nbrHi[2] != MPI_PROC_NULL) {
        const Real* rb = d.recvX[1].data();
        for (size_t z = 0; z < lnz; ++z)
            for (size_t y = 0; y < lny; ++y)
                in[d.lidx(lnx + 1, y + 1, z + 1)] = rb[z * lny + y];
    }

    // --- compute the faces of the block that depend on halo data ---
    if (!innerOk) {
        stencilRegion(in, out, sx, sy, d.cx0, d.cx1, d.cy0, d.cy1, d.cz0, d.cz1);
        return;
    }
    // z slabs (full x/y extent)
    stencilRegion(in, out, sx, sy, d.cx0, d.cx1, d.cy0, d.cy1, d.cz0, iz0);
    stencilRegion(in, out, sx, sy, d.cx0, d.cx1, d.cy0, d.cy1, iz1, d.cz1);
    // y slabs (inner z, full x extent)
    stencilRegion(in, out, sx, sy, d.cx0, d.cx1, d.cy0, iy0, iz0, iz1);
    stencilRegion(in, out, sx, sy, d.cx0, d.cx1, iy1, d.cy1, iz0, iz1);
    // x slabs (inner y/z)
    stencilRegion(in, out, sx, sy, d.cx0, ix0, iy0, iy1, iz0, iz1);
    stencilRegion(in, out, sx, sy, ix1, d.cx1, iy0, iy1, iz0, iz1);
    // Boundary values: the owned global-boundary cells are never written and
    // both buffers hold the same initial values there, so they are "copied".
}

// Gather the distributed grid into a full global array on rank 0.
static void gatherGrid(const std::vector<Real>& local, const Domain& d, std::vector<Real>& full) {
    const size_t nx = d.nx, ny = d.ny, nz = d.nz;

    // Pack owned block contiguously.
    std::vector<Real> block(d.lnx * d.lny * d.lnz);
    for (size_t z = 0; z < d.lnz; ++z)
        for (size_t y = 0; y < d.lny; ++y)
            std::memcpy(&block[(z * d.lny + y) * d.lnx], &local[d.lidx(1, y + 1, z + 1)], d.lnx * sizeof(Real));

    // Every rank tells root where its block lives.
    long long meta[6] = {(long long)d.gx0, (long long)d.gy0, (long long)d.gz0,
                         (long long)d.lnx, (long long)d.lny, (long long)d.lnz};
    std::vector<long long> allMeta(d.rank == 0 ? 6 * (size_t)d.size : 0);
    MPI_Gather(meta, 6, MPI_LONG_LONG, allMeta.data(), 6, MPI_LONG_LONG, 0, d.cart);

    if (d.rank == 0) {
        full.assign(nx * ny * nz, 0.0);
        std::vector<MPI_Request> reqs;
        std::vector<MPI_Datatype> types;
        reqs.reserve(d.size);
        for (int r = 0; r < d.size; ++r) {
            const long long* m = &allMeta[6 * (size_t)r];
            if (m[3] == 0 || m[4] == 0 || m[5] == 0) continue;
            if (r == 0) continue;
            const int sizes[3] = {(int)nz, (int)ny, (int)nx};
            const int subs[3] = {(int)m[5], (int)m[4], (int)m[3]};
            const int starts[3] = {(int)m[2], (int)m[1], (int)m[0]};
            MPI_Datatype t;
            MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
            types.push_back(t);
            reqs.emplace_back();
            MPI_Irecv(full.data(), 1, t, r, 100, d.cart, &reqs.back());
        }
        // Own block.
        for (size_t z = 0; z < d.lnz; ++z)
            for (size_t y = 0; y < d.lny; ++y)
                std::memcpy(&full[idx3(d.gx0, d.gy0 + y, d.gz0 + z, nx, ny)],
                            &block[(z * d.lny + y) * d.lnx], d.lnx * sizeof(Real));
        if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        for (auto& t : types) MPI_Type_free(&t);
    } else if (!block.empty()) {
        MPI_Send(block.data(), (int)block.size(), MPI_DOUBLE, 0, 100, d.cart);
    }
}

bool validateResult(const std::vector<Real>& grid, const Domain& d) {
    // Simple sanity checks, evaluated over the owned cells and reduced globally.
    int badLocal = 0;
    Real minVal = INFINITY, maxVal = -INFINITY;
    for (size_t z = 0; z < d.lnz; ++z) {
        for (size_t y = 0; y < d.lny; ++y) {
            const Real* row = &grid[d.lidx(1, y + 1, z + 1)];
            for (size_t x = 0; x < d.lnx; ++x) {
                const Real val = row[x];
                if (std::isnan(val) || std::isinf(val)) badLocal = 1;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }
    int bad = 0;
    Real gmin = 0, gmax = 0;
    MPI_Allreduce(&badLocal, &bad, 1, MPI_INT, MPI_MAX, d.cart);
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, d.cart);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, d.cart);

    // 1. No NaN or Inf values
    if (bad) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);

    // After averaging, values should be somewhat bounded
    if (gmax > 1e6 || gmin < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are never written, so they equal the initial values)

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
    const bool isRoot = (worldRank == 0);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
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
    setupDomain(d, nx, ny, nz);

    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid %d x %d x %d)\n", d.size, d.dims[2], d.dims[1], d.dims[0]);
    }

    const size_t localSize = d.sx * d.sy * d.sz;

    // Allocate grids (double buffering)
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize
    if (isRoot) printf("Initializing grid...\n");
    initializeGrid(grid1, d);
    grid2 = grid1;  // boundary cells must hold initial values in both buffers

    // Run stencil iterations
    if (isRoot) printf("Running stencil computation...\n");
    MPI_Barrier(d.cart);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, d);
        } else {
            stencilIteration(grid2, grid1, d);
        }
    }

    const double end = MPI_Wtime();
    const double localElapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Allreduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, d.cart);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (isRoot) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / maxElapsed / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> full;
        gatherGrid(finalGrid, d, full);
        if (isRoot) print_results(full, "Grid");
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        bool valid = validateResult(finalGrid, d);

        if (valid) {
            if (isRoot) printf("Validation: PASSED\n");
            rc = 0;
        } else {
            if (isRoot) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Comm_free(&d.cart);
    MPI_Finalize();
    return rc;
}
