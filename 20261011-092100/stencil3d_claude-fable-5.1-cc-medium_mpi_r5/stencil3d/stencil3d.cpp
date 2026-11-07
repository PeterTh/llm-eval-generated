#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (global grid, used for initialization semantics)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Domain decomposition description
// ---------------------------------------------------------------------------
struct Decomp {
    int dims[3];      // process grid
    int coords[3];    // my coordinates in the process grid
    int nbr[3][2];    // neighbor ranks: [dim][0]=lower, [dim][1]=upper (MPI_PROC_NULL if none)
    int start[3];     // global start index (x,y,z) of my block
    int len[3];       // local block extents (x,y,z) without halo
};

// Split n cells over p blocks; block i gets its start and length.
static inline void blockRange(const size_t n, const int p, const int i, int& start, int& len) {
    const size_t base = n / (size_t)p;
    const size_t rem = n % (size_t)p;
    const size_t s = (size_t)i * base + std::min((size_t)i, rem);
    const size_t l = base + ((size_t)i < rem ? 1 : 0);
    start = (int)s;
    len = (int)l;
}

// Choose a 3D process grid minimizing halo surface, subject to every block
// being non-empty (and, if possible, every block having at least one
// interior cell in every dimension so that halo exchange is useful).
static void chooseDims(const int size, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    const size_t n[3] = {nx, ny, nz};
    double bestCost = -1.0;
    int best[3] = {1, 1, size};
    bool haveFeasible = false;
    for (int dx = 1; dx <= size; ++dx) {
        if (size % dx) continue;
        const int rest = size / dx;
        for (int dy = 1; dy <= rest; ++dy) {
            if (rest % dy) continue;
            const int dz = rest / dy;
            const int d[3] = {dx, dy, dz};
            bool ok = true;
            for (int k = 0; k < 3; ++k) {
                if ((size_t)d[k] > n[k]) ok = false;
            }
            if (!ok) continue;
            // Communication volume estimate: sum of block face areas.
            const double bx = (double)nx / dx, by = (double)ny / dy, bz = (double)nz / dz;
            const double cost = (dx > 1 ? by * bz : 0.0) + (dy > 1 ? bx * bz : 0.0) + (dz > 1 ? bx * by : 0.0);
            if (!haveFeasible || cost < bestCost) {
                bestCost = cost;
                best[0] = dx; best[1] = dy; best[2] = dz;
                haveFeasible = true;
            }
        }
    }
    if (!haveFeasible) {
        // Grid too small for this many ranks; fall back to MPI's choice and
        // tolerate empty blocks (they simply do no work).
        best[0] = best[1] = best[2] = 0;
        MPI_Dims_create(size, 3, best);
    }
    dims[0] = best[0]; dims[1] = best[1]; dims[2] = best[2];
}

// ---------------------------------------------------------------------------
// Local grid with one-cell halo in every direction.
// Local index (i,j,k) with i in [0, len0+1] etc.; cell (i,j,k) corresponds to
// global (start0+i-1, start1+j-1, start2+k-1).
// ---------------------------------------------------------------------------
struct LocalGrid {
    int lx, ly, lz;      // interior extents
    int sx, sy, sz;      // strides (sx = 1)
    size_t total;
    inline size_t id(const int i, const int j, const int k) const noexcept {
        return (size_t)k * sz + (size_t)j * sy + (size_t)i;
    }
};

static void initializeLocalGrid(std::vector<Real>& grid, const LocalGrid& lg, const Decomp& dc,
                                const size_t nx, const size_t ny) {
    for (int k = 1; k <= lg.lz; ++k) {
        const size_t gz = (size_t)(dc.start[2] + k - 1);
        for (int j = 1; j <= lg.ly; ++j) {
            const size_t gy = (size_t)(dc.start[1] + j - 1);
            Real* row = grid.data() + lg.id(0, j, k);
            const size_t rowBase = idx3((size_t)dc.start[0], gy, gz, nx, ny);
            for (int i = 1; i <= lg.lx; ++i) {
                const size_t gidx = rowBase + (size_t)(i - 1);
                row[i] = (Real)(gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on a local sub-box [i0,i1] x [j0,j1] x [k0,k1] (inclusive)
static inline void stencilRegion(const Real* __restrict__ in, Real* __restrict__ out, const LocalGrid& lg,
                                 const int i0, const int i1, const int j0, const int j1,
                                 const int k0, const int k1) {
    const int sy = lg.sy;
    const int sz = lg.sz;
    for (int k = k0; k <= k1; ++k) {
        for (int j = j0; j <= j1; ++j) {
            const size_t base = lg.id(0, j, k);
            const Real* __restrict__ c = in + base;
            const Real* __restrict__ ym = c - sy;
            const Real* __restrict__ yp = c + sy;
            const Real* __restrict__ zm = c - sz;
            const Real* __restrict__ zp = c + sz;
            Real* __restrict__ o = out + base;
            #pragma GCC ivdep
            for (int i = i0; i <= i1; ++i) {
                o[i] = (c[i] + c[i - 1] + c[i + 1] + ym[i] + yp[i] + zm[i] + zp[i]) / 7.0;
            }
        }
    }
}

// Halo exchange machinery
struct HaloComm {
    MPI_Datatype faceType[3];  // datatype of one face slab perpendicular to dim d
    MPI_Comm comm;
};

static void buildHaloTypes(HaloComm& hc, const LocalGrid& lg) {
    const int sizes[3] = {lg.lz + 2, lg.ly + 2, lg.lx + 2};  // C order: z, y, x
    for (int d = 0; d < 3; ++d) {
        int subsizes[3] = {lg.lz, lg.ly, lg.lx};
        // Face perpendicular to dim d has thickness 1 in that dim.
        // d=0 -> x (C index 2), d=1 -> y (C index 1), d=2 -> z (C index 0)
        subsizes[2 - d] = 1;
        for (int q = 0; q < 3; ++q) subsizes[q] = std::max(subsizes[q], 1);
        const int starts[3] = {0, 0, 0};
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &hc.faceType[d]);
        MPI_Type_commit(&hc.faceType[d]);
    }
}

// Post all halo sends/receives for the given buffer. Returns the number of requests.
static int startHaloExchange(Real* buf, const LocalGrid& lg, const Decomp& dc, const HaloComm& hc,
                             MPI_Request reqs[12]) {
    int n = 0;
    const int ext[3] = {lg.lx, lg.ly, lg.lz};
    for (int d = 0; d < 3; ++d) {
        if (ext[d] == 0) continue;
        // Local offsets of first interior face and last interior face, and halo faces.
        int lo[3] = {1, 1, 1};
        int hi[3] = {1, 1, 1};
        int loHalo[3] = {1, 1, 1};
        int hiHalo[3] = {1, 1, 1};
        hi[d] = ext[d];
        loHalo[d] = 0;
        hiHalo[d] = ext[d] + 1;
        // Receive from lower neighbor into lower halo; send our lowest face to it.
        MPI_Irecv(buf + lg.id(loHalo[0], loHalo[1], loHalo[2]), 1, hc.faceType[d], dc.nbr[d][0], d * 2 + 1, hc.comm, &reqs[n++]);
        MPI_Irecv(buf + lg.id(hiHalo[0], hiHalo[1], hiHalo[2]), 1, hc.faceType[d], dc.nbr[d][1], d * 2 + 0, hc.comm, &reqs[n++]);
        MPI_Isend(buf + lg.id(lo[0], lo[1], lo[2]), 1, hc.faceType[d], dc.nbr[d][0], d * 2 + 0, hc.comm, &reqs[n++]);
        MPI_Isend(buf + lg.id(hi[0], hi[1], hi[2]), 1, hc.faceType[d], dc.nbr[d][1], d * 2 + 1, hc.comm, &reqs[n++]);
    }
    return n;
}

// One stencil iteration: exchange halos of `input`, compute global-interior
// points into `output`. Global boundary cells are never written (both
// buffers hold the initial values there), matching the original copy semantics.
static void stencilIteration(std::vector<Real>& input, std::vector<Real>& output, const LocalGrid& lg,
                             const Decomp& dc, const HaloComm& hc,
                             const size_t nx, const size_t ny, const size_t nz) {
    // Compute range of local indices that are global-interior points.
    const size_t N[3] = {nx, ny, nz};
    int lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
        const long s = dc.start[d];
        const long len = (d == 0 ? lg.lx : (d == 1 ? lg.ly : lg.lz));
        lo[d] = (int)std::max(1L, 2L - s);
        hi[d] = (int)std::min(len, (long)N[d] - 1L - s);
    }
    if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) {
        // Nothing to compute locally, but still participate in the exchange.
        MPI_Request reqs[12];
        const int n = startHaloExchange(input.data(), lg, dc, hc, reqs);
        MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
        return;
    }

    MPI_Request reqs[12];
    const int n = startHaloExchange(input.data(), lg, dc, hc, reqs);

    const Real* in = input.data();
    Real* out = output.data();

    // Core region: cells whose neighbors are all local interior cells.
    const int ci0 = std::max(lo[0], 2), ci1 = std::min(hi[0], lg.lx - 1);
    const int cj0 = std::max(lo[1], 2), cj1 = std::min(hi[1], lg.ly - 1);
    const int ck0 = std::max(lo[2], 2), ck1 = std::min(hi[2], lg.lz - 1);
    const bool hasCore = (ci0 <= ci1) && (cj0 <= cj1) && (ck0 <= ck1);
    if (hasCore) {
        stencilRegion(in, out, lg, ci0, ci1, cj0, cj1, ck0, ck1);
    }

    MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);

    if (!hasCore) {
        stencilRegion(in, out, lg, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        return;
    }
    // Boundary shells of the local block (depend on halo data).
    // z slabs
    if (lo[2] < ck0) stencilRegion(in, out, lg, lo[0], hi[0], lo[1], hi[1], lo[2], ck0 - 1);
    if (hi[2] > ck1) stencilRegion(in, out, lg, lo[0], hi[0], lo[1], hi[1], ck1 + 1, hi[2]);
    // y slabs (within core z range)
    if (lo[1] < cj0) stencilRegion(in, out, lg, lo[0], hi[0], lo[1], cj0 - 1, ck0, ck1);
    if (hi[1] > cj1) stencilRegion(in, out, lg, lo[0], hi[0], cj1 + 1, hi[1], ck0, ck1);
    // x slabs (within core y,z range)
    if (lo[0] < ci0) stencilRegion(in, out, lg, lo[0], ci0 - 1, cj0, cj1, ck0, ck1);
    if (hi[0] > ci1) stencilRegion(in, out, lg, ci1 + 1, hi[0], cj0, cj1, ck0, ck1);
}

// Gather the distributed grid into a full global grid on rank 0.
static void gatherGrid(const std::vector<Real>& local, const LocalGrid& lg, const Decomp& dc,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int rank, const int size, MPI_Comm comm, std::vector<Real>& global) {
    const int localSizes[3] = {lg.lz + 2, lg.ly + 2, lg.lx + 2};
    const int localSub[3] = {std::max(lg.lz, 1), std::max(lg.ly, 1), std::max(lg.lx, 1)};
    const int localStarts[3] = {1, 1, 1};
    const bool nonEmpty = lg.lx > 0 && lg.ly > 0 && lg.lz > 0;

    if (rank != 0) {
        if (nonEmpty) {
            MPI_Datatype t;
            MPI_Type_create_subarray(3, localSizes, localSub, localStarts, MPI_ORDER_C, MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
            MPI_Send(local.data(), 1, t, 0, 100, comm);
            MPI_Type_free(&t);
        }
        return;
    }

    global.assign(nx * ny * nz, 0.0);
    const int gsizes[3] = {(int)nz, (int)ny, (int)nx};
    for (int r = 0; r < size; ++r) {
        int coords[3];
        MPI_Cart_coords(comm, r, 3, coords);
        int st[3], ln[3];
        blockRange(nx, dc.dims[0], coords[0], st[0], ln[0]);
        blockRange(ny, dc.dims[1], coords[1], st[1], ln[1]);
        blockRange(nz, dc.dims[2], coords[2], st[2], ln[2]);
        if (ln[0] == 0 || ln[1] == 0 || ln[2] == 0) continue;
        if (r == 0) {
            for (int k = 1; k <= lg.lz; ++k)
                for (int j = 1; j <= lg.ly; ++j)
                    std::memcpy(global.data() + idx3((size_t)st[0], (size_t)(st[1] + j - 1), (size_t)(st[2] + k - 1), nx, ny),
                                local.data() + lg.id(1, j, k), sizeof(Real) * (size_t)lg.lx);
        } else {
            const int gsub[3] = {ln[2], ln[1], ln[0]};
            const int gst[3] = {st[2], st[1], st[0]};
            MPI_Datatype t;
            MPI_Type_create_subarray(3, gsizes, gsub, gst, MPI_ORDER_C, MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
            MPI_Recv(global.data(), 1, t, r, 100, comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&t);
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // ---- Domain decomposition ----
    Decomp dc;
    chooseDims(size, nx, ny, nz, dc.dims);
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dc.dims, periods, 1, &cart);
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, dc.coords);
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &dc.nbr[d][0], &dc.nbr[d][1]);
    }
    blockRange(nx, dc.dims[0], dc.coords[0], dc.start[0], dc.len[0]);
    blockRange(ny, dc.dims[1], dc.coords[1], dc.start[1], dc.len[1]);
    blockRange(nz, dc.dims[2], dc.coords[2], dc.start[2], dc.len[2]);
    // Ranks owning an empty block (only possible when ranks outnumber cells in
    // some dimension) take no part in halo exchange: they and their neighbors
    // treat the link as MPI_PROC_NULL.
    {
        const size_t N[3] = {nx, ny, nz};
        auto blockEmpty = [&](const int c[3]) {
            for (int d = 0; d < 3; ++d) {
                int s, l;
                blockRange(N[d], dc.dims[d], c[d], s, l);
                if (l == 0) return true;
            }
            return false;
        };
        const bool meEmpty = blockEmpty(dc.coords);
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                if (dc.nbr[d][side] == MPI_PROC_NULL) continue;
                int c[3] = {dc.coords[0], dc.coords[1], dc.coords[2]};
                c[d] += (side == 0 ? -1 : 1);
                if (meEmpty || blockEmpty(c)) dc.nbr[d][side] = MPI_PROC_NULL;
            }
        }
    }
    if (rank == 0) {
        printf("Process grid: %d x %d x %d\n", dc.dims[0], dc.dims[1], dc.dims[2]);
    }

    LocalGrid lg;
    lg.lx = dc.len[0]; lg.ly = dc.len[1]; lg.lz = dc.len[2];
    lg.sx = 1;
    lg.sy = lg.lx + 2;
    lg.sz = (lg.lx + 2) * (lg.ly + 2);
    lg.total = (size_t)lg.sz * (size_t)(lg.lz + 2);

    HaloComm hc;
    hc.comm = cart;
    buildHaloTypes(hc, lg);
    
    // Allocate grids (double buffering), both initialized so that global
    // boundary cells (never updated) hold the initial values in either buffer.
    std::vector<Real> grid1(lg.total, 0.0);
    std::vector<Real> grid2(lg.total, 0.0);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeLocalGrid(grid1, lg, dc, nx, ny);
    grid2 = grid1;
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double tStart = MPI_Wtime();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, lg, dc, hc, nx, ny, nz);
        } else {
            stencilIteration(grid2, grid1, lg, dc, hc, nx, ny, nz);
        }
    }
    
    double tLocal = MPI_Wtime() - tStart;
    double tMax = tLocal;
    MPI_Reduce(&tLocal, &tMax, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    const long durationMs = (long)(tMax * 1000.0);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    int ret = 0;
    // Print results for external validation
    std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        gatherGrid(finalLocal, lg, dc, nx, ny, nz, rank, size, cart, finalGrid);
        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }
            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(finalGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, cart);
    }

    for (int d = 0; d < 3; ++d) MPI_Type_free(&hc.faceType[d]);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return ret;
}
