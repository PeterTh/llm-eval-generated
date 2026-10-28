#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Only the root rank produces console output.
static bool g_isRoot = false;

__attribute__((format(printf, 1, 2)))
static void rootPrintf(const char* fmt, ...) {
    if (!g_isRoot) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

// 3D index calculation (global grid, x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Dimension order used throughout: 0 = x, 1 = y, 2 = z.
//
// Each rank owns a contiguous block of the global grid, stored in a local array
// surrounded by one ghost layer. Owned cells have local coordinates 1..len[d];
// ghost cells have local coordinates 0 and len[d]+1.
struct Subdomain {
    size_t gn[3];     // global grid extents
    size_t start[3];  // global index of the first owned cell
    size_t len[3];    // number of owned cells
    size_t sy, sz;    // strides of the ghosted local array (x stride is 1)
    size_t total;     // size of the ghosted local array

    size_t at(size_t i, size_t j, size_t k) const noexcept { return i + j * sy + k * sz; }
};

static size_t blockStart(size_t c, size_t n, size_t p) noexcept { return c * n / p; }

static Subdomain makeSubdomain(const size_t gn[3], const int dims[3], const int coords[3]) {
    Subdomain s{};
    for (int d = 0; d < 3; ++d) {
        s.gn[d] = gn[d];
        const size_t p = static_cast<size_t>(dims[d]);
        const size_t c = static_cast<size_t>(coords[d]);
        s.start[d] = blockStart(c, gn[d], p);
        s.len[d] = blockStart(c + 1, gn[d], p) - s.start[d];
    }
    s.sy = s.len[0] + 2;
    s.sz = s.sy * (s.len[1] + 2);
    s.total = s.sz * (s.len[2] + 2);
    return s;
}

// Chooses a 3D process grid minimizing the total halo volume (ghost cells
// exchanged per iteration). Each dimension must be no larger than the grid.
static bool chooseDims(int P, const size_t gn[3], int dims[3]) {
    bool found = false;
    double best = 0.0;
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0) continue;
        const int rest = P / px;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py != 0) continue;
            const int pz = rest / py;
            const int d[3] = {px, py, pz};
            bool ok = true;
            for (int k = 0; k < 3; ++k) ok = ok && static_cast<size_t>(d[k]) <= gn[k];
            if (!ok) continue;
            const double cost = double(px - 1) * double(gn[1]) * double(gn[2]) +
                                double(py - 1) * double(gn[0]) * double(gn[2]) +
                                double(pz - 1) * double(gn[0]) * double(gn[1]);
            if (!found || cost < best) {
                found = true;
                best = cost;
                std::copy(d, d + 3, dims);
            }
        }
    }
    return found;
}

template <typename F>
static void forEachOwned(const Subdomain& s, F&& f) {
    for (size_t k = 1; k <= s.len[2]; ++k)
        for (size_t j = 1; j <= s.len[1]; ++j)
            for (size_t i = 1; i <= s.len[0]; ++i) f(i, j, k);
}

// Visits every owned cell on the face orthogonal to dimension d at local layer `layer`,
// in a fixed order shared by sender and receiver.
template <typename F>
static void forEachFaceCell(const Subdomain& s, int d, size_t layer, F&& f) {
    const int a = (d + 1) % 3;
    const int b = (d + 2) % 3;
    size_t c[3];
    c[d] = layer;
    for (size_t cb = 1; cb <= s.len[b]; ++cb) {
        c[b] = cb;
        for (size_t ca = 1; ca <= s.len[a]; ++ca) {
            c[a] = ca;
            f(s.at(c[0], c[1], c[2]));
        }
    }
}

// Halo exchange of the six faces of the local block. Non-blocking, with
// pack/unpack into contiguous buffers.
class HaloExchanger {
  public:
    HaloExchanger(const Subdomain& sd, const int nbr[3][2], MPI_Comm comm) : sd_(sd), comm_(comm) {
        for (int d = 0; d < 3; ++d)
            for (int s = 0; s < 2; ++s) nbr_[d][s] = nbr[d][s];
        for (int d = 0; d < 3; ++d) {
            const int a = (d + 1) % 3;
            const int b = (d + 2) % 3;
            const size_t n = sd.len[a] * sd.len[b];
            for (int s = 0; s < 2; ++s) {
                send_[2 * d + s].resize(n);
                recv_[2 * d + s].resize(n);
            }
        }
    }

    // Packs and posts all halo transfers for grid `g`.
    void begin(const Real* g) {
        int n = 0;
        for (int d = 0; d < 3; ++d) {
            for (int s = 0; s < 2; ++s) {
                if (nbr_[d][s] == MPI_PROC_NULL) continue;
                // Data from neighbour on side s was sent through its face (1 - s).
                const int f = 2 * d + s;
                MPI_Irecv(recv_[f].data(), count(d), MPI_DOUBLE, nbr_[d][s], 2 * d + (1 - s), comm_, &req_[n++]);
            }
        }
        for (int d = 0; d < 3; ++d) {
            for (int s = 0; s < 2; ++s) {
                if (nbr_[d][s] == MPI_PROC_NULL) continue;
                const int f = 2 * d + s;
                Real* buf = send_[f].data();
                size_t m = 0;
                forEachFaceCell(sd_, d, s == 0 ? 1 : sd_.len[d], [&](size_t idx) { buf[m++] = g[idx]; });
                MPI_Isend(buf, count(d), MPI_DOUBLE, nbr_[d][s], 2 * d + s, comm_, &req_[n++]);
            }
        }
        nreq_ = n;
    }

    // Completes all transfers and unpacks received data into the ghost layers of `g`.
    void end(Real* g) {
        MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 3; ++d) {
            for (int s = 0; s < 2; ++s) {
                if (nbr_[d][s] == MPI_PROC_NULL) continue;
                const Real* buf = recv_[2 * d + s].data();
                size_t m = 0;
                forEachFaceCell(sd_, d, s == 0 ? 0 : sd_.len[d] + 1, [&](size_t idx) { g[idx] = buf[m++]; });
            }
        }
    }

  private:
    int count(int d) const { return static_cast<int>(sd_.len[(d + 1) % 3] * sd_.len[(d + 2) % 3]); }

    const Subdomain& sd_;
    MPI_Comm comm_;
    int nbr_[3][2];
    std::vector<Real> send_[6], recv_[6];
    MPI_Request req_[12];
    int nreq_ = 0;
};

// Inclusive box in local (owned) coordinates.
struct Box {
    long long lo[3];
    long long hi[3];
};

// Range of owned cells that are interior in the global grid (not fixed boundary).
static Box interiorRange(const Subdomain& sd) {
    Box r{};
    for (int d = 0; d < 3; ++d) {
        const long long start = static_cast<long long>(sd.start[d]);
        const long long gn = static_cast<long long>(sd.gn[d]);
        const long long len = static_cast<long long>(sd.len[d]);
        r.lo[d] = std::max<long long>(1, 2 - start);
        r.hi[d] = std::min<long long>(len, gn - 1 - start);
    }
    return r;
}

// Splits the owned block into an inner box (reads owned cells only) and a shell of
// disjoint boxes that read ghost cells. Empty boxes are allowed; they are skipped at run time.
static void buildBoxes(const Subdomain& sd, Box& inner, std::vector<Box>& shell) {
    const long long Lx = static_cast<long long>(sd.len[0]);
    const long long Ly = static_cast<long long>(sd.len[1]);
    const long long Lz = static_cast<long long>(sd.len[2]);
    auto add = [&](long long x0, long long x1, long long y0, long long y1, long long z0, long long z1) {
        shell.push_back(Box{{x0, y0, z0}, {x1, y1, z1}});
    };
    add(1, Lx, 1, Ly, 1, 1);                                      // z = 1 face
    add(1, Lx, 1, Ly, std::max(Lz, 2LL), Lz);                     // z = Lz face
    add(1, Lx, 1, 1, 2, Lz - 1);                                  // y = 1 face
    add(1, Lx, std::max(Ly, 2LL), Ly, 2, Lz - 1);                 // y = Ly face
    add(1, 1, 2, Ly - 1, 2, Lz - 1);                              // x = 1 face
    add(std::max(Lx, 2LL), Lx, 2, Ly - 1, 2, Lz - 1);             // x = Lx face
    inner = Box{{2, 2, 2}, {Lx - 1, Ly - 1, Lz - 1}};
}

void initializeGrid(std::vector<Real>& grid, const Subdomain& sd, const size_t nx, const size_t ny) {
    forEachOwned(sd, [&](size_t i, size_t j, size_t k) {
        const size_t gx = sd.start[0] + i - 1;
        const size_t gy = sd.start[1] + j - 1;
        const size_t gz = sd.start[2] + k - 1;
        const size_t idx = idx3(gx, gy, gz, nx, ny);
        grid[sd.at(i, j, k)] = (idx % 19) * 1.0;
    });
}

// 7-point stencil over the clipped box. Arithmetic order matches the serial code.
static void stencilBox(const Subdomain& sd, const Box& interior, const Box& box, const Real* __restrict in, Real* __restrict out) {
    size_t lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
        const long long l = std::max(box.lo[d], interior.lo[d]);
        const long long h = std::min(box.hi[d], interior.hi[d]);
        if (l > h) return;
        lo[d] = static_cast<size_t>(l);
        hi[d] = static_cast<size_t>(h);
    }
    const size_t sy = sd.sy;
    const size_t sz = sd.sz;
    for (size_t k = lo[2]; k <= hi[2]; ++k) {
        for (size_t j = lo[1]; j <= hi[1]; ++j) {
            const size_t base = sd.at(0, j, k);
            for (size_t i = lo[0]; i <= hi[0]; ++i) {
                const size_t idx = base + i;
                const Real center = in[idx];
                const Real left = in[idx - 1];
                const Real right = in[idx + 1];
                const Real front = in[idx - sy];
                const Real back = in[idx + sy];
                const Real bottom = in[idx - sz];
                const Real top = in[idx + sz];
                out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const Subdomain& sd, const Real* grid, MPI_Comm comm) {
    // Reductions over all ranks; every rank takes the same branch.
    int localBad = 0;
    Real localMin = grid[sd.at(1, 1, 1)];
    Real localMax = localMin;
    forEachOwned(sd, [&](size_t i, size_t j, size_t k) {
        const Real v = grid[sd.at(i, j, k)];
        if (std::isnan(v) || std::isinf(v)) localBad = 1;
        localMin = std::min(localMin, v);
        localMax = std::max(localMax, v);
    });
    int globalBad = 0;
    Real minVal = 0.0, maxVal = 0.0;
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    // 1. No NaN or Inf values
    if (globalBad) {
        rootPrintf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    rootPrintf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        rootPrintf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    rootPrintf("Usage: %s [options]\n", progName);
    rootPrintf("Options:\n");
    rootPrintf("  -x <num>     Grid size in X dimension (default: 128)\n");
    rootPrintf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    rootPrintf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    rootPrintf("  -i <num>     Number of iterations (default: 10)\n");
    rootPrintf("  -v           Enable validation\n");
    rootPrintf("  -r           Print results for external validation\n");
    rootPrintf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    g_isRoot = (worldRank == 0);

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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    rootPrintf("3D Stencil Benchmark\n");
    rootPrintf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    rootPrintf("Iterations: %d\n", iterations);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gn[3] = {nx, ny, nz};
    const size_t gridSize = nx * ny * nz;

    // Use the largest number of ranks (up to all of them) that can decompose the grid.
    // Any surplus ranks (only possible for grids smaller than the rank count) are idle.
    int dims[3] = {1, 1, 1};
    int procs = worldSize;
    while (procs >= 1 && !chooseDims(procs, gn, dims)) --procs;
    if (procs < 1) {
        rootPrintf("Error: cannot decompose %zu x %zu x %zu grid\n", nx, ny, nz);
        MPI_Finalize();
        return 1;
    }
    if (procs < worldSize) {
        rootPrintf("Note: grid too small for %d ranks; using %d, the rest are idle\n", worldSize, procs);
    }

    // Cartesian communicator (non-periodic); rank order is preserved.
    // Ranks beyond the process grid receive MPI_COMM_NULL and exit.
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    if (cart == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart, rank, 3, coords);
    int nbr[3][2];  // [d][0] = minus neighbour, [d][1] = plus neighbour
    for (int d = 0; d < 3; ++d) MPI_Cart_shift(cart, d, 1, &nbr[d][0], &nbr[d][1]);

    rootPrintf("MPI ranks: %d (process grid %d x %d x %d)\n", procs, dims[0], dims[1], dims[2]);

    const Subdomain sd = makeSubdomain(gn, dims, coords);
    HaloExchanger halo(sd, nbr, cart);

    Box inner{};
    std::vector<Box> shell;
    buildBoxes(sd, inner, shell);
    const Box interior = interiorRange(sd);

    // Allocate grids (double buffering). Both buffers get the same initial values,
    // so fixed global-boundary cells are correct in either buffer, as in the serial code.
    std::vector<Real> grid1(sd.total, 0.0);
    std::vector<Real> grid2(sd.total, 0.0);

    rootPrintf("Initializing grid...\n");
    initializeGrid(grid1, sd, nx, ny);
    grid2 = grid1;

    rootPrintf("Running stencil computation...\n");
    Real* cur = grid1.data();
    Real* nxt = grid2.data();

    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        // Overlap: inner cells need no halo data, so compute them while messages are in flight.
        halo.begin(cur);
        stencilBox(sd, interior, inner, cur, nxt);
        halo.end(cur);
        for (const Box& b : shell) stencilBox(sd, interior, b, cur, nxt);
        std::swap(cur, nxt);
    }
    const double localElapsed = MPI_Wtime() - t0;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    rootPrintf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / elapsed / 1e6;  // Million cell updates per second
    rootPrintf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Gather the final grid on the root (only needed for result output) in global order.
    const bool needGlobal = printResults;
    std::vector<Real> finalGrid;
    if (needGlobal) {
        std::vector<Real> tmp;
        if (rank == 0) {
            finalGrid.assign(gridSize, 0.0);
            auto place = [&](const Subdomain& s, const Real* src, size_t& m) {
                forEachOwned(s, [&](size_t i, size_t j, size_t k) {
                    finalGrid[idx3(s.start[0] + i - 1, s.start[1] + j - 1, s.start[2] + k - 1, nx, ny)] = src[m++];
                });
            };
            // Own block
            tmp.resize(sd.len[0] * sd.len[1] * sd.len[2]);
            size_t m = 0;
            forEachOwned(sd, [&](size_t i, size_t j, size_t k) { tmp[m++] = cur[sd.at(i, j, k)]; });
            m = 0;
            place(sd, tmp.data(), m);
            // Other blocks
            for (int r = 1; r < procs; ++r) {
                int rc[3];
                MPI_Cart_coords(cart, r, 3, rc);
                const Subdomain sr = makeSubdomain(gn, dims, rc);
                const size_t n = sr.len[0] * sr.len[1] * sr.len[2];
                tmp.resize(n);
                MPI_Recv(tmp.data(), static_cast<int>(n), MPI_DOUBLE, r, 100, cart, MPI_STATUS_IGNORE);
                m = 0;
                place(sr, tmp.data(), m);
            }
        } else {
            const size_t n = sd.len[0] * sd.len[1] * sd.len[2];
            tmp.resize(n);
            size_t m = 0;
            forEachOwned(sd, [&](size_t i, size_t j, size_t k) { tmp[m++] = cur[sd.at(i, j, k)]; });
            MPI_Send(tmp.data(), static_cast<int>(n), MPI_DOUBLE, 0, 100, cart);
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        rootPrintf("Validating result...\n");
        const bool valid = validateResult(sd, cur, cart);
        if (valid) {
            rootPrintf("Validation: PASSED\n");
        } else {
            rootPrintf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return exitCode;
}
