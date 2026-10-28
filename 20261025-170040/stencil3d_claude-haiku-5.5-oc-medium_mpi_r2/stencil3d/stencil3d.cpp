#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Index = std::ptrdiff_t;

static_assert(std::is_same_v<Real, double>, "MPI_DOUBLE is used for Real");

// 3D index calculation (global grid, x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

namespace {

constexpr int kTagGather = 1000;

// Global problem description (identical on every rank)
struct Global {
    size_t N[3];  // global extents nx, ny, nz
    int P[3];     // process grid extents
};

// Subdomain owned by one rank. Local storage is padded by one ghost layer
// on every side; owned cells use local indices 1..L[d].
struct Block {
    Index L[3];   // owned extents
    Index g0[3];  // global 0-based index of owned local cell 1
    Index N[3];   // global extents
    Index S[3];   // strides into the padded local array (S[0] == 1)
    Index total;  // padded local array size
};

// Half-open-free inclusive box in local coordinates
struct Box {
    Index lo[3];
    Index hi[3];
};

Box mkBox(Index xl, Index xh, Index yl, Index yh, Index zl, Index zh) {
    Box b;
    b.lo[0] = xl; b.hi[0] = xh;
    b.lo[1] = yl; b.hi[1] = yh;
    b.lo[2] = zl; b.hi[2] = zh;
    return b;
}

// Split n cells into p contiguous blocks; block c gets [start, start+len)
void blockRange(size_t n, int p, int c, size_t& start, size_t& len) {
    const size_t P = static_cast<size_t>(p);
    const size_t C = static_cast<size_t>(c);
    const size_t base = n / P;
    const size_t rem = n % P;
    start = C * base + std::min(C, rem);
    len = base + (C < rem ? 1 : 0);
}

Block makeBlock(const Global& g, const int coords[3]) {
    Block b;
    for (int d = 0; d < 3; ++d) {
        size_t start = 0, len = 0;
        blockRange(g.N[d], g.P[d], coords[d], start, len);
        b.L[d] = static_cast<Index>(len);
        b.g0[d] = static_cast<Index>(start);
        b.N[d] = static_cast<Index>(g.N[d]);
    }
    const Index e0 = b.L[0] + 2, e1 = b.L[1] + 2, e2 = b.L[2] + 2;
    b.S[0] = 1;
    b.S[1] = e0;
    b.S[2] = e0 * e1;
    b.total = e0 * e1 * e2;
    return b;
}

// Visit owned cells in (z, y, x) order: f(ordinal, localIndex, globalIndex)
template <class F>
void forEachOwned(const Block& b, F&& f) {
    Index n = 0;
    for (Index k = 1; k <= b.L[2]; ++k)
        for (Index j = 1; j <= b.L[1]; ++j)
            for (Index i = 1; i <= b.L[0]; ++i, ++n) {
                const Index p = i * b.S[0] + j * b.S[1] + k * b.S[2];
                const Index gi = (b.g0[0] + i - 1) +
                                 b.N[0] * ((b.g0[1] + j - 1) + b.N[1] * (b.g0[2] + k - 1));
                f(n, p, gi);
            }
}

// Visit the owned face cells lying in plane `plane` of dimension d, in an
// order that depends only on d (so sender and receiver agree on packing).
template <class F>
void forEachFaceCell(const Block& b, int d, Index plane, F&& f) {
    const int o1 = (d + 1) % 3;
    const int o2 = (d + 2) % 3;
    for (Index q = 1; q <= b.L[o2]; ++q)
        for (Index r = 1; r <= b.L[o1]; ++r) {
            Index c[3];
            c[d] = plane;
            c[o1] = r;
            c[o2] = q;
            f(c[0] * b.S[0] + c[1] * b.S[1] + c[2] * b.S[2]);
        }
}

// Choose a 3D process grid minimizing total halo surface area.
bool chooseDims(int P, const size_t N[3], int dims[3]) {
    double best = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0) continue;
        if (static_cast<size_t>(px) > N[0]) break;
        const int rest = P / px;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py != 0) continue;
            if (static_cast<size_t>(py) > N[1]) break;
            const int pz = rest / py;
            if (static_cast<size_t>(pz) > N[2]) continue;
            const double cost = (px - 1) * static_cast<double>(N[1]) * static_cast<double>(N[2]) +
                                (py - 1) * static_cast<double>(N[0]) * static_cast<double>(N[2]) +
                                (pz - 1) * static_cast<double>(N[0]) * static_cast<double>(N[1]);
            if (cost < best) {
                best = cost;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
                found = true;
            }
        }
    }
    return found;
}

// Face halo exchange with the six Cartesian neighbours.
// Face f = 2*d + s: s = 0 is the low side, s = 1 the high side of dimension d.
struct Halo {
    MPI_Comm comm = MPI_COMM_NULL;
    Block b{};
    int nb[3][2] = {{MPI_PROC_NULL, MPI_PROC_NULL}, {MPI_PROC_NULL, MPI_PROC_NULL}, {MPI_PROC_NULL, MPI_PROC_NULL}};
    std::vector<Real> sendBuf[6];
    std::vector<Real> recvBuf[6];
    MPI_Request req[12];

    void init(MPI_Comm cart, const Block& blk) {
        comm = cart;
        b = blk;
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(cart, d, 1, &nb[d][0], &nb[d][1]);  // source = low, dest = high
        }
        for (int d = 0; d < 3; ++d)
            for (int s = 0; s < 2; ++s) {
                const int o1 = (d + 1) % 3, o2 = (d + 2) % 3;
                const size_t cnt = static_cast<size_t>(b.L[o1] * b.L[o2]);
                sendBuf[2 * d + s].assign(cnt, 0.0);
                recvBuf[2 * d + s].assign(cnt, 0.0);
            }
    }

    // Pack owned faces and post all receives/sends. Ghost cells of `g` are
    // written only in finish(), so the owned data may be read concurrently.
    void post(const Real* g) {
        int k = 0;
        for (int d = 0; d < 3; ++d)
            for (int s = 0; s < 2; ++s) {
                const int f = 2 * d + s;
                const Index plane = (s == 0) ? 1 : b.L[d];
                Real* sb = sendBuf[f].data();
                Index n = 0;
                forEachFaceCell(b, d, plane, [&](Index p) { sb[n++] = g[p]; });
                const int cnt = static_cast<int>(sendBuf[f].size());
                MPI_Irecv(recvBuf[f].data(), cnt, MPI_DOUBLE, nb[d][s], 2 * d + (1 - s), comm, &req[k++]);
                MPI_Isend(sb, cnt, MPI_DOUBLE, nb[d][s], 2 * d + s, comm, &req[k++]);
            }
    }

    // Wait for all messages and unpack received faces into ghost layers.
    void finish(Real* g) {
        MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 3; ++d)
            for (int s = 0; s < 2; ++s) {
                if (nb[d][s] == MPI_PROC_NULL) continue;
                const int f = 2 * d + s;
                const Index plane = (s == 0) ? 0 : b.L[d] + 1;
                const Real* rb = recvBuf[f].data();
                Index n = 0;
                forEachFaceCell(b, d, plane, [&](Index p) { g[p] = rb[n++]; });
            }
    }
};

// 7-point averaging stencil over an inclusive box of owned, globally
// interior cells. Arithmetic order matches the serial reference exactly.
void sweep(const Real* __restrict in, Real* __restrict out, const Block& b, const Box& bx) {
    const Index S1 = b.S[1];
    const Index S2 = b.S[2];
    for (Index k = bx.lo[2]; k <= bx.hi[2]; ++k)
        for (Index j = bx.lo[1]; j <= bx.hi[1]; ++j) {
            const Index base = j * S1 + k * S2;
            for (Index i = bx.lo[0]; i <= bx.hi[0]; ++i) {
                const Index p = base + i;
                const Real center = in[p];
                const Real left = in[p - 1];
                const Real right = in[p + 1];
                const Real front = in[p - S1];
                const Real back = in[p + S1];
                const Real bottom = in[p - S2];
                const Real top = in[p + S2];
                out[p] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
}

// Global boundary cells are never updated (their value is constant, as in
// the serial code, which copies them through unchanged).
struct Regions {
    Box update;  // owned cells that are globally interior
    Box core;    // subset of update whose stencil reads only owned cells
    bool hasCore;
};

Regions computeRegions(const Block& b) {
    Index ulo[3], uhi[3], clo[3], chi[3];
    bool hasCore = true;
    for (int d = 0; d < 3; ++d) {
        ulo[d] = std::max<Index>(1, 2 - b.g0[d]);
        uhi[d] = std::min<Index>(b.L[d], b.N[d] - 1 - b.g0[d]);
        clo[d] = std::max<Index>(ulo[d], 2);
        chi[d] = std::min<Index>(uhi[d], b.L[d] - 1);
        if (clo[d] > chi[d]) hasCore = false;
    }
    Regions r;
    r.update = mkBox(ulo[0], uhi[0], ulo[1], uhi[1], ulo[2], uhi[2]);
    r.core = mkBox(clo[0], chi[0], clo[1], chi[1], clo[2], chi[2]);
    r.hasCore = hasCore;
    return r;
}

// Disjoint boxes covering update \ core (only valid when core is non-empty).
void shellBoxes(const Regions& r, Box out[6]) {
    const Box& U = r.update;
    const Box& C = r.core;
    out[0] = mkBox(U.lo[0], U.hi[0], U.lo[1], U.hi[1], U.lo[2], C.lo[2] - 1);
    out[1] = mkBox(U.lo[0], U.hi[0], U.lo[1], U.hi[1], C.hi[2] + 1, U.hi[2]);
    out[2] = mkBox(U.lo[0], U.hi[0], U.lo[1], C.lo[1] - 1, C.lo[2], C.hi[2]);
    out[3] = mkBox(U.lo[0], U.hi[0], C.hi[1] + 1, U.hi[1], C.lo[2], C.hi[2]);
    out[4] = mkBox(U.lo[0], C.lo[0] - 1, C.lo[1], C.hi[1], C.lo[2], C.hi[2]);
    out[5] = mkBox(C.hi[0] + 1, U.hi[0], C.lo[1], C.hi[1], C.lo[2], C.hi[2]);
}

// Distributed version of validateResult(): same checks and messages.
bool validateDistributed(const Real* g, const Block& b, MPI_Comm comm, bool root) {
    int bad = 0;
    Real mn = std::numeric_limits<Real>::infinity();
    Real mx = -std::numeric_limits<Real>::infinity();
    forEachOwned(b, [&](Index, Index p, Index) {
        const Real v = g[p];
        if (std::isnan(v) || std::isinf(v)) bad = 1;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    });

    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    MPI_Allreduce(MPI_IN_PLACE, &mn, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (root) printf("Value range: [%.6f, %.6f]\n", mn, mx);

    if (mx > 1e6 || mn < -1e6) {
        if (root) printf("Validation failed: values out of expected range\n");
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int parseRc = -1;  // -1: continue, otherwise exit code
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
            if (root) printUsage(argv[0]);
            parseRc = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseRc = 1;
            break;
        }
    }
    if (parseRc != -1) {
        MPI_Finalize();
        return parseRc;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx < 1 || ny < 1 || nz < 1) {
        if (root) printf("Error: grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    Global g{};
    g.N[0] = nx;
    g.N[1] = ny;
    g.N[2] = nz;
    if (!chooseDims(worldSize, g.N, g.P)) {
        if (root) printf("Error: cannot decompose %zu x %zu x %zu grid over %d ranks\n", nx, ny, nz, worldSize);
        MPI_Finalize();
        return 1;
    }

    // Cartesian process topology; reorder=0 keeps cart rank == world rank
    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, g.P, periods, 0, &cart);
    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart, rank, 3, coords);

    const Block blk = makeBlock(g, coords);
    Halo halo;
    halo.init(cart, blk);
    const Regions regions = computeRegions(blk);

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;

    // Allocate local double-buffered blocks (with ghost layers)
    std::vector<Real> grid1(static_cast<size_t>(blk.total), 0.0);
    std::vector<Real> grid2(static_cast<size_t>(blk.total), 0.0);

    // Initialize
    if (root) printf("Initializing grid...\n");
    forEachOwned(blk, [&](Index, Index p, Index gi) {
        grid1[static_cast<size_t>(p)] = (static_cast<size_t>(gi) % 19) * 1.0;
    });
    // Boundary cells are never rewritten, so both buffers must hold them.
    grid2 = grid1;

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* out = (iter % 2 == 0) ? grid2.data() : grid1.data();

        halo.post(in);
        if (regions.hasCore) sweep(in, out, blk, regions.core);
        halo.finish(in);

        if (regions.hasCore) {
            Box shell[6];
            shellBoxes(regions, shell);
            for (const Box& bx : shell) sweep(in, out, blk, bx);
        } else {
            sweep(in, out, blk, regions.update);
        }
    }

    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(maxElapsed));

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics (global cell count)
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Final grid (identical buffer selection to the serial code)
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* fg = finalGrid.data();

    // Print results for external validation: gather the full grid on rank 0
    if (printResults) {
        std::vector<Real> full;
        if (rank == 0) {
            full.assign(gridSize, 0.0);
            forEachOwned(blk, [&](Index, Index p, Index gi) {
                full[static_cast<size_t>(gi)] = fg[p];
            });
            std::vector<Real> buf;
            for (int r = 1; r < worldSize; ++r) {
                int rc[3] = {0, 0, 0};
                MPI_Cart_coords(cart, r, 3, rc);
                const Block br = makeBlock(g, rc);
                const int cnt = static_cast<int>(br.L[0] * br.L[1] * br.L[2]);
                buf.resize(static_cast<size_t>(cnt));
                MPI_Recv(buf.data(), cnt, MPI_DOUBLE, r, kTagGather, cart, MPI_STATUS_IGNORE);
                forEachOwned(br, [&](Index n, Index, Index gi) {
                    full[static_cast<size_t>(gi)] = buf[static_cast<size_t>(n)];
                });
            }
            print_results(full, "Grid");
        } else {
            const int cnt = static_cast<int>(blk.L[0] * blk.L[1] * blk.L[2]);
            std::vector<Real> buf(static_cast<size_t>(cnt));
            forEachOwned(blk, [&](Index n, Index p, Index) {
                buf[static_cast<size_t>(n)] = fg[p];
            });
            MPI_Send(buf.data(), cnt, MPI_DOUBLE, 0, kTagGather, cart);
        }
    }

    int rc = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateDistributed(fg, blk, cart, root);
        if (valid) {
            if (root) printf("Validation: PASSED\n");
            rc = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return rc;
}
