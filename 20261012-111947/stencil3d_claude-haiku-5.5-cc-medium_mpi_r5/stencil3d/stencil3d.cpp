#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
static_assert(std::is_same_v<Real, double>, "halo exchange uses MPI_DOUBLE");

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Axes are ordered z, y, x (slowest to fastest in memory), matching idx3.
// Face f = 2 * axis + side, where side 0 is the low and side 1 the high end of the axis.
constexpr int kFaces = 6;

// Owned cells of one rank plus a one-cell halo on every side.
struct Block {
    size_t start[3];  // global index of the first owned cell, per axis
    size_t len[3];    // number of owned cells, per axis
    size_t sx;        // padded stride along y
    size_t sxy;       // padded stride along z
    size_t total;     // padded element count

    // Padded index of an owned cell given owned-local coordinates
    size_t at(size_t z, size_t y, size_t x) const { return (z + 1) * sxy + (y + 1) * sx + (x + 1); }
};

// Start and length of block c when n cells are split over p blocks
inline void blockExtent(const size_t n, const int p, const int c, size_t& start, size_t& len) {
    start = n * c / p;
    len = n * (c + 1) / p - start;
}

// Chooses a 3D process grid (z, y, x) with at most one block per cell along each axis.
// Picks the factorization with the least total internal face area, i.e. the least halo traffic.
bool chooseProcessGrid(const int nprocs, const size_t n[3], int p[3]) {
    bool found = false;
    double best = 0.0;
    for (int pz = 1; pz <= nprocs; ++pz) {
        if (nprocs % pz) continue;
        const int rest = nprocs / pz;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py) continue;
            const int px = rest / py;
            if (static_cast<size_t>(pz) > n[0] || static_cast<size_t>(py) > n[1] || static_cast<size_t>(px) > n[2]) continue;
            const double cost = (pz - 1.0) * n[1] * n[2] + (py - 1.0) * n[0] * n[2] + (px - 1.0) * n[0] * n[1];
            if (!found || cost < best) {
                found = true;
                best = cost;
                p[0] = pz;
                p[1] = py;
                p[2] = px;
            }
        }
    }
    return found;
}

void initializeBlock(std::vector<Real>& buf, const Block& b, const size_t nx, const size_t ny) {
    for (size_t z = 0; z < b.len[0]; ++z) {
        for (size_t y = 0; y < b.len[1]; ++y) {
            for (size_t x = 0; x < b.len[2]; ++x) {
                const size_t idx = idx3(b.start[2] + x, b.start[1] + y, b.start[0] + z, nx, ny);
                buf[b.at(z, y, x)] = (idx % 19) * 1.0;
            }
        }
    }
}

// Half-open range of owned-local indices along one axis
struct Range {
    size_t lo;
    size_t hi;
    bool empty() const { return lo >= hi; }
};

// Owned-local cells along one axis that the stencil updates. Global boundary cells are fixed.
Range updateRange(const size_t start, const size_t len, const size_t n) {
    const size_t lo = start >= 1 ? 0 : std::min<size_t>(len, 1 - start);
    const size_t hi = start + 1 < n ? std::min<size_t>(len, n - 1 - start) : 0;
    return {lo, hi};
}

struct Box {
    Range r[3];
    bool empty() const { return r[0].empty() || r[1].empty() || r[2].empty(); }
};

// Applies the 7-point averaging stencil to every cell of a box. Expression order matches the serial code.
void sweep(const Real* __restrict in, Real* __restrict out, const Box& b, const Block& blk) {
    if (b.empty()) return;
    const size_t count = b.r[2].hi - b.r[2].lo;
    for (size_t z = b.r[0].lo; z < b.r[0].hi; ++z) {
        for (size_t y = b.r[1].lo; y < b.r[1].hi; ++y) {
            const size_t base = blk.at(z, y, b.r[2].lo);
            for (size_t i = 0; i < count; ++i) {
                const size_t p = base + i;
                out[p] = (in[p] + in[p - 1] + in[p + 1] + in[p - blk.sx] + in[p + blk.sx] + in[p - blk.sxy] + in[p + blk.sxy]) / 7.0;
            }
        }
    }
}

// Subarray datatype selecting one face: the outermost owned layer (send) or the halo layer just outside (receive)
MPI_Datatype makeFaceType(const Block& b, const int face, const bool send) {
    const int axis = face / 2;
    const bool high = face % 2 == 1;
    int sizes[3] = {static_cast<int>(b.len[0] + 2), static_cast<int>(b.len[1] + 2), static_cast<int>(b.len[2] + 2)};
    int subsizes[3] = {static_cast<int>(b.len[0]), static_cast<int>(b.len[1]), static_cast<int>(b.len[2])};
    int starts[3] = {1, 1, 1};
    subsizes[axis] = 1;
    if (send) {
        starts[axis] = high ? static_cast<int>(b.len[axis]) : 1;
    } else {
        starts[axis] = high ? static_cast<int>(b.len[axis]) + 1 : 0;
    }
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Persistent halo exchange for both grid buffers. Requests [0, kFaces) receive, [kFaces, 2*kFaces) send.
struct Halo {
    MPI_Datatype sendType[kFaces];
    MPI_Datatype recvType[kFaces];
    MPI_Request req[2][2 * kFaces];
};

void setupHalo(Halo& halo, const Block& b, MPI_Comm cart, const int nbr[kFaces], Real* const bufs[2]) {
    for (int f = 0; f < kFaces; ++f) {
        halo.sendType[f] = makeFaceType(b, f, true);
        halo.recvType[f] = makeFaceType(b, f, false);
    }
    for (int k = 0; k < 2; ++k) {
        for (int f = 0; f < kFaces; ++f) {
            // A face-f message is tagged f by the sender; the neighbour on side f sends toward us on its face f ^ 1
            MPI_Recv_init(bufs[k], 1, halo.recvType[f], nbr[f], f ^ 1, cart, &halo.req[k][f]);
            MPI_Send_init(bufs[k], 1, halo.sendType[f], nbr[f], f, cart, &halo.req[k][kFaces + f]);
        }
    }
}

void freeHalo(Halo& halo) {
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 2 * kFaces; ++i) {
            MPI_Request_free(&halo.req[k][i]);
        }
    }
    for (int f = 0; f < kFaces; ++f) {
        MPI_Type_free(&halo.sendType[f]);
        MPI_Type_free(&halo.recvType[f]);
    }
}

// Copies the owned cells of a block, in z, y, x order, into a dense array
void packBlock(const Block& b, const Real* buf, std::vector<Real>& packed) {
    size_t p = 0;
    for (size_t z = 0; z < b.len[0]; ++z) {
        for (size_t y = 0; y < b.len[1]; ++y) {
            for (size_t x = 0; x < b.len[2]; ++x) {
                packed[p++] = buf[b.at(z, y, x)];
            }
        }
    }
}

// Writes a dense block into its place in the global grid (global layout is idx3 order)
void placeBlock(std::vector<Real>& global, const size_t n[3], const size_t start[3], const size_t len[3], const Real* packed) {
    size_t p = 0;
    for (size_t z = 0; z < len[0]; ++z) {
        for (size_t y = 0; y < len[1]; ++y) {
            for (size_t x = 0; x < len[2]; ++x) {
                global[idx3(start[2] + x, start[1] + y, start[0] + z, n[2], n[1])] = packed[p++];
            }
        }
    }
}

// Gathers the full grid on rank 0 (used only for printing results)
std::vector<Real> gatherGrid(MPI_Comm cart, const int nprocs, const int procs[3], const size_t n[3], const Block& b, const Real* buf) {
    const size_t count = b.len[0] * b.len[1] * b.len[2];
    std::vector<Real> packed(count);
    packBlock(b, buf, packed);

    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    if (rank != 0) {
        MPI_Send(packed.data(), static_cast<int>(count), MPI_DOUBLE, 0, 0, cart);
        return {};
    }

    std::vector<Real> global(n[0] * n[1] * n[2]);
    placeBlock(global, n, b.start, b.len, packed.data());
    for (int r = 1; r < nprocs; ++r) {
        int coord[3];
        MPI_Cart_coords(cart, r, 3, coord);
        size_t start[3];
        size_t len[3];
        for (int a = 0; a < 3; ++a) {
            blockExtent(n[a], procs[a], coord[a], start[a], len[a]);
        }
        const size_t rcount = len[0] * len[1] * len[2];
        packed.resize(rcount);
        MPI_Recv(packed.data(), static_cast<int>(rcount), MPI_DOUBLE, r, 0, cart, MPI_STATUS_IGNORE);
        placeBlock(global, n, start, len, packed.data());
    }
    return global;
}

// Same checks as the serial validateResult(), evaluated collectively. Reports on the root rank only.
bool validateDistributed(MPI_Comm cart, const bool root, const Block& b, const Real* buf) {
    long long nonFinite = 0;
    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();
    for (size_t z = 0; z < b.len[0]; ++z) {
        for (size_t y = 0; y < b.len[1]; ++y) {
            for (size_t x = 0; x < b.len[2]; ++x) {
                const Real val = buf[b.at(z, y, x)];
                if (!std::isfinite(val)) {
                    ++nonFinite;
                } else {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
            }
        }
    }

    // 1. No NaN or Inf values
    long long totalNonFinite = 0;
    MPI_Allreduce(&nonFinite, &totalNonFinite, 1, MPI_LONG_LONG, MPI_SUM, cart);
    if (totalNonFinite > 0) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, cart);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, cart);

    if (root) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
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

int runBenchmark(int argc, char** argv) {
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const bool worldRoot = worldRank == 0;

    int nxArg = 128;
    int nyArg = 0;  // Will be set to nx if not specified
    int nzArg = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nxArg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            nyArg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nzArg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRoot) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (nyArg == 0) nyArg = nxArg;
    if (nzArg == 0) nzArg = nxArg;

    if (nxArg < 1 || nyArg < 1 || nzArg < 1 || iterations < 0) {
        if (worldRoot) {
            printf("Grid dimensions must be positive and the iteration count non-negative\n");
            printUsage(argv[0]);
        }
        return 1;
    }

    const size_t nx = nxArg;
    const size_t ny = nyArg;
    const size_t nz = nzArg;
    const size_t n[3] = {nz, ny, nx};

    // Use the largest rank count that can split the grid; any remaining ranks sit out
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int nprocs = worldSize;
    int procs[3] = {1, 1, 1};
    while (!chooseProcessGrid(nprocs, n, procs)) --nprocs;

    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < nprocs ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (active == MPI_COMM_NULL) return 0;

    // Reordering may renumber ranks, so the Cartesian communicator is used for everything from here on
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(active, 3, procs, periods, 1, &cart);
    MPI_Comm_free(&active);
    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    const bool root = rank == 0;

    int coord[3];
    MPI_Cart_coords(cart, rank, 3, coord);

    Block blk{};
    for (int a = 0; a < 3; ++a) {
        blockExtent(n[a], procs[a], coord[a], blk.start[a], blk.len[a]);
    }
    blk.sx = blk.len[2] + 2;
    blk.sxy = blk.sx * (blk.len[1] + 2);
    blk.total = blk.sxy * (blk.len[0] + 2);

    int nbr[kFaces];
    for (int a = 0; a < 3; ++a) {
        MPI_Cart_shift(cart, a, 1, &nbr[2 * a], &nbr[2 * a + 1]);
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Both buffers start identical, so their global boundary cells match from the first iteration on
    std::vector<Real> grid1(blk.total, 0.0);
    std::vector<Real> grid2(blk.total, 0.0);
    initializeBlock(grid1, blk, nx, ny);
    grid2 = grid1;
    Real* const grids[2] = {grid1.data(), grid2.data()};

    Halo halo;
    setupHalo(halo, blk, cart, nbr, grids);

    // Cells updated by the stencil, split into an inner part that needs no halo data and
    // the outer shell that does. The inner part is computed while the halo messages are in flight.
    Box upd;
    for (int a = 0; a < 3; ++a) {
        upd.r[a] = updateRange(blk.start[a], blk.len[a], n[a]);
    }
    Box inner;
    for (int a = 0; a < 3; ++a) {
        inner.r[a] = {std::max<size_t>(upd.r[a].lo, 1), std::min(upd.r[a].hi, blk.len[a] - 1)};
    }
    std::vector<Box> outer;
    if (inner.empty()) {
        outer.push_back(upd);
    } else {
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                Box b = upd;
                for (int k = 0; k < a; ++k) b.r[k] = inner.r[k];
                b.r[a] = side == 0 ? Range{upd.r[a].lo, inner.r[a].lo} : Range{inner.r[a].hi, upd.r[a].hi};
                outer.push_back(b);
            }
        }
    }

    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        const int cur = iter % 2;
        const Real* in = grids[cur];
        Real* out = grids[1 - cur];

        MPI_Startall(2 * kFaces, halo.req[cur]);
        sweep(in, out, inner, blk);
        MPI_Waitall(2 * kFaces, halo.req[cur], MPI_STATUSES_IGNORE);
        for (const Box& b : outer) {
            sweep(in, out, b, blk);
        }
    }

    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    const long ms = static_cast<long>(maxElapsed * 1000.0);
    if (root) printf("Computation time: %ld ms\n", ms);

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
    if (root) printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalBuf = grids[iterations % 2];

    // Print results for external validation
    if (printResults) {
        std::vector<Real> global = gatherGrid(cart, nprocs, procs, n, blk, finalBuf);
        if (root) print_results(global, "Grid");
    }

    int exitCode = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateDistributed(cart, root, blk, finalBuf);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        exitCode = valid ? 0 : 1;
    }

    freeHalo(halo);
    MPI_Comm_free(&cart);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
