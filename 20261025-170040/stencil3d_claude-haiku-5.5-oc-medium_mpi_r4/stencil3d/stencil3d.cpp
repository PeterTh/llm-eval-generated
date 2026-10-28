#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (global, x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Array-order convention used for all index math: axis 0 = z (slowest),
// axis 1 = y, axis 2 = x (contiguous). Matches the layout produced by idx3().
namespace {

// Half-open range [lo, hi) of local indices
struct Range {
    size_t lo;
    size_t hi;
};

inline bool contains(const Range& r, size_t i) { return i >= r.lo && i < r.hi; }

// Geometry of one rank's block. Local storage is (alloc[0] x alloc[1] x alloc[2])
// with one ghost layer on every side; owned cells occupy local indices [1, len].
struct Decomp {
    size_t n[3];        // global extents
    size_t start[3];    // global index of first owned cell
    size_t len[3];      // owned cell count per axis
    size_t alloc[3];    // len + 2 ghost layers
    Range interior[3];  // owned local indices that are not global boundary cells
};

// Balanced block partition of n cells over p parts: part c starts at n*c/p
inline size_t blockStart(size_t n, size_t p, size_t c) { return n * c / p; }

Decomp makeDecomp(const size_t n[3], const int p[3], const int coord[3]) {
    Decomp D{};
    for (int k = 0; k < 3; ++k) {
        D.n[k] = n[k];
        const size_t pk = static_cast<size_t>(p[k]);
        const size_t ck = static_cast<size_t>(coord[k]);
        D.start[k] = blockStart(n[k], pk, ck);
        D.len[k] = blockStart(n[k], pk, ck + 1) - D.start[k];
        D.alloc[k] = D.len[k] + 2;
        // Global index g = start + i - 1 is interior iff 1 <= g <= n - 2
        const size_t lo = (D.start[k] == 0) ? 2 : 1;
        const size_t hiIncl = std::min(D.len[k], n[k] - 1 - D.start[k]);
        D.interior[k] = {lo, hiIncl + 1};
    }
    return D;
}

// Choose a process grid (array order) using as many ranks as possible, never
// splitting an axis finer than its extent, and among those minimise the total
// halo surface area.
void chooseProcessGrid(const size_t n[3], int nprocs, int p[3]) {
    size_t bestProd = 0;
    double bestCost = 0.0;
    p[0] = p[1] = p[2] = 1;
    const size_t np = static_cast<size_t>(nprocs);
    for (size_t a = 1; a <= std::min(np, n[0]); ++a) {
        for (size_t b = 1; b <= std::min(np / a, n[1]); ++b) {
            for (size_t c = 1; c <= std::min(np / (a * b), n[2]); ++c) {
                const size_t prod = a * b * c;
                const double L0 = static_cast<double>(n[0]) / a;
                const double L1 = static_cast<double>(n[1]) / b;
                const double L2 = static_cast<double>(n[2]) / c;
                double cost = 0.0;
                if (a > 1) cost += L1 * L2;
                if (b > 1) cost += L0 * L2;
                if (c > 1) cost += L0 * L1;
                if (prod > bestProd || (prod == bestProd && cost < bestCost)) {
                    bestProd = prod;
                    bestCost = cost;
                    p[0] = static_cast<int>(a);
                    p[1] = static_cast<int>(b);
                    p[2] = static_cast<int>(c);
                }
            }
        }
    }
}

// Stencil update of the box `r` (local indices). Cells on the global domain
// boundary are copied; all other cells get the 7-point average. Every cell
// read lies within the owned region or the ghost layers.
void sweep(const Real* __restrict in, Real* __restrict out, const Decomp& D, const Range r[3]) {
    const size_t sx = D.alloc[2];
    const size_t sxy = D.alloc[2] * D.alloc[1];
    for (size_t k0 = r[0].lo; k0 < r[0].hi; ++k0) {
        const bool inZ = contains(D.interior[0], k0);
        for (size_t k1 = r[1].lo; k1 < r[1].hi; ++k1) {
            const size_t base = k0 * sxy + k1 * sx;
            const bool inRow = inZ && contains(D.interior[1], k1);
            const size_t xs = inRow ? std::max(r[2].lo, D.interior[2].lo) : r[2].hi;
            const size_t xe = inRow ? std::min(r[2].hi, D.interior[2].hi) : r[2].hi;
            if (xs >= xe) {
                for (size_t x = r[2].lo; x < r[2].hi; ++x) out[base + x] = in[base + x];
                continue;
            }
            for (size_t x = r[2].lo; x < xs; ++x) out[base + x] = in[base + x];
            for (size_t x = xs; x < xe; ++x) {
                const size_t c = base + x;
                // Same summation order as the serial reference
                out[c] = (in[c] + in[c - 1] + in[c + 1] + in[c - sx] + in[c + sx] +
                          in[c - sxy] + in[c + sxy]) / 7.0;
            }
            for (size_t x = xe; x < r[2].hi; ++x) out[base + x] = in[base + x];
        }
    }
}

// Update the whole owned block: interior core first (no ghost reads), then the
// shell of owned cells that touch ghost layers (run after halos have arrived).
void computeCore(const Real* in, Real* out, const Decomp& D) {
    const Range core[3] = {{2, D.len[0]}, {2, D.len[1]}, {2, D.len[2]}};
    sweep(in, out, D, core);
}

void computeShell(const Real* in, Real* out, const Decomp& D) {
    // Partition of the owned-minus-core shell: the axis d is the first one on a
    // face; earlier axes are restricted to their core range, later ones are full.
    for (size_t d = 0; d < 3; ++d) {
        for (size_t side = 0; side < 2; ++side) {
            if (side == 1 && D.len[d] == 1) continue;  // same face as side 0
            const size_t face = (side == 0) ? 1 : D.len[d];
            Range r[3];
            for (size_t e = 0; e < 3; ++e) {
                if (e < d) {
                    r[e] = {2, D.len[e]};
                } else if (e == d) {
                    r[e] = {face, face + 1};
                } else {
                    r[e] = {1, D.len[e] + 1};
                }
            }
            sweep(in, out, D, r);
        }
    }
}

void initializeGrid(Real* local, const Decomp& D, const size_t nx, const size_t ny) {
    const size_t sx = D.alloc[2];
    const size_t sxy = D.alloc[2] * D.alloc[1];
    for (size_t k0 = 1; k0 <= D.len[0]; ++k0) {
        const size_t gz = D.start[0] + k0 - 1;
        for (size_t k1 = 1; k1 <= D.len[1]; ++k1) {
            const size_t gy = D.start[1] + k1 - 1;
            for (size_t k2 = 1; k2 <= D.len[2]; ++k2) {
                const size_t gx = D.start[2] + k2 - 1;
                const size_t gidx = idx3(gx, gy, gz, nx, ny);
                local[k0 * sxy + k1 * sx + k2] = (gidx % 19) * 1.0;
            }
        }
    }
}

// Persistent halo exchange with the six face neighbours (the 7-point stencil
// needs no edge or corner ghosts). One set of requests per buffer parity.
struct Halo {
    static constexpr int kReqPerBuffer = 12;  // 3 axes * 2 sides * (send + recv)
    int nbr[3][2];                  // [axis][0] = minus neighbour, [axis][1] = plus
    MPI_Datatype faceSend[3][2];    // owned boundary layer sent to nbr[axis][side]
    MPI_Datatype faceRecv[3][2];    // ghost layer filled from nbr[axis][side]
    MPI_Request req[2][kReqPerBuffer];
};

MPI_Datatype makeSubarray(const int sizes[3], const int sub[3], const int starts[3]) {
    MPI_Datatype t;
    MPI_Type_create_subarray(3, sizes, sub, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}

void setupHalo(Halo& H, MPI_Comm cart, const Decomp& D, Real* const bufs[2]) {
    for (int k = 0; k < 3; ++k) {
        MPI_Cart_shift(cart, k, 1, &H.nbr[k][0], &H.nbr[k][1]);
    }

    int sizes[3], sub[3], starts[3];
    for (int k = 0; k < 3; ++k) {
        for (int e = 0; e < 3; ++e) {
            sizes[e] = static_cast<int>(D.alloc[e]);
        }
        for (int s = 0; s < 2; ++s) {
            // Owned face on side s (sent) and ghost layer on side s (received)
            for (int e = 0; e < 3; ++e) {
                sub[e] = static_cast<int>(D.len[e]);
                starts[e] = 1;
            }
            sub[k] = 1;
            starts[k] = (s == 0) ? 1 : static_cast<int>(D.len[k]);
            H.faceSend[k][s] = makeSubarray(sizes, sub, starts);

            starts[k] = (s == 0) ? 0 : static_cast<int>(D.len[k]) + 1;
            H.faceRecv[k][s] = makeSubarray(sizes, sub, starts);
        }
    }

    for (int b = 0; b < 2; ++b) {
        int r = 0;
        for (int k = 0; k < 3; ++k) {
            for (int s = 0; s < 2; ++s) {
                // Tag 2k+s marks data travelling toward the side-s neighbour
                MPI_Recv_init(bufs[b], 1, H.faceRecv[k][s], H.nbr[k][s], 2 * k + (1 - s), cart,
                              &H.req[b][r++]);
                MPI_Send_init(bufs[b], 1, H.faceSend[k][s], H.nbr[k][s], 2 * k + s, cart,
                              &H.req[b][r++]);
            }
        }
    }
}

void freeHalo(Halo& H) {
    for (int b = 0; b < 2; ++b) {
        for (int i = 0; i < Halo::kReqPerBuffer; ++i) {
            MPI_Request_free(&H.req[b][i]);
        }
    }
    for (int k = 0; k < 3; ++k) {
        for (int s = 0; s < 2; ++s) {
            MPI_Type_free(&H.faceSend[k][s]);
            MPI_Type_free(&H.faceRecv[k][s]);
        }
    }
}

Decomp decompOf(MPI_Comm cart, const size_t n[3], const int p[3], int rank) {
    int coord[3];
    MPI_Cart_coords(cart, rank, 3, coord);
    return makeDecomp(n, p, coord);
}

// Assemble the full grid in global order on rank 0 (used only for -r output)
std::vector<Real> gatherGlobal(MPI_Comm cart, const Decomp& D, const size_t n[3], const int p[3],
                               int nActive, const Real* local) {
    int rank;
    MPI_Comm_rank(cart, &rank);
    const int tag = 7;
    std::vector<Real> out;

    int sizes[3], sub[3], starts[3];
    for (int e = 0; e < 3; ++e) {
        sizes[e] = static_cast<int>(D.alloc[e]);
        sub[e] = static_cast<int>(D.len[e]);
        starts[e] = 1;
    }
    MPI_Datatype ownedT = makeSubarray(sizes, sub, starts);

    if (rank == 0) {
        out.assign(n[0] * n[1] * n[2], 0.0);
        int gsizes[3] = {static_cast<int>(n[0]), static_cast<int>(n[1]), static_cast<int>(n[2])};
        for (int r = 0; r < nActive; ++r) {
            const Decomp Dr = decompOf(cart, n, p, r);
            if (r == 0) {
                for (size_t i0 = 0; i0 < D.len[0]; ++i0) {
                    for (size_t i1 = 0; i1 < D.len[1]; ++i1) {
                        const Real* src = local + ((i0 + 1) * D.alloc[1] + (i1 + 1)) * D.alloc[2] + 1;
                        Real* dst = out.data() + ((D.start[0] + i0) * n[1] + (D.start[1] + i1)) * n[2] + D.start[2];
                        std::copy_n(src, D.len[2], dst);
                    }
                }
            } else {
                int rsub[3], rst[3];
                for (int e = 0; e < 3; ++e) {
                    rsub[e] = static_cast<int>(Dr.len[e]);
                    rst[e] = static_cast<int>(Dr.start[e]);
                }
                MPI_Datatype gT = makeSubarray(gsizes, rsub, rst);
                MPI_Recv(out.data(), 1, gT, r, tag, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&gT);
            }
        }
    } else {
        MPI_Send(local, 1, ownedT, 0, tag, cart);
    }
    MPI_Type_free(&ownedT);
    return out;
}

// Distributed version of validateResult(): same checks and messages, but the
// reductions are done across ranks instead of gathering the whole grid.
bool validateDistributed(MPI_Comm cart, const Decomp& D, const Real* local, bool isRoot) {
    Real lmin = std::numeric_limits<Real>::infinity();
    Real lmax = -std::numeric_limits<Real>::infinity();
    int lbad = 0;
    const size_t sx = D.alloc[2];
    const size_t sxy = D.alloc[2] * D.alloc[1];
    for (size_t k0 = 1; k0 <= D.len[0]; ++k0) {
        for (size_t k1 = 1; k1 <= D.len[1]; ++k1) {
            for (size_t k2 = 1; k2 <= D.len[2]; ++k2) {
                const Real v = local[k0 * sxy + k1 * sx + k2];
                if (!std::isfinite(v)) {
                    ++lbad;
                } else {
                    lmin = std::min(lmin, v);
                    lmax = std::max(lmax, v);
                }
            }
        }
    }

    int gbad = 0;
    Real gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&lbad, &gbad, 1, MPI_INT, MPI_SUM, cart);
    MPI_Allreduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, cart);
    MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, cart);

    if (gbad > 0) {
        if (isRoot) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (isRoot) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 1e6 || gmin < -1e6) {
        if (isRoot) printf("Validation failed: values out of expected range\n");
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

// Runs the benchmark on the ranks of `cart` (the active ranks). Returns the exit code.
int runBenchmark(MPI_Comm cart, const size_t n[3], const int p[3], int nActive, int iterations,
                 bool validate, bool printResults, bool isRoot) {
    int rank;
    MPI_Comm_rank(cart, &rank);
    const Decomp D = decompOf(cart, n, p, rank);

    const size_t localSize = D.alloc[0] * D.alloc[1] * D.alloc[2];
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize, 0.0);
    Real* bufs[2] = {grid1.data(), grid2.data()};

    if (isRoot) printf("Initializing grid...\n");
    initializeGrid(grid1.data(), D, n[2], n[1]);

    Halo H;
    setupHalo(H, cart, D, bufs);

    if (isRoot) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    // `cur` holds the current input; after the loop it holds the final grid
    int cur = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = bufs[cur];
        Real* out = bufs[1 - cur];
        MPI_Startall(Halo::kReqPerBuffer, H.req[cur]);
        computeCore(in, out, D);
        MPI_Waitall(Halo::kReqPerBuffer, H.req[cur], MPI_STATUSES_IGNORE);
        computeShell(in, out, D);
        cur = 1 - cur;
    }

    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (isRoot) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate performance metrics (same formula as the serial version)
        double cellUpdates = (double)((n[2] - 2) * (n[1] - 2) * (n[0] - 2)) * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const Real* finalLocal = bufs[cur];
    if (printResults) {
        std::vector<Real> global = gatherGlobal(cart, D, n, p, nActive, finalLocal);
        if (isRoot) print_results(global, "Grid");
    }

    int exitCode = 0;
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        const bool valid = validateDistributed(cart, D, finalLocal, isRoot);
        if (isRoot) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        if (!valid) exitCode = 1;
    }

    freeHalo(H);
    return exitCode;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool isRoot = (worldRank == 0);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    int exitCode = 0;
    bool stop = false;

    // Parse command line arguments
    for (int i = 1; i < argc && !stop; ++i) {
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
            stop = true;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            stop = true;
        }
    }

    if (!stop) {
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
        const size_t maxDim = static_cast<size_t>(INT_MAX);
        if (nx == 0 || ny == 0 || nz == 0 || nx > maxDim || ny > maxDim || nz > maxDim) {
            if (isRoot) printf("Error: grid dimensions must be between 1 and %d\n", INT_MAX);
            exitCode = 1;
            stop = true;
        }
    }

    if (stop) {
        MPI_Finalize();
        return exitCode;
    }

    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n[3] = {nz, ny, nx};  // array order (z, y, x)
    int p[3];
    chooseProcessGrid(n, worldSize, p);
    const int nActive = p[0] * p[1] * p[2];

    // Ranks beyond nActive get MPI_COMM_NULL and sit out the computation
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, p, periods, 0, &cart);
    if (cart != MPI_COMM_NULL) {
        exitCode = runBenchmark(cart, n, p, nActive, iterations, validate, printResults, isRoot);
        MPI_Comm_free(&cart);
    }

    // Every rank (including idle ones) returns the same status
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
