#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Index = std::ptrdiff_t;

// 3D index calculation (global, x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Global extent [0, n) is split over p blocks; block c owns [offset, offset + length).
inline Index blockOffset(const Index n, const Index p, const Index c) noexcept {
    return c * (n / p) + std::min(c, n % p);
}

inline Index blockLength(const Index n, const Index p, const Index c) noexcept {
    return n / p + (c < n % p ? 1 : 0);
}

// Choose a 3D process grid (px, py, pz) with px*py*pz == size, no dimension larger than
// the matching grid extent, minimizing the per-rank halo surface. Returns {0,0,0} if none.
std::array<int, 3> chooseProcessGrid(const int size, const std::array<Index, 3>& n) {
    std::array<int, 3> best{0, 0, 0};
    double bestCost = 0.0;
    for (int px = 1; px <= size; ++px) {
        if (size % px != 0) continue;
        const int rem = size / px;
        for (int py = 1; py <= rem; ++py) {
            if (rem % py != 0) continue;
            const int pz = rem / py;
            if (px > n[0] || py > n[1] || pz > n[2]) continue;
            const double bx = double(n[0]) / px;
            const double by = double(n[1]) / py;
            const double bz = double(n[2]) / pz;
            const double cost = bx * by + by * bz + bx * bz;
            if (best[0] == 0 || cost < bestCost) {
                best = {px, py, pz};
                bestCost = cost;
            }
        }
    }
    return best;
}

// Decomposition of the global grid onto a 3D Cartesian communicator.
// Local storage is padded by one ghost layer on every side.
struct Domain {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    std::array<int, 3> procs{};   // process grid extents (x, y, z)
    std::array<int, 3> coords{};  // this rank's position in the process grid
    std::array<Index, 3> n{};     // global grid extents
    std::array<Index, 3> off{};   // global offset of owned block
    std::array<Index, 3> len{};   // owned block extents
    std::array<int, 6> nb{};      // neighbour ranks: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z
    Index sx = 0, sy = 0, sz = 0; // padded local extents
    Index sxy = 0;
};

// Index of local padded cell (x, y, z), with x,y,z in [0, len+2).
inline Index lidx(const Domain& d, const Index x, const Index y, const Index z) noexcept {
    return z * d.sxy + y * d.sx + x;
}

// Owned cells in padded local coordinates are [1, len] on every axis.
struct Range {
    std::array<Index, 3> lo{};
    std::array<Index, 3> hi{};
    bool empty() const noexcept {
        return lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2];
    }
};

// Face datatype for direction dir (0:-x 1:+x 2:-y 3:+y 4:-z 5:+z).
// ghost == false: owned boundary layer on that side (data to send).
// ghost == true:  ghost layer on that side (data to receive).
MPI_Datatype makeFaceType(const Domain& d, const int dir, const bool ghost) {
    const int axis = dir / 2;
    const int side = dir % 2;
    std::array<Index, 3> sizes = {d.sx, d.sy, d.sz};
    std::array<Index, 3> subs = {d.len[0], d.len[1], d.len[2]};
    std::array<Index, 3> starts = {1, 1, 1};
    subs[axis] = 1;
    if (!ghost) {
        starts[axis] = (side == 0) ? 1 : d.len[axis];
    } else {
        starts[axis] = (side == 0) ? 0 : d.len[axis] + 1;
    }
    // MPI expects C order (slowest-varying dimension first): z, y, x.
    int msizes[3], msubs[3], mstarts[3];
    for (int k = 0; k < 3; ++k) {
        msizes[k] = static_cast<int>(sizes[2 - k]);
        msubs[k] = static_cast<int>(subs[2 - k]);
        mstarts[k] = static_cast<int>(starts[2 - k]);
    }
    MPI_Datatype t;
    MPI_Type_create_subarray(3, msizes, msubs, mstarts, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}

// Interior range that needs ghost data: owned cells that are global interior points
// (global boundary cells are never updated by the stencil).
Range stencilRange(const Domain& d) {
    Range r;
    for (int a = 0; a < 3; ++a) {
        r.lo[a] = std::max<Index>(1, 2 - d.off[a]);
        r.hi[a] = std::min<Index>(d.len[a], d.n[a] - 1 - d.off[a]);
    }
    return r;
}

// Subset of the stencil range that only reads owned cells (no ghost data needed).
Range innerRange(const Domain& d, const Range& r) {
    Range in = r;
    for (int a = 0; a < 3; ++a) {
        in.lo[a] = std::max<Index>(r.lo[a], 2);
        in.hi[a] = std::min<Index>(r.hi[a], d.len[a] - 1);
    }
    return in;
}

// Disjoint decomposition of (stencil range minus inner range) into slabs.
std::vector<Range> shellRanges(const Domain& d, const Range& r, const Range& inner) {
    std::vector<Range> out;
    auto add = [&out](const Range& x) {
        if (!x.empty()) out.push_back(x);
    };

    // z slabs (full x, y extent)
    Index zLowHi = std::min<Index>(r.hi[2], 1);
    Range t = r;
    t.hi[2] = zLowHi;
    add(t);
    Index zHighLo = std::max<Index>({r.lo[2], d.len[2], zLowHi + 1});
    t = r;
    t.lo[2] = zHighLo;
    add(t);

    // Middle z: y slabs (full x extent)
    Range mz = r;
    mz.lo[2] = inner.lo[2];
    mz.hi[2] = inner.hi[2];
    Index yLowHi = std::min<Index>(r.hi[1], 1);
    t = mz;
    t.hi[1] = yLowHi;
    add(t);
    Index yHighLo = std::max<Index>({r.lo[1], d.len[1], yLowHi + 1});
    t = mz;
    t.lo[1] = yHighLo;
    t.hi[1] = r.hi[1];
    add(t);

    // Middle y: x slabs
    Range my = mz;
    my.lo[1] = inner.lo[1];
    my.hi[1] = inner.hi[1];
    Index xLowHi = std::min<Index>(r.hi[0], 1);
    t = my;
    t.hi[0] = xLowHi;
    add(t);
    Index xHighLo = std::max<Index>({r.lo[0], d.len[0], xLowHi + 1});
    t = my;
    t.lo[0] = xHighLo;
    t.hi[0] = r.hi[0];
    add(t);

    return out;
}

// 7-point averaging stencil over a local range. Summation order matches the serial code.
void stencilRangeKernel(const Real* __restrict in, Real* __restrict out, const Range& r,
                        const Index sx, const Index sxy) {
    for (Index z = r.lo[2]; z <= r.hi[2]; ++z) {
        for (Index y = r.lo[1]; y <= r.hi[1]; ++y) {
            const Index base = z * sxy + y * sx;
            for (Index x = r.lo[0]; x <= r.hi[0]; ++x) {
                const Index i = base + x;
                out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - sx] + in[i + sx] +
                          in[i - sxy] + in[i + sxy]) / 7.0;
            }
        }
    }
}

// Global boundary cells keep their input value.
void copyGlobalBoundary(const Domain& d, const Real* in, Real* out) {
    for (Index z = 1; z <= d.len[2]; ++z) {
        const Index gz = d.off[2] + z - 1;
        const bool zb = (gz == 0 || gz == d.n[2] - 1);
        for (Index y = 1; y <= d.len[1]; ++y) {
            const Index gy = d.off[1] + y - 1;
            const bool yb = zb || gy == 0 || gy == d.n[1] - 1;
            const Index base = lidx(d, 0, y, z);
            if (yb) {
                for (Index x = 1; x <= d.len[0]; ++x) out[base + x] = in[base + x];
            } else {
                if (d.off[0] == 0) out[base + 1] = in[base + 1];
                if (d.off[0] + d.len[0] - 1 == d.n[0] - 1) out[base + d.len[0]] = in[base + d.len[0]];
            }
        }
    }
}

void initializeGrid(const Domain& d, Real* grid) {
    for (Index z = 1; z <= d.len[2]; ++z) {
        for (Index y = 1; y <= d.len[1]; ++y) {
            for (Index x = 1; x <= d.len[0]; ++x) {
                const size_t gidx = idx3(static_cast<size_t>(d.off[0] + x - 1),
                                         static_cast<size_t>(d.off[1] + y - 1),
                                         static_cast<size_t>(d.off[2] + z - 1),
                                         static_cast<size_t>(d.n[0]), static_cast<size_t>(d.n[1]));
                grid[lidx(d, x, y, z)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// Collective: returns the same result on all ranks; rank 0 prints.
bool validateResult(const Domain& d, const Real* grid) {
    // 1. No NaN or Inf values
    long long bad = 0;
    Real minVal = grid[lidx(d, 1, 1, 1)];
    Real maxVal = minVal;
    for (Index z = 1; z <= d.len[2]; ++z)
        for (Index y = 1; y <= d.len[1]; ++y)
            for (Index x = 1; x <= d.len[0]; ++x) {
                const Real v = grid[lidx(d, x, y, z)];
                if (std::isnan(v) || std::isinf(v)) {
                    ++bad;
                } else {
                    minVal = std::min(minVal, v);
                    maxVal = std::max(maxVal, v);
                }
            }

    long long totalBad = 0;
    MPI_Allreduce(&bad, &totalBad, 1, MPI_LONG_LONG, MPI_SUM, d.cart);
    if (totalBad > 0) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real gMin = 0.0, gMax = 0.0;
    MPI_Allreduce(&minVal, &gMin, 1, MPI_DOUBLE, MPI_MIN, d.cart);
    MPI_Allreduce(&maxVal, &gMax, 1, MPI_DOUBLE, MPI_MAX, d.cart);

    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", gMin, gMax);

    if (gMax > 1e6 || gMin < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

// Collective: gathers the full grid (x fastest) onto rank 0.
void gatherGrid(const Domain& d, const Real* local, std::vector<Real>& global) {
    if (d.rank == 0) {
        global.assign(static_cast<size_t>(d.n[0] * d.n[1] * d.n[2]), 0.0);
    }

    // Owned block inside the padded local array
    int lsizes[3] = {int(d.sz), int(d.sy), int(d.sx)};
    int lsubs[3] = {int(d.len[2]), int(d.len[1]), int(d.len[0])};
    int lstarts[3] = {1, 1, 1};
    MPI_Datatype localType;
    MPI_Type_create_subarray(3, lsizes, lsubs, lstarts, MPI_ORDER_C, MPI_DOUBLE, &localType);
    MPI_Type_commit(&localType);

    if (d.rank == 0) {
        // Own block: direct copy
        for (Index z = 0; z < d.len[2]; ++z)
            for (Index y = 0; y < d.len[1]; ++y)
                for (Index x = 0; x < d.len[0]; ++x) {
                    const size_t g = idx3(static_cast<size_t>(d.off[0] + x), static_cast<size_t>(d.off[1] + y),
                                          static_cast<size_t>(d.off[2] + z), static_cast<size_t>(d.n[0]),
                                          static_cast<size_t>(d.n[1]));
                    global[g] = local[lidx(d, x + 1, y + 1, z + 1)];
                }
        for (int r = 1; r < d.size; ++r) {
            std::array<int, 3> rc{};
            MPI_Cart_coords(d.cart, r, 3, rc.data());
            std::array<Index, 3> roff{}, rlen{};
            for (int a = 0; a < 3; ++a) {
                roff[a] = blockOffset(d.n[a], d.procs[a], rc[a]);
                rlen[a] = blockLength(d.n[a], d.procs[a], rc[a]);
            }
            int msizes[3] = {int(d.n[2]), int(d.n[1]), int(d.n[0])};
            int msubs[3] = {int(rlen[2]), int(rlen[1]), int(rlen[0])};
            int mstarts[3] = {int(roff[2]), int(roff[1]), int(roff[0])};
            MPI_Datatype gtype;
            MPI_Type_create_subarray(3, msizes, msubs, mstarts, MPI_ORDER_C, MPI_DOUBLE, &gtype);
            MPI_Type_commit(&gtype);
            MPI_Recv(global.data(), 1, gtype, r, 0, d.cart, MPI_STATUS_IGNORE);
            MPI_Type_free(&gtype);
        }
    } else {
        MPI_Send(local, 1, localType, 0, 0, d.cart);
    }
    MPI_Type_free(&localType);
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

int run(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments (identical on every rank)
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
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
    }

    Domain d;
    d.size = size;
    d.n = {static_cast<Index>(nx), static_cast<Index>(ny), static_cast<Index>(nz)};
    d.procs = chooseProcessGrid(size, d.n);
    if (d.procs[0] == 0) {
        if (rank == 0) {
            printf("Cannot decompose %zu x %zu x %zu grid over %d MPI ranks\n", nx, ny, nz, size);
        }
        return 1;
    }

    // Cartesian topology (non-periodic); reorder allowed for locality on the cluster
    int periods[3] = {0, 0, 0};
    int dims[3] = {d.procs[0], d.procs[1], d.procs[2]};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    MPI_Cart_coords(d.cart, d.rank, 3, d.coords.data());
    // From here on, all root-only work uses the Cartesian communicator's rank 0
    rank = d.rank;

    for (int a = 0; a < 3; ++a) {
        int src = MPI_PROC_NULL, dst = MPI_PROC_NULL;
        MPI_Cart_shift(d.cart, a, 1, &src, &dst);
        d.nb[2 * a] = src;      // -a neighbour
        d.nb[2 * a + 1] = dst;  // +a neighbour
        d.off[a] = blockOffset(d.n[a], d.procs[a], d.coords[a]);
        d.len[a] = blockLength(d.n[a], d.procs[a], d.coords[a]);
    }
    d.sx = d.len[0] + 2;
    d.sy = d.len[1] + 2;
    d.sz = d.len[2] + 2;
    d.sxy = d.sx * d.sy;
    const size_t total = static_cast<size_t>(d.sxy * d.sz);

    std::array<MPI_Datatype, 6> sendT{}, recvT{};
    for (int dir = 0; dir < 6; ++dir) {
        sendT[dir] = makeFaceType(d, dir, false);
        recvT[dir] = makeFaceType(d, dir, true);
    }

    {
        std::vector<Real> bufA(total, 0.0);
        std::vector<Real> bufB(total, 0.0);

        if (rank == 0) printf("Initializing grid...\n");
        initializeGrid(d, bufA.data());

        const Range stencil = stencilRange(d);
        const Range inner = innerRange(d, stencil);
        const std::vector<Range> shell = shellRanges(d, stencil, inner);

        if (rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(d.cart);
        const double t0 = MPI_Wtime();

        for (int iter = 0; iter < iterations; ++iter) {
            Real* in = (iter % 2 == 0) ? bufA.data() : bufB.data();
            Real* out = (iter % 2 == 0) ? bufB.data() : bufA.data();

            // Post halo exchange: receive into ghost layers, send owned boundary layers
            std::array<MPI_Request, 12> reqs{};
            int nreq = 0;
            for (int dir = 0; dir < 6; ++dir) {
                MPI_Irecv(in, 1, recvT[dir], d.nb[dir], dir ^ 1, d.cart, &reqs[nreq++]);
            }
            for (int dir = 0; dir < 6; ++dir) {
                MPI_Isend(in, 1, sendT[dir], d.nb[dir], dir, d.cart, &reqs[nreq++]);
            }

            // Overlap: work that needs only owned data, plus global boundary copy
            stencilRangeKernel(in, out, inner, d.sx, d.sxy);
            copyGlobalBoundary(d, in, out);

            MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);

            // Remaining cells depend on ghost layers
            for (const Range& r : shell) {
                stencilRangeKernel(in, out, r, d.sx, d.sxy);
            }
        }

        const double elapsed = MPI_Wtime() - t0;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, d.cart);

        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(maxElapsed));

        if (rank == 0) {
            printf("Computation time: %ld ms\n", duration.count());

            // Calculate performance metrics
            double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
            double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // Final grid lives in bufA for even iteration counts, bufB otherwise
        Real* finalGrid = (iterations % 2 == 0) ? bufA.data() : bufB.data();

        if (printResults) {
            std::vector<Real> global;
            gatherGrid(d, finalGrid, global);
            if (rank == 0) print_results(global, "Grid");
        }

        int rc = 0;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const bool valid = validateResult(d, finalGrid);
            if (valid) {
                if (rank == 0) printf("Validation: PASSED\n");
            } else {
                if (rank == 0) printf("Validation: FAILED\n");
                rc = 1;
            }
        }

        for (int dir = 0; dir < 6; ++dir) {
            MPI_Type_free(&sendT[dir]);
            MPI_Type_free(&recvT[dir]);
        }
        MPI_Comm_free(&d.cart);
        return rc;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = run(argc, argv);
    MPI_Finalize();
    return rc;
}
