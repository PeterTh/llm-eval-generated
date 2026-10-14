#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// Per-dimension triple, index 0 = x (fastest varying in memory), 1 = y, 2 = z
using Dims3 = std::array<size_t, 3>;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Half-open box [lo, hi) in local storage coordinates
struct Box {
    Dims3 lo;
    Dims3 hi;

    bool empty() const { return lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2]; }
};

// Splits n cells into p contiguous blocks; block c covers [start, start + len)
inline void splitRange(const size_t n, const int p, const int c, size_t& start, size_t& len) noexcept {
    const size_t parts = static_cast<size_t>(p);
    const size_t part = static_cast<size_t>(c);
    start = part * (n / parts) + std::min(part, n % parts);
    len = n / parts + (part < n % parts ? 1 : 0);
}

// Local storage holds the owned cells plus one ghost layer on each face.
// Storage coordinate s = owned index + 1, so ghost layers sit at s = 0 and s = ext + 1.
inline size_t storageIndex(const Dims3& s, const Dims3& stride) noexcept {
    return s[0] + s[1] * stride[1] + s[2] * stride[2];
}

// Chooses the 3D process grid with the least halo traffic per rank, subject to every
// block holding at least one cell per dimension. Returns false if no such grid exists.
bool chooseProcs(const int nprocs, const Dims3& n, std::array<int, 3>& procs) {
    bool found = false;
    double bestCost = 0.0;
    for (int px = 1; px <= nprocs; ++px) {
        if (nprocs % px != 0) continue;
        const int rest = nprocs / px;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py != 0) continue;
            const std::array<int, 3> p{px, py, rest / py};
            if (static_cast<size_t>(p[0]) > n[0] || static_cast<size_t>(p[1]) > n[1] || static_cast<size_t>(p[2]) > n[2]) continue;

            // Each partitioned dimension exchanges two faces of the block's cross-section
            double cost = 0.0;
            for (size_t d = 0; d < 3; ++d) {
                if (p[d] == 1) continue;
                double face = 1.0;
                for (size_t e = 0; e < 3; ++e) {
                    if (e != d) face *= static_cast<double>(n[e]) / p[e];
                }
                cost += face;
            }
            if (!found || cost < bestCost) {
                found = true;
                bestCost = cost;
                procs = p;
            }
        }
    }
    return found;
}

// 7-point stencil over box b. Same expression and evaluation order as the serial code,
// so results are bitwise identical to a single-process run.
void stencilBox(const Real* __restrict input, Real* __restrict output, const Dims3& stride, const Box& b) {
    const size_t sy = stride[1];
    const size_t sz = stride[2];
    for (size_t z = b.lo[2]; z < b.hi[2]; ++z) {
        for (size_t y = b.lo[1]; y < b.hi[1]; ++y) {
            const size_t row = z * sz + y * sy;
            for (size_t x = b.lo[0]; x < b.hi[0]; ++x) {
                const size_t idx = row + x;

                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - sy];
                const Real back = input[idx + sy];
                const Real bottom = input[idx - sz];
                const Real top = input[idx + sz];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// Copies storage box b of grid into buf (x fastest)
void packBox(const Real* grid, Real* buf, const Dims3& stride, const Box& b) {
    size_t k = 0;
    for (size_t z = b.lo[2]; z < b.hi[2]; ++z) {
        for (size_t y = b.lo[1]; y < b.hi[1]; ++y) {
            const Real* row = grid + z * stride[2] + y * stride[1];
            for (size_t x = b.lo[0]; x < b.hi[0]; ++x) buf[k++] = row[x];
        }
    }
}

// Copies buf (x fastest) into storage box b of grid
void unpackBox(Real* grid, const Real* buf, const Dims3& stride, const Box& b) {
    size_t k = 0;
    for (size_t z = b.lo[2]; z < b.hi[2]; ++z) {
        for (size_t y = b.lo[1]; y < b.hi[1]; ++y) {
            Real* row = grid + z * stride[2] + y * stride[1];
            for (size_t x = b.lo[0]; x < b.hi[0]; ++x) row[x] = buf[k++];
        }
    }
}

// Storage box of the owned layer (ghost == false) or the ghost layer (ghost == true)
// on the low (side 0) or high (side 1) face along dimension d
Box faceBox(const Dims3& ext, const size_t d, const size_t side, const bool ghost) {
    Box b{{1, 1, 1}, {ext[0] + 1, ext[1] + 1, ext[2] + 1}};
    if (ghost) {
        b.lo[d] = side == 0 ? 0 : ext[d] + 1;
    } else {
        b.lo[d] = side == 0 ? 1 : ext[d];
    }
    b.hi[d] = b.lo[d] + 1;
    return b;
}

// This rank's part of the global grid under a 3D Cartesian decomposition.
// Each rank owns one block and keeps a ghost layer filled from its face neighbours.
struct Decomp {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int nranks = 1;
    Dims3 n{};                               // global grid size
    std::array<int, 3> procs{};              // process grid
    Dims3 start{};                           // global index of the first owned cell
    Dims3 ext{};                             // owned cells per dimension
    Dims3 stride{};                          // storage strides including ghost layers
    size_t storageSize = 0;
    std::array<std::array<int, 2>, 3> nbr{}; // neighbour ranks [dim][0 = low, 1 = high]
    std::array<std::array<std::vector<Real>, 2>, 3> sendBuf, recvBuf;
    std::array<MPI_Request, 12> reqs{};
    int nreq = 0;
    Box inner{};                             // cells whose stencil reads owned data only
    std::vector<Box> shell;                  // remaining cells, updated once the halo has arrived

    Decomp(MPI_Comm world, const Dims3& global, const std::array<int, 3>& p) : n(global), procs(p) {
        const std::array<int, 3> periods{0, 0, 0};
        MPI_Cart_create(world, 3, procs.data(), periods.data(), 1, &cart);
        MPI_Comm_rank(cart, &rank);
        MPI_Comm_size(cart, &nranks);

        blockOf(rank, start, ext);
        stride = {1, ext[0] + 2, (ext[0] + 2) * (ext[1] + 2)};
        storageSize = (ext[0] + 2) * (ext[1] + 2) * (ext[2] + 2);

        for (size_t d = 0; d < 3; ++d) {
            int lowNbr = MPI_PROC_NULL;
            int highNbr = MPI_PROC_NULL;
            MPI_Cart_shift(cart, static_cast<int>(d), 1, &lowNbr, &highNbr);
            nbr[d] = {lowNbr, highNbr};

            size_t count = 1;
            for (size_t e = 0; e < 3; ++e) {
                if (e != d) count *= ext[e];
            }
            for (size_t side = 0; side < 2; ++side) {
                sendBuf[d][side].resize(count);
                recvBuf[d][side].resize(count);
            }
        }

        // Global boundary cells (index 0 and n-1) are never updated; interior cells use the stencil
        Box compute;
        for (size_t d = 0; d < 3; ++d) {
            compute.lo[d] = start[d] == 0 ? 2 : 1;
            compute.hi[d] = std::min(ext[d] + 1, n[d] - start[d]);
        }

        // Inner cells skip the first and last owned layer, so they never read ghost data
        Box in;
        for (size_t d = 0; d < 3; ++d) {
            in.lo[d] = std::max(compute.lo[d], size_t{2});
            in.hi[d] = std::min(compute.hi[d], ext[d]);
        }

        if (in.empty()) {
            if (!compute.empty()) shell.push_back(compute);
        } else {
            inner = in;
            // Partition compute \ inner into disjoint boxes, one per face slab
            for (size_t d = 0; d < 3; ++d) {
                for (size_t side = 0; side < 2; ++side) {
                    Box b = compute;
                    for (size_t e = 0; e < d; ++e) {
                        b.lo[e] = in.lo[e];
                        b.hi[e] = in.hi[e];
                    }
                    if (side == 0) {
                        b.hi[d] = in.lo[d];
                    } else {
                        b.lo[d] = in.hi[d];
                    }
                    if (!b.empty()) shell.push_back(b);
                }
            }
        }
    }

    Box ownedBox() const { return Box{{1, 1, 1}, {ext[0] + 1, ext[1] + 1, ext[2] + 1}}; }

    // Start and extent of the block owned by rank r of the Cartesian communicator
    void blockOf(const int r, Dims3& s, Dims3& e) const {
        std::array<int, 3> c{};
        MPI_Cart_coords(cart, r, 3, c.data());
        for (size_t d = 0; d < 3; ++d) splitRange(n[d], procs[d], c[d], s[d], e[d]);
    }

    // Fills owned cells with the values the serial code assigns at the same global positions
    void initialize(Real* grid) const {
        const Box own = ownedBox();
        for (size_t z = own.lo[2]; z < own.hi[2]; ++z) {
            for (size_t y = own.lo[1]; y < own.hi[1]; ++y) {
                for (size_t x = own.lo[0]; x < own.hi[0]; ++x) {
                    const size_t idx = idx3(start[0] + x - 1, start[1] + y - 1, start[2] + z - 1, n[0], n[1]);
                    grid[storageIndex({x, y, z}, stride)] = (idx % 19) * 1.0;
                }
            }
        }
    }

    // Packs boundary layers and starts nonblocking halo communication
    void postHalo(const Real* grid) {
        nreq = 0;
        for (size_t d = 0; d < 3; ++d) {
            for (size_t side = 0; side < 2; ++side) {
                if (nbr[d][side] == MPI_PROC_NULL) continue;
                packBox(grid, sendBuf[d][side].data(), stride, faceBox(ext, d, side, false));
            }
        }
        for (size_t d = 0; d < 3; ++d) {
            for (size_t side = 0; side < 2; ++side) {
                if (nbr[d][side] == MPI_PROC_NULL) continue;
                const int count = static_cast<int>(sendBuf[d][side].size());
                // Data sent towards side s carries tag 2d + s; the neighbour on side s sent to us
                // towards its opposite side, so we receive with tag 2d + (1 - s)
                const int recvTag = static_cast<int>(2 * d + (1 - side));
                const int sendTag = static_cast<int>(2 * d + side);
                MPI_Irecv(recvBuf[d][side].data(), count, MPI_DOUBLE, nbr[d][side], recvTag, cart, &reqs[nreq++]);
                MPI_Isend(sendBuf[d][side].data(), count, MPI_DOUBLE, nbr[d][side], sendTag, cart, &reqs[nreq++]);
            }
        }
    }

    // Waits for the halo messages and writes them into the ghost layers
    void finishHalo(Real* grid) {
        MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);
        for (size_t d = 0; d < 3; ++d) {
            for (size_t side = 0; side < 2; ++side) {
                if (nbr[d][side] == MPI_PROC_NULL) continue;
                unpackBox(grid, recvBuf[d][side].data(), stride, faceBox(ext, d, side, true));
            }
        }
    }

    // One sweep: out = stencil(in). The halo exchange overlaps with the interior update.
    void step(Real* in, Real* out) {
        postHalo(in);
        stencilBox(in, out, stride, inner);
        finishHalo(in);
        for (const Box& b : shell) stencilBox(in, out, stride, b);
    }

    // Assembles the whole grid on rank 0 in serial order (x fastest). Other ranks get an empty vector.
    std::vector<Real> gather(const Real* grid) const {
        std::vector<Real> mine(ext[0] * ext[1] * ext[2]);
        packBox(grid, mine.data(), stride, ownedBox());

        std::vector<int> counts(nranks);
        std::vector<int> displs(nranks);
        size_t total = 0;
        for (int r = 0; r < nranks; ++r) {
            Dims3 s{};
            Dims3 e{};
            blockOf(r, s, e);
            counts[r] = static_cast<int>(e[0] * e[1] * e[2]);
            displs[r] = static_cast<int>(total);
            total += counts[r];
        }

        std::vector<Real> all(rank == 0 ? total : 0);
        MPI_Gatherv(mine.data(), static_cast<int>(mine.size()), MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, cart);
        if (rank != 0) return {};

        std::vector<Real> global(n[0] * n[1] * n[2]);
        for (int r = 0; r < nranks; ++r) {
            Dims3 s{};
            Dims3 e{};
            blockOf(r, s, e);
            size_t k = static_cast<size_t>(displs[r]);
            for (size_t z = 0; z < e[2]; ++z) {
                for (size_t y = 0; y < e[1]; ++y) {
                    for (size_t x = 0; x < e[0]; ++x) {
                        global[idx3(s[0] + x, s[1] + y, s[2] + z, n[0], n[1])] = all[k++];
                    }
                }
            }
        }
        return global;
    }

    // Same checks as the serial validation, evaluated across all ranks. Messages come from rank 0.
    bool validate(const Real* grid) const {
        Real minVal = std::numeric_limits<Real>::infinity();
        Real maxVal = -std::numeric_limits<Real>::infinity();
        long long bad = 0;
        const Box own = ownedBox();
        for (size_t z = own.lo[2]; z < own.hi[2]; ++z) {
            for (size_t y = own.lo[1]; y < own.hi[1]; ++y) {
                for (size_t x = own.lo[0]; x < own.hi[0]; ++x) {
                    const Real val = grid[storageIndex({x, y, z}, stride)];
                    if (!std::isfinite(val)) ++bad;
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
            }
        }

        // 1. No NaN or Inf values
        long long badTotal = 0;
        MPI_Allreduce(&bad, &badTotal, 1, MPI_LONG_LONG, MPI_SUM, cart);
        if (badTotal > 0) {
            if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        // 2. Values should be reasonable (bounded)
        MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, cart);
        if (rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

        if (maxVal > 1e6 || minVal < -1e6) {
            if (rank == 0) printf("Validation failed: values out of expected range\n");
            return false;
        }

        // 3. Boundary values should not change significantly
        // (they are copied, so they should be close to initial values)

        return true;
    }
};

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

int runBenchmark(int argc, char** argv, int rank, int nprocs) {
    const bool root = rank == 0;

    long nx = 128;
    long ny = 0;  // Will be set to nx if not specified
    long nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses the same way)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atol(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atol(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atol(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx < 1 || ny < 1 || nz < 1 || iterations < 0) {
        if (root) {
            printf("Grid sizes must be positive and the iteration count non-negative\n");
            printUsage(argv[0]);
        }
        return 1;
    }
    const Dims3 n{static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)};

    std::array<int, 3> procs{};
    if (!chooseProcs(nprocs, n, procs)) {
        if (root) {
            printf("Cannot split a %zu x %zu x %zu grid over %d ranks (each block needs at least one cell per dimension)\n",
                   n[0], n[1], n[2], nprocs);
        }
        return 1;
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", n[0], n[1], n[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid %d x %d x %d)\n", nprocs, procs[0], procs[1], procs[2]);
    }

    Decomp dd(MPI_COMM_WORLD, n, procs);

    // Allocate local grids (double buffering, with ghost layers). Both buffers start from the
    // initial field so the boundary cells, which are never updated, match in both.
    std::vector<Real> grid1(dd.storageSize);
    std::vector<Real> grid2(dd.storageSize);

    // Initialize
    if (root) printf("Initializing grid...\n");
    dd.initialize(grid1.data());
    grid2 = grid1;

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            dd.step(grid1.data(), grid2.data());
        } else {
            dd.step(grid2.data(), grid1.data());
        }
    }

    const double localSeconds = MPI_Wtime() - t0;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);  // slowest rank

    if (root) {
        const long ms = static_cast<long>(seconds * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate performance metrics
        double cellUpdates = (double)((n[0]-2) * (n[1]-2) * (n[2]-2)) * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.data() : grid2.data();
    if (printResults) {
        const std::vector<Real> global = dd.gather(finalGrid);
        if (root) print_results(global, "Grid");
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = dd.validate(finalGrid);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int rc = runBenchmark(argc, argv, rank, nprocs);

    MPI_Finalize();
    return rc;
}
