#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;  // transferred with MPI_DOUBLE

// 3D index calculation (global grid, serial layout)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Part of the global grid owned by one rank. Storage carries one ghost layer on every side,
// so owned cell (x,y,z) with 1 <= x <= n[0] (etc.) lives at at(x,y,z).
struct Block {
    size_t n[3];       // owned cells per axis (always >= 1)
    size_t lo[3];      // global coordinate of the first owned cell per axis
    size_t stride[3];  // element strides of the local array

    size_t at(size_t x, size_t y, size_t z) const { return x + y * stride[1] + z * stride[2]; }
    size_t allocSize() const { return stride[2] * (n[2] + 2); }
};

Block makeBlock(const int coords[3], const int dims[3], const size_t N[3]) {
    Block b;
    for (int d = 0; d < 3; ++d) {
        b.lo[d] = N[d] * coords[d] / dims[d];
        b.n[d] = N[d] * (coords[d] + 1) / dims[d] - b.lo[d];
    }
    b.stride[0] = 1;
    b.stride[1] = b.n[0] + 2;
    b.stride[2] = b.stride[1] * (b.n[1] + 2);
    return b;
}

// Picks a 3D process grid (product == nprocs) that minimises the halo surface of a block.
// Every process needs at least one cell per axis. Returns false if no such grid exists.
bool chooseDims(int nprocs, const size_t N[3], int dims[3]) {
    double best = std::numeric_limits<double>::max();
    bool found = false;
    for (int a = 1; a <= nprocs; ++a) {
        if (nprocs % a != 0 || static_cast<size_t>(a) > N[0]) continue;
        const int rest = nprocs / a;
        for (int b = 1; b <= rest; ++b) {
            if (rest % b != 0) continue;
            const int c = rest / b;
            if (static_cast<size_t>(b) > N[1] || static_cast<size_t>(c) > N[2]) continue;
            const double bx = static_cast<double>(N[0]) / a;
            const double by = static_cast<double>(N[1]) / b;
            const double bz = static_cast<double>(N[2]) / c;
            const double surface = (a > 1 ? by * bz : 0.0) + (b > 1 ? bx * bz : 0.0) + (c > 1 ? bx * by : 0.0);
            if (surface < best) {
                best = surface;
                dims[0] = a;
                dims[1] = b;
                dims[2] = c;
                found = true;
            }
        }
    }
    return found;
}

// Inclusive range of local owned-cell coordinates; empty if any axis has lo > hi.
struct Box {
    long lo[3];
    long hi[3];

    bool empty() const { return lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]; }
};

// Owned cells that are updated by the stencil: global coordinates 1..N-2 on every axis.
Box updateBox(const Block& b, const size_t N[3]) {
    Box u;
    for (int d = 0; d < 3; ++d) {
        const long lo = static_cast<long>(b.lo[d]);
        u.lo[d] = std::max(1L, 2L - lo);                                      // global >= 1
        u.hi[d] = std::min(static_cast<long>(b.n[d]), static_cast<long>(N[d]) - 1 - lo);  // global <= N-2
    }
    return u;
}

// Splits the update box into the shell of cells on its six faces and the core in the middle.
// Shell cells may read ghost data; core cells read owned data only. Together they partition u.
std::vector<Box> shellBoxes(const Box& u) {
    std::vector<Box> shell;
    for (int d = 0; d < 3; ++d) {
        Box s = u;
        for (int e = 0; e < d; ++e) {  // axes already covered by earlier planes are shrunk
            s.lo[e] = u.lo[e] + 1;
            s.hi[e] = u.hi[e] - 1;
        }
        s.lo[d] = s.hi[d] = u.lo[d];
        shell.push_back(s);
        if (u.hi[d] > u.lo[d]) {
            s.lo[d] = s.hi[d] = u.hi[d];
            shell.push_back(s);
        }
    }
    return shell;
}

// Initial data depends only on the global index, exactly as in the serial version.
void initializeBlock(std::vector<Real>& grid, const Block& b, const size_t nx, const size_t ny) {
    for (size_t z = 1; z <= b.n[2]; ++z) {
        for (size_t y = 1; y <= b.n[1]; ++y) {
            for (size_t x = 1; x <= b.n[0]; ++x) {
                const size_t idx = idx3(b.lo[0] + x - 1, b.lo[1] + y - 1, b.lo[2] + z - 1, nx, ny);
                grid[b.at(x, y, z)] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil over a box of owned cells. Same summation order as the serial code.
void stencilBox(const Real* __restrict input, Real* __restrict output, const Block& b, const Box& box) {
    if (box.empty()) return;
    const size_t sy = b.stride[1];
    const size_t sz = b.stride[2];
    for (long z = box.lo[2]; z <= box.hi[2]; ++z) {
        for (long y = box.lo[1]; y <= box.hi[1]; ++y) {
            const size_t row = b.at(box.lo[0], y, z);
            const size_t end = row + static_cast<size_t>(box.hi[0] - box.lo[0]);
            for (size_t c = row; c <= end; ++c) {
                output[c] = (input[c] + input[c - 1] + input[c + 1] + input[c - sy] + input[c + sy] +
                             input[c - sz] + input[c + sz]) / 7.0;
            }
        }
    }
}

// Calls visit(index) for each cell of the plane at local coordinate `pos` along `axis`
// (owned or ghost layer). The order depends only on the block shape, so a packed face
// unpacks correctly on the neighbour, which has the same extents in the other two axes.
template <typename Visit>
void forEachPlaneCell(const Block& b, int axis, size_t pos, Visit&& visit) {
    size_t lo[3] = {1, 1, 1};
    size_t hi[3] = {b.n[0], b.n[1], b.n[2]};
    lo[axis] = hi[axis] = pos;
    for (size_t z = lo[2]; z <= hi[2]; ++z) {
        for (size_t y = lo[1]; y <= hi[1]; ++y) {
            for (size_t x = lo[0]; x <= hi[0]; ++x) {
                visit(b.at(x, y, z));
            }
        }
    }
}

// Non-blocking exchange of the six face halos with the Cartesian neighbours.
// Face f = 2*axis + side (side 0 = low, 1 = high). Faces without a neighbour
// (MPI_PROC_NULL, i.e. the global domain boundary) are skipped.
class HaloExchange {
  public:
    HaloExchange(MPI_Comm comm, const int* neighbours, const Block& b) : comm_(comm) {
        for (int f = 0; f < 6; ++f) {
            nb_[f] = neighbours[f];
            const int axis = f / 2;
            size_t count = 1;
            for (int e = 0; e < 3; ++e) {
                if (e != axis) count *= b.n[e];
            }
            send_[f].resize(count);
            recv_[f].resize(count);
        }
        std::fill(req_, req_ + 12, MPI_REQUEST_NULL);
    }

    // Packs the owned boundary planes of `a` and posts the receives and sends.
    void start(const Real* a, const Block& b) {
        for (int f = 0; f < 6; ++f) {
            if (nb_[f] == MPI_PROC_NULL) continue;
            const int axis = f / 2;
            const size_t pos = (f % 2 == 0) ? 1 : b.n[axis];  // owned plane facing the neighbour
            size_t p = 0;
            forEachPlaneCell(b, axis, pos, [&](size_t c) { send_[f][p++] = a[c]; });
            const int count = static_cast<int>(send_[f].size());
            // The neighbour sends its face facing us with tag f^1, and we send ours with tag f.
            MPI_Irecv(recv_[f].data(), count, MPI_DOUBLE, nb_[f], f ^ 1, comm_, &req_[2 * f]);
            MPI_Isend(send_[f].data(), count, MPI_DOUBLE, nb_[f], f, comm_, &req_[2 * f + 1]);
        }
    }

    // Waits for all transfers and unpacks the received planes into the ghost layer of `a`.
    void finish(Real* a, const Block& b) {
        MPI_Waitall(12, req_, MPI_STATUSES_IGNORE);
        for (int f = 0; f < 6; ++f) {
            if (nb_[f] == MPI_PROC_NULL) continue;
            const int axis = f / 2;
            const size_t pos = (f % 2 == 0) ? 0 : b.n[axis] + 1;  // ghost plane on this face
            size_t p = 0;
            forEachPlaneCell(b, axis, pos, [&](size_t c) { a[c] = recv_[f][p++]; });
        }
    }

  private:
    MPI_Comm comm_;
    int nb_[6];
    std::vector<Real> send_[6];
    std::vector<Real> recv_[6];
    MPI_Request req_[12];
};

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

struct Options {
    size_t nx;
    size_t ny;
    size_t nz;
    int iterations;
    bool validate;
    bool printResults;
};

// Runs the distributed benchmark on `comm`, whose size equals the product of dims.
// Returns the validation status (0 = OK), which is only meaningful on rank 0.
int runDistributed(MPI_Comm comm, const Options& opt, const int dims[3]) {
    const size_t N[3] = {opt.nx, opt.ny, opt.nz};
    const size_t nx = opt.nx;
    const size_t ny = opt.ny;
    const size_t nz = opt.nz;
    const size_t gridSize = nx * ny * nz;

    // Cartesian topology; reorder=0 keeps comm rank 0 as the Cartesian root
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(comm, 3, dims, periods, 0, &cart);
    int myRank = 0;
    int nprocs = 1;
    MPI_Comm_rank(cart, &myRank);
    MPI_Comm_size(cart, &nprocs);
    const bool root = myRank == 0;

    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart, myRank, 3, coords);

    int nb[6];  // low/high neighbour per axis; MPI_PROC_NULL at the domain boundary
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &nb[2 * d], &nb[2 * d + 1]);
    }

    const Block blk = makeBlock(coords, dims, N);

    // Allocate local grids (double buffering). Boundary cells are never written,
    // so both buffers keep their initial values there, as in the serial version.
    std::vector<Real> grid1(blk.allocSize(), 0.0);
    std::vector<Real> grid2(blk.allocSize(), 0.0);

    // Initialize
    if (root) printf("Initializing grid...\n");
    initializeBlock(grid1, blk, nx, ny);
    initializeBlock(grid2, blk, nx, ny);

    const Box upd = updateBox(blk, N);
    const bool haveCells = !upd.empty();
    Box core = upd;  // interior cells whose neighbours are all owned
    for (int d = 0; d < 3; ++d) {
        core.lo[d] += 1;
        core.hi[d] -= 1;
    }
    const std::vector<Box> shell = haveCells ? shellBoxes(upd) : std::vector<Box>{};

    HaloExchange halo(cart, nb, blk);

    // Run stencil iterations; halo transfers overlap with the core update
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    Real* in = grid1.data();
    Real* out = grid2.data();
    for (int iter = 0; iter < opt.iterations; ++iter) {
        halo.start(in, blk);
        if (haveCells) stencilBox(in, out, blk, core);
        halo.finish(in, blk);
        for (const Box& s : shell) {
            stencilBox(in, out, blk, s);
        }
        std::swap(in, out);
    }

    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    // Gather the final owned cells on rank 0 (`in` holds the latest result)
    std::vector<Real> packed;
    packed.reserve(blk.n[0] * blk.n[1] * blk.n[2]);
    for (size_t z = 1; z <= blk.n[2]; ++z) {
        for (size_t y = 1; y <= blk.n[1]; ++y) {
            for (size_t x = 1; x <= blk.n[0]; ++x) {
                packed.push_back(in[blk.at(x, y, z)]);
            }
        }
    }

    std::vector<Block> blocks;
    std::vector<int> counts(nprocs, 0);
    std::vector<int> displs(nprocs, 0);
    std::vector<Real> staging;
    std::vector<Real> finalGrid;
    if (root) {
        blocks.resize(nprocs);
        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int c[3];
            MPI_Cart_coords(cart, r, 3, c);
            blocks[r] = makeBlock(c, dims, N);
            counts[r] = static_cast<int>(blocks[r].n[0] * blocks[r].n[1] * blocks[r].n[2]);
            displs[r] = offset;
            offset += counts[r];
        }
        staging.resize(gridSize);
        finalGrid.resize(gridSize);
    }
    MPI_Gatherv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                staging.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, cart);

    int status = 0;
    if (root) {
        for (int r = 0; r < nprocs; ++r) {
            const Block& bb = blocks[r];
            size_t p = static_cast<size_t>(displs[r]);
            for (size_t z = 1; z <= bb.n[2]; ++z) {
                for (size_t y = 1; y <= bb.n[1]; ++y) {
                    for (size_t x = 1; x <= bb.n[0]; ++x) {
                        finalGrid[idx3(bb.lo[0] + x - 1, bb.lo[1] + y - 1, bb.lo[2] + z - 1, nx, ny)] = staging[p++];
                    }
                }
            }
        }

        // Calculate performance metrics (slowest rank determines the time)
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * opt.iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (opt.printResults) {
            print_results(finalGrid, "Grid");
        }

        // Validation
        if (opt.validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Comm_free(&cart);
    return status;
}

int runBenchmark(int argc, char** argv) {
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = rank == 0;

    Options opt{128, 0, 0, 10, false, false};  // ny/nz set to nx if not specified

    // Parse command line arguments (identical on every rank, so all ranks take the same path)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            opt.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            opt.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            opt.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            opt.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            opt.printResults = true;
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

    if (opt.ny == 0) opt.ny = opt.nx;
    if (opt.nz == 0) opt.nz = opt.nx;

    if (opt.nx == 0 || opt.ny == 0 || opt.nz == 0) {
        if (root) printf("Error: grid dimensions must be at least 1\n");
        return 1;
    }

    // Use the largest number of ranks that yields a valid decomposition (>= 1 cell per rank
    // per axis); any remaining ranks stay idle.
    const size_t N[3] = {opt.nx, opt.ny, opt.nz};
    int dims[3] = {1, 1, 1};
    int used = 1;
    for (int p = nprocs; p >= 1; --p) {
        if (chooseDims(p, N, dims)) {
            used = p;
            break;
        }
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", opt.nx, opt.ny, opt.nz);
        printf("Iterations: %d\n", opt.iterations);
        printf("Validation: %s\n", opt.validate ? "enabled" : "disabled");
        printf("MPI processes: %d of %d (%d x %d x %d)\n", used, nprocs, dims[0], dims[1], dims[2]);
    }

    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < used ? 0 : MPI_UNDEFINED, rank, &active);

    int status = 0;
    if (active != MPI_COMM_NULL) {
        status = runDistributed(active, opt, dims);
        MPI_Comm_free(&active);
    }

    // All ranks, including idle ones, return the same status
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int status = runBenchmark(argc, argv);
    MPI_Finalize();
    return status;
}
