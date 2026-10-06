#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Domain decomposition: the grid is split into blocks along Z and Y (X is kept
// whole so rows stay contiguous for vectorization). Each rank owns lz x ly x nx
// cells and stores them with a one-cell halo in Z and Y.
struct Domain {
    size_t nx, ny, nz;      // global grid size
    size_t z0, y0;          // global offset of first owned cell
    size_t lz, ly;          // number of owned cells in Z and Y
    size_t LY;              // local row count per plane (ly + 2)
    size_t plane;           // local plane size (LY * nx)
    int down, up;           // Z neighbors
    int south, north;       // Y neighbors
    MPI_Datatype yRowType;  // lz rows of nx values, strided by one plane

    size_t localSize() const noexcept { return (lz + 2) * plane; }
    size_t lidx(const size_t x, const size_t y, const size_t z) const noexcept { return (z * LY + y) * nx + x; }
};

static void blockRange(const size_t n, const int p, const int c, size_t& start, size_t& count) {
    const size_t base = n / p, rem = n % p;
    count = base + (static_cast<size_t>(c) < rem ? 1 : 0);
    start = c * base + std::min<size_t>(c, rem);
}

// Choose the number of active ranks and a pz x py process grid minimizing halo volume
static void chooseDecomposition(const int nprocs, const size_t ny, const size_t nz, int& pz, int& py) {
    for (int p = nprocs; p >= 1; --p) {
        double bestCost = -1.0;
        for (int a = 1; a <= p; ++a) {
            if (p % a) continue;
            const int b = p / a;
            if (static_cast<size_t>(a) > nz || static_cast<size_t>(b) > ny) continue;
            const double cost = (double)ny * (a - 1) + (double)nz * (b - 1);
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                pz = a;
                py = b;
            }
        }
        if (bestCost >= 0.0) return;
    }
    pz = py = 1;
}

void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t idx = idx3(x, d.y0 + y - 1, d.z0 + z - 1, d.nx, d.ny);
                grid[d.lidx(x, y, z)] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on local rows [zb,ze) x [yb,ye); global boundary cells are copied
static void stencilRows(const Real* __restrict in, Real* __restrict out, const Domain& d,
                        const size_t zb, const size_t ze, const size_t yb, const size_t ye) {
    const size_t nx = d.nx;
    const size_t plane = d.plane;
    for (size_t z = zb; z < ze; ++z) {
        const size_t gz = d.z0 + z - 1;
        const bool zBoundary = (gz == 0 || gz == d.nz - 1);
        for (size_t y = yb; y < ye; ++y) {
            const size_t gy = d.y0 + y - 1;
            const size_t base = d.lidx(0, y, z);
            const Real* __restrict c = in + base;
            Real* __restrict o = out + base;
            if (zBoundary || gy == 0 || gy == d.ny - 1) {
                std::memcpy(o, c, nx * sizeof(Real));
                continue;
            }
            o[0] = c[0];
            o[nx - 1] = c[nx - 1];
            const Real* __restrict f = c - nx;
            const Real* __restrict bk = c + nx;
            const Real* __restrict bo = c - plane;
            const Real* __restrict tp = c + plane;
            #pragma GCC ivdep
            for (size_t x = 1; x + 1 < nx; ++x) {
                // Simple averaging stencil (same operation order as the reference)
                o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + bk[x] + bo[x] + tp[x]) / 7.0;
            }
        }
    }
}

// One stencil iteration: halo exchange overlapped with interior computation
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const Domain& d, MPI_Comm comm) {
    // Halo exchange touches only halo cells of input, which are never written here
    Real* in = const_cast<Real*>(input.data());
    Real* out = output.data();
    const int faceCount = static_cast<int>(d.ly * d.nx);
    MPI_Request reqs[8];
    int nreq = 0;

    MPI_Irecv(in + d.lidx(0, 1, 0), faceCount, MPI_DOUBLE, d.down, 0, comm, &reqs[nreq++]);
    MPI_Irecv(in + d.lidx(0, 1, d.lz + 1), faceCount, MPI_DOUBLE, d.up, 1, comm, &reqs[nreq++]);
    MPI_Irecv(in + d.lidx(0, 0, 1), 1, d.yRowType, d.south, 2, comm, &reqs[nreq++]);
    MPI_Irecv(in + d.lidx(0, d.ly + 1, 1), 1, d.yRowType, d.north, 3, comm, &reqs[nreq++]);
    MPI_Isend(in + d.lidx(0, 1, d.lz), faceCount, MPI_DOUBLE, d.up, 0, comm, &reqs[nreq++]);
    MPI_Isend(in + d.lidx(0, 1, 1), faceCount, MPI_DOUBLE, d.down, 1, comm, &reqs[nreq++]);
    MPI_Isend(in + d.lidx(0, d.ly, 1), 1, d.yRowType, d.north, 2, comm, &reqs[nreq++]);
    MPI_Isend(in + d.lidx(0, 1, 1), 1, d.yRowType, d.south, 3, comm, &reqs[nreq++]);

    // Interior cells not depending on halos
    if (d.lz > 2 && d.ly > 2) {
        stencilRows(in, out, d, 2, d.lz, 2, d.ly);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Outer shell of the local block
    stencilRows(in, out, d, 1, 2, 1, d.ly + 1);
    if (d.lz > 1) stencilRows(in, out, d, d.lz, d.lz + 1, 1, d.ly + 1);
    if (d.lz > 2) {
        stencilRows(in, out, d, 2, d.lz, 1, 2);
        if (d.ly > 1) stencilRows(in, out, d, 2, d.lz, d.ly, d.ly + 1);
    }
}

bool validateResult(const std::vector<Real>& grid, const Domain& d, MPI_Comm comm, const bool isRoot) {
    // Simple sanity checks (on the distributed grid)

    // 1. No NaN or Inf values
    int localBad = 0;
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            const Real* row = grid.data() + d.lidx(0, y, z);
            for (size_t x = 0; x < d.nx; ++x) {
                const Real val = row[x];
                if (std::isnan(val) || std::isinf(val)) localBad = 1;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }
    int bad = 0;
    MPI_Allreduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, comm);
    if (bad) {
        if (isRoot) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real gMin, gMax;
    MPI_Allreduce(&minVal, &gMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (isRoot) printf("Value range: [%.6f, %.6f]\n", gMin, gMax);

    // After averaging, values should be somewhat bounded
    if (gMax > 1e6 || gMin < -1e6) {
        if (isRoot) printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

// Collect the distributed grid into a full global array on rank 0 of comm
static void gatherGrid(const std::vector<Real>& local, const Domain& d, MPI_Comm comm,
                       const int pz, const int py, std::vector<Real>& global) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Request sendReq;
    MPI_Datatype sendType;
    {
        const int sizes[3] = {(int)d.lz + 2, (int)d.LY, (int)d.nx};
        const int subs[3] = {(int)d.lz, (int)d.ly, (int)d.nx};
        const int starts[3] = {1, 1, 0};
        MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &sendType);
        MPI_Type_commit(&sendType);
        MPI_Isend(local.data(), 1, sendType, 0, 99, comm, &sendReq);
    }

    if (rank == 0) {
        global.assign(d.nx * d.ny * d.nz, 0.0);
        std::vector<MPI_Request> reqs(nprocs);
        std::vector<MPI_Datatype> types(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            int coords[2];
            MPI_Cart_coords(comm, r, 2, coords);
            size_t zs, zc, ys, yc;
            blockRange(d.nz, pz, coords[0], zs, zc);
            blockRange(d.ny, py, coords[1], ys, yc);
            const int sizes[3] = {(int)d.nz, (int)d.ny, (int)d.nx};
            const int subs[3] = {(int)zc, (int)yc, (int)d.nx};
            const int starts[3] = {(int)zs, (int)ys, 0};
            MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &types[r]);
            MPI_Type_commit(&types[r]);
            MPI_Irecv(global.data(), 1, types[r], r, 99, comm, &reqs[r]);
        }
        MPI_Waitall(nprocs, reqs.data(), MPI_STATUSES_IGNORE);
        for (auto& t : types) MPI_Type_free(&t);
    }
    MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
    MPI_Type_free(&sendType);
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
    
    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Set up the process grid (ranks beyond what the grid can use stay idle)
    int pz = 1, py = 1;
    chooseDecomposition(worldSize, ny, nz, pz, py);
    const int activeProcs = pz * py;
    const bool active = worldRank < activeProcs;
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);

    int exitCode = 0;
    if (active) {
        MPI_Comm cart;
        const int dims[2] = {pz, py};
        const int periods[2] = {0, 0};
        MPI_Cart_create(activeComm, 2, dims, periods, 0, &cart);
        int cartRank, coords[2];
        MPI_Comm_rank(cart, &cartRank);
        MPI_Cart_coords(cart, cartRank, 2, coords);

        Domain d{};
        d.nx = nx; d.ny = ny; d.nz = nz;
        blockRange(nz, pz, coords[0], d.z0, d.lz);
        blockRange(ny, py, coords[1], d.y0, d.ly);
        d.LY = d.ly + 2;
        d.plane = d.LY * nx;
        MPI_Cart_shift(cart, 0, 1, &d.down, &d.up);
        MPI_Cart_shift(cart, 1, 1, &d.south, &d.north);
        MPI_Type_vector((int)d.lz, (int)nx, (int)d.plane, MPI_DOUBLE, &d.yRowType);
        MPI_Type_commit(&d.yRowType);

        // Allocate local grids with halos (double buffering)
        std::vector<Real> grid1(d.localSize(), 0.0);
        std::vector<Real> grid2(d.localSize(), 0.0);

        // Initialize
        if (isRoot) printf("Initializing grid...\n");
        initializeGrid(grid1, d);

        // Run stencil iterations
        if (isRoot) printf("Running stencil computation...\n");
        MPI_Barrier(cart);
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, d, cart);
            } else {
                stencilIteration(grid2, grid1, d, cart);
            }
        }

        MPI_Barrier(cart);
        auto end = std::chrono::high_resolution_clock::now();
        const long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        long maxDuration = 0;
        MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, cart);
        auto duration = std::chrono::milliseconds(maxDuration);

        if (isRoot) {
            printf("Computation time: %ld ms\n", duration.count());

            // Calculate performance metrics
            double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
            double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // Print results for external validation
        const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
        if (printResults) {
            std::vector<Real> globalGrid;
            gatherGrid(finalGrid, d, cart, pz, py, globalGrid);
            if (isRoot) print_results(globalGrid, "Grid");
        }

        // Validation
        if (validate) {
            if (isRoot) printf("Validating result...\n");
            bool valid = validateResult(finalGrid, d, cart, isRoot);

            if (valid) {
                if (isRoot) printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                if (isRoot) printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }

        MPI_Type_free(&d.yRowType);
        MPI_Comm_free(&cart);
        MPI_Comm_free(&activeComm);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
