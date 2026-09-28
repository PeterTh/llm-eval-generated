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

#define MPI_REAL_TYPE MPI_DOUBLE

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Domain decomposition helpers
// ---------------------------------------------------------------------------

// Dimension ordering used throughout: 0 -> z (slowest), 1 -> y, 2 -> x (fastest).
enum { DZ = 0, DY = 1, DX = 2 };

// Block-distribution of n cells over p parts: part i gets blockSize cells
// starting at blockOffset.
inline size_t blockSize(const size_t n, const int p, const int i) noexcept {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    return base + (static_cast<size_t>(i) < rem ? 1 : 0);
}

inline size_t blockOffset(const size_t n, const int p, const int i) noexcept {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    return static_cast<size_t>(i) * base + std::min(static_cast<size_t>(i), rem);
}

// Greedy factorization of the rank count onto the three grid dimensions:
// every prime factor is given to the dimension that currently has the largest
// local extent. Ties prefer z, then y, then x, which keeps the (contiguous)
// x-direction split as little as possible.
void computeDims(const int numRanks, const size_t n[3], int dims[3]) {
    dims[0] = dims[1] = dims[2] = 1;

    std::vector<int> factors;
    int rest = numRanks;
    for (int f = 2; f * f <= rest; ++f) {
        while (rest % f == 0) {
            factors.push_back(f);
            rest /= f;
        }
    }
    if (rest > 1) factors.push_back(rest);
    std::sort(factors.begin(), factors.end(), std::greater<int>());

    for (const int f : factors) {
        int best = 0;
        size_t bestExtent = 0;
        for (int d = 0; d < 3; ++d) {
            const size_t extent = n[d] / static_cast<size_t>(dims[d]);
            if (extent > bestExtent) {
                bestExtent = extent;
                best = d;
            }
        }
        dims[best] *= f;
    }
}

// ---------------------------------------------------------------------------
// Local grid description
// ---------------------------------------------------------------------------

struct LocalGrid {
    size_t n[3] = {0, 0, 0};      // global sizes (z, y, x)
    size_t loc[3] = {0, 0, 0};    // owned local sizes (z, y, x)
    size_t off[3] = {0, 0, 0};    // global offset of the owned block
    size_t full[3] = {0, 0, 0};   // local sizes including one halo layer
    size_t stride[3] = {0, 0, 0}; // linear strides for z, y, x

    // Half-open local index range [lo, hi) of cells that are updated by the
    // stencil in dimension d (owned cells are 1..loc[d], global boundary cells
    // are excluded because they are simply copied).
    size_t interiorLo(const int d) const noexcept { return off[d] == 0 ? 2 : 1; }
    size_t interiorHi(const int d) const noexcept { return off[d] + loc[d] == n[d] ? loc[d] : loc[d] + 1; }

    size_t size() const noexcept { return full[0] * full[1] * full[2]; }
};

// Initialize including halo cells; halos get the value their owner would
// compute, so the initial state is globally consistent.
void initializeGrid(std::vector<Real>& grid, const LocalGrid& g, const size_t nx, const size_t ny) {
    for (size_t lz = 0; lz < g.full[DZ]; ++lz) {
        const long gz = static_cast<long>(g.off[DZ]) + static_cast<long>(lz) - 1;
        for (size_t ly = 0; ly < g.full[DY]; ++ly) {
            const long gy = static_cast<long>(g.off[DY]) + static_cast<long>(ly) - 1;
            for (size_t lx = 0; lx < g.full[DX]; ++lx) {
                const long gx = static_cast<long>(g.off[DX]) + static_cast<long>(lx) - 1;
                const size_t local = lz * g.stride[DZ] + ly * g.stride[DY] + lx;

                // Cells outside the global domain are never read.
                if (gz < 0 || gy < 0 || gx < 0 || static_cast<size_t>(gz) >= g.n[DZ] || static_cast<size_t>(gy) >= g.n[DY] ||
                    static_cast<size_t>(gx) >= g.n[DX]) {
                    grid[local] = 0.0;
                    continue;
                }

                const size_t idx = idx3(static_cast<size_t>(gx), static_cast<size_t>(gy), static_cast<size_t>(gz), nx, ny);
                grid[local] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil over the half-open local box [lo, hi) in each dimension.
inline void stencilBox(const Real* __restrict__ input, Real* __restrict__ output, const LocalGrid& g, const size_t lo[3],
                       const size_t hi[3]) {
    const size_t sz = g.stride[DZ];
    const size_t sy = g.stride[DY];

    for (size_t z = lo[DZ]; z < hi[DZ]; ++z) {
        for (size_t y = lo[DY]; y < hi[DY]; ++y) {
            const size_t row = z * sz + y * sy;
            const Real* __restrict__ in = input + row;
            Real* __restrict__ out = output + row;
            for (size_t x = lo[DX]; x < hi[DX]; ++x) {
                out[x] = (in[x] + in[x - 1] + in[x + 1] + in[x - sy] + in[x + sy] + in[x - sz] + in[x + sz]) / 7.0;
            }
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

// Gather the distributed grid into the full global array on the root rank.
void gatherGrid(const std::vector<Real>& local, const LocalGrid& g, std::vector<Real>& global, MPI_Comm comm, const int rank,
                const int numRanks, const int dims[3], const int root) {
    const size_t ownedCount = g.loc[DZ] * g.loc[DY] * g.loc[DX];

    std::vector<Real> sendBuf(ownedCount);
    for (size_t z = 0; z < g.loc[DZ]; ++z) {
        for (size_t y = 0; y < g.loc[DY]; ++y) {
            const Real* src = local.data() + (z + 1) * g.stride[DZ] + (y + 1) * g.stride[DY] + 1;
            std::copy(src, src + g.loc[DX], sendBuf.data() + (z * g.loc[DY] + y) * g.loc[DX]);
        }
    }

    std::vector<int> counts, displs;
    std::vector<Real> recvBuf;
    if (rank == root) {
        counts.resize(numRanks);
        displs.resize(numRanks);
        size_t total = 0;
        for (int r = 0; r < numRanks; ++r) {
            int coords[3];
            MPI_Cart_coords(comm, r, 3, coords);
            const size_t c = blockSize(g.n[DZ], dims[DZ], coords[DZ]) * blockSize(g.n[DY], dims[DY], coords[DY]) *
                             blockSize(g.n[DX], dims[DX], coords[DX]);
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(total);
            total += c;
        }
        recvBuf.resize(total);
    }

    MPI_Gatherv(sendBuf.data(), static_cast<int>(ownedCount), MPI_REAL_TYPE, recvBuf.data(), counts.data(), displs.data(),
                MPI_REAL_TYPE, root, comm);

    if (rank != root) return;

    global.resize(g.n[DZ] * g.n[DY] * g.n[DX]);
    for (int r = 0; r < numRanks; ++r) {
        int coords[3];
        MPI_Cart_coords(comm, r, 3, coords);
        const size_t lz = blockSize(g.n[DZ], dims[DZ], coords[DZ]);
        const size_t ly = blockSize(g.n[DY], dims[DY], coords[DY]);
        const size_t lx = blockSize(g.n[DX], dims[DX], coords[DX]);
        const size_t oz = blockOffset(g.n[DZ], dims[DZ], coords[DZ]);
        const size_t oy = blockOffset(g.n[DY], dims[DY], coords[DY]);
        const size_t ox = blockOffset(g.n[DX], dims[DX], coords[DX]);

        const Real* src = recvBuf.data() + displs[r];
        for (size_t z = 0; z < lz; ++z) {
            for (size_t y = 0; y < ly; ++y) {
                Real* dst = global.data() + idx3(ox, oy + y, oz + z, g.n[DX], g.n[DY]);
                std::copy(src, src + lx, dst);
                src += lx;
            }
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const size_t globalN[3] = {nz, ny, nx};

    int dims[3];
    computeDims(numRanks, globalN, dims);

    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);

    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    const bool isRoot = (rank == 0);

    if (isRoot) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (decomposition %d x %d x %d in z x y x x)\n", numRanks, dims[DZ], dims[DY], dims[DX]);
    }

    LocalGrid g;
    for (int d = 0; d < 3; ++d) {
        g.n[d] = globalN[d];
        g.loc[d] = blockSize(globalN[d], dims[d], coords[d]);
        g.off[d] = blockOffset(globalN[d], dims[d], coords[d]);
        g.full[d] = g.loc[d] + 2;
    }
    g.stride[DX] = 1;
    g.stride[DY] = g.full[DX];
    g.stride[DZ] = g.full[DX] * g.full[DY];

    if (g.loc[DZ] == 0 || g.loc[DY] == 0 || g.loc[DX] == 0) {
        if (isRoot) printf("Error: too many MPI ranks for the given grid size\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate grids (double buffering)
    std::vector<Real> grid1(g.size());
    std::vector<Real> grid2(g.size());

    // Initialize. Global boundary cells are only ever copied by the stencil, so
    // initializing both buffers keeps them at their initial values throughout.
    if (isRoot) printf("Initializing grid...\n");
    initializeGrid(grid1, g, nx, ny);
    grid2 = grid1;

    // ---- halo exchange setup ------------------------------------------------
    int neighbor[3][2];
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &neighbor[d][0], &neighbor[d][1]);
    }

    const int fullSizes[3] = {static_cast<int>(g.full[DZ]), static_cast<int>(g.full[DY]), static_cast<int>(g.full[DX])};
    const int ownedSizes[3] = {static_cast<int>(g.loc[DZ]), static_cast<int>(g.loc[DY]), static_cast<int>(g.loc[DX])};

    MPI_Datatype sendType[3][2], recvType[3][2];
    for (int d = 0; d < 3; ++d) {
        for (int side = 0; side < 2; ++side) {
            int sub[3] = {ownedSizes[0], ownedSizes[1], ownedSizes[2]};
            sub[d] = 1;

            int sendStart[3] = {1, 1, 1};
            sendStart[d] = (side == 0) ? 1 : ownedSizes[d];
            int recvStart[3] = {1, 1, 1};
            recvStart[d] = (side == 0) ? 0 : ownedSizes[d] + 1;

            MPI_Type_create_subarray(3, fullSizes, sub, sendStart, MPI_ORDER_C, MPI_REAL_TYPE, &sendType[d][side]);
            MPI_Type_commit(&sendType[d][side]);
            MPI_Type_create_subarray(3, fullSizes, sub, recvStart, MPI_ORDER_C, MPI_REAL_TYPE, &recvType[d][side]);
            MPI_Type_commit(&recvType[d][side]);
        }
    }

    // Persistent requests, one set per buffer (halos are exchanged on the
    // current input buffer).
    MPI_Request requests[2][12];
    std::vector<Real>* buffers[2] = {&grid1, &grid2};
    for (int b = 0; b < 2; ++b) {
        Real* data = buffers[b]->data();
        int k = 0;
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                MPI_Recv_init(data, 1, recvType[d][side], neighbor[d][side], d * 2 + (1 - side), cart, &requests[b][k++]);
                MPI_Send_init(data, 1, sendType[d][side], neighbor[d][side], d * 2 + side, cart, &requests[b][k++]);
            }
        }
    }

    // ---- computation boxes --------------------------------------------------
    // Cells to update, restricted to the global interior.
    size_t lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
        lo[d] = g.interiorLo(d);
        hi[d] = g.interiorHi(d);
        if (hi[d] < lo[d]) hi[d] = lo[d];
    }

    // Split into an inner part (independent of halo data) and the surrounding
    // shell, so that the halo exchange can overlap with the inner computation.
    size_t innerLo[3], innerHi[3];
    for (int d = 0; d < 3; ++d) {
        innerLo[d] = std::max(lo[d], std::min<size_t>(2, g.loc[d] + 1));
        innerHi[d] = std::min(hi[d], std::max<size_t>(g.loc[d], innerLo[d]));
        if (innerHi[d] < innerLo[d]) innerHi[d] = innerLo[d];
    }

    // The six shell slabs (half-open boxes).
    size_t shellLo[6][3], shellHi[6][3];
    int numShell = 0;
    {
        auto addBox = [&](const size_t a[3], const size_t b[3]) {
            for (int d = 0; d < 3; ++d) {
                if (b[d] <= a[d]) return;
            }
            for (int d = 0; d < 3; ++d) {
                shellLo[numShell][d] = a[d];
                shellHi[numShell][d] = b[d];
            }
            ++numShell;
        };

        size_t a[3], b[3];
        // z slabs (full y/x extent)
        a[DZ] = lo[DZ];       b[DZ] = innerLo[DZ];  a[DY] = lo[DY]; b[DY] = hi[DY]; a[DX] = lo[DX]; b[DX] = hi[DX];
        addBox(a, b);
        a[DZ] = innerHi[DZ];  b[DZ] = hi[DZ];
        addBox(a, b);
        // y slabs (inner z, full x extent)
        a[DZ] = innerLo[DZ];  b[DZ] = innerHi[DZ];
        a[DY] = lo[DY];       b[DY] = innerLo[DY];
        addBox(a, b);
        a[DY] = innerHi[DY];  b[DY] = hi[DY];
        addBox(a, b);
        // x slabs (inner z/y)
        a[DY] = innerLo[DY];  b[DY] = innerHi[DY];
        a[DX] = lo[DX];       b[DX] = innerLo[DX];
        addBox(a, b);
        a[DX] = innerHi[DX];  b[DX] = hi[DX];
        addBox(a, b);
    }

    // Run stencil iterations
    if (isRoot) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        const int b = iter % 2;
        const Real* in = buffers[b]->data();
        Real* out = buffers[1 - b]->data();

        MPI_Startall(12, requests[b]);
        stencilBox(in, out, g, innerLo, innerHi);
        MPI_Waitall(12, requests[b], MPI_STATUSES_IGNORE);
        for (int s = 0; s < numShell; ++s) {
            stencilBox(in, out, g, shellLo[s], shellHi[s]);
        }
    }

    MPI_Barrier(cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDurationMs = duration.count();
    long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG, MPI_MAX, 0, cart);

    if (isRoot) {
        printf("Computation time: %ld ms\n", globalDurationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (globalDurationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        gatherGrid(localFinal, g, finalGrid, cart, rank, numRanks, dims, 0);

        if (isRoot) {
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
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, cart);
    }

    for (int b = 0; b < 2; ++b) {
        for (int k = 0; k < 12; ++k) MPI_Request_free(&requests[b][k]);
    }
    for (int d = 0; d < 3; ++d) {
        for (int side = 0; side < 2; ++side) {
            MPI_Type_free(&sendType[d][side]);
            MPI_Type_free(&recvType[d][side]);
        }
    }
    MPI_Comm_free(&cart);
    MPI_Finalize();

    return exitCode;
}
