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

// 3D index calculation (global grid, used for initialization / gathering)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Block distribution of n cells over p parts: part i gets n/p (+1 for the first n%p parts)
inline void blockRange(const long n, const long p, const long i, long& begin, long& count) {
    const long base = n / p;
    const long rem = n % p;
    count = base + (i < rem ? 1 : 0);
    begin = i * base + std::min(i, rem);
}

// Local (per-rank) domain description, including one layer of ghost cells in every direction
struct Domain {
    long x0 = 0, y0 = 0, z0 = 0;      // global index of first owned cell
    long lnx = 0, lny = 0, lnz = 0;   // number of owned cells per dimension
    long sx = 0, sy = 0, sz = 0;      // padded (ghost-included) extents
    size_t plane = 0;                 // sx * sy
    size_t size = 0;                  // sx * sy * sz

    // index into the padded local array; lx/ly/lz are ghost-inclusive (owned cells are 1..ln*)
    inline size_t at(const long lx, const long ly, const long lz) const noexcept {
        return static_cast<size_t>((lz * sy + ly) * sx + lx);
    }
};

// Initialize the owned cells of the local block exactly as the serial code initializes the global grid
void initializeGrid(std::vector<Real>& grid, const Domain& d, const size_t nx, const size_t ny) {
    for (long lz = 1; lz <= d.lnz; ++lz) {
        const size_t gz = static_cast<size_t>(d.z0 + lz - 1);
        for (long ly = 1; ly <= d.lny; ++ly) {
            const size_t gy = static_cast<size_t>(d.y0 + ly - 1);
            const size_t gbase = gz * (nx * ny) + gy * nx + static_cast<size_t>(d.x0);
            Real* __restrict out = grid.data() + d.at(1, ly, lz);
            for (long lx = 0; lx < d.lnx; ++lx) {
                out[lx] = static_cast<Real>((gbase + static_cast<size_t>(lx)) % 19) * 1.0;
            }
        }
    }
}

// Ghost cell exchange for a 7-point stencil: only face neighbours are required (no edges/corners).
struct HaloExchange {
    MPI_Comm cart = MPI_COMM_NULL;
    int nbXm = MPI_PROC_NULL, nbXp = MPI_PROC_NULL;
    int nbYm = MPI_PROC_NULL, nbYp = MPI_PROC_NULL;
    int nbZm = MPI_PROC_NULL, nbZp = MPI_PROC_NULL;

    // Packed send/receive buffers for the strided X and Y faces
    std::vector<Real> sendXm, sendXp, recvXm, recvXp;
    std::vector<Real> sendYm, sendYp, recvYm, recvYp;

    MPI_Request reqs[12];
    int nreq = 0;

    void setup(MPI_Comm c, const Domain& d) {
        cart = c;
        MPI_Cart_shift(cart, 2, 1, &nbXm, &nbXp);
        MPI_Cart_shift(cart, 1, 1, &nbYm, &nbYp);
        MPI_Cart_shift(cart, 0, 1, &nbZm, &nbZp);

        const size_t nX = static_cast<size_t>(d.lny) * static_cast<size_t>(d.lnz);
        const size_t nY = static_cast<size_t>(d.lnx) * static_cast<size_t>(d.lnz);
        sendXm.resize(nX); sendXp.resize(nX); recvXm.resize(nX); recvXp.resize(nX);
        sendYm.resize(nY); sendYp.resize(nY); recvYm.resize(nY); recvYp.resize(nY);
    }

    // Pack the outgoing faces and post all non-blocking transfers
    void start(std::vector<Real>& g, const Domain& d) {
        Real* __restrict data = g.data();
        nreq = 0;

        // --- Z faces: whole XY planes are contiguous, no packing required ---
        if (d.lnz > 0) {
            MPI_Irecv(data + d.at(0, 0, 0), static_cast<int>(d.plane), MPI_DOUBLE, nbZm, 0, cart, &reqs[nreq++]);
            MPI_Irecv(data + d.at(0, 0, d.lnz + 1), static_cast<int>(d.plane), MPI_DOUBLE, nbZp, 1, cart, &reqs[nreq++]);
            MPI_Isend(data + d.at(0, 0, 1), static_cast<int>(d.plane), MPI_DOUBLE, nbZm, 1, cart, &reqs[nreq++]);
            MPI_Isend(data + d.at(0, 0, d.lnz), static_cast<int>(d.plane), MPI_DOUBLE, nbZp, 0, cart, &reqs[nreq++]);
        }

        // --- Y faces: pack the owned x-rows of the first/last owned y-plane ---
        const int cntY = static_cast<int>(sendYm.size());
        if (cntY > 0) {
            for (long lz = 0; lz < d.lnz; ++lz) {
                std::memcpy(sendYm.data() + lz * d.lnx, data + d.at(1, 1, lz + 1), sizeof(Real) * d.lnx);
                std::memcpy(sendYp.data() + lz * d.lnx, data + d.at(1, d.lny, lz + 1), sizeof(Real) * d.lnx);
            }
            MPI_Irecv(recvYm.data(), cntY, MPI_DOUBLE, nbYm, 2, cart, &reqs[nreq++]);
            MPI_Irecv(recvYp.data(), cntY, MPI_DOUBLE, nbYp, 3, cart, &reqs[nreq++]);
            MPI_Isend(sendYm.data(), cntY, MPI_DOUBLE, nbYm, 3, cart, &reqs[nreq++]);
            MPI_Isend(sendYp.data(), cntY, MPI_DOUBLE, nbYp, 2, cart, &reqs[nreq++]);
        }

        // --- X faces: strided gather of the first/last owned x-column ---
        const int cntX = static_cast<int>(sendXm.size());
        if (cntX > 0) {
            size_t k = 0;
            for (long lz = 1; lz <= d.lnz; ++lz) {
                for (long ly = 1; ly <= d.lny; ++ly, ++k) {
                    sendXm[k] = data[d.at(1, ly, lz)];
                    sendXp[k] = data[d.at(d.lnx, ly, lz)];
                }
            }
            MPI_Irecv(recvXm.data(), cntX, MPI_DOUBLE, nbXm, 4, cart, &reqs[nreq++]);
            MPI_Irecv(recvXp.data(), cntX, MPI_DOUBLE, nbXp, 5, cart, &reqs[nreq++]);
            MPI_Isend(sendXm.data(), cntX, MPI_DOUBLE, nbXm, 5, cart, &reqs[nreq++]);
            MPI_Isend(sendXp.data(), cntX, MPI_DOUBLE, nbXp, 4, cart, &reqs[nreq++]);
        }

    }

    // Wait for the transfers and unpack the incoming faces into the ghost layers
    void finish(std::vector<Real>& g, const Domain& d) {
        Real* __restrict data = g.data();
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        nreq = 0;

        const int cntY = static_cast<int>(sendYm.size());
        const int cntX = static_cast<int>(sendXm.size());
        if (cntY > 0) {
            for (long lz = 0; lz < d.lnz; ++lz) {
                if (nbYm != MPI_PROC_NULL) {
                    std::memcpy(data + d.at(1, 0, lz + 1), recvYm.data() + lz * d.lnx, sizeof(Real) * d.lnx);
                }
                if (nbYp != MPI_PROC_NULL) {
                    std::memcpy(data + d.at(1, d.lny + 1, lz + 1), recvYp.data() + lz * d.lnx, sizeof(Real) * d.lnx);
                }
            }
        }
        if (cntX > 0) {
            size_t k = 0;
            for (long lz = 1; lz <= d.lnz; ++lz) {
                for (long ly = 1; ly <= d.lny; ++ly, ++k) {
                    if (nbXm != MPI_PROC_NULL) data[d.at(0, ly, lz)] = recvXm[k];
                    if (nbXp != MPI_PROC_NULL) data[d.at(d.lnx + 1, ly, lz)] = recvXp[k];
                }
            }
        }
    }
};

// 7-point stencil computation on the owned interior cells of the local block.
// Global boundary cells are never written: both buffers hold the (constant) initial boundary
// values, which is equivalent to the serial code's explicit boundary copy.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const Domain& d,
                      const long xb, const long xe,
                      const long yb, const long ye,
                      const long zb, const long ze) {
    const Real* __restrict in = input.data();
    Real* __restrict out = output.data();
    const long sx = d.sx;
    const long plane = static_cast<long>(d.plane);

    for (long lz = zb; lz <= ze; ++lz) {
        for (long ly = yb; ly <= ye; ++ly) {
            const size_t base = d.at(0, ly, lz);
            const Real* __restrict c = in + base;
            Real* __restrict o = out + base;
            for (long lx = xb; lx <= xe; ++lx) {
                const Real center = c[lx];
                const Real left = c[lx - 1];
                const Real right = c[lx + 1];
                const Real front = c[lx - sx];
                const Real back = c[lx + sx];
                const Real bottom = c[lx - plane];
                const Real top = c[lx + plane];

                // Simple averaging stencil
                o[lx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, const Domain& d, MPI_Comm comm, const int rank) {
    // Simple sanity checks (distributed over all ranks, identical semantics to the serial version)

    // 1. No NaN or Inf values
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (long lz = 1; lz <= d.lnz; ++lz) {
        for (long ly = 1; ly <= d.lny; ++ly) {
            const Real* __restrict c = grid.data() + d.at(1, ly, lz);
            for (long lx = 0; lx < d.lnx; ++lx) {
                const Real val = c[lx];
                if (std::isnan(val) || std::isinf(val)) {
                    localBad = 1;
                } else {
                    localMin = std::min(localMin, val);
                    localMax = std::max(localMax, val);
                }
            }
        }
    }

    int globalBad = 0;
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_LOR, comm);
    if (globalBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = 0.0, maxVal = 0.0;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

// Collect the distributed grid into a single global array on rank 0 (only needed for -r output)
void gatherGrid(const std::vector<Real>& local, const Domain& d, MPI_Comm cart, const int rank, const int nranks,
                const size_t nx, const size_t ny, const size_t nz, std::vector<Real>& global) {
    // Pack the owned cells contiguously
    const size_t localCount = static_cast<size_t>(d.lnx) * static_cast<size_t>(d.lny) * static_cast<size_t>(d.lnz);
    std::vector<Real> packed(localCount);
    {
        size_t k = 0;
        for (long lz = 1; lz <= d.lnz; ++lz) {
            for (long ly = 1; ly <= d.lny; ++ly) {
                std::memcpy(packed.data() + k, local.data() + d.at(1, ly, lz), sizeof(Real) * d.lnx);
                k += static_cast<size_t>(d.lnx);
            }
        }
    }

    if (rank != 0) {
        MPI_Send(packed.data(), static_cast<int>(localCount), MPI_DOUBLE, 0, 7, cart);
        return;
    }

    global.assign(nx * ny * nz, Real{0});
    std::vector<Real> recv;

    auto unpack = [&](const Real* src, const long x0, const long y0, const long z0,
                      const long cx, const long cy, const long cz) {
        size_t k = 0;
        for (long z = 0; z < cz; ++z) {
            for (long y = 0; y < cy; ++y) {
                const size_t dst = idx3(static_cast<size_t>(x0), static_cast<size_t>(y0 + y), static_cast<size_t>(z0 + z), nx, ny);
                std::memcpy(global.data() + dst, src + k, sizeof(Real) * cx);
                k += static_cast<size_t>(cx);
            }
        }
    };

    int dims[3], periods[3], coords[3];
    MPI_Cart_get(cart, 3, dims, periods, coords);

    unpack(packed.data(), d.x0, d.y0, d.z0, d.lnx, d.lny, d.lnz);

    for (int r = 1; r < nranks; ++r) {
        int rc[3];
        MPI_Cart_coords(cart, r, 3, rc);
        long x0, y0, z0, cx, cy, cz;
        blockRange(static_cast<long>(nx), dims[2], rc[2], x0, cx);
        blockRange(static_cast<long>(ny), dims[1], rc[1], y0, cy);
        blockRange(static_cast<long>(nz), dims[0], rc[0], z0, cz);

        const size_t cnt = static_cast<size_t>(cx) * static_cast<size_t>(cy) * static_cast<size_t>(cz);
        recv.resize(cnt);
        MPI_Recv(recv.data(), static_cast<int>(cnt), MPI_DOUBLE, r, 7, cart, MPI_STATUS_IGNORE);
        unpack(recv.data(), x0, y0, z0, cx, cy, cz);
    }
}

// Pick a process grid dims[] = {Z, Y, X} whose extent per dimension does not exceed the grid
// extent (every rank must own at least one cell in each dimension) and which minimizes the
// halo surface of a local block. Returns the number of ranks actually used (<= P).
int chooseDims(const long P, const long nz, const long ny, const long nx, int dims[3]) {
    for (long p = P; p >= 1; --p) {
        double best = -1.0;
        for (long a = 1; a <= std::min(p, nz); ++a) {
            if (p % a != 0) continue;
            const long rest = p / a;
            for (long b = 1; b <= std::min(rest, ny); ++b) {
                if (rest % b != 0) continue;
                const long c = rest / b;
                if (c > nx) continue;
                const double lx = static_cast<double>(nx) / c;
                const double ly = static_cast<double>(ny) / b;
                const double lz = static_cast<double>(nz) / a;
                const double cost = lx * ly + lx * lz + ly * lz;  // halo surface of a local block
                if (best < 0.0 || cost < best) {
                    best = cost;
                    dims[0] = static_cast<int>(a);
                    dims[1] = static_cast<int>(b);
                    dims[2] = static_cast<int>(c);
                }
            }
        }
        if (best >= 0.0) return static_cast<int>(p);
    }
    dims[0] = dims[1] = dims[2] = 1;
    return 1;
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
    }

    // --- Build a 3D Cartesian process grid (dims[0]=Z, dims[1]=Y, dims[2]=X) ---
    int dims[3] = {0, 0, 0};
    const int active = chooseDims(nranks, static_cast<long>(nz), static_cast<long>(ny), static_cast<long>(nx), dims);

    // If the grid is too small to give every rank at least one cell per dimension, the surplus
    // ranks (the highest world ranks) stay idle so that the process grid stays well formed.
    MPI_Comm work = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &work);
    if (work == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(work, 3, dims, periods, /*reorder=*/0, &cart);
    MPI_Comm_free(&work);

    int cartRank = 0;
    MPI_Comm_rank(cart, &cartRank);
    int coords[3];
    MPI_Cart_coords(cart, cartRank, 3, coords);

    Domain d;
    blockRange(static_cast<long>(nx), dims[2], coords[2], d.x0, d.lnx);
    blockRange(static_cast<long>(ny), dims[1], coords[1], d.y0, d.lny);
    blockRange(static_cast<long>(nz), dims[0], coords[0], d.z0, d.lnz);
    d.sx = d.lnx + 2;
    d.sy = d.lny + 2;
    d.sz = d.lnz + 2;
    d.plane = static_cast<size_t>(d.sx) * static_cast<size_t>(d.sy);
    d.size = d.plane * static_cast<size_t>(d.sz);

    // Allocate local grids with ghost layers (double buffering)
    std::vector<Real> grid1(d.size, Real{0});
    std::vector<Real> grid2(d.size, Real{0});

    // Initialize; both buffers carry the (constant) global boundary values
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, d, nx, ny);
    grid2 = grid1;

    HaloExchange halo;
    halo.setup(cart, d);

    // Loop bounds over owned cells that are global interior cells (global index in [1, n-2])
    const long xb = std::max(1L, 2L - d.x0), xe = std::min(d.lnx, static_cast<long>(nx) - 1 - d.x0);
    const long yb = std::max(1L, 2L - d.y0), ye = std::min(d.lny, static_cast<long>(ny) - 1 - d.y0);
    const long zb = std::max(1L, 2L - d.z0), ze = std::min(d.lnz, static_cast<long>(nz) - 1 - d.z0);

    // Sub-box that does not depend on any ghost cell; it can be updated while the halo
    // exchange is still in flight. The remaining shell is updated once the halos arrived.
    const long xib = std::max(xb, 2L), xie = std::min(xe, d.lnx - 1);
    const long yib = std::max(yb, 2L), yie = std::min(ye, d.lny - 1);
    const long zib = std::max(zb, 2L), zie = std::min(ze, d.lnz - 1);
    const bool hasInner = (xib <= xie) && (yib <= yie) && (zib <= zie);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;

        halo.start(in, d);

        if (hasInner) {
            stencilIteration(in, out, d, xib, xie, yib, yie, zib, zie);
        }

        halo.finish(in, d);

        if (!hasInner) {
            stencilIteration(in, out, d, xb, xe, yb, ye, zb, ze);
        } else {
            // Shell around the already updated inner box (disjoint slabs, no cell computed twice)
            stencilIteration(in, out, d, xb, xe, yb, ye, zb, zib - 1);
            stencilIteration(in, out, d, xb, xe, yb, ye, zie + 1, ze);
            stencilIteration(in, out, d, xb, xe, yb, yib - 1, zib, zie);
            stencilIteration(in, out, d, xb, xe, yie + 1, ye, zib, zie);
            stencilIteration(in, out, d, xb, xib - 1, yib, yie, zib, zie);
            stencilIteration(in, out, d, xie + 1, xe, yib, yie, zib, zie);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Allreduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, cart);

    const long durationMs = static_cast<long>(elapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> global;
        gatherGrid(finalGrid, d, cart, cartRank, active, nx, ny, nz, global);
        if (cartRank == 0) print_results(global, "Grid");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(finalGrid, d, cart, rank);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return ret;
}
