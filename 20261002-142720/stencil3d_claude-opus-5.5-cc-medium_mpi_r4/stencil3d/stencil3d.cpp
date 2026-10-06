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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local subdomain of the global grid owned by one MPI rank (3D block decomposition).
// The local array stores the owned block plus a one-cell halo on every side.
struct Domain {
    size_t nx, ny, nz;              // global sizes
    size_t lx, ly, lz;              // owned block sizes
    size_t ox, oy, oz;              // global offset of owned block
    size_t sx, sy;                  // local array strides (lx+2, (lx+2)*(ly+2))
    size_t localSize;
    int nbrLo[3], nbrHi[3];         // neighbors per dimension (0=x,1=y,2=z)
    MPI_Datatype sendLo[3], sendHi[3], recvLo[3], recvHi[3];
    MPI_Comm comm;

    inline size_t lidx(size_t x, size_t y, size_t z) const noexcept { return z * sy + y * sx + x; }
};

static void blockRange(size_t n, int p, int c, size_t& off, size_t& len) {
    const size_t base = n / p, rem = n % p;
    off = c * base + std::min<size_t>(c, rem);
    len = base + (static_cast<size_t>(c) < rem ? 1 : 0);
}

// Choose a process grid (px,py,pz) with px*py*pz == P that minimizes halo traffic.
// Returns false if no factorization gives every rank at least one cell per dimension.
static bool chooseDims(int P, size_t nx, size_t ny, size_t nz, int dims[3]) {
    double best = -1.0;
    for (int px = 1; px <= P; ++px) {
        if (P % px) continue;
        for (int py = 1; py <= P / px; ++py) {
            if ((P / px) % py) continue;
            const int pz = P / px / py;
            if ((size_t)px > nx || (size_t)py > ny || (size_t)pz > nz) continue;
            // Per-rank exchanged face area; x-faces are strided, so weight them a bit more.
            const double bx = (double)nx / px, by = (double)ny / py, bz = (double)nz / pz;
            const double cost = (px > 1 ? 1.5 * 2.0 * by * bz : 0.0)
                              + (py > 1 ? 2.0 * bx * bz : 0.0)
                              + (pz > 1 ? 2.0 * bx * by : 0.0);
            if (best < 0.0 || cost < best - 1e-9) {
                best = cost;
                dims[0] = px; dims[1] = py; dims[2] = pz;
            }
        }
    }
    return best >= 0.0;
}

static void setupDomain(Domain& d, MPI_Comm comm, const int dims[3]) {
    int periods[3] = {0, 0, 0};
    int mpiDims[3] = {dims[2], dims[1], dims[0]};  // MPI order: z, y, x
    MPI_Cart_create(comm, 3, mpiDims, periods, 0, &d.comm);  // no reorder: rank 0 stays the output rank
    int rank;
    MPI_Comm_rank(d.comm, &rank);
    int coords[3];
    MPI_Cart_coords(d.comm, rank, 3, coords);
    blockRange(d.nz, dims[2], coords[0], d.oz, d.lz);
    blockRange(d.ny, dims[1], coords[1], d.oy, d.ly);
    blockRange(d.nx, dims[0], coords[2], d.ox, d.lx);
    MPI_Cart_shift(d.comm, 2, 1, &d.nbrLo[0], &d.nbrHi[0]);
    MPI_Cart_shift(d.comm, 1, 1, &d.nbrLo[1], &d.nbrHi[1]);
    MPI_Cart_shift(d.comm, 0, 1, &d.nbrLo[2], &d.nbrHi[2]);

    d.sx = d.lx + 2;
    d.sy = d.sx * (d.ly + 2);
    d.localSize = d.sy * (d.lz + 2);

    // Face subarray types, in MPI (z,y,x) order
    const int sizes[3] = {(int)d.lz + 2, (int)d.ly + 2, (int)d.lx + 2};
    const int l[3] = {(int)d.lx, (int)d.ly, (int)d.lz};
    for (int dim = 0; dim < 3; ++dim) {
        const int m = 2 - dim;
        int sub[3] = {l[2], l[1], l[0]};
        sub[m] = 1;
        int st[3] = {1, 1, 1};
        auto make = [&](int pos, MPI_Datatype& t) {
            st[m] = pos;
            MPI_Type_create_subarray(3, sizes, sub, st, MPI_ORDER_C, MPI_DOUBLE, &t);
            MPI_Type_commit(&t);
        };
        make(1, d.sendLo[dim]);
        make(l[dim], d.sendHi[dim]);
        make(0, d.recvLo[dim]);
        make(l[dim] + 1, d.recvHi[dim]);
    }
}

static void freeDomain(Domain& d) {
    for (int dim = 0; dim < 3; ++dim) {
        MPI_Type_free(&d.sendLo[dim]);
        MPI_Type_free(&d.sendHi[dim]);
        MPI_Type_free(&d.recvLo[dim]);
        MPI_Type_free(&d.recvHi[dim]);
    }
    MPI_Comm_free(&d.comm);
}

void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    // Initialize owned cells and in-domain halo cells from the global formula
    for (size_t lz = 0; lz < d.lz + 2; ++lz) {
        const long gz = (long)d.oz + (long)lz - 1;
        if (gz < 0 || gz >= (long)d.nz) continue;
        for (size_t ly = 0; ly < d.ly + 2; ++ly) {
            const long gy = (long)d.oy + (long)ly - 1;
            if (gy < 0 || gy >= (long)d.ny) continue;
            for (size_t lx = 0; lx < d.lx + 2; ++lx) {
                const long gx = (long)d.ox + (long)lx - 1;
                if (gx < 0 || gx >= (long)d.nx) continue;
                const size_t idx = idx3(gx, gy, gz, d.nx, d.ny);
                grid[d.lidx(lx, ly, lz)] = (idx % 19) * 1.0;
            }
        }
    }
}

struct Box { size_t x0, x1, y0, y1, z0, z1; };  // half-open local index ranges

// 7-point stencil computation on a local box
static void stencilBox(const Real* __restrict in, Real* __restrict out, const Domain& d, const Box& b) {
    if (b.x0 >= b.x1 || b.y0 >= b.y1 || b.z0 >= b.z1) return;
    const size_t sx = d.sx, sy = d.sy;
    for (size_t z = b.z0; z < b.z1; ++z) {
        for (size_t y = b.y0; y < b.y1; ++y) {
            const size_t row = z * sy + y * sx;
            const Real* __restrict c = in + row;
            const Real* __restrict f = c - sx;
            const Real* __restrict k = c + sx;
            const Real* __restrict bo = c - sy;
            const Real* __restrict t = c + sy;
            Real* __restrict o = out + row;
#pragma GCC ivdep
            for (size_t x = b.x0; x < b.x1; ++x) {
                // Simple averaging stencil (same operation order as the original)
                o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + k[x] + bo[x] + t[x]) / 7.0;
            }
        }
    }
}

// Local index range [lo,hi) of globally interior cells within the owned block
static void interiorRange(size_t n, size_t off, size_t len, size_t& lo, size_t& hi) {
    const size_t g0 = std::max<size_t>(off, 1);
    const size_t g1 = std::min<size_t>(off + len, n >= 1 ? n - 1 : 0);
    if (g1 <= g0) { lo = hi = 1; return; }
    lo = g0 - off + 1;
    hi = g1 - off + 1;
}

// One stencil iteration with the halo exchange overlapped with the inner computation.
// Global boundary cells are never written: both buffers start with identical boundary
// values, which is equivalent to copying them every iteration.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output, const Domain& d, const Box& upd) {
    MPI_Request reqs[12];
    int nreq = 0;
    Real* inp = input.data();
    for (int dim = 0; dim < 3; ++dim) {
        MPI_Irecv(inp, 1, d.recvLo[dim], d.nbrLo[dim], 2 * dim, d.comm, &reqs[nreq++]);
        MPI_Irecv(inp, 1, d.recvHi[dim], d.nbrHi[dim], 2 * dim + 1, d.comm, &reqs[nreq++]);
    }
    for (int dim = 0; dim < 3; ++dim) {
        MPI_Isend(inp, 1, d.sendHi[dim], d.nbrHi[dim], 2 * dim, d.comm, &reqs[nreq++]);
        MPI_Isend(inp, 1, d.sendLo[dim], d.nbrLo[dim], 2 * dim + 1, d.comm, &reqs[nreq++]);
    }

    // Inner box: cells that do not depend on neighbor halos
    Box in = upd;
    if (d.nbrLo[0] != MPI_PROC_NULL) in.x0 = std::max<size_t>(in.x0, 2);
    if (d.nbrHi[0] != MPI_PROC_NULL) in.x1 = std::min<size_t>(in.x1, d.lx);
    if (d.nbrLo[1] != MPI_PROC_NULL) in.y0 = std::max<size_t>(in.y0, 2);
    if (d.nbrHi[1] != MPI_PROC_NULL) in.y1 = std::min<size_t>(in.y1, d.ly);
    if (d.nbrLo[2] != MPI_PROC_NULL) in.z0 = std::max<size_t>(in.z0, 2);
    if (d.nbrHi[2] != MPI_PROC_NULL) in.z1 = std::min<size_t>(in.z1, d.lz);
    const bool innerValid = in.x0 < in.x1 && in.y0 < in.y1 && in.z0 < in.z1;

    Real* out = output.data();
    if (innerValid) stencilBox(inp, out, d, in);

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (!innerValid) {
        stencilBox(inp, out, d, upd);
        return;
    }
    // Shell = upd \ in
    stencilBox(inp, out, d, {upd.x0, upd.x1, upd.y0, upd.y1, upd.z0, in.z0});
    stencilBox(inp, out, d, {upd.x0, upd.x1, upd.y0, upd.y1, in.z1, upd.z1});
    stencilBox(inp, out, d, {upd.x0, upd.x1, upd.y0, in.y0, in.z0, in.z1});
    stencilBox(inp, out, d, {upd.x0, upd.x1, in.y1, upd.y1, in.z0, in.z1});
    stencilBox(inp, out, d, {upd.x0, in.x0, in.y0, in.y1, in.z0, in.z1});
    stencilBox(inp, out, d, {in.x1, upd.x1, in.y0, in.y1, in.z0, in.z1});
}

// Gather the distributed grid into a global array on rank 0
static void gatherGrid(const std::vector<Real>& local, const Domain& d, std::vector<Real>& global) {
    int rank, size;
    MPI_Comm_rank(d.comm, &rank);
    MPI_Comm_size(d.comm, &size);
    std::vector<Real> packed(d.lx * d.ly * d.lz);
    size_t p = 0;
    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            const Real* src = local.data() + d.lidx(1, y, z);
            std::copy(src, src + d.lx, packed.data() + p);
            p += d.lx;
        }
    }
    unsigned long long meta[6] = {d.ox, d.oy, d.oz, d.lx, d.ly, d.lz};
    std::vector<unsigned long long> allMeta(rank == 0 ? 6 * size : 0);
    MPI_Gather(meta, 6, MPI_UNSIGNED_LONG_LONG, allMeta.data(), 6, MPI_UNSIGNED_LONG_LONG, 0, d.comm);

    if (rank != 0) {
        MPI_Send(packed.data(), (int)packed.size(), MPI_DOUBLE, 0, 100, d.comm);
        return;
    }
    // Receive blocks one at a time (avoids int overflow of Gatherv displacements on huge grids)
    std::vector<Real> buf;
    for (int r = 0; r < size; ++r) {
        const unsigned long long* m = &allMeta[6 * r];
        const Real* src;
        if (r == 0) {
            src = packed.data();
        } else {
            buf.resize(m[3] * m[4] * m[5]);
            MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 100, d.comm, MPI_STATUS_IGNORE);
            src = buf.data();
        }
        for (size_t z = 0; z < m[5]; ++z) {
            for (size_t y = 0; y < m[4]; ++y) {
                std::copy(src, src + m[3], global.data() + idx3(m[0], m[1] + y, m[2] + z, d.nx, d.ny));
                src += m[3];
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t gridSize = nx * ny * nz;

    // Use the largest number of ranks that admits a decomposition with non-empty blocks
    int dims[3] = {1, 1, 1};
    int active = worldSize;
    while (active > 1 && !chooseDims(active, nx, ny, nz, dims)) --active;
    if (active == 1) { dims[0] = dims[1] = dims[2] = 1; }
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);

    int exitCode = 0;
    if (activeComm != MPI_COMM_NULL) {
        Domain d;
        d.nx = nx; d.ny = ny; d.nz = nz;
        setupDomain(d, activeComm, dims);

        // Allocate local grids with halos (double buffering)
        std::vector<Real> grid1(d.localSize, 0.0);

        // Initialize
        if (root) printf("Initializing grid...\n");
        initializeGrid(grid1, d);
        std::vector<Real> grid2(grid1);  // boundary values stay identical in both buffers

        Box upd;
        interiorRange(nx, d.ox, d.lx, upd.x0, upd.x1);
        interiorRange(ny, d.oy, d.ly, upd.y0, upd.y1);
        interiorRange(nz, d.oz, d.lz, upd.z0, upd.z1);

        // Run stencil iterations
        if (root) printf("Running stencil computation...\n");
        MPI_Barrier(d.comm);
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, d, upd);
            } else {
                stencilIteration(grid2, grid1, d, upd);
            }
        }

        MPI_Barrier(d.comm);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (root) {
            printf("Computation time: %ld ms\n", duration.count());

            // Calculate performance metrics
            double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
            double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults || validate) {
            const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;
            std::vector<Real> finalGrid(root ? gridSize : 0);
            gatherGrid(localFinal, d, finalGrid);

            if (root) {
                // Print results for external validation
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
        }
        freeDomain(d);
        MPI_Comm_free(&activeComm);
    }

    fflush(stdout);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
