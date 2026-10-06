#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Balanced block distribution of n points over p parts: start of part c
static inline long blockStart(const long n, const int p, const int c) {
    return (n / p) * c + std::min<long>(c, n % p);
}

// Local subdomain of the global grid, stored with a one-cell halo on every side.
// Local array dimensions: (lx+2) x (ly+2) x (lz+2), x fastest.
struct Domain {
    long nx, ny, nz;         // global sizes
    int px, py, pz;          // process grid
    int cx, cy, cz;          // coordinates of this rank
    long ox, oy, oz;         // global offset of first owned cell
    long lx, ly, lz;         // owned cell counts
    long sx, sy;             // local strides (sx = lx+2, sy = (lx+2)*(ly+2))
    // Range (local, inclusive) of owned cells that are global interior points
    long xlo, xhi, ylo, yhi, zlo, zhi;
    MPI_Comm cart;
    int nbr[3][2];           // neighbor ranks [dim x,y,z][low,high]

    size_t localSize() const { return (size_t)sy * (lz + 2); }
    size_t at(long i, long j, long k) const { return (size_t)k * sy + (size_t)j * sx + (size_t)i; }
};

// Choose a process grid that minimizes halo communication volume.
// x-direction faces are strided (more expensive) and x-splits hurt vectorization, so they are penalized.
static void chooseProcessGrid(const int size, const long nx, const long ny, const long nz, int& px, int& py, int& pz) {
    double best = std::numeric_limits<double>::max();
    px = py = pz = -1;
    for (int a = 1; a <= size; ++a) {
        if (size % a) continue;
        for (int b = 1; b <= size / a; ++b) {
            if ((size / a) % b) continue;
            const int c = size / a / b;
            if (a > nx || b > ny || c > nz) continue;
            const double lx = (double)nx / a, ly = (double)ny / b, lz = (double)nz / c;
            double cost = 0.0;
            if (a > 1) cost += 4.0 * ly * lz;   // strided faces, penalized
            if (b > 1) cost += 2.0 * lx * lz * 1.2;
            if (c > 1) cost += 2.0 * lx * ly;
            if (cost < best) {
                best = cost;
                px = a; py = b; pz = c;
            }
        }
    }
}

static void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    for (long k = 0; k < d.lz; ++k) {
        for (long j = 0; j < d.ly; ++j) {
            const size_t gbase = idx3((size_t)d.ox, (size_t)(d.oy + j), (size_t)(d.oz + k), (size_t)d.nx, (size_t)d.ny);
            Real* row = grid.data() + d.at(1, j + 1, k + 1);
            for (long i = 0; i < d.lx; ++i) {
                row[i] = ((gbase + i) % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on a local box (inclusive bounds, local coordinates)
static inline void stencilBox(const Real* __restrict in, Real* __restrict out, const Domain& d,
                              const long x0, const long x1, const long y0, const long y1, const long z0, const long z1) {
    if (x0 > x1 || y0 > y1 || z0 > z1) return;
    const long sx = d.sx, sy = d.sy;
    for (long k = z0; k <= z1; ++k) {
        for (long j = y0; j <= y1; ++j) {
            const size_t base = d.at(0, j, k);
            const Real* __restrict c = in + base;
            const Real* __restrict f = c - sx;
            const Real* __restrict b = c + sx;
            const Real* __restrict lo = c - sy;
            const Real* __restrict hi = c + sy;
            Real* __restrict o = out + base;
#pragma GCC ivdep
            for (long i = x0; i <= x1; ++i) {
                // Simple averaging stencil (same summation order as the reference)
                o[i] = (c[i] + c[i - 1] + c[i + 1] + f[i] + b[i] + lo[i] + hi[i]) / 7.0;
            }
        }
    }
}

// Persistent halo exchange requests for one buffer
struct Halo {
    std::vector<MPI_Request> reqs;
};

static void buildHalo(Halo& h, std::vector<Real>& buf, const Domain& d, MPI_Datatype faces[3]) {
    const int tagBase = 100;
    for (int dim = 0; dim < 3; ++dim) {
        const long l = (dim == 0) ? d.lx : (dim == 1) ? d.ly : d.lz;
        for (int side = 0; side < 2; ++side) {
            const int peer = d.nbr[dim][side];
            if (peer == MPI_PROC_NULL) continue;
            const long sendPos = (side == 0) ? 1 : l;
            const long recvPos = (side == 0) ? 0 : l + 1;
            size_t so, ro;
            if (dim == 0) { so = d.at(sendPos, 1, 1); ro = d.at(recvPos, 1, 1); }
            else if (dim == 1) { so = d.at(1, sendPos, 1); ro = d.at(1, recvPos, 1); }
            else { so = d.at(1, 1, sendPos); ro = d.at(1, 1, recvPos); }
            MPI_Request r;
            // Message tag identifies direction of travel: data sent to the low side carries tag (dim,0)
            MPI_Recv_init(buf.data() + ro, 1, faces[dim], peer, tagBase + dim * 2 + (1 - side), d.cart, &r);
            h.reqs.push_back(r);
            MPI_Send_init(buf.data() + so, 1, faces[dim], peer, tagBase + dim * 2 + side, d.cart, &r);
            h.reqs.push_back(r);
        }
    }
}

// One stencil iteration with communication/computation overlap
static void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output, const Domain& d, Halo& halo) {
    if (!halo.reqs.empty()) MPI_Startall((int)halo.reqs.size(), halo.reqs.data());

    const Real* in = input.data();
    Real* out = output.data();

    // Inner region: cells whose neighbors are all owned locally
    const long ix0 = std::max(d.xlo, 2L), ix1 = std::min(d.xhi, d.lx - 1);
    const long iy0 = std::max(d.ylo, 2L), iy1 = std::min(d.yhi, d.ly - 1);
    const long iz0 = std::max(d.zlo, 2L), iz1 = std::min(d.zhi, d.lz - 1);
    stencilBox(in, out, d, ix0, ix1, iy0, iy1, iz0, iz1);

    if (!halo.reqs.empty()) MPI_Waitall((int)halo.reqs.size(), halo.reqs.data(), MPI_STATUSES_IGNORE);

    // Shell: remaining cells adjacent to the halo
    // z planes
    stencilBox(in, out, d, d.xlo, d.xhi, d.ylo, d.yhi, d.zlo, std::min(iz0 - 1, d.zhi));
    stencilBox(in, out, d, d.xlo, d.xhi, d.ylo, d.yhi, std::max(iz1 + 1, iz0), d.zhi);
    if (iz0 > iz1) return;
    // y rows within inner z
    stencilBox(in, out, d, d.xlo, d.xhi, d.ylo, std::min(iy0 - 1, d.yhi), iz0, iz1);
    stencilBox(in, out, d, d.xlo, d.xhi, std::max(iy1 + 1, iy0), d.yhi, iz0, iz1);
    if (iy0 > iy1) return;
    // x columns within inner y,z
    stencilBox(in, out, d, d.xlo, std::min(ix0 - 1, d.xhi), iy0, iy1, iz0, iz1);
    stencilBox(in, out, d, std::max(ix1 + 1, ix0), d.xhi, iy0, iy1, iz0, iz1);
    // Global boundary cells are never written: both buffers hold the initial values,
    // which is equivalent to copying boundary values every iteration.
}

// Gather the full grid on rank 0 (returns empty vector on other ranks)
static std::vector<Real> gatherGrid(const std::vector<Real>& local, const Domain& d, const int rank, const int size) {
    std::vector<Real> packed((size_t)d.lx * d.ly * d.lz);
    size_t p = 0;
    for (long k = 1; k <= d.lz; ++k)
        for (long j = 1; j <= d.ly; ++j) {
            const Real* row = local.data() + d.at(1, j, k);
            std::copy(row, row + d.lx, packed.data() + p);
            p += d.lx;
        }

    std::vector<Real> full;
    if (rank != 0) {
        MPI_Send(packed.data(), (int)packed.size(), MPI_DOUBLE, 0, 200, d.cart);
        return full;
    }

    full.resize((size_t)d.nx * d.ny * d.nz);
    std::vector<Real> recv;
    for (int r = 0; r < size; ++r) {
        int c[3];  // cartesian coords in (z, y, x) order
        MPI_Cart_coords(d.cart, r, 3, c);
        const long ox = blockStart(d.nx, d.px, c[2]), lx = blockStart(d.nx, d.px, c[2] + 1) - ox;
        const long oy = blockStart(d.ny, d.py, c[1]), ly = blockStart(d.ny, d.py, c[1] + 1) - oy;
        const long oz = blockStart(d.nz, d.pz, c[0]), lz = blockStart(d.nz, d.pz, c[0] + 1) - oz;
        const Real* src = packed.data();
        if (r != 0) {
            recv.resize((size_t)lx * ly * lz);
            MPI_Recv(recv.data(), (int)recv.size(), MPI_DOUBLE, r, 200, d.cart, MPI_STATUS_IGNORE);
            src = recv.data();
        }
        for (long k = 0; k < lz; ++k)
            for (long j = 0; j < ly; ++j) {
                std::copy(src, src + lx, full.data() + idx3(ox, oy + j, oz + k, d.nx, d.ny));
                src += lx;
            }
    }
    return full;
}

// Distributed validation; result is identical on all ranks
bool validateResult(const std::vector<Real>& grid, const Domain& d, const int rank) {
    // Simple sanity checks
    int bad = 0;
    Real mm[2] = {std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity()}; // {min, -max}
    for (long k = 1; k <= d.lz; ++k)
        for (long j = 1; j <= d.ly; ++j) {
            const Real* row = grid.data() + d.at(1, j, k);
            for (long i = 0; i < d.lx; ++i) {
                const Real val = row[i];
                // 1. No NaN or Inf values
                if (std::isnan(val) || std::isinf(val)) bad = 1;
                mm[0] = std::min(mm[0], val);
                mm[1] = std::min(mm[1], -val);
            }
        }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, d.cart);
    if (bad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_DOUBLE, MPI_MIN, d.cart);
    const Real minVal = mm[0];
    const Real maxVal = -mm[1];

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

static int run(int argc, char** argv, const int rank, const int size) {
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

    // Domain decomposition
    Domain d;
    d.nx = (long)nx; d.ny = (long)ny; d.nz = (long)nz;
    chooseProcessGrid(size, d.nx, d.ny, d.nz, d.px, d.py, d.pz);
    if (d.px < 0) {
        if (rank == 0) printf("Error: cannot decompose %zu x %zu x %zu grid over %d processes\n", nx, ny, nz, size);
        return 1;
    }
    {
        // MPI dims order: [z, y, x] so that rank order follows memory order
        int dims[3] = {d.pz, d.py, d.px};
        int periods[3] = {0, 0, 0};
        MPI_Comm c;
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &c);
        d.cart = c;
        int coords[3];
        MPI_Cart_coords(d.cart, rank, 3, coords);
        d.cz = coords[0]; d.cy = coords[1]; d.cx = coords[2];
        MPI_Cart_shift(d.cart, 2, 1, &d.nbr[0][0], &d.nbr[0][1]);
        MPI_Cart_shift(d.cart, 1, 1, &d.nbr[1][0], &d.nbr[1][1]);
        MPI_Cart_shift(d.cart, 0, 1, &d.nbr[2][0], &d.nbr[2][1]);
    }
    d.ox = blockStart(d.nx, d.px, d.cx); d.lx = blockStart(d.nx, d.px, d.cx + 1) - d.ox;
    d.oy = blockStart(d.ny, d.py, d.cy); d.ly = blockStart(d.ny, d.py, d.cy + 1) - d.oy;
    d.oz = blockStart(d.nz, d.pz, d.cz); d.lz = blockStart(d.nz, d.pz, d.cz + 1) - d.oz;
    d.sx = d.lx + 2;
    d.sy = d.sx * (d.ly + 2);
    // Local range of global interior points: global g in [1, n-2], local i = g - o + 1
    d.xlo = std::max(1L, 2 - d.ox); d.xhi = std::min(d.lx, d.nx - 1 - d.ox);
    d.ylo = std::max(1L, 2 - d.oy); d.yhi = std::min(d.ly, d.ny - 1 - d.oy);
    d.zlo = std::max(1L, 2 - d.oz); d.zhi = std::min(d.lz, d.nz - 1 - d.oz);

    // Face datatypes for halo exchange (owned extent only; no edges/corners needed for a 7-point stencil)
    MPI_Datatype faces[3];
    {
        const int sizes[3] = {(int)(d.lz + 2), (int)(d.ly + 2), (int)(d.lx + 2)};
        const int start[3] = {0, 0, 0};
        const int sub[3][3] = {{(int)d.lz, (int)d.ly, 1}, {(int)d.lz, 1, (int)d.lx}, {1, (int)d.ly, (int)d.lx}};
        for (int i = 0; i < 3; ++i) {
            MPI_Type_create_subarray(3, sizes, sub[i], start, MPI_ORDER_C, MPI_DOUBLE, &faces[i]);
            MPI_Type_commit(&faces[i]);
        }
    }

    // Allocate grids (double buffering)
    std::vector<Real> grid1(d.localSize(), 0.0);
    std::vector<Real> grid2(d.localSize(), 0.0);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, d);
    grid2 = grid1;  // boundary values are identical in both buffers

    Halo halo1, halo2;
    buildHalo(halo1, grid1, d, faces);
    buildHalo(halo2, grid2, d, faces);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(d.cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, d, halo1);
        } else {
            stencilIteration(grid2, grid1, d, halo2);
        }
    }

    MPI_Barrier(d.cart);
    auto end = std::chrono::high_resolution_clock::now();
    long ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Bcast(&ms, 1, MPI_LONG, 0, d.cart);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", ms);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> full = gatherGrid(finalGrid, d, rank, size);
        if (rank == 0) print_results(full, "Grid");
    }

    int ret = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(finalGrid, d, rank);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    for (auto& r : halo1.reqs) MPI_Request_free(&r);
    for (auto& r : halo2.reqs) MPI_Request_free(&r);
    for (auto& t : faces) MPI_Type_free(&t);
    MPI_Comm_free(&d.cart);
    return ret;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ret = run(argc, argv, rank, size);

    MPI_Finalize();
    return ret;
}
