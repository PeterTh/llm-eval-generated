#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Block distribution of n cells over p parts: start of part i
inline size_t blockStart(const size_t n, const int p, const int i) noexcept {
    return (n / p) * i + std::min<size_t>(i, n % p);
}
inline size_t blockSize(const size_t n, const int p, const int i) noexcept {
    return n / p + ((size_t)i < n % p ? 1 : 0);
}

// Local subdomain with one ghost layer on each side
struct Domain {
    size_t nx, ny, nz;          // global sizes
    int px, py, pz;             // process grid
    int cx, cy, cz;             // coordinates in process grid
    size_t ox, oy, oz;          // global offset of first owned cell
    size_t lx, ly, lz;          // owned sizes
    size_t sx, sy, sz;          // allocated sizes (owned + 2 ghosts)
    // update range in local coordinates (inclusive), empty if lo > hi
    long ux0, ux1, uy0, uy1, uz0, uz1;

    size_t at(const size_t x, const size_t y, const size_t z) const noexcept {
        return z * (sx * sy) + y * sx + x;
    }
};

void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            for (size_t x = 1; x <= d.lx; ++x) {
                const size_t gidx = idx3(d.ox + x - 1, d.oy + y - 1, d.oz + z - 1, d.nx, d.ny);
                grid[d.at(x, y, z)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on box [x0,x1]x[y0,y1]x[z0,z1] (local coords, inclusive)
static void stencilBox(const Real* __restrict in, Real* __restrict out, const Domain& d,
                       long x0, long x1, long y0, long y1, long z0, long z1) {
    if (x0 > x1 || y0 > y1 || z0 > z1) return;
    const size_t sx = d.sx;
    const size_t sxy = d.sx * d.sy;
    for (long z = z0; z <= z1; ++z) {
        for (long y = y0; y <= y1; ++y) {
            const size_t base = d.at(0, y, z);
            const Real* __restrict c = in + base;
            const Real* __restrict f = c - sx;
            const Real* __restrict b = c + sx;
            const Real* __restrict bo = c - sxy;
            const Real* __restrict t = c + sxy;
            Real* __restrict o = out + base;
#pragma GCC ivdep
            for (long x = x0; x <= x1; ++x) {
                // Simple averaging stencil (same operation order as the original)
                o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + b[x] + bo[x] + t[x]) / 7.0;
            }
        }
    }
}

// Shell of box U minus inner box I (I assumed contained in U and non-empty)
static void stencilShell(const Real* in, Real* out, const Domain& d,
                         long ax0, long ax1, long ay0, long ay1, long az0, long az1) {
    stencilBox(in, out, d, d.ux0, d.ux1, d.uy0, d.uy1, d.uz0, az0 - 1);
    stencilBox(in, out, d, d.ux0, d.ux1, d.uy0, d.uy1, az1 + 1, d.uz1);
    stencilBox(in, out, d, d.ux0, d.ux1, d.uy0, ay0 - 1, az0, az1);
    stencilBox(in, out, d, d.ux0, d.ux1, ay1 + 1, d.uy1, az0, az1);
    stencilBox(in, out, d, d.ux0, ax0 - 1, ay0, ay1, az0, az1);
    stencilBox(in, out, d, ax1 + 1, d.ux1, ay0, ay1, az0, az1);
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

// Choose a process grid (px,py,pz) using as many ranks as possible (each rank
// must own at least one cell per dimension) while minimizing halo traffic.
static void chooseProcessGrid(const int nprocs, const size_t nx, const size_t ny, const size_t nz,
                              int& bpx, int& bpy, int& bpz) {
    bpx = bpy = bpz = 1;
    long bestUsed = 1;
    double bestCost = 1e300;
    for (int pz = 1; pz <= nprocs; ++pz) {
        if ((size_t)pz > nz) break;
        for (int py = 1; pz * py <= nprocs; ++py) {
            if ((size_t)py > ny) break;
            for (int px = 1; pz * py * px <= nprocs; ++px) {
                if ((size_t)px > nx) break;
                const long used = (long)px * py * pz;
                const double lx = std::ceil((double)nx / px);
                const double ly = std::ceil((double)ny / py);
                const double lz = std::ceil((double)nz / pz);
                // x-faces are strided (one value per row) -> weight them higher
                double cost = 0.0;
                if (pz > 1) cost += 2.0 * lx * ly;
                if (py > 1) cost += 2.0 * lx * lz * 1.2;
                if (px > 1) cost += 2.0 * ly * lz * 4.0;
                if (used > bestUsed || (used == bestUsed && cost < bestCost)) {
                    bestUsed = used;
                    bestCost = cost;
                    bpx = px; bpy = py; bpz = pz;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
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
    
    // Process grid; surplus ranks (more ranks than cells) stay idle
    int px, py, pz;
    chooseProcessGrid(worldSize, std::max<size_t>(nx, 1), std::max<size_t>(ny, 1), std::max<size_t>(nz, 1), px, py, pz);
    const int used = px * py * pz;
    const bool active = worldRank < used;

    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);

    int exitCode = 0;
    if (active) {
        int dims[3] = {pz, py, px};
        int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(activeComm, 3, dims, periods, 0, &cart);
        int rank;
        MPI_Comm_rank(cart, &rank);
        int coords[3];
        MPI_Cart_coords(cart, rank, 3, coords);

        Domain d;
        d.nx = nx; d.ny = ny; d.nz = nz;
        d.px = px; d.py = py; d.pz = pz;
        d.cz = coords[0]; d.cy = coords[1]; d.cx = coords[2];
        d.ox = blockStart(nx, px, d.cx); d.lx = blockSize(nx, px, d.cx);
        d.oy = blockStart(ny, py, d.cy); d.ly = blockSize(ny, py, d.cy);
        d.oz = blockStart(nz, pz, d.cz); d.lz = blockSize(nz, pz, d.cz);
        d.sx = d.lx + 2; d.sy = d.ly + 2; d.sz = d.lz + 2;
        // Global interior is [1, n-2]; local coord = global - o + 1
        auto range = [](size_t n, size_t o, size_t l, long& lo, long& hi) {
            const long glo = std::max<long>(1, (long)o);
            const long ghi = std::min<long>((long)n - 2, (long)(o + l) - 1);
            lo = glo - (long)o + 1;
            hi = ghi - (long)o + 1;
        };
        range(nx, d.ox, d.lx, d.ux0, d.ux1);
        range(ny, d.oy, d.ly, d.uy0, d.uy1);
        range(nz, d.oz, d.lz, d.uz0, d.uz1);

        const size_t localSize = d.sx * d.sy * d.sz;

        // Allocate grids (double buffering). Global boundary cells never change
        // (they are copied each iteration), so both buffers start identical.
        std::vector<Real> grid1(localSize, 0.0);
        initializeGrid(grid1, d);
        std::vector<Real> grid2(grid1);

        // Neighbours and halo datatypes (faces only; 7-point stencil needs no edges/corners)
        int nbLo[3], nbHi[3];
        for (int dim = 0; dim < 3; ++dim) MPI_Cart_shift(cart, dim, 1, &nbLo[dim], &nbHi[dim]);

        const int sizes[3] = {(int)d.sz, (int)d.sy, (int)d.sx};
        const int sub[3][3] = {
            {1, (int)d.ly, (int)d.lx},   // z-face
            {(int)d.lz, 1, (int)d.lx},   // y-face
            {(int)d.lz, (int)d.ly, 1},   // x-face
        };
        const int lens[3] = {(int)d.lz, (int)d.ly, (int)d.lx};
        MPI_Datatype sendLo[3], sendHi[3], recvLo[3], recvHi[3];
        for (int dim = 0; dim < 3; ++dim) {
            int st[3] = {1, 1, 1};
            auto mk = [&](int pos, MPI_Datatype& t) {
                st[dim] = pos;
                MPI_Type_create_subarray(3, sizes, sub[dim], st, MPI_ORDER_C, MPI_DOUBLE, &t);
                MPI_Type_commit(&t);
            };
            mk(0, recvLo[dim]);
            mk(1, sendLo[dim]);
            mk(lens[dim], sendHi[dim]);
            mk(lens[dim] + 1, recvHi[dim]);
        }

        // Persistent requests for both buffer parities
        std::vector<MPI_Request> reqs[2];
        Real* bufs[2] = {grid1.data(), grid2.data()};
        for (int p = 0; p < 2; ++p) {
            Real* buf = bufs[p];
            for (int dim = 0; dim < 3; ++dim) {
                MPI_Request r;
                if (nbLo[dim] != MPI_PROC_NULL) {
                    MPI_Recv_init(buf, 1, recvLo[dim], nbLo[dim], 2 * dim + 1, cart, &r); reqs[p].push_back(r);
                    MPI_Send_init(buf, 1, sendLo[dim], nbLo[dim], 2 * dim, cart, &r); reqs[p].push_back(r);
                }
                if (nbHi[dim] != MPI_PROC_NULL) {
                    MPI_Recv_init(buf, 1, recvHi[dim], nbHi[dim], 2 * dim, cart, &r); reqs[p].push_back(r);
                    MPI_Send_init(buf, 1, sendHi[dim], nbHi[dim], 2 * dim + 1, cart, &r); reqs[p].push_back(r);
                }
            }
        }

        // Inner box: cells whose stencil does not touch ghost layers
        const long ax0 = std::max<long>(d.ux0, 2), ax1 = std::min<long>(d.ux1, (long)d.lx - 1);
        const long ay0 = std::max<long>(d.uy0, 2), ay1 = std::min<long>(d.uy1, (long)d.ly - 1);
        const long az0 = std::max<long>(d.uz0, 2), az1 = std::min<long>(d.uz1, (long)d.lz - 1);
        const bool innerEmpty = ax0 > ax1 || ay0 > ay1 || az0 > az1;

        if (root) {
            printf("Initializing grid...\n");
            printf("Running stencil computation...\n");
            fflush(stdout);
        }
        MPI_Barrier(cart);
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            const int p = iter % 2;
            const Real* in = bufs[p];
            Real* out = bufs[1 - p];
            std::vector<MPI_Request>& rq = reqs[p];
            if (!rq.empty()) MPI_Startall((int)rq.size(), rq.data());
            if (innerEmpty) {
                if (!rq.empty()) MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
                stencilBox(in, out, d, d.ux0, d.ux1, d.uy0, d.uy1, d.uz0, d.uz1);
            } else {
                stencilBox(in, out, d, ax0, ax1, ay0, ay1, az0, az1);
                if (!rq.empty()) MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
                stencilShell(in, out, d, ax0, ax1, ay0, ay1, az0, az1);
            }
        }

        MPI_Barrier(cart);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        long localDurationMs = duration.count();
        long maxDurationMs = 0;
        MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, cart);

        if (root) {
            printf("Computation time: %ld ms\n", maxDurationMs);
            // Calculate performance metrics
            double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
            double mcups = cellUpdates / (maxDurationMs / 1000.0) / 1e6;  // Million cell updates per second
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        for (int p = 0; p < 2; ++p)
            for (auto& r : reqs[p]) MPI_Request_free(&r);
        for (int dim = 0; dim < 3; ++dim) {
            MPI_Type_free(&sendLo[dim]); MPI_Type_free(&sendHi[dim]);
            MPI_Type_free(&recvLo[dim]); MPI_Type_free(&recvHi[dim]);
        }

        if (printResults || validate) {
            const std::vector<Real>& local = (iterations % 2 == 0) ? grid1 : grid2;
            std::vector<Real> packed(d.lx * d.ly * d.lz);
            {
                size_t k = 0;
                for (size_t z = 1; z <= d.lz; ++z)
                    for (size_t y = 1; y <= d.ly; ++y)
                        for (size_t x = 1; x <= d.lx; ++x) packed[k++] = local[d.at(x, y, z)];
            }
            if (rank == 0) {
                std::vector<Real> finalGrid(nx * ny * nz);
                std::vector<Real> tmp;
                for (int r = 0; r < used; ++r) {
                    int c[3];
                    MPI_Cart_coords(cart, r, 3, c);
                    const size_t ox = blockStart(nx, px, c[2]), lx = blockSize(nx, px, c[2]);
                    const size_t oy = blockStart(ny, py, c[1]), ly = blockSize(ny, py, c[1]);
                    const size_t oz = blockStart(nz, pz, c[0]), lz = blockSize(nz, pz, c[0]);
                    const Real* src;
                    if (r == 0) {
                        src = packed.data();
                    } else {
                        tmp.resize(lx * ly * lz);
                        MPI_Recv(tmp.data(), (int)tmp.size(), MPI_DOUBLE, r, 99, cart, MPI_STATUS_IGNORE);
                        src = tmp.data();
                    }
                    size_t k = 0;
                    for (size_t z = 0; z < lz; ++z)
                        for (size_t y = 0; y < ly; ++y) {
                            std::memcpy(&finalGrid[idx3(ox, oy + y, oz + z, nx, ny)], src + k, lx * sizeof(Real));
                            k += lx;
                        }
                }

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
                fflush(stdout);
            } else {
                MPI_Send(packed.data(), (int)packed.size(), MPI_DOUBLE, 0, 99, cart);
            }
        }

        MPI_Comm_free(&cart);
        MPI_Comm_free(&activeComm);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
