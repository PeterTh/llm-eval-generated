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

// Describes the sub-domain owned by one MPI rank.
// Local arrays carry one ghost layer on every side: local dims are (lnx+2, lny+2, lnz+2),
// local index 1..ln maps to global index g0..g1-1.
struct Domain {
    size_t nx, ny, nz;          // global sizes
    size_t gx0, gx1;            // owned global range in x: [gx0, gx1)
    size_t gy0, gy1;
    size_t gz0, gz1;
    size_t lnx, lny, lnz;       // owned extents
    size_t pnx, pny, pnz;       // padded extents (owned + 2 ghost)
    int nbr[3][2];              // neighbour ranks: [dim][0=low,1=high]
    int rank, size;
    MPI_Comm comm;
};

static void splitRange(const size_t n, const int parts, const int part, size_t& lo, size_t& hi) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t rem = n % static_cast<size_t>(parts);
    const size_t p = static_cast<size_t>(part);
    lo = p * base + std::min(p, rem);
    hi = lo + base + (p < rem ? 1 : 0);
}

// Initialize owned cells of the local (ghost-padded) grid with the same values as the
// original sequential code: value = (global_linear_index % 19).
void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    for (size_t z = 0; z < d.lnz; ++z) {
        for (size_t y = 0; y < d.lny; ++y) {
            Real* row = &grid[idx3(1, y + 1, z + 1, d.pnx, d.pny)];
            const size_t gbase = idx3(d.gx0, d.gy0 + y, d.gz0 + z, d.nx, d.ny);
            for (size_t x = 0; x < d.lnx; ++x) {
                row[x] = static_cast<Real>((gbase + x) % 19);
            }
        }
    }
}

// 7-point stencil on the local box [xs,xe) x [ys,ye) x [zs,ze) (local padded coordinates).
static inline void stencilBox(const Real* __restrict in, Real* __restrict out,
                              const size_t pnx, const size_t pny,
                              const size_t xs, const size_t xe,
                              const size_t ys, const size_t ye,
                              const size_t zs, const size_t ze) {
    if (xs >= xe || ys >= ye || zs >= ze) return;
    const size_t plane = pnx * pny;
    for (size_t z = zs; z < ze; ++z) {
        for (size_t y = ys; y < ye; ++y) {
            const size_t base = z * plane + y * pnx;
            const Real* __restrict c = in + base;
            const Real* __restrict f = in + base - pnx;
            const Real* __restrict b = in + base + pnx;
            const Real* __restrict lo = in + base - plane;
            const Real* __restrict hi = in + base + plane;
            Real* __restrict o = out + base;
            for (size_t x = xs; x < xe; ++x) {
                // Same summation order as the original code for bit-identical results.
                o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + b[x] + lo[x] + hi[x]) / 7.0;
            }
        }
    }
}

struct HaloTypes {
    MPI_Datatype face[3];  // subarray type for a face normal to dim 0/1/2 (x/y/z)
};

static HaloTypes createHaloTypes(const Domain& d) {
    HaloTypes t;
    const int sizes[3] = {static_cast<int>(d.pnz), static_cast<int>(d.pny), static_cast<int>(d.pnx)};
    // x-face: 1 x lny x lnz ; y-face: lnx x 1 x lnz ; z-face: lnx x lny x 1   (C order: z,y,x)
    const int subX[3] = {static_cast<int>(d.lnz), static_cast<int>(d.lny), 1};
    const int subY[3] = {static_cast<int>(d.lnz), 1, static_cast<int>(d.lnx)};
    const int subZ[3] = {1, static_cast<int>(d.lny), static_cast<int>(d.lnx)};
    const int starts[3] = {0, 0, 0};
    MPI_Type_create_subarray(3, sizes, subX, starts, MPI_ORDER_C, MPI_DOUBLE, &t.face[0]);
    MPI_Type_create_subarray(3, sizes, subY, starts, MPI_ORDER_C, MPI_DOUBLE, &t.face[1]);
    MPI_Type_create_subarray(3, sizes, subZ, starts, MPI_ORDER_C, MPI_DOUBLE, &t.face[2]);
    for (int i = 0; i < 3; ++i) MPI_Type_commit(&t.face[i]);
    return t;
}

static void freeHaloTypes(HaloTypes& t) {
    for (int i = 0; i < 3; ++i) MPI_Type_free(&t.face[i]);
}

// Post non-blocking halo exchange for all six faces of `grid`.
static void startHaloExchange(std::vector<Real>& grid, const Domain& d, const HaloTypes& t,
                              MPI_Request* reqs) {
    const size_t ln[3] = {d.lnx, d.lny, d.lnz};
    for (int i = 0; i < 12; ++i) reqs[i] = MPI_REQUEST_NULL;
    int r = 0;
    for (int dim = 0; dim < 3; ++dim) {
        // offsets along `dim` for: send-low (1), recv-low (0), send-high (ln), recv-high (ln+1)
        auto off = [&](size_t k) {
            size_t x = 1, y = 1, z = 1;
            (dim == 0 ? x : dim == 1 ? y : z) = k;
            return idx3(x, y, z, d.pnx, d.pny);
        };
        const int tagLo = 2 * dim;      // message travelling toward lower rank
        const int tagHi = 2 * dim + 1;  // message travelling toward higher rank
        // Receive from low neighbour into ghost 0; receive from high neighbour into ghost ln+1
        MPI_Irecv(grid.data() + off(0), 1, t.face[dim], d.nbr[dim][0], tagHi, d.comm, &reqs[r++]);
        MPI_Irecv(grid.data() + off(ln[dim] + 1), 1, t.face[dim], d.nbr[dim][1], tagLo, d.comm, &reqs[r++]);
        // Send owned face 1 to low neighbour; send owned face ln to high neighbour
        MPI_Isend(grid.data() + off(1), 1, t.face[dim], d.nbr[dim][0], tagLo, d.comm, &reqs[r++]);
        MPI_Isend(grid.data() + off(ln[dim]), 1, t.face[dim], d.nbr[dim][1], tagHi, d.comm, &reqs[r++]);
    }
}

// One stencil iteration on the distributed grid. Global boundary cells are never written,
// so they keep their initial values in both buffers (equivalent to the original boundary copy).
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const Domain& d, const HaloTypes& t) {
    MPI_Request reqs[12];
    startHaloExchange(input, d, t, reqs);

    // Range of locally computed cells (local padded coords), excluding global boundary
    const size_t cxs = (d.gx0 == 0) ? 2 : 1, cxe = (d.gx1 == d.nx) ? d.lnx : d.lnx + 1;
    const size_t cys = (d.gy0 == 0) ? 2 : 1, cye = (d.gy1 == d.ny) ? d.lny : d.lny + 1;
    const size_t czs = (d.gz0 == 0) ? 2 : 1, cze = (d.gz1 == d.nz) ? d.lnz : d.lnz + 1;

    // Inner block that does not touch any ghost cell: can be computed during communication
    // (bounds are clamped into the compute range so the shells below never leave it)
    const size_t ixs = std::min(std::max<size_t>(cxs, 2), cxe), ixe = std::max(ixs, std::min(cxe, d.lnx));
    const size_t iys = std::min(std::max<size_t>(cys, 2), cye), iye = std::max(iys, std::min(cye, d.lny));
    const size_t izs = std::min(std::max<size_t>(czs, 2), cze), ize = std::max(izs, std::min(cze, d.lnz));

    const Real* in = input.data();
    Real* out = output.data();
    stencilBox(in, out, d.pnx, d.pny, ixs, ixe, iys, iye, izs, ize);

    MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);

    // Remaining shells adjacent to the ghost layers
    stencilBox(in, out, d.pnx, d.pny, cxs, cxe, cys, cye, czs, izs);   // z-low plane(s)
    stencilBox(in, out, d.pnx, d.pny, cxs, cxe, cys, cye, ize, cze);   // z-high plane(s)
    stencilBox(in, out, d.pnx, d.pny, cxs, cxe, cys, iys, izs, ize);   // y-low
    stencilBox(in, out, d.pnx, d.pny, cxs, cxe, iye, cye, izs, ize);   // y-high
    stencilBox(in, out, d.pnx, d.pny, cxs, ixs, iys, iye, izs, ize);   // x-low
    stencilBox(in, out, d.pnx, d.pny, ixe, cxe, iys, iye, izs, ize);   // x-high
}

// Gather the distributed grid into a full global grid on rank 0 (global layout as original).
static void gatherGrid(const std::vector<Real>& local, const Domain& d, std::vector<Real>& global) {
    const int n = static_cast<int>(d.lnx * d.lny * d.lnz);
    std::vector<Real> packed(static_cast<size_t>(n));
    for (size_t z = 0; z < d.lnz; ++z)
        for (size_t y = 0; y < d.lny; ++y)
            std::memcpy(&packed[(z * d.lny + y) * d.lnx],
                        &local[idx3(1, y + 1, z + 1, d.pnx, d.pny)], d.lnx * sizeof(Real));

    std::vector<int> counts, displs;
    std::vector<Real> recv;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
    }
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, d.comm);
    if (d.rank == 0) {
        size_t total = 0;
        for (int r = 0; r < d.size; ++r) { displs[r] = static_cast<int>(total); total += counts[r]; }
        recv.resize(total);
    }
    MPI_Gatherv(packed.data(), n, MPI_DOUBLE, recv.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, d.comm);

    if (d.rank == 0) {
        global.resize(d.nx * d.ny * d.nz);
        int dims[3], periods[3], coords[3];
        MPI_Cart_get(d.comm, 3, dims, periods, coords);
        for (int r = 0; r < d.size; ++r) {
            MPI_Cart_coords(d.comm, r, 3, coords);
            size_t x0, x1, y0, y1, z0, z1;
            splitRange(d.nz, dims[0], coords[0], z0, z1);
            splitRange(d.ny, dims[1], coords[1], y0, y1);
            splitRange(d.nx, dims[2], coords[2], x0, x1);
            const size_t lx = x1 - x0, ly = y1 - y0;
            const Real* src = recv.data() + displs[r];
            for (size_t z = z0; z < z1; ++z)
                for (size_t y = y0; y < y1; ++y)
                    std::memcpy(&global[idx3(x0, y, z, d.nx, d.ny)],
                                src + ((z - z0) * ly + (y - y0)) * lx, lx * sizeof(Real));
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
    
    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    // 3D Cartesian decomposition: choose the factorisation of the rank count that
    // minimises the halo surface per rank, never splitting a dimension finer than its
    // size and preferring to keep the contiguous x rows long (ties favour fewer x parts).
    // dims[] is ordered (z, y, x) to match the C-order local arrays.
    int dims[3] = {0, 0, 0};
    {
        const double ext[3] = {static_cast<double>(nz), static_cast<double>(ny), static_cast<double>(nx)};
        double best = -1.0;
        for (int pz = 1; pz <= worldSize; ++pz) {
            if (worldSize % pz != 0 || static_cast<size_t>(pz) > nz) continue;
            const int rest = worldSize / pz;
            for (int py = 1; py <= rest; ++py) {
                if (rest % py != 0 || static_cast<size_t>(py) > ny) continue;
                const int px = rest / py;
                if (static_cast<size_t>(px) > nx) continue;
                const double lz = ext[0] / pz, ly = ext[1] / py, lx = ext[2] / px;
                const double surface = (pz > 1 ? lx * ly : 0.0) + (py > 1 ? lx * lz : 0.0) + (px > 1 ? ly * lz : 0.0);
                if (best < 0.0 || surface < best - 1e-9 ||
                    (std::fabs(surface - best) <= 1e-9 &&
                     (px < dims[2] || (px == dims[2] && py < dims[1])))) {
                    best = surface;
                    dims[0] = pz; dims[1] = py; dims[2] = px;
                }
            }
        }
        if (best < 0.0) {
            if (worldRank == 0)
                fprintf(stderr, "Error: %d MPI ranks cannot be laid out on a %zu x %zu x %zu grid\n",
                        worldSize, nx, ny, nz);
            MPI_Finalize();
            return 1;
        }
    }
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);

    Domain d;
    d.nx = nx; d.ny = ny; d.nz = nz;
    d.comm = cart;
    MPI_Comm_rank(cart, &d.rank);
    MPI_Comm_size(cart, &d.size);
    int coords[3];
    MPI_Cart_coords(cart, d.rank, 3, coords);
    splitRange(nz, dims[0], coords[0], d.gz0, d.gz1);
    splitRange(ny, dims[1], coords[1], d.gy0, d.gy1);
    splitRange(nx, dims[2], coords[2], d.gx0, d.gx1);
    d.lnx = d.gx1 - d.gx0; d.lny = d.gy1 - d.gy0; d.lnz = d.gz1 - d.gz0;
    d.pnx = d.lnx + 2; d.pny = d.lny + 2; d.pnz = d.lnz + 2;
    MPI_Cart_shift(cart, 2, 1, &d.nbr[0][0], &d.nbr[0][1]);  // x
    MPI_Cart_shift(cart, 1, 1, &d.nbr[1][0], &d.nbr[1][1]);  // y
    MPI_Cart_shift(cart, 0, 1, &d.nbr[2][0], &d.nbr[2][1]);  // z

    if (d.rank == 0) {
        printf("Decomposition: %d x %d x %d (x,y,z)\n", dims[2], dims[1], dims[0]);
    }

    const size_t localSize = d.pnx * d.pny * d.pnz;
    
    // Allocate grids (double buffering), ghost-padded
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize, 0.0);
    
    // Initialize both buffers so that global boundary cells (never recomputed) hold their
    // initial values in whichever buffer ends up being the final one.
    if (d.rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, d);
    initializeGrid(grid2, d);

    HaloTypes halo = createHaloTypes(d);
    
    // Run stencil iterations
    if (d.rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, d, halo);
        } else {
            stencilIteration(grid2, grid1, d, halo);
        }
    }
    
    MPI_Barrier(cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, cart);
    
    freeHaloTypes(halo);

    int exitCode = 0;
    if (d.rank == 0) {
        printf("Computation time: %ld ms\n", global_ms);
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (global_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults || validate) {
        std::vector<Real> finalGrid;
        gatherGrid(finalLocal, d, finalGrid);

        if (d.rank == 0) {
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
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, cart);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return exitCode;
}
