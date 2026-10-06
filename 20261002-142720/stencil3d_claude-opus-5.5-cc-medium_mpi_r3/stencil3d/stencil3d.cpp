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

// Local subdomain of a 2D (y,z) block decomposition. Each rank holds the full
// x extent (keeps the unit-stride inner loop long and halo rows contiguous)
// plus one ghost layer in y and z.
struct Domain {
    size_t nx = 0, ny = 0, nz = 0;      // global sizes
    size_t y0 = 0, y1 = 0, z0 = 0, z1 = 0; // owned global ranges [y0,y1) x [z0,z1)
    size_t lny = 0, lnz = 0;            // local sizes including ghost layers
    int up = MPI_PROC_NULL, down = MPI_PROC_NULL;   // z+1 / z-1 neighbours
    int north = MPI_PROC_NULL, south = MPI_PROC_NULL; // y+1 / y-1 neighbours
    // Local ranges of points to update (global interior points we own)
    size_t ly_lo = 0, ly_hi = 0, lz_lo = 0, lz_hi = 0;

    size_t lidx(const size_t x, const size_t ly, const size_t lz) const noexcept {
        return (lz * lny + ly) * nx + x;
    }
};

// Update local points in [zlo,zhi) x [ylo,yhi) x [1,nx-1)
static void stencilBox(const Real* __restrict in, Real* __restrict out, const Domain& d,
                       const size_t zlo, const size_t zhi, const size_t ylo, const size_t yhi) {
    const size_t nx = d.nx;
    const size_t sy = nx;
    const size_t sz = nx * d.lny;
    for (size_t z = zlo; z < zhi; ++z) {
        for (size_t y = ylo; y < yhi; ++y) {
            const size_t base = z * sz + y * sy;
            const Real* __restrict c = in + base;
            const Real* __restrict f = c - sy;
            const Real* __restrict b = c + sy;
            const Real* __restrict lo = c - sz;
            const Real* __restrict hi = c + sz;
            Real* __restrict o = out + base;
#pragma GCC ivdep
            for (size_t x = 1; x < nx - 1; ++x) {
                // Same operand order as the reference for bitwise-identical results
                o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + b[x] + lo[x] + hi[x]) / 7.0;
            }
        }
    }
}

// One stencil iteration with halo exchange overlapped with the inner update.
// Global boundary values are never modified, so both buffers keep them from init.
static void stencilIteration(Real* in, Real* out, const Domain& d,
                             MPI_Comm comm, MPI_Datatype yFace) {
    MPI_Request req[8];
    const size_t nx = d.nx;
    const size_t planeCount = nx * d.lny;
    const int pc = static_cast<int>(planeCount);
    const size_t ownZ = d.lnz - 2;
    const size_t ownY = d.lny - 2;

    // z-direction faces: whole contiguous planes
    MPI_Irecv(in + d.lidx(0, 0, 0), pc, MPI_DOUBLE, d.down, 0, comm, &req[0]);
    MPI_Irecv(in + d.lidx(0, 0, ownZ + 1), pc, MPI_DOUBLE, d.up, 1, comm, &req[1]);
    // y-direction faces: one row per owned z plane
    MPI_Irecv(in + d.lidx(0, 0, 1), 1, yFace, d.south, 2, comm, &req[2]);
    MPI_Irecv(in + d.lidx(0, ownY + 1, 1), 1, yFace, d.north, 3, comm, &req[3]);
    MPI_Isend(in + d.lidx(0, 0, ownZ), pc, MPI_DOUBLE, d.up, 0, comm, &req[4]);
    MPI_Isend(in + d.lidx(0, 0, 1), pc, MPI_DOUBLE, d.down, 1, comm, &req[5]);
    MPI_Isend(in + d.lidx(0, ownY, 1), 1, yFace, d.north, 2, comm, &req[6]);
    MPI_Isend(in + d.lidx(0, 1, 1), 1, yFace, d.south, 3, comm, &req[7]);

    const size_t zlo = d.lz_lo, zhi = d.lz_hi, ylo = d.ly_lo, yhi = d.ly_hi;
    const bool any = zlo < zhi && ylo < yhi && nx > 2;

    // Inner part: does not touch ghost layers
    if (any && zlo + 1 < zhi - 1 && ylo + 1 < yhi - 1) {
        stencilBox(in, out, d, zlo + 1, zhi - 1, ylo + 1, yhi - 1);
    }

    MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

    // Outer shell (depends on ghost data)
    if (any) {
        stencilBox(in, out, d, zlo, zlo + 1, ylo, yhi);
        if (zhi - 1 > zlo) stencilBox(in, out, d, zhi - 1, zhi, ylo, yhi);
        if (zhi - 1 > zlo + 1) {
            stencilBox(in, out, d, zlo + 1, zhi - 1, ylo, ylo + 1);
            if (yhi - 1 > ylo) stencilBox(in, out, d, zlo + 1, zhi - 1, yhi - 1, yhi);
        }
    }
}

// Choose process grid (py x pz) minimizing halo traffic, with each rank owning
// at least one y row and one z plane. Returns number of active ranks.
static int chooseGrid(const int nprocs, const size_t ny, const size_t nz, int& py, int& pz) {
    for (int p = nprocs; p >= 1; --p) {
        double best = -1.0;
        for (int a = 1; a <= p; ++a) {
            if (p % a) continue;
            const int b = p / a; // a = pz, b = py
            if (static_cast<size_t>(a) > nz || static_cast<size_t>(b) > ny) continue;
            // communicated elements per x-row: z-cuts * ny + y-cuts * nz
            const double cost = static_cast<double>(a - 1) * ny + static_cast<double>(b - 1) * nz;
            if (best < 0.0 || cost < best) { best = cost; pz = a; py = b; }
        }
        if (best >= 0.0) return p;
    }
    py = pz = 1;
    return 1;
}

static inline void blockRange(const size_t n, const int p, const int i, size_t& lo, size_t& hi) {
    const size_t base = n / p, rem = n % p;
    const size_t ui = static_cast<size_t>(i);
    lo = ui * base + std::min(ui, rem);
    hi = lo + base + (ui < rem ? 1 : 0);
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
    
    // Parse command line arguments (every rank parses identically)
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

    // Domain decomposition over (y,z); ranks beyond what the grid supports idle
    int py = 1, pz = 1;
    const int active = chooseGrid(worldSize, ny, nz, py, pz);
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    Domain d;
    d.nx = nx; d.ny = ny; d.nz = nz;
    std::vector<Real> grid1, grid2;
    MPI_Datatype yFace = MPI_DATATYPE_NULL;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm cart;
        int dims[2] = {pz, py};
        int periods[2] = {0, 0};
        MPI_Cart_create(comm, 2, dims, periods, 0, &cart);
        MPI_Comm_free(&comm);
        comm = cart;
        int rank, coords[2];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);
        MPI_Cart_shift(comm, 0, 1, &d.down, &d.up);
        MPI_Cart_shift(comm, 1, 1, &d.south, &d.north);
        blockRange(nz, pz, coords[0], d.z0, d.z1);
        blockRange(ny, py, coords[1], d.y0, d.y1);
        d.lnz = d.z1 - d.z0 + 2;
        d.lny = d.y1 - d.y0 + 2;
        // Local index l <-> global index g0 + l - 1; update global 1..n-2
        const size_t gzlo = std::max<size_t>(d.z0, 1);
        const size_t gzhi = std::min(d.z1, nz > 0 ? nz - 1 : 0);
        const size_t gylo = std::max<size_t>(d.y0, 1);
        const size_t gyhi = std::min(d.y1, ny > 0 ? ny - 1 : 0);
        d.lz_lo = gzlo - d.z0 + 1;
        d.lz_hi = gzhi > gzlo ? gzhi - d.z0 + 1 : d.lz_lo;
        d.ly_lo = gylo - d.y0 + 1;
        d.ly_hi = gyhi > gylo ? gyhi - d.y0 + 1 : d.ly_lo;

        MPI_Type_vector(static_cast<int>(d.lnz - 2), static_cast<int>(nx),
                        static_cast<int>(nx * d.lny), MPI_DOUBLE, &yFace);
        MPI_Type_commit(&yFace);

        // Allocate local grids (double buffering) with ghost layers
        const size_t localSize = nx * d.lny * d.lnz;
        grid1.resize(localSize);
        grid2.resize(localSize);
    }

    // Initialize (each rank its own block plus ghosts; both buffers, so the
    // fixed global boundary values are present in either one)
    if (root) printf("Initializing grid...\n");
    if (comm != MPI_COMM_NULL) {
        for (size_t lz = 0; lz < d.lnz; ++lz) {
            if (d.z0 + lz < 1 || d.z0 + lz - 1 >= nz) continue;
            const size_t gz = d.z0 + lz - 1;
            for (size_t ly = 0; ly < d.lny; ++ly) {
                if (d.y0 + ly < 1 || d.y0 + ly - 1 >= ny) continue;
                const size_t gy = d.y0 + ly - 1;
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, gy, gz, nx, ny);
                    const size_t l = d.lidx(x, ly, lz);
                    grid1[l] = grid2[l] = (idx % 19) * 1.0;
                }
            }
        }
    }
    
    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (comm != MPI_COMM_NULL) {
        Real* a = grid1.data();
        Real* b = grid2.data();
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration(a, b, d, comm, yFace);
            std::swap(a, b);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the final grid onto rank 0 (world rank 0 is always active rank 0)
        std::vector<Real> finalGrid;
        if (comm != MPI_COMM_NULL) {
            const std::vector<Real>& local = (iterations % 2 == 0) ? grid1 : grid2;
            const size_t ownY = d.lny - 2, ownZ = d.lnz - 2;
            std::vector<Real> packed(nx * ownY * ownZ);
            for (size_t lz = 0; lz < ownZ; ++lz)
                for (size_t ly = 0; ly < ownY; ++ly)
                    std::memcpy(&packed[(lz * ownY + ly) * nx], &local[d.lidx(0, ly + 1, lz + 1)],
                                nx * sizeof(Real));
            int nActive, myRank;
            MPI_Comm_size(comm, &nActive);
            MPI_Comm_rank(comm, &myRank);
            std::vector<int> counts, displs;
            const int myCount = static_cast<int>(packed.size());
            if (myRank == 0) { counts.resize(nActive); displs.resize(nActive); }
            MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
            std::vector<Real> all;
            if (myRank == 0) {
                size_t off = 0;
                for (int r = 0; r < nActive; ++r) { displs[r] = static_cast<int>(off); off += counts[r]; }
                all.resize(off);
            }
            MPI_Gatherv(packed.data(), myCount, MPI_DOUBLE, all.data(), counts.data(),
                        displs.data(), MPI_DOUBLE, 0, comm);
            if (myRank == 0) {
                finalGrid.resize(gridSize);
                for (int r = 0; r < nActive; ++r) {
                    int c[2];
                    MPI_Cart_coords(comm, r, 2, c);
                    size_t zb, ze, yb, ye;
                    blockRange(nz, pz, c[0], zb, ze);
                    blockRange(ny, py, c[1], yb, ye);
                    const Real* src = all.data() + displs[r];
                    for (size_t z = zb; z < ze; ++z)
                        for (size_t y = yb; y < ye; ++y) {
                            std::memcpy(&finalGrid[idx3(0, y, z, nx, ny)], src, nx * sizeof(Real));
                            src += nx;
                        }
                }
            }
        }

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
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (yFace != MPI_DATATYPE_NULL) MPI_Type_free(&yFace);
    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
