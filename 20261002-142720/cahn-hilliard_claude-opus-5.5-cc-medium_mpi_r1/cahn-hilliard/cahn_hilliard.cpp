#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Block distribution of n items over p parts: returns start and count of part i
static inline void blockRange(const size_t n, const int p, const int i, size_t& start, size_t& count) {
    const size_t base = n / p;
    const size_t rem = n % p;
    const size_t ui = static_cast<size_t>(i);
    count = base + (ui < rem ? 1 : 0);
    start = ui * base + std::min(ui, rem);
}

// Local subdomain (2D decomposition over z and y; x is kept whole and contiguous).
// Local arrays carry one ghost layer on every side: dimensions (nx+2) x (ly+2) x (lz+2).
// Ghost cells at the global boundary replicate the boundary value, which reproduces the
// clamped boundary conditions of the original code exactly.
struct Domain {
    size_t nx, ny, nz;          // global sizes
    size_t ly, lz;              // local interior sizes
    size_t y0, z0;              // global offsets
    size_t X, Y, Z;             // padded local sizes
    int nbrZlo, nbrZhi, nbrYlo, nbrYhi;
    MPI_Comm comm;
    MPI_Datatype yFace;         // one y-row (full padded x) per interior z plane
};

// Copy x-ghosts of one row from its boundary values
static inline void fillRowGhosts(double* __restrict row, const size_t nx) {
    row[0] = row[1];
    row[nx + 1] = row[nx];
}

// Chemical potential on box z in [za,zb), y in [ya,yb) (local padded indices)
static void computeChemicalPotentialBox(const double* __restrict c, double* __restrict mu, const Domain& d,
                                        const size_t za, const size_t zb, const size_t ya, const size_t yb,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t X = d.X;
    const size_t XY = d.X * d.Y;
    const size_t nx = d.nx;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    for (size_t z = za; z < zb; ++z) {
        for (size_t y = ya; y < yb; ++y) {
            const size_t row = z * XY + y * X;
            const double* __restrict cr = c + row;
            const double* __restrict cym = cr - X;
            const double* __restrict cyp = cr + X;
            const double* __restrict czm = cr - XY;
            const double* __restrict czp = cr + XY;
            double* __restrict mr = mu + row;
#pragma GCC ivdep
            for (size_t x = 1; x <= nx; ++x) {
                const double cv = cr[x];
                const double cxx = (cr[x + 1] + cr[x - 1] - 2.0 * cv) / dx2;
                const double cyy = (cyp[x] + cym[x] - 2.0 * cv) / dy2;
                const double czz = (czp[x] + czm[x] - 2.0 * cv) / dz2;
                mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * (cxx + cyy + czz);
            }
            fillRowGhosts(mr, nx);
        }
    }
}

// Cahn-Hilliard update on box z in [za,zb), y in [ya,yb) (local padded indices)
static void cahnHilliardUpdateBox(double* __restrict cnew, const double* __restrict cold,
                                  const double* __restrict mu, const Domain& d,
                                  const size_t za, const size_t zb, const size_t ya, const size_t yb,
                                  const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t X = d.X;
    const size_t XY = d.X * d.Y;
    const size_t nx = d.nx;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const double dtD = dt * D;
    for (size_t z = za; z < zb; ++z) {
        for (size_t y = ya; y < yb; ++y) {
            const size_t row = z * XY + y * X;
            const double* __restrict mr = mu + row;
            const double* __restrict mym = mr - X;
            const double* __restrict myp = mr + X;
            const double* __restrict mzm = mr - XY;
            const double* __restrict mzp = mr + XY;
            const double* __restrict co = cold + row;
            double* __restrict cn = cnew + row;
#pragma GCC ivdep
            for (size_t x = 1; x <= nx; ++x) {
                const double mv = mr[x];
                const double mxx = (mr[x + 1] + mr[x - 1] - 2.0 * mv) / dx2;
                const double myy = (myp[x] + mym[x] - 2.0 * mv) / dy2;
                const double mzz = (mzp[x] + mzm[x] - 2.0 * mv) / dz2;
                cn[x] = co[x] + dtD * (mxx + myy + mzz);
            }
            fillRowGhosts(cn, nx);
        }
    }
}

// Apply a kernel over the boundary shell of the local interior (cells that depend on ghosts
// received from neighbors), or over the deep interior.
template <typename F>
static void forInterior(const Domain& d, F&& f) {
    // z in [2, lz), y in [2, ly)
    if (d.lz > 2 && d.ly > 2) f(2, d.lz, 2, d.ly);
}

template <typename F>
static void forShell(const Domain& d, F&& f) {
    const size_t lz = d.lz, ly = d.ly;
    f(1, 2, 1, ly + 1);                       // z = 1 plane
    if (lz > 1) f(lz, lz + 1, 1, ly + 1);     // z = lz plane
    if (lz > 2) {
        f(2, lz, 1, 2);                       // y = 1 rows
        if (ly > 1) f(2, lz, ly, ly + 1);     // y = ly rows
    }
}

// Start halo exchange of a field; global-boundary ghosts are filled locally (clamped BC).
static void startHalo(double* f, const Domain& d, MPI_Request* req) {
    const size_t X = d.X, XY = d.X * d.Y;
    // z faces cover only interior y rows (contiguous), so they never overlap the y ghosts
    const size_t planeLen = d.ly * X;
    const int plane = static_cast<int>(planeLen);
    double* zlo_ghost = f + X;
    double* zlo_send = f + XY + X;
    double* zhi_send = f + d.lz * XY + X;
    double* zhi_ghost = f + (d.lz + 1) * XY + X;
    // Local fills first so no buffer is touched while communication is pending
    if (d.nbrZlo == MPI_PROC_NULL) std::memcpy(zlo_ghost, zlo_send, planeLen * sizeof(double));
    if (d.nbrZhi == MPI_PROC_NULL) std::memcpy(zhi_ghost, zhi_send, planeLen * sizeof(double));
    if (d.nbrYlo == MPI_PROC_NULL) {
        for (size_t z = 1; z <= d.lz; ++z)
            std::memcpy(f + z * XY, f + z * XY + X, X * sizeof(double));
    }
    if (d.nbrYhi == MPI_PROC_NULL) {
        for (size_t z = 1; z <= d.lz; ++z)
            std::memcpy(f + z * XY + (d.ly + 1) * X, f + z * XY + d.ly * X, X * sizeof(double));
    }

    MPI_Irecv(zlo_ghost, plane, MPI_DOUBLE, d.nbrZlo, 0, d.comm, &req[0]);
    MPI_Irecv(zhi_ghost, plane, MPI_DOUBLE, d.nbrZhi, 1, d.comm, &req[1]);
    MPI_Irecv(f + XY, 1, d.yFace, d.nbrYlo, 2, d.comm, &req[2]);
    MPI_Irecv(f + XY + (d.ly + 1) * X, 1, d.yFace, d.nbrYhi, 3, d.comm, &req[3]);
    MPI_Isend(zhi_send, plane, MPI_DOUBLE, d.nbrZhi, 0, d.comm, &req[4]);
    MPI_Isend(zlo_send, plane, MPI_DOUBLE, d.nbrZlo, 1, d.comm, &req[5]);
    MPI_Isend(f + XY + d.ly * X, 1, d.yFace, d.nbrYhi, 2, d.comm, &req[6]);
    MPI_Isend(f + XY + X, 1, d.yFace, d.nbrYlo, 3, d.comm, &req[7]);
}

static inline void finishHalo(MPI_Request* req) {
    MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
}

// Initialize concentration field (local part, using global linear indices)
void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t nx = d.nx, ny = d.ny, nz = d.nz;
    const size_t vol = nx * ny * nz;
    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t gz = d.z0 + z - 1;
        for (size_t y = 1; y <= d.ly; ++y) {
            const size_t gy = d.y0 + y - 1;
            double* row = c.data() + z * d.X * d.Y + y * d.X;
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx3(x, gy, gz, nx, ny);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[x + 1] = -1.0 + 2.0 * pseudo;
            }
            fillRowGhosts(row, nx);
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Choose process grid (pz, py) minimizing halo volume per rank; prefer z splits (contiguous faces).
static bool chooseProcGrid(const int P, const size_t ny, const size_t nz, int& pz, int& py) {
    double best = -1.0;
    for (int a = 1; a <= P; ++a) {
        if (P % a) continue;
        const int b = P / a;
        if (static_cast<size_t>(a) > nz || static_cast<size_t>(b) > ny) continue;
        const double lz = std::ceil(static_cast<double>(nz) / a);
        const double ly = std::ceil(static_cast<double>(ny) / b);
        // y faces are strided and slightly more costly
        const double cost = (a > 1 ? 2.0 * ly : 0.0) + (b > 1 ? 2.0 * 1.1 * lz : 0.0);
        if (best < 0.0 || cost < best) {
            best = cost;
            pz = a;
            py = b;
        }
    }
    return best >= 0.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;

    // Domain decomposition
    int pz = 1, py = 1;
    if (!chooseProcGrid(nprocs, ny, nz, pz, py)) {
        if (root) fprintf(stderr, "Error: cannot decompose %zu x %zu (y x z) over %d processes\n", ny, nz, nprocs);
        MPI_Finalize();
        return 1;
    }
    Domain d{};
    d.nx = nx; d.ny = ny; d.nz = nz;
    {
        int dims[2] = {pz, py};
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &d.comm);
        int crank, coords[2];
        MPI_Comm_rank(d.comm, &crank);
        MPI_Cart_coords(d.comm, crank, 2, coords);
        blockRange(nz, pz, coords[0], d.z0, d.lz);
        blockRange(ny, py, coords[1], d.y0, d.ly);
        MPI_Cart_shift(d.comm, 0, 1, &d.nbrZlo, &d.nbrZhi);
        MPI_Cart_shift(d.comm, 1, 1, &d.nbrYlo, &d.nbrYhi);
    }
    d.X = nx + 2; d.Y = d.ly + 2; d.Z = d.lz + 2;
    MPI_Type_vector(static_cast<int>(d.lz), static_cast<int>(d.X), static_cast<int>(d.X * d.Y), MPI_DOUBLE, &d.yFace);
    MPI_Type_commit(&d.yFace);

    const size_t localSize = d.X * d.Y * d.Z;
    
    // Allocate arrays
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);
    
    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold, d);
    
    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Request req[8];
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        double* c = cold.data();
        double* m = mu.data();
        double* cn = cnew.data();

        // Compute chemical potential
        startHalo(c, d, req);
        auto muKernel = [&](size_t za, size_t zb, size_t ya, size_t yb) {
            computeChemicalPotentialBox(c, m, d, za, zb, ya, yb, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        };
        forInterior(d, muKernel);
        finishHalo(req);
        forShell(d, muKernel);
        
        // Update concentration
        startHalo(m, d, req);
        auto updKernel = [&](size_t za, size_t zb, size_t ya, size_t yb) {
            cahnHilliardUpdateBox(cn, c, m, d, za, zb, ya, yb, D, dt, dx, dy, dz);
        };
        forInterior(d, updKernel);
        finishHalo(req);
        forShell(d, updKernel);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localElapsedMs = duration.count(), maxElapsedMs = 0;
    MPI_Reduce(&localElapsedMs, &maxElapsedMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (root) {
        printf("Computation time: %ld ms\n", maxElapsedMs);
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxElapsedMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the global field on rank 0 in original memory order
        const size_t localCount = d.lz * d.ly * nx;
        std::vector<double> sendbuf(localCount);
        for (size_t z = 0; z < d.lz; ++z)
            for (size_t y = 0; y < d.ly; ++y)
                std::memcpy(&sendbuf[(z * d.ly + y) * nx], &cold[(z + 1) * d.X * d.Y + (y + 1) * d.X + 1],
                            nx * sizeof(double));

        std::vector<int> counts, displs;
        std::vector<double> recvbuf;
        if (root) {
            counts.resize(nprocs);
            displs.resize(nprocs);
            size_t off = 0;
            for (int r = 0; r < nprocs; ++r) {
                int coords[2];
                MPI_Cart_coords(d.comm, r, 2, coords);
                size_t s, lz, ly;
                blockRange(nz, pz, coords[0], s, lz);
                blockRange(ny, py, coords[1], s, ly);
                counts[r] = static_cast<int>(lz * ly * nx);
                displs[r] = static_cast<int>(off);
                off += lz * ly * nx;
            }
            recvbuf.resize(gridSize);
        }
        MPI_Gatherv(sendbuf.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    recvbuf.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, d.comm);

        if (root) {
            std::vector<double> global(gridSize);
            for (int r = 0; r < nprocs; ++r) {
                int coords[2];
                MPI_Cart_coords(d.comm, r, 2, coords);
                size_t gz0, lz, gy0, ly;
                blockRange(nz, pz, coords[0], gz0, lz);
                blockRange(ny, py, coords[1], gy0, ly);
                const double* src = recvbuf.data() + displs[r];
                for (size_t z = 0; z < lz; ++z)
                    for (size_t y = 0; y < ly; ++y)
                        std::memcpy(&global[idx3(0, gy0 + y, gz0 + z, nx, ny)], src + (z * ly + y) * nx,
                                    nx * sizeof(double));
            }
            recvbuf.clear();
            recvbuf.shrink_to_fit();

            // Print results for external validation
            if (printResults) {
                print_results(global, "Concentration");
            }
            
            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global, nx, ny, nz);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&d.yFace);
    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return exitCode;
}
