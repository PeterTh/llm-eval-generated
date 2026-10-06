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

// Local subdomain of a 2D (z, y) Cartesian domain decomposition.
// Each rank owns the full x extent and a block of y rows and z planes.
// Local arrays are padded with one ghost layer in every direction:
// ghost cells hold neighbour data (halo exchange) or, at the global boundary,
// a copy of the boundary value, which reproduces the clamped boundary condition exactly.
struct Domain {
    size_t nx, ny, nz;          // global sizes
    size_t nyl, nzl;            // owned local sizes
    size_t y0, z0;              // global offsets of owned block
    size_t sx, sy, sz;          // padded strides (sx = 1)
    size_t nxp, nyp, nzp;       // padded sizes
    int rank, size;
    MPI_Comm comm;
    int zlo, zhi, ylo, yhi;     // neighbour ranks (MPI_PROC_NULL at global boundary)
    MPI_Datatype zface, yface;  // halo datatypes

    size_t at(size_t x, size_t y, size_t z) const noexcept { return z * sz + y * sy + x; }
};

static void blockRange(size_t n, int parts, int coord, size_t& off, size_t& len) {
    const size_t base = n / parts, rem = n % parts;
    const size_t c = static_cast<size_t>(coord);
    len = base + (c < rem ? 1 : 0);
    off = c * base + std::min(c, rem);
}

static void setupDomain(Domain& d, size_t nx, size_t ny, size_t nz) {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Choose pz * py = worldSize minimising halo surface, with pz <= nz and py <= ny
    int bestPz = -1, bestPy = -1;
    double bestCost = 0.0;
    for (int pz = 1; pz <= worldSize; ++pz) {
        if (worldSize % pz) continue;
        const int py = worldSize / pz;
        if (static_cast<size_t>(pz) > nz || static_cast<size_t>(py) > ny) continue;
        const double lz = static_cast<double>(nz) / pz, ly = static_cast<double>(ny) / py;
        // halo volume per rank (x is full); prefer z splits (contiguous faces) on ties
        const double cost = (pz > 1 ? 2.0 * ly : 0.0) + (py > 1 ? 2.0 * lz * 1.05 : 0.0);
        if (bestPz < 0 || cost < bestCost) { bestCost = cost; bestPz = pz; bestPy = py; }
    }
    if (bestPz < 0) {
        int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
        if (r == 0) fprintf(stderr, "Error: cannot decompose %zu x %zu (y x z) onto %d ranks\n", ny, nz, worldSize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[2] = {bestPz, bestPy};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &d.comm);
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Comm_size(d.comm, &d.size);
    int coords[2];
    MPI_Cart_coords(d.comm, d.rank, 2, coords);
    MPI_Cart_shift(d.comm, 0, 1, &d.zlo, &d.zhi);
    MPI_Cart_shift(d.comm, 1, 1, &d.ylo, &d.yhi);

    d.nx = nx; d.ny = ny; d.nz = nz;
    blockRange(nz, dims[0], coords[0], d.z0, d.nzl);
    blockRange(ny, dims[1], coords[1], d.y0, d.nyl);
    d.nxp = nx + 2; d.nyp = d.nyl + 2; d.nzp = d.nzl + 2;
    d.sx = 1; d.sy = d.nxp; d.sz = d.nxp * d.nyp;

    // z face: owned rows y=1..nyl of one plane (contiguous)
    MPI_Type_contiguous(static_cast<int>(d.nyl * d.nxp), MPI_DOUBLE, &d.zface);
    MPI_Type_commit(&d.zface);
    // y face: one row in each owned plane z=1..nzl
    MPI_Type_vector(static_cast<int>(d.nzl), static_cast<int>(d.nxp), static_cast<int>(d.sz), MPI_DOUBLE, &d.yface);
    MPI_Type_commit(&d.yface);
}

// Start halo exchange of field f; global-boundary ghosts are filled locally (clamped BC).
static int startHalo(const Domain& d, double* f, MPI_Request* req) {
    int n = 0;
    const size_t zf = d.at(0, 1, 0);       // start of owned rows in plane 0
    const size_t yf = d.at(0, 0, 1);       // start of row 0 in plane 1
    // z direction
    if (d.zlo != MPI_PROC_NULL) {
        MPI_Irecv(f + zf, 1, d.zface, d.zlo, 0, d.comm, &req[n++]);
        MPI_Isend(f + zf + d.sz, 1, d.zface, d.zlo, 1, d.comm, &req[n++]);
    } else {
        std::memcpy(f + zf, f + zf + d.sz, d.nyl * d.nxp * sizeof(double));
    }
    if (d.zhi != MPI_PROC_NULL) {
        MPI_Irecv(f + zf + (d.nzl + 1) * d.sz, 1, d.zface, d.zhi, 1, d.comm, &req[n++]);
        MPI_Isend(f + zf + d.nzl * d.sz, 1, d.zface, d.zhi, 0, d.comm, &req[n++]);
    } else {
        std::memcpy(f + zf + (d.nzl + 1) * d.sz, f + zf + d.nzl * d.sz, d.nyl * d.nxp * sizeof(double));
    }
    // y direction
    if (d.ylo != MPI_PROC_NULL) {
        MPI_Irecv(f + yf, 1, d.yface, d.ylo, 2, d.comm, &req[n++]);
        MPI_Isend(f + yf + d.sy, 1, d.yface, d.ylo, 3, d.comm, &req[n++]);
    } else {
        for (size_t z = 1; z <= d.nzl; ++z)
            std::memcpy(f + d.at(0, 0, z), f + d.at(0, 1, z), d.nxp * sizeof(double));
    }
    if (d.yhi != MPI_PROC_NULL) {
        MPI_Irecv(f + yf + (d.nyl + 1) * d.sy, 1, d.yface, d.yhi, 3, d.comm, &req[n++]);
        MPI_Isend(f + yf + d.nyl * d.sy, 1, d.yface, d.yhi, 2, d.comm, &req[n++]);
    } else {
        for (size_t z = 1; z <= d.nzl; ++z)
            std::memcpy(f + d.at(0, d.nyl + 1, z), f + d.at(0, d.nyl, z), d.nxp * sizeof(double));
    }
    return n;
}

// Chemical potential on one row (owned y, z), including x ghosts
static inline void muRow(const Domain& d, const double* __restrict c, double* __restrict mu, size_t y, size_t z,
                         const double dx, const double dy, const double dz,
                         const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t row = d.at(0, y, z);
    const size_t sy = d.sy, sz = d.sz, nx = d.nx;
    const double* __restrict cr = c + row;
    double* __restrict mr = mu + row;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma GCC ivdep
    for (size_t x = 1; x <= nx; ++x) {
        const double cv = cr[x];
        const double cxx = (cr[x + 1] + cr[x - 1] - 2.0 * cv) / dx2;
        const double cyy = (cr[x + sy] + cr[x - sy] - 2.0 * cv) / dy2;
        const double czz = (cr[x + sz] + cr[x - sz] - 2.0 * cv) / dz2;
        mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * (cxx + cyy + czz);
    }
    mr[0] = mr[1];
    mr[nx + 1] = mr[nx];
}

// Cahn-Hilliard update on one row (owned y, z), including x ghosts
static inline void updateRow(const Domain& d, double* __restrict cnew, const double* __restrict cold,
                             const double* __restrict mu, size_t y, size_t z,
                             const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t row = d.at(0, y, z);
    const size_t sy = d.sy, sz = d.sz, nx = d.nx;
    const double* __restrict mr = mu + row;
    const double* __restrict co = cold + row;
    double* __restrict cn = cnew + row;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma GCC ivdep
    for (size_t x = 1; x <= nx; ++x) {
        const double mv = mr[x];
        const double cxx = (mr[x + 1] + mr[x - 1] - 2.0 * mv) / dx2;
        const double cyy = (mr[x + sy] + mr[x - sy] - 2.0 * mv) / dy2;
        const double czz = (mr[x + sz] + mr[x - sz] - 2.0 * mv) / dz2;
        cn[x] = co[x] + dt * D * (cxx + cyy + czz);
    }
    cn[0] = cn[1];
    cn[nx + 1] = cn[nx];
}

// Apply rowFn over owned rows: interior first (overlapping communication), then the boundary shell
template <typename RowFn>
static void sweepOverlapped(const Domain& d, MPI_Request* req, int nreq, RowFn&& rowFn) {
    const size_t nyl = d.nyl, nzl = d.nzl;
    for (size_t z = 2; z + 1 <= nzl; ++z)
        for (size_t y = 2; y + 1 <= nyl; ++y)
            rowFn(y, z);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    // planes z = 1 and z = nzl
    for (size_t y = 1; y <= nyl; ++y) rowFn(y, 1);
    if (nzl > 1)
        for (size_t y = 1; y <= nyl; ++y) rowFn(y, nzl);
    // rows y = 1 and y = nyl of inner planes
    for (size_t z = 2; z + 1 <= nzl; ++z) {
        rowFn(1, z);
        if (nyl > 1) rowFn(nyl, z);
    }
}

// Initialize concentration field (owned cells, global indexing) and x ghosts
void initializeConcentration(const Domain& d, std::vector<double>& c) {
    const size_t nx = d.nx, ny = d.ny, nz = d.nz;
    const size_t vol = nx * ny * nz;
    for (size_t z = 1; z <= d.nzl; ++z) {
        for (size_t y = 1; y <= d.nyl; ++y) {
            const size_t gz = d.z0 + z - 1, gy = d.y0 + y - 1;
            double* row = c.data() + d.at(0, y, z);
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx3(x, gy, gz, nx, ny);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[x + 1] = -1.0 + 2.0 * pseudo;
            }
            row[0] = row[1];
            row[nx + 1] = row[nx];
        }
    }
}

// Gather owned blocks into a global array on rank 0 (returns empty vector elsewhere)
static std::vector<double> gatherGlobal(const Domain& d, const std::vector<double>& c) {
    const size_t nx = d.nx;
    std::vector<double> local(nx * d.nyl * d.nzl);
    size_t k = 0;
    for (size_t z = 1; z <= d.nzl; ++z)
        for (size_t y = 1; y <= d.nyl; ++y) {
            std::memcpy(local.data() + k, c.data() + d.at(1, y, z), nx * sizeof(double));
            k += nx;
        }

    long long info[4] = {static_cast<long long>(d.y0), static_cast<long long>(d.nyl),
                         static_cast<long long>(d.z0), static_cast<long long>(d.nzl)};
    std::vector<long long> allInfo(d.rank == 0 ? 4 * d.size : 0);
    MPI_Gather(info, 4, MPI_LONG_LONG, allInfo.data(), 4, MPI_LONG_LONG, 0, d.comm);

    std::vector<int> counts, displs;
    std::vector<double> recv;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
        size_t off = 0;
        for (int r = 0; r < d.size; ++r) {
            const size_t cnt = nx * static_cast<size_t>(allInfo[4 * r + 1]) * static_cast<size_t>(allInfo[4 * r + 3]);
            counts[r] = static_cast<int>(cnt);
            displs[r] = static_cast<int>(off);
            off += cnt;
        }
        recv.resize(off);
    }
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                recv.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, d.comm);

    std::vector<double> global;
    if (d.rank == 0) {
        global.resize(nx * d.ny * d.nz);
        for (int r = 0; r < d.size; ++r) {
            const size_t ry0 = allInfo[4 * r], rnyl = allInfo[4 * r + 1];
            const size_t rz0 = allInfo[4 * r + 2], rnzl = allInfo[4 * r + 3];
            const double* src = recv.data() + displs[r];
            for (size_t z = 0; z < rnzl; ++z)
                for (size_t y = 0; y < rnyl; ++y) {
                    std::memcpy(global.data() + idx3(0, ry0 + y, rz0 + z, nx, d.ny), src, nx * sizeof(double));
                    src += nx;
                }
        }
    }
    return global;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const bool root = (worldRank == 0);

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

    Domain d;
    setupDomain(d, nx, ny, nz);
    const size_t localSize = d.nxp * d.nyp * d.nzp;
    
    // Allocate padded local arrays
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);
    
    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(d, cold);
    
    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    MPI_Request req[8];
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        {
            const double* c = cold.data();
            double* m = mu.data();
            const int nreq = startHalo(d, cold.data(), req);
            sweepOverlapped(d, req, nreq, [&](size_t y, size_t z) {
                muRow(d, c, m, y, z, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            });
        }
        
        // Update concentration
        {
            const double* co = cold.data();
            const double* m = mu.data();
            double* cn = cnew.data();
            const int nreq = startHalo(d, mu.data(), req);
            sweepOverlapped(d, req, nreq, [&](size_t y, size_t z) {
                updateRow(d, cn, co, m, y, z, D, dt, dx, dy, dz);
            });
        }
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(d.comm);
    auto end = std::chrono::high_resolution_clock::now();
    const long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(maxDuration);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    int ret = 0;
    if (printResults || validate) {
        std::vector<double> global = gatherGlobal(d, cold);
        
        if (root) {
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
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, d.comm);
    }
    
    MPI_Type_free(&d.zface);
    MPI_Type_free(&d.yface);
    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return ret;
}
