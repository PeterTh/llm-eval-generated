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

// Local subdomain of the global grid owned by one MPI rank.
// Arrays are stored with one ghost layer on each side: padded sizes are
// (lnx+2) x (lny+2) x (lnz+2), interior cells have local indices 1..ln*.
struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};
    int nbrLo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbrHi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    size_t n[3] = {0, 0, 0};       // global sizes
    size_t ln[3] = {0, 0, 0};      // local interior sizes
    size_t off[3] = {0, 0, 0};     // global offset of first interior cell
    size_t px = 0, py = 0, pz = 0; // padded sizes
    size_t sy = 0, sz = 0;         // strides
    MPI_Datatype sendLo[3], sendHi[3], recvLo[3], recvHi[3];
};

static void blockRange(const size_t n, const int p, const int c, size_t& len, size_t& off) {
    const size_t base = n / p, rem = n % p;
    len = base + (static_cast<size_t>(c) < rem ? 1 : 0);
    off = c * base + std::min<size_t>(c, rem);
}

// Choose a process grid minimizing halo traffic; x-faces are strided and
// therefore weighted more heavily so that splits along z/y are preferred.
// Returns false if P ranks cannot be arranged with at least one cell each.
static bool chooseDims(const int P, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    double best = -1.0;
    dims[0] = dims[1] = dims[2] = 1;
    for (int a = 1; a <= P; ++a) {
        if (P % a) continue;
        for (int b = 1; b <= P / a; ++b) {
            if ((P / a) % b) continue;
            const int c = P / a / b;
            if (static_cast<size_t>(a) > nx || static_cast<size_t>(b) > ny || static_cast<size_t>(c) > nz) continue;
            const double lx = std::ceil(double(nx) / a), ly = std::ceil(double(ny) / b), lz = std::ceil(double(nz) / c);
            double cost = 0.0;
            if (a > 1) cost += 4.0 * ly * lz;
            if (b > 1) cost += 1.5 * lx * lz;
            if (c > 1) cost += 1.0 * lx * ly;
            cost += 1e-3 * (a + b); // tie-break: prefer z, then y splits
            if (best < 0.0 || cost < best) {
                best = cost;
                dims[0] = a; dims[1] = b; dims[2] = c;
            }
        }
    }
    return best >= 0.0;
}

// Returns false on ranks that are left idle (only if the grid cannot be split
// among all ranks, e.g. more ranks than cells).
static bool setupDomain(Domain& d, const size_t nx, const size_t ny, const size_t nz) {
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    d.n[0] = nx; d.n[1] = ny; d.n[2] = nz;
    int dimsXYZ[3] = {1, 1, 1};
    int active = worldSize;
    while (active > 1 && !chooseDims(active, nx, ny, nz, dimsXYZ)) --active;
    if (active == 1) dimsXYZ[0] = dimsXYZ[1] = dimsXYZ[2] = 1;

    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (activeComm == MPI_COMM_NULL) return false;

    // MPI Cartesian ordering: last dim varies fastest -> use (z, y, x) order
    int cdims[3] = {dimsXYZ[2], dimsXYZ[1], dimsXYZ[0]};
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(activeComm, 3, cdims, periods, 0, &d.comm);
    MPI_Comm_free(&activeComm);
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Comm_size(d.comm, &d.size);
    int cc[3];
    MPI_Cart_coords(d.comm, d.rank, 3, cc);
    for (int k = 0; k < 3; ++k) {
        d.dims[k] = dimsXYZ[k];
        d.coords[k] = cc[2 - k];
        MPI_Cart_shift(d.comm, 2 - k, 1, &d.nbrLo[k], &d.nbrHi[k]);
        blockRange(d.n[k], d.dims[k], d.coords[k], d.ln[k], d.off[k]);
    }
    d.px = d.ln[0] + 2; d.py = d.ln[1] + 2; d.pz = d.ln[2] + 2;
    d.sy = d.px; d.sz = d.px * d.py;

    // Face datatypes (subarrays of the padded array, C order = z, y, x)
    const int sizes[3] = {static_cast<int>(d.pz), static_cast<int>(d.py), static_cast<int>(d.px)};
    for (int k = 0; k < 3; ++k) {
        const int ck = 2 - k; // dimension index in (z, y, x) order
        int sub[3] = {static_cast<int>(d.ln[2]), static_cast<int>(d.ln[1]), static_cast<int>(d.ln[0])};
        sub[ck] = 1;
        int sLo[3] = {1, 1, 1}, sHi[3] = {1, 1, 1}, rLo[3] = {1, 1, 1}, rHi[3] = {1, 1, 1};
        sHi[ck] = static_cast<int>(d.ln[k]);
        rLo[ck] = 0;
        rHi[ck] = static_cast<int>(d.ln[k]) + 1;
        MPI_Type_create_subarray(3, sizes, sub, sLo, MPI_ORDER_C, MPI_DOUBLE, &d.sendLo[k]);
        MPI_Type_create_subarray(3, sizes, sub, sHi, MPI_ORDER_C, MPI_DOUBLE, &d.sendHi[k]);
        MPI_Type_create_subarray(3, sizes, sub, rLo, MPI_ORDER_C, MPI_DOUBLE, &d.recvLo[k]);
        MPI_Type_create_subarray(3, sizes, sub, rHi, MPI_ORDER_C, MPI_DOUBLE, &d.recvHi[k]);
        MPI_Type_commit(&d.sendLo[k]); MPI_Type_commit(&d.sendHi[k]);
        MPI_Type_commit(&d.recvLo[k]); MPI_Type_commit(&d.recvHi[k]);
    }
    return true;
}

static void freeDomain(Domain& d) {
    for (int k = 0; k < 3; ++k) {
        MPI_Type_free(&d.sendLo[k]); MPI_Type_free(&d.sendHi[k]);
        MPI_Type_free(&d.recvLo[k]); MPI_Type_free(&d.recvHi[k]);
    }
    MPI_Comm_free(&d.comm);
}

// Post non-blocking halo exchange of all six faces
static int startHalo(const Domain& d, double* a, MPI_Request* req) {
    int nr = 0;
    for (int k = 0; k < 3; ++k) {
        if (d.nbrLo[k] != MPI_PROC_NULL) {
            MPI_Irecv(a, 1, d.recvLo[k], d.nbrLo[k], 2 * k + 1, d.comm, &req[nr++]);
            MPI_Isend(a, 1, d.sendLo[k], d.nbrLo[k], 2 * k, d.comm, &req[nr++]);
        }
        if (d.nbrHi[k] != MPI_PROC_NULL) {
            MPI_Irecv(a, 1, d.recvHi[k], d.nbrHi[k], 2 * k, d.comm, &req[nr++]);
            MPI_Isend(a, 1, d.sendHi[k], d.nbrHi[k], 2 * k + 1, d.comm, &req[nr++]);
        }
    }
    return nr;
}

// Clamped boundary condition at physical domain edges: ghost = boundary cell
static void fillPhysicalGhosts(const Domain& d, double* a) {
    const size_t lx = d.ln[0], ly = d.ln[1], lz = d.ln[2];
    const size_t sy = d.sy, sz = d.sz;
    if (d.nbrLo[0] == MPI_PROC_NULL || d.nbrHi[0] == MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z)
            for (size_t y = 1; y <= ly; ++y) {
                double* row = a + z * sz + y * sy;
                if (d.nbrLo[0] == MPI_PROC_NULL) row[0] = row[1];
                if (d.nbrHi[0] == MPI_PROC_NULL) row[lx + 1] = row[lx];
            }
    }
    for (size_t z = 1; z <= lz; ++z) {
        if (d.nbrLo[1] == MPI_PROC_NULL)
            std::memcpy(a + z * sz + 0 * sy + 1, a + z * sz + 1 * sy + 1, lx * sizeof(double));
        if (d.nbrHi[1] == MPI_PROC_NULL)
            std::memcpy(a + z * sz + (ly + 1) * sy + 1, a + z * sz + ly * sy + 1, lx * sizeof(double));
    }
    for (size_t y = 1; y <= ly; ++y) {
        if (d.nbrLo[2] == MPI_PROC_NULL)
            std::memcpy(a + 0 * sz + y * sy + 1, a + 1 * sz + y * sy + 1, lx * sizeof(double));
        if (d.nbrHi[2] == MPI_PROC_NULL)
            std::memcpy(a + (lz + 1) * sz + y * sy + 1, a + lz * sz + y * sy + 1, lx * sizeof(double));
    }
}

struct Box { size_t x0, x1, y0, y1, z0, z1; };

// Compute chemical potential on a box of local (padded) indices
static void computeChemicalPotential(const double* __restrict c, double* __restrict mu, const Domain& d, const Box& b,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t sy = d.sy, sz = d.sz;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    for (size_t z = b.z0; z < b.z1; ++z) {
        for (size_t y = b.y0; y < b.y1; ++y) {
            const size_t base = z * sz + y * sy;
            const double* __restrict cr = c + base;
            const double* __restrict cyp = cr + sy;
            const double* __restrict cyn = cr - sy;
            const double* __restrict czp = cr + sz;
            const double* __restrict czn = cr - sz;
            double* __restrict mr = mu + base;
#pragma GCC ivdep
            for (size_t x = b.x0; x < b.x1; ++x) {
                const double cv = cr[x];
                const double cxx = (cr[x + 1] + cr[x - 1] - 2.0 * cv) / dx2;
                const double cyy = (cyp[x] + cyn[x] - 2.0 * cv) / dy2;
                const double czz = (czp[x] + czn[x] - 2.0 * cv) / dz2;
                const double lap = cxx + cyy + czz;
                mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update step on a box of local (padded) indices
static void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold, const double* __restrict mu,
                               const Domain& d, const Box& b,
                               const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t sy = d.sy, sz = d.sz;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    for (size_t z = b.z0; z < b.z1; ++z) {
        for (size_t y = b.y0; y < b.y1; ++y) {
            const size_t base = z * sz + y * sy;
            const double* __restrict m = mu + base;
            const double* __restrict myp = m + sy;
            const double* __restrict myn = m - sy;
            const double* __restrict mzp = m + sz;
            const double* __restrict mzn = m - sz;
            const double* __restrict co = cold + base;
            double* __restrict cn = cnew + base;
#pragma GCC ivdep
            for (size_t x = b.x0; x < b.x1; ++x) {
                const double mv = m[x];
                const double cxx = (m[x + 1] + m[x - 1] - 2.0 * mv) / dx2;
                const double cyy = (myp[x] + myn[x] - 2.0 * mv) / dy2;
                const double czz = (mzp[x] + mzn[x] - 2.0 * mv) / dz2;
                cn[x] = co[x] + dt * D * (cxx + cyy + czz);
            }
        }
    }
}

// Split the interior into an inner box (independent of ghost cells) and a
// shell of up to six boxes that depend on halo data.
static void splitBoxes(const Domain& d, Box& inner, Box* shell, int& nShell) {
    const size_t lx = d.ln[0], ly = d.ln[1], lz = d.ln[2];
    nShell = 0;
    if (lx < 3 || ly < 3 || lz < 3) {
        inner = {1, 1, 1, 1, 1, 1};
        shell[nShell++] = {1, lx + 1, 1, ly + 1, 1, lz + 1};
        return;
    }
    inner = {2, lx, 2, ly, 2, lz};
    shell[nShell++] = {1, lx + 1, 1, ly + 1, 1, 2};           // z low
    shell[nShell++] = {1, lx + 1, 1, ly + 1, lz, lz + 1};     // z high
    shell[nShell++] = {1, lx + 1, 1, 2, 2, lz};               // y low
    shell[nShell++] = {1, lx + 1, ly, ly + 1, 2, lz};         // y high
    shell[nShell++] = {1, 2, 2, ly, 2, lz};                   // x low
    shell[nShell++] = {lx, lx + 1, 2, ly, 2, lz};             // x high
}

// Initialize concentration field (local part, global pseudo-random pattern)
static void initializeConcentration(double* c, const Domain& d) {
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    const size_t vol = nx * ny * nz;
    for (size_t z = 1; z <= d.ln[2]; ++z) {
        for (size_t y = 1; y <= d.ln[1]; ++y) {
            for (size_t x = 1; x <= d.ln[0]; ++x) {
                const size_t gx = d.off[0] + x - 1, gy = d.off[1] + y - 1, gz = d.off[2] + z - 1;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[z * d.sz + y * d.sy + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather the distributed field into a global array on rank 0
static void gatherField(const Domain& d, const double* a, std::vector<double>& global) {
    const size_t localCount = d.ln[0] * d.ln[1] * d.ln[2];
    std::vector<double> buf(localCount);
    size_t p = 0;
    for (size_t z = 1; z <= d.ln[2]; ++z)
        for (size_t y = 1; y <= d.ln[1]; ++y)
            for (size_t x = 1; x <= d.ln[0]; ++x)
                buf[p++] = a[z * d.sz + y * d.sy + x];
    if (d.rank != 0) {
        MPI_Send(buf.data(), static_cast<int>(localCount), MPI_DOUBLE, 0, 99, d.comm);
        return;
    }
    const size_t nx = d.n[0], ny = d.n[1];
    global.assign(d.n[0] * d.n[1] * d.n[2], 0.0);
    std::vector<double> rbuf;
    for (int r = 0; r < d.size; ++r) {
        int cc[3];
        MPI_Cart_coords(d.comm, r, 3, cc);
        size_t ln[3], off[3];
        for (int k = 0; k < 3; ++k) blockRange(d.n[k], d.dims[k], cc[2 - k], ln[k], off[k]);
        const size_t cnt = ln[0] * ln[1] * ln[2];
        const double* src;
        if (r == 0) {
            src = buf.data();
        } else {
            rbuf.resize(cnt);
            MPI_Recv(rbuf.data(), static_cast<int>(cnt), MPI_DOUBLE, r, 99, d.comm, MPI_STATUS_IGNORE);
            src = rbuf.data();
        }
        size_t q = 0;
        for (size_t z = 0; z < ln[2]; ++z)
            for (size_t y = 0; y < ln[1]; ++y) {
                std::memcpy(&global[idx3(off[0], off[1] + y, off[2] + z, nx, ny)], src + q, ln[0] * sizeof(double));
                q += ln[0];
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
    const bool active = setupDomain(d, nx, ny, nz);
    if (!active) {
        // Idle rank: grid too small to give this rank any cells
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        int valid = 1;
        if (validate) MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validate && !valid ? 1 : 0;
    }
    const size_t localSize = d.px * d.py * d.pz;
    
    // Allocate local arrays (with ghost layers)
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    Box inner;
    Box shell[6];
    int nShell = 0;
    splitBoxes(d, inner, shell, nShell);
    
    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), d);
    
    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    MPI_Request req[12];
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential (overlap halo exchange of c with interior work)
        int nr = startHalo(d, cold.data(), req);
        fillPhysicalGhosts(d, cold.data());
        computeChemicalPotential(cold.data(), mu.data(), d, inner, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        for (int s = 0; s < nShell; ++s)
            computeChemicalPotential(cold.data(), mu.data(), d, shell[s], dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        
        // Update concentration (overlap halo exchange of mu with interior work)
        nr = startHalo(d, mu.data(), req);
        fillPhysicalGhosts(d, mu.data());
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d, inner, D, dt, dx, dy, dz);
        MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        for (int s = 0; s < nShell; ++s)
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d, shell[s], D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> global;
    if (printResults || validate) gatherField(d, cold.data(), global);
    
    // Print results for external validation
    if (printResults && root) {
        print_results(global, "Concentration");
    }
    
    int ret = 0;
    // Validation
    if (validate) {
        int valid = 1;
        if (root) {
            printf("Validating result...\n");
            valid = validateResult(global, nx, ny, nz) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ret = valid ? 0 : 1;
    }
    
    freeDomain(d);
    MPI_Finalize();
    return ret;
}
