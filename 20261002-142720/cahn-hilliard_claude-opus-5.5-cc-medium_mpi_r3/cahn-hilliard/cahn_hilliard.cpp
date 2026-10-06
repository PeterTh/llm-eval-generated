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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local subdomain of a 3D Cartesian block decomposition.
// Fields are stored with a one-cell ghost layer on every side; ghost cells either hold
// neighbor-rank data or (at the global boundary) a copy of the adjacent boundary cell,
// which reproduces the clamped boundary condition of the original stencil exactly.
struct Domain {
    MPI_Comm comm;
    int rank, size;
    int dims[3], coords[3];
    int nbrLo[3], nbrHi[3];          // neighbor ranks in x, y, z (MPI_PROC_NULL at boundary)
    size_t n[3];                     // global sizes
    size_t l[3];                     // local sizes
    size_t off[3];                   // global offsets
    size_t sx, sxy;                  // strides of the padded array
    size_t padded;                   // padded element count
    MPI_Datatype face[3];            // face datatypes (one layer, interior extent)
};

static void blockRange(size_t n, int p, int c, size_t& len, size_t& off) {
    const size_t base = n / p, rem = n % p;
    len = base + (static_cast<size_t>(c) < rem ? 1 : 0);
    off = c * base + std::min<size_t>(c, rem);
}

// Choose a process grid minimizing halo communication volume
static void chooseDims(int P, const size_t n[3], int dims[3]) {
    double best = std::numeric_limits<double>::max();
    dims[0] = dims[1] = 1; dims[2] = P;
    bool found = false;
    for (int px = 1; px <= P; ++px) {
        if (P % px) continue;
        for (int py = 1; py <= P / px; ++py) {
            if ((P / px) % py) continue;
            const int pz = P / px / py;
            if (static_cast<size_t>(px) > n[0] || static_cast<size_t>(py) > n[1] || static_cast<size_t>(pz) > n[2]) continue;
            const double lx = std::ceil(double(n[0]) / px), ly = std::ceil(double(n[1]) / py), lz = std::ceil(double(n[2]) / pz);
            // x-faces are strided (more expensive), weight them slightly higher
            double cost = 0.0;
            if (px > 1) cost += 1.5 * ly * lz;
            if (py > 1) cost += 1.1 * lx * lz;
            if (pz > 1) cost += 1.0 * lx * ly;
            if (!found || cost < best - 1e-9) { best = cost; dims[0] = px; dims[1] = py; dims[2] = pz; found = true; }
        }
    }
    if (!found) {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) printf("Error: too many MPI ranks (%d) for grid size\n", P);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void setupDomain(Domain& d, size_t nx, size_t ny, size_t nz) {
    int P; MPI_Comm_size(MPI_COMM_WORLD, &P);
    d.n[0] = nx; d.n[1] = ny; d.n[2] = nz;
    // MPI Cartesian dims are ordered slowest-first: (z, y, x)
    int dimsXYZ[3];
    chooseDims(P, d.n, dimsXYZ);
    int cdims[3] = {dimsXYZ[2], dimsXYZ[1], dimsXYZ[0]};
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, cdims, periods, 1, &d.comm);
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Comm_size(d.comm, &d.size);
    int cc[3];
    MPI_Cart_coords(d.comm, d.rank, 3, cc);
    for (int a = 0; a < 3; ++a) {
        d.dims[a] = dimsXYZ[a];
        d.coords[a] = cc[2 - a];
        MPI_Cart_shift(d.comm, 2 - a, 1, &d.nbrLo[a], &d.nbrHi[a]);
        blockRange(d.n[a], d.dims[a], d.coords[a], d.l[a], d.off[a]);
    }
    d.sx = d.l[0] + 2;
    d.sxy = d.sx * (d.l[1] + 2);
    d.padded = d.sxy * (d.l[2] + 2);

    const int sizes[3] = {int(d.l[2] + 2), int(d.l[1] + 2), int(d.l[0] + 2)};
    const int starts[3] = {0, 0, 0};
    const int subX[3] = {int(d.l[2]), int(d.l[1]), 1};
    const int subY[3] = {int(d.l[2]), 1, int(d.l[0])};
    const int subZ[3] = {1, int(d.l[1]), int(d.l[0])};
    MPI_Type_create_subarray(3, sizes, subX, starts, MPI_ORDER_C, MPI_DOUBLE, &d.face[0]);
    MPI_Type_create_subarray(3, sizes, subY, starts, MPI_ORDER_C, MPI_DOUBLE, &d.face[1]);
    MPI_Type_create_subarray(3, sizes, subZ, starts, MPI_ORDER_C, MPI_DOUBLE, &d.face[2]);
    for (int a = 0; a < 3; ++a) MPI_Type_commit(&d.face[a]);
}

// Padded-array index for local interior coordinates (may be -1 .. l)
static inline size_t pidx(const Domain& d, long x, long y, long z) {
    return size_t(z + 1) * d.sxy + size_t(y + 1) * d.sx + size_t(x + 1);
}

// Start non-blocking exchange of all six faces (7-point stencil needs no edges/corners)
static void startHalo(const Domain& d, double* f, MPI_Request req[12]) {
    int nr = 0;
    const long lx = d.l[0], ly = d.l[1], lz = d.l[2];
    for (int a = 0; a < 3; ++a) {
        long lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0}, glo[3] = {0, 0, 0}, ghi[3] = {0, 0, 0};
        const long la = (a == 0) ? lx : (a == 1) ? ly : lz;
        lo[a] = 0; hi[a] = la - 1; glo[a] = -1; ghi[a] = la;
        // receive into ghosts
        MPI_Irecv(f + pidx(d, glo[0], glo[1], glo[2]), 1, d.face[a], d.nbrLo[a], 2 * a, d.comm, &req[nr++]);
        MPI_Irecv(f + pidx(d, ghi[0], ghi[1], ghi[2]), 1, d.face[a], d.nbrHi[a], 2 * a + 1, d.comm, &req[nr++]);
        // send boundary layers
        MPI_Isend(f + pidx(d, lo[0], lo[1], lo[2]), 1, d.face[a], d.nbrLo[a], 2 * a + 1, d.comm, &req[nr++]);
        MPI_Isend(f + pidx(d, hi[0], hi[1], hi[2]), 1, d.face[a], d.nbrHi[a], 2 * a, d.comm, &req[nr++]);
    }
}

// Fill ghost layers at the physical boundary with copies of the boundary cells (clamping)
static void fillPhysicalGhosts(const Domain& d, double* f) {
    const long lx = d.l[0], ly = d.l[1], lz = d.l[2];
    if (d.nbrLo[2] == MPI_PROC_NULL)
        for (long y = 0; y < ly; ++y) for (long x = 0; x < lx; ++x) f[pidx(d, x, y, -1)] = f[pidx(d, x, y, 0)];
    if (d.nbrHi[2] == MPI_PROC_NULL)
        for (long y = 0; y < ly; ++y) for (long x = 0; x < lx; ++x) f[pidx(d, x, y, lz)] = f[pidx(d, x, y, lz - 1)];
    if (d.nbrLo[1] == MPI_PROC_NULL)
        for (long z = 0; z < lz; ++z) for (long x = 0; x < lx; ++x) f[pidx(d, x, -1, z)] = f[pidx(d, x, 0, z)];
    if (d.nbrHi[1] == MPI_PROC_NULL)
        for (long z = 0; z < lz; ++z) for (long x = 0; x < lx; ++x) f[pidx(d, x, ly, z)] = f[pidx(d, x, ly - 1, z)];
    if (d.nbrLo[0] == MPI_PROC_NULL)
        for (long z = 0; z < lz; ++z) for (long y = 0; y < ly; ++y) f[pidx(d, -1, y, z)] = f[pidx(d, 0, y, z)];
    if (d.nbrHi[0] == MPI_PROC_NULL)
        for (long z = 0; z < lz; ++z) for (long y = 0; y < ly; ++y) f[pidx(d, lx, y, z)] = f[pidx(d, lx - 1, y, z)];
}

struct Box { long x0, x1, y0, y1, z0, z1; };

// Chemical potential over a box of local interior coordinates
static void muBox(const Domain& d, const Box& b, const double* __restrict c, double* __restrict mu,
                  const double dx, const double dy, const double dz,
                  const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t sx = d.sx, sxy = d.sxy;
    const double idx2 = dx * dx, idy2 = dy * dy, idz2 = dz * dz;
    for (long z = b.z0; z < b.z1; ++z) {
        for (long y = b.y0; y < b.y1; ++y) {
            const size_t row = pidx(d, 0, y, z);
            const double* __restrict cr = c + row;
            double* __restrict mr = mu + row;
#pragma omp simd
            for (long x = b.x0; x < b.x1; ++x) {
                const double cv = cr[x];
                const double cxx = (cr[x + 1] + cr[x - 1] - 2.0 * cv) / idx2;
                const double cyy = (cr[x + sx] + cr[x - sx] - 2.0 * cv) / idy2;
                const double czz = (cr[x + sxy] + cr[x - sxy] - 2.0 * cv) / idz2;
                mr[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update over a box of local interior coordinates
static void updateBox(const Domain& d, const Box& b, double* __restrict cnew, const double* __restrict cold,
                      const double* __restrict mu, const double D, const double dt,
                      const double dx, const double dy, const double dz) {
    const size_t sx = d.sx, sxy = d.sxy;
    const double idx2 = dx * dx, idy2 = dy * dy, idz2 = dz * dz;
    for (long z = b.z0; z < b.z1; ++z) {
        for (long y = b.y0; y < b.y1; ++y) {
            const size_t row = pidx(d, 0, y, z);
            const double* __restrict mr = mu + row;
            const double* __restrict co = cold + row;
            double* __restrict cn = cnew + row;
#pragma omp simd
            for (long x = b.x0; x < b.x1; ++x) {
                const double mv = mr[x];
                const double cxx = (mr[x + 1] + mr[x - 1] - 2.0 * mv) / idx2;
                const double cyy = (mr[x + sx] + mr[x - sx] - 2.0 * mv) / idy2;
                const double czz = (mr[x + sxy] + mr[x - sxy] - 2.0 * mv) / idz2;
                cn[x] = co[x] + dt * D * (cxx + cyy + czz);
            }
        }
    }
}

// Split local domain into an interior box (independent of ghost cells) and boundary shells
static void makeBoxes(const Domain& d, Box& inner, std::vector<Box>& shells) {
    const long lx = d.l[0], ly = d.l[1], lz = d.l[2];
    const long ix0 = std::min(1L, lx), ix1 = std::max(ix0, lx - 1);
    const long iy0 = std::min(1L, ly), iy1 = std::max(iy0, ly - 1);
    const long iz0 = std::min(1L, lz), iz1 = std::max(iz0, lz - 1);
    inner = {ix0, ix1, iy0, iy1, iz0, iz1};
    shells.clear();
    auto add = [&](Box b) { if (b.x1 > b.x0 && b.y1 > b.y0 && b.z1 > b.z0) shells.push_back(b); };
    add({0, lx, 0, ly, 0, iz0});
    add({0, lx, 0, ly, iz1, lz});
    add({0, lx, 0, iy0, iz0, iz1});
    add({0, lx, iy1, ly, iz0, iz1});
    add({0, ix0, iy0, iy1, iz0, iz1});
    add({ix1, lx, iy0, iy1, iz0, iz1});
}

// Initialize concentration field (local part, using global linear ids)
void initializeConcentration(const Domain& d, std::vector<double>& c) {
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < d.l[2]; ++z) {
        for (size_t y = 0; y < d.l[1]; ++y) {
            for (size_t x = 0; x < d.l[0]; ++x) {
                const size_t gx = x + d.off[0], gy = y + d.off[1], gz = z + d.off[2];
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[pidx(d, x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather the distributed field into a global array on rank 0
static void gatherField(const Domain& d, const std::vector<double>& c, std::vector<double>& global) {
    const size_t lx = d.l[0], ly = d.l[1], lz = d.l[2];
    std::vector<double> buf(lx * ly * lz);
    size_t k = 0;
    for (size_t z = 0; z < lz; ++z)
        for (size_t y = 0; y < ly; ++y)
            for (size_t x = 0; x < lx; ++x) buf[k++] = c[pidx(d, x, y, z)];

    if (d.rank != 0) {
        size_t sent = 0;
        while (sent < buf.size()) {
            const int cnt = int(std::min<size_t>(buf.size() - sent, size_t(1) << 30));
            MPI_Send(buf.data() + sent, cnt, MPI_DOUBLE, 0, 100, d.comm);
            sent += cnt;
        }
        return;
    }
    const size_t nx = d.n[0], ny = d.n[1];
    global.assign(nx * ny * d.n[2], 0.0);
    std::vector<double> rbuf;
    for (int r = 0; r < d.size; ++r) {
        int cc[3];
        MPI_Cart_coords(d.comm, r, 3, cc);
        size_t rl[3], ro[3];
        for (int a = 0; a < 3; ++a) blockRange(d.n[a], d.dims[a], cc[2 - a], rl[a], ro[a]);
        const size_t cnt = rl[0] * rl[1] * rl[2];
        const double* src;
        if (r == 0) {
            src = buf.data();
        } else {
            rbuf.resize(cnt);
            size_t got = 0;
            while (got < cnt) {
                const int m = int(std::min<size_t>(cnt - got, size_t(1) << 30));
                MPI_Recv(rbuf.data() + got, m, MPI_DOUBLE, r, 100, d.comm, MPI_STATUS_IGNORE);
                got += m;
            }
            src = rbuf.data();
        }
        size_t q = 0;
        for (size_t z = 0; z < rl[2]; ++z)
            for (size_t y = 0; y < rl[1]; ++y) {
                double* dst = &global[idx3(ro[0], ro[1] + y, ro[2] + z, nx, ny)];
                std::memcpy(dst, src + q, rl[0] * sizeof(double));
                q += rl[0];
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

static int run(int argc, char** argv) {
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
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
    
    // Allocate arrays (local, with ghost layers)
    std::vector<double> cold(d.padded, 0.0);
    std::vector<double> cnew(d.padded, 0.0);
    std::vector<double> mu(d.padded, 0.0);

    Box inner;
    std::vector<Box> shells;
    makeBoxes(d, inner, shells);
    
    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(d, cold);
    
    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    MPI_Request req[12];
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential (overlap halo exchange of c with interior work)
        startHalo(d, cold.data(), req);
        fillPhysicalGhosts(d, cold.data());
        muBox(d, inner, cold.data(), mu.data(), dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
        for (const Box& b : shells)
            muBox(d, b, cold.data(), mu.data(), dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        
        // Update concentration (overlap halo exchange of mu with interior work)
        startHalo(d, mu.data(), req);
        fillPhysicalGhosts(d, mu.data());
        updateBox(d, inner, cnew.data(), cold.data(), mu.data(), D, dt, dx, dy, dz);
        MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
        for (const Box& b : shells)
            updateBox(d, b, cnew.data(), cold.data(), mu.data(), D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(d.comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
    
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int ret = 0;
    if (printResults || validate) {
        std::vector<double> global;
        gatherField(d, cold, global);
        if (d.rank == 0) {
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

    for (int a = 0; a < 3; ++a) MPI_Type_free(&d.face[a]);
    MPI_Comm_free(&d.comm);
    return ret;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int ret = run(argc, argv);
    MPI_Finalize();
    return ret;
}
