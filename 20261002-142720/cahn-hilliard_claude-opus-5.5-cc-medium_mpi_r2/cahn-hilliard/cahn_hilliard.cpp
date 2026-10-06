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

// Block distribution of n cells over p parts
static inline size_t blockStart(const size_t c, const size_t p, const size_t n) noexcept {
    return c * (n / p) + std::min(c, n % p);
}

// Local subdomain with one ghost layer on every side.
// Ghost cells on the physical boundary hold a copy of the adjacent boundary cell,
// which reproduces the clamped boundary condition of the original stencil exactly.
struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int dims[3] = {1, 1, 1};     // process grid (z, y, x)
    int coords[3] = {0, 0, 0};   // (z, y, x)
    int nbLo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL}; // (x, y, z)
    int nbHi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    size_t n[3] = {0, 0, 0};     // local interior size (x, y, z)
    size_t off[3] = {0, 0, 0};   // global offset (x, y, z)
    size_t sx = 0, sy = 0, sz = 0; // padded sizes
    std::vector<double> sendBuf[6];
    std::vector<double> recvBuf[6];

    inline size_t at(const size_t i, const size_t j, const size_t k) const noexcept { return (k * sy + j) * sx + i; }
    size_t padded() const noexcept { return sx * sy * sz; }
};

// Choose process grid minimizing the maximal per-rank halo surface
static bool chooseDims(const int p, const size_t nx, const size_t ny, const size_t nz, int out[3]) {
    double best = -1.0;
    for (int px = 1; px <= p; ++px) {
        if (p % px) continue;
        for (int py = 1; py <= p / px; ++py) {
            if ((p / px) % py) continue;
            const int pz = p / px / py;
            if ((size_t)px > nx || (size_t)py > ny || (size_t)pz > nz) continue;
            const double lx = std::ceil((double)nx / px), ly = std::ceil((double)ny / py), lz = std::ceil((double)nz / pz);
            // x faces are strided -> slightly more expensive
            double cost = (px > 1 ? 2.0 * 1.25 * ly * lz : 0.0) + (py > 1 ? 2.0 * lx * lz : 0.0) + (pz > 1 ? 2.0 * lx * ly : 0.0);
            if (best < 0.0 || cost < best - 1e-9) {
                best = cost;
                out[0] = pz; out[1] = py; out[2] = px;
            }
        }
    }
    return best >= 0.0;
}

// Fill physical-boundary ghosts (local copy, clamped BC)
static void fillPhysicalGhosts(const Domain& d, double* __restrict f) {
    const size_t lx = d.n[0], ly = d.n[1], lz = d.n[2];
    if (d.nbLo[0] == MPI_PROC_NULL || d.nbHi[0] == MPI_PROC_NULL) {
        for (size_t k = 1; k <= lz; ++k)
            for (size_t j = 1; j <= ly; ++j) {
                double* row = f + d.at(0, j, k);
                if (d.nbLo[0] == MPI_PROC_NULL) row[0] = row[1];
                if (d.nbHi[0] == MPI_PROC_NULL) row[lx + 1] = row[lx];
            }
    }
    if (d.nbLo[1] == MPI_PROC_NULL)
        for (size_t k = 1; k <= lz; ++k) std::memcpy(f + d.at(1, 0, k), f + d.at(1, 1, k), lx * sizeof(double));
    if (d.nbHi[1] == MPI_PROC_NULL)
        for (size_t k = 1; k <= lz; ++k) std::memcpy(f + d.at(1, ly + 1, k), f + d.at(1, ly, k), lx * sizeof(double));
    if (d.nbLo[2] == MPI_PROC_NULL)
        for (size_t j = 1; j <= ly; ++j) std::memcpy(f + d.at(1, j, 0), f + d.at(1, j, 1), lx * sizeof(double));
    if (d.nbHi[2] == MPI_PROC_NULL)
        for (size_t j = 1; j <= ly; ++j) std::memcpy(f + d.at(1, j, lz + 1), f + d.at(1, j, lz), lx * sizeof(double));
}

// Face iteration helper: face f = 2*dim + side (side 0 = low, 1 = high).
// layer: index along dim (interior layer for send, ghost layer for recv).
template <typename Fn>
static inline void forFace(const Domain& d, const int dim, const size_t layer, Fn&& fn) {
    const size_t lx = d.n[0], ly = d.n[1], lz = d.n[2];
    size_t m = 0;
    if (dim == 0) {
        for (size_t k = 1; k <= lz; ++k)
            for (size_t j = 1; j <= ly; ++j) fn(m++, d.at(layer, j, k));
    } else if (dim == 1) {
        for (size_t k = 1; k <= lz; ++k)
            for (size_t i = 1; i <= lx; ++i) fn(m++, d.at(i, layer, k));
    } else {
        for (size_t j = 1; j <= ly; ++j)
            for (size_t i = 1; i <= lx; ++i) fn(m++, d.at(i, j, layer));
    }
}

// Start non-blocking halo exchange of field f
static int startHalo(Domain& d, double* f, MPI_Request* reqs) {
    int nr = 0;
    for (int dim = 0; dim < 3; ++dim) {
        for (int side = 0; side < 2; ++side) {
            const int nb = side ? d.nbHi[dim] : d.nbLo[dim];
            if (nb == MPI_PROC_NULL) continue;
            const int face = 2 * dim + side;
            // message received from low neighbour was sent "upwards" (tag 2*dim+1)
            MPI_Irecv(d.recvBuf[face].data(), (int)d.recvBuf[face].size(), MPI_DOUBLE, nb,
                      2 * dim + (side ? 0 : 1), d.comm, &reqs[nr++]);
        }
    }
    for (int dim = 0; dim < 3; ++dim) {
        for (int side = 0; side < 2; ++side) {
            const int nb = side ? d.nbHi[dim] : d.nbLo[dim];
            if (nb == MPI_PROC_NULL) continue;
            const int face = 2 * dim + side;
            const size_t layer = side ? d.n[dim] : 1;
            double* buf = d.sendBuf[face].data();
            forFace(d, dim, layer, [&](size_t m, size_t id) { buf[m] = f[id]; });
            MPI_Isend(buf, (int)d.sendBuf[face].size(), MPI_DOUBLE, nb, 2 * dim + side, d.comm, &reqs[nr++]);
        }
    }
    return nr;
}

static void finishHalo(Domain& d, double* f, MPI_Request* reqs, const int nr) {
    MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
    for (int dim = 0; dim < 3; ++dim) {
        for (int side = 0; side < 2; ++side) {
            const int nb = side ? d.nbHi[dim] : d.nbLo[dim];
            if (nb == MPI_PROC_NULL) continue;
            const int face = 2 * dim + side;
            const size_t layer = side ? d.n[dim] + 1 : 0;
            const double* buf = d.recvBuf[face].data();
            forFace(d, dim, layer, [&](size_t m, size_t id) { f[id] = buf[m]; });
        }
    }
}

struct Params {
    double dx, dy, dz, dt, gamma, e_AA, e_BB, e_AB, D;
};

// Chemical potential on box [i0,i1) x [j0,j1) x [k0,k1) (local padded coordinates)
static void muBox(const Domain& d, const double* __restrict c, double* __restrict mu, const Params& p,
                  size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1) {
    const size_t sx = d.sx, sxy = d.sx * d.sy;
    const double dx2 = p.dx * p.dx, dy2 = p.dy * p.dy, dz2 = p.dz * p.dz;
    for (size_t k = k0; k < k1; ++k)
        for (size_t j = j0; j < j1; ++j) {
            const size_t base = d.at(0, j, k);
            const double* __restrict cc = c + base;
#pragma GCC ivdep
            for (size_t i = i0; i < i1; ++i) {
                const double cv = cc[i];
                const double cxx = (cc[i + 1] + cc[i - 1] - 2.0 * cv) / dx2;
                const double cyy = (cc[i + sx] + cc[i - sx] - 2.0 * cv) / dy2;
                const double czz = (cc[i + sxy] + cc[i - sxy] - 2.0 * cv) / dz2;
                mu[base + i] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                             + 3.0 * cv + cv * cv * cv
                             - p.gamma * (cxx + cyy + czz);
            }
        }
}

// Concentration update on box
static void updBox(const Domain& d, double* __restrict cnew, const double* __restrict cold,
                   const double* __restrict mu, const Params& p,
                   size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1) {
    const size_t sx = d.sx, sxy = d.sx * d.sy;
    const double dx2 = p.dx * p.dx, dy2 = p.dy * p.dy, dz2 = p.dz * p.dz;
    for (size_t k = k0; k < k1; ++k)
        for (size_t j = j0; j < j1; ++j) {
            const size_t base = d.at(0, j, k);
            const double* __restrict m = mu + base;
#pragma GCC ivdep
            for (size_t i = i0; i < i1; ++i) {
                const double mv = m[i];
                const double cxx = (m[i + 1] + m[i - 1] - 2.0 * mv) / dx2;
                const double cyy = (m[i + sx] + m[i - sx] - 2.0 * mv) / dy2;
                const double czz = (m[i + sxy] + m[i - sxy] - 2.0 * mv) / dz2;
                cnew[base + i] = cold[base + i] + p.dt * p.D * (cxx + cyy + czz);
            }
        }
}

// Apply kernel over the whole interior, overlapping the halo exchange of `f`
// with computation of cells that do not depend on remote ghosts.
template <typename Kernel>
static void haloOverlapped(Domain& d, double* f, Kernel&& kern) {
    MPI_Request reqs[12];
    const int nr = startHalo(d, f, reqs);
    fillPhysicalGhosts(d, f);

    size_t lo[3], hi[3]; // inner box [lo, hi) not touching remote ghosts
    bool innerEmpty = false;
    for (int dim = 0; dim < 3; ++dim) {
        lo[dim] = 1 + (d.nbLo[dim] != MPI_PROC_NULL ? 1 : 0);
        hi[dim] = d.n[dim] + 1 - (d.nbHi[dim] != MPI_PROC_NULL ? 1 : 0);
        if (hi[dim] <= lo[dim]) innerEmpty = true;
    }
    const size_t X = d.n[0] + 1, Y = d.n[1] + 1, Z = d.n[2] + 1;

    if (innerEmpty || nr == 0) {
        finishHalo(d, f, reqs, nr);
        kern(1, X, 1, Y, 1, Z);
        return;
    }
    kern(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
    finishHalo(d, f, reqs, nr);
    // shell: z slabs, then y slabs, then x columns
    if (lo[2] > 1) kern(1, X, 1, Y, 1, lo[2]);
    if (hi[2] < Z) kern(1, X, 1, Y, hi[2], Z);
    if (lo[1] > 1) kern(1, X, 1, lo[1], lo[2], hi[2]);
    if (hi[1] < Y) kern(1, X, hi[1], Y, lo[2], hi[2]);
    if (lo[0] > 1) kern(1, lo[0], lo[1], hi[1], lo[2], hi[2]);
    if (hi[0] < X) kern(hi[0], X, lo[1], hi[1], lo[2], hi[2]);
}

// Initialize concentration field (local block, global pseudo-random pattern)
void initializeConcentration(const Domain& d, std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t k = 1; k <= d.n[2]; ++k) {
        const size_t z = d.off[2] + k - 1;
        for (size_t j = 1; j <= d.n[1]; ++j) {
            const size_t y = d.off[1] + j - 1;
            for (size_t i = 1; i <= d.n[0]; ++i) {
                const size_t x = d.off[0] + i - 1;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[d.at(i, j, k)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather full global field on rank 0 (natural x-fastest ordering)
static std::vector<double> gatherGlobal(const Domain& d, const std::vector<double>& c,
                                        const size_t nx, const size_t ny, const size_t nz) {
    int nprocs;
    MPI_Comm_size(d.comm, &nprocs);
    std::vector<double> local(d.n[0] * d.n[1] * d.n[2]);
    size_t m = 0;
    for (size_t k = 1; k <= d.n[2]; ++k)
        for (size_t j = 1; j <= d.n[1]; ++j)
            for (size_t i = 1; i <= d.n[0]; ++i) local[m++] = c[d.at(i, j, k)];

    std::vector<double> global;
    if (d.rank != 0) {
        MPI_Send(local.data(), (int)local.size(), MPI_DOUBLE, 0, 99, d.comm);
        return global;
    }
    global.resize(nx * ny * nz);
    std::vector<double> buf;
    for (int r = 0; r < nprocs; ++r) {
        int rc[3];
        MPI_Cart_coords(d.comm, r, 3, rc);
        size_t o[3], n[3];
        const size_t N[3] = {nx, ny, nz};
        for (int dim = 0; dim < 3; ++dim) {
            const int pc = rc[2 - dim], pd = d.dims[2 - dim];
            o[dim] = blockStart(pc, pd, N[dim]);
            n[dim] = blockStart(pc + 1, pd, N[dim]) - o[dim];
        }
        const double* src;
        if (r == 0) {
            src = local.data();
        } else {
            buf.resize(n[0] * n[1] * n[2]);
            MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 99, d.comm, MPI_STATUS_IGNORE);
            src = buf.data();
        }
        size_t q = 0;
        for (size_t k = 0; k < n[2]; ++k)
            for (size_t j = 0; j < n[1]; ++j) {
                std::memcpy(&global[idx3(o[0], o[1] + j, o[2] + k, nx, ny)], src + q, n[0] * sizeof(double));
                q += n[0];
            }
    }
    return global;
}

bool validateResult(const Domain& d, const std::vector<double>& c, const bool active, const int worldRank) {
    // Check for NaN or Inf
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    if (active) {
        for (size_t k = 1; k <= d.n[2]; ++k)
            for (size_t j = 1; j <= d.n[1]; ++j)
                for (size_t i = 1; i <= d.n[0]; ++i) {
                    const double val = c[d.at(i, j, k)];
                    if (std::isnan(val) || std::isinf(val)) bad = 1;
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (bad) {
        if (worldRank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (worldRank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        if (worldRank == 0) printf("Validation failed: values out of expected range\n");
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

static int runMain(int argc, char** argv, const int worldRank, const int worldSize) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    const bool root = (worldRank == 0);
    
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
    Params p;
    p.dx = 1.0;
    p.dy = 1.0;
    p.dz = 1.0;
    p.dt = 0.01;
    p.e_AA = -(2.0 / 9.0);
    p.e_BB = -(2.0 / 9.0);
    p.e_AB = (2.0 / 9.0);
    p.gamma = 0.5;
    p.D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    
    // Domain decomposition: use the largest number of ranks that fits the grid
    Domain d;
    int dims[3] = {1, 1, 1};
    int nActive = worldSize;
    while (nActive > 1 && !chooseDims(nActive, nx, ny, nz, dims)) --nActive;
    if (nActive == 1) { dims[0] = dims[1] = dims[2] = 1; }
    const bool active = worldRank < nActive;
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    
    std::vector<double> cold, cnew, mu;
    if (active) {
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(activeComm, 3, dims, periods, 0, &d.comm);
        MPI_Comm_free(&activeComm);
        MPI_Comm_rank(d.comm, &d.rank);
        MPI_Cart_coords(d.comm, d.rank, 3, d.coords);
        for (int k = 0; k < 3; ++k) d.dims[k] = dims[k];
        const size_t N[3] = {nx, ny, nz};
        for (int dim = 0; dim < 3; ++dim) {
            const int cd = 2 - dim; // cart dimension index (z, y, x ordering)
            d.off[dim] = blockStart(d.coords[cd], dims[cd], N[dim]);
            d.n[dim] = blockStart(d.coords[cd] + 1, dims[cd], N[dim]) - d.off[dim];
            MPI_Cart_shift(d.comm, cd, 1, &d.nbLo[dim], &d.nbHi[dim]);
        }
        d.sx = d.n[0] + 2; d.sy = d.n[1] + 2; d.sz = d.n[2] + 2;
        const size_t faceSize[3] = {d.n[1] * d.n[2], d.n[0] * d.n[2], d.n[0] * d.n[1]};
        for (int f = 0; f < 6; ++f) {
            d.sendBuf[f].resize(faceSize[f / 2]);
            d.recvBuf[f].resize(faceSize[f / 2]);
        }
        
        // Allocate arrays (local block with ghost layer)
        cold.assign(d.padded(), 0.0);
        cnew.assign(d.padded(), 0.0);
        mu.assign(d.padded(), 0.0);
    }
    
    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    if (active) initializeConcentration(d, cold, nx, ny, nz);
    
    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (active) {
        for (int t = 0; t < iterations; ++t) {
            double* c = cold.data();
            double* m = mu.data();
            double* cn = cnew.data();
            // Compute chemical potential
            haloOverlapped(d, c, [&](size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1) {
                muBox(d, c, m, p, i0, i1, j0, j1, k0, k1);
            });
            
            // Update concentration
            haloOverlapped(d, m, [&](size_t i0, size_t i1, size_t j0, size_t j1, size_t k0, size_t k1) {
                updBox(d, cn, c, m, p, i0, i1, j0, j1, k0, k1);
            });
            
            // Swap buffers
            std::swap(cold, cnew);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    const long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    if (active) MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, d.comm);
    auto duration = std::chrono::milliseconds(maxDuration);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults && active) {
        std::vector<double> global = gatherGlobal(d, cold, nx, ny, nz);
        if (root) print_results(global, "Concentration");
    }
    
    // Validation
    int ret = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateResult(d, cold, active, worldRank);
        
        if (valid) {
            if (root) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            ret = 1;
        }
    }
    
    if (active) MPI_Comm_free(&d.comm);
    return ret;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const int ret = runMain(argc, argv, worldRank, worldSize);
    fflush(stdout);
    MPI_Finalize();
    return ret;
}
