#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// Distributed-memory version: the z-axis is split into contiguous slabs, one
// per MPI rank. Each slab is stored with kGhost ghost planes on each side.
// Local plane k (-kGhost <= k < nzl + kGhost) starts at offset (k + kGhost) * plane.
// Because z is the slowest-varying index, the owned planes of all ranks, taken
// in rank order, form the global field in linear order. This lets the final
// gather produce exactly the serial array, so output and hash are identical.

namespace {

constexpr std::ptrdiff_t kGhost = 2;  // ghost planes per side (one exchange per step)
constexpr int kTagUp = 0;             // message travelling from lower rank to upper rank
constexpr int kTagDown = 1;           // message travelling from upper rank to lower rank

struct Params {
    double dx, dy, dz;
    double dt;
    double e_AA, e_BB, e_AB;
    double gamma;
    double D;
};

// Description of the z-slab owned by one rank.
struct Slab {
    size_t z0 = 0;              // first owned global z-plane
    size_t nzl = 0;             // number of owned z-planes (0 for idle ranks)
    bool first = false;         // owns global z = 0 (clamped boundary)
    bool last = false;          // owns global z = nz-1 (clamped boundary)
    int up = -1;                // rank holding the next slab, or -1
    int down = -1;              // rank holding the previous slab, or -1
};

// Number of active ranks. Each active rank owns at least 2 planes so that
// the two-plane halo can always be sent from the owned region.
int activeRanks(size_t nz, int size) {
    if (nz < 2) return 1;
    const size_t limit = nz / 2;
    return static_cast<int>(std::min<size_t>(static_cast<size_t>(size), limit));
}

Slab makeSlab(int rank, int P, size_t nz) {
    Slab s;
    if (rank >= P) return s;  // idle rank: owns nothing
    const size_t base = nz / static_cast<size_t>(P);
    const size_t rem = nz % static_cast<size_t>(P);
    const size_t r = static_cast<size_t>(rank);
    s.nzl = base + (r < rem ? 1 : 0);
    s.z0 = r * base + std::min(r, rem);
    s.first = (rank == 0);
    s.last = (rank == P - 1);
    s.up = (rank + 1 < P) ? rank + 1 : -1;
    s.down = (rank > 0) ? rank - 1 : -1;
    return s;
}

template <class T>
inline T* planePtr(T* base, size_t plane, std::ptrdiff_t k) {
    return base + (k + kGhost) * static_cast<std::ptrdiff_t>(plane);
}

// 7-point Laplacian from the neighbour values of one cell. The operation
// order matches the serial reference code exactly.
inline double lapFromNeighbours(double cv, double xp, double xn, double yp, double yn,
                                double zp, double zn,
                                double dx, double dy, double dz) {
    const double cxx = (xp + xn - 2.0 * cv) / (dx * dx);
    const double cyy = (yp + yn - 2.0 * cv) / (dy * dy);
    const double czz = (zp + zn - 2.0 * cv) / (dz * dz);
    return cxx + cyy + czz;
}

// Apply f(x, lap(x)) across one xy-row. The x-direction clamping only affects
// the two edge columns, so the interior loop is branch-free and vectorizes.
// Row pointers: c (this row), cyp/cyn (clamped y-neighbour rows), zp/zn (z-neighbour rows).
template <class F>
inline void forEachLaplacian(const double* __restrict c, const double* __restrict cyp,
                             const double* __restrict cyn, const double* __restrict zp,
                             const double* __restrict zn, size_t nx,
                             double dx, double dy, double dz, F&& f) {
    if (nx == 1) {
        f(size_t{0}, lapFromNeighbours(c[0], c[0], c[0], cyp[0], cyn[0], zp[0], zn[0], dx, dy, dz));
        return;
    }
    f(size_t{0}, lapFromNeighbours(c[0], c[1], c[0], cyp[0], cyn[0], zp[0], zn[0], dx, dy, dz));
    for (size_t x = 1; x + 1 < nx; ++x) {
        f(x, lapFromNeighbours(c[x], c[x + 1], c[x - 1], cyp[x], cyn[x], zp[x], zn[x], dx, dy, dz));
    }
    const size_t xl = nx - 1;
    f(xl, lapFromNeighbours(c[xl], c[xl], c[xl - 1], cyp[xl], cyn[xl], zp[xl], zn[xl], dx, dy, dz));
}

// Chemical potential on local planes [kBegin, kEnd). Reads c at planes k-1..k+1.
void computeChemicalPotential(const double* c, double* mu, size_t nx, size_t ny, size_t plane,
                              std::ptrdiff_t kBegin, std::ptrdiff_t kEnd, const Params& p) {
    for (std::ptrdiff_t k = kBegin; k < kEnd; ++k) {
        const double* cz = planePtr(c, plane, k);
        const double* czp = cz + plane;
        const double* czn = cz - plane;
        double* m = planePtr(mu, plane, k);
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* rc = cz + y * nx;
            double* rm = m + y * nx;
            forEachLaplacian(rc, cz + yp * nx, cz + yn * nx, czp + y * nx, czn + y * nx, nx,
                             p.dx, p.dy, p.dz, [&](size_t x, double lap) {
                const double cv = rc[x];
                rm[x] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - p.gamma * lap;
            });
        }
    }
}

// Cahn-Hilliard update on local planes [kBegin, kEnd). Reads mu at planes k-1..k+1.
void cahnHilliardUpdate(const double* cold, double* cnew, const double* mu,
                        size_t nx, size_t ny, size_t plane,
                        std::ptrdiff_t kBegin, std::ptrdiff_t kEnd, const Params& p) {
    for (std::ptrdiff_t k = kBegin; k < kEnd; ++k) {
        const double* mz = planePtr(mu, plane, k);
        const double* mzp = mz + plane;
        const double* mzn = mz - plane;
        const double* co = planePtr(cold, plane, k);
        double* cn = planePtr(cnew, plane, k);
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* rm = mz + y * nx;
            const double* ro = co + y * nx;
            double* rn = cn + y * nx;
            forEachLaplacian(rm, mz + yp * nx, mz + yn * nx, mzp + y * nx, mzn + y * nx, nx,
                             p.dx, p.dy, p.dz, [&](size_t x, double lap) {
                rn[x] = ro[x] + p.dt * p.D * lap;
            });
        }
    }
}

// Fill ghost planes that lie beyond the global z-boundary by replicating the
// boundary plane. This reproduces the serial clamped boundary exactly.
void replicateBoundaryGhosts(double* a, size_t plane, std::ptrdiff_t width, const Slab& s) {
    const std::ptrdiff_t nzl = static_cast<std::ptrdiff_t>(s.nzl);
    const size_t n = plane;
    if (s.first) {
        const double* src = planePtr(a, plane, 0);
        for (std::ptrdiff_t g = 1; g <= width; ++g) {
            std::copy(src, src + n, planePtr(a, plane, -g));
        }
    }
    if (s.last) {
        const double* src = planePtr(a, plane, nzl - 1);
        for (std::ptrdiff_t g = 0; g < width; ++g) {
            std::copy(src, src + n, planePtr(a, plane, nzl + g));
        }
    }
}

// Post nonblocking exchange of kGhost planes with both neighbours.
// Send/receive buffers are contiguous, so no packing is needed.
int postHaloExchange(double* c, size_t plane, MPI_Datatype planeType, const Slab& s,
                     MPI_Request* reqs) {
    const std::ptrdiff_t nzl = static_cast<std::ptrdiff_t>(s.nzl);
    int n = 0;
    if (s.down >= 0) {
        MPI_Irecv(planePtr(c, plane, -kGhost), static_cast<int>(kGhost), planeType,
                  s.down, kTagUp, MPI_COMM_WORLD, &reqs[n++]);
        MPI_Isend(planePtr(c, plane, 0), static_cast<int>(kGhost), planeType,
                  s.down, kTagDown, MPI_COMM_WORLD, &reqs[n++]);
    }
    if (s.up >= 0) {
        MPI_Irecv(planePtr(c, plane, nzl), static_cast<int>(kGhost), planeType,
                  s.up, kTagDown, MPI_COMM_WORLD, &reqs[n++]);
        MPI_Isend(planePtr(c, plane, nzl - kGhost), static_cast<int>(kGhost), planeType,
                  s.up, kTagUp, MPI_COMM_WORLD, &reqs[n++]);
    }
    return n;
}

// Initialize concentration on the owned planes. Uses global linear ids so the
// field matches the serial initialization.
void initializeConcentration(double* c, size_t nx, size_t ny, const Slab& s, size_t vol) {
    const size_t plane = nx * ny;
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(s.nzl); ++k) {
        double* cz = planePtr(c, plane, k);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = (s.z0 + static_cast<size_t>(k)) * plane + y * nx + x;
                // Pseudo-random value in [-1, 1]
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cz[y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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

int runBenchmark(int argc, char** argv, int rank, int size) {
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Error: grid dimensions must be positive\n");
        return 1;
    }

    if (root) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Physical parameters
    const Params p{
        1.0,            // dx
        1.0,            // dy
        1.0,            // dz
        0.01,           // dt
        -(2.0 / 9.0),   // e_AA
        -(2.0 / 9.0),   // e_BB
        (2.0 / 9.0),    // e_AB
        0.5,            // gamma
        1.0             // D
    };

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    // Domain decomposition over z
    const int P = activeRanks(nz, size);
    const Slab slab = makeSlab(rank, P, nz);
    const bool active = (rank < P);

    // MPI datatype: one full xy-plane (contiguous in memory)
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Local storage (ghost planes included). Idle ranks keep an allocation
    // so that data pointers are always valid.
    const size_t localPlanes = slab.nzl + 2 * static_cast<size_t>(kGhost);
    std::vector<double> cold(localPlanes * plane, 0.0);
    std::vector<double> cnew(localPlanes * plane, 0.0);
    std::vector<double> mu(localPlanes * plane, 0.0);

    if (root) printf("Initializing concentration field...\n");
    if (active) initializeConcentration(cold.data(), nx, ny, slab, gridSize);

    if (root) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Request reqs[4];
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        if (!active) continue;
        const std::ptrdiff_t nzl = static_cast<std::ptrdiff_t>(slab.nzl);

        // Start the halo exchange of the concentration field
        const int nReq = postHaloExchange(cold.data(), plane, planeType, slab, reqs);

        // Interior planes of mu depend only on owned data, so compute them while
        // messages are in flight. This requires nzl > 2.
        if (nzl > 2) {
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, plane, 1, nzl - 1, p);
        }

        MPI_Waitall(nReq, reqs, MPI_STATUSES_IGNORE);
        replicateBoundaryGhosts(cold.data(), plane, kGhost, slab);

        // Remaining planes of mu, including the ghost planes at -1 and nzl
        if (nzl > 2) {
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, plane, -1, 1, p);
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, plane, nzl - 1, nzl + 1, p);
        } else {
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, plane, -1, nzl + 1, p);
        }

        // mu at the global boundary: clamped copies, so the update reads the
        // same values the serial code reads.
        replicateBoundaryGhosts(mu.data(), plane, 1, slab);

        // Update concentration on owned planes
        cahnHilliardUpdate(cold.data(), cnew.data(), mu.data(), nx, ny, plane, 0, nzl, p);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const long ms = static_cast<long>(maxElapsed * 1000.0);
    if (root) {
        printf("Computation time: %ld ms\n", ms);

        // Performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field on the root rank only when it is needed
    if (printResults || validate) {
        std::vector<int> counts(static_cast<size_t>(size));
        std::vector<int> displs(static_cast<size_t>(size));
        for (int r = 0; r < size; ++r) {
            const Slab sr = makeSlab(r, P, nz);
            counts[static_cast<size_t>(r)] = static_cast<int>(sr.nzl);
            displs[static_cast<size_t>(r)] = static_cast<int>(sr.z0);
        }

        std::vector<double> full;
        if (root) full.resize(gridSize);

        MPI_Gatherv(planePtr(cold.data(), plane, 0), static_cast<int>(slab.nzl), planeType,
                    root ? full.data() : nullptr, counts.data(), displs.data(), planeType,
                    0, MPI_COMM_WORLD);

        int status = 0;
        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
        if (validate) {
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        MPI_Type_free(&planeType);
        return status;
    }

    MPI_Type_free(&planeType);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
