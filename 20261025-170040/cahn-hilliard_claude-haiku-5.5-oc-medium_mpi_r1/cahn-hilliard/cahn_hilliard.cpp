#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed-memory Cahn-Hilliard solver.
//
// The domain is split into contiguous slabs of z-planes, one slab per MPI rank
// (ranks beyond nz own nothing). Each field is stored with one ghost plane below
// (local index 0) and one above (local index nzl+1); owned planes are 1..nzl.
// Ghost planes that lie across a rank boundary are filled by nonblocking halo
// exchange. Ghost planes that lie outside the global domain hold a copy of the
// nearest owned plane, which reproduces the clamped (Neumann-like) boundary of
// the serial code exactly.
//
// Each time step is organised to hide communication:
//   1. start halo exchange of c
//   2. compute mu on interior planes (needs only owned c)
//   3. finish halo exchange of c; compute mu on the two boundary planes
//   4. start halo exchange of mu
//   5. compute c_new on interior planes (needs only owned mu)
//   6. finish halo exchange of mu; compute c_new on the two boundary planes
// All stencil arithmetic is performed in the same order as the serial code, so
// results are bitwise identical for any number of ranks.

namespace {

struct Params {
    double dx, dy, dz;
    double dt;
    double e_AA, e_BB, e_AB;
    double gamma;
    double D;
};

// Owned z-planes of rank r: first plane index and count.
void planeRange(size_t nz, size_t P, size_t r, size_t& z0, size_t& nzl) {
    const size_t base = nz / P;
    const size_t rem = nz % P;
    nzl = base + (r < rem ? 1 : 0);
    z0 = r * base + std::min(r, rem);
}

struct Decomp {
    MPI_Comm comm;
    size_t nx, ny, nz;
    size_t plane;   // nx * ny
    size_t z0;      // first global z-plane owned by this rank
    size_t nzl;     // number of owned z-planes
    int down;       // rank owning the planes below, or MPI_PROC_NULL
    int up;         // rank owning the planes above, or MPI_PROC_NULL
};

Decomp makeDecomp(MPI_Comm comm, int rank, int size, size_t nx, size_t ny, size_t nz) {
    const size_t P = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t active = std::min(P, nz);  // ranks [0, active) own at least one plane

    Decomp d{};
    d.comm = comm;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.plane = nx * ny;
    planeRange(nz, P, r, d.z0, d.nzl);
    d.down = (r > 0 && r < active) ? rank - 1 : MPI_PROC_NULL;
    d.up = (r + 1 < active) ? rank + 1 : MPI_PROC_NULL;
    return d;
}

// Message tags: TAG_UP carries data moving to a higher rank, TAG_DOWN to a lower one.
constexpr int TAG_UP = 0;
constexpr int TAG_DOWN = 1;

// Post nonblocking exchange of the boundary planes of field f (4 requests).
void haloStart(const Decomp& d, double* f, MPI_Request* req) {
    const int n = static_cast<int>(d.plane);
    double* lowGhost = f;
    double* lowOwned = f + d.plane;
    double* highOwned = f + d.nzl * d.plane;
    double* highGhost = f + (d.nzl + 1) * d.plane;

    MPI_Irecv(lowGhost, n, MPI_DOUBLE, d.down, TAG_UP, d.comm, &req[0]);
    MPI_Irecv(highGhost, n, MPI_DOUBLE, d.up, TAG_DOWN, d.comm, &req[1]);
    MPI_Isend(highOwned, n, MPI_DOUBLE, d.up, TAG_UP, d.comm, &req[2]);
    MPI_Isend(lowOwned, n, MPI_DOUBLE, d.down, TAG_DOWN, d.comm, &req[3]);
}

// Complete the exchange and refresh ghost planes on the global domain boundary.
void haloFinish(const Decomp& d, double* f, MPI_Request* req) {
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (d.nzl == 0) return;

    const size_t pl = d.plane;
    if (d.down == MPI_PROC_NULL) {
        std::copy(f + pl, f + 2 * pl, f);
    }
    if (d.up == MPI_PROC_NULL) {
        std::copy(f + d.nzl * pl, f + (d.nzl + 1) * pl, f + (d.nzl + 1) * pl);
    }
}

// Visit every x index of a row together with its clamped neighbours xp and xn.
template <typename F>
inline void forEachX(size_t nx, F&& body) {
    if (nx == 0) return;
    if (nx == 1) {
        body(size_t{0}, size_t{0}, size_t{0});
        return;
    }
    body(size_t{0}, size_t{1}, size_t{0});
    for (size_t x = 1; x + 1 < nx; ++x) {
        body(x, x + 1, x - 1);
    }
    body(nx - 1, nx - 1, nx - 2);
}

// Pointers to the centre row and its six neighbour rows (clamped in y).
struct Row {
    const double* c;
    const double* up;
    const double* dn;
    const double* zp;
    const double* zm;
};

inline Row rowAt(const double* f, const Decomp& d, size_t p, size_t y) {
    const double* row = f + p * d.plane + y * d.nx;
    return Row{
        row,
        (y + 1 < d.ny) ? row + d.nx : row,
        (y > 0) ? row - d.nx : row,
        row + d.plane,
        row - d.plane,
    };
}

// 7-point Laplacian with clamped boundaries; same operation order as the serial code.
inline double laplacian(const Row& r, size_t x, size_t xp, size_t xn, const Params& P) {
    const double cv = r.c[x];
    const double cxx = (r.c[xp] + r.c[xn] - 2.0 * cv) / (P.dx * P.dx);
    const double cyy = (r.up[x] + r.dn[x] - 2.0 * cv) / (P.dy * P.dy);
    const double czz = (r.zp[x] + r.zm[x] - 2.0 * cv) / (P.dz * P.dz);
    return cxx + cyy + czz;
}

// Chemical potential on owned plane p (1..nzl) from c (ghosts must be valid).
void computeMuPlane(const double* c, double* mu, const Decomp& d, size_t p, const Params& P) {
    for (size_t y = 0; y < d.ny; ++y) {
        const Row r = rowAt(c, d, p, y);
        double* out = mu + p * d.plane + y * d.nx;
        forEachX(d.nx, [&](size_t x, size_t xp, size_t xn) {
            const double cv = r.c[x];
            out[x] = 4.5 * ((cv + 1.0) * P.e_AA + (cv - 1.0) * P.e_BB - 2.0 * cv * P.e_AB)
                     + 3.0 * cv + cv * cv * cv
                     - P.gamma * laplacian(r, x, xp, xn, P);
        });
    }
}

// Explicit update on owned plane p from cold and mu (mu ghosts must be valid).
void computeCnewPlane(const double* cold, double* cnew, const double* mu, const Decomp& d,
                      size_t p, const Params& P) {
    for (size_t y = 0; y < d.ny; ++y) {
        const Row m = rowAt(mu, d, p, y);
        const double* cr = cold + p * d.plane + y * d.nx;
        double* out = cnew + p * d.plane + y * d.nx;
        forEachX(d.nx, [&](size_t x, size_t xp, size_t xn) {
            out[x] = cr[x] + P.dt * P.D * laplacian(m, x, xp, xn, P);
        });
    }
}

// One Cahn-Hilliard step: cnew = f(cold), with communication overlapped.
void step(const Decomp& d, const Params& P, double* cold, double* cnew, double* mu,
          MPI_Request* reqC, MPI_Request* reqM) {
    const size_t nzl = d.nzl;

    // Phase 1: exchange c ghosts while computing mu on interior planes.
    haloStart(d, cold, reqC);
    for (size_t p = 2; p < nzl; ++p) {
        computeMuPlane(cold, mu, d, p, P);
    }
    haloFinish(d, cold, reqC);

    // Phase 2: boundary planes of mu (need c ghosts), then exchange mu ghosts.
    if (nzl >= 1) computeMuPlane(cold, mu, d, 1, P);
    if (nzl >= 2) computeMuPlane(cold, mu, d, nzl, P);
    haloStart(d, mu, reqM);

    // Phase 3: interior c_new planes while mu ghosts are in flight.
    for (size_t p = 2; p < nzl; ++p) {
        computeCnewPlane(cold, cnew, mu, d, p, P);
    }
    haloFinish(d, mu, reqM);

    // Phase 4: boundary c_new planes (need mu ghosts).
    if (nzl >= 1) computeCnewPlane(cold, cnew, mu, d, 1, P);
    if (nzl >= 2) computeCnewPlane(cold, cnew, mu, d, nzl, P);
}

// Initialize owned planes; global linear index defines the pseudo-random pattern.
void initializeConcentration(double* c, const Decomp& d) {
    const size_t vol = d.nx * d.ny * d.nz;
    for (size_t p = 1; p <= d.nzl; ++p) {
        const size_t z = d.z0 + (p - 1);
        for (size_t y = 0; y < d.ny; ++y) {
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t idx = p * d.plane + y * d.nx + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * d.plane + y * d.nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather the owned portions of the field on rank 0 in global z order.
std::vector<double> gatherField(const Decomp& d, const double* f, int rank, int size) {
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t z0 = 0, nzl = 0;
        planeRange(d.nz, static_cast<size_t>(size), static_cast<size_t>(r), z0, nzl);
        counts[r] = static_cast<int>(nzl * d.plane);
        displs[r] = static_cast<int>(z0 * d.plane);
    }
    std::vector<double> full;
    if (rank == 0) full.resize(d.nx * d.ny * d.nz);
    MPI_Gatherv(f + d.plane, static_cast<int>(d.nzl * d.plane), MPI_DOUBLE,
                rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, d.comm);
    return full;
}

// Distributed version of the serial validation; all ranks reach the same verdict.
bool validateResult(const Decomp& d, const double* f, int rank) {
    int localBad = 0;
    double lo = HUGE_VAL;
    double hi = -HUGE_VAL;
    for (size_t i = d.plane; i < (d.nzl + 1) * d.plane; ++i) {
        const double v = f[i];
        if (std::isnan(v) || std::isinf(v)) localBad = 1;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }

    int globalBad = 0;
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_MAX, d.comm);
    if (globalBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (d.nx * d.ny * d.nz == 0) return true;

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", lo, hi);

    if (hi > 10.0 || lo < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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

int run(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks agree on exit)
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
    const Params P{
        1.0,                    // dx
        1.0,                    // dy
        1.0,                    // dz
        0.01,                   // dt
        -(2.0 / 9.0),           // e_AA
        -(2.0 / 9.0),           // e_BB
        (2.0 / 9.0),            // e_AB
        0.5,                    // gamma
        1.0,                    // D
    };

    const Decomp d = makeDecomp(MPI_COMM_WORLD, rank, size, nx, ny, nz);
    const size_t gridSize = nx * ny * nz;

    // Each buffer holds the owned planes plus one ghost plane on each side
    const size_t bufSize = (d.nzl + 2) * d.plane;
    std::vector<double> coldV(bufSize, 0.0);
    std::vector<double> cnewV(bufSize, 0.0);
    std::vector<double> muV(bufSize, 0.0);
    double* cold = coldV.data();
    double* cnew = cnewV.data();
    double* mu = muV.data();

    if (root) printf("Initializing concentration field...\n");
    initializeConcentration(cold, d);

    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Request reqC[4];
    MPI_Request reqM[4];

    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        step(d, P, cold, cnew, mu, reqC, reqM);
        std::swap(cold, cnew);
    }

    const double elapsed = MPI_Wtime() - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Integer milliseconds, as in the serial version
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gathered in global order)
    if (printResults) {
        std::vector<double> full = gatherField(d, cold, rank, size);
        if (root) {
            print_results(full, "Concentration");
        }
    }

    int rc = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateResult(d, cold, rank);
        if (root) {
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
        rc = valid ? 0 : 1;
    }

    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = run(argc, argv);
    MPI_Finalize();
    return rc;
}
