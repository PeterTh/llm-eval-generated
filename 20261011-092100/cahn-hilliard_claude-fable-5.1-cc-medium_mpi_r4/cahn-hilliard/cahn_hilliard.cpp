#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// MPI parallelization: 1D slab decomposition along the Z axis.
//
// Each rank owns a contiguous range of Z planes [z0, z0 + lnz). The local
// arrays hold (lnz + 2) planes: plane 0 is the lower ghost plane and plane
// lnz + 1 is the upper ghost plane. Ghost planes are filled either by
// exchanging with the neighbouring rank (interior sub-domain boundaries) or by
// copying the adjacent owned plane (global domain boundaries, which reproduces
// the clamped boundary condition of the original code exactly).
//
// Halo depth is chosen globally (identically on all ranks):
//   * depth 2 (default, requires every rank to own >= 2 planes): a single halo
//     exchange of c per time step; mu is recomputed redundantly on the one
//     ghost plane adjacent to each interior sub-domain boundary, so cnew can be
//     updated without exchanging mu.
//   * depth 1 (fallback when nz / size < 2): two halo exchanges per time step
//     (c before computing mu, mu before computing cnew).
// Each exchange is non-blocking and overlapped with the computation of the
// interior planes that do not depend on ghost data. The arithmetic performed
// per cell is identical to the original code, so the results are bit-for-bit
// reproducible independent of the rank count.
// ---------------------------------------------------------------------------

// 3D index calculation (local plane index z includes the ghost offset)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Decomp {
    int rank = 0;
    int size = 1;
    size_t nx = 0, ny = 0, nz = 0;  // global grid
    size_t z0 = 0;                  // first owned global plane
    size_t lnz = 0;                 // number of owned planes
    int lower = MPI_PROC_NULL;      // rank owning plane z0 - 1 (or MPI_PROC_NULL)
    int upper = MPI_PROC_NULL;      // rank owning plane z0 + lnz (or MPI_PROC_NULL)
    size_t g = 1;                   // halo depth (number of ghost planes per side)
    size_t plane() const noexcept { return nx * ny; }
    size_t first() const noexcept { return g; }          // first owned local plane
    size_t last() const noexcept { return g + lnz; }     // one past last owned local plane
    size_t localPlanes() const noexcept { return lnz + 2 * g; }
};

// Compute the plane range owned by a rank: the first (nz % size) ranks get one extra plane.
static void planeRange(const size_t nz, const int size, const int rank, size_t& z0, size_t& lnz) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    lnz = base + (r < rem ? 1 : 0);
    z0 = r * base + std::min(r, rem);
}

static Decomp makeDecomp(const size_t nx, const size_t ny, const size_t nz, MPI_Comm comm) {
    Decomp d;
    MPI_Comm_rank(comm, &d.rank);
    MPI_Comm_size(comm, &d.size);
    d.nx = nx; d.ny = ny; d.nz = nz;
    planeRange(nz, d.size, d.rank, d.z0, d.lnz);
    // Ranks with planes form a prefix of the rank sequence, so the neighbours
    // of a non-empty rank are simply rank -/+ 1 when those ranks own planes.
    if (d.lnz > 0) {
        if (d.z0 > 0) d.lower = d.rank - 1;
        if (d.z0 + d.lnz < nz) d.upper = d.rank + 1;
    }
    // A 2-deep halo needs every rank to own at least two planes; the minimum
    // slab size is nz / size, which is identical on all ranks.
    d.g = (nz / static_cast<size_t>(d.size) >= 2) ? 2 : 1;
    return d;
}

// Non-blocking halo exchange of a field with (lnz + 2 * depth) planes.
// Sends the `depth` outermost owned planes to each neighbour and receives its
// `depth` ghost planes. Global domain boundaries replicate the boundary plane
// into the ghost planes (clamped boundary condition).
struct HaloExchange {
    MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    void start(std::vector<double>& f, const Decomp& d, const size_t depth, MPI_Comm comm, const int tag) {
        if (d.lnz == 0) return;
        const size_t p = d.plane();
        const size_t g = d.g;
        double* lowerGhost = f.data() + (g - depth) * p;       // `depth` planes just below first owned
        double* firstOwned = f.data() + g * p;                 // `depth` planes starting at first owned
        double* lastOwned  = f.data() + (g + d.lnz - depth) * p;  // `depth` planes ending at last owned
        double* upperGhost = f.data() + (g + d.lnz) * p;       // `depth` planes just above last owned
        const int n = static_cast<int>(depth * p);

        if (d.lower != MPI_PROC_NULL) {
            MPI_Irecv(lowerGhost, n, MPI_DOUBLE, d.lower, tag, comm, &reqs[0]);
            MPI_Isend(firstOwned, n, MPI_DOUBLE, d.lower, tag, comm, &reqs[1]);
        } else {
            for (size_t k = 0; k < depth; ++k)
                std::memcpy(lowerGhost + k * p, firstOwned, p * sizeof(double));
        }
        if (d.upper != MPI_PROC_NULL) {
            MPI_Irecv(upperGhost, n, MPI_DOUBLE, d.upper, tag, comm, &reqs[2]);
            MPI_Isend(lastOwned, n, MPI_DOUBLE, d.upper, tag, comm, &reqs[3]);
        } else {
            const double* lastPlane = f.data() + (g + d.lnz - 1) * p;
            for (size_t k = 0; k < depth; ++k)
                std::memcpy(upperGhost + k * p, lastPlane, p * sizeof(double));
        }
    }

    void finish() {
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    }
};

// Compute Laplacian with clamped boundary conditions in X/Y; Z neighbours are
// always available through the ghost planes (z is a local plane index in [1, lnz]).
inline double computeLaplacian(const double* __restrict c, const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z) noexcept {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Apply a stencil kernel to local planes [zb, ze). The kernel receives the
// linear local index and the Laplacian of `in` at that cell.
template <typename Kernel>
inline void applyStencil(const double* __restrict in, const size_t nx, const size_t ny,
                         const double dx, const double dy, const double dz,
                         const size_t zb, const size_t ze, Kernel&& kernel) {
    const size_t plane = nx * ny;
    for (size_t z = zb; z < ze; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const double* __restrict row  = in + z * plane + y * nx;
            const double* __restrict rowp = in + z * plane + yp * nx;
            const double* __restrict rown = in + z * plane + yn * nx;
            const double* __restrict rowu = in + (z + 1) * plane + y * nx;
            const double* __restrict rowd = in + (z - 1) * plane + y * nx;
            const size_t rowIdx = z * plane + y * nx;

            // Boundary cells in X use the generic clamped formulation.
            kernel(rowIdx, computeLaplacian(in, nx, ny, dx, dy, dz, 0, y, z));
            // Interior cells: branch-free, vectorizable. Same operation order as above.
            for (size_t x = 1; x + 1 < nx; ++x) {
                const double cc = row[x];
                const double cxx = (row[x + 1] + row[x - 1] - 2.0 * cc) / (dx * dx);
                const double cyy = (rowp[x] + rown[x] - 2.0 * cc) / (dy * dy);
                const double czz = (rowu[x] + rowd[x] - 2.0 * cc) / (dz * dz);
                kernel(rowIdx + x, cxx + cyy + czz);
            }
            if (nx > 1) {
                kernel(rowIdx + nx - 1, computeLaplacian(in, nx, ny, dx, dy, dz, nx - 1, y, z));
            }
        }
    }
}

// Compute chemical potential on local planes [zb, ze)
inline void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                                     const size_t nx, const size_t ny,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                     const size_t zb, const size_t ze) {
    const double* __restrict cp = c.data();
    double* __restrict mp = mu.data();
    applyStencil(cp, nx, ny, dx, dy, dz, zb, ze, [=](const size_t idx, const double lap) {
        const double cv = cp[idx];
        mp[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * lap;
    });
}

// Cahn-Hilliard update step on local planes [zb, ze)
inline void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               const size_t nx, const size_t ny,
                               const double D, const double dt, const double dx, const double dy, const double dz,
                               const size_t zb, const size_t ze) {
    const double* __restrict op = cold.data();
    double* __restrict np = cnew.data();
    applyStencil(mu.data(), nx, ny, dx, dy, dz, zb, ze, [=](const size_t idx, const double lap) {
        np[idx] = op[idx] + dt * D * lap;
    });
}

// Initialize the locally owned part of the concentration field
void initializeConcentration(std::vector<double>& c, const Decomp& d) {
    const size_t nx = d.nx, ny = d.ny;
    const size_t vol = d.nx * d.ny * d.nz;

    for (size_t lz = 0; lz < d.lnz; ++lz) {
        const size_t z = d.z0 + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz + d.first(), nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validate the distributed field; result is identical on all ranks, output on rank 0.
bool validateResult(const std::vector<double>& c, const Decomp& d, MPI_Comm comm) {
    const size_t p = d.plane();
    const double* begin = c.data() + d.first() * p;
    const double* end = c.data() + d.last() * p;

    // Check for NaN or Inf
    int bad = 0;
    for (const double* it = begin; it != end; ++it) {
        if (std::isnan(*it) || std::isinf(*it)) { bad = 1; break; }
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = INFINITY;
    double maxVal = -INFINITY;
    for (const double* it = begin; it != end; ++it) {
        minVal = std::min(minVal, *it);
        maxVal = std::max(maxVal, *it);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (d.rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Gather the distributed field into a full global array on rank 0.
std::vector<double> gatherField(const std::vector<double>& c, const Decomp& d, MPI_Comm comm) {
    std::vector<double> full;
    std::vector<int> counts, displs;
    if (d.rank == 0) {
        full.resize(d.nx * d.ny * d.nz);
        counts.resize(d.size);
        displs.resize(d.size);
        for (int r = 0; r < d.size; ++r) {
            size_t z0, lnz;
            planeRange(d.nz, d.size, r, z0, lnz);
            counts[r] = static_cast<int>(lnz * d.plane());
            displs[r] = static_cast<int>(z0 * d.plane());
        }
    }
    MPI_Gatherv(c.data() + d.first() * d.plane(), static_cast<int>(d.lnz * d.plane()), MPI_DOUBLE,
                full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    return full;
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const Decomp d = makeDecomp(nx, ny, nz, comm);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (halo depth %zu)\n", d.size, d.g);
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
    const size_t localSize = d.localPlanes() * d.plane();

    // Allocate arrays (owned planes plus two ghost planes)
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, d);

    // Local plane bookkeeping. Owned planes are [zFirst, zLast). The chemical
    // potential must be available on [muB, muE): one plane beyond the owned
    // range towards each interior neighbour (depth 2: recomputed locally from
    // the 2-deep c halo; depth 1: received from the neighbour). Planes in
    // [zIntB, zIntE) only touch owned data and are computed while the halo
    // exchange is in flight.
    const size_t p = d.plane();
    const size_t zFirst = d.first();
    const size_t zLast = d.last();
    const size_t muB = (d.g == 2 && d.lower != MPI_PROC_NULL) ? zFirst - 1 : zFirst;
    const size_t muE = (d.g == 2 && d.upper != MPI_PROC_NULL) ? zLast + 1 : zLast;
    const size_t zIntB = std::min<size_t>(zFirst + 1, zLast);
    const size_t zIntE = (d.lnz >= 2) ? zLast - 1 : zIntB;

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double tStart = MPI_Wtime();

    HaloExchange halo;
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        halo.start(cold, d, d.g, comm, 0);
        computeChemicalPotential(cold, mu, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB, zIntB, zIntE);
        halo.finish();
        computeChemicalPotential(cold, mu, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB, muB, zIntB);
        computeChemicalPotential(cold, mu, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB, zIntE, muE);

        if (d.g == 2) {
            // Ghost planes of mu at global boundaries follow the clamped condition.
            if (d.lnz > 0 && d.lower == MPI_PROC_NULL)
                std::memcpy(mu.data() + (zFirst - 1) * p, mu.data() + zFirst * p, p * sizeof(double));
            if (d.lnz > 0 && d.upper == MPI_PROC_NULL)
                std::memcpy(mu.data() + zLast * p, mu.data() + (zLast - 1) * p, p * sizeof(double));
            // Update concentration (mu halo already available locally)
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, zFirst, zLast);
        } else {
            // Update concentration after exchanging the mu halo
            halo.start(mu, d, 1, comm, 1);
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, zIntB, zIntE);
            halo.finish();
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, zFirst, zIntB);
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, D, dt, dx, dy, dz, zIntE, zLast);
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(comm);
    const double tEnd = MPI_Wtime();
    const long durationMs = static_cast<long>((tEnd - tStart) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full = gatherField(cold, d, comm);
        if (rank == 0) print_results(full, "Concentration");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, d, comm);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    MPI_Finalize();
    return ret;
}
