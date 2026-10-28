#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

// Physical parameters (grid spacing is unit; constants are folded at compile time)
constexpr double kDx = 1.0;
constexpr double kDy = 1.0;
constexpr double kDz = 1.0;
constexpr double kDt = 0.01;
constexpr double kEAA = -(2.0 / 9.0);
constexpr double kEBB = -(2.0 / 9.0);
constexpr double kEAB = (2.0 / 9.0);
constexpr double kGamma = 0.5;
constexpr double kD = 1.0;

// Halo message tags
constexpr int kTagToDown = 0;
constexpr int kTagToUp = 1;

struct Params {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
};

// 3D index calculation (z slowest, x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Z-slab decomposition of the global grid. Each rank owns global planes [z0, z0+nzl).
// Local buffers hold nzl+2 planes: plane 0 and plane nzl+1 are ghost planes.
struct Slab {
    size_t nx = 0, ny = 0, nz = 0;
    size_t plane = 0;   // nx*ny
    size_t z0 = 0;      // first owned global plane
    size_t nzl = 0;     // number of owned planes
    int down = MPI_PROC_NULL;  // neighbour with lower z (or MPI_PROC_NULL at global boundary)
    int up = MPI_PROC_NULL;    // neighbour with higher z (or MPI_PROC_NULL at global boundary)
};

// Visit the 5-point-per-axis Laplacian of plane-local field f for every cell of one plane.
// fzm / fzp are the planes below / above. In-plane neighbours use clamped indices (zero-flux),
// matching the original computeLaplacian boundary handling. The arithmetic order is identical
// to the original code.
template <typename Op>
inline void forEachLaplacian(const double* f, const double* fzm, const double* fzp,
                             const size_t nx, const size_t ny, Op&& op) {
    for (size_t y = 0; y < ny; ++y) {
        const size_t row = y * nx;
        const size_t rowYp = (y + 1 < ny ? y + 1 : y) * nx;
        const size_t rowYn = (y > 0 ? y - 1 : 0) * nx;

        auto cell = [&](const size_t x, const size_t xm, const size_t xp) {
            const size_t i = row + x;
            const double cv = f[i];
            const double cxx = (f[row + xp] + f[row + xm] - 2.0 * cv) / (kDx * kDx);
            const double cyy = (f[rowYp + x] + f[rowYn + x] - 2.0 * cv) / (kDy * kDy);
            const double czz = (fzp[i] + fzm[i] - 2.0 * cv) / (kDz * kDz);
            op(i, cxx + cyy + czz);
        };

        // Peeled boundary cells in x, branch-free interior for vectorization
        cell(0, 0, nx > 1 ? 1 : 0);
        for (size_t x = 1; x + 1 < nx; ++x) {
            cell(x, x - 1, x + 1);
        }
        if (nx > 1) {
            cell(nx - 1, nx - 2, nx - 1);
        }
    }
}

// Chemical potential for local plane p (c and mu are full local buffers)
inline void chemicalPotentialPlane(const double* c, double* mu, const size_t nx, const size_t ny,
                                   const size_t plane, const size_t p) {
    const double* cc = c + p * plane;
    double* out = mu + p * plane;
    forEachLaplacian(cc, c + (p - 1) * plane, c + (p + 1) * plane, nx, ny,
                     [&](const size_t i, const double lap) {
                         const double cv = cc[i];
                         out[i] = 4.5 * ((cv + 1.0) * kEAA + (cv - 1.0) * kEBB - 2.0 * cv * kEAB)
                                  + 3.0 * cv + cv * cv * cv
                                  - kGamma * lap;
                     });
}

// Cahn-Hilliard update for local plane p
inline void updatePlane(const double* cold, double* cnew, const double* mu, const size_t nx, const size_t ny,
                        const size_t plane, const size_t p) {
    const double* mm = mu + p * plane;
    double* out = cnew + p * plane;
    const double* co = cold + p * plane;
    forEachLaplacian(mm, mu + (p - 1) * plane, mu + (p + 1) * plane, nx, ny,
                     [&](const size_t i, const double lap) {
                         out[i] = co[i] + kDt * kD * lap;
                     });
}

// Post non-blocking exchange of the first/last owned planes with the z-neighbours.
// Receives go into the ghost planes. req must have room for 4 requests.
void postHaloExchange(MPI_Comm comm, const Slab& s, double* f, MPI_Datatype planeType, MPI_Request* req) {
    MPI_Irecv(f, 1, planeType, s.down, kTagToUp, comm, &req[0]);
    MPI_Irecv(f + (s.nzl + 1) * s.plane, 1, planeType, s.up, kTagToDown, comm, &req[1]);
    MPI_Isend(f + s.plane, 1, planeType, s.down, kTagToDown, comm, &req[2]);
    MPI_Isend(f + s.nzl * s.plane, 1, planeType, s.up, kTagToUp, comm, &req[3]);
}

// At global z boundaries there is no neighbour: the ghost plane replicates the edge plane,
// which is exactly the clamped boundary condition of the original code.
void fillPhysicalGhosts(const Slab& s, double* f) {
    if (s.down == MPI_PROC_NULL) {
        std::copy(f + s.plane, f + 2 * s.plane, f);
    }
    if (s.up == MPI_PROC_NULL) {
        std::copy(f + s.nzl * s.plane, f + (s.nzl + 1) * s.plane, f + (s.nzl + 1) * s.plane);
    }
}

// One time step. Halo traffic is overlapped with computation on interior planes.
void timeStep(MPI_Comm comm, const Slab& s, MPI_Datatype planeType,
              std::vector<double>& cold, std::vector<double>& cnew, std::vector<double>& mu) {
    MPI_Request req[4];
    double* c = cold.data();
    double* m = mu.data();
    double* cn = cnew.data();
    const size_t nx = s.nx, ny = s.ny, plane = s.plane;

    // Stage 1: chemical potential needs cold's ghost planes.
    postHaloExchange(comm, s, c, planeType, req);
    for (size_t p = 2; p < s.nzl; ++p) {  // interior: needs only owned planes
        chemicalPotentialPlane(c, m, nx, ny, plane, p);
    }
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    fillPhysicalGhosts(s, c);
    chemicalPotentialPlane(c, m, nx, ny, plane, 1);
    if (s.nzl > 1) {
        chemicalPotentialPlane(c, m, nx, ny, plane, s.nzl);
    }

    // Stage 2: update needs mu's ghost planes.
    postHaloExchange(comm, s, m, planeType, req);
    for (size_t p = 2; p < s.nzl; ++p) {
        updatePlane(c, cn, m, nx, ny, plane, p);
    }
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    fillPhysicalGhosts(s, m);
    updatePlane(c, cn, m, nx, ny, plane, 1);
    if (s.nzl > 1) {
        updatePlane(c, cn, m, nx, ny, plane, s.nzl);
    }
}

// Initialize owned planes using global linear indices (identical to the serial code)
void initializeConcentration(std::vector<double>& c, const Slab& s) {
    const size_t vol = s.nx * s.ny * s.nz;
    for (size_t z = 0; z < s.nzl; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t gz = s.z0 + z;
                const size_t idx = (z + 1) * s.plane + y * s.nx + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx3(x, y, gz, s.nx, s.ny);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Distributed validation: every rank checks its owned planes, reductions give the global result.
// Returns the same value on all ranks.
bool validateResult(MPI_Comm comm, const Slab& s, const std::vector<double>& c, const bool root) {
    int localBad = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (size_t i = s.plane; i < (s.nzl + 1) * s.plane; ++i) {
        const double v = c[i];
        if (std::isnan(v) || std::isinf(v)) {
            localBad = 1;
        }
    }
    int globalBad = 0;
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_LOR, comm);
    if (globalBad) {
        if (root) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    for (size_t i = s.plane; i < (s.nzl + 1) * s.plane; ++i) {
        localMin = std::min(localMin, c[i]);
        localMax = std::max(localMax, c[i]);
    }
    double minVal = 0.0, maxVal = 0.0;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (root) {
        printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    }

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        if (root) {
            printf("Validation failed: values out of expected range\n");
        }
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

// Runs the simulation on communicator comm (all ranks in comm own at least one plane).
int simulate(MPI_Comm comm, const Params& params) {
    int rank = 0, nproc = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    const bool root = (rank == 0);

    const size_t nx = params.nx, ny = params.ny, nz = params.nz;
    const size_t gridSize = nx * ny * nz;

    Slab s;
    s.nx = nx;
    s.ny = ny;
    s.nz = nz;
    s.plane = nx * ny;
    const size_t base = nz / static_cast<size_t>(nproc);
    const size_t rem = nz % static_cast<size_t>(nproc);
    const size_t r = static_cast<size_t>(rank);
    s.nzl = base + (r < rem ? 1 : 0);
    s.z0 = r * base + std::min(r, rem);
    s.down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    s.up = rank < nproc - 1 ? rank + 1 : MPI_PROC_NULL;

    // One MPI datatype element == one full xy-plane
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(s.plane), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Local buffers with ghost planes
    const size_t localSize = (s.nzl + 2) * s.plane;
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    if (root) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, s);

    if (root) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    for (int t = 0; t < params.iterations; ++t) {
        timeStep(comm, s, planeType, cold, cnew, mu);
        std::swap(cold, cnew);
    }

    MPI_Barrier(comm);
    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        const long long ms = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", ms);

        const double cellUpdates = static_cast<double>(gridSize) * params.iterations;
        const double mcups = cellUpdates / (static_cast<double>(ms) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gathered in global z order)
    if (params.printResults) {
        std::vector<int> counts(nproc), displs(nproc);
        for (int q = 0; q < nproc; ++q) {
            const size_t uq = static_cast<size_t>(q);
            counts[q] = static_cast<int>(base + (uq < rem ? 1 : 0));
            displs[q] = static_cast<int>(uq * base + std::min(uq, rem));
        }
        std::vector<double> full;
        if (root) {
            full.resize(gridSize);
        }
        MPI_Gatherv(cold.data() + s.plane, static_cast<int>(s.nzl), planeType,
                    root ? full.data() : nullptr, counts.data(), displs.data(), planeType, 0, comm);
        if (root) {
            print_results(full, "Concentration");
        }
    }

    int rc = 0;
    if (params.validate) {
        if (root) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(comm, s, cold, root);
        if (root) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        rc = valid ? 0 : 1;
    }

    MPI_Type_free(&planeType);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const bool root = (rank == 0);

    Params params;
    bool exitEarly = false;
    int status = 0;

    // Parse command line arguments (identical on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            params.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            params.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            params.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            params.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            params.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            params.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            exitEarly = true;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitEarly = true;
            status = 1;
            break;
        }
    }

    if (!exitEarly) {
        if (params.ny == 0) params.ny = params.nx;
        if (params.nz == 0) params.nz = params.nx;

        if (params.nx == 0 || params.ny == 0 || params.nz == 0 || params.nx * params.ny > INT_MAX) {
            if (root) {
                printf("Error: grid dimensions must be positive and nx*ny must fit in int\n");
            }
            status = 1;
        } else {
            if (root) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", params.nx, params.ny, params.nz);
                printf("Time steps: %d\n", params.iterations);
                printf("Validation: %s\n", params.validate ? "enabled" : "disabled");
            }

            // Ranks beyond nz own no planes and do not take part in the simulation
            const int active = static_cast<int>(std::min(static_cast<size_t>(nproc), params.nz));
            MPI_Comm comm = MPI_COMM_NULL;
            MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
            if (comm != MPI_COMM_NULL) {
                status = simulate(comm, params);
                MPI_Comm_free(&comm);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
