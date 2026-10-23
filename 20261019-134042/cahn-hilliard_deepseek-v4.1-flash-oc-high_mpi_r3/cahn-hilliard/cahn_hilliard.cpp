#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static bool factor3D(int p, int nx, int ny, int nz, int& px, int& py, int& pz) {
    bool found = false;
    double bestScore = std::numeric_limits<double>::infinity();
    int bestX = 1;
    int bestY = 1;
    int bestZ = 1;

    for (int z = 1; z <= std::min(p, nz); ++z) {
        if (p % z != 0) continue;
        const int r = p / z;
        for (int y = 1; y <= std::min(r, ny); ++y) {
            if (r % y != 0) continue;
            const int x = r / y;
            if (x > nx) continue;

            const double lx = static_cast<double>(nx) / x;
            const double ly = static_cast<double>(ny) / y;
            const double lz = static_cast<double>(nz) / z;
            const double score = 1.0 / lx + 1.0 / ly + 1.0 / lz;

            if (!found || score < bestScore - 1e-12 ||
                (std::abs(score - bestScore) <= 1e-12 && z > bestZ)) {
                found = true;
                bestScore = score;
                bestX = x;
                bestY = y;
                bestZ = z;
            }
        }
    }

    if (found) {
        px = bestX;
        py = bestY;
        pz = bestZ;
    }
    return found;
}

// Compute Laplacian with clamped boundary conditions
static inline double computeLaplacian(const double* c, const int x, const int y, const int z,
                                      const int sy, const int sz,
                                      const double dx, const double dy, const double dz) {
    const int idx = z * sz + y * sy + x;
    const double center = c[idx];
    const double cxx = (c[idx + 1] + c[idx - 1] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx + sy] + c[idx - sy] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx + sz] + c[idx - sz] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

struct HaloTypes {
    MPI_Datatype sendLow[3];
    MPI_Datatype sendHigh[3];
    MPI_Datatype recvLow[3];
    MPI_Datatype recvHigh[3];
};

static MPI_Datatype makeSubarray(const int lx, const int ly, const int lz,
                                 const int s0, const int s1, const int s2,
                                 const int n0, const int n1, const int n2) {
    int sizes[3] = {lz + 2, ly + 2, lx + 2};
    int subsizes[3] = {s0, s1, s2};
    int starts[3] = {n0, n1, n2};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

static HaloTypes createHaloTypes(const int lx, const int ly, const int lz) {
    HaloTypes t;
    t.sendLow[0] = makeSubarray(lx, ly, lz, lz, ly, 1, 1, 1, 1);
    t.recvLow[0] = makeSubarray(lx, ly, lz, lz, ly, 1, 1, 1, 0);
    t.sendHigh[0] = makeSubarray(lx, ly, lz, lz, ly, 1, 1, 1, lx);
    t.recvHigh[0] = makeSubarray(lx, ly, lz, lz, ly, 1, 1, 1, lx + 1);
    t.sendLow[1] = makeSubarray(lx, ly, lz, lz, 1, lx, 1, 1, 1);
    t.recvLow[1] = makeSubarray(lx, ly, lz, lz, 1, lx, 1, 0, 1);
    t.sendHigh[1] = makeSubarray(lx, ly, lz, lz, 1, lx, 1, ly, 1);
    t.recvHigh[1] = makeSubarray(lx, ly, lz, lz, 1, lx, 1, ly + 1, 1);
    t.sendLow[2] = makeSubarray(lx, ly, lz, 1, ly, lx, 1, 1, 1);
    t.recvLow[2] = makeSubarray(lx, ly, lz, 1, ly, lx, 0, 1, 1);
    t.sendHigh[2] = makeSubarray(lx, ly, lz, 1, ly, lx, lz, 1, 1);
    t.recvHigh[2] = makeSubarray(lx, ly, lz, 1, ly, lx, lz + 1, 1, 1);
    return t;
}

static void freeHaloTypes(HaloTypes& t) {
    for (int d = 0; d < 3; ++d) {
        MPI_Type_free(&t.sendLow[d]);
        MPI_Type_free(&t.sendHigh[d]);
        MPI_Type_free(&t.recvLow[d]);
        MPI_Type_free(&t.recvHigh[d]);
    }
}

static void exchangeHalos(std::vector<double>& a,
                          const int lx, const int ly, const int lz,
                          const int sy, const int sz,
                          const int lower[3], const int upper[3],
                          const HaloTypes& types, MPI_Comm comm) {
    MPI_Request req[12];
    int nreq = 0;

    for (int d = 0; d < 3; ++d) {
        const int tag = 100 + d;
        if (lower[d] != MPI_PROC_NULL) {
            MPI_Isend(a.data(), 1, types.sendLow[d], lower[d], tag, comm, &req[nreq++]);
            MPI_Irecv(a.data(), 1, types.recvLow[d], lower[d], tag, comm, &req[nreq++]);
        }
        if (upper[d] != MPI_PROC_NULL) {
            MPI_Isend(a.data(), 1, types.sendHigh[d], upper[d], tag, comm, &req[nreq++]);
            MPI_Irecv(a.data(), 1, types.recvHigh[d], upper[d], tag, comm, &req[nreq++]);
        }
    }

    if (nreq > 0) {
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    }

    if (lower[0] == MPI_PROC_NULL) {
        for (int k = 1; k <= lz; ++k)
            for (int j = 1; j <= ly; ++j) {
                const int b = k * sz + j * sy;
                a[b] = a[b + 1];
            }
    }
    if (upper[0] == MPI_PROC_NULL) {
        for (int k = 1; k <= lz; ++k)
            for (int j = 1; j <= ly; ++j) {
                const int b = k * sz + j * sy + lx;
                a[b + 1] = a[b];
            }
    }
    if (lower[1] == MPI_PROC_NULL) {
        for (int k = 1; k <= lz; ++k)
            for (int i = 1; i <= lx; ++i) {
                const int b = k * sz + i;
                a[b] = a[b + sy];
            }
    }
    if (upper[1] == MPI_PROC_NULL) {
        for (int k = 1; k <= lz; ++k)
            for (int i = 1; i <= lx; ++i) {
                const int b = k * sz + ly * sy + i;
                a[b + sy] = a[b];
            }
    }
    if (lower[2] == MPI_PROC_NULL) {
        std::memcpy(a.data(), a.data() + sz, sizeof(double) * static_cast<size_t>(sz));
    }
    if (upper[2] == MPI_PROC_NULL) {
        std::memcpy(a.data() + static_cast<size_t>(lz + 1) * sz,
                    a.data() + static_cast<size_t>(lz) * sz,
                    sizeof(double) * static_cast<size_t>(sz));
    }
}

// Compute chemical potential
static void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                                     const int lx, const int ly, const int lz,
                                     const int sy, const int sz,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB,
                                     const double e_AB) {
    for (int z = 1; z <= lz; ++z) {
        for (int y = 1; y <= ly; ++y) {
            for (int x = 1; x <= lx; ++x) {
                const int idx = z * sz + y * sy + x;
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c.data(), x, y, z, sy, sz, dx, dy, dz);
            }
        }
    }
}

// Cahn-Hilliard update step
static void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               const int lx, const int ly, const int lz,
                               const int sy, const int sz,
                               const double D, const double dt,
                               const double dx, const double dy, const double dz) {
    for (int z = 1; z <= lz; ++z) {
        for (int y = 1; y <= ly; ++y) {
            for (int x = 1; x <= lx; ++x) {
                const int idx = z * sz + y * sy + x;
                cnew[idx] = cold[idx] + dt * D *
                            computeLaplacian(mu.data(), x, y, z, sy, sz, dx, dy, dz);
            }
        }
    }
}

// Initialize concentration field
static void initializeConcentration(std::vector<double>& c,
                                    const int nx, const int ny, const int nz,
                                    const int startX, const int startY, const int startZ,
                                    const int lx, const int ly, const int lz,
                                    const int sy, const int sz) {
    const size_t vol = static_cast<size_t>(nx) * ny * nz;

    for (int k = 1; k <= lz; ++k) {
        const int gz = startZ + k - 1;
        for (int j = 1; j <= ly; ++j) {
            const int gy = startY + j - 1;
            for (int i = 1; i <= lx; ++i) {
                const int gx = startX + i - 1;
                const size_t linear_id = static_cast<size_t>(gz) * (nx * ny)
                                       + static_cast<size_t>(gy) * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) /
                                       static_cast<double>(vol));
                c[k * sz + j * sy + i] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResult(const std::vector<double>& c,
                           const int lx, const int ly, const int lz,
                           const int sy, const int sz,
                           int rank, MPI_Comm comm) {
    bool localFinite = true;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();

    for (int z = 1; z <= lz; ++z) {
        for (int y = 1; y <= ly; ++y) {
            for (int x = 1; x <= lx; ++x) {
                const double val = c[z * sz + y * sy + x];
                if (std::isnan(val) || std::isinf(val)) localFinite = false;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }

    int finiteFlag = localFinite ? 0 : 1;
    int globalFinite = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Reduce(&finiteFlag, &globalFinite, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    bool valid = true;
    if (rank == 0) {
        if (globalFinite) {
            printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (globalMax > 10.0 || globalMin < -10.0) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }
    }

    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, comm);
    return valid;
}

static void printUsage(const char* progName) {
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int nx = 64;
    int ny = 0;
    int nz = 0;
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    int activeRanks = worldSize;
    int px = 1;
    int py = 1;
    int pz = 1;
    while (activeRanks > 0 &&
           !factor3D(activeRanks, nx, ny, nz, px, py, pz)) {
        --activeRanks;
    }
    if (activeRanks < 1) activeRanks = 1;

    const bool active = worldRank < activeRanks;
    MPI_Comm comm = MPI_COMM_NULL;
    if (active) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, worldRank, &comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, worldRank, &comm);
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
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

    int dims[3] = {px, py, pz};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(comm, 3, dims, periods, 0, &cart);

    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart, rank, 3, coords);

    int lower[3];
    int upper[3];
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &lower[d], &upper[d]);
    }

    int globalDim[3] = {nx, ny, nz};
    int start[3] = {0, 0, 0};
    int len[3] = {0, 0, 0};
    for (int d = 0; d < 3; ++d) {
        const int base = globalDim[d] / dims[d];
        const int rem = globalDim[d] % dims[d];
        len[d] = base + (coords[d] < rem ? 1 : 0);
        start[d] = coords[d] * base + std::min(coords[d], rem);
    }

    const int lx = len[0];
    const int ly = len[1];
    const int lz = len[2];
    const int sy = lx + 2;
    const int sz = (lx + 2) * (ly + 2);
    const size_t localAlloc = static_cast<size_t>(sz) * static_cast<size_t>(lz + 2);

    // Allocate arrays
    std::vector<double> cold(localAlloc);
    std::vector<double> cnew(localAlloc);
    std::vector<double> mu(localAlloc);

    HaloTypes haloTypes = createHaloTypes(lx, ly, lz);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz,
                            start[0], start[1], start[2],
                            lx, ly, lz, sy, sz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double startTime = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, lx, ly, lz, sy, sz, lower, upper, haloTypes, cart);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, lx, ly, lz, sy, sz,
                                 dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        exchangeHalos(mu, lx, ly, lz, sy, sz, lower, upper, haloTypes, cart);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, lx, ly, lz, sy, sz,
                           D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(comm);
    const double endTime = MPI_Wtime();
    const double localDuration = endTime - startTime;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const size_t gridSize = static_cast<size_t>(nx) * ny * nz;

    std::vector<double> localData(static_cast<size_t>(lx) * ly * lz);
    size_t localCount = 0;
    for (int k = 1; k <= lz; ++k) {
        for (int j = 1; j <= ly; ++j) {
            for (int i = 1; i <= lx; ++i) {
                localData[localCount++] = cold[k * sz + j * sy + i];
            }
        }
    }

    int meta[6] = {start[0], start[1], start[2], lx, ly, lz};
    std::vector<int> allMeta(static_cast<size_t>(6) * size);
    MPI_Allgather(meta, 6, MPI_INT, allMeta.data(), 6, MPI_INT, comm);

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    long long totalCount = 0;
    for (int r = 0; r < size; ++r) {
        const long long c = static_cast<long long>(allMeta[r * 6 + 3]) *
                            allMeta[r * 6 + 4] * allMeta[r * 6 + 5];
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(totalCount);
        totalCount += c;
    }

    std::vector<double> gathered;
    double* recvBuf = nullptr;
    if (rank == 0) {
        gathered.resize(static_cast<size_t>(totalCount));
        recvBuf = gathered.data();
    }

    MPI_Gatherv(localData.data(), static_cast<int>(localData.size()), MPI_DOUBLE,
                recvBuf, rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, comm);

    std::vector<double> globalField;
    if (rank == 0) {
        globalField.resize(gridSize);
        for (int r = 0; r < size; ++r) {
            const int sx = allMeta[r * 6 + 0];
            const int syStart = allMeta[r * 6 + 1];
            const int szStart = allMeta[r * 6 + 2];
            const int rx = allMeta[r * 6 + 3];
            const int ry = allMeta[r * 6 + 4];
            const int rz = allMeta[r * 6 + 5];
            const double* chunk = gathered.data() + displs[r];
            size_t q = 0;
            for (int k = 0; k < rz; ++k) {
                const int gz = szStart + k;
                for (int j = 0; j < ry; ++j) {
                    const int gy = syStart + j;
                    const size_t base = (static_cast<size_t>(gz) * ny + gy) * nx + sx;
                    for (int i = 0; i < rx; ++i) {
                        globalField[base + i] = chunk[q++];
                    }
                }
            }
        }
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));

        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (printResults) {
            print_results(globalField, "Concentration");
        }
    }

    // Validation
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(cold, lx, ly, lz, sy, sz, rank, comm);
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    freeHaloTypes(haloTypes);
    MPI_Comm_free(&cart);
    MPI_Comm_free(&comm);
    MPI_Finalize();

    return valid ? 0 : 1;
}
