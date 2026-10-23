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

inline size_t part_start(const size_t n, const int p, const int i) noexcept {
    const size_t pi = static_cast<size_t>(p);
    const size_t base = n / pi;
    const size_t rem = n % pi;
    const size_t ii = static_cast<size_t>(i);
    return ii * base + std::min(ii, rem);
}

inline size_t part_count(const size_t n, const int p, const int i) noexcept {
    const size_t pi = static_cast<size_t>(p);
    const size_t base = n / pi;
    const size_t rem = n % pi;
    return base + (static_cast<size_t>(i) < rem ? 1 : 0);
}

struct Domain {
    size_t nx = 0, ny = 0, nz = 0;
    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};
    size_t lnx = 0, lny = 0, lnz = 0;
    size_t ox = 0, oy = 0, oz = 0;
    size_t pnx = 2, pny = 2, pnz = 2;
    bool active = false;
    int nbrLow[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbrHigh[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
};

static void chooseDims(const int nprocs, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    int best[3] = {1, 1, nprocs};
    long long bestActive = -1;
    double bestCost = std::numeric_limits<double>::infinity();

    for (int px = 1; px <= nprocs; ++px) {
        if (nprocs % px != 0) continue;
        const int remx = nprocs / px;
        for (int py = 1; py <= remx; ++py) {
            if (remx % py != 0) continue;
            const int pz = remx / py;

            const long long ax = std::min<long long>(px, static_cast<long long>(nx));
            const long long ay = std::min<long long>(py, static_cast<long long>(ny));
            const long long az = std::min<long long>(pz, static_cast<long long>(nz));
            const long long active = ax * ay * az;

            const double mx = static_cast<double>((nx + static_cast<size_t>(px) - 1) / static_cast<size_t>(px));
            const double my = static_cast<double>((ny + static_cast<size_t>(py) - 1) / static_cast<size_t>(py));
            const double mz = static_cast<double>((nz + static_cast<size_t>(pz) - 1) / static_cast<size_t>(pz));

            const double volume = mx * my * mz;
            const double surface = mx * my + my * mz + mx * mz;
            const double cost = volume + surface;

            if (active > bestActive || (active == bestActive && cost < bestCost)) {
                bestActive = active;
                bestCost = cost;
                best[0] = px;
                best[1] = py;
                best[2] = pz;
            }
        }
    }

    dims[0] = best[0];
    dims[1] = best[1];
    dims[2] = best[2];
}

static void setupNeighbors(Domain& d, const MPI_Comm cart) {
    for (int dim = 0; dim < 3; ++dim) {
        int src = MPI_PROC_NULL;
        int dst = MPI_PROC_NULL;
        MPI_Cart_shift(cart, dim, 1, &src, &dst);

        const size_t nlen = (dim == 0) ? d.nx : (dim == 1) ? d.ny : d.nz;

        int low = src;
        int high = dst;

        if (low != MPI_PROC_NULL) {
            const int nc = d.coords[dim] - 1;
            if (nc < 0 || static_cast<size_t>(nc) >= nlen) low = MPI_PROC_NULL;
        }
        if (high != MPI_PROC_NULL) {
            const int nc = d.coords[dim] + 1;
            if (nc < 0 || static_cast<size_t>(nc) >= nlen) high = MPI_PROC_NULL;
        }

        d.nbrLow[dim] = low;
        d.nbrHigh[dim] = high;
    }
}

static void createHaloTypes(const Domain& d, MPI_Datatype types[3]) {
    MPI_Datatype tmp;

    MPI_Type_vector(static_cast<int>(d.lny), 1, static_cast<int>(d.pnx), MPI_DOUBLE, &tmp);
    MPI_Type_create_hvector(static_cast<int>(d.lnz), 1,
                            static_cast<MPI_Aint>(d.pnx) * static_cast<MPI_Aint>(d.pny) *
                                static_cast<MPI_Aint>(sizeof(double)),
                            tmp, &types[0]);
    MPI_Type_commit(&types[0]);
    MPI_Type_free(&tmp);

    MPI_Type_vector(static_cast<int>(d.lnz), static_cast<int>(d.lnx),
                    static_cast<int>(d.pnx * d.pny), MPI_DOUBLE, &types[1]);
    MPI_Type_commit(&types[1]);

    MPI_Type_vector(static_cast<int>(d.lny), static_cast<int>(d.lnx),
                    static_cast<int>(d.pnx), MPI_DOUBLE, &types[2]);
    MPI_Type_commit(&types[2]);
}

static void haloExchange(std::vector<double>& a, const Domain& d,
                         const MPI_Datatype types[3], const MPI_Comm cart) {
    if (!d.active) return;

    const size_t pnx = d.pnx;
    const size_t plane = d.pnx * d.pny;
    auto P = [&](const size_t x, const size_t y, const size_t z) -> size_t {
        return z * plane + y * pnx + x;
    };

    MPI_Request reqs[12];
    int nreq = 0;

    for (int dim = 0; dim < 3; ++dim) {
        size_t hLow, hHigh, iLow, iHigh;
        if (dim == 0) {
            hLow = P(0, 1, 1);
            hHigh = P(d.lnx + 1, 1, 1);
            iLow = P(1, 1, 1);
            iHigh = P(d.lnx, 1, 1);
        } else if (dim == 1) {
            hLow = P(1, 0, 1);
            hHigh = P(1, d.lny + 1, 1);
            iLow = P(1, 1, 1);
            iHigh = P(1, d.lny, 1);
        } else {
            hLow = P(1, 1, 0);
            hHigh = P(1, 1, d.lnz + 1);
            iLow = P(1, 1, 1);
            iHigh = P(1, 1, d.lnz);
        }

        const int low = d.nbrLow[dim];
        if (low == MPI_PROC_NULL) {
            if (dim == 0) {
                for (size_t z = 1; z <= d.lnz; ++z)
                    for (size_t y = 1; y <= d.lny; ++y)
                        a[P(0, y, z)] = a[P(1, y, z)];
            } else if (dim == 1) {
                for (size_t z = 1; z <= d.lnz; ++z)
                    for (size_t x = 1; x <= d.lnx; ++x)
                        a[P(x, 0, z)] = a[P(x, 1, z)];
            } else {
                for (size_t y = 1; y <= d.lny; ++y)
                    for (size_t x = 1; x <= d.lnx; ++x)
                        a[P(x, y, 0)] = a[P(x, y, 1)];
            }
        } else {
            MPI_Irecv(&a[hLow], 1, types[dim], low, 100 + dim, cart, &reqs[nreq++]);
            MPI_Isend(&a[iLow], 1, types[dim], low, 100 + dim, cart, &reqs[nreq++]);
        }

        const int high = d.nbrHigh[dim];
        if (high == MPI_PROC_NULL) {
            if (dim == 0) {
                for (size_t z = 1; z <= d.lnz; ++z)
                    for (size_t y = 1; y <= d.lny; ++y)
                        a[P(d.lnx + 1, y, z)] = a[P(d.lnx, y, z)];
            } else if (dim == 1) {
                for (size_t z = 1; z <= d.lnz; ++z)
                    for (size_t x = 1; x <= d.lnx; ++x)
                        a[P(x, d.lny + 1, z)] = a[P(x, d.lny, z)];
            } else {
                for (size_t y = 1; y <= d.lny; ++y)
                    for (size_t x = 1; x <= d.lnx; ++x)
                        a[P(x, y, d.lnz + 1)] = a[P(x, y, d.lnz)];
            }
        } else {
            MPI_Irecv(&a[hHigh], 1, types[dim], high, 100 + dim, cart, &reqs[nreq++]);
            MPI_Isend(&a[iHigh], 1, types[dim], high, 100 + dim, cart, &reqs[nreq++]);
        }
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

// Compute chemical potential
static void computeChemicalPotential(const double* __restrict__ c,
                                     double* __restrict__ mu,
                                     const Domain& d,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA,
                                     const double e_BB, const double e_AB) {
    const size_t pnx = d.pnx;
    const size_t plane = d.pnx * d.pny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    for (size_t z = 0; z < d.lnz; ++z) {
        const size_t zbase = (z + 1) * plane;
        for (size_t y = 0; y < d.lny; ++y) {
            const size_t base = zbase + (y + 1) * pnx + 1;
            for (size_t x = 0; x < d.lnx; ++x) {
                const size_t i = base + x;
                const double cv = c[i];

                const double cxx = (c[i + 1] + c[i - 1] - 2.0 * cv) / dx2;
                const double cyy = (c[i + pnx] + c[i - pnx] - 2.0 * cv) / dy2;
                const double czz = (c[i + plane] + c[i - plane] - 2.0 * cv) / dz2;
                const double lap = cxx + cyy + czz;

                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv
                      - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update step
static void cahnHilliardUpdate(double* __restrict__ cnew,
                               const double* __restrict__ cold,
                               const double* __restrict__ mu,
                               const Domain& d,
                               const double D, const double dt,
                               const double dx, const double dy, const double dz) {
    const size_t pnx = d.pnx;
    const size_t plane = d.pnx * d.pny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    for (size_t z = 0; z < d.lnz; ++z) {
        const size_t zbase = (z + 1) * plane;
        for (size_t y = 0; y < d.lny; ++y) {
            const size_t base = zbase + (y + 1) * pnx + 1;
            for (size_t x = 0; x < d.lnx; ++x) {
                const size_t i = base + x;

                const double mxx = (mu[i + 1] + mu[i - 1] - 2.0 * mu[i]) / dx2;
                const double myy = (mu[i + pnx] + mu[i - pnx] - 2.0 * mu[i]) / dy2;
                const double mzz = (mu[i + plane] + mu[i - plane] - 2.0 * mu[i]) / dz2;
                const double lap = mxx + myy + mzz;

                cnew[i] = cold[i] + dt * D * lap;
            }
        }
    }
}

// Initialize concentration field
static void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t vol = d.nx * d.ny * d.nz;
    const size_t nxny = d.nx * d.ny;
    const size_t pnx = d.pnx;
    const size_t plane = d.pnx * d.pny;

    for (size_t z = 0; z < d.lnz; ++z) {
        const size_t gz = d.oz + z;
        const size_t zbase = (z + 1) * plane;
        for (size_t y = 0; y < d.lny; ++y) {
            const size_t gy = d.oy + y;
            const size_t gbase = gz * nxny + gy * d.nx + d.ox;
            const size_t base = zbase + (y + 1) * pnx + 1;
            for (size_t x = 0; x < d.lnx; ++x) {
                const size_t linear_id = gbase + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[base + x] = -1.0 + 2.0 * pseudo;
            }
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

static void gatherGlobal(const std::vector<double>& local, const Domain& d,
                         const MPI_Comm cart, const int rank, const int nprocs,
                         std::vector<double>& global) {
    const size_t pnx = d.pnx;
    const size_t plane = d.pnx * d.pny;
    const size_t localCells = d.lnx * d.lny * d.lnz;

    std::vector<double> sendbuf(localCells);
    if (d.active) {
        size_t pos = 0;
        for (size_t z = 0; z < d.lnz; ++z) {
            const size_t zbase = (z + 1) * plane;
            for (size_t y = 0; y < d.lny; ++y) {
                const size_t base = zbase + (y + 1) * pnx + 1;
                for (size_t x = 0; x < d.lnx; ++x) {
                    sendbuf[pos++] = local[base + x];
                }
            }
        }
    }

    const int sendcount = static_cast<int>(localCells);

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> recvbuf;

    if (rank == 0) {
        counts.resize(static_cast<size_t>(nprocs));
        displs.resize(static_cast<size_t>(nprocs));
        long long total = 0;
        for (int r = 0; r < nprocs; ++r) {
            int c[3];
            MPI_Cart_coords(cart, r, 3, c);
            const size_t lx = part_count(d.nx, d.dims[0], c[0]);
            const size_t ly = part_count(d.ny, d.dims[1], c[1]);
            const size_t lz = part_count(d.nz, d.dims[2], c[2]);
            const long long cnt = static_cast<long long>(lx) * static_cast<long long>(ly) *
                                  static_cast<long long>(lz);
            counts[static_cast<size_t>(r)] = static_cast<int>(cnt);
            displs[static_cast<size_t>(r)] = static_cast<int>(total);
            total += cnt;
        }
        recvbuf.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(sendbuf.data(), sendcount, MPI_DOUBLE,
                rank == 0 ? recvbuf.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, cart);

    if (rank == 0) {
        global.assign(d.nx * d.ny * d.nz, 0.0);
        const size_t nxny = d.nx * d.ny;

        for (int r = 0; r < nprocs; ++r) {
            int c[3];
            MPI_Cart_coords(cart, r, 3, c);
            const size_t lx = part_count(d.nx, d.dims[0], c[0]);
            const size_t ly = part_count(d.ny, d.dims[1], c[1]);
            const size_t lz = part_count(d.nz, d.dims[2], c[2]);
            if (lx == 0 || ly == 0 || lz == 0) continue;

            const size_t ox = part_start(d.nx, d.dims[0], c[0]);
            const size_t oy = part_start(d.ny, d.dims[1], c[1]);
            const size_t oz = part_start(d.nz, d.dims[2], c[2]);

            const double* src = recvbuf.data() + displs[static_cast<size_t>(r)];
            size_t pos = 0;
            for (size_t z = 0; z < lz; ++z) {
                for (size_t y = 0; y < ly; ++y) {
                    const size_t gbase = (oz + z) * nxny + (oy + y) * d.nx + ox;
                    for (size_t x = 0; x < lx; ++x) {
                        global[gbase + x] = src[pos++];
                    }
                }
            }
        }
    }
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

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badArg = false;
    const char* badArgText = nullptr;

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
            showHelp = true;
            break;
        } else {
            badArg = true;
            badArgText = argv[i];
            break;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    if (badArg) {
        if (rank == 0) {
            printf("Unknown option: %s\n", badArgText);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    int dims[3];
    chooseDims(nprocs, nx, ny, nz, dims);
    const int periods[3] = {0, 0, 0};

    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    int cartRank = 0;
    MPI_Comm_rank(cart, &cartRank);

    Domain d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    for (int i = 0; i < 3; ++i) d.dims[i] = dims[i];
    MPI_Cart_coords(cart, cartRank, 3, d.coords);

    d.lnx = part_count(nx, dims[0], d.coords[0]);
    d.lny = part_count(ny, dims[1], d.coords[1]);
    d.lnz = part_count(nz, dims[2], d.coords[2]);
    d.ox = part_start(nx, dims[0], d.coords[0]);
    d.oy = part_start(ny, dims[1], d.coords[1]);
    d.oz = part_start(nz, dims[2], d.coords[2]);
    d.pnx = d.lnx + 2;
    d.pny = d.lny + 2;
    d.pnz = d.lnz + 2;
    d.active = (d.lnx > 0 && d.lny > 0 && d.lnz > 0);

    setupNeighbors(d, cart);

    if (rank == 0) {
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

    const size_t gridSize = nx * ny * nz;
    const size_t localSize = d.pnx * d.pny * d.pnz;

    // Allocate arrays
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    MPI_Datatype types[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    if (d.active) createHaloTypes(d, types);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    if (d.active) initializeConcentration(cold, d);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        haloExchange(cold, d, types, cart);

        // Compute chemical potential
        if (d.active) {
            computeChemicalPotential(cold.data(), mu.data(), d, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
        }

        haloExchange(mu, d, types, cart);

        // Update concentration
        if (d.active) {
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d,
                               D, dt, dx, dy, dz);
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    const double end = MPI_Wtime();
    const double localDuration = end - start;
    double maxDuration = localDuration;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        const long long millis = static_cast<long long>(maxDuration * 1000.0);
        printf("Computation time: %lld ms\n", millis);

        const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = cellUpdates / maxDuration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int rc = 0;

    if (printResults || validate) {
        std::vector<double> global;
        gatherGlobal(cold, d, cart, cartRank, nprocs, global);

        if (cartRank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(global, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(global, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                    rc = 0;
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, cart);

    if (d.active) {
        for (int i = 0; i < 3; ++i) MPI_Type_free(&types[i]);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return rc;
}
