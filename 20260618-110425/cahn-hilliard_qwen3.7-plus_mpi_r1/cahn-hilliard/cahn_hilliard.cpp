#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (z includes ghost layers in local coordinates)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost layers between neighboring MPI ranks along Z.
// Local z layout: [0] = bottom ghost, [1..nz_local] = owned, [nz_local+1] = top ghost.
// Clamped (zero-gradient) BCs applied at global domain boundaries.
static void exchangeGhosts(double* __restrict__ data, size_t nz_local,
                           size_t nx, size_t ny,
                           int rank, int nprocs, MPI_Comm comm) {
    const int plane = static_cast<int>(nx * ny);
    const size_t plane_bytes = static_cast<size_t>(plane) * sizeof(double);

    double* bot_ghost  = data;                                    // local z=0
    double* bot_owned  = data + plane;                            // local z=1
    double* top_owned  = data + nz_local * static_cast<size_t>(plane); // local z=nz_local
    double* top_ghost  = data + (nz_local + 1) * static_cast<size_t>(plane); // local z=nz_local+1

    MPI_Request reqs[4];
    int nreqs = 0;

    // Post receives first so incoming data can be buffered immediately
    if (rank > 0)
        MPI_Irecv(bot_ghost, plane, MPI_DOUBLE, rank - 1, 10, comm, &reqs[nreqs++]);
    if (rank < nprocs - 1)
        MPI_Irecv(top_ghost, plane, MPI_DOUBLE, rank + 1, 20, comm, &reqs[nreqs++]);

    // Post sends
    if (rank < nprocs - 1)
        MPI_Isend(top_owned, plane, MPI_DOUBLE, rank + 1, 10, comm, &reqs[nreqs++]);
    if (rank > 0)
        MPI_Isend(bot_owned, plane, MPI_DOUBLE, rank - 1, 20, comm, &reqs[nreqs++]);

    // Clamped BCs at global boundaries (overlaps with communication)
    if (rank == 0)
        std::memcpy(bot_ghost, bot_owned, plane_bytes);
    if (rank == nprocs - 1)
        std::memcpy(top_ghost, top_owned, plane_bytes);

    if (nreqs > 0)
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
}

// Compute Laplacian at a local grid point using ghost layers for Z neighbors.
// local_z is in [1, nz_local]; ghost layers at 0 and nz_local+1 are pre-populated.
inline double computeLaplacian(const double* __restrict__ c,
                               const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t local_z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0)     ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0)     ? y - 1 : 0;
    const size_t zp = local_z + 1;
    const size_t zn = local_z - 1;

    const size_t nxy = nx * ny;
    const size_t base = local_z * nxy;
    const double c_center = c[base + y * nx + x];

    const double cxx = (c[base + y * nx + xp] + c[base + y * nx + xn] - 2.0 * c_center) / (dx * dx);
    const double cyy = (c[base + yp * nx + x] + c[base + yn * nx + x] - 2.0 * c_center) / (dy * dy);
    const double czz = (c[(zp) * nxy + y * nx + x] + c[(zn) * nxy + y * nx + x] - 2.0 * c_center) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential over locally owned Z-layers [1..nz_local]
static void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                     const size_t nx, const size_t ny, const size_t nz_local,
                                     const double dx, const double dy, const double dz,
                                     const double gamma,
                                     const double e_AA, const double e_BB, const double e_AB) {
    const size_t nxy = nx * ny;
    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t zbase = z * nxy;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = zbase + y * nx + x;
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard concentration update over locally owned Z-layers [1..nz_local]
static void cahnHilliardUpdate(double* __restrict__ cnew,
                               const double* __restrict__ cold,
                               const double* __restrict__ mu,
                               const size_t nx, const size_t ny, const size_t nz_local,
                               const double D, const double dt,
                               const double dx, const double dy, const double dz) {
    const size_t nxy = nx * ny;
    const double dtD = dt * D;
    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t zbase = z * nxy;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = zbase + y * nx + x;
                cnew[idx] = cold[idx] + dtD *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field using global coordinates for reproducibility
static void initializeConcentration(double* __restrict__ c,
                                    const size_t nx, const size_t ny, const size_t nz_local,
                                    const size_t z_offset, const size_t vol) {
    const size_t nxy = nx * ny;
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        const size_t gz = z_offset + (lz - 1);
        const size_t zbase = lz * nxy;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = gz * nxy + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol)
                                       / static_cast<double>(vol));
                c[zbase + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResult(const std::vector<double>& c,
                           [[maybe_unused]] const size_t nx,
                           [[maybe_unused]] const size_t ny,
                           [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0], maxVal = c[0];
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // ---- Parse arguments on rank 0 and broadcast ----
    int i_nx = 64, i_ny = 0, i_nz = 0;
    int iterations = 20;
    int validate = 0, printResults = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc)      i_nx = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) i_ny = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) i_nz = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-v") == 0)  validate = 1;
            else if (std::strcmp(argv[i], "-r") == 0)  printResults = 1;
            else if (std::strcmp(argv[i], "-h") == 0)  { printUsage(argv[0]); MPI_Finalize(); return 0; }
            else {
                if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
                MPI_Finalize();
                return 1;
            }
        }
        if (i_ny == 0) i_ny = i_nx;
        if (i_nz == 0) i_nz = i_nx;
    }

    MPI_Bcast(&i_nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i_ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i_nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int flags[2] = {validate, printResults};
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0]; printResults = flags[1];

    const size_t nx = static_cast<size_t>(i_nx);
    const size_t ny = static_cast<size_t>(i_ny);
    const size_t nz = static_cast<size_t>(i_nz);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- 1-D slab decomposition along Z ----
    const size_t base_nz   = nz / static_cast<size_t>(nprocs);
    const size_t remainder = nz % static_cast<size_t>(nprocs);
    const size_t r         = static_cast<size_t>(rank);
    const size_t nz_local  = (r < remainder) ? base_nz + 1 : base_nz;
    const size_t z_offset  = (r < remainder)
                             ? r * (base_nz + 1)
                             : remainder * (base_nz + 1) + (r - remainder) * base_nz;

    if (nz_local == 0) {
        printf("Rank %d: no Z-layers assigned (nz=%zu < nprocs=%d). Aborting.\n",
               rank, nz, nprocs);
        MPI_Finalize();
        return 1;
    }

    // ---- Physical parameters ----
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma_val = 0.5, D = 1.0;

    const size_t global_vol = nx * ny * nz;
    const size_t local_vol  = (nz_local + 2) * nx * ny; // +2 for ghost layers

    // ---- Allocate local arrays (with ghost layers) ----
    std::vector<double> cold(local_vol, 0.0);
    std::vector<double> cnew(local_vol, 0.0);
    std::vector<double> mu(local_vol, 0.0);

    // ---- Initialize concentration field ----
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold.data(), nx, ny, nz_local, z_offset, global_vol);

    // ---- Run simulation ----
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers for concentration
        exchangeGhosts(cold.data(), nz_local, nx, ny, rank, nprocs, MPI_COMM_WORLD);

        // Compute chemical potential (owned layers)
        computeChemicalPotential(cold.data(), mu.data(), nx, ny, nz_local,
                                 dx, dy, dz, gamma_val, e_AA, e_BB, e_AB);

        // Exchange ghost layers for chemical potential
        exchangeGhosts(mu.data(), nz_local, nx, ny, rank, nprocs, MPI_COMM_WORLD);

        // Update concentration (owned layers)
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, nz_local,
                           D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // ---- Gather results to rank 0 for output / validation ----
    const int local_count = static_cast<int>(nz_local * nx * ny);
    const double* send_buf = cold.data() + nx * ny; // skip bottom ghost

    std::vector<int> recvcounts, displs;
    std::vector<double> global_data;
    if (rank == 0) {
        recvcounts.resize(nprocs);
        displs.resize(nprocs);
        global_data.resize(global_vol);
    }

    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int p = 1; p < nprocs; ++p)
            displs[p] = displs[p - 1] + recvcounts[p - 1];
    }

    MPI_Gatherv(send_buf, local_count, MPI_DOUBLE,
                global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- Report timing and results on rank 0 ----
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        long long ms = duration.count();

        printf("Computation time: %lld ms\n", ms);

        double cellUpdates = static_cast<double>(global_vol) * iterations;
        double mcups = (ms > 0) ? cellUpdates / (ms / 1000.0) / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults)
            print_results(global_data, "Concentration");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_data, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
