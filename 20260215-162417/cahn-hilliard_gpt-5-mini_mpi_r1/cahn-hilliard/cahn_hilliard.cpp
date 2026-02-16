#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Local 3D index calculation (z fast: z*(nx*ny)+y*nx+x)
inline constexpr size_t idx3_local(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions using local arrays with halos
inline double computeLaplacianLocal(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local_with_halo,
                                    const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z_local) {
    // z_local is index INCLUDING halo (0..nz_local+1)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z_local < nz_local_with_halo - 1) ? z_local + 1 : z_local;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z_local > 0) ? z_local - 1 : 0;

    const double center = c[idx3_local(x, y, z_local, nx, ny)];
    const double cxx = (c[idx3_local(xp, y, z_local, nx, ny)] + c[idx3_local(xn, y, z_local, nx, ny)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3_local(x, yp, z_local, nx, ny)] + c[idx3_local(x, yn, z_local, nx, ny)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3_local(x, y, zp, nx, ny)] + c[idx3_local(x, y, zn, nx, ny)] - 2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

// Exchange Z-direction halo planes (one plane = nx*ny contiguous doubles)
static void exchange_z_planes(std::vector<double>& buf, const size_t nx, const size_t ny, const int rank, const int size, const int prev, const int next, const size_t nz_local) {
    const int plane = static_cast<int>(nx * ny);
    MPI_Status st1, st2;

    // Send first interior plane (z=1) to prev -> receive into z=0
    if (prev != MPI_PROC_NULL) {
        MPI_Sendrecv(&buf[idx3_local(0,0,1,nx,ny)], plane, MPI_DOUBLE, prev, 0,
                     &buf[idx3_local(0,0,0,nx,ny)], plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &st1);
    } else {
        // Clamp: copy first interior to halo
        std::memcpy(&buf[idx3_local(0,0,0,nx,ny)], &buf[idx3_local(0,0,1,nx,ny)], plane * sizeof(double));
    }

    // Send last interior plane (z=nz_local) to next -> receive into z=nz_local+1
    if (next != MPI_PROC_NULL) {
        MPI_Sendrecv(&buf[idx3_local(0,0,nz_local,nx,ny)], plane, MPI_DOUBLE, next, 1,
                     &buf[idx3_local(0,0,nz_local+1,nx,ny)], plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &st2);
    } else {
        // Clamp: copy last interior to halo
        std::memcpy(&buf[idx3_local(0,0,nz_local+1,nx,ny)], &buf[idx3_local(0,0,nz_local,nx,ny)], plane * sizeof(double));
    }
}

// Compute chemical potential on local domain (excluding halos)
void computeChemicalPotentialLocal(const std::vector<double>& cold_local, std::vector<double>& mu_local,
                                   const size_t nx, const size_t ny, const size_t nz_local,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nzp1 = nz_local + 2; // including halos
    for (size_t zl = 1; zl <= nz_local; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, zl, nx, ny);
                const double cv = cold_local[idx];
                mu_local[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv
                              - gamma * computeLaplacianLocal(cold_local, nx, ny, nzp1, dx, dy, dz, x, y, zl);
            }
        }
    }
}

// Cahn-Hilliard update on local domain (excluding halos)
void cahnHilliardUpdateLocal(std::vector<double>& cnew_local, const std::vector<double>& cold_local,
                             const std::vector<double>& mu_local,
                             const size_t nx, const size_t ny, const size_t nz_local,
                             const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nzp1 = nz_local + 2;
    for (size_t zl = 1; zl <= nz_local; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, zl, nx, ny);
                cnew_local[idx] = cold_local[idx] + dt * D * computeLaplacianLocal(mu_local, nx, ny, nzp1, dx, dy, dz, x, y, zl);
            }
        }
    }
}

// Initialize concentration field for local partition
void initializeConcentrationLocal(std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t nz_local,
                                  const size_t z_start_global, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    for (size_t zl = 1; zl <= nz_local; ++zl) {
        size_t gz = z_start_global + (zl - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx_local = idx3_local(x, y, zl, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c_local[idx_local] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResultGlobal(const std::vector<double>& local_c, const size_t nx, const size_t ny, const size_t nz_local, const int rank, const int size) {
    // Local checks
    bool local_bad = false;
    double local_min = local_c[idx3_local(0,0,1,nx,ny)];
    double local_max = local_min;
    for (size_t zl = 1; zl <= nz_local; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double v = local_c[idx3_local(x,y,zl,nx,ny)];
                if (std::isnan(v) || std::isinf(v)) local_bad = true;
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }

    int any_bad = local_bad ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &any_bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (any_bad != 0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    if (global_max > 10.0 || global_min < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    if (progName) printf("Usage: %s [options]\n", progName);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0 to keep semantics)
    if (rank == 0) {
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
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else if (i > 0) {
                // ignore unknown on other ranks
            }
        }
    }
    // Broadcast parsed options to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

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

    const size_t nz_global = nz;
    const size_t global_grid = nx * ny * nz_global;

    // Partition in Z dimension (block distribution)
    size_t base = nz_global / static_cast<size_t>(size);
    size_t rem = nz_global % static_cast<size_t>(size);
    size_t nz_local = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    size_t z_start = (base * static_cast<size_t>(rank)) + std::min(static_cast<size_t>(rank), rem);

    // Local arrays include 2 halo planes (z=0 and z=nz_local+1)
    const size_t nz_local_with_halo = nz_local + 2;
    const size_t local_size_with_halo = nx * ny * nz_local_with_halo;

    std::vector<double> cold_local(local_size_with_halo, 0.0);
    std::vector<double> cnew_local(local_size_with_halo, 0.0);
    std::vector<double> mu_local(local_size_with_halo, 0.0);

    // Initialize local concentration (interior only)
    initializeConcentrationLocal(cold_local, nx, ny, nz_local, z_start, nz_global);

    // Setup neighbor ranks
    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Warm up halo exchange for initial boundaries (clamping or neighbor copy)
    exchange_z_planes(cold_local, nx, ny, rank, size, prev, next, nz_local);

    // Synchronize and start timing using MPI_Wtime
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Ensure halos of cold are up-to-date
        exchange_z_planes(cold_local, nx, ny, rank, size, prev, next, nz_local);

        // Compute chemical potential on local interior
        computeChemicalPotentialLocal(cold_local, mu_local, nx, ny, nz_local, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange halos of mu before update
        exchange_z_planes(mu_local, nx, ny, rank, size, prev, next, nz_local);

        // Update concentration on local interior
        cahnHilliardUpdateLocal(cnew_local, cold_local, mu_local, nx, ny, nz_local, D, dt, dx, dy, dz);

        // Swap pointers (swap vectors)
        cold_local.swap(cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // Reduce timing to root (take max)
    double t_max = 0.0;
    MPI_Reduce(&elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f s\n", t_max);
        double cellUpdates = static_cast<double>(global_grid) * iterations;
        double mcups = cellUpdates / t_max / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to root for printing/validation
    // Prepare sendcounts and displacements
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t local_nz_r = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        sendcounts[r] = static_cast<int>(local_nz_r * nx * ny);
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + sendcounts[r-1];

    std::vector<double> gather_buf;
    if (rank == 0) gather_buf.resize(static_cast<size_t>(global_grid));

    // Prepare local contiguous buffer (interior only, without halos)
    std::vector<double> local_interior(sendcounts[rank]);
    size_t pos = 0;
    for (size_t zl = 1; zl <= nz_local; ++zl) {
        size_t base_idx = idx3_local(0,0,zl,nx,ny);
        std::memcpy(&local_interior[pos], &cold_local[base_idx], nx * ny * sizeof(double));
        pos += nx * ny;
    }

    MPI_Gatherv(local_interior.data(), static_cast<int>(local_interior.size()), MPI_DOUBLE,
                gather_buf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(gather_buf, "Concentration");
    }

    if (validate) {
        bool ok = validateResultGlobal(cold_local, nx, ny, nz_local, rank, size);
        if (rank == 0) {
            if (ok) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
        MPI_Finalize();
        return ok ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
