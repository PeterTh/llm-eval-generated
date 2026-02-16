#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Local 3D index for arrays with ghost layers in Z (z from 0..local_nz+1)
inline constexpr size_t idx3_local(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny, const size_t nz_local_with_ghost) noexcept {
    (void)nz_local_with_ghost;
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost layers in Z for a slab-decomposed array
void exchange_ghost_layers(std::vector<double>& arr, const size_t nx, const size_t ny, const size_t local_nz, int rank, int size, MPI_Comm comm) {
    const size_t slice = nx * ny;
    // send first interior z=1 to rank-1, receive into z=0
    if (size == 1) {
        // No communication: apply clamped boundary by copying edge into ghost
        for (size_t i = 0; i < slice; ++i) {
            arr[i] = arr[slice + i]; // z=0 <- z=1
            arr[(local_nz + 1) * slice + i] = arr[local_nz * slice + i]; // last ghost <- last interior
        }
        return;
    }

    MPI_Status status;
    // Top neighbor (rank-1)
    if (rank > 0) {
        MPI_Sendrecv(&arr[idx3_local(0,0,1,nx,ny,local_nz+2)], slice, MPI_DOUBLE, rank-1, 0,
                     &arr[idx3_local(0,0,0,nx,ny,local_nz+2)], slice, MPI_DOUBLE, rank-1, 1,
                     comm, &status);
    } else {
        // clamped
        memcpy(&arr[idx3_local(0,0,0,nx,ny,local_nz+2)], &arr[idx3_local(0,0,1,nx,ny,local_nz+2)], slice * sizeof(double));
    }

    // Bottom neighbor (rank+1)
    if (rank < size - 1) {
        MPI_Sendrecv(&arr[idx3_local(0,0,local_nz,nx,ny,local_nz+2)], slice, MPI_DOUBLE, rank+1, 1,
                     &arr[idx3_local(0,0,local_nz+1,nx,ny,local_nz+2)], slice, MPI_DOUBLE, rank+1, 0,
                     comm, &status);
    } else {
        // clamped
        memcpy(&arr[idx3_local(0,0,local_nz+1,nx,ny,local_nz+2)], &arr[idx3_local(0,0,local_nz,nx,ny,local_nz+2)], slice * sizeof(double));
    }
}

// Compute Laplacian with clamped boundaries using local arrays with ghost layers
inline double computeLaplacian_local(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz_with_ghost,
                                     const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    // z is local index including ghost (0..local_nz+1)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1;

    const double c_center = c[idx3_local(x,y,z,nx,ny,local_nz_with_ghost)];
    const double cxx = (c[idx3_local(xp, y, z, nx, ny, local_nz_with_ghost)] + c[idx3_local(xn, y, z, nx, ny, local_nz_with_ghost)] - 2.0 * c_center) / (dx * dx);
    const double cyy = (c[idx3_local(x, yp, z, nx, ny, local_nz_with_ghost)] + c[idx3_local(x, yn, z, nx, ny, local_nz_with_ghost)] - 2.0 * c_center) / (dy * dy);
    const double czz = (c[idx3_local(x, y, zp, nx, ny, local_nz_with_ghost)] + c[idx3_local(x, y, zn, nx, ny, local_nz_with_ghost)] - 2.0 * c_center) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local slabs (interior z from 1..local_nz)
void computeChemicalPotential_local(const std::vector<double>& c_local, std::vector<double>& mu_local,
                                   const size_t nx, const size_t ny, const size_t local_nz,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t local_nz_with_ghost = local_nz + 2;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x,y,z,nx,ny,local_nz_with_ghost);
                const double cv = c_local[idx];
                mu_local[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv
                                - gamma * computeLaplacian_local(c_local, nx, ny, local_nz_with_ghost, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update on local slab
void cahnHilliardUpdate_local(std::vector<double>& cnew_local, const std::vector<double>& cold_local,
                              const std::vector<double>& mu_local,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t local_nz_with_ghost = local_nz + 2;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x,y,z,nx,ny,local_nz_with_ghost);
                cnew_local[idx] = cold_local[idx] + dt * D * computeLaplacian_local(mu_local, nx, ny, local_nz_with_ghost, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field on local slab using global z to make deterministic same as serial
void initializeConcentration_local(std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t local_nz, const size_t global_z_start) {
    const size_t slice = nx * ny;
    const size_t vol = nx * ny * (local_nz + 0); // not used for modulus across global domain but keep local
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        size_t gz = global_z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx_local = idx3_local(x,y,lz,nx,ny,local_nz+2);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % (nx * ny * (local_nz + 0 ? 1 : 1))) / static_cast<double>(nx * ny * (local_nz + 0 ? 1 : 1)));
                c_local[idx_local] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult_local(const std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t local_nz) {
    const size_t local_nz_with_ghost = local_nz + 2;
    // Check for NaN/Inf and local min/max
    double local_min = 1e300;
    double local_max = -1e300;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double v = c_local[idx3_local(x,y,z,nx,ny,local_nz_with_ghost)];
                if (std::isnan(v) || std::isinf(v)) return false;
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }
    // set outputs via MPI reductions outside
    (void)local_min; (void)local_max;
    return true;
}

void printUsage(const char* progName) {
    if (MPI::COMM_WORLD.Get_rank() == 0) {
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
}

int main(int argc, char** argv) {
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
    
    // Parse command line arguments (only rank 0 prints errors/help)
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
    
    size_t global_gridSize = nx * ny * nz;

    // Decompose along Z
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    int base = nz / size;
    int rem = nz % size;
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int lnz = base + (r < rem ? 1 : 0);
        counts[r] = lnz * nx * ny;
        displs[r] = offset * nx * ny;
        offset += lnz;
    }
    int local_nz = counts[rank] / (nx * ny);
    // local arrays include 2 ghost layers in z
    const size_t local_nz_with_ghost = local_nz + 2;
    const size_t local_size_with_ghost = local_nz_with_ghost * nx * ny;

    std::vector<double> cold_local(local_size_with_ghost, 0.0);
    std::vector<double> cnew_local(local_size_with_ghost, 0.0);
    std::vector<double> mu_local(local_size_with_ghost, 0.0);

    // Initialize local concentration deterministically matching global layout
    // compute global start z for this rank
    int gz_start = 0;
    for (int r = 0; r < rank; ++r) gz_start += counts[r] / (nx * ny);
    initializeConcentration_local(cold_local, nx, ny, local_nz, gz_start);

    if (rank == 0) printf("Initializing concentration field...\n");

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Main time-stepping loop
    for (int t = 0; t < iterations; ++t) {
        // exchange cold ghost layers
        exchange_ghost_layers(cold_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        // compute chemical potential (fills mu interior)
        computeChemicalPotential_local(cold_local, mu_local, nx, ny, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        // exchange mu ghost layers
        exchange_ghost_layers(mu_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        // update concentration into cnew (interior only)
        cahnHilliardUpdate_local(cnew_local, cold_local, mu_local, nx, ny, local_nz, D, dt, dx, dy, dz);
        // swap local buffers
        std::swap(cold_local, cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long ms = (long)(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double cellUpdates = (double)global_gridSize * iterations;
        double mcups = cellUpdates / (max_elapsed) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for printing/validation
    std::vector<double> full_result;
    if (rank == 0) full_result.resize(global_gridSize);
    // prepare send buffer: interior only
    std::vector<int> counts_bytes(size);
    for (int r = 0; r < size; ++r) counts_bytes[r] = counts[r];
    MPI_Gatherv(&cold_local[idx3_local(0,0,1,nx,ny,local_nz_with_ghost)], counts[rank], MPI_DOUBLE,
                full_result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(full_result, "Concentration");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool local_ok = validateResult_local(cold_local, nx, ny, local_nz);
        int ok_int = local_ok ? 1 : 0;
        int global_ok = 0;
        MPI_Reduce(&ok_int, &global_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (global_ok) {
                // also compute global min/max for reporting
                double local_min = 1e300, local_max = -1e300;
                for (size_t z = 0; z < nz; ++z) (void)z; // silence
                for (size_t i = 0; i < global_gridSize; ++i) {
                    local_min = std::min(local_min, full_result[i]);
                    local_max = std::max(local_max, full_result[i]);
                }
                printf("Concentration range: [%.6f, %.6f]\n", local_min, local_max);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
