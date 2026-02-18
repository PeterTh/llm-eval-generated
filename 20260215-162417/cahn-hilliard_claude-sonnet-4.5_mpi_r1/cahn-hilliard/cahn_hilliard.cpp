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

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z,
                        const size_t z_global, const size_t nz_global) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // Local z boundaries
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    // Apply clamping at global boundaries
    size_t zp_use = zp;
    size_t zn_use = zn;
    if (z_global == 0) zn_use = 0;
    if (z_global == nz_global - 1) zp_use = z;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp_use, nx, ny)] + c[idx3(x, y, zn_use, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t z_offset, const size_t nz_global) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                const size_t z_global = z_offset + z;
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z, z_global, nz_global);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t z_offset, const size_t nz_global) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t z_global = z_offset + z;
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z, z_global, nz_global);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z_offset, const size_t nx_global, const size_t ny_global, const size_t nz_global) {
    const size_t vol = nx_global * ny_global * nz_global;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value using global coordinates
                const size_t z_global = z_offset + z;
                const size_t linear_id = z_global * (nx_global * ny_global) + y * nx_global + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    
    // Global dimensions
    const size_t nx_global = nx;
    const size_t ny_global = ny;
    const size_t nz_global = nz;
    
    // Domain decomposition in Z direction with ghost cells
    size_t nz_per_rank = nz_global / size;
    size_t remainder = nz_global % size;
    
    // Distribute remainder to first ranks
    size_t z_start, z_end, nz_local;
    if (rank < (int)remainder) {
        nz_local = nz_per_rank + 1;
        z_start = rank * nz_local;
    } else {
        nz_local = nz_per_rank;
        z_start = remainder * (nz_per_rank + 1) + (rank - remainder) * nz_per_rank;
    }
    z_end = z_start + nz_local;
    
    // Add ghost cells
    const bool has_lower = (rank > 0);
    const bool has_upper = (rank < size - 1);
    const size_t nz_with_ghost = nz_local + (has_lower ? 1 : 0) + (has_upper ? 1 : 0);
    const size_t ghost_lower = has_lower ? 1 : 0;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx_global, ny_global, nz_global);
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
    
    size_t local_gridSize = nx_global * ny_global * nz_with_ghost;
    
    // Allocate arrays
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx_global, ny_global, nz_with_ghost, 
                           z_start - ghost_lower, nx_global, ny_global, nz_global);
    
    // Setup neighbor communication
    const int rank_lower = has_lower ? rank - 1 : MPI_PROC_NULL;
    const int rank_upper = has_upper ? rank + 1 : MPI_PROC_NULL;
    const size_t plane_size = nx_global * ny_global;
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost cells for concentration field
        if (has_lower) {
            MPI_Sendrecv(&cold[idx3(0, 0, ghost_lower, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_lower, 0,
                        &cold[idx3(0, 0, 0, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_lower, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (has_upper) {
            MPI_Sendrecv(&cold[idx3(0, 0, ghost_lower + nz_local - 1, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_upper, 0,
                        &cold[idx3(0, 0, ghost_lower + nz_local, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_upper, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx_global, ny_global, nz_with_ghost, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, z_start - ghost_lower, nz_global);
        
        // Exchange ghost cells for chemical potential
        if (has_lower) {
            MPI_Sendrecv(&mu[idx3(0, 0, ghost_lower, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_lower, 1,
                        &mu[idx3(0, 0, 0, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_lower, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (has_upper) {
            MPI_Sendrecv(&mu[idx3(0, 0, ghost_lower + nz_local - 1, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_upper, 1,
                        &mu[idx3(0, 0, ghost_lower + nz_local, nx_global, ny_global)], plane_size, MPI_DOUBLE, rank_upper, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx_global, ny_global, nz_with_ghost, D, dt, dx, dy, dz,
                          z_start - ghost_lower, nz_global);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx_global * ny_global * nz_global) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation or printing
    std::vector<double> global_result;
    if (rank == 0) {
        global_result.resize(nx_global * ny_global * nz_global);
    }
    
    // Extract local data without ghost cells
    std::vector<double> local_data(nx_global * ny_global * nz_local);
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny_global; ++y) {
            for (size_t x = 0; x < nx_global; ++x) {
                local_data[idx3(x, y, z, nx_global, ny_global)] = 
                    cold[idx3(x, y, z + ghost_lower, nx_global, ny_global)];
            }
        }
    }
    
    // Gather sizes and displacements
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    int local_count = nx_global * ny_global * nz_local;
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(local_data.data(), local_count, MPI_DOUBLE,
                global_result.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(global_result, "Concentration");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(global_result, nx_global, ny_global, nz_global);
            
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
