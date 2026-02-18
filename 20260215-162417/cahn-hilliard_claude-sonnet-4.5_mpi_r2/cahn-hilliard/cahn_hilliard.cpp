#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (now with ghost zones)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange halo regions between neighboring MPI ranks
void exchangeHalos(std::vector<double>& data, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size) {
    const size_t layer_size = nx * ny;
    
    // Send to upper neighbor (rank+1), receive from lower neighbor (rank-1)
    if (rank < size - 1) {
        MPI_Send(&data[idx3(0, 0, local_nz, nx, ny)], layer_size, MPI_DOUBLE, 
                 rank + 1, 0, MPI_COMM_WORLD);
    }
    if (rank > 0) {
        MPI_Recv(&data[idx3(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, 
                 rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Send to lower neighbor (rank-1), receive from upper neighbor (rank+1)
    if (rank > 0) {
        MPI_Send(&data[idx3(0, 0, 1, nx, ny)], layer_size, MPI_DOUBLE, 
                 rank - 1, 1, MPI_COMM_WORLD);
    }
    if (rank < size - 1) {
        MPI_Recv(&data[idx3(0, 0, local_nz + 1, nx, ny)], layer_size, MPI_DOUBLE, 
                 rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Compute Laplacian with clamped boundary conditions (MPI version)
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                        const double dx, const double dy, const double dz, 
                        const size_t x, const size_t y, const size_t z,
                        const size_t global_z_start, const size_t global_nz,
                        const int rank, const int size) {
    const size_t global_z = global_z_start + z - 1; // -1 because z=1 is first interior point
    
    // X and Y boundaries (clamped)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // Z boundaries (considering both local and global boundaries)
    size_t zp, zn;
    
    // Handle Z-minus boundary
    if (z == 1) { // At local lower boundary
        if (rank == 0) { // Global lower boundary
            zn = z; // Clamped
        } else {
            zn = z - 1; // Ghost zone
        }
    } else {
        zn = z - 1;
    }
    
    // Handle Z-plus boundary
    if (z == local_nz) { // At local upper boundary
        if (rank == size - 1) { // Global upper boundary
            zp = z; // Clamped
        } else {
            zp = z + 1; // Ghost zone
        }
    } else {
        zp = z + 1;
    }
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential (MPI version)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t global_z_start, const size_t global_nz,
                              const int rank, const int size) {
    // Loop over interior points (z from 1 to local_nz, excluding ghost zones at 0 and local_nz+1)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, dx, dy, dz, x, y, z, 
                                                    global_z_start, global_nz, rank, size);
            }
        }
    }
}

// Cahn-Hilliard update step (MPI version)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t global_z_start, const size_t global_nz,
                        const int rank, const int size) {
    // Loop over interior points
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, dx, dy, dz, x, y, z,
                                           global_z_start, global_nz, rank, size);
            }
        }
    }
}

// Initialize concentration field (MPI version - each rank initializes its local portion)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                            const size_t global_z_start, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    
    // Initialize interior points (z from 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = global_z_start + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] based on global coordinates
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz) {
    // Check for NaN or Inf in local data (interior points only)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double val = c[idx];
                if (std::isnan(val) || std::isinf(val)) {
                    printf("Validation failed: found NaN or Inf value\n");
                    return false;
                }
            }
        }
    }
    
    // Find local min/max
    double localMin = c[idx3(0, 0, 1, nx, ny)];
    double localMax = c[idx3(0, 0, 1, nx, ny)];
    
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double val = c[idx];
                localMin = std::min(localMin, val);
                localMax = std::max(localMax, val);
            }
        }
    }
    
    // Global reduction to find overall min/max
    double globalMin, globalMax;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        
        // Values should generally stay within reasonable bounds
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
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
    // Initialize MPI
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
    
    // Domain decomposition along Z-axis
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    
    // Distribute remainder evenly among first 'remainder' ranks
    size_t global_z_start;
    if (rank < (int)remainder) {
        local_nz++;
        global_z_start = rank * local_nz;
    } else {
        global_z_start = remainder * (local_nz + 1) + (rank - remainder) * local_nz;
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("MPI processes: %d\n", size);
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
    
    // Allocate local arrays with ghost zones (local_nz + 2 layers)
    size_t local_size = nx * ny * (local_nz + 2);
    
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, local_nz, global_z_start, nz);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration field
        exchangeHalos(cold, nx, ny, local_nz, rank, size);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, global_z_start, nz, rank, size);
        
        // Exchange halos for chemical potential
        exchangeHalos(mu, nx, ny, local_nz, rank, size);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz,
                          global_z_start, nz, rank, size);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double duration = end - start;
    
    if (rank == 0) {
        printf("Computation time: %.3f s\n", duration);
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        // Gather all data to rank 0
        std::vector<double> global_data;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        if (rank == 0) {
            global_data.resize(nx * ny * nz);
        }
        
        // Calculate receive counts and displacements
        for (int r = 0; r < size; ++r) {
            size_t r_local_nz = nz / size;
            if (r < (int)(nz % size)) {
                r_local_nz++;
            }
            recvcounts[r] = nx * ny * r_local_nz;
            
            if (r == 0) {
                displs[r] = 0;
            } else {
                displs[r] = displs[r-1] + recvcounts[r-1];
            }
        }
        
        // Extract interior data (without ghost zones)
        std::vector<double> local_interior(nx * ny * local_nz);
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    local_interior[z * nx * ny + y * nx + x] = cold[idx3(x, y, z + 1, nx, ny)];
                }
            }
        }
        
        MPI_Gatherv(local_interior.data(), nx * ny * local_nz, MPI_DOUBLE,
                   global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_data, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(cold, nx, ny, local_nz);
        
        // Global AND reduction
        int local_valid = valid ? 1 : 0;
        int global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return global_valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
