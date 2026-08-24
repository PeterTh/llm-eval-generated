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
// For MPI: handles local domain with ghost layers
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                        const double dx, const double dy, const double dz, 
                        const size_t x, const size_t y, const size_t z,
                        const size_t global_z, const size_t global_nz,
                        const bool has_lower_neighbor, const bool has_upper_neighbor) {
    // z is local index (1 to local_nz for interior points)
    // global_z is the corresponding global index
    
    // X and Y boundaries use clamped conditions
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // Z boundaries depend on whether we have neighbors or are at global boundaries
    size_t zp, zn;
    
    // Upper Z neighbor
    if (z < local_nz) {
        zp = z + 1;  // Can use ghost layer or next interior point
    } else {
        // At local upper boundary
        if (has_upper_neighbor) {
            zp = z + 1;  // Ghost layer from upper neighbor
        } else {
            // Global upper boundary: clamp
            zp = z;
        }
    }
    
    // Lower Z neighbor
    if (z > 1) {
        zn = z - 1;  // Interior point
    } else {
        // At local lower boundary (z == 1)
        if (has_lower_neighbor) {
            zn = 0;  // Ghost layer from lower neighbor
        } else {
            // Global lower boundary: clamp
            zn = 1;
        }
    }
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Exchange ghost layers via MPI
void exchangeGhostLayers(std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                         const int rank, const int size) {
    const size_t layer_size = nx * ny;
    
    // Exchange with lower neighbor (rank - 1)
    if (rank > 0) {
        // Send our lower interior layer (z=1) to rank-1
        // Receive into our lower ghost layer (z=0) from rank-1
        MPI_Sendrecv(&c[idx3(0, 0, 1, nx, ny)], layer_size, MPI_DOUBLE, rank - 1, 0,
                     &c[idx3(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Exchange with upper neighbor (rank + 1)
    if (rank < size - 1) {
        // Send our upper interior layer (z=local_nz) to rank+1
        // Receive into our upper ghost layer (z=local_nz+1) from rank+1
        MPI_Sendrecv(&c[idx3(0, 0, local_nz, nx, ny)], layer_size, MPI_DOUBLE, rank + 1, 1,
                     &c[idx3(0, 0, local_nz + 1, nx, ny)], layer_size, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t z_offset, const size_t global_nz,
                              const int rank, const int size) {
    const bool has_lower = (rank > 0);
    const bool has_upper = (rank < size - 1);
    
    // Process interior points (z from 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_offset + (z - 1);  // Convert to global Z index
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, dx, dy, dz, x, y, z, 
                                                    global_z, global_nz, has_lower, has_upper);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t z_offset, const size_t global_nz,
                        const int rank, const int size) {
    const bool has_lower = (rank > 0);
    const bool has_upper = (rank < size - 1);
    
    // Process interior points (z from 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_offset + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, dx, dy, dz, x, y, z,
                                          global_z, global_nz, has_lower, has_upper);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, 
                            const size_t local_nz, const size_t z_offset, 
                            const size_t global_nx, const size_t global_ny, const size_t global_nz) {
    const size_t global_vol = global_nx * global_ny * global_nz;
    
    // Initialize interior points (z from 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_offset + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Use global linear ID for deterministic initialization
                const size_t global_linear_id = global_z * (global_nx * global_ny) + y * global_nx + x;
                const double pseudo = ((((global_linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather local data to rank 0 for validation/output
void gatherToRoot(const std::vector<double>& local_c, std::vector<double>& global_c,
                  const size_t nx, const size_t ny, const size_t local_nz,
                  const int rank, const int size, const size_t global_nz) {
    // Extract interior data (without ghost layers)
    std::vector<double> interior_data(nx * ny * local_nz);
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t interior_idx = (z - 1) * (nx * ny) + y * nx + x;
                interior_data[interior_idx] = local_c[local_idx];
            }
        }
    }
    
    // Compute send counts and displacements
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; ++r) {
        size_t rank_nz = global_nz / size;
        if (r < (int)(global_nz % size)) rank_nz++;
        recvcounts[r] = nx * ny * rank_nz;
        displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
    }
    
    // Gather to rank 0
    if (rank == 0) {
        global_c.resize(nx * ny * global_nz);
    }
    
    MPI_Gatherv(interior_data.data(), interior_data.size(), MPI_DOUBLE,
                global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
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
    
    // Domain decomposition in Z direction
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_offset = rank * local_nz + std::min((size_t)rank, remainder);
    
    if (rank < (int)remainder) {
        local_nz++;
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", size);
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
    
    // Local grid size includes ghost layers: (local_nz + 2) in Z direction
    // Ghost layers at z=0 and z=local_nz+1
    // Interior points at z=1 to z=local_nz
    size_t local_grid_size = nx * ny * (local_nz + 2);
    
    // Allocate arrays
    std::vector<double> cold(local_grid_size, 0.0);
    std::vector<double> cnew(local_grid_size, 0.0);
    std::vector<double> mu(local_grid_size, 0.0);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, local_nz, z_offset, nx, ny, nz);
    
    // Exchange initial ghost layers
    exchangeGhostLayers(cold, nx, ny, local_nz, rank, size);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers for concentration
        exchangeGhostLayers(cold, nx, ny, local_nz, rank, size);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, z_offset, nz, rank, size);
        
        // Exchange ghost layers for chemical potential
        exchangeGhostLayers(mu, nx, ny, local_nz, rank, size);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz,
                          z_offset, nz, rank, size);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate performance
        size_t global_grid_size = nx * ny * nz;
        double cellUpdates = (double)global_grid_size * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation and output
    std::vector<double> global_c;
    gatherToRoot(cold, global_c, nx, ny, local_nz, rank, size, nz);
    
    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
        print_results(global_c, "Concentration");
    }
    
    // Validation (rank 0 only)
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(global_c, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            result = 0;
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    MPI_Finalize();
    return result;
}
