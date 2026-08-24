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
// z is the local index in the buffer (including ghost layers)
// local_nz is the number of real layers
// rank and size are MPI rank and size
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z,
                        int rank, int size) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    
    // Z boundary handling
    size_t zp = z + 1;
    size_t zn = z - 1;

    // Clamped boundary condition at global boundaries
    // If we are at the very top (rank size-1, z == local_nz)
    if (rank == size - 1 && z == local_nz) {
        zp = z; // Clamp to self
    }
    
    // If we are at the very bottom (rank 0, z == 1)
    if (rank == 0 && z == 1) {
        zn = z; // Clamp to self
    }

    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    // For z, we use indices which might be in ghost layers
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              int rank, int size) {
    // Iterate over real cells (indices 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, dx, dy, dz, x, y, z, rank, size);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        int rank, int size) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, dx, dy, dz, x, y, z, rank, size);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_global, 
                             const size_t z_start, const size_t local_nz) {
    const size_t vol = nx * ny * nz_global;
    
    // Fill real cells (1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] using global index
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchangeGhosts(std::vector<double>& data, size_t nx, size_t ny, size_t local_nz, int rank, int size) {
    size_t layer_size = nx * ny;
    MPI_Status status;
    
    // Exchange with up (rank + 1)
    // Send top real layer (local_nz) to up's bottom ghost (0)
    // Recv from up's bottom real layer (1) into top ghost (local_nz + 1)
    int up_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int down_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    // Send to down, Recv from up
    MPI_Sendrecv(
        &data[idx3(0, 0, 1, nx, ny)], layer_size, MPI_DOUBLE, down_rank, 0,
        &data[idx3(0, 0, local_nz + 1, nx, ny)], layer_size, MPI_DOUBLE, up_rank, 0,
        MPI_COMM_WORLD, &status
    );

    // Send to up, Recv from down
    MPI_Sendrecv(
        &data[idx3(0, 0, local_nz, nx, ny)], layer_size, MPI_DOUBLE, up_rank, 1,
        &data[idx3(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, down_rank, 1,
        MPI_COMM_WORLD, &status
    );
}

// Gather all data to rank 0 for validation/printing
std::vector<double> gatherData(const std::vector<double>& local_data, size_t nx, size_t ny, size_t nz_global, 
                               size_t local_nz, int rank, int size) {
    std::vector<double> global_data;
    if (rank == 0) {
        global_data.resize(nx * ny * nz_global);
    }

    size_t layer_size = nx * ny;
    size_t local_count = local_nz * layer_size;

    // Prepare receive counts and displacements for Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    if (rank == 0) {
        int current_disp = 0;
        for (int r = 0; r < size; ++r) {
            size_t r_nz_per_rank = nz_global / size;
            size_t r_remainder = nz_global % size;
            size_t r_local_nz = r_nz_per_rank + (static_cast<size_t>(r) < r_remainder ? 1 : 0);
            
            recvcounts[r] = static_cast<int>(r_local_nz * layer_size);
            displs[r] = current_disp;
            current_disp += recvcounts[r];
        }
    }

    // Send only real data (skip ghosts: start at local index 1)
    MPI_Gatherv(
        &local_data[idx3(0, 0, 1, nx, ny)], static_cast<int>(local_count), MPI_DOUBLE,
        global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    return global_data;
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
    
    // Domain decomposition
    size_t nz_per_rank = nz / size;
    size_t remainder = nz % size;
    size_t local_nz = nz_per_rank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t z_start = static_cast<size_t>(rank) * nz_per_rank + (static_cast<size_t>(rank) < remainder ? static_cast<size_t>(rank) : remainder);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
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
    
    // Allocate arrays with ghost layers
    // Size = (local_nz + 2) * nx * ny
    size_t local_gridSize = (local_nz + 2) * nx * ny;
    
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z_start, local_nz);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghosts for concentration
        exchangeGhosts(cold, nx, ny, local_nz, rank, size);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, rank, size);
        
        // Exchange ghosts for chemical potential
        exchangeGhosts(mu, nx, ny, local_nz, rank, size);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz, rank, size);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration = duration.count();
    long global_duration;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration);
        
        // Calculate performance
        size_t global_gridSize = nx * ny * nz;
        double cellUpdates = (double)global_gridSize * iterations;
        double mcups = cellUpdates / (global_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation/output if requested
    if (printResults || validate) {
        if (rank == 0) printf("Gathering results...\n");
        std::vector<double> global_c = gatherData(cold, nx, ny, nz, local_nz, rank, size);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(global_c, "Concentration");
            }
            
            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_c, nx, ny, nz);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
