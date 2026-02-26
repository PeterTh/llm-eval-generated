#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for local domain with ghost layers
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost layers with neighboring MPI ranks
void exchangeGhostLayers(std::vector<double>& data, const size_t nx, const size_t ny, const size_t local_nz,
                         const int rank, const int size) {
    const size_t slice_size = nx * ny;
    
    MPI_Request requests[4];
    int req_count = 0;
    
    // Send to rank above (rank+1), receive from rank above
    if (rank < size - 1) {
        const size_t send_idx = idx3(0, 0, local_nz, nx, ny);
        MPI_Isend(&data[send_idx], slice_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        
        const size_t recv_idx = idx3(0, 0, local_nz + 1, nx, ny);
        MPI_Irecv(&data[recv_idx], slice_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    // Send to rank below (rank-1), receive from rank below
    if (rank > 0) {
        const size_t send_idx = idx3(0, 0, 1, nx, ny);
        MPI_Isend(&data[send_idx], slice_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        
        const size_t recv_idx = idx3(0, 0, 0, nx, ny);
        MPI_Irecv(&data[recv_idx], slice_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

// Compute Laplacian with boundary conditions for MPI domain
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, 
                        [[maybe_unused]] const size_t local_nz,
                        const double dx, const double dy, const double dz, 
                        const size_t x, const size_t y, const size_t z,
                        const int rank, const int size, const size_t global_nz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // Z boundary handling accounting for MPI domain decomposition
    size_t zp = z + 1;
    size_t zn = z - 1;
    
    // Check global boundaries
    const size_t global_z_start = rank * (global_nz / size);
    const size_t global_z = global_z_start + (z - 1); // z=1 is first interior point
    
    if (global_z == 0) {
        zn = z; // Clamp at global lower boundary
    }
    if (global_z == global_nz - 1) {
        zp = z; // Clamp at global upper boundary
    }
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential for local domain
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const int rank, const int size, const size_t global_nz) {
    // Process interior cells only (z=1 to local_nz, ghost cells at z=0 and z=local_nz+1)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, dx, dy, dz, x, y, z, rank, size, global_nz);
            }
        }
    }
}

// Cahn-Hilliard update step for local domain
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const int rank, const int size, const size_t global_nz) {
    // Process interior cells only (z=1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, dx, dy, dz, x, y, z, rank, size, global_nz);
            }
        }
    }
}

// Initialize concentration field for local domain
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                             const int rank, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const size_t global_z_start = rank * (global_nz / size);
    
    // Initialize interior cells (z=1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = global_z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] using global coordinates
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                    const int rank) {
    // Check for NaN or Inf in local domain
    bool local_valid = true;
    double local_min = 1e100;
    double local_max = -1e100;
    
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double val = c[idx];
                if (std::isnan(val) || std::isinf(val)) {
                    local_valid = false;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }
    
    // Reduce validation results across all ranks
    int valid_flag = local_valid ? 1 : 0;
    int global_valid;
    MPI_Allreduce(&valid_flag, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    double global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (!global_valid) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        
        // Values should generally stay within reasonable bounds
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }
    
    return global_valid != 0;
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
    
    // Parse command line arguments (all ranks parse)
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
    
    // Decompose Z dimension across ranks
    const size_t global_nz = nz;
    const size_t local_nz = global_nz / size;
    
    if (global_nz % size != 0) {
        if (rank == 0) {
            printf("Error: Z dimension (%zu) must be divisible by number of MPI ranks (%d)\n", global_nz, size);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", size);
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, global_nz);
        printf("Local grid size per rank: %zu x %zu x %zu\n", nx, ny, local_nz);
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
    
    // Allocate local arrays with ghost layers (local_nz + 2 for top and bottom ghosts)
    const size_t local_size = nx * ny * (local_nz + 2);
    
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, local_nz, rank, global_nz);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers
        exchangeGhostLayers(cold, nx, ny, local_nz, rank, size);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, rank, size, global_nz);
        
        // Exchange ghost layers for mu
        exchangeGhostLayers(mu, nx, ny, local_nz, rank, size);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz, rank, size, global_nz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * global_nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for printing if requested
    if (printResults) {
        std::vector<double> global_result;
        if (rank == 0) {
            global_result.resize(nx * ny * global_nz);
        }
        
        // Extract interior cells (without ghost layers)
        std::vector<double> local_interior(nx * ny * local_nz);
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t local_idx = idx3(x, y, z + 1, nx, ny); // z+1 because ghost at z=0
                    const size_t interior_idx = z * (nx * ny) + y * nx + x;
                    local_interior[interior_idx] = cold[local_idx];
                }
            }
        }
        
        // Gather all local interiors to rank 0
        MPI_Gather(local_interior.data(), nx * ny * local_nz, MPI_DOUBLE,
                   global_result.data(), nx * ny * local_nz, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_result, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(cold, nx, ny, local_nz, rank);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
