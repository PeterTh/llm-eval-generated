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
inline constexpr size_t idx3_local(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 3D index calculation for global domain
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with ghost layer data and clamped boundaries
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z,
                        const int rank, const int size, [[maybe_unused]] const size_t z_offset, [[maybe_unused]] const size_t nz_global,
                        const std::vector<double>* ghost_prev, const std::vector<double>* ghost_next) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // Get z neighbors, using ghost layers when available
    double c_zp, c_zn;
    
    if (z < nz_local - 1) {
        c_zp = c[idx3_local(x, y, z + 1, nx, ny)];
    } else {
        // At z=nz_local-1
        if (rank == size - 1) {
            // Global boundary: clamp
            c_zp = c[idx3_local(x, y, z, nx, ny)];
        } else {
            // Interior boundary: use ghost from next rank
            c_zp = (*ghost_next)[idx3_local(x, y, 0, nx, ny)];
        }
    }
    
    if (z > 0) {
        c_zn = c[idx3_local(x, y, z - 1, nx, ny)];
    } else {
        // At z=0
        if (rank == 0) {
            // Global boundary: clamp
            c_zn = c[idx3_local(x, y, z, nx, ny)];
        } else {
            // Interior boundary: use ghost from prev rank
            c_zn = (*ghost_prev)[idx3_local(x, y, 0, nx, ny)];
        }
    }
    
    const double c_center = c[idx3_local(x, y, z, nx, ny)];
    const double cxx = (c[idx3_local(xp, y, z, nx, ny)] + c[idx3_local(xn, y, z, nx, ny)] - 
                  2.0 * c_center) / (dx * dx);
    const double cyy = (c[idx3_local(x, yp, z, nx, ny)] + c[idx3_local(x, yn, z, nx, ny)] - 
                  2.0 * c_center) / (dy * dy);
    const double czz = (c_zp + c_zn - 2.0 * c_center) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Exchange ghost layers for z-dimension with neighboring ranks
void exchangeGhostLayers(std::vector<double>& c, std::vector<double>& ghost_prev, std::vector<double>& ghost_next,
                         const size_t nx, const size_t ny, const size_t nz_local,
                         const int rank, const int size) {
    if (size == 1) return;  // No communication needed for single rank
    
    const size_t layer_size = nx * ny;
    const int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Exchange boundaries for Laplacian stencil
    if (prev_rank != MPI_PROC_NULL && next_rank != MPI_PROC_NULL) {
        MPI_Request requests[4];
        MPI_Isend(&c[idx3_local(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(ghost_prev.data(), layer_size, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&c[idx3_local(0, 0, nz_local - 1, nx, ny)], layer_size, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(ghost_next.data(), layer_size, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    } else if (prev_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(&c[idx3_local(0, 0, 0, nx, ny)], layer_size, MPI_DOUBLE, prev_rank, 0,
                     ghost_prev.data(), layer_size, MPI_DOUBLE, prev_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (next_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(&c[idx3_local(0, 0, nz_local - 1, nx, ny)], layer_size, MPI_DOUBLE, next_rank, 1,
                     ghost_next.data(), layer_size, MPI_DOUBLE, next_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Compute chemical potential (local computation after halo exchange)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const int rank, const int size, const size_t z_offset, const size_t nz_global,
                              const std::vector<double>& ghost_prev, const std::vector<double>& ghost_next) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz_local, dx, dy, dz, x, y, z,
                                                    rank, size, z_offset, nz_global, &ghost_prev, &ghost_next);
            }
        }
    }
}

// Cahn-Hilliard update step (local computation after halo exchange)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const int rank, const int size, const size_t z_offset, const size_t nz_global,
                        const std::vector<double>& ghost_prev, const std::vector<double>& ghost_next) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz_local, dx, dy, dz, x, y, z,
                                          rank, size, z_offset, nz_global, &ghost_prev, &ghost_next);
            }
        }
    }
}

// Initialize concentration field (local partition)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local,
                             const size_t z_offset, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z, nx, ny);
                // Use global z coordinate for consistent pseudo-random initialization
                const size_t global_z = z + z_offset;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, 
                   [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz_local,
                   const int rank) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Rank %d: Validation failed: found NaN or Inf value\n", rank);
            return false;
        }
    }
    
    // Find local min/max
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    // Gather global min/max
    double global_min, global_max;
    MPI_Allreduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    }
    
    // Check bounds
    if (global_max > 10.0 || global_min < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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
    
    // Parse command line arguments (only rank 0)
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
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition: distribute z-dimension among ranks
    size_t nz_global = nz;
    size_t nz_local = nz / size;
    size_t z_offset = rank * nz_local;
    
    // Handle remainder: give extra slices to last ranks
    if (rank >= (int)(size - (nz % size))) {
        nz_local += 1;
        z_offset = nz - (size - rank) * (nz_local);
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz_global);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
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
    
    size_t gridSize = nx * ny * nz_local;
    
    // Allocate local arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    std::vector<double> ghost_prev(nx * ny);  // Ghost layer from previous rank
    std::vector<double> ghost_next(nx * ny);  // Ghost layer from next rank
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz_local, z_offset, nz_global);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers for concentration field
        exchangeGhostLayers(cold, ghost_prev, ghost_next, nx, ny, nz_local, rank, size);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz_local, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, rank, size, z_offset, nz_global,
                                ghost_prev, ghost_next);
        
        // Exchange ghost layers for chemical potential
        exchangeGhostLayers(mu, ghost_prev, ghost_next, nx, ny, nz_local, rank, size);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local, D, dt, dx, dy, dz,
                          rank, size, z_offset, nz_global, ghost_prev, ghost_next);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration = duration.count();
    long global_duration;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration);
        
        // Calculate performance (total grid updates)
        double cellUpdates = (double)nz_global * nx * ny * iterations;
        double mcups = cellUpdates / (global_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation (gather to rank 0)
    if (printResults && rank == 0) {
        // Allocate global array on rank 0
        std::vector<double> c_global(nx * ny * nz_global);
        
        // Receive data from all ranks
        for (int r = 0; r < size; ++r) {
            size_t local_z_count = nz_global / size;
            size_t local_z_start = r * local_z_count;
            if (r >= (int)(size - (nz_global % size))) {
                local_z_count += 1;
                local_z_start = nz_global - (size - r) * local_z_count;
            }
            
            if (r == 0) {
                std::copy(cold.begin(), cold.end(), 
                         c_global.begin() + local_z_start * nx * ny);
            } else {
                std::vector<double> recv_data(nx * ny * local_z_count);
                MPI_Recv(recv_data.data(), nx * ny * local_z_count, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::copy(recv_data.begin(), recv_data.end(),
                         c_global.begin() + local_z_start * nx * ny);
            }
        }
        print_results(c_global, "Concentration");
    } else if (printResults && rank != 0) {
        // Send data to rank 0
        MPI_Send(cold.data(), cold.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz_local, rank);
        
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
