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
// z_offset is the global z coordinate of local z=0
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z,
                        const size_t z_offset, const size_t nz_global) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // For z, need to consider global boundaries
    const size_t z_global = z_offset + z;
    const size_t zp = (z_global < nz_global - 1) ? z + 1 : z;
    const size_t zn = (z_global > 0) ? z - 1 : z;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t z_offset, const size_t nz_global) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz_local, dx, dy, dz, x, y, z, z_offset, nz_global);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t z_offset, const size_t nz_global) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz_local, dx, dy, dz, x, y, z, z_offset, nz_global);
            }
        }
    }
}

// Initialize concentration field (local portion)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local, 
                            const size_t z_offset, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] using global z coordinate
                const size_t z_global = z_offset + z;
                const size_t linear_id = z_global * (nx * ny) + y * nx + x;
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
    
    const size_t nz_global = nz;
    
    // Domain decomposition in Z dimension
    const size_t nz_per_rank = nz_global / size;
    const size_t nz_remainder = nz_global % size;
    
    // Distribute remainder among first ranks
    size_t nz_local = nz_per_rank + (static_cast<size_t>(rank) < nz_remainder ? 1 : 0);
    size_t z_offset = rank * nz_per_rank + std::min(static_cast<size_t>(rank), nz_remainder);
    
    // Allocate with ghost cells (1 on each side in z)
    const size_t nz_with_ghost = nz_local + 2;
    const size_t gridSize_local = nx * ny * nz_local;
    const size_t gridSize_with_ghost = nx * ny * nz_with_ghost;
    
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
    
    // Allocate arrays with ghost cells
    std::vector<double> cold_ghost(gridSize_with_ghost);
    std::vector<double> cnew_ghost(gridSize_with_ghost);
    std::vector<double> mu_ghost(gridSize_with_ghost);
    
    // Views into interior (excluding ghost cells)
    auto get_interior_ptr = [&](std::vector<double>& vec) {
        return vec.data() + nx * ny;  // Skip first ghost layer
    };
    
    // Initialize concentration field (interior only)
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> cold_interior(gridSize_local);
    initializeConcentration(cold_interior, nx, ny, nz_local, z_offset, nz_global);
    
    // Copy interior to ghost array
    std::copy(cold_interior.begin(), cold_interior.end(), get_interior_ptr(cold_ghost));
    
    // Communication buffers for ghost cell exchange
    std::vector<double> send_buf_top(nx * ny);
    std::vector<double> send_buf_bottom(nx * ny);
    std::vector<double> recv_buf_top(nx * ny);
    std::vector<double> recv_buf_bottom(nx * ny);
    
    const int rank_prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost cells for cold_ghost
        // Bottom boundary (send to rank-1, recv from rank-1)
        std::copy_n(get_interior_ptr(cold_ghost), nx * ny, send_buf_bottom.data());
        
        // Top boundary (send to rank+1, recv from rank+1)
        std::copy_n(get_interior_ptr(cold_ghost) + (nz_local - 1) * nx * ny, nx * ny, send_buf_top.data());
        
        MPI_Request reqs[4];
        MPI_Isend(send_buf_bottom.data(), nx * ny, MPI_DOUBLE, rank_prev, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buf_bottom.data(), nx * ny, MPI_DOUBLE, rank_prev, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(send_buf_top.data(), nx * ny, MPI_DOUBLE, rank_next, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recv_buf_top.data(), nx * ny, MPI_DOUBLE, rank_next, 0, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        
        // Copy received ghost cells
        if (rank_prev != MPI_PROC_NULL) {
            std::copy_n(recv_buf_bottom.data(), nx * ny, cold_ghost.data());
        } else {
            // Clamp boundary - copy interior
            std::copy_n(get_interior_ptr(cold_ghost), nx * ny, cold_ghost.data());
        }
        
        if (rank_next != MPI_PROC_NULL) {
            std::copy_n(recv_buf_top.data(), nx * ny, cold_ghost.data() + (nz_local + 1) * nx * ny);
        } else {
            // Clamp boundary - copy interior
            std::copy_n(get_interior_ptr(cold_ghost) + (nz_local - 1) * nx * ny, nx * ny,
                       cold_ghost.data() + (nz_local + 1) * nx * ny);
        }
        
        // Compute chemical potential (work on interior with ghost access)
        std::vector<double> cold_view(get_interior_ptr(cold_ghost), 
                                     get_interior_ptr(cold_ghost) + gridSize_local);
        std::vector<double> mu_view(gridSize_local);
        
        // Need to work with ghost array for stencil access
        for (size_t z = 0; z < nz_local; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx_interior = idx3(x, y, z, nx, ny);
                    const size_t idx_ghost = idx3(x, y, z + 1, nx, ny);  // +1 for ghost offset
                    const double cv = cold_ghost[idx_ghost];
                    
                    // Compute Laplacian with ghost cells
                    const size_t xp = (x < nx - 1) ? x + 1 : x;
                    const size_t yp = (y < ny - 1) ? y + 1 : y;
                    const size_t xn = (x > 0) ? x - 1 : 0;
                    const size_t yn = (y > 0) ? y - 1 : 0;
                    
                    const double cxx = (cold_ghost[idx3(xp, y, z + 1, nx, ny)] + 
                                       cold_ghost[idx3(xn, y, z + 1, nx, ny)] - 
                                       2.0 * cv) / (dx * dx);
                    const double cyy = (cold_ghost[idx3(x, yp, z + 1, nx, ny)] + 
                                       cold_ghost[idx3(x, yn, z + 1, nx, ny)] - 
                                       2.0 * cv) / (dy * dy);
                    const double czz = (cold_ghost[idx3(x, y, z + 2, nx, ny)] + 
                                       cold_ghost[idx3(x, y, z, nx, ny)] - 
                                       2.0 * cv) / (dz * dz);
                    
                    const double laplacian = cxx + cyy + czz;
                    
                    mu_view[idx_interior] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                                          + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                    mu_ghost[idx_ghost] = mu_view[idx_interior];
                }
            }
        }
        
        // Exchange ghost cells for mu_ghost
        std::copy_n(get_interior_ptr(mu_ghost), nx * ny, send_buf_bottom.data());
        std::copy_n(get_interior_ptr(mu_ghost) + (nz_local - 1) * nx * ny, nx * ny, send_buf_top.data());
        
        MPI_Isend(send_buf_bottom.data(), nx * ny, MPI_DOUBLE, rank_prev, 2, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buf_bottom.data(), nx * ny, MPI_DOUBLE, rank_prev, 3, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(send_buf_top.data(), nx * ny, MPI_DOUBLE, rank_next, 3, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recv_buf_top.data(), nx * ny, MPI_DOUBLE, rank_next, 2, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        
        if (rank_prev != MPI_PROC_NULL) {
            std::copy_n(recv_buf_bottom.data(), nx * ny, mu_ghost.data());
        } else {
            std::copy_n(get_interior_ptr(mu_ghost), nx * ny, mu_ghost.data());
        }
        
        if (rank_next != MPI_PROC_NULL) {
            std::copy_n(recv_buf_top.data(), nx * ny, mu_ghost.data() + (nz_local + 1) * nx * ny);
        } else {
            std::copy_n(get_interior_ptr(mu_ghost) + (nz_local - 1) * nx * ny, nx * ny,
                       mu_ghost.data() + (nz_local + 1) * nx * ny);
        }
        
        // Update concentration
        std::vector<double> cnew_view(gridSize_local);
        for (size_t z = 0; z < nz_local; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx_interior = idx3(x, y, z, nx, ny);
                    const size_t idx_ghost = idx3(x, y, z + 1, nx, ny);
                    
                    const size_t xp = (x < nx - 1) ? x + 1 : x;
                    const size_t yp = (y < ny - 1) ? y + 1 : y;
                    const size_t xn = (x > 0) ? x - 1 : 0;
                    const size_t yn = (y > 0) ? y - 1 : 0;
                    
                    const double mxx = (mu_ghost[idx3(xp, y, z + 1, nx, ny)] + 
                                       mu_ghost[idx3(xn, y, z + 1, nx, ny)] - 
                                       2.0 * mu_ghost[idx_ghost]) / (dx * dx);
                    const double myy = (mu_ghost[idx3(x, yp, z + 1, nx, ny)] + 
                                       mu_ghost[idx3(x, yn, z + 1, nx, ny)] - 
                                       2.0 * mu_ghost[idx_ghost]) / (dy * dy);
                    const double mzz = (mu_ghost[idx3(x, y, z + 2, nx, ny)] + 
                                       mu_ghost[idx3(x, y, z, nx, ny)] - 
                                       2.0 * mu_ghost[idx_ghost]) / (dz * dz);
                    
                    const double mu_laplacian = mxx + myy + mzz;
                    
                    cnew_view[idx_interior] = cold_ghost[idx_ghost] + dt * D * mu_laplacian;
                    cnew_ghost[idx_ghost] = cnew_view[idx_interior];
                }
            }
        }
        
        // Swap buffers
        std::swap(cold_ghost, cnew_ghost);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather timing statistics
    long local_time = duration.count();
    long max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_time);
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz_global) * iterations;
        double mcups = cellUpdates / (max_time / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results from all ranks for validation/output
    std::vector<double> final_result;
    if (rank == 0) {
        final_result.resize(nx * ny * nz_global);
    }
    
    // Gather counts and displacements
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    int local_count = static_cast<int>(gridSize_local);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    
    // Extract interior data
    std::vector<double> local_interior(gridSize_local);
    std::copy_n(get_interior_ptr(cold_ghost), gridSize_local, local_interior.data());
    
    MPI_Gatherv(local_interior.data(), gridSize_local, MPI_DOUBLE,
                final_result.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(final_result, "Concentration");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(final_result, nx, ny, nz_global);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
