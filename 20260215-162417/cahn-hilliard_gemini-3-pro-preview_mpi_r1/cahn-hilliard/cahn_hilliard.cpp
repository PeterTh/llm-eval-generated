#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
// Modified for local buffer with ghost cells
// z is local index, range [0, local_nz+1]
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
// z is the local index in the buffer (including ghost layers)
// global_z is the physical z coordinate
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, [[maybe_unused]] const size_t local_nz,
                        const size_t global_nz, const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z, const size_t global_z) {
    
    // X and Y boundaries are physical boundaries (clamped)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // Z boundaries:
    // If we are at global physical boundary, clamp.
    // If we are at internal boundary, use neighbor (which is in ghost layer).
    
    // z is local index. Real data is from 1 to local_nz.
    // Ghost layers are at 0 and local_nz + 1.
    
    size_t zp = z + 1;
    size_t zn = z - 1;

    // Apply physical boundary conditions for Z
    if (global_z == global_nz - 1) zp = z; // Clamp at top physical boundary
    if (global_z == 0) zn = z;             // Clamp at bottom physical boundary

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Exchange ghost cells
void exchangeGhostCells(std::vector<double>& data, int rank, int size, size_t nx, size_t ny, size_t local_nz, size_t global_nz) {
    size_t xy_plane = nx * ny;
    MPI_Request requests[4];
    int num_requests = 0;

    // If this rank has no data, it shouldn't participate in exchange
    if (local_nz == 0) return;

    // Helper to check neighbor's load
    auto get_neighbor_nz = [&](int r) -> size_t {
        if (r < 0 || r >= size) return 0;
        size_t base = global_nz / size;
        size_t rem = global_nz % size;
        return base + (r < (int)rem ? 1 : 0);
    };

    // Up neighbor (rank + 1)
    if (rank < size - 1) {
        size_t up_nz = get_neighbor_nz(rank + 1);
        if (up_nz > 0) {
            // Send to up (rank + 1)
            MPI_Isend(&data[idx3(0, 0, local_nz, nx, ny)], xy_plane, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
            // Receive from up (rank + 1)
            MPI_Irecv(&data[idx3(0, 0, local_nz + 1, nx, ny)], xy_plane, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        }
    }

    // Down neighbor (rank - 1)
    if (rank > 0) {
        size_t down_nz = get_neighbor_nz(rank - 1);
        if (down_nz > 0) {
            // Receive from down (rank - 1)
            MPI_Irecv(&data[idx3(0, 0, 0, nx, ny)], xy_plane, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
            // Send to down (rank - 1)
            MPI_Isend(&data[idx3(0, 0, 1, nx, ny)], xy_plane, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        }
    }
    
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
}
    
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz, const size_t global_nz,
                              const size_t z_start,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    
    // Iterate over real local cells (z from 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, global_nz, dx, dy, dz, x, y, z, global_z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz, const size_t global_nz,
                        const size_t z_start,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    
    // Iterate over real local cells (z from 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, global_nz, dx, dy, dz, x, y, z, global_z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentrationGlobal(std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz, 
                                  const size_t global_nz, const size_t z_start) {
    const size_t global_vol = nx * ny * global_nz;
    
    for (size_t z = 1; z <= local_nz; ++z) {
        size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz) {
    // Check for NaN or Inf
    // Only check real cells
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    // printf("Validation failed: found NaN or Inf value\n"); // Print only once
                    return false;
                }
            }
        }
    }
    
    // Global reduction for min/max would be needed for full validation like original
    // But for basic check, local is fine.
    
    double minVal = c[idx3(0,0,1,nx,ny)]; // Start with first real element
    double maxVal = c[idx3(0,0,1,nx,ny)];
    
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                double val = c[idx3(x, y, z, nx, ny)];
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }
    
    // We need to reduce minVal and maxVal across all ranks
    double globalMin, globalMax;
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    } else {
        // Return result of rank 0 check? 
        // Actually we should broadcast decision or just return true/false locally based on global values
        if (globalMax > 10.0 || globalMin < -10.0) return false;
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
    // All ranks parse arguments
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
        } 
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Domain decomposition (1D along Z)
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * local_nz;
    
    if (rank < (int)remainder) {
        local_nz++;
        z_start += rank;
    } else {
        z_start += remainder;
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
    
    // Allocation size includes 2 ghost layers (top and bottom)
    size_t bufferSize = nx * ny * (local_nz + 2);
    
    // Allocate arrays
    std::vector<double> cold(bufferSize);
    std::vector<double> cnew(bufferSize);
    std::vector<double> mu(bufferSize);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentrationGlobal(cold, nx, ny, local_nz, nz, z_start);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost cells for c
        exchangeGhostCells(cold, rank, size, nx, ny, local_nz, nz);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, nz, z_start, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Exchange ghost cells for mu
        exchangeGhostCells(mu, rank, size, nx, ny, local_nz, nz);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, nz, z_start, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        // Gather all data to rank 0 or write to shared file?
        // print_results writes to stdout/file.
        // For simplicity, gather to rank 0 then print.
        // BEWARE: This might consume a lot of memory on rank 0 for large grids.
        // But benchmarks usually run small grids for correctness check or assume scalable I/O.
        // Given existing print_results, let's gather.
        
        std::vector<double> global_c;
        if (rank == 0) global_c.resize(nx * ny * nz);
        
        // We need to gather variable sized chunks if remainder exists
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = nx * ny * local_nz;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        // Gather the real data (skip ghost layers)
        // We need to pack the real data into a contiguous buffer if it's not
        // Here, real data is from index nx*ny to nx*ny*(local_nz+1).
        // It is contiguous in memory.
        
        MPI_Gatherv(&cold[nx * ny], my_count, MPI_DOUBLE, 
                   global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);
                   
        if (rank == 0) {
            print_results(global_c, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, local_nz);
        
        // We need to ensure all ranks agree on validity
        int local_valid = valid ? 1 : 0;
        int global_valid = 1;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        if (!global_valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
