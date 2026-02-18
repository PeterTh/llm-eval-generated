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
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
// z is the local index
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    
    // Z boundaries handled by ghost layers
    // Real data is from z=1 to z=nz.
    // Ghost layers at z=0 and z=nz+1 are populated by exchange_halos.
    
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
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
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    // Iterate over real domain (1..nz)
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz, 
                           const size_t global_nz, const size_t z_start) {
    const size_t vol = nx * ny * global_nz;
    
    // Fill real domain 1..nz
    for (size_t z = 1; z <= nz; ++z) {
        size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchange_halos(std::vector<double>& data, size_t nx, size_t ny, size_t nz, int rank, int size) {
    size_t plane_size = nx * ny;
    MPI_Status status;
    
    // Tags
    int tag_up = 1;
    int tag_down = 2;
    
    // Exchange with neighbor below (rank-1)
    // Send 1st real layer (index 1), Recv into 0th ghost layer (index 0)
    // Only if rank > 0.
    // Rank 0 has no lower neighbor.
    
    // Exchange with neighbor above (rank+1)
    // Send last real layer (index nz), Recv into upper ghost layer (index nz+1)
    // Only if rank < size-1.
    // Rank size-1 has no upper neighbor.

    // To avoid deadlock, use Sendrecv or careful ordering.
    // Let's use Sendrecv for simplicity and safety.
    
    // Upward exchange: Rank i sends to i+1, Rank i+1 receives from i
    // Rank i also receives from i+1 (downward flow), Rank i+1 sends to i
    
    // Let's do it in two steps using Sendrecv.
    
    // Step 1: Exchange with Upper Neighbor (Rank i <-> Rank i+1)
    // Rank i sends up (tag_up), receives from up (tag_down - wait, tags need to match sender's intent)
    // Rank i sends its top layer (nz) to i+1. Rank i+1 receives it into its bottom ghost (0).
    // Rank i+1 sends its bottom layer (1) to i. Rank i receives it into its top ghost (nz+1).
    
    int up_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int down_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    
    // Send to Up, Recv from Up
    // We send our top real layer (nz) to up_neighbor.
    // We receive from up_neighbor into our top ghost layer (nz+1).
    // Wait, if I send to up, up receives from down.
    
    // Let's use a simpler pattern: Odd/Even or just Sendrecv.
    
    // Send Down (to rank-1), Recv Up (from rank+1)
    // Send Up (to rank+1), Recv Down (from rank-1)
    
    // Send buffer for Up: &data[idx3(0, 0, nz, nx, ny)]
    // Recv buffer from Up: &data[idx3(0, 0, nz+1, nx, ny)]
    // Send buffer for Down: &data[idx3(0, 0, 1, nx, ny)]
    // Recv buffer from Down: &data[idx3(0, 0, 0, nx, ny)]
    
    // 1. Send to Up, Recv from Down
    MPI_Sendrecv(
        &data[idx3(0, 0, nz, nx, ny)], plane_size, MPI_DOUBLE, up_neighbor, tag_up,
        &data[idx3(0, 0, 0, nx, ny)], plane_size, MPI_DOUBLE, down_neighbor, tag_up,
        MPI_COMM_WORLD, &status
    );
    
    // 2. Send to Down, Recv from Up
    MPI_Sendrecv(
        &data[idx3(0, 0, 1, nx, ny)], plane_size, MPI_DOUBLE, down_neighbor, tag_down,
        &data[idx3(0, 0, nz + 1, nx, ny)], plane_size, MPI_DOUBLE, up_neighbor, tag_down,
        MPI_COMM_WORLD, &status
    );
    
    // Apply boundary conditions for global boundaries
    if (rank == 0) {
        // Bottom boundary: copy layer 1 to layer 0
        // This is necessary because the Recv from down_neighbor (PROC_NULL) didn't do anything.
        std::copy(data.begin() + idx3(0, 0, 1, nx, ny),
                  data.begin() + idx3(0, 0, 2, nx, ny),
                  data.begin() + idx3(0, 0, 0, nx, ny));
    }
    if (rank == size - 1) {
        // Top boundary: copy layer nz to layer nz+1
         // This is necessary because the Recv from up_neighbor (PROC_NULL) didn't do anything.
        std::copy(data.begin() + idx3(0, 0, nz, nx, ny),
                  data.begin() + idx3(0, 0, nz + 1, nx, ny),
                  data.begin() + idx3(0, 0, nz + 1, nx, ny));
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
    int rank = 0;
    int size = 1;
    MPI_Init(&argc, &argv);
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
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

    // Determine local domain size
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if ((size_t)rank < remainder) {
        local_nz++;
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
    
    // Size including ghost layers (one top, one bottom)
    size_t local_grid_size = nx * ny * (local_nz + 2);
    
    // Allocate arrays
    std::vector<double> cold(local_grid_size);
    std::vector<double> cnew(local_grid_size);
    std::vector<double> mu(local_grid_size);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz, nz, z_start);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    
    // Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration field before computing chemical potential
        exchange_halos(cold, nx, ny, local_nz, rank, size);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Exchange halos for chemical potential before updating concentration
        exchange_halos(mu, nx, ny, local_nz, rank, size);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    // Barrier after timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        size_t totalGridSize = nx * ny * nz;
        double cellUpdates = (double)totalGridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation and printing
    std::vector<double> global_c;
    if (validate || printResults) {
        if (rank == 0) {
            global_c.resize(nx * ny * nz);
        }

        // Prepare counts and displacements for Gatherv
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        // Each rank sends its real data (excluding ghost layers)
        // Size of real data block in elements
        int sendcount = (int)(nx * ny * local_nz);
        
        // Gather counts on root
        MPI_Gather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Calculate displacements
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        // Gatherv
        // Send buffer is start of real data (skipping bottom ghost layer)
        // Note: global_c is only significant on root, but argument must be valid (or NULL on non-root)
        MPI_Gatherv(&cold[idx3(0, 0, 1, nx, ny)], sendcount, MPI_DOUBLE,
                    global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(global_c, "Concentration");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(global_c, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    MPI_Finalize();
    return 0;
}

