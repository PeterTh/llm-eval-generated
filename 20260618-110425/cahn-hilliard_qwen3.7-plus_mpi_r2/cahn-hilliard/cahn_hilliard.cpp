#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for local arrays with ghost layers
// Local array dimensions: nx * ny * (nz_local + 2)
// z_local=0 is bottom ghost, z_local=1..nz_local are owned, z_local=nz_local+1 is top ghost
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost layers between neighboring processes
void exchangeGhostLayers(double* data, const size_t nx, const size_t ny, const size_t nz_local,
                         const int rank, const int nprocs, MPI_Comm cart_comm) {
    const size_t plane_size = nx * ny;
    const int bottom = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int top = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Send top owned plane to top neighbor, receive into top ghost from top neighbor
    // Send bottom owned plane to bottom neighbor, receive into bottom ghost from bottom neighbor
    MPI_Sendrecv(
        data + idx3(0, 0, nz_local, nx, ny), plane_size, MPI_DOUBLE, top, 0,
        data + idx3(0, 0, 0, nx, ny), plane_size, MPI_DOUBLE, bottom, 0,
        cart_comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(
        data + idx3(0, 0, 1, nx, ny), plane_size, MPI_DOUBLE, bottom, 1,
        data + idx3(0, 0, nz_local + 1, nx, ny), plane_size, MPI_DOUBLE, top, 1,
        cart_comm, MPI_STATUS_IGNORE);
    
    // Apply clamped boundary conditions at global boundaries
    if (rank == 0) {
        // Bottom ghost = bottom owned (clamped)
        memcpy(data + idx3(0, 0, 0, nx, ny), data + idx3(0, 0, 1, nx, ny), plane_size * sizeof(double));
    }
    if (rank == nprocs - 1) {
        // Top ghost = top owned (clamped)
        memcpy(data + idx3(0, 0, nz_local + 1, nx, ny), data + idx3(0, 0, nz_local, nx, ny), plane_size * sizeof(double));
    }
}

// Compute Laplacian with clamped boundary conditions for X and Y, ghost layers for Z
inline double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z_local) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z_local + 1;
    const size_t zn = z_local - 1;
    
    const size_t idx_center = idx3(x, y, z_local, nx, ny);
    
    const double cxx = (c[idx3(xp, y, z_local, nx, ny)] + c[idx3(xn, y, z_local, nx, ny)] - 
                  2.0 * c[idx_center]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z_local, nx, ny)] + c[idx3(x, yn, z_local, nx, ny)] - 
                  2.0 * c[idx_center]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx_center]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nz_local; ++z) {
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
void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field using global indices for reproducibility
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz_local,
                             const size_t z_start, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    
    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
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
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 needs to, but all do for simplicity)
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
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // 1D domain decomposition along Z-axis
    const size_t base_nz = nz / nprocs;
    const size_t remainder = nz % nprocs;
    const size_t nz_local = base_nz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    
    size_t z_start = 0;
    if (static_cast<size_t>(rank) < remainder) {
        z_start = rank * (base_nz + 1);
    } else {
        z_start = remainder * (base_nz + 1) + (rank - remainder) * base_nz;
    }
    
    if (rank == 0) {
        printf("Decomposition: Z-axis, %zu layers per process (with remainder distribution)\n", base_nz);
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
    
    // Local array dimensions with ghost layers: nx * ny * (nz_local + 2)
    const size_t local_vol = nx * ny * (nz_local + 2);
    
    // Allocate arrays
    std::vector<double> cold(local_vol, 0.0);
    std::vector<double> cnew(local_vol, 0.0);
    std::vector<double> mu(local_vol, 0.0);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold.data(), nx, ny, nz_local, z_start, nz);
    
    // Exchange ghost layers for initial data
    exchangeGhostLayers(cold.data(), nx, ny, nz_local, rank, nprocs, MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold.data(), mu.data(), nx, ny, nz_local, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Exchange ghost layers for mu
        exchangeGhostLayers(mu.data(), nx, ny, nz_local, rank, nprocs, MPI_COMM_WORLD);
        
        // Update concentration
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, nz_local, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
        
        // Exchange ghost layers for cold
        exchangeGhostLayers(cold.data(), nx, ny, nz_local, rank, nprocs, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration = duration.count();
    
    // Get maximum time across all processes
    long long max_duration;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration);
        
        // Calculate performance
        double cellUpdates = (double)nx * ny * nz * iterations;
        double mcups = cellUpdates / (max_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for output and validation
    std::vector<double> global_c;
    if (rank == 0) {
        global_c.resize(nx * ny * nz);
    }
    
    // Extract owned data (skip ghost layers)
    std::vector<double> local_data(nx * ny * nz_local);
    for (size_t z = 1; z <= nz_local; ++z) {
        memcpy(local_data.data() + (z - 1) * nx * ny, 
               cold.data() + idx3(0, 0, z, nx, ny), 
               nx * ny * sizeof(double));
    }
    
    // Gather using MPI_Gatherv
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const size_t r_nz = base_nz + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(nx * ny * r_nz);
        size_t r_z_start = 0;
        if (static_cast<size_t>(r) < remainder) {
            r_z_start = r * (base_nz + 1);
        } else {
            r_z_start = remainder * (base_nz + 1) + (r - remainder) * base_nz;
        }
        displs[r] = static_cast<int>(r_z_start * nx * ny);
    }
    
    MPI_Gatherv(local_data.data(), static_cast<int>(nx * ny * nz_local), MPI_DOUBLE,
                global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(global_c, "Concentration");
    }
    
    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(global_c, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
