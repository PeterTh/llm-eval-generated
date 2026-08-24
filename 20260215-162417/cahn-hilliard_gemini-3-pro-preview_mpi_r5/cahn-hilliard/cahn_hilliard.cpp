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
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, [[maybe_unused]] const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1; // Always valid due to ghost cells
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1; // Always valid due to ghost cells
    
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
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    // Process only the local real domain (indices 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    // Process only the local real domain (indices 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz, const size_t local_nz, const size_t z_start) {
    const size_t vol = nx * ny * nz;
    
    // Initialize only the local real domain (indices 1 to local_nz)
    for (size_t z = 1; z <= local_nz; ++z) {
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

void exchangeGhosts(std::vector<double>& field, int nx, int ny, int local_nz, int rank, int size) {
    size_t plane_size = nx * ny;
    int top_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int bottom_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    // Exchange Upward (Send Top Real -> Top Neighbor; Recv from Bottom Neighbor -> Bottom Ghost)
    MPI_Sendrecv(
        &field[idx3(0, 0, local_nz, nx, ny)], plane_size, MPI_DOUBLE, top_neighbor, 0,
        &field[idx3(0, 0, 0, nx, ny)],        plane_size, MPI_DOUBLE, bottom_neighbor, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Exchange Downward (Send Bottom Real -> Bottom Neighbor; Recv from Top Neighbor -> Top Ghost)
    MPI_Sendrecv(
        &field[idx3(0, 0, 1, nx, ny)],          plane_size, MPI_DOUBLE, bottom_neighbor, 1,
        &field[idx3(0, 0, local_nz + 1, nx, ny)], plane_size, MPI_DOUBLE, top_neighbor, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Boundary conditions (Clamped)
    if (rank == 0) {
        // z(-1) = z(0) => Ghost 0 = Real 1
        std::copy(
            field.begin() + idx3(0, 0, 1, nx, ny),
            field.begin() + idx3(0, 0, 2, nx, ny),
            field.begin() + idx3(0, 0, 0, nx, ny)
        );
    }
    if (rank == size - 1) {
        // z(nz) = z(nz-1) => Ghost local_nz+1 = Real local_nz
        std::copy(
            field.begin() + idx3(0, 0, local_nz, nx, ny),
            field.begin() + idx3(0, 0, local_nz + 1, nx, ny),
            field.begin() + idx3(0, 0, local_nz + 1, nx, ny)
        );
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } 
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nz < (size_t)size) {
        if (rank == 0) {
            printf("Error: grid size Z (%zu) must be at least equal to number of MPI ranks (%d)\n", nz, size);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Domain decomposition
    size_t base_local_nz = nz / size;
    size_t remainder = nz % size;
    size_t local_nz = base_local_nz + ((size_t)rank < remainder ? 1 : 0);
    size_t z_start = 0;
    for (int r = 0; r < rank; ++r) {
        z_start += base_local_nz + ((size_t)r < remainder ? 1 : 0);
    }
    
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
    
    size_t plane_size = nx * ny;
    // Allocate arrays with ghost layers (top and bottom)
    size_t local_grid_size = (local_nz + 2) * plane_size;
    
    std::vector<double> cold(local_grid_size);
    std::vector<double> cnew(local_grid_size);
    std::vector<double> mu(local_grid_size);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, local_nz, z_start);
    
    // Initial ghost exchange
    exchangeGhosts(cold, nx, ny, local_nz, rank, size);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        exchangeGhosts(mu, nx, ny, local_nz, rank, size);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        
        exchangeGhosts(cnew, nx, ny, local_nz, rank, size);

        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation/output
    std::vector<double> c_global;
    if (validate || printResults) {
        if (rank == 0) c_global.resize(nx * ny * nz);
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = local_nz * plane_size;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[r] = displs[r-1] + recvcounts[r-1];
            }
        }
        
        // Gather only real data (skip bottom ghost)
        MPI_Gatherv(&cold[plane_size], my_count, MPI_DOUBLE, 
                   c_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(c_global, "Concentration");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(c_global, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}
