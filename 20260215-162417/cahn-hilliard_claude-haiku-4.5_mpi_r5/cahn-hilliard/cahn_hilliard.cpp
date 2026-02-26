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

// MPI helper: get rank and size
int get_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int get_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

// Compute local z-range for a given rank
void compute_z_range(const size_t nz, int rank, int nprocs, size_t& z_start, size_t& z_end) {
    size_t z_per_proc = nz / nprocs;
    size_t remainder = nz % nprocs;
    
    if (rank < (int)remainder) {
        z_start = rank * (z_per_proc + 1);
        z_end = z_start + z_per_proc + 1;
    } else {
        z_start = remainder * (z_per_proc + 1) + (rank - remainder) * z_per_proc;
        z_end = z_start + z_per_proc;
    }
}

// Compute Laplacian with clamped boundary conditions (local to rank)
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, [[maybe_unused]] const size_t nz_global,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z,
                        [[maybe_unused]] const size_t z_start, const size_t z_local_size) {
    // Handle Z boundaries: use ghost cells or clamped boundary conditions
    size_t zp = (z < z_local_size - 1) ? z + 1 : z;
    size_t zn = (z > 0) ? z - 1 : 0;
    
    // Handle X and Y boundaries with clamped conditions
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential (local computation)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_global,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t z_start, const size_t z_local_size) {
    for (size_t z = 0; z < z_local_size; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz_global, dx, dy, dz, x, y, z, z_start, z_local_size);
            }
        }
    }
}

// Cahn-Hilliard update step (local computation)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_global,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t z_start, const size_t z_local_size) {
    for (size_t z = 0; z < z_local_size; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz_global, dx, dy, dz, x, y, z, z_start, z_local_size);
            }
        }
    }
}

// Initialize concentration field (local to rank)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_global,
                             const size_t z_start, const size_t z_local_size) {
    for (size_t z = 0; z < z_local_size; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] based on global coordinates
                const size_t global_z = z_start + z;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const size_t vol = nx * ny * nz_global;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, 
                    [[maybe_unused]] const size_t nz_local) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            if (get_rank() == 0) printf("Validation failed: found NaN or Inf value\n");
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
    
    // Find global min/max across all ranks
    double global_min = minVal;
    double global_max = maxVal;
    MPI_Allreduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (get_rank() == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    }
    
    // Values should generally stay within reasonable bounds
    if (global_max > 10.0 || global_min < -10.0) {
        if (get_rank() == 0) printf("Validation failed: values out of expected range\n");
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
    printf("Note: Program uses MPI for distributed memory parallelization\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = get_rank();
    int nprocs = get_size();
    
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
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI Processes: %d\n", nprocs);
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
    
    // Compute local z-range for this rank
    size_t z_start, z_end;
    compute_z_range(nz, rank, nprocs, z_start, z_end);
    size_t z_local_size = z_end - z_start;
    
    size_t local_grid_size = nx * ny * z_local_size;
    
    // Allocate local arrays
    std::vector<double> cold(local_grid_size);
    std::vector<double> cnew(local_grid_size);
    std::vector<double> mu(local_grid_size);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z_start, z_local_size);
    
    // Barrier to ensure all ranks are ready
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, z_start, z_local_size);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz, z_start, z_local_size);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    // Synchronize before timing
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
    
    // Gather results for printing on rank 0 if requested
    if (printResults) {
        // Gather all local arrays to rank 0
        std::vector<double> global_result;
        int *recvcounts = nullptr;
        int *displs = nullptr;
        
        if (rank == 0) {
            global_result.resize(nx * ny * nz);
            recvcounts = new int[nprocs];
            displs = new int[nprocs];
            
            for (int i = 0; i < nprocs; ++i) {
                size_t z_start_i, z_end_i;
                compute_z_range(nz, i, nprocs, z_start_i, z_end_i);
                recvcounts[i] = (z_end_i - z_start_i) * nx * ny;
                displs[i] = z_start_i * nx * ny;
            }
        }
        
        MPI_Gatherv(cold.data(), local_grid_size, MPI_DOUBLE,
                   rank == 0 ? global_result.data() : nullptr,
                   rank == 0 ? recvcounts : nullptr,
                   rank == 0 ? displs : nullptr,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_result, "Concentration");
            delete[] recvcounts;
            delete[] displs;
        }
    }
    
    // Validation
    int return_code = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, z_local_size);
        
        // All ranks must agree on validation result
        int valid_int = valid ? 1 : 0;
        int global_valid_int;
        MPI_Allreduce(&valid_int, &global_valid_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        bool global_valid = (global_valid_int != 0);
        
        if (rank == 0) {
            printf("Validation: %s\n", global_valid ? "PASSED" : "FAILED");
            return_code = global_valid ? 0 : 1;
        } else {
            return_code = global_valid ? 0 : 1;
        }
    }
    
    MPI_Finalize();
    return return_code;
}

