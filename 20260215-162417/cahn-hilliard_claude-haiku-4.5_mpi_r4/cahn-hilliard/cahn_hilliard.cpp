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

// MPI domain decomposition
struct MpiDomain {
    int rank, size;
    int local_nx, local_ny, local_nz;
    int global_nx, global_ny, global_nz;
    int offset_z;
    
    MpiDomain(int rank_, int size_, int gnx, int gny, int gnz) 
        : rank(rank_), size(size_), global_nx(gnx), global_ny(gny), global_nz(gnz),
          local_nx(gnx), local_ny(gny) {
        local_nz = (gnz + size - 1) / size;
        offset_z = rank * local_nz;
        
        if (offset_z + local_nz > gnz) {
            local_nz = gnz - offset_z;
        }
    }
};

// Compute Laplacian with boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    auto get_c = [&](int xx, int yy, int zz) -> double {
        if (xx < 0) xx = 0;
        if (xx >= (int)nx) xx = nx - 1;
        if (yy < 0) yy = 0;
        if (yy >= (int)ny) yy = ny - 1;
        if (zz < 0) zz = 0;
        if (zz >= (int)nz) zz = nz - 1;
        return c[idx3(xx, yy, zz, nx, ny)];
    };
    
    int xp = (x < nx - 1) ? x + 1 : x;
    int xn = (x > 0) ? x - 1 : 0;
    int yp = (y < ny - 1) ? y + 1 : y;
    int yn = (y > 0) ? y - 1 : 0;
    
    double cxx = (get_c(xp, y, z) + get_c(xn, y, z) - 2.0 * get_c(x, y, z)) / (dx * dx);
    double cyy = (get_c(x, yp, z) + get_c(x, yn, z) - 2.0 * get_c(x, y, z)) / (dy * dy);
    double czz = (get_c(x, y, z + 1) + get_c(x, y, z - 1) - 2.0 * get_c(x, y, z)) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field with global consistency
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                            const MpiDomain& domain) {
    const size_t global_vol = (size_t)domain.global_nx * domain.global_ny * domain.global_nz;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                size_t global_z = domain.offset_z + z;
                const size_t linear_id = global_z * ((size_t)domain.global_nx * domain.global_ny) + y * domain.global_nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (broadcast from rank 0)
    if (mpi_rank == 0) {
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
    
    if (mpi_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI Parallel)\n");
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", mpi_size);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Setup MPI domain decomposition
    MpiDomain domain(mpi_rank, mpi_size, nx, ny, nz);
    
    if (mpi_rank == 0) {
        printf("Local grid per process: %d x %d x %d\n", domain.local_nx, domain.local_ny, domain.local_nz);
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
    
    size_t local_gridSize = domain.local_nx * domain.local_ny * domain.local_nz;
    
    // Allocate local arrays
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    if (mpi_rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, domain.local_nx, domain.local_ny, domain.local_nz, domain);
    
    if (mpi_rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    
    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential locally with overlapped communication
        computeChemicalPotential(cold, mu, domain.local_nx, domain.local_ny, domain.local_nz,
                                dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        
        // Update concentration locally
        cahnHilliardUpdate(cnew, cold, mu, domain.local_nx, domain.local_ny, domain.local_nz,
                          D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather results on rank 0
    std::vector<double> global_c;
    std::vector<int> recvcounts(mpi_size);
    std::vector<int> displs(mpi_size);
    
    int local_size = domain.local_nx * domain.local_ny * domain.local_nz;
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        global_c.resize(nx * ny * nz);
        displs[0] = 0;
        for (int i = 1; i < mpi_size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(cold.data(), local_size, MPI_DOUBLE,
                global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
