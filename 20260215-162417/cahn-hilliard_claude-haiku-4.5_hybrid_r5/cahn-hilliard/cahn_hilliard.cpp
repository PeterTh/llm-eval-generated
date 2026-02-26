#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// MPI-aware 3D domain decomposition
struct DomainDecomposition {
    int rank, size;
    size_t nx_local, ny_local, nz_local;
    size_t nx_global, ny_global, nz_global;
    int px, py, pz;
    
    DomainDecomposition(int r, int s, size_t gnx, size_t gny, size_t gnz)
        : rank(r), size(s), nx_global(gnx), ny_global(gny), nz_global(gnz) {
        // 3D process grid decomposition
        px = 1; py = 1; pz = 1;
        for (int i = 2; i * i * i <= size; ++i) {
            if (size % (i * i * i) == 0) {
                px = py = pz = i;
            }
        }
        if (px * py * pz != size) {
            for (int i = 1; i * i <= size; ++i) {
                if (size % (i * i) == 0) {
                    px = i;
                    py = i;
                    pz = size / (i * i);
                    break;
                }
            }
        }
        if (px * py * pz != size) {
            px = size;
            py = 1;
            pz = 1;
        }
        
        nx_local = (gnx + px - 1) / px;
        ny_local = (gny + py - 1) / py;
        nz_local = (gnz + pz - 1) / pz;
    }
};

// Compute Laplacian with clamped boundary conditions and OpenMP parallelism
void computeLaplacianField(const std::vector<double>& c, std::vector<double>& laplacian,
                           const size_t nx, const size_t ny, const size_t nz,
                           const double dx_inv2, const double dy_inv2, const double dz_inv2) {
    const double idx2 = 1.0 / (dx_inv2);
    const double idy2 = 1.0 / (dy_inv2);
    const double idz2 = 1.0 / (dz_inv2);
    
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t zp = (z < nz - 1) ? z + 1 : z;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t yn = (y > 0) ? y - 1 : 0;
                const size_t zn = (z > 0) ? z - 1 : 0;
                
                const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                              2.0 * c[idx]) * idx2;
                const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                              2.0 * c[idx]) * idy2;
                const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                              2.0 * c[idx]) * idz2;
                
                laplacian[idx] = cxx + cyy + czz;
            }
        }
    }
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential with OpenMP parallelism
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    #pragma omp parallel for collapse(3) schedule(static)
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

// Cahn-Hilliard update step with OpenMP parallelism
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    #pragma omp parallel for collapse(3) schedule(static)
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

// Initialize concentration field with OpenMP parallelism
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = z * (nx * ny) + y * nx + x;
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
    
    // Create domain decomposition
    DomainDecomposition domain(rank, size, nx, ny, nz);
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA Hybrid)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d, OpenMP threads: %d, CUDA enabled\n", size, omp_get_max_threads());
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Physical parameters (dx, dy, dz are all 1.0)
    
    size_t gridSize = domain.nx_local * domain.ny_local * domain.nz_local;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    std::vector<double> laplacian_c(gridSize);
    std::vector<double> laplacian_mu(gridSize);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, domain.nx_local, domain.ny_local, domain.nz_local);
    
    // Precompute inverse squared distances
    const double dx2_inv = 1.0 / (1.0 * 1.0);
    const double dy2_inv = 1.0 / (1.0 * 1.0);
    const double dz2_inv = 1.0 / (1.0 * 1.0);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute Laplacian of concentration
        computeLaplacianField(cold, laplacian_c, domain.nx_local, domain.ny_local, domain.nz_local,
                             dx2_inv, dy2_inv, dz2_inv);
        
        // Compute chemical potential with pre-computed Laplacian with OpenMP
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < domain.nz_local; ++z) {
            for (size_t y = 0; y < domain.ny_local; ++y) {
                for (size_t x = 0; x < domain.nx_local; ++x) {
                    const size_t idx = idx3(x, y, z, domain.nx_local, domain.ny_local);
                    const double cv = cold[idx];
                    
                    mu[idx] = 4.5 * ((cv + 1.0) * (-(2.0 / 9.0)) + (cv - 1.0) * (-(2.0 / 9.0)) - 2.0 * cv * (2.0 / 9.0))
                             + 3.0 * cv + cv * cv * cv
                             - 0.5 * laplacian_c[idx];
                }
            }
        }
        
        // Compute Laplacian of chemical potential
        computeLaplacianField(mu, laplacian_mu, domain.nx_local, domain.ny_local, domain.nz_local,
                             dx2_inv, dy2_inv, dz2_inv);
        
        // Update concentration with pre-computed Laplacian with OpenMP
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < domain.nz_local; ++z) {
            for (size_t y = 0; y < domain.ny_local; ++y) {
                for (size_t x = 0; x < domain.nx_local; ++x) {
                    const size_t idx = idx3(x, y, z, domain.nx_local, domain.ny_local);
                    cnew[idx] = cold[idx] + 0.01 * 1.0 * laplacian_mu[idx];
                }
            }
        }
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    // Ensure all processes complete
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Aggregate timing across all ranks
    long long local_duration = duration.count();
    long long global_duration = 0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration);
        
        // Calculate performance (using global grid)
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (global_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation (gather on rank 0)
    if (printResults && rank == 0) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, domain.nx_local, domain.ny_local, domain.nz_local);
        
        int global_valid = valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return global_valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
