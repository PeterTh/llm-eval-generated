#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian using halo planes in z (clamped at global boundaries)
inline double computeLaplacianLocal(const double* c, const size_t nx, const size_t ny,
                                    const size_t x, const size_t y, const size_t z,
                                    const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    const size_t idx = idx3(x, y, z, nx, ny);
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx]) * inv_dx2;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx]) * inv_dy2;
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] - 
                  2.0 * c[idx]) * inv_dz2;
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double inv_dx2, const double inv_dy2, const double inv_dz2,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double* cdata = c.data();
    double* mudata = mu.data();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = cdata[idx];
                
                mudata[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianLocal(cdata, nx, ny, x, y, z, inv_dx2, inv_dy2, inv_dz2);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt,
                        const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const double* mudata = mu.data();
    const double* colddata = cold.data();
    double* cnewdata = cnew.data();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnewdata[idx] = colddata[idx] + dt * D * 
                           computeLaplacianLocal(mudata, nx, ny, x, y, z, inv_dx2, inv_dy2, inv_dz2);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t start_z, const size_t local_nz) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < local_nz; ++z) {
        const size_t gz = start_z + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                    const int rank, MPI_Comm comm) {
    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_bad = 1;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }
    
    int global_bad = 0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);
    double minVal = 0.0;
    double maxVal = 0.0;
    MPI_Allreduce(&local_min, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    }
    
    if (global_bad) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }
    
    if (maxVal > 10.0 || minVal < -10.0) {
        if (rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
        return false;
    }
    
    return true;
}

void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size, MPI_Comm comm) {
    const size_t plane = nx * ny;
    double* data = field.data();
    const size_t lower_plane = idx3(0, 0, 1, nx, ny);
    const size_t upper_plane = idx3(0, 0, local_nz, nx, ny);
    const size_t lower_halo = idx3(0, 0, 0, nx, ny);
    const size_t upper_halo = idx3(0, 0, local_nz + 1, nx, ny);
    
    MPI_Status status;
    if (rank > 0) {
        MPI_Sendrecv(data + lower_plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                     data + lower_halo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1,
                     comm, &status);
    } else {
        std::memcpy(data + lower_halo, data + lower_plane, plane * sizeof(double));
    }
    
    if (rank < size - 1) {
        MPI_Sendrecv(data + upper_plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1,
                     data + upper_halo, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                     comm, &status);
    } else {
        std::memcpy(data + upper_halo, data + upper_plane, plane * sizeof(double));
    }
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    int run = 1;
    int exit_code = 0;
    
    if (rank == 0) {
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
                printUsage(argv[0]);
                run = 0;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                run = 0;
                exit_code = 1;
                break;
            }
        }
        
        if (run) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }
    
    MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!run) {
        MPI_Finalize();
        return exit_code;
    }
    
    long long dims[3];
    int settings[3];
    if (rank == 0) {
        dims[0] = static_cast<long long>(nx);
        dims[1] = static_cast<long long>(ny);
        dims[2] = static_cast<long long>(nz);
        settings[0] = iterations;
        settings[1] = validate ? 1 : 0;
        settings[2] = printResults ? 1 : 0;
    }
    MPI_Bcast(dims, 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(settings, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        nx = static_cast<size_t>(dims[0]);
        ny = static_cast<size_t>(dims[1]);
        nz = static_cast<size_t>(dims[2]);
        iterations = settings[0];
        validate = settings[1] != 0;
        printResults = settings[2] != 0;
    }
    
    if (size > static_cast<int>(nz)) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds nz (%zu)\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
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
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    
    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t start_z = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
    const size_t local_grid = (local_nz + 2) * plane;
    
    // Allocate arrays with halo planes in z
    std::vector<double> cold(local_grid);
    std::vector<double> cnew(local_grid);
    std::vector<double> mu(local_grid);
    
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, nz, start_z, local_nz);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        exchangeHalos(cold, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        exchangeHalos(mu, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, inv_dx2, inv_dy2, inv_dz2);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double local_duration = end - start;
    double max_duration = 0.0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_duration * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / max_duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> global;
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            global.resize(gridSize);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t rbase = nz / static_cast<size_t>(size);
                const size_t rrem = nz % static_cast<size_t>(size);
                const size_t rlocal_nz = rbase + (static_cast<size_t>(r) < rrem ? 1 : 0);
                const size_t rstart_z = rbase * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), rrem);
                counts[r] = static_cast<int>(rlocal_nz * plane);
                displs[r] = static_cast<int>(rstart_z * plane);
            }
        }
        MPI_Gatherv(cold.data() + idx3(0, 0, 1, nx, ny), static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(cold, nx, ny, local_nz, rank, MPI_COMM_WORLD);
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exit_code = valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return exit_code;
}
