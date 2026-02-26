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

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nz_with_ghost = local_nz + 2;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz_with_ghost, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nz_with_ghost = local_nz + 2;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz_with_ghost, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t local_nz, const size_t z_start) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;
    
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z_start + z) * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny,
                   const size_t local_nz, const int rank, const int size, MPI_Comm comm) {
    if (local_nz == 0) {
        return;
    }

    const size_t plane = nx * ny;
    double* data = field.data();
    double* send_lower = data + plane;
    double* send_upper = data + local_nz * plane;
    double* recv_lower = data;
    double* recv_upper = data + (local_nz + 1) * plane;

    if (rank > 0) {
        MPI_Sendrecv(send_lower, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                     recv_lower, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                     comm, MPI_STATUS_IGNORE);
    } else {
        std::memcpy(recv_lower, send_lower, plane * sizeof(double));
    }

    if (rank + 1 < size) {
        MPI_Sendrecv(send_upper, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                     recv_upper, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                     comm, MPI_STATUS_IGNORE);
    } else {
        std::memcpy(recv_upper, send_upper, plane * sizeof(double));
    }
}

bool validateResultDistributed(const std::vector<double>& c, const size_t nx, const size_t ny,
                               const size_t local_nz, const int rank, MPI_Comm comm) {
    bool local_invalid = false;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_invalid = true;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    int local_invalid_i = local_invalid ? 1 : 0;
    int global_invalid_i = 0;
    MPI_Allreduce(&local_invalid_i, &global_invalid_i, 1, MPI_INT, MPI_LOR, comm);

    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    const bool valid = (global_invalid_i == 0) && (global_max <= 10.0) && (global_min >= -10.0);

    if (rank == 0) {
        if (global_invalid_i != 0) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);

        if (!valid) {
            printf("Validation failed: values out of expected range\n");
        }
    }

    return valid;
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

        if (run != 0) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }

    MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (run == 0) {
        MPI_Finalize();
        return exit_code;
    }

    unsigned long long nx_u = static_cast<unsigned long long>(nx);
    unsigned long long ny_u = static_cast<unsigned long long>(ny);
    unsigned long long nz_u = static_cast<unsigned long long>(nz);
    MPI_Bcast(&nx_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    nx = static_cast<size_t>(nx_u);
    ny = static_cast<size_t>(ny_u);
    nz = static_cast<size_t>(nz_u);
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    if (static_cast<size_t>(size) > nz) {
        if (rank == 0) {
            printf("Error: MPI ranks (%d) exceed grid size in Z (%zu)\n", size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t rank_size = static_cast<size_t>(rank);
    const size_t size_size = static_cast<size_t>(size);
    const size_t base = nz / size_size;
    const size_t rem = nz % size_size;
    const size_t local_nz = base + (rank_size < rem ? 1 : 0);
    const size_t z_start = rank_size * base + std::min(rank_size, rem);

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
    
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t local_size = (local_nz + 2) * plane;
    
    // Allocate arrays with ghost layers in Z
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, nz, local_nz, z_start);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        exchangeHalos(mu, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> gathered;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const size_t r_size = static_cast<size_t>(r);
                const size_t r_local = base + (r_size < rem ? 1 : 0);
                counts[r] = static_cast<int>(r_local * plane);
                displs[r] = static_cast<int>(offset);
                offset += r_local * plane;
            }
            gathered.resize(gridSize);
        }

        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(gathered, "Concentration");
        }
    }
    
    int status = 0;
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResultDistributed(cold, nx, ny, local_nz, rank, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        if (!valid) {
            status = 1;
        }
    }
    
    MPI_Finalize();
    return status;
}
