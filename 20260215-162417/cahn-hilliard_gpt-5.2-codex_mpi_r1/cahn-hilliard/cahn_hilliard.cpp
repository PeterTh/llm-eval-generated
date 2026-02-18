#include <algorithm>
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

inline size_t localNzForRank(const size_t nz, const int size, const int rank) noexcept {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return base + (static_cast<size_t>(rank) < rem ? 1u : 0u);
}

inline size_t zStartForRank(const size_t nz, const int size, const int rank) noexcept {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const size_t local_nz, const size_t global_nz, const size_t z_start,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    (void)local_nz;
    const size_t global_z = z_start + (z - 1);
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (global_z < global_nz - 1) ? z + 1 : z;
    const size_t zn = (global_z > 0) ? z - 1 : z;
    
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
                              const size_t global_nz, const size_t z_start,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, local_nz, global_nz, z_start, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const size_t global_nz, const size_t z_start,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, local_nz, global_nz, z_start, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                             const size_t global_nz, const size_t z_start) {
    const size_t vol = nx * ny * global_nz;
    
    for (size_t z = 0; z < local_nz; ++z) {
        const size_t global_z = z_start + z;
        const size_t local_z = z + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, local_z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny,
                   const size_t local_nz, const int rank, const int size, MPI_Comm comm) {
    const size_t plane_size = nx * ny;
    const double* send_lower = field.data() + plane_size;
    const double* send_upper = field.data() + plane_size * local_nz;
    double* recv_lower = field.data();
    double* recv_upper = field.data() + plane_size * (local_nz + 1);

    MPI_Status status;
    if (rank > 0) {
        MPI_Sendrecv(send_lower, static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 0,
                     recv_lower, static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 1,
                     comm, &status);
    }
    if (rank < size - 1) {
        MPI_Sendrecv(send_upper, static_cast<int>(plane_size), MPI_DOUBLE, rank + 1, 1,
                     recv_upper, static_cast<int>(plane_size), MPI_DOUBLE, rank + 1, 0,
                     comm, &status);
    }
    if (rank == 0) {
        std::memcpy(recv_lower, send_lower, plane_size * sizeof(double));
    }
    if (rank == size - 1) {
        std::memcpy(recv_upper, send_upper, plane_size * sizeof(double));
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny,
                    const size_t local_nz, MPI_Comm comm, const int rank) {
    bool local_ok = true;
    double local_min = std::numeric_limits<double>::max();
    double local_max = -std::numeric_limits<double>::max();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_ok = false;
                    continue;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    int ok_flag = local_ok ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&ok_flag, &global_ok, 1, MPI_INT, MPI_MIN, comm);

    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    bool ok = global_ok != 0;
    if (ok && (global_max > 10.0 || global_min < -10.0)) {
        ok = false;
    }

    if (rank == 0) {
        if (!global_ok) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
            if (!ok) {
                printf("Validation failed: values out of expected range\n");
            }
        }
    }

    return ok;
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
    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
    
    if (size > static_cast<int>(nz)) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds grid size in Z (%zu)\n", size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
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
    
    const size_t gridSize = nx * ny * nz;
    const size_t local_nz = localNzForRank(nz, size, rank);
    const size_t z_start = zStartForRank(nz, size, rank);
    const size_t plane_size = nx * ny;
    
    // Allocate arrays
    std::vector<double> cold((local_nz + 2) * plane_size);
    std::vector<double> cnew((local_nz + 2) * plane_size);
    std::vector<double> mu((local_nz + 2) * plane_size);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, local_nz, nz, z_start);
    exchangeHalos(cold, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        exchangeHalos(cold, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, local_nz, nz, z_start, dx, dy, dz, 
                                 gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        exchangeHalos(mu, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, nz, z_start, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double local_seconds = end - start;
    double max_seconds = 0.0;
    MPI_Reduce(&local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_seconds * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / max_seconds / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> gathered;
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            gathered.resize(gridSize);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t r_local_nz = localNzForRank(nz, size, r);
                const size_t r_z_start = zStartForRank(nz, size, r);
                counts[r] = static_cast<int>(r_local_nz * plane_size);
                displs[r] = static_cast<int>(r_z_start * plane_size);
            }
        }
        const int local_count = static_cast<int>(local_nz * plane_size);
        MPI_Gatherv(cold.data() + plane_size, local_count, MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(gathered, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(cold, nx, ny, local_nz, MPI_COMM_WORLD, rank);
        
        if (valid) {
            if (rank == 0) {
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
