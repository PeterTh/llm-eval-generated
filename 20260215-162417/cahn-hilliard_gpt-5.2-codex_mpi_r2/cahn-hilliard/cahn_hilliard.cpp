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

struct Decomposition {
    size_t local_nz;
    size_t z_start;
};

Decomposition decomposeZ(const size_t nz, const int size, const int rank) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t start = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
    return {local, start};
}

void exchangeHalo(std::vector<double>& field, const size_t nx, const size_t ny,
                  const size_t local_nz, const int rank, const int size, MPI_Comm comm) {
    if (local_nz == 0) {
        return;
    }
    const size_t plane = nx * ny;
    double* data = field.data();
    MPI_Status status;

    if (rank > 0) {
        MPI_Sendrecv(data + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                     data, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &status);
    } else {
        std::copy_n(data + plane, plane, data);
    }

    if (rank + 1 < size) {
        MPI_Sendrecv(data + plane * local_nz, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1,
                     data + plane * (local_nz + 1), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &status);
    } else {
        std::copy_n(data + plane * local_nz, plane, data + plane * (local_nz + 1));
    }
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;
    
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
    for (size_t z = 1; z <= local_nz; ++z) {
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
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= local_nz; ++z) {
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
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t z_start, const size_t global_nz) {
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

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny,
                    const size_t local_nz, const int rank, MPI_Comm comm) {
    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_bad = 1;
                    continue;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }

    int global_bad = 0;
    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (global_bad) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    }

    if (global_max > 10.0 || global_min < -10.0) {
        if (rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    
    // Parse command line arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = -1;
                break;
            }
        }

        if (parseStatus == 0) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, comm);
    if (parseStatus != 0) {
        MPI_Finalize();
        return (parseStatus > 0) ? 0 : 1;
    }

    unsigned long long nx_u = nx;
    unsigned long long ny_u = ny;
    unsigned long long nz_u = nz;
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&nx_u, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&ny_u, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&nz_u, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, comm);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, comm);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, comm);
    nx = static_cast<size_t>(nx_u);
    ny = static_cast<size_t>(ny_u);
    nz = static_cast<size_t>(nz_u);
    validate = (validate_i != 0);
    printResults = (print_i != 0);
    
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
    
    size_t gridSize = nx * ny * nz;
    Decomposition decomp = decomposeZ(nz, size, rank);
    const size_t local_nz = decomp.local_nz;
    
    // Allocate arrays
    const size_t plane = nx * ny;
    std::vector<double> cold((local_nz + 2) * plane, 0.0);
    std::vector<double> cnew((local_nz + 2) * plane, 0.0);
    std::vector<double> mu((local_nz + 2) * plane, 0.0);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, local_nz, decomp.z_start, nz);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        exchangeHalo(cold, nx, ny, local_nz, rank, size, comm);
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        exchangeHalo(mu, nx, ny, local_nz, rank, size, comm);
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(comm);
    double end = MPI_Wtime();
    double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_time * 1000.0));
        
        // Calculate performance
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / max_time / 1e6;
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
                Decomposition rdecomp = decomposeZ(nz, size, r);
                counts[r] = static_cast<int>(rdecomp.local_nz * plane);
                displs[r] = static_cast<int>(rdecomp.z_start * plane);
            }
        }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) {
            print_results(gathered, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(cold, nx, ny, local_nz, rank, comm);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
