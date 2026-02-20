#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions (z halo provided)
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, [[maybe_unused]] const size_t nz,
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
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nz; ++z) {
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
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z_start, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    
    for (size_t z = 1; z <= nz; ++z) {
        const size_t global_z = z_start + (z - 1);
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

void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny, const size_t nz,
                   const int rank, const int size, MPI_Comm comm) {
    if (nz == 0) {
        return;
    }
    const size_t planeSize = nx * ny;
    double* data = field.data();
    double* lowerSend = data + idx3(0, 0, 1, nx, ny);
    double* lowerRecv = data + idx3(0, 0, 0, nx, ny);
    double* upperSend = data + idx3(0, 0, nz, nx, ny);
    double* upperRecv = data + idx3(0, 0, nz + 1, nx, ny);

    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

    if (prev == MPI_PROC_NULL) {
        std::copy_n(lowerSend, planeSize, lowerRecv);
    } else {
        MPI_Sendrecv(lowerSend, static_cast<int>(planeSize), MPI_DOUBLE, prev, 0,
                     lowerRecv, static_cast<int>(planeSize), MPI_DOUBLE, prev, 1,
                     comm, MPI_STATUS_IGNORE);
    }

    if (next == MPI_PROC_NULL) {
        std::copy_n(upperSend, planeSize, upperRecv);
    } else {
        MPI_Sendrecv(upperSend, static_cast<int>(planeSize), MPI_DOUBLE, next, 1,
                     upperRecv, static_cast<int>(planeSize), MPI_DOUBLE, next, 0,
                     comm, MPI_STATUS_IGNORE);
    }
}

bool validateResultParallel(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                            MPI_Comm comm, const int rank) {
    int localBad = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();

    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    localBad = 1;
                } else {
                    localMin = std::min(localMin, val);
                    localMax = std::max(localMax, val);
                }
            }
        }
    }

    int globalBad = 0;
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_LOR, comm);

    if (globalBad) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
        }
    }

    int globalValid = (globalMax <= 10.0 && globalMin >= -10.0) ? 1 : 0;
    MPI_Bcast(&globalValid, 1, MPI_INT, 0, comm);
    return globalValid == 1;
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
    int showHelp = 0;
    int parseError = 0;
    
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
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseError = 1;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        } else if (parseError) {
            printUsage(argv[0]);
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    MPI_Bcast(&showHelp, 1, MPI_INT, 0, comm);
    MPI_Bcast(&parseError, 1, MPI_INT, 0, comm);
    if (showHelp || parseError) {
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    uint64_t nx64 = static_cast<uint64_t>(nx);
    uint64_t ny64 = static_cast<uint64_t>(ny);
    uint64_t nz64 = static_cast<uint64_t>(nz);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;

    MPI_Bcast(&nx64, 1, MPI_UINT64_T, 0, comm);
    MPI_Bcast(&ny64, 1, MPI_UINT64_T, 0, comm);
    MPI_Bcast(&nz64, 1, MPI_UINT64_T, 0, comm);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, comm);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, comm);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, comm);

    nx = static_cast<size_t>(nx64);
    ny = static_cast<size_t>(ny64);
    nz = static_cast<size_t>(nz64);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

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
    
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
    const size_t z_start = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);

    const size_t planeSize = nx * ny;
    const size_t localGridSize = (local_nz + 2) * planeSize;
    const size_t globalGridSize = nx * ny * nz;
    
    // Allocate arrays with halo planes
    std::vector<double> cold(localGridSize, 0.0);
    std::vector<double> cnew(localGridSize, 0.0);
    std::vector<double> mu(localGridSize, 0.0);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, local_nz, z_start, nz);
    exchangeHalos(cold, nx, ny, local_nz, rank, size, comm);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, local_nz, rank, size, comm);
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        exchangeHalos(mu, nx, ny, local_nz, rank, size, comm);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(comm);
    const double end = MPI_Wtime();
    const double localDuration = end - start;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxDuration * 1000.0);
    }
    
    // Calculate performance
    if (rank == 0) {
        double cellUpdates = static_cast<double>(globalGridSize) * iterations;
        double mcups = cellUpdates / maxDuration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        int localCount = static_cast<int>(local_nz * planeSize);
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> gathered;
        if (rank == 0) {
            counts.resize(size);
        }
        MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
        if (rank == 0) {
            displs.resize(size);
            int offset = 0;
            for (int i = 0; i < size; ++i) {
                displs[i] = offset;
                offset += counts[i];
            }
            gathered.resize(globalGridSize);
        }
        MPI_Gatherv(cold.data() + idx3(0, 0, 1, nx, ny), localCount, MPI_DOUBLE,
                    gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) {
            print_results(gathered, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResultParallel(cold, nx, ny, local_nz, comm, rank);
        
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
