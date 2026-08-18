#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// z is local and includes the two ghost planes (0 and localNz + 1).
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline double laplacian(const double* const field, const size_t nx, const size_t ny,
                        const double invDx2, const double invDy2, const double invDz2,
                        const size_t x, const size_t y, const size_t z) noexcept {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t i = idx3(x, y, z, nx, ny);
    return (field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] - 2.0 * field[i]) * invDx2
         + (field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] - 2.0 * field[i]) * invDy2
         + (field[i + nx * ny] + field[i - nx * ny] - 2.0 * field[i]) * invDz2;
}

void computeChemicalPotentialRange(const std::vector<double>& c, std::vector<double>& mu,
                                   const size_t nx, const size_t ny, const size_t firstZ, const size_t lastZ,
                                   const double invDx2, const double invDy2, const double invDz2,
                                   const double gamma, const double eAA, const double eBB, const double eAB) {
    const double* const cdata = c.data();
    double* const mudata = mu.data();
    for (size_t z = firstZ; z <= lastZ; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = cdata[idx];
                mudata[idx] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * laplacian(cdata, nx, ny, invDx2, invDy2, invDz2, x, y, z);
            }
        }
    }
}

void cahnHilliardUpdateRange(std::vector<double>& cnew, const std::vector<double>& cold, const std::vector<double>& mu,
                             const size_t nx, const size_t ny, const size_t firstZ, const size_t lastZ,
                             const double factor, const double invDx2, const double invDy2, const double invDz2) {
    const double* const coldData = cold.data();
    const double* const muData = mu.data();
    double* const cnewData = cnew.data();
    for (size_t z = firstZ; z <= lastZ; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnewData[idx] = coldData[idx] + factor * laplacian(muData, nx, ny, invDx2, invDy2, invDz2, x, y, z);
            }
        }
    }
}

struct HaloExchange {
    MPI_Request requests[4];
    int requestCount = 0;
};

HaloExchange beginHaloExchange(std::vector<double>& field, const size_t planeSize, const size_t localNz,
                               const int previous, const int next, MPI_Comm communicator) {
    // Filling physical-boundary ghosts here gives exactly the original clamped stencil.
    std::memcpy(field.data(), field.data() + planeSize, planeSize * sizeof(double));
    std::memcpy(field.data() + (localNz + 1) * planeSize, field.data() + localNz * planeSize, planeSize * sizeof(double));

    HaloExchange exchange;
    if (previous != MPI_PROC_NULL) {
        MPI_Irecv(field.data(), static_cast<int>(planeSize), MPI_DOUBLE, previous, 17, communicator, &exchange.requests[exchange.requestCount++]);
        MPI_Isend(field.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE, previous, 18, communicator, &exchange.requests[exchange.requestCount++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(field.data() + (localNz + 1) * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, next, 18, communicator, &exchange.requests[exchange.requestCount++]);
        MPI_Isend(field.data() + localNz * planeSize, static_cast<int>(planeSize), MPI_DOUBLE, next, 17, communicator, &exchange.requests[exchange.requestCount++]);
    }
    return exchange;
}

void finishHaloExchange(HaloExchange& exchange) {
    MPI_Waitall(exchange.requestCount, exchange.requests, MPI_STATUSES_IGNORE);
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t localNz,
                             const size_t globalZOffset, const size_t globalNz) {
    const size_t volume = nx * ny * globalNz;
    for (size_t z = 1; z <= localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linearId = (globalZOffset + z - 1) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linearId + 1) * 1299709) % volume) / static_cast<double>(volume));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t planeSize, const size_t localNz, const int rank, MPI_Comm communicator) {
    bool localValid = true;
    double minVal = c[planeSize];
    double maxVal = minVal;
    for (size_t i = planeSize; i < (localNz + 1) * planeSize; ++i) {
        const double val = c[i];
        if (!std::isfinite(val)) localValid = false;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    int validInt = localValid ? 1 : 0;
    int globalValid = 0;
    double globalMin = 0.0, globalMax = 0.0;
    MPI_Allreduce(&validInt, &globalValid, 1, MPI_INT, MPI_LAND, communicator);
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, communicator);
    if (rank == 0) {
        if (!globalValid) printf("Validation failed: found NaN or Inf value\n");
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
    }
    return globalValid && globalMax <= 10.0 && globalMin >= -10.0;
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nz < static_cast<size_t>(ranks)) {
        if (rank == 0) {
            printf("Invalid configuration: dimensions must be positive, iterations non-negative, and z must be at least the MPI process count.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = nx * ny;
    if (planeSize > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) printf("Grid plane is too large for MPI count arguments.\n");
        MPI_Finalize();
        return 1;
    }

    const size_t basePlanes = nz / static_cast<size_t>(ranks);
    const size_t extraPlanes = nz % static_cast<size_t>(ranks);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < extraPlanes ? 1 : 0);
    const size_t globalZOffset = static_cast<size_t>(rank) * basePlanes + std::min(static_cast<size_t>(rank), extraPlanes);
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI processes: %d\n", ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Physical parameters
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    const size_t gridSize = nx * ny * nz;
    const double invDx2 = 1.0;
    const double invDy2 = 1.0;
    const double invDz2 = 1.0;
    
    // Allocate arrays
    std::vector<double> cold((localNz + 2) * planeSize);
    std::vector<double> cnew((localNz + 2) * planeSize);
    std::vector<double> mu((localNz + 2) * planeSize);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, localNz, globalZOffset, nz);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // Interior planes do not depend on incoming halos, so communication is hidden by computation.
        HaloExchange coldExchange = beginHaloExchange(cold, planeSize, localNz, previous, next, MPI_COMM_WORLD);
        if (localNz > 2) {
            computeChemicalPotentialRange(cold, mu, nx, ny, 2, localNz - 1, invDx2, invDy2, invDz2,
                                          gamma, e_AA, e_BB, e_AB);
        }
        finishHaloExchange(coldExchange);
        computeChemicalPotentialRange(cold, mu, nx, ny, 1, 1, invDx2, invDy2, invDz2, gamma, e_AA, e_BB, e_AB);
        if (localNz > 1) {
            computeChemicalPotentialRange(cold, mu, nx, ny, localNz, localNz, invDx2, invDy2, invDz2,
                                          gamma, e_AA, e_BB, e_AB);
        }

        HaloExchange muExchange = beginHaloExchange(mu, planeSize, localNz, previous, next, MPI_COMM_WORLD);
        if (localNz > 2) {
            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, 2, localNz - 1, dt * D, invDx2, invDy2, invDz2);
        }
        finishHaloExchange(muExchange);
        cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, 1, 1, dt * D, invDx2, invDy2, invDz2);
        if (localNz > 1) {
            cahnHilliardUpdateRange(cnew, cold, mu, nx, ny, localNz, localNz, dt * D, invDx2, invDy2, invDz2);
        }
        std::swap(cold, cnew);
    }
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> globalConcentration;
        if (rank == 0) globalConcentration.resize(gridSize);
        const int localCount = static_cast<int>(localNz * planeSize);
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int process = 0; process < ranks; ++process) {
                const size_t processPlanes = basePlanes + (static_cast<size_t>(process) < extraPlanes ? 1 : 0);
                counts[process] = static_cast<int>(processPlanes * planeSize);
                displacements[process] = static_cast<int>((static_cast<size_t>(process) * basePlanes +
                                      std::min(static_cast<size_t>(process), extraPlanes)) * planeSize);
            }
        }
        MPI_Gatherv(cold.data() + planeSize, localCount, MPI_DOUBLE, globalConcentration.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(globalConcentration, "Concentration");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold, planeSize, localNz, rank, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Finalize();
    return 0;
}
