#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (local grid)
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

// Compute chemical potential on local domain
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

// Cahn-Hilliard update step on local domain
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

// Initialize concentration field on local domain
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nzLocal, const size_t zStart, const size_t nzGlobal) {
    const size_t vol = nx * ny * nzGlobal;
    for (size_t z = 0; z < nzLocal; ++z) {
        const size_t zGlobal = zStart + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] using global coordinates
                const size_t linear_id = zGlobal * (nx * ny) + y * nx + x;
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

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
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

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along Z axis
    const size_t nzPerRank = nz / numRanks;
    const size_t remainder = nz % numRanks;
    const size_t zStart = rank * nzPerRank + (rank < static_cast<int>(remainder) ? rank : remainder);
    const size_t zEnd = zStart + (rank < static_cast<int>(remainder) ? nzPerRank + 1 : nzPerRank);

    // Local Z size includes 1 ghost cell on each side
    const size_t nzLocal = (zEnd - zStart) + 2;
    const size_t localSize = nx * ny * nzLocal;

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

    // Print info on rank 0
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local arrays
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    // Initialize interior concentration field
    initializeConcentration(cold, nx, ny, nzLocal, zStart, nz);

    // Initialize ghost cells with clamped boundary values
    // Bottom ghost cell (local z=0): use first interior value
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            cold[idx3(x, y, 0, nx, ny)] = cold[idx3(x, y, 1, nx, ny)];
            cnew[idx3(x, y, 0, nx, ny)] = cold[idx3(x, y, 1, nx, ny)];
        }
    }
    // Top ghost cell (local z=nzLocal-1): use last interior value
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            cold[idx3(x, y, nzLocal - 1, nx, ny)] = cold[idx3(x, y, nzLocal - 2, nx, ny)];
            cnew[idx3(x, y, nzLocal - 1, nx, ny)] = cold[idx3(x, y, nzLocal - 2, nx, ny)];
        }
    }

    // Ghost cell communication buffers (one Z-slice each)
    const size_t sliceSize = nx * ny;
    std::vector<double> sendBufTop(sliceSize);
    std::vector<double> sendBufBot(sliceSize);
    std::vector<double> recvBufTop(sliceSize);
    std::vector<double> recvBufBot(sliceSize);

    // Determine neighbor ranks
    const int rankBelow = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rankAbove = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    // Communication request handles for non-blocking operations
    MPI_Request reqs[4];

    // Exchange ghost cells: send interior boundaries, receive ghost cells
    // Uses non-blocking communication for performance
    auto exchangeGhostCells = [&](std::vector<double>& field) {
        // Copy interior boundary data to send buffers
        if (rankAbove != MPI_PROC_NULL) {
            const size_t srcIdx = idx3(0, 0, nzLocal - 2, nx, ny);
            std::memcpy(sendBufTop.data(), field.data() + srcIdx, sliceSize * sizeof(double));
        }
        if (rankBelow != MPI_PROC_NULL) {
            const size_t srcIdx = idx3(0, 0, 1, nx, ny);
            std::memcpy(sendBufBot.data(), field.data() + srcIdx, sliceSize * sizeof(double));
        }

        // Post non-blocking sends and receives simultaneously
        // Tags: 0 = top slice sent to rankAbove, 1 = bottom slice sent to rankBelow
        MPI_Isend(sendBufTop.data(), sliceSize, MPI_DOUBLE, rankAbove, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sendBufBot.data(), sliceSize, MPI_DOUBLE, rankBelow, 1, MPI_COMM_WORLD, &reqs[1]);
        // Receive matching tags: rankAbove sends bottom (tag 1), rankBelow sends top (tag 0)
        MPI_Irecv(recvBufTop.data(), sliceSize, MPI_DOUBLE, rankAbove, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recvBufBot.data(), sliceSize, MPI_DOUBLE, rankBelow, 0, MPI_COMM_WORLD, &reqs[3]);

        // Wait for all communications to complete
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Copy received data into ghost cells
        if (rankAbove != MPI_PROC_NULL) {
            const size_t dstIdx = idx3(0, 0, nzLocal - 1, nx, ny);
            std::memcpy(field.data() + dstIdx, recvBufTop.data(), sliceSize * sizeof(double));
        }
        if (rankBelow != MPI_PROC_NULL) {
            const size_t dstIdx = idx3(0, 0, 0, nx, ny);
            std::memcpy(field.data() + dstIdx, recvBufBot.data(), sliceSize * sizeof(double));
        }
    };

    // Initial ghost cell exchange before simulation
    exchangeGhostCells(cold);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential on local domain
        computeChemicalPotential(cold, mu, nx, ny, nzLocal, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);

        // Update concentration on local domain
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzLocal, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);

        // Exchange ghost cells for next iteration
        exchangeGhostCells(cold);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Calculate performance using global grid size
    const size_t gridSize = nx * ny * nz;
    double cellUpdates = static_cast<double>(gridSize) * iterations;
    double mcups = cellUpdates / (static_cast<double>(duration.count()) / 1000.0) / 1e6;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather all data to rank 0 for validation and results printing
    std::vector<double> globalCold;
    if (rank == 0) {
        globalCold.resize(gridSize);
    }

    // Gather local sizes
    std::vector<int> localSizes(numRanks);
    const int localSizeInt = static_cast<int>(localSize);
    MPI_Gather(&localSizeInt, 1, MPI_INT, localSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacements for gatherv
    std::vector<int> displs(numRanks, 0);
    for (int r = 1; r < numRanks; ++r) {
        displs[r] = displs[r - 1] + localSizes[r - 1];
    }

    // Gather all local arrays to rank 0
    if (rank == 0) {
        const int totalLocalSize = displs[numRanks - 1] + localSizes[numRanks - 1];
        std::vector<double> gathered(totalLocalSize);
        MPI_Gatherv(cold.data(), localSizeInt, MPI_DOUBLE,
                    gathered.data(), localSizes.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        // Reorder into global grid layout (z-major: each rank's data is contiguous in Z)
        for (int r = 0; r < numRanks; ++r) {
            const size_t rZStart = r * nzPerRank + (r < static_cast<int>(remainder) ? r : remainder);
            const size_t rZCount = (r < static_cast<int>(remainder) ? nzPerRank + 1 : nzPerRank);
            // Copy interior data (skip ghost cells) from gathered buffer
            for (size_t z = 0; z < rZCount; ++z) {
                const size_t srcOff = static_cast<size_t>(displs[r]) + (z + 1) * sliceSize;
                const size_t dstOff = (rZStart + z) * sliceSize;
                std::memcpy(globalCold.data() + dstOff, gathered.data() + srcOff, sliceSize * sizeof(double));
            }
        }
    } else {
        MPI_Gatherv(cold.data(), localSizeInt, MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(globalCold, "Concentration");
    }

    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(globalCold, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();

    return 0;
}
