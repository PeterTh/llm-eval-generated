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

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void computeZDecomposition(const size_t nz, const int size, const int rank, size_t& zStart, size_t& localNz) {
    const size_t sizeCount = static_cast<size_t>(size);
    const size_t base = nz / sizeCount;
    const size_t rem = nz % sizeCount;
    if (static_cast<size_t>(rank) < rem) {
        localNz = base + 1;
        zStart = static_cast<size_t>(rank) * (base + 1);
    } else {
        localNz = base;
        zStart = rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
    }
}

void initializeLocalGrid(std::vector<Real>& grid,
                         const size_t nx,
                         const size_t ny,
                         const size_t localNz,
                         const size_t zStart) {
    const size_t plane = nx * ny;
    std::fill(grid.begin(), grid.end(), 0.0);
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = zStart + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIdx = idx3(x, y, globalZ, nx, ny);
                const size_t localIdx = z * plane + y * nx + x;
                grid[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& grid,
                   const size_t nx,
                   const size_t ny,
                   const size_t localNz,
                   const int rank,
                   const int size,
                   const MPI_Comm comm) {
    if (localNz == 0 || size == 1) {
        return;
    }
    const size_t plane = nx * ny;
    Real* data = grid.data();
    MPI_Request requests[4];
    int requestCount = 0;

    if (rank > 0) {
        MPI_Irecv(data, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[requestCount++]);
        MPI_Isend(data + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[requestCount++]);
    }
    if (rank + 1 < size) {
        MPI_Irecv(data + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[requestCount++]);
        MPI_Isend(data + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[requestCount++]);
    }

    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
}

// 7-point stencil computation on local domain with halos
void stencilIterationLocal(const std::vector<Real>& input,
                           std::vector<Real>& output,
                           const size_t nx,
                           const size_t ny,
                           const size_t localNz,
                           const size_t zStart,
                           const size_t nz) {
    if (localNz == 0) {
        return;
    }
    const size_t plane = nx * ny;
    const Real* in = input.data();
    Real* out = output.data();

    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = zStart + (z - 1);
        const size_t zOffset = z * plane;

        if (globalZ == 0 || globalZ + 1 == nz) {
            std::memcpy(out + zOffset, in + zOffset, plane * sizeof(Real));
            continue;
        }

        std::memcpy(out + zOffset, in + zOffset, nx * sizeof(Real));
        std::memcpy(out + zOffset + (ny - 1) * nx, in + zOffset + (ny - 1) * nx, nx * sizeof(Real));

        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = zOffset + y * nx;
            out[row] = in[row];
            out[row + (nx - 1)] = in[row + (nx - 1)];
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = row + x;
                const Real center = in[idx];
                out[idx] = (center +
                            in[idx - 1] + in[idx + 1] +
                            in[idx - nx] + in[idx + nx] +
                            in[idx - plane] + in[idx + plane]) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
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

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool shouldExit = false;
    int exitCode = 0;
    
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
            shouldExit = true;
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            shouldExit = true;
            exitCode = 1;
            break;
        }
    }

    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Grid plane too large for MPI counts\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t zStart = 0;
    size_t localNz = 0;
    computeZDecomposition(nz, size, rank, zStart, localNz);

    if (localNz * plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Local grid too large for MPI counts\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate grids with halo layers (double buffering)
    std::vector<Real> grid1((localNz + 2) * plane);
    std::vector<Real> grid2((localNz + 2) * plane);

    // Initialize
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeLocalGrid(grid1, nx, ny, localNz, zStart);

    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeHalos(grid1, nx, ny, localNz, rank, size, MPI_COMM_WORLD);
            stencilIterationLocal(grid1, grid2, nx, ny, localNz, zStart, nz);
        } else {
            exchangeHalos(grid2, nx, ny, localNz, rank, size, MPI_COMM_WORLD);
            stencilIterationLocal(grid2, grid1, nx, ny, localNz, zStart, nz);
        }
    }

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / maxElapsed / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        if (rank == 0) {
            const size_t gridSize = nx * ny * nz;
            if (gridSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
                printf("Global grid too large for MPI gather\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            finalGrid.resize(gridSize);
        }

        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            recvCounts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                size_t rStart = 0;
                size_t rNz = 0;
                computeZDecomposition(nz, size, r, rStart, rNz);
                const size_t count = rNz * plane;
                if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    rStart * plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    printf("MPI gather counts too large\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                recvCounts[r] = static_cast<int>(count);
                displs[r] = static_cast<int>(rStart * plane);
            }
        }

        const int sendCount = static_cast<int>(localNz * plane);
        const Real* sendBuf = (localNz > 0) ? (finalLocal.data() + plane) : finalLocal.data();
        MPI_Gatherv(sendBuf,
                    sendCount,
                    MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        int validInt = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateResult(finalGrid, nx, ny, nz);
            validInt = valid ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validInt == 1 ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
