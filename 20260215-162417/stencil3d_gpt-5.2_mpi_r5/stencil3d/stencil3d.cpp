#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void partitionZ(const size_t nz, const int commSize, const int rank, size_t& zStart, size_t& localNz) {
    const size_t p = static_cast<size_t>(commSize);
    const size_t r = static_cast<size_t>(rank);

    const size_t base = (p == 0) ? 0 : (nz / p);
    const size_t rem = (p == 0) ? 0 : (nz % p);

    localNz = base + ((r < rem) ? 1 : 0);
    zStart = r * base + ((r < rem) ? r : rem);
}

inline void stencilPlane(const Real* __restrict in, Real* __restrict out,
                         const size_t lz, const size_t zStart,
                         const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    const size_t gZ = zStart + (lz - 1);

    const Real* __restrict in_c = in + lz * plane;
    Real* __restrict out_c = out + lz * plane;

    if (gZ == 0 || gZ + 1 == nz || nx < 3 || ny < 3) {
        std::memcpy(out_c, in_c, plane * sizeof(Real));
        return;
    }

    // Copy y-boundary rows
    std::memcpy(out_c, in_c, nx * sizeof(Real));
    std::memcpy(out_c + (ny - 1) * nx, in_c + (ny - 1) * nx, nx * sizeof(Real));

    const Real* __restrict in_zm1 = in + (lz - 1) * plane;
    const Real* __restrict in_zp1 = in + (lz + 1) * plane;

    constexpr Real inv7 = Real(1.0 / 7.0);

    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = y * nx;

        // Copy x-boundary values
        out_c[row] = in_c[row];
        out_c[row + (nx - 1)] = in_c[row + (nx - 1)];

        const Real* __restrict in_row = in_c + row;
        const Real* __restrict in_row_m = in_c + row - nx;
        const Real* __restrict in_row_p = in_c + row + nx;
        const Real* __restrict in_row_zm1 = in_zm1 + row;
        const Real* __restrict in_row_zp1 = in_zp1 + row;
        Real* __restrict out_row = out_c + row;

        for (size_t x = 1; x + 1 < nx; ++x) {
            out_row[x] = (in_row[x] + in_row[x - 1] + in_row[x + 1] +
                          in_row_m[x] + in_row_p[x] +
                          in_row_zm1[x] + in_row_zp1[x]) * inv7;
        }
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
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
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", commSize);
    }

    size_t zStart = 0;
    size_t localNz = 0;
    partitionZ(nz, commSize, rank, zStart, localNz);

    const size_t plane = nx * ny;
    const int planeCount = static_cast<int>(plane);
    const size_t localSize = (localNz + 2) * plane;

    // Allocate local grids with Z-halos (double buffering)
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize owned region with the same global indexing formula as the serial code
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t gZ = zStart + (lz - 1);
        Real* __restrict planePtr = grid1.data() + lz * plane;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gIdx = idx3(x, y, gZ, nx, ny);
                planePtr[y * nx + x] = static_cast<Real>(gIdx % 19);
            }
        }
    }

    const int prev = (localNz == 0 || zStart == 0) ? MPI_PROC_NULL : (rank - 1);
    const int next = (localNz == 0 || (zStart + localNz >= nz)) ? MPI_PROC_NULL : (rank + 1);

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* __restrict in = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* __restrict out = (iter % 2 == 0) ? grid2.data() : grid1.data();

        if (localNz > 0) {
            MPI_Request reqs[4];
            int nreq = 0;

            // Exchange Z halo planes
            if (prev != MPI_PROC_NULL) {
                MPI_Irecv(in + 0 * plane, planeCount, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[nreq++]);
                MPI_Isend(in + 1 * plane, planeCount, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            }
            if (next != MPI_PROC_NULL) {
                MPI_Irecv(in + (localNz + 1) * plane, planeCount, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[nreq++]);
                MPI_Isend(in + localNz * plane, planeCount, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            }

            // Overlap computation for planes that do not depend on halos
            if (localNz > 2) {
                for (size_t lz = 2; lz + 1 <= localNz; ++lz) {
                    stencilPlane(in, out, lz, zStart, nx, ny, nz);
                }
            }

            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

            // Compute planes adjacent to halos
            stencilPlane(in, out, 1, zStart, nx, ny, nz);
            if (localNz >= 2) {
                stencilPlane(in, out, localNz, zStart, nx, ny, nz);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double cellUpdates = static_cast<double>((nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0)) * iterations;
        const double mcups = (maxTime > 0.0) ? (cellUpdates / maxTime / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Final local buffer (including halos)
    Real* __restrict finalLocal = (iterations % 2 == 0) ? grid1.data() : grid2.data();

    int validInt = 1;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        std::vector<int> recvcounts;
        std::vector<int> displs;

        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
            recvcounts.resize(commSize);
            displs.resize(commSize);

            for (int r = 0; r < commSize; ++r) {
                size_t zs = 0, ln = 0;
                partitionZ(nz, commSize, r, zs, ln);
                recvcounts[r] = static_cast<int>(ln * plane);
                displs[r] = static_cast<int>(zs * plane);
            }
        }

        const int sendcount = static_cast<int>(localNz * plane);
        MPI_Gatherv(finalLocal + 1 * plane, sendcount, MPI_DOUBLE,
                    (rank == 0) ? finalGrid.data() : nullptr,
                    (rank == 0) ? recvcounts.data() : nullptr,
                    (rank == 0) ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool ok = validateResult(finalGrid, nx, ny, nz);
                validInt = ok ? 1 : 0;
                printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            }
        }
    }

    if (validate) {
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validInt ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
