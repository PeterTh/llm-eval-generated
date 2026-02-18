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

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// 3D index calculation (global)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline void decompose1d(const size_t n, const int rank, const int size, size_t& start, size_t& count) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    count = base + (r < rem ? 1u : 0u);
    start = r * base + (r < rem ? r : rem);
}

static inline void initializeLocalGrid(std::vector<Real>& gridWithGhost,
                                      const size_t nx, const size_t ny,
                                      const size_t z0, const size_t localZ) {
    const size_t plane = nx * ny;
    // Owned planes are [1..localZ]; ghosts are [0] and [localZ+1]
    for (size_t lz = 1; lz <= localZ; ++lz) {
        const size_t gz = z0 + (lz - 1);
        Real* RESTRICT planePtr = gridWithGhost.data() + lz * plane;
        const size_t baseIdx = gz * plane;
        for (size_t i = 0; i < plane; ++i) {
            planePtr[i] = static_cast<Real>((baseIdx + i) % 19u);
        }
    }
}

static inline void computePlane(const Real* RESTRICT in, Real* RESTRICT out,
                               const size_t nx, const size_t ny,
                               const size_t plane, const size_t lz,
                               const size_t gz, const size_t nzGlobal) {
    const Real* RESTRICT inPlane = in + lz * plane;
    Real* RESTRICT outPlane = out + lz * plane;

    // Global Z-boundary: copy entire plane
    if (gz == 0 || gz + 1 == nzGlobal) {
        std::memcpy(outPlane, inPlane, plane * sizeof(Real));
        return;
    }

    // Copy X/Y boundaries for this plane
    for (size_t x = 0; x < nx; ++x) {
        outPlane[x] = inPlane[x];
        outPlane[(ny - 1) * nx + x] = inPlane[(ny - 1) * nx + x];
    }
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = y * nx;
        outPlane[row] = inPlane[row];
        outPlane[row + (nx - 1)] = inPlane[row + (nx - 1)];
    }

    const Real* RESTRICT inBelow = in + (lz - 1) * plane;
    const Real* RESTRICT inAbove = in + (lz + 1) * plane;

    const Real inv7 = static_cast<Real>(1.0 / 7.0);
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            const Real center = inPlane[i];
            const Real left = inPlane[i - 1];
            const Real right = inPlane[i + 1];
            const Real front = inPlane[i - nx];
            const Real back = inPlane[i + nx];
            const Real bottom = inBelow[i];
            const Real top = inAbove[i];
            outPlane[i] = (center + left + right + front + back + bottom + top) * inv7;
        }
    }
}

static inline void stencilIterationMPI(std::vector<Real>& in, std::vector<Real>& out,
                                      const size_t nx, const size_t ny, const size_t nzGlobal,
                                      const size_t z0, const size_t localZ,
                                      const int rank, const int size, MPI_Comm comm) {
    const size_t plane = nx * ny;

    // Exchange halos in Z (ghost planes of input)
    MPI_Request reqs[4];
    int reqCount = 0;

    const int prev = rank - 1;
    const int next = rank + 1;

    if (prev >= 0 && localZ > 0) {
        MPI_Irecv(in.data() + 0 * plane, static_cast<int>(plane), MPI_DOUBLE, prev, 100, comm, &reqs[reqCount++]);
        MPI_Isend(in.data() + 1 * plane, static_cast<int>(plane), MPI_DOUBLE, prev, 200, comm, &reqs[reqCount++]);
    }
    if (next < size && localZ > 0) {
        MPI_Irecv(in.data() + (localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 200, comm, &reqs[reqCount++]);
        MPI_Isend(in.data() + localZ * plane, static_cast<int>(plane), MPI_DOUBLE, next, 100, comm, &reqs[reqCount++]);
    }

    // Compute bulk planes that do not depend on halos
    if (localZ >= 3) {
        for (size_t lz = 2; lz <= localZ - 1; ++lz) {
            const size_t gz = z0 + (lz - 1);
            computePlane(in.data(), out.data(), nx, ny, plane, lz, gz, nzGlobal);
        }
    }

    if (reqCount > 0) {
        MPI_Waitall(reqCount, reqs, MPI_STATUSES_IGNORE);
    }

    // Compute edge planes (may depend on halos)
    if (localZ > 0) {
        const size_t gzFirst = z0;
        computePlane(in.data(), out.data(), nx, ny, plane, 1, gzFirst, nzGlobal);
        if (localZ > 1) {
            const size_t gzLast = z0 + (localZ - 1);
            computePlane(in.data(), out.data(), nx, ny, plane, localZ, gzLast, nzGlobal);
        }
    }
}

static inline bool validateResultMPI(const std::vector<Real>& localGridWithGhost,
                                    const size_t nx, const size_t ny,
                                    const size_t localZ, MPI_Comm comm, const int rank) {
    const size_t plane = nx * ny;
    const size_t n = localZ * plane;
    const Real* RESTRICT data = localGridWithGhost.data() + 1 * plane;

    int localOk = 1;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();

    for (size_t i = 0; i < n; ++i) {
        const Real v = data[i];
        if (std::isnan(v) || std::isinf(v)) {
            localOk = 0;
            break;
        }
        localMin = std::min(localMin, v);
        localMax = std::max(localMax, v);
    }

    int globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_LAND, comm);

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    if (!globalOk) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation (gather to rank 0)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    int validate = 0;
    int printResults = 0;

    int earlyExit = 0;
    int earlyCode = 0;

    if (worldRank == 0) {
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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                earlyExit = 1;
                earlyCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                earlyExit = 1;
                earlyCode = 1;
                break;
            }
        }

        if (!earlyExit) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }

    // Broadcast parameters + early-exit
    {
        unsigned long long params[3] = {static_cast<unsigned long long>(nx), static_cast<unsigned long long>(ny),
                                        static_cast<unsigned long long>(nz)};
        MPI_Bcast(params, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        nx = static_cast<size_t>(params[0]);
        ny = static_cast<size_t>(params[1]);
        nz = static_cast<size_t>(params[2]);
        MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&earlyExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&earlyCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (earlyExit) {
        MPI_Finalize();
        return earlyCode;
    }

    // Use at most nz ranks (avoid empty ranks at the end participating as neighbors)
    const int activeSize = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    const int color = (worldRank < activeSize) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &comm);

    if (color == MPI_UNDEFINED) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    size_t z0 = 0, localZ = 0;
    decompose1d(nz, rank, size, z0, localZ);

    const size_t plane = nx * ny;
    std::vector<Real> grid1((localZ + 2) * plane);
    std::vector<Real> grid2((localZ + 2) * plane);

    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeLocalGrid(grid1, nx, ny, z0, localZ);

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIterationMPI(grid1, grid2, nx, ny, nz, z0, localZ, rank, size, comm);
        } else {
            stencilIterationMPI(grid2, grid1, nx, ny, nz, z0, localZ, rank, size, comm);
        }
    }

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();

    const auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDuration);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * static_cast<double>(iterations);
        const double mcups = cellUpdates / (static_cast<double>(globalDuration) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults) {
        std::vector<Real> global;
        std::vector<int> counts;
        std::vector<int> displs;

        if (rank == 0) {
            global.resize(nx * ny * nz);
            counts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                size_t s = 0, c = 0;
                decompose1d(nz, r, size, s, c);
                counts[static_cast<size_t>(r)] = static_cast<int>(c * plane);
                displs[static_cast<size_t>(r)] = static_cast<int>(s * plane);
            }
        }

        const int sendCount = static_cast<int>(localZ * plane);
        MPI_Gatherv(const_cast<Real*>(finalLocal.data() + 1 * plane), sendCount, MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, comm);

        if (rank == 0) {
            print_results(global, "Grid");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResultMPI(finalLocal, nx, ny, localZ, comm, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Comm_free(&comm);
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
