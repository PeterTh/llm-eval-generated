#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

// The global grid is split into contiguous z slabs.  Two extra xy planes hold
// halos, making every message contiguous and avoiding pack/unpack overhead.
struct Slab {
    size_t firstZ;
    size_t planes;
};

Slab decompose(const size_t nz, const int rank, const int ranks) noexcept {
    const size_t quotient = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    const size_t planes = quotient + (r < remainder ? 1 : 0);
    const size_t first = r * quotient + std::min(r, remainder);
    return {first, planes};
}

inline size_t localIndex(const size_t x, const size_t y, const size_t z,
                         const size_t nx, const size_t planeSize) noexcept {
    return z * planeSize + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t nz, const Slab slab) {
    const size_t planeSize = nx * ny;
    // Initialize owned planes and both halos.  Only owned values are observed;
    // initialized halos also make physical-edge behavior deterministic.
    for (size_t localZ = 0; localZ < slab.planes + 2; ++localZ) {
        const long long globalZ = static_cast<long long>(slab.firstZ) +
                                  static_cast<long long>(localZ) - 1;
        if (globalZ < 0 || globalZ >= static_cast<long long>(nz)) {
            continue;
        }
        const size_t globalBase = static_cast<size_t>(globalZ) * planeSize;
        Real* const plane = grid.data() + localZ * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            plane[i] = static_cast<Real>((globalBase + i) % 19);
        }
    }
}

// Compute a range of owned local z planes. x/y boundaries and the two global
// z boundaries are deliberately untouched: both buffers were initialized to
// their invariant values before the iteration loop.
inline void computePlanes(const Real* __restrict input,
                          Real* __restrict output, const size_t nx,
                          const size_t ny, const size_t planeSize,
                          const size_t firstGlobalZ, const size_t globalNz,
                          const size_t beginLocalZ,
                          const size_t endLocalZ) noexcept {
    constexpr Real oneSeventh = 1.0 / 7.0;
    for (size_t z = beginLocalZ; z < endLocalZ; ++z) {
        const size_t globalZ = firstGlobalZ + z - 1;
        if (globalZ == 0 || globalZ + 1 == globalNz) {
            continue;
        }
        const size_t zOffset = z * planeSize;
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t begin = zOffset + y * nx + 1;
            const size_t end = zOffset + y * nx + nx - 1;
            for (size_t i = begin; i < end; ++i) {
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - nx] + input[i + nx] +
                             input[i - planeSize] + input[i + planeSize]) *
                            oneSeventh;
            }
        }
    }
}

void stencilIteration(std::vector<Real>& input,
                      std::vector<Real>& output, const size_t nx,
                      const size_t ny, const size_t globalNz, const Slab slab,
                      const int rank, const int activeRanks,
                      MPI_Comm communicator) {
    if (slab.planes == 0) {
        return;
    }

    const size_t planeSize = nx * ny;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
    MPI_Request requests[4];

    // Tags identify the direction in which a plane travels.
    MPI_Irecv(input.data(), static_cast<int>(planeSize), MPI_DOUBLE, lower, 1,
              communicator, &requests[0]);
    MPI_Irecv(input.data() + (slab.planes + 1) * planeSize,
              static_cast<int>(planeSize), MPI_DOUBLE, upper, 0, communicator,
              &requests[1]);
    MPI_Isend(input.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
              lower, 0, communicator, &requests[2]);
    MPI_Isend(input.data() + slab.planes * planeSize,
              static_cast<int>(planeSize), MPI_DOUBLE, upper, 1, communicator,
              &requests[3]);

    // Planes 2..N-1 cannot depend on either incoming halo.
    if (slab.planes > 2) {
        computePlanes(input.data(), output.data(), nx, ny, planeSize,
                      slab.firstZ, globalNz, 2, slab.planes);
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    computePlanes(input.data(), output.data(), nx, ny, planeSize, slab.firstZ,
                  globalNz, 1, 2);
    if (slab.planes > 1) {
        computePlanes(input.data(), output.data(), nx, ny, planeSize,
                      slab.firstZ, globalNz, slab.planes, slab.planes + 1);
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t planeSize,
                    const Slab slab, MPI_Comm communicator, const int rank) {
    bool localFinite = true;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    const Real* const begin = grid.data() + planeSize;
    const Real* const end = begin + slab.planes * planeSize;
    for (const Real* value = begin; value != end; ++value) {
        localFinite = localFinite && std::isfinite(*value);
        localMin = std::min(localMin, *value);
        localMax = std::max(localMax, *value);
    }

    int finite = localFinite ? 1 : 0;
    int allFinite = 0;
    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_LAND, communicator);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, communicator);

    if (rank == 0) {
        if (!allFinite) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return allFinite && globalMax <= 1e6 && globalMin >= -1e6;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool parseError = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parseError = true;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (help || parseError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    const bool dimensionsValid = nx >= 2 && ny >= 2 && nz >= 2;
    const bool planeFitsMpi = ny == 0 || nx <=
        static_cast<size_t>(std::numeric_limits<int>::max()) / ny;
    const bool allocationFits = planeFitsMpi && nz <=
        std::numeric_limits<size_t>::max() / (nx * ny);
    if (!dimensionsValid || iterations < 0 || !allocationFits ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Grid dimensions must be at least 2, iterations must "
                         "be nonnegative, and each xy plane must fit an MPI "
                         "message.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = nx * ny;
    const Slab slab = decompose(nz, rank, ranks);
    const int activeRanks = static_cast<int>(std::min(nz,
                                         static_cast<size_t>(ranks)));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> grid1((slab.planes + 2) * planeSize);
    std::vector<Real> grid2((slab.planes + 2) * planeSize);
    initializeGrid(grid1, nx, ny, nz, slab);
    grid2 = grid1;

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if ((iteration & 1) == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, slab, rank,
                             activeRanks, MPI_COMM_WORLD);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, slab, rank,
                             activeRanks, MPI_COMM_WORLD);
        }
    }

    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const std::vector<Real>& finalGrid =
        (iterations & 1) == 0 ? grid1 : grid2;
    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double updates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
        const double mcups = seconds > 0.0 ? updates / seconds / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<Real> gathered;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            for (int process = 0; process < ranks; ++process) {
                const Slab part = decompose(nz, process, ranks);
                counts[static_cast<size_t>(process)] =
                    static_cast<int>(part.planes * planeSize);
                displacements[static_cast<size_t>(process)] =
                    static_cast<int>(part.firstZ * planeSize);
            }
            gathered.resize(nx * ny * nz);
        }
        MPI_Gatherv(finalGrid.data() + planeSize,
                    static_cast<int>(slab.planes * planeSize), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) print_results(gathered, "Grid");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(finalGrid, planeSize, slab, MPI_COMM_WORLD, rank)
                    ? 1
                    : 0;
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
