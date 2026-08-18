#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Slab {
    size_t firstZ;
    size_t nz;
};

static Slab decompose(const size_t nz, const int rank, const int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, extra), base + (r < extra ? 1U : 0U)};
}

static void initializeGrid(std::vector<Real>& grid, const size_t nx,
                           const size_t ny, const Slab slab) {
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= slab.nz; ++lz) {
        const size_t globalBase = (slab.firstZ + lz - 1) * plane;
        Real* const dst = grid.data() + lz * plane;
        for (size_t i = 0; i < plane; ++i)
            dst[i] = static_cast<Real>((globalBase + i) % 19);
    }
}

// Update one owned plane.  Physical boundary cells retain their input value.
static inline void updatePlane(const Real* input, Real* output,
                               const size_t lz, const size_t globalZ,
                               const size_t nx, const size_t ny,
                               const size_t globalNz) {
    const size_t plane = nx * ny;
    const size_t base = lz * plane;
    if (globalZ == 0 || globalZ + 1 == globalNz) {
        std::memcpy(output + base, input + base, plane * sizeof(Real));
        return;
    }

    // Copy the four xy boundary edges (corners may harmlessly be copied twice).
    std::memcpy(output + base, input + base, nx * sizeof(Real));
    std::memcpy(output + base + (ny - 1) * nx,
                input + base + (ny - 1) * nx, nx * sizeof(Real));
    for (size_t y = 1; y + 1 < ny; ++y) {
        output[base + y * nx] = input[base + y * nx];
        output[base + y * nx + nx - 1] = input[base + y * nx + nx - 1];
    }

    constexpr Real oneSeventh = Real{1.0 / 7.0};
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - plane] + input[i + plane]) * oneSeventh;
        }
    }
}

static void stencilIteration(std::vector<Real>& input,
                             std::vector<Real>& output, const size_t nx,
                             const size_t ny, const size_t globalNz,
                             const Slab slab, const int rank, const int ranks,
                             MPI_Comm comm) {
    const size_t plane = nx * ny;
    MPI_Request requests[4];
    int nreq = 0;

    // Receive directly into ghost planes and send directly from owned planes.
    if (rank > 0) {
        MPI_Irecv(input.data(), static_cast<int>(plane),
                  MPI_DOUBLE, rank - 1, 101, comm, &requests[nreq++]);
        MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank - 1, 102, comm, &requests[nreq++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(input.data() + (slab.nz + 1) * plane,
                  static_cast<int>(plane), MPI_DOUBLE, rank + 1, 102, comm,
                  &requests[nreq++]);
        MPI_Isend(input.data() + slab.nz * plane, static_cast<int>(plane),
                  MPI_DOUBLE, rank + 1, 101, comm, &requests[nreq++]);
    }

    // These planes depend only on locally owned data, overlapping halo traffic.
    for (size_t lz = 2; lz < slab.nz; ++lz)
        updatePlane(input.data(), output.data(), lz, slab.firstZ + lz - 1,
                    nx, ny, globalNz);

    if (nreq != 0)
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    updatePlane(input.data(), output.data(), 1, slab.firstZ,
                nx, ny, globalNz);
    if (slab.nz > 1)
        updatePlane(input.data(), output.data(), slab.nz,
                    slab.firstZ + slab.nz - 1, nx, ny, globalNz);
}

static bool validateResult(const std::vector<Real>& grid, const size_t plane,
                           const size_t localNz, const int rank, MPI_Comm comm) {
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
        const Real value = grid[i];
        localBad |= !std::isfinite(value);
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }

    int bad = 0;
    Real minVal = 0.0, maxVal = 0.0;
    MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, 0, comm);
    MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) return true;

    if (bad) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldRanks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    bool help = false, argsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc)
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc)
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc)
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else {
            if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argsValid = false;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || !argsValid) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0)
            std::fprintf(stderr, "Grid dimensions must be at least 2 and each xy plane must fit an MPI message\n");
        MPI_Finalize();
        return 1;
    }

    // Never assign empty slabs. Extra launched ranks wait in MPI_Finalize.
    const int activeRanks = nz < static_cast<size_t>(worldRanks)
                                ? static_cast<int>(nz) : worldRanks;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const Slab slab = decompose(nz, rank, activeRanks);
    const size_t plane = nx * ny;
    if (nz > std::numeric_limits<size_t>::max() / plane ||
        slab.nz + 2 > std::numeric_limits<size_t>::max() / plane) {
        if (rank == 0) std::fprintf(stderr, "Grid is too large\n");
        MPI_Abort(comm, 1);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }
    std::vector<Real> grid1((slab.nz + 2) * plane);
    std::vector<Real> grid2((slab.nz + 2) * plane);
    initializeGrid(grid1, nx, ny, slab);

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0)
            stencilIteration(grid1, grid2, nx, ny, nz, slab, rank, activeRanks, comm);
        else
            stencilIteration(grid2, grid1, nx, ny, nz, slab, rank, activeRanks, comm);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double updates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<Real> gathered;
        const size_t totalElements = plane * nz;
        if (slab.nz * plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            totalElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI count limits\n");
            MPI_Abort(comm, 1);
        }
        if (rank == 0) {
            counts.resize(activeRanks);
            displacements.resize(activeRanks);
            gathered.resize(totalElements);
            for (int r = 0; r < activeRanks; ++r) {
                const Slab s = decompose(nz, r, activeRanks);
                counts[r] = static_cast<int>(s.nz * plane);
                displacements[r] = static_cast<int>(s.firstZ * plane);
            }
        }
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(slab.nz * plane),
                    MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(gathered, "Grid");
    }

    int returnCode = 0;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, plane, slab.nz, rank, comm);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
        MPI_Bcast(&returnCode, 1, MPI_INT, 0, comm);
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return returnCode;
}
