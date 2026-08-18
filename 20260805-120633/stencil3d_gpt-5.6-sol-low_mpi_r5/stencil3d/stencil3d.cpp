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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

struct Slab {
    size_t begin;
    size_t count;
};

static Slab slabForRank(size_t nz, int rank, int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    return {static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra),
            base + (static_cast<size_t>(rank) < extra)};
}

static void initializeGrids(std::vector<Real>& a, std::vector<Real>& b,
                            size_t nx, size_t ny, const Slab& slab) {
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= slab.count; ++lz) {
        const size_t gz = slab.begin + lz - 1;
        Real* const ap = a.data() + lz * plane;
        Real* const bp = b.data() + lz * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>((gz * plane + i) % 19);
            ap[i] = value;
            // Global boundary cells never change.  Pre-populating both buffers
            // removes boundary copies from every timed iteration.
            const size_t y = i / nx;
            const size_t x = i - y * nx;
            if (gz == 0 || x == 0 || x + 1 == nx || y == 0 || y + 1 == ny)
                bp[i] = value;
        }
    }
}

static inline void stencilPlane(const Real* __restrict input,
                                Real* __restrict output, size_t lz, size_t gz,
                                size_t nx, size_t ny, size_t nz) {
    if (gz == 0 || gz + 1 == nz) return;
    const size_t plane = nx * ny;
    const Real oneSeventh = 1.0 / 7.0;
    const size_t zoff = lz * plane;
    for (size_t y = 1; y + 1 < ny; ++y) {
        size_t i = zoff + y * nx + 1;
        const size_t end = zoff + y * nx + nx - 1;
        for (; i < end; ++i) {
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] + input[i - plane] +
                         input[i + plane]) * oneSeventh;
        }
    }
}

static void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                             size_t nx, size_t ny, size_t nz, const Slab& slab,
                             int rank, int activeRanks, MPI_Comm comm) {
    if (slab.count == 0) return;
    const size_t plane = nx * ny;
    const int mpiPlane = static_cast<int>(plane);
    MPI_Request requests[4];
    int nreq = 0;

    if (rank > 0) {
        MPI_Irecv(input.data(), mpiPlane, MPI_DOUBLE, rank - 1, 101, comm, &requests[nreq++]);
        MPI_Isend(input.data() + plane, mpiPlane, MPI_DOUBLE, rank - 1, 100, comm,
                  &requests[nreq++]);
    }
    if (rank + 1 < activeRanks) {
        MPI_Irecv(input.data() + (slab.count + 1) * plane, mpiPlane, MPI_DOUBLE,
                  rank + 1, 100, comm, &requests[nreq++]);
        MPI_Isend(input.data() + slab.count * plane, mpiPlane, MPI_DOUBLE,
                  rank + 1, 101, comm, &requests[nreq++]);
    }

    // These planes do not depend on newly received halos.
    for (size_t lz = 2; lz < slab.count; ++lz)
        stencilPlane(input.data(), output.data(), lz, slab.begin + lz - 1, nx, ny, nz);

    if (nreq) MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    stencilPlane(input.data(), output.data(), 1, slab.begin, nx, ny, nz);
    if (slab.count > 1)
        stencilPlane(input.data(), output.data(), slab.count,
                     slab.begin + slab.count - 1, nx, ny, nz);
}

static bool validateResult(const std::vector<Real>& grid, size_t plane,
                           const Slab& slab, int rank, MPI_Comm comm) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localValid = 1;
    for (size_t i = plane; i < (slab.count + 1) * plane; ++i) {
        const Real value = grid[i];
        if (!std::isfinite(value)) localValid = 0;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }
    Real globalMin, globalMax;
    int globalValid;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, 0, comm);
    if (rank != 0) return true;
    if (!globalValid) std::printf("Validation failed: found NaN or Inf value\n");
    std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 1e6 || globalMin < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        globalValid = 0;
    }
    return globalValid != 0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false;
    bool argsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); argsValid = false; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t intMax = static_cast<size_t>(std::numeric_limits<int>::max());
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 || nx > intMax ||
        ny > intMax / nx || nx * ny > intMax || nz > intMax / (nx * ny)) argsValid = false;
    if (help || !argsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }

    const Slab slab = slabForRank(nz, rank, ranks);
    const int activeRanks = std::min<size_t>(nz, static_cast<size_t>(ranks));
    const size_t plane = nx * ny;
    std::vector<Real> grid1((slab.count + 2) * plane);
    std::vector<Real> grid2((slab.count + 2) * plane);
    initializeGrids(grid1, grid2, nx, ny, slab);
    // Fix the two global Z boundary planes in the second buffer.
    if (slab.count && slab.begin == 0)
        std::copy_n(grid1.data() + plane, plane, grid2.data() + plane);
    if (slab.count && slab.begin + slab.count == nz)
        std::copy_n(grid1.data() + slab.count * plane, plane, grid2.data() + slab.count * plane);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\nInitializing grid...\n", iterations,
                    validate ? "enabled" : "disabled");
        std::printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) stencilIteration(grid1, grid2, nx, ny, nz, slab, rank, activeRanks, MPI_COMM_WORLD);
        else stencilIteration(grid2, grid1, nx, ny, nz, slab, rank, activeRanks, MPI_COMM_WORLD);
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        std::printf("Performance: %.3f MCellUpdates/s\n", updates / elapsed / 1e6);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<int> counts(ranks), displs(ranks);
        for (int r = 0; r < ranks; ++r) {
            const Slab s = slabForRank(nz, r, ranks);
            counts[r] = static_cast<int>(s.count * plane);
            displs[r] = static_cast<int>(s.begin * plane);
        }
        std::vector<Real> gathered;
        if (rank == 0) gathered.resize(nx * ny * nz);
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(slab.count * plane), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) print_results(gathered, "Grid");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(finalGrid, plane, slab, rank, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return (rank == 0 && !valid) ? 1 : 0;
}
