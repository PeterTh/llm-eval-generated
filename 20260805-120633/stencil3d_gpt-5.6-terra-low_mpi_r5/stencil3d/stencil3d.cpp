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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                           size_t localNz, size_t globalZ0) {
    for (size_t z = 0; z < localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = idx3(x, y, globalZ0 + z, nx, ny);
                grid[idx3(x, y, z + 1, nx, ny)] = static_cast<Real>(globalIndex % 19);
            }
}

// Update one owned plane.  Local storage has one ghost plane at either end.
static inline void updatePlane(const Real* input, Real* output, size_t z,
                               size_t globalZ, size_t nx, size_t ny, size_t nz) {
    if (globalZ == 0 || globalZ + 1 == nz)
        return;
    const size_t plane = nx * ny;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = z * plane + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - plane] + input[i + plane]) / 7.0;
        }
    }
}

static void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                             size_t nx, size_t ny, size_t nz, size_t localNz,
                             size_t globalZ0, int prev, int next, MPI_Comm comm) {
    const size_t plane = nx * ny;
    // This also preserves all global x/y boundaries exactly as in the serial code.
    std::memcpy(output.data() + plane, input.data() + plane, localNz * plane * sizeof(Real));

    MPI_Request requests[4];
    int nrequests = 0;
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1, comm, &requests[nrequests++]);
        MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0, comm, &requests[nrequests++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(input.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, comm, &requests[nrequests++]);
        MPI_Isend(input.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1, comm, &requests[nrequests++]);
    }

    // Planes independent of halos run while communication is in flight.
    for (size_t localZ = 1; localZ + 1 < localNz; ++localZ)
        updatePlane(input.data(), output.data(), localZ + 1, globalZ0 + localZ, nx, ny, nz);

    MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
    updatePlane(input.data(), output.data(), 1, globalZ0, nx, ny, nz);
    if (localNz > 1)
        updatePlane(input.data(), output.data(), localNz, globalZ0 + localNz - 1, nx, ny, nz);
}

static bool validateResult(const std::vector<Real>& localGrid, size_t nx, size_t ny, size_t localNz,
                           MPI_Comm comm, int rank) {
    const size_t plane = nx * ny;
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
        localBad |= !std::isfinite(localGrid[i]);
        localMin = std::min(localMin, localGrid[i]);
        localMax = std::max(localMax, localGrid[i]);
    }
    int bad = 0;
    Real minVal, maxVal;
    MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, 0, comm);
    MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) return true;
    if (bad) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) { std::printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("  -x <num>  Grid X size (default: 128)\n  -y <num>  Grid Y size (default: X)\n");
    std::printf("  -z <num>  Grid Z size (default: X)\n  -i <num>  Iterations (default: 10)\n");
    std::printf("  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!worldRank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!worldRank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!worldRank) std::printf("Grid dimensions must be at least 3 and plane size must fit MPI counts.\n");
        MPI_Finalize(); return 1;
    }

    const int activeSize = std::min<size_t>(worldSize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeSize) { MPI_Finalize(); return 0; }
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    const size_t base = nz / size, extra = nz % size;
    const size_t localNz = base + (static_cast<size_t>(rank) < extra);
    const size_t globalZ0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, globalZ0);
    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

    if (!rank) {
        std::printf("3D Stencil Benchmark (MPI ranks: %d)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n",
                    size, nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (!(iter & 1)) stencilIteration(grid1, grid2, nx, ny, nz, localNz, globalZ0, prev, next, comm);
        else stencilIteration(grid2, grid1, nx, ny, nz, localNz, globalZ0, prev, next, comm);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const std::vector<Real>& finalGrid = (iterations & 1) ? grid2 : grid1;
    if (!rank) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }
    if (printResults) {
        std::vector<int> counts(size), offsets(size);
        for (int r = 0; r < size; ++r) { const size_t slabs = base + (static_cast<size_t>(r) < extra); counts[r] = static_cast<int>(slabs * plane); offsets[r] = r ? offsets[r-1] + counts[r-1] : 0; }
        std::vector<Real> fullGrid(rank == 0 ? nx * ny * nz : 0);
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE, fullGrid.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        if (!rank) print_results(fullGrid, "Grid");
    }
    if (validate) { if (!rank) std::printf("Validating result...\n"); const bool valid = validateResult(finalGrid, nx, ny, localNz, comm, rank); if (!valid) exitCode = 1; MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm); if (!rank) std::printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED"); }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
