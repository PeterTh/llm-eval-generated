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

static inline size_t planeIndex(size_t x, size_t y, size_t z, size_t nx,
                                size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                           size_t localNz, size_t globalZ0) {
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t globalBase = (globalZ0 + lz - 1) * plane;
        Real* out = grid.data() + lz * plane;
        for (size_t i = 0; i < plane; ++i)
            out[i] = static_cast<Real>((globalBase + i) % 19);
    }
}

static inline void updatePlanes(const Real* input, Real* output, size_t nx,
                                size_t ny, size_t localBegin, size_t localEnd,
                                size_t globalZ0, size_t globalNz) {
    const size_t plane = nx * ny;
    constexpr Real oneSeventh = 1.0 / 7.0;
    for (size_t z = localBegin; z < localEnd; ++z) {
        const size_t gz = globalZ0 + z - 1;
        Real* dst = output + z * plane;
        const Real* src = input + z * plane;

        // The original program preserves all six global boundary faces.
        if (gz == 0 || gz + 1 == globalNz) {
            std::copy_n(src, plane, dst);
            continue;
        }
        std::copy_n(src, nx, dst);
        std::copy_n(src + (ny - 1) * nx, nx, dst + (ny - 1) * nx);
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = y * nx;
            dst[row] = src[row];
            dst[row + nx - 1] = src[row + nx - 1];
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = row + x;
                dst[i] = (src[i] + src[i - 1] + src[i + 1] +
                          src[i - nx] + src[i + nx] + src[i - plane] +
                          src[i + plane]) * oneSeventh;
            }
        }
    }
}

static void stencilIteration(const std::vector<Real>& input,
                             std::vector<Real>& output, size_t nx, size_t ny,
                             size_t localNz, size_t globalZ0, size_t globalNz,
                             int rank, int ranks, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const int count = static_cast<int>(plane);
    MPI_Request requests[4];
    int nreq = 0;

    // Receives are posted first. Sends use the owned end planes directly.
    if (rank > 0) {
        MPI_Irecv(const_cast<Real*>(input.data()), count, MPI_DOUBLE, rank - 1,
                  1, comm, &requests[nreq++]);
        MPI_Isend(input.data() + plane, count, MPI_DOUBLE, rank - 1, 2, comm,
                  &requests[nreq++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(const_cast<Real*>(input.data()) + (localNz + 1) * plane,
                  count, MPI_DOUBLE, rank + 1, 2, comm, &requests[nreq++]);
        MPI_Isend(input.data() + localNz * plane, count, MPI_DOUBLE, rank + 1,
                  1, comm, &requests[nreq++]);
    }

    // Planes not adjacent to a rank boundary do not depend on communication.
    if (localNz > 2)
        updatePlanes(input.data(), output.data(), nx, ny, 2, localNz,
                     globalZ0, globalNz);
    if (nreq) MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
    if (localNz >= 1)
        updatePlanes(input.data(), output.data(), nx, ny, 1, 2, globalZ0,
                     globalNz);
    if (localNz >= 2)
        updatePlanes(input.data(), output.data(), nx, ny, localNz,
                     localNz + 1, globalZ0, globalNz);
}

static bool validateResult(const std::vector<Real>& grid, size_t plane,
                           size_t localNz, int rank, MPI_Comm comm) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localValid = 1;
    for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
        const Real v = grid[i];
        if (!std::isfinite(v)) localValid = 0;
        localMin = std::min(localMin, v);
        localMax = std::max(localMax, v);
    }
    Real globalMin, globalMax;
    int globalValid;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, 0, comm);
    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        globalValid = globalValid && globalMax <= 1e6 && globalMin >= -1e6;
    }
    MPI_Bcast(&globalValid, 1, MPI_INT, 0, comm);
    return globalValid != 0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  Grid X size (default: 128)\n"
                "  -y <num>  Grid Y size (default: X)\n"
                "  -z <num>  Grid Z size (default: X)\n"
                "  -i <num>  Iterations (default: 10)\n"
                "  -v        Enable validation\n"
                "  -r        Print results for external validation\n"
                "  -h        Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else badArgs = true;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t plane = nx * ny;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        static_cast<size_t>(ranks) > nz || plane > static_cast<size_t>(std::numeric_limits<int>::max()))
        badArgs = true;
    if (help || badArgs) {
        if (rank == 0) {
            if (badArgs) std::fprintf(stderr, "Invalid dimensions/options, too many ranks, or MPI plane too large.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < extra);
    const size_t globalZ0 = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);
    std::vector<Real> grid1((localNz + 2) * plane);
    std::vector<Real> grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, globalZ0);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nMPI ranks: %d\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled", ranks);
        std::printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0)
            stencilIteration(grid1, grid2, nx, ny, localNz, globalZ0, nz, rank, ranks, MPI_COMM_WORLD);
        else
            stencilIteration(grid2, grid1, nx, ny, localNz, globalZ0, nz, rank, ranks, MPI_COMM_WORLD);
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const std::vector<Real>& finalGrid = (iterations & 1) ? grid2 : grid1;

    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0.0);
    }

    if (printResults) {
        const int localCount = static_cast<int>(localNz * plane);
        std::vector<int> counts, displs;
        std::vector<Real> gathered;
        if (rank == 0) {
            counts.resize(ranks); displs.resize(ranks);
            size_t offset = 0;
            for (int r = 0; r < ranks; ++r) {
                const size_t rnz = base + (static_cast<size_t>(r) < extra);
                if (rnz * plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    offset > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    std::fprintf(stderr, "Result is too large for MPI_Gatherv.\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                counts[r] = static_cast<int>(rnz * plane);
                displs[r] = static_cast<int>(offset);
                offset += rnz * plane;
            }
            gathered.resize(nx * ny * nz);
        }
        MPI_Gatherv(finalGrid.data() + plane, localCount, MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(gathered, "Grid");
    }

    int rc = 0;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, plane, localNz, rank, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        rc = valid ? 0 : 1;
    }
    MPI_Finalize();
    return rc;
}
