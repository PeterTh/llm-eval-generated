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

inline constexpr size_t idx3(size_t x, size_t y, size_t z,
                             size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

// Initialize owned planes exactly as the serial program did.  Local plane zero
// and localNz+1 are communication halos.
static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                           size_t localNz, size_t zBegin) {
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t globalBase = (zBegin + lz - 1) * plane;
        Real* const p = grid.data() + lz * plane;
        for (size_t i = 0; i < plane; ++i)
            p[i] = static_cast<Real>((globalBase + i) % 19);
    }
}

static inline void computePlane(const Real* __restrict input,
                                Real* __restrict output, size_t nx, size_t ny,
                                size_t lz, size_t globalZ, size_t nz) {
    const size_t plane = nx * ny;
    const Real* const in = input + lz * plane;
    Real* const out = output + lz * plane;

    if (globalZ == 0 || globalZ + 1 == nz) {
        std::copy_n(in, plane, out);
        return;
    }

    // Fixed X/Y boundary cells.
    std::copy_n(in, nx, out);
    std::copy_n(in + (ny - 1) * nx, nx, out + (ny - 1) * nx);
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = y * nx;
        out[row] = in[row];
        out[row + nx - 1] = in[row + nx - 1];
        #pragma omp simd
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t i = row + x;
            out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] +
                      in[i + nx] + in[i - plane] + in[i + plane]) / 7.0;
        }
    }
}

static void stencilIteration(const std::vector<Real>& input,
                             std::vector<Real>& output, size_t nx, size_t ny,
                             size_t localNz, size_t zBegin, size_t nz,
                             int rank, int ranks, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Request req[4];

    MPI_Irecv(const_cast<Real*>(input.data()), static_cast<int>(plane), MPI_DOUBLE,
              lower, 11, comm, &req[0]);
    MPI_Irecv(const_cast<Real*>(input.data()) + (localNz + 1) * plane,
              static_cast<int>(plane), MPI_DOUBLE, upper, 10, comm, &req[1]);
    MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
              lower, 10, comm, &req[2]);
    MPI_Isend(input.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
              upper, 11, comm, &req[3]);

    // Planes not adjacent to a rank boundary do not depend on halo traffic.
    for (size_t lz = 2; lz < localNz; ++lz)
        computePlane(input.data(), output.data(), nx, ny, lz,
                     zBegin + lz - 1, nz);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    computePlane(input.data(), output.data(), nx, ny, 1, zBegin, nz);
    if (localNz > 1)
        computePlane(input.data(), output.data(), nx, ny, localNz,
                     zBegin + localNz - 1, nz);
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  Grid size in X (default: 128)\n");
    std::printf("  -y <num>  Grid size in Y (default: same as X)\n");
    std::printf("  -z <num>  Grid size in Z (default: same as X)\n");
    std::printf("  -i <num>  Number of iterations (default: 10)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else { if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]); bad = true; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        nx > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ny > static_cast<size_t>(std::numeric_limits<int>::max()) / nx) bad = true;
    if (help || bad) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }

    // Extra ranks are excluded when there are fewer Z planes than processes.
    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const size_t base = nz / activeRanks, remainder = nz % activeRanks;
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t zBegin = static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\nMPI processes: %d\n",
                    iterations, validate ? "enabled" : "disabled", activeRanks);
        std::printf("Initializing grid...\n");
    }
    std::vector<Real> grid1((localNz + 2) * plane);
    std::vector<Real> grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, zBegin);

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) stencilIteration(grid1, grid2, nx, ny, localNz, zBegin, nz, rank, activeRanks, comm);
        else stencilIteration(grid2, grid1, nx, ny, localNz, zBegin, nz, rank, activeRanks, comm);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const long ms = static_cast<long>(seconds * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Computation time: %ld ms\n", ms);
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
    }

    const std::vector<Real>& finalGrid = (iterations & 1) ? grid2 : grid1;
    if (printResults) {
        std::vector<Real> global;
        if (rank == 0) {
            global.resize(nx * ny * nz);
            std::copy_n(finalGrid.data() + plane, localNz * plane, global.data());
            // One plane per message avoids MPI-3's signed-int total-count limit.
            for (int r = 1; r < activeRanks; ++r) {
                const size_t rnz = base + (static_cast<size_t>(r) < remainder);
                const size_t rz = static_cast<size_t>(r) * base +
                                  std::min(static_cast<size_t>(r), remainder);
                for (size_t p = 0; p < rnz; ++p)
                    MPI_Recv(global.data() + (rz + p) * plane, static_cast<int>(plane),
                             MPI_DOUBLE, r, 20, comm, MPI_STATUS_IGNORE);
            }
            print_results(global, "Grid");
        } else {
            for (size_t p = 0; p < localNz; ++p)
                MPI_Send(finalGrid.data() + (p + 1) * plane, static_cast<int>(plane),
                         MPI_DOUBLE, 0, 20, comm);
        }
    }

    int localFinite = 1;
    Real localMin = finalGrid[plane], localMax = finalGrid[plane];
    for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
        localFinite &= std::isfinite(finalGrid[i]);
        localMin = std::min(localMin, finalGrid[i]);
        localMax = std::max(localMax, finalGrid[i]);
    }
    int finite = 0;
    Real minVal = 0, maxVal = 0;
    MPI_Reduce(&localFinite, &finite, 1, MPI_INT, MPI_LAND, 0, comm);
    MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int status = 0;
    if (rank == 0 && validate) {
        std::printf("Validating result...\nValue range: [%.6f, %.6f]\n", minVal, maxVal);
        status = (!finite || maxVal > 1e6 || minVal < -1e6) ? 1 : 0;
        std::printf("Validation: %s\n", status ? "FAILED" : "PASSED");
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
