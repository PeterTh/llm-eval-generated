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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

// Planes 1..localNz are owned by this rank; planes 0 and localNz+1 are halos.
void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t firstZ) {
    const size_t plane = nx * ny;
    for (size_t z = 0; z < localNz; ++z)
        for (size_t i = 0; i < plane; ++i)
            grid[(z + 1) * plane + i] = ((firstZ + z) * plane + i) % 19;
}

inline void stencilPlane(const std::vector<Real>& in, std::vector<Real>& out,
                         size_t localZ, size_t globalZ, size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const size_t base = localZ * plane;
    if (globalZ == 0 || globalZ + 1 == nz) {
        std::copy_n(in.data() + base, plane, out.data() + base);
        return;
    }
    // The two y-boundary rows are immutable, and copying whole rows is faster than branching per cell.
    std::copy_n(in.data() + base, nx, out.data() + base);
    std::copy_n(in.data() + base + (ny - 1) * nx, nx, out.data() + base + (ny - 1) * nx);
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        out[row] = in[row];
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t p = row + x;
            out[p] = (in[p] + in[p - 1] + in[p + 1] + in[p - nx] + in[p + nx]
                      + in[p - plane] + in[p + plane]) / 7.0;
        }
        out[row + nx - 1] = in[row + nx - 1];
    }
}

bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, MPI_Comm comm) {
    const size_t plane = nx * ny;
    int localFinite = 1;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t i = 0; i < plane; ++i) {
            const Real v = grid[z * plane + i];
            localFinite &= std::isfinite(v);
            localMin = std::min(localMin, v);
            localMax = std::max(localMax, v);
        }
    int finite;
    Real minVal, maxVal;
    MPI_Allreduce(&localFinite, &finite, 1, MPI_INT, MPI_LAND, comm);
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        if (!finite) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
        if (maxVal > 1e6 || minVal < -1e6) finite = 0;
    }
    MPI_Bcast(&finite, 1, MPI_INT, 0, comm);
    return finite != 0;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 128)\n  -y <num>  Grid Y (default X)\n"
                "  -z <num>  Grid Z (default X)\n  -i <num>  Iterations (default 10)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
    if (worldRank == 0 && (help || badArgs || nx < 3 || ny < 3 || nz < 3 || iterations < 0)) {
        if (badArgs) std::printf("Invalid option or grid dimensions\n");
        printUsage(argv[0]);
    }
    const int localError = help || badArgs || nx < 3 || ny < 3 || nz < 3 || iterations < 0;
    int error;
    MPI_Allreduce(&localError, &error, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (error) { MPI_Finalize(); return help && !badArgs ? 0 : 1; }

    // At most nz ranks can own a plane.  Extra ranks remain synchronized but do no work.
    const int active = worldRank < static_cast<int>(nz);
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (!active) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    const size_t base = nz / ranks, rem = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < rem);
    const size_t firstZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, firstZ);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI ranks: %d)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n",
                    worldSize, nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        auto& in = (iter & 1) ? grid2 : grid1;
        auto& out = (iter & 1) ? grid1 : grid2;
        MPI_Request req[4]; int nreq = 0;
        if (rank > 0) {
            MPI_Irecv(in.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &req[nreq++]);
            MPI_Isend(in.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &req[nreq++]);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(in.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &req[nreq++]);
            MPI_Isend(in.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &req[nreq++]);
        }
        for (size_t z = 2; z < localNz; ++z) stencilPlane(in, out, z, firstZ + z - 1, nx, ny, nz);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        stencilPlane(in, out, 1, firstZ, nx, ny, nz);
        if (localNz > 1) stencilPlane(in, out, localNz, firstZ + localNz - 1, nx, ny, nz);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    auto& finalGrid = (iterations & 1) ? grid2 : grid1;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", maxElapsed * 1000.0,
                    static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations / maxElapsed / 1e6);
    }
    if (printResults) {
        std::vector<int> counts(ranks), displs(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t n = base + (static_cast<size_t>(r) < rem);
            counts[r] = static_cast<int>(n * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem)) * plane);
        }
        std::vector<Real> global(rank == 0 ? nx * ny * nz : 0);
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    global.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(global, "Grid");
    }
    const bool valid = !validate || validateResult(finalGrid, nx, ny, localNz, comm);
    if (validate && rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
