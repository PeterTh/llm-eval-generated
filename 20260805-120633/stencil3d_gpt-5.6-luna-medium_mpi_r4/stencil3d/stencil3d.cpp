#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Block {
    size_t n = 0;
    size_t begin = 0;
    size_t extent = 0;
};

Block split(size_t global, int parts, int coordinate) {
    const size_t base = global / static_cast<size_t>(parts);
    const size_t extra = global % static_cast<size_t>(parts);
    const size_t extent = base + (static_cast<size_t>(coordinate) < extra ? 1 : 0);
    const size_t begin = static_cast<size_t>(coordinate) * base +
                         std::min(static_cast<size_t>(coordinate), extra);
    return {extent, begin, extent};
}

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t nz,
                    size_t x0, size_t y0, size_t z0, size_t globalX, size_t globalY) {
    const size_t pitch = nx + 2;
    const size_t plane = pitch * (ny + 2);
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 1; y <= ny; ++y) {
            for (size_t x = 1; x <= nx; ++x) {
                const size_t global = idx3(x0 + x - 1, y0 + y - 1, z0 + z - 1,
                                           globalX, globalY);
                grid[z * plane + y * pitch + x] = static_cast<Real>(global % 19);
            }
        }
    }
}

void makeFaceTypes(size_t nx, size_t ny, size_t nz, MPI_Datatype& xInterior,
                   MPI_Datatype& xHalo, MPI_Datatype& yInterior, MPI_Datatype& yHalo,
                   MPI_Datatype& zBottomInterior, MPI_Datatype& zTopInterior,
                   MPI_Datatype& zBottomHalo, MPI_Datatype& zTopHalo) {
    const int sizes[3] = {static_cast<int>(nz + 2), static_cast<int>(ny + 2), static_cast<int>(nx + 2)};
    const int xSub[3] = {static_cast<int>(nz), static_cast<int>(ny), 1};
    const int ySub[3] = {static_cast<int>(nz), 1, static_cast<int>(nx)};
    const int zSub[3] = {1, static_cast<int>(ny), static_cast<int>(nx)};
    int start[3] = {1, 1, 1};
    MPI_Type_create_subarray(3, sizes, xSub, start, MPI_ORDER_C, MPI_DOUBLE, &xInterior);
    start[2] = 0;
    MPI_Type_create_subarray(3, sizes, xSub, start, MPI_ORDER_C, MPI_DOUBLE, &xHalo);
    start[1] = 1; start[2] = 1;
    MPI_Type_create_subarray(3, sizes, ySub, start, MPI_ORDER_C, MPI_DOUBLE, &yInterior);
    start[1] = 0;
    MPI_Type_create_subarray(3, sizes, ySub, start, MPI_ORDER_C, MPI_DOUBLE, &yHalo);
    start[0] = 1; start[1] = 1; start[2] = 1;
    MPI_Type_create_subarray(3, sizes, zSub, start, MPI_ORDER_C, MPI_DOUBLE, &zBottomInterior);
    start[0] = static_cast<int>(nz);
    MPI_Type_create_subarray(3, sizes, zSub, start, MPI_ORDER_C, MPI_DOUBLE, &zTopInterior);
    start[0] = 0;
    start[0] = 0;
    MPI_Type_create_subarray(3, sizes, zSub, start, MPI_ORDER_C, MPI_DOUBLE, &zBottomHalo);
    start[0] = static_cast<int>(nz + 1);
    MPI_Type_create_subarray(3, sizes, zSub, start, MPI_ORDER_C, MPI_DOUBLE, &zTopHalo);
    MPI_Type_commit(&xInterior); MPI_Type_commit(&xHalo);
    MPI_Type_commit(&yInterior); MPI_Type_commit(&yHalo);
    MPI_Type_commit(&zBottomInterior); MPI_Type_commit(&zTopInterior);
    MPI_Type_commit(&zBottomHalo); MPI_Type_commit(&zTopHalo);
}

void exchangeHalos(std::vector<Real>& grid, size_t nx, size_t ny,
                   const int neighbors[6], MPI_Datatype xInterior, MPI_Datatype xHalo,
                   MPI_Datatype yInterior, MPI_Datatype yHalo,
                   MPI_Datatype zBottomInterior, MPI_Datatype zTopInterior,
                   MPI_Datatype zBottomHalo, MPI_Datatype zTopHalo) {
    const size_t pitch = nx + 2;
    MPI_Request requests[12];
    int requestCount = 0;
    // Receive tags are the opposite sender's direction.  MPI_PROC_NULL makes
    // physical global boundaries cost-free while retaining the same schedule.
    MPI_Irecv(grid.data(), 1, xHalo, neighbors[0], 11, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Irecv(grid.data() + nx + 1, 1, xHalo, neighbors[1], 10, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Irecv(grid.data(), 1, yHalo, neighbors[2], 21, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Irecv(grid.data() + (ny + 1) * pitch, 1, yHalo, neighbors[3], 20, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Irecv(grid.data(), 1, zBottomHalo, neighbors[4], 31, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Irecv(grid.data(), 1, zTopHalo, neighbors[5], 30, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Isend(grid.data(), 1, xInterior, neighbors[0], 10, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Isend(grid.data() + nx - 1, 1, xInterior, neighbors[1], 11, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Isend(grid.data(), 1, yInterior, neighbors[2], 20, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Isend(grid.data() + (ny - 1) * pitch, 1, yInterior, neighbors[3], 21, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Isend(grid.data(), 1, zBottomInterior, neighbors[4], 30, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Isend(grid.data(), 1, zTopInterior, neighbors[5], 31, MPI_COMM_WORLD, &requests[requestCount++]);
    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
}

void stencilIteration(const std::vector<Real>& in, std::vector<Real>& out,
                      size_t nx, size_t ny, size_t nz, size_t x0, size_t y0, size_t z0,
                      size_t globalX, size_t globalY, size_t globalZ) {
    const size_t pitch = nx + 2;
    const size_t plane = pitch * (ny + 2);
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 1; y <= ny; ++y) {
            for (size_t x = 1; x <= nx; ++x) {
                const size_t outIndex = z * plane + y * pitch + x;
                const size_t gx = x0 + x - 1, gy = y0 + y - 1, gz = z0 + z - 1;
                if (gx == 0 || gx + 1 == globalX || gy == 0 || gy + 1 == globalY ||
                    gz == 0 || gz + 1 == globalZ) {
                    out[outIndex] = in[outIndex];
                } else {
                    out[outIndex] = (in[outIndex] + in[outIndex - 1] + in[outIndex + 1] +
                                     in[outIndex - pitch] + in[outIndex + pitch] +
                                     in[outIndex - plane] + in[outIndex + plane]) / 7.0;
                }
            }
        }
    }
}

std::vector<Real> packOwned(const std::vector<Real>& grid, size_t nx, size_t ny, size_t nz) {
    std::vector<Real> packed(nx * ny * nz);
    const size_t pitch = nx + 2, plane = pitch * (ny + 2);
    size_t p = 0;
    for (size_t z = 1; z <= nz; ++z)
        for (size_t y = 1; y <= ny; ++y)
            for (size_t x = 1; x <= nx; ++x)
                packed[p++] = grid[z * plane + y * pitch + x];
    return packed;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num>  Grid size in X (default: 128)\n  -y <num>  Grid size in Y (default: X)\n");
    printf("  -z <num>  Grid size in Z (default: X)\n  -i <num>  Iterations (default: 10)\n");
    printf("  -v        Enable validation\n  -r        Print results\n  -h        Show this help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        if (rank == 0) printf("Grid dimensions must be at least 3 and iterations non-negative.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(world, 3, dims);
    if (static_cast<size_t>(dims[0]) > nx || static_cast<size_t>(dims[1]) > ny || static_cast<size_t>(dims[2]) > nz) {
        if (rank == 0) printf("Too many MPI ranks for the requested grid.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int periods[3] = {0, 0, 0}; MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int coords[3]; MPI_Cart_coords(cart, rank, 3, coords);
    Block bx = split(nx, dims[0], coords[0]), by = split(ny, dims[1], coords[1]), bz = split(nz, dims[2], coords[2]);
    std::vector<Real> grid1((bx.n + 2) * (by.n + 2) * (bz.n + 2));
    std::vector<Real> grid2(grid1.size());
    initializeGrid(grid1, bx.n, by.n, bz.n, bx.begin, by.begin, bz.begin, nx, ny);
    int neighbors[6];
    MPI_Cart_shift(cart, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(cart, 1, 1, &neighbors[2], &neighbors[3]);
    MPI_Cart_shift(cart, 2, 1, &neighbors[4], &neighbors[5]);
    // The Cartesian communicator is rank-compatible with MPI_COMM_WORLD (reorder=0).
    MPI_Datatype xInterior, xHalo, yInterior, yHalo, zBottomInterior, zTopInterior, zBottomHalo, zTopHalo;
    makeFaceTypes(bx.n, by.n, bz.n, xInterior, xHalo, yInterior, yHalo, zBottomInterior, zTopInterior, zBottomHalo, zTopHalo);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\nInitializing grid...\nRunning stencil computation...\n", world);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;
        exchangeHalos(in, bx.n, by.n, neighbors, xInterior, xHalo, yInterior, yHalo, zBottomInterior, zTopInterior, zBottomHalo, zTopHalo);
        stencilIteration(in, out, bx.n, by.n, bz.n, bx.begin, by.begin, bz.begin, nx, ny, nz);
    }
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
    }
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;
    int localCount = static_cast<int>(bx.n * by.n * bz.n);
    std::vector<int> counts, displacements;
    std::vector<Real> gathered;
    std::vector<int> metadata;
    if (printResults && rank == 0) { counts.resize(world); displacements.resize(world); metadata.resize(world * 6); }
    if (printResults) {
        std::vector<Real> packed = packOwned(finalLocal, bx.n, by.n, bz.n);
        MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int mine[6] = {static_cast<int>(bx.begin), static_cast<int>(by.begin), static_cast<int>(bz.begin),
                       static_cast<int>(bx.n), static_cast<int>(by.n), static_cast<int>(bz.n)};
        MPI_Gather(mine, 6, MPI_INT, rank == 0 ? metadata.data() : nullptr, 6, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            int total = 0; for (int r = 0; r < world; ++r) { displacements[r] = total; total += counts[r]; }
            gathered.resize(static_cast<size_t>(total));
        }
        MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<Real> global(nx * ny * nz);
            for (int r = 0; r < world; ++r) {
                const int x0 = metadata[6*r], y0 = metadata[6*r+1], z0 = metadata[6*r+2];
                const int lx = metadata[6*r+3], ly = metadata[6*r+4], lz = metadata[6*r+5];
                size_t p = static_cast<size_t>(displacements[r]);
                for (int z = 0; z < lz; ++z) for (int y = 0; y < ly; ++y) for (int x = 0; x < lx; ++x)
                    global[idx3(x0+x, y0+y, z0+z, nx, ny)] = gathered[p++];
            }
            print_results(global, "Grid");
        }
    }
    if (validate) {
        bool localValid = true; Real localMin = std::numeric_limits<Real>::max(), localMax = -localMin;
        const size_t pch = bx.n + 2, pln = pch * (by.n + 2);
        for (size_t z = 1; z <= bz.n; ++z) for (size_t y = 1; y <= by.n; ++y) for (size_t x = 1; x <= bx.n; ++x) {
            Real v = finalLocal[z*pln + y*pch + x]; localValid &= std::isfinite(v); localMin = std::min(localMin, v); localMax = std::max(localMax, v);
        }
        int validInt = localValid ? 1 : 0, allValid = 0; Real minVal = 0, maxVal = 0;
        MPI_Reduce(&validInt, &allValid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) { printf("Validating result...\nValue range: [%.6f, %.6f]\n", minVal, maxVal); printf("Validation: %s\n", allValid && minVal >= -1e6 && maxVal <= 1e6 ? "PASSED" : "FAILED"); }
        int validationStatus = (rank == 0 && allValid && minVal >= -1e6 && maxVal <= 1e6) ? 0 : 1;
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (validationStatus != 0) { MPI_Finalize(); return 1; }
    }
    MPI_Type_free(&xInterior); MPI_Type_free(&xHalo); MPI_Type_free(&yInterior); MPI_Type_free(&yHalo);
    MPI_Type_free(&zBottomInterior); MPI_Type_free(&zTopInterior);
    MPI_Type_free(&zBottomHalo); MPI_Type_free(&zTopHalo);
    MPI_Comm_free(&cart); MPI_Finalize(); return 0;
}
