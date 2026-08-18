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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Range {
    int begin;
    int size;
};

Range splitRange(const int n, const int part, const int parts) {
    const int remainder = n % parts;
    return {part * (n / parts) + std::min(part, remainder),
            n / parts + (part < remainder ? 1 : 0)};
}

void initializeGrid(std::vector<Real>& grid, const int nx, const int ny, const int nz,
                    const int x0, const int y0, const int z0,
                    const int globalNx, const int globalNy) {
    const int sx = nx + 2;
    const int sy = ny + 2;
    for (int z = 1; z <= nz; ++z) {
        for (int y = 1; y <= ny; ++y) {
            for (int x = 1; x <= nx; ++x) {
                const size_t globalIndex = idx3(static_cast<size_t>(x0 + x - 1),
                                                static_cast<size_t>(y0 + y - 1),
                                                static_cast<size_t>(z0 + z - 1),
                                                static_cast<size_t>(globalNx),
                                                static_cast<size_t>(globalNy));
                grid[(static_cast<size_t>(z) * sy + y) * sx + x] =
                    static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& input, const int nx, const int ny, const int nz,
                   const int west, const int east, const int south, const int north,
                   const int down, const int up, MPI_Comm comm) {
    const int sx = nx + 2;
    const int sy = ny + 2;
    const size_t plane = static_cast<size_t>(sx) * sy;
    std::vector<Real> sendX(static_cast<size_t>(nz) * ny), recvX(sendX.size());
    std::vector<Real> sendYS(static_cast<size_t>(nz) * nx), sendYN(sendYS.size());
    std::vector<Real> recvYS(sendYS.size()), recvYN(sendYS.size());
    std::vector<Real> sendZD(static_cast<size_t>(ny) * nx), sendZU(sendZD.size());
    std::vector<Real> recvZD(sendZD.size()), recvZU(sendZD.size());
    for (int z = 1; z <= nz; ++z) for (int y = 1; y <= ny; ++y) {
        sendX[static_cast<size_t>(z-1) * ny + y-1] = input[static_cast<size_t>(z) * plane + y * sx + 1];
        for (int x = 1; x <= nx; ++x) sendYS[static_cast<size_t>(z-1) * nx + x-1] = input[static_cast<size_t>(z) * plane + sx + x];
    }
    std::vector<Real> sendXEast(sendX.size()), recvXEast(sendX.size());
    for (int z = 1; z <= nz; ++z) for (int y = 1; y <= ny; ++y)
        sendXEast[static_cast<size_t>(z-1) * ny + y-1] = input[static_cast<size_t>(z) * plane + y * sx + nx];
    MPI_Sendrecv(sendX.data(), static_cast<int>(sendX.size()), MPI_DOUBLE, west, 10,
                 recvX.data(), static_cast<int>(recvX.size()), MPI_DOUBLE, east, 10, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendXEast.data(), static_cast<int>(sendXEast.size()), MPI_DOUBLE, east, 11,
                 recvXEast.data(), static_cast<int>(recvXEast.size()), MPI_DOUBLE, west, 11, comm, MPI_STATUS_IGNORE);
    for (int z = 1; z <= nz; ++z) for (int y = 1; y <= ny; ++y) {
        input[static_cast<size_t>(z) * plane + y * sx] = recvXEast[static_cast<size_t>(z-1) * ny + y-1];
        input[static_cast<size_t>(z) * plane + y * sx + nx + 1] = recvX[static_cast<size_t>(z-1) * ny + y-1];
    }
    for (int z = 1; z <= nz; ++z) for (int x = 1; x <= nx; ++x)
        sendYN[static_cast<size_t>(z-1) * nx + x-1] = input[static_cast<size_t>(z) * plane + ny * sx + x];
    MPI_Sendrecv(sendYS.data(), static_cast<int>(sendYS.size()), MPI_DOUBLE, south, 20,
                 recvYN.data(), static_cast<int>(recvYN.size()), MPI_DOUBLE, north, 20, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendYN.data(), static_cast<int>(sendYN.size()), MPI_DOUBLE, north, 21,
                 recvYS.data(), static_cast<int>(recvYS.size()), MPI_DOUBLE, south, 21, comm, MPI_STATUS_IGNORE);
    for (int z = 1; z <= nz; ++z) for (int x = 1; x <= nx; ++x) {
        input[static_cast<size_t>(z) * plane + x] = recvYS[static_cast<size_t>(z-1) * nx + x-1];
        input[static_cast<size_t>(z) * plane + (ny + 1) * sx + x] = recvYN[static_cast<size_t>(z-1) * nx + x-1];
    }
    for (int y = 1; y <= ny; ++y) for (int x = 1; x <= nx; ++x) {
        sendZD[static_cast<size_t>(y-1) * nx + x-1] = input[plane + y * sx + x];
    }
    MPI_Sendrecv(sendZD.data(), static_cast<int>(sendZD.size()), MPI_DOUBLE, down, 30,
                 recvZU.data(), static_cast<int>(recvZU.size()), MPI_DOUBLE, up, 30, comm, MPI_STATUS_IGNORE);
    for (int y = 1; y <= ny; ++y) for (int x = 1; x <= nx; ++x)
        sendZU[static_cast<size_t>(y-1) * nx + x-1] = input[static_cast<size_t>(nz) * plane + y * sx + x];
    MPI_Sendrecv(sendZU.data(), static_cast<int>(sendZU.size()), MPI_DOUBLE, up, 31,
                 recvZD.data(), static_cast<int>(recvZD.size()), MPI_DOUBLE, down, 31, comm, MPI_STATUS_IGNORE);
    for (int y = 1; y <= ny; ++y) for (int x = 1; x <= nx; ++x) {
        input[y * sx + x] = recvZD[static_cast<size_t>(y-1) * nx + x-1];
        input[static_cast<size_t>(nz + 1) * plane + y * sx + x] = recvZU[static_cast<size_t>(y-1) * nx + x-1];
    }
}

void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      const int nx, const int ny, const int nz,
                      const int x0, const int y0, const int z0,
                      const int globalNx, const int globalNy, const int globalNz) {
    const int sx = nx + 2;
    const int sy = ny + 2;
    const size_t plane = static_cast<size_t>(sx) * sy;
    for (int z = 1; z <= nz; ++z) {
        for (int y = 1; y <= ny; ++y) {
            for (int x = 1; x <= nx; ++x) {
                const size_t p = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * sx + x;
                const bool boundary = x0 + x == 1 || x0 + x == globalNx ||
                                       y0 + y == 1 || y0 + y == globalNy ||
                                       z0 + z == 1 || z0 + z == globalNz;
                if (boundary) {
                    output[p] = input[p];
                } else {
                    output[p] = (input[p] + input[p - 1] + input[p + 1] +
                                 input[p - sx] + input[p + sx] +
                                 input[p - plane] + input[p + plane]) / 7.0;
                }
            }
        }
    }
}

bool validateLocal(const std::vector<Real>& grid, const int nx, const int ny, const int nz,
                   Real& minValue, Real& maxValue) {
    minValue = std::numeric_limits<Real>::infinity();
    maxValue = -std::numeric_limits<Real>::infinity();
    bool valid = true;
    const int sx = nx + 2;
    const int sy = ny + 2;
    for (int z = 1; z <= nz; ++z) {
        for (int y = 1; y <= ny; ++y) {
            for (int x = 1; x <= nx; ++x) {
                const Real value = grid[(static_cast<size_t>(z) * sy + y) * sx + x];
                valid = valid && std::isfinite(value);
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
        }
    }
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false, parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parseError = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 || parseError) {
        if (rank == 0) { printf("Invalid arguments.\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(worldSize, 3, dims);
    if (dims[0] > nx || dims[1] > ny || dims[2] > nz) {
        if (rank == 0) printf("Too many MPI ranks for the requested grid.\n");
        MPI_Finalize(); return 1;
    }
    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    int west, east, south, north, down, up;
    MPI_Cart_shift(cart, 0, 1, &west, &east);
    MPI_Cart_shift(cart, 1, 1, &south, &north);
    MPI_Cart_shift(cart, 2, 1, &down, &up);
    const Range rx = splitRange(nx, coords[0], dims[0]);
    const Range ry = splitRange(ny, coords[1], dims[1]);
    const Range rz = splitRange(nz, coords[2], dims[2]);
    const int lx = rx.size, ly = ry.size, lz = rz.size;
    const int sx = lx + 2, sy = ly + 2;
    const size_t localSize = static_cast<size_t>(lx + 2) * (ly + 2) * (lz + 2);
    std::vector<Real> grid1(localSize), grid2(localSize);
    initializeGrid(grid1, lx, ly, lz, rx.begin, ry.begin, rz.begin, nx, ny);

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI processes: %d (decomposition %d x %d x %d)\n", worldSize, dims[0], dims[1], dims[2]);
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;
        exchangeHalos(input, lx, ly, lz, west, east, south, north, down, up, cart);
        stencilIteration(input, output, lx, ly, lz, rx.begin, ry.begin, rz.begin, nx, ny, nz);
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        const double cellUpdates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;
    if (validate) {
        Real localMin, localMax, globalMin, globalMax;
        const bool localValid = validateLocal(finalLocal, lx, ly, lz, localMin, localMax);
        int validInt = localValid ? 1 : 0, globalValid = 0;
        MPI_Reduce(&validInt, &globalValid, 1, MPI_INT, MPI_MIN, 0, cart);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalMin, globalMax);
            const bool valid = globalValid && globalMax <= 1e6 && globalMin >= -1e6;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) { MPI_Comm_free(&cart); MPI_Finalize(); return 1; }
        }
    }

    if (printResults) {
        const int localCount = lx * ly * lz;
        std::vector<Real> packed(static_cast<size_t>(localCount));
        size_t p = 0;
        for (int z = 1; z <= lz; ++z) for (int y = 1; y <= ly; ++y)
            for (int x = 1; x <= lx; ++x) packed[p++] = finalLocal[(static_cast<size_t>(z) * sy + y) * sx + x];
        std::vector<int> metadata(rank == 0 ? static_cast<size_t>(worldSize) * 6 : 0);
        int localMeta[6] = {rx.begin, ry.begin, rz.begin, lx, ly, lz};
        MPI_Gather(localMeta, 6, MPI_INT, metadata.data(), 6, MPI_INT, 0, cart);
        std::vector<int> counts(rank == 0 ? worldSize : 0), displacements(rank == 0 ? worldSize : 0);
        std::vector<Real> gatheredPacked(rank == 0 ? static_cast<size_t>(nx) * ny * nz : 0);
        std::vector<Real> global(rank == 0 ? static_cast<size_t>(nx) * ny * nz : 0);
        if (rank == 0) {
            int total = 0;
            for (int r = 0; r < worldSize; ++r) {
                counts[r] = metadata[6*r+3] * metadata[6*r+4] * metadata[6*r+5];
                displacements[r] = total; total += counts[r];
            }
        }
        MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, gatheredPacked.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, cart);
        if (rank == 0) {
            for (int r = 0; r < worldSize; ++r) {
                const int xStart = metadata[6*r], yStart = metadata[6*r+1], zStart = metadata[6*r+2];
                const int bx = metadata[6*r+3], by = metadata[6*r+4], bz = metadata[6*r+5];
                const int rankOffset = displacements[r];
                for (int z = 0; z < bz; ++z) for (int y = 0; y < by; ++y) for (int x = 0; x < bx; ++x)
                    global[idx3(xStart+x, yStart+y, zStart+z, nx, ny)] =
                        gatheredPacked[rankOffset + (static_cast<size_t>(z) * by + y) * bx + x];
            }
            print_results(global, "Grid");
        }
    }
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
