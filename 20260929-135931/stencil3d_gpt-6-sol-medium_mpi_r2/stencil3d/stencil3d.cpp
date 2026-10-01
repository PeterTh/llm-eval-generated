#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

using Real = double;

static size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return (z * ny + y) * nx + x;
}

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t nz) {
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                grid[idx3(x, y, z, nx, ny)] = Real(idx3(x, y, z, nx, ny) % 19);
}

static bool validateResult(const std::vector<Real>& grid) {
    Real minVal = grid[0], maxVal = grid[0];
    for (Real value : grid) {
        if (!std::isfinite(value)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
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
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Select a factorization that fits the interior and minimizes exchanged face area.
static int chooseTopology(int worldSize, const int extent[3], int dims[3]) {
    const long long volume = 1LL * extent[0] * extent[1] * extent[2];
    for (int ranks = static_cast<int>(std::min<long long>(worldSize, volume)); ranks >= 1; --ranks) {
        double best = std::numeric_limits<double>::infinity();
        for (int px = 1; px <= extent[0] && px <= ranks; ++px) {
            if (ranks % px) continue;
            const int rest = ranks / px;
            for (int py = 1; py <= extent[1] && py <= rest; ++py) {
                if (rest % py) continue;
                const int pz = rest / py;
                if (pz > extent[2]) continue;
                const double area = double(px - 1) * extent[1] * extent[2]
                                  + double(py - 1) * extent[0] * extent[2]
                                  + double(pz - 1) * extent[0] * extent[1];
                if (area < best) {
                    best = area;
                    dims[0] = px; dims[1] = py; dims[2] = pz;
                }
            }
        }
        if (best < std::numeric_limits<double>::infinity()) return ranks;
    }
    return 1;
}

static int blockSize(int n, int parts, int coordinate) {
    return n / parts + (coordinate < n % parts);
}
static int blockStart(int n, int parts, int coordinate) {
    return coordinate * (n / parts) + std::min(coordinate, n % parts) + 1;
}

static MPI_Datatype faceType(const int sizes[3], int axis) {
    // MPI subarray dimensions are ordered z, y, x.
    int subsizes[3] = {sizes[0] - 2, sizes[1] - 2, sizes[2] - 2};
    subsizes[2 - axis] = 1;
    const int starts[3] = {0, 0, 0};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

static void computeBox(const Real* input, Real* output, int sx, int sy,
                       int x0, int x1, int y0, int y1, int z0, int z1) {
    const size_t row = sx, plane = size_t(sx) * sy;
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y) {
            size_t i = (size_t(z) * sy + y) * sx + x0;
            for (int x = x0; x <= x1; ++x, ++i)
                output[i] = (input[i] + input[i - 1] + input[i + 1]
                           + input[i - row] + input[i + row]
                           + input[i - plane] + input[i + plane]) / 7.0;
        }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else { parseStatus = 1; if (worldRank == 0) printf("Unknown option: %s\n", argv[i]); break; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (parseStatus || nx == 0 || ny == 0 || nz == 0 ||
        nx > INT_MAX || ny > INT_MAX || nz > INT_MAX || iterations < 0) {
        if (worldRank == 0) {
            if (!parseStatus) printf("Invalid grid size or iteration count\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (worldRank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    const int extent[3] = {int(nx) - 2, int(ny) - 2, int(nz) - 2};
    const bool hasInterior = extent[0] > 0 && extent[1] > 0 && extent[2] > 0;
    int dims[3] = {1, 1, 1};
    const int activeRanks = hasInterior ? chooseTopology(worldSize, extent, dims) : 1;
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (worldRank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }
    int status = 0;
    if (!hasInterior) {
        // Every point is a fixed boundary point.
        const double start = MPI_Wtime();
        const double elapsed = MPI_Wtime() - start;
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", elapsed * 1000, 0.0);
        if (printResults || validate) {
            std::vector<Real> grid(nx * ny * nz);
            initializeGrid(grid, nx, ny, nz);
            if (printResults) print_results(grid, "Grid");
            if (validate) {
                printf("Validating result...\n");
                status = !validateResult(grid);
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        MPI_Comm_free(&active);
        MPI_Finalize();
        return status;
    }

    MPI_Comm cart;
    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(active, 3, dims, periods, 0, &cart);
    int rank, coords[3];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, coords);
    const int lx = blockSize(extent[0], dims[0], coords[0]);
    const int ly = blockSize(extent[1], dims[1], coords[1]);
    const int lz = blockSize(extent[2], dims[2], coords[2]);
    const int gx = blockStart(extent[0], dims[0], coords[0]);
    const int gy = blockStart(extent[1], dims[1], coords[1]);
    const int gz = blockStart(extent[2], dims[2], coords[2]);
    const int sx = lx + 2, sy = ly + 2, sz = lz + 2;
    const size_t localVolume = size_t(sx) * sy * sz;
    std::vector<Real> grid1(localVolume, 0.0), grid2(localVolume, 0.0);
    for (int z = 0; z < sz; ++z)
        for (int y = 0; y < sy; ++y)
            for (int x = 0; x < sx; ++x) {
                const size_t globalX = size_t(gx + x - 1);
                const size_t globalY = size_t(gy + y - 1);
                const size_t globalZ = size_t(gz + z - 1);
                const bool owned = x > 0 && x <= lx && y > 0 && y <= ly && z > 0 && z <= lz;
                const bool boundary = globalX == 0 || globalX == nx - 1 ||
                                      globalY == 0 || globalY == ny - 1 ||
                                      globalZ == 0 || globalZ == nz - 1;
                if (owned || boundary) {
                    const Real value = Real(idx3(globalX, globalY, globalZ, nx, ny) % 19);
                    const size_t i = idx3(x, y, z, sx, sy);
                    grid1[i] = value;
                    if (boundary) grid2[i] = value;
                }
            }
    const int sizes[3] = {sz, sy, sx};
    MPI_Datatype faces[3];
    int neighbors[3][2];
    for (int axis = 0; axis < 3; ++axis) {
        faces[axis] = faceType(sizes, axis);
        MPI_Cart_shift(cart, axis, 1, &neighbors[axis][0], &neighbors[axis][1]);
    }
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0 ? grid1 : grid2).data();
        Real* output = (iter % 2 == 0 ? grid2 : grid1).data();
        MPI_Request requests[12];
        int count = 0;
        const int len[3] = {lx, ly, lz};
        for (int axis = 0; axis < 3; ++axis) {
            int pos[3] = {1, 1, 1};
            pos[axis] = 0;
            MPI_Irecv(input + idx3(pos[0], pos[1], pos[2], sx, sy), 1, faces[axis],
                      neighbors[axis][0], axis * 2 + 1, cart, &requests[count++]);
            pos[axis] = len[axis] + 1;
            MPI_Irecv(input + idx3(pos[0], pos[1], pos[2], sx, sy), 1, faces[axis],
                      neighbors[axis][1], axis * 2, cart, &requests[count++]);
            pos[axis] = 1;
            MPI_Isend(input + idx3(pos[0], pos[1], pos[2], sx, sy), 1, faces[axis],
                      neighbors[axis][0], axis * 2, cart, &requests[count++]);
            pos[axis] = len[axis];
            MPI_Isend(input + idx3(pos[0], pos[1], pos[2], sx, sy), 1, faces[axis],
                      neighbors[axis][1], axis * 2 + 1, cart, &requests[count++]);
        }
        computeBox(input, output, sx, sy, 2, lx - 1, 2, ly - 1, 2, lz - 1);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Six disjoint outer slabs cover all cells not computed above.
        computeBox(input, output, sx, sy, 1, 1, 1, ly, 1, lz);
        if (lx > 1) computeBox(input, output, sx, sy, lx, lx, 1, ly, 1, lz);
        computeBox(input, output, sx, sy, 2, lx - 1, 1, 1, 1, lz);
        if (ly > 1) computeBox(input, output, sx, sy, 2, lx - 1, ly, ly, 1, lz);
        computeBox(input, output, sx, sy, 2, lx - 1, 2, ly - 1, 1, 1);
        if (lz > 1) computeBox(input, output, sx, sy, 2, lx - 1, 2, ly - 1, lz, lz);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        const double updates = double(extent[0]) * extent[1] * extent[2] * iterations;
        printf("Computation time: %.3f ms\n", elapsed * 1000);
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0.0);
    }
    if (printResults || validate) {
        const Real* finalGrid = (iterations % 2 == 0 ? grid1 : grid2).data();
        std::vector<Real> packed(size_t(lx) * ly * lz);
        size_t next = 0;
        for (int z = 1; z <= lz; ++z)
            for (int y = 1; y <= ly; ++y)
                for (int x = 1; x <= lx; ++x)
                    packed[next++] = finalGrid[idx3(x, y, z, sx, sy)];
        const int localCount = static_cast<int>(packed.size());
        std::vector<int> counts, displacements;
        if (rank == 0) { counts.resize(activeRanks); displacements.resize(activeRanks); }
        MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr,
                   1, MPI_INT, 0, cart);
        std::vector<Real> received;
        if (rank == 0) {
            int offset = 0;
            for (int r = 0; r < activeRanks; ++r) {
                displacements[r] = offset;
                offset += counts[r];
            }
            received.resize(offset);
        }
        MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? received.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, cart);
        if (rank == 0) {
            std::vector<Real> grid(nx * ny * nz);
            initializeGrid(grid, nx, ny, nz);
            for (int r = 0; r < activeRanks; ++r) {
                int c[3];
                MPI_Cart_coords(cart, r, 3, c);
                const int bx = blockSize(extent[0], dims[0], c[0]);
                const int by = blockSize(extent[1], dims[1], c[1]);
                const int bz = blockSize(extent[2], dims[2], c[2]);
                const int ox = blockStart(extent[0], dims[0], c[0]);
                const int oy = blockStart(extent[1], dims[1], c[1]);
                const int oz = blockStart(extent[2], dims[2], c[2]);
                size_t p = displacements[r];
                for (int z = 0; z < bz; ++z)
                    for (int y = 0; y < by; ++y)
                        for (int x = 0; x < bx; ++x)
                            grid[idx3(ox + x, oy + y, oz + z, nx, ny)] = received[p++];
            }
            if (printResults) print_results(grid, "Grid");
            if (validate) {
                printf("Validating result...\n");
                status = !validateResult(grid);
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, cart);
    for (MPI_Datatype& face : faces) MPI_Type_free(&face);
    MPI_Comm_free(&cart);
    MPI_Comm_free(&active);
    MPI_Finalize();
    return status;
}
