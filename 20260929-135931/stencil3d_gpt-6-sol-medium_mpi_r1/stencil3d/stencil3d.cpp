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

// All ranks use the same decomposition, including when there are more ranks than planes.
size_t planeCount(size_t nz, int ranks, int rank) {
    return nz / static_cast<size_t>(ranks) +
           (static_cast<size_t>(rank) < nz % static_cast<size_t>(ranks));
}

size_t firstPlane(size_t nz, int ranks, int rank) {
    return (nz / static_cast<size_t>(ranks)) * static_cast<size_t>(rank) +
           std::min(static_cast<size_t>(rank), nz % static_cast<size_t>(ranks));
}

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                    size_t localNz, size_t globalFirst) {
    const size_t plane = nx * ny;
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalBase = (globalFirst + z) * plane;
        const size_t localBase = (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            grid[localBase + i] = static_cast<Real>((globalBase + i) % 19);
        }
    }
}

// Update only interior cells. Both buffers start with identical boundary values,
// so the fixed global boundaries need no per-iteration copy.
void updatePlanes(const Real* input, Real* output, size_t nx, size_t ny,
                  size_t localFirst, size_t localLast, size_t globalFirst,
                  size_t globalNz) {
    const size_t plane = nx * ny;
    if (nx < 3 || ny < 3 || globalNz < 3) return;
    for (size_t z = localFirst; z <= localLast; ++z) {
        const size_t globalZ = globalFirst + z - 1;
        if (globalZ == 0 || globalZ + 1 == globalNz) continue;
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
}

void printUsage(const char* progName) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int argumentStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            argumentStatus = 2;
            break;
        } else {
            if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
            argumentStatus = 1;
            break;
        }
    }
    if (argumentStatus) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentStatus == 2 ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or unsupported grid size\n");
        MPI_Finalize();
        return 1;
    }

    const int ranks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= ranks) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const size_t plane = nx * ny;
    const size_t localNz = planeCount(nz, ranks, rank);
    const size_t globalFirst = firstPlane(nz, ranks, rank);
    std::vector<Real> grid1((localNz + 2) * plane);
    std::vector<Real> grid2((localNz + 2) * plane);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, localNz, globalFirst);
    initializeGrid(grid2, nx, ny, localNz, globalFirst);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;
        MPI_Request requests[4];
        int nrequests = 0;
        if (rank > 0) {
            MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 1, comm, &requests[nrequests++]);
            MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 0, comm, &requests[nrequests++]);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(input.data() + (localNz + 1) * plane,
                      static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                      comm, &requests[nrequests++]);
            MPI_Isend(input.data() + localNz * plane,
                      static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1,
                      comm, &requests[nrequests++]);
        }
        if (localNz > 2)
            updatePlanes(input.data(), output.data(), nx, ny, 2, localNz - 1,
                         globalFirst, nz);
        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        updatePlanes(input.data(), output.data(), nx, ny, 1, 1, globalFirst, nz);
        if (localNz > 1)
            updatePlanes(input.data(), output.data(), nx, ny, localNz, localNz,
                         globalFirst, nz);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n",
               maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> fullGrid;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
            std::copy_n(finalGrid.data() + plane, localNz * plane, fullGrid.data());
            for (int source = 1; source < ranks; ++source) {
                const size_t count = planeCount(nz, ranks, source) * plane;
                Real* destination = fullGrid.data() + firstPlane(nz, ranks, source) * plane;
                size_t offset = 0;
                while (offset < count) {
                    const int chunk = static_cast<int>(std::min(
                        count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Recv(destination + offset, chunk, MPI_DOUBLE, source, 2,
                             comm, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(chunk);
                }
            }
            print_results(fullGrid, "Grid");
        } else {
            const size_t count = localNz * plane;
            size_t offset = 0;
            while (offset < count) {
                const int chunk = static_cast<int>(std::min(
                    count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Send(finalGrid.data() + plane + offset, chunk, MPI_DOUBLE,
                         0, 2, comm);
                offset += static_cast<size_t>(chunk);
            }
        }
    }

    int status = 0;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int localFinite = 1;
        for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
            const Real value = finalGrid[i];
            localFinite &= std::isfinite(value) ? 1 : 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real minVal, maxVal;
        int allFinite;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&localFinite, &allFinite, 1, MPI_INT, MPI_MIN, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) {
                printf("Validation failed: found NaN or Inf value\n");
                status = 1;
            } else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    status = 1;
                }
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
