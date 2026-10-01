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

// Every rank owns a contiguous range of global Z planes. Empty ranges are
// placed at the end so that active ranks have adjacent rank numbers.
static size_t planeCount(int rank, size_t nz, int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    return base + (static_cast<size_t>(rank) < nz % static_cast<size_t>(ranks));
}

static size_t firstPlane(int rank, size_t nz, int ranks) {
    return static_cast<size_t>(rank) * (nz / static_cast<size_t>(ranks)) +
           std::min(static_cast<size_t>(rank), nz % static_cast<size_t>(ranks));
}

static void updatePlane(const Real* input, Real* output, size_t plane,
                        size_t nx, size_t ny, size_t globalZ, size_t nz) {
    const size_t area = nx * ny;
    const Real* current = input + plane * area;
    Real* next = output + plane * area;
    if (globalZ == 0 || globalZ + 1 == nz || nx < 3 || ny < 3) {
        std::memcpy(next, current, area * sizeof(Real));
        return;
    }

    std::memcpy(next, current, nx * sizeof(Real));
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = y * nx;
        next[row] = current[row];
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            next[i] = (current[i] + current[i - 1] + current[i + 1] +
                       current[i - nx] + current[i + nx] +
                       input[(plane - 1) * area + i] +
                       input[(plane + 1) * area + i]) / 7.0;
        }
        next[row + nx - 1] = current[row + nx - 1];
    }
    std::memcpy(next + (ny - 1) * nx, current + (ny - 1) * nx,
                nx * sizeof(Real));
}

// Results output requires global order, so assemble only when -r is set.
// Chunked sends avoid MPI's int count limit for large local grids.
static void gatherResults(const Real* local, size_t localCells,
                          size_t nx, size_t ny, size_t nz,
                          int rank, int ranks) {
    const size_t area = nx * ny;
    if (rank == 0) {
        std::vector<Real> full(area * nz);
        std::copy_n(local, localCells, full.data());
        for (int source = 1; source < ranks; ++source) {
            const size_t offset = firstPlane(source, nz, ranks) * area;
            const size_t count = planeCount(source, nz, ranks) * area;
            for (size_t done = 0; done < count;) {
                const int chunk = static_cast<int>(std::min(count - done, static_cast<size_t>(INT_MAX)));
                MPI_Recv(full.data() + offset + done, chunk, MPI_DOUBLE,
                         source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                done += static_cast<size_t>(chunk);
            }
        }
        print_results(full, "Grid");
    } else {
        for (size_t done = 0; done < localCells;) {
            const int chunk = static_cast<int>(std::min(localCells - done, static_cast<size_t>(INT_MAX)));
            MPI_Send(local + done, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            done += static_cast<size_t>(chunk);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > static_cast<size_t>(INT_MAX) / ny ||
        nx * ny > static_cast<size_t>(INT_MAX) ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny) - 2) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    const size_t area = nx * ny;
    const size_t localZ = planeCount(rank, nz, ranks);
    const size_t zStart = firstPlane(rank, nz, ranks);
    const size_t localCells = localZ * area;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const int prev = rank > 0 && rank < active ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
    std::vector<Real> grid1((localZ + 2) * area);
    std::vector<Real> grid2((localZ + 2) * area);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    for (size_t z = 0; z < localZ; ++z) {
        const size_t globalBase = (zStart + z) * area;
        Real* plane = grid1.data() + (z + 1) * area;
        for (size_t i = 0; i < area; ++i)
            plane[i] = static_cast<Real>((globalBase + i) % 19);
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0 ? grid1 : grid2).data();
        Real* output = (iter % 2 == 0 ? grid2 : grid1).data();
        MPI_Request requests[4];
        int requestCount = 0;
        if (localZ && nx >= 3 && ny >= 3 && nz >= 3) {
            if (prev != MPI_PROC_NULL) {
                MPI_Irecv(input, static_cast<int>(area), MPI_DOUBLE, prev, 0,
                          MPI_COMM_WORLD, &requests[requestCount++]);
                MPI_Isend(input + area, static_cast<int>(area), MPI_DOUBLE, prev, 1,
                          MPI_COMM_WORLD, &requests[requestCount++]);
            }
            if (next != MPI_PROC_NULL) {
                MPI_Irecv(input + (localZ + 1) * area, static_cast<int>(area),
                          MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[requestCount++]);
                MPI_Isend(input + localZ * area, static_cast<int>(area),
                          MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &requests[requestCount++]);
            }
        }
        for (size_t z = 2; z < localZ; ++z)
            updatePlane(input, output, z, nx, ny, zStart + z - 1, nz);
        if (requestCount) MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (localZ) updatePlane(input, output, 1, nx, ny, zStart, nz);
        if (localZ > 1) updatePlane(input, output, localZ, nx, ny, zStart + localZ - 1, nz);
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(duration * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double cells = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                             static_cast<double>(ny > 2 ? ny - 2 : 0) *
                             static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? cells / duration / 1e6 : 0.0);
    }

    const Real* finalGrid = (iterations % 2 == 0 ? grid1 : grid2).data() + area;
    if (printResults) gatherResults(finalGrid, localCells, nx, ny, nz, rank, ranks);

    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        for (size_t i = 0; i < localCells; ++i) {
            const Real value = finalGrid[i];
            if (!std::isfinite(value)) valid = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real globalMin = 0, globalMax = 0;
        int allFinite = 0;
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&valid, &allFinite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (globalMax > 1e6 || globalMin < -1e6)
                printf("Validation failed: values out of expected range\n");
            valid = allFinite && globalMax <= 1e6 && globalMin >= -1e6;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
