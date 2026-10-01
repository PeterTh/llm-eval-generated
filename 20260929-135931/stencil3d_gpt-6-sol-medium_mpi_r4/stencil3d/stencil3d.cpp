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

// Each rank owns [firstZ, firstZ + localZ); planes 0 and localZ + 1 are halos.
void initializeGrid(std::vector<Real>& grid, size_t plane, size_t firstZ, size_t localZ) {
    for (size_t z = 1; z <= localZ; ++z) {
        const size_t globalBase = (firstZ + z - 1) * plane;
        const size_t localBase = z * plane;
        for (size_t i = 0; i < plane; ++i)
            grid[localBase + i] = static_cast<Real>((globalBase + i) % 19);
    }
}

// Update an inclusive range of local planes. Boundary cells stay initialized.
void stencilPlanes(const Real* __restrict input, Real* __restrict output,
                   size_t nx, size_t ny, size_t firstZ, size_t nz,
                   size_t begin, size_t end) {
    const size_t plane = nx * ny;
    for (size_t z = begin; z <= end; ++z) {
        const size_t globalZ = firstZ + z - 1;
        if (globalZ == 0 || globalZ == nz - 1) continue;
        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = row + x;
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - nx];
                const Real back = input[idx + nx];
                const Real bottom = input[idx - plane];
                const Real top = input[idx + plane];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, size_t plane, size_t localZ, int rank) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localValid = 1;
    for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
        const Real val = grid[i];
        if (!std::isfinite(val)) localValid = 0;
        localMin = std::min(localMin, val);
        localMax = std::max(localMax, val);
    }
    Real minVal, maxVal;
    int valid;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        if (!valid) printf("Validation failed: found NaN or Inf value\n");
        printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
        if (maxVal > 1e6 || minVal < -1e6)
            printf("Validation failed: values out of expected range\n");
    }
    return valid && maxVal <= 1e6 && minVal >= -1e6;
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
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (nx < 2 || ny < 2 || nz < 2 || nx > maxSize / ny ||
        nx * ny > maxSize / nz || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or MPI plane too large\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t activeRanks = std::min(nz, static_cast<size_t>(worldSize));
    const size_t base = nz / activeRanks;
    const size_t extra = nz % activeRanks;
    const size_t localZ = static_cast<size_t>(rank) < activeRanks
                            ? base + (static_cast<size_t>(rank) < extra) : 0;
    const size_t firstZ = static_cast<size_t>(rank) < activeRanks
                            ? static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra) : nz;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localZ ? (localZ + 2) * plane : 0);
    std::vector<Real> grid2(localZ ? (localZ + 2) * plane : 0);
    initializeGrid(grid1, plane, firstZ, localZ);
    initializeGrid(grid2, plane, firstZ, localZ);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    for (int iter = 0; iter < iterations; ++iter) {
        if (localZ) {
            const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
            const int upper = static_cast<size_t>(rank + 1) < activeRanks ? rank + 1 : MPI_PROC_NULL;
            MPI_Request requests[4];
            MPI_Irecv(input, static_cast<int>(plane), MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(input + (localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(input + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(input + localZ * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &requests[3]);

            if (localZ > 2)
                stencilPlanes(input, output, nx, ny, firstZ, nz, 2, localZ - 1);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            stencilPlanes(input, output, nx, ny, firstZ, nz, 1, 1);
            if (localZ > 1)
                stencilPlanes(input, output, nx, ny, firstZ, nz, localZ, localZ);
        }
        std::swap(input, output);
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double cellUpdates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / elapsed / 1e6);
    }

    if (printResults) {
        std::vector<Real> fullGrid;
        if (rank == 0) fullGrid.resize(plane * nz);
        // Transfer contiguous slabs only when the complete result is requested.
        const size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
        if (rank == 0) {
            std::copy(input + plane, input + (localZ + 1) * plane, fullGrid.begin());
            for (size_t source = 1; source < activeRanks; ++source) {
                const size_t sourceZ = base + (source < extra);
                const size_t sourceFirst = source * base + std::min(source, extra);
                const size_t offset = sourceFirst * plane;
                const size_t count = sourceZ * plane;
                for (size_t pos = 0; pos < count; pos += chunk)
                    MPI_Recv(fullGrid.data() + offset + pos, static_cast<int>(std::min(chunk, count - pos)),
                             MPI_DOUBLE, static_cast<int>(source), 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            print_results(fullGrid, "Grid");
        } else if (localZ) {
            const size_t count = localZ * plane;
            for (size_t pos = 0; pos < count; pos += chunk)
                MPI_Send(input + plane + pos, static_cast<int>(std::min(chunk, count - pos)),
                         MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }

    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult((iterations % 2 == 0) ? grid1 : grid2, plane, localZ, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        status = valid ? 0 : 1;
    }
    MPI_Finalize();
    return status;
}
