#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Each process owns contiguous z planes, with a ghost plane on each side.
// Both buffers retain the initialized physical boundary values throughout.
static void updatePlanes(const Real* input, Real* output, size_t nx, size_t ny,
                         size_t first, size_t last) {
    const size_t plane = nx * ny;
    for (size_t z = first; z < last; ++z) {
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

// MPI's traditional count argument is an int. Split large planes and result
// transfers into chunks so no size_t-to-int conversion can truncate a count.
static void postPlane(Real* data, size_t count, int peer, int tag, bool receive,
                      MPI_Comm comm, std::vector<MPI_Request>& requests) {
    while (count != 0) {
        const int n = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive) MPI_Irecv(data, n, MPI_DOUBLE, peer, tag, comm, &request);
        else MPI_Isend(data, n, MPI_DOUBLE, peer, tag, comm, &request);
        requests.push_back(request);
        data += n;
        count -= static_cast<size_t>(n);
    }
}

static void transferGrid(Real* data, size_t count, int peer, bool receive, MPI_Comm comm) {
    while (count != 0) {
        const int n = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (receive) MPI_Recv(data, n, MPI_DOUBLE, peer, 2, comm, MPI_STATUS_IGNORE);
        else MPI_Send(data, n, MPI_DOUBLE, peer, 2, comm);
        data += n;
        count -= static_cast<size_t>(n);
    }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > SIZE_MAX / sizeof(Real) ||
        nz > SIZE_MAX / (nx * ny) / sizeof(Real)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or too large grid size\n");
        MPI_Finalize();
        return 1;
    }

    // Exclude empty slabs when the job has more processes than z planes.
    const int activeSize = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeSize) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const size_t base = nz / activeSize, remainder = nz % activeSize;
    const size_t localZ = base + (static_cast<size_t>(rank) < remainder);
    const size_t zStart = base * static_cast<size_t>(rank) +
                          std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    if (localZ > SIZE_MAX / plane - 2 || (localZ + 2) * plane > SIZE_MAX / sizeof(Real)) {
        if (rank == 0) fprintf(stderr, "Local grid is too large\n");
        MPI_Abort(comm, 1);
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1((localZ + 2) * plane), grid2((localZ + 2) * plane);
    for (size_t z = 0; z < localZ; ++z) {
        const size_t globalOffset = (zStart + z) * plane;
        const size_t localOffset = (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>((globalOffset + i) % 19);
            grid1[localOffset + i] = value;
            if (iterations > 0) grid2[localOffset + i] = value;
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    if (nx > 2 && ny > 2 && nz > 2) {
        std::vector<MPI_Request> requests;
        for (int iter = 0; iter < iterations; ++iter) {
            requests.clear();
            if (rank > 0) {
                postPlane(input, plane, rank - 1, 1, true, comm, requests);
                postPlane(input + plane, plane, rank - 1, 0, false, comm, requests);
            }
            if (rank + 1 < activeSize) {
                postPlane(input + (localZ + 1) * plane, plane, rank + 1, 0, true, comm, requests);
                postPlane(input + localZ * plane, plane, rank + 1, 1, false, comm, requests);
            }
            // These planes depend only on local data, so they overlap the exchange.
            if (localZ > 2) updatePlanes(input, output, nx, ny, 2, localZ);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (zStart > 0 && zStart + 1 < nz) updatePlanes(input, output, nx, ny, 1, 2);
            if (localZ > 1 && zStart + localZ - 1 > 0 && zStart + localZ < nz)
                updatePlanes(input, output, nx, ny, localZ, localZ + 1);
            std::swap(input, output);
        }
    } else if (iterations > 0 && iterations % 2 != 0) {
        std::swap(input, output);
    }
    // Match the original buffer choice even for negative iteration counts.
    if (iterations < 0 && iterations % 2 != 0) input = grid2.data();
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> globalGrid(plane * nz);
            std::copy(input + plane, input + (localZ + 1) * plane, globalGrid.begin());
            for (int p = 1; p < activeSize; ++p) {
                const size_t countZ = base + (static_cast<size_t>(p) < remainder);
                const size_t startZ = base * static_cast<size_t>(p) + std::min(static_cast<size_t>(p), remainder);
                transferGrid(globalGrid.data() + startZ * plane, countZ * plane, p, true, comm);
            }
            print_results(globalGrid, "Grid");
        } else {
            transferGrid(input + plane, localZ * plane, 0, false, comm);
        }
    }
    int valid = 1;
    if (validate) {
        Real localMin = input[plane], localMax = input[plane];
        int localFinite = 1;
        for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
            const Real value = input[i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real globalMin = 0, globalMax = 0;
        int finite = 0;
        MPI_Reduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, 0, comm);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = 0;
            } else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
