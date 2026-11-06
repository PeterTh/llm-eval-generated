#include <algorithm>
#include <chrono>
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

// A plane can exceed the int count accepted by MPI. Keep chunks in order;
// MPI guarantees ordering for messages from the same source with the same tag.
void exchangePlane(Real* buffer, size_t count, int peer, int tag, bool receive,
                   std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive)
            MPI_Irecv(buffer + offset, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else
            MPI_Isend(buffer + offset, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        offset += static_cast<size_t>(chunk);
    }
}

void transferGrid(Real* buffer, size_t count, int peer, bool receive) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        if (receive)
            MPI_Recv(buffer + offset, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Send(buffer + offset, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD);
        offset += static_cast<size_t>(chunk);
    }
}

// Local plane indices start at 1; planes 0 and localNz+1 are halos.
void updatePlanes(const Real* __restrict input, Real* __restrict output,
                  size_t nx, size_t ny, size_t first, size_t end) {
    const size_t plane = nx * ny;
    for (size_t z = first; z < end; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t p = row + x;
                output[p] = (input[p] + input[p - 1] + input[p + 1] +
                             input[p - nx] + input[p + nx] +
                             input[p - plane] + input[p + plane]) / 7.0;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
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
    const size_t limit = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > limit / ny || nx * ny > limit / nz ||
        nx * ny * nz > limit / sizeof(Real)) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    if (localNz > limit / plane - std::min<size_t>(2, limit / plane)) {
        if (rank == 0) fprintf(stderr, "Local grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    // Both buffers start with the fixed physical boundary values. Interior
    // values in the destination are overwritten on every iteration.
    std::vector<Real> grid1(localNz ? (localNz + 2) * plane : 0);
    std::vector<Real> grid2(grid1.size());
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalOffset = (zStart + z) * plane;
        const size_t localOffset = (z + 1) * plane;
        for (size_t p = 0; p < plane; ++p)
            grid1[localOffset + p] = grid2[localOffset + p] = static_cast<Real>((globalOffset + p) % 19);
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    std::vector<MPI_Request> requests;
    for (int iter = 0; iter < iterations; ++iter) {
        if (localNz && hasInterior) {
            requests.clear();
            const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
            const int upper = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
            if (lower != MPI_PROC_NULL) {
                exchangePlane(input, plane, lower, 1, true, requests);
                exchangePlane(input + plane, plane, lower, 0, false, requests);
            }
            if (upper != MPI_PROC_NULL) {
                exchangePlane(input + (localNz + 1) * plane, plane, upper, 0, true, requests);
                exchangePlane(input + localNz * plane, plane, upper, 1, false, requests);
            }
            // Local planes away from slab ends do not depend on incoming halos.
            if (localNz > 2) updatePlanes(input, output, nx, ny, 2, localNz);
            if (!requests.empty())
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (zStart > 0 && zStart < nz - 1)
                updatePlanes(input, output, nx, ny, 1, 2);
            if (localNz > 1 && zStart + localNz - 1 > 0 && zStart + localNz - 1 < nz - 1)
                updatePlanes(input, output, nx, ny, localNz, localNz + 1);
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> globalGrid(gridSize);
            std::copy_n(input + plane, localNz * plane, globalGrid.data());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t count = base + (static_cast<size_t>(source) < extra ? 1 : 0);
                const size_t startZ = static_cast<size_t>(source) * base + std::min(static_cast<size_t>(source), extra);
                transferGrid(globalGrid.data() + startZ * plane, count * plane, source, true);
            }
            print_results(globalGrid, "Grid");
        } else if (localNz) {
            transferGrid(input + plane, localNz * plane, 0, false);
        }
    }

    int valid = 1;
    if (validate) {
        int localFinite = 1;
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        for (size_t p = 0; p < localNz * plane; ++p) {
            const Real value = input[plane + p];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int allFinite;
        Real minVal, maxVal;
        MPI_Reduce(&localFinite, &allFinite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = 0;
            } else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
