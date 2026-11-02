#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

// Work on complete owned planes in [begin, end), skipping global Z boundaries.
static void updatePlanes(const Real* __restrict__ input, Real* __restrict__ output,
                         size_t nx, size_t ny, size_t nz, size_t zStart,
                         size_t begin, size_t end) {
    const size_t plane = nx * ny;
    for (size_t z = begin; z < end; ++z) {
        const size_t globalZ = zStart + z - 1;
        if (globalZ == 0 || globalZ == nz - 1) continue;
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

// Split large transfers into MPI counts representable by an int.
static void postHalo(Real* data, size_t count, int peer, int tag, bool receive,
                     std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive) MPI_Irecv(data, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else MPI_Isend(data, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void transferResult(Real* data, size_t count, int peer, bool receive) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (receive) MPI_Recv(data, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else MPI_Send(data, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static bool parseSize(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0, iterationCount = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value;
            if (!parseSize(argv[++i], value) || (option == 'i' && value > INT_MAX)) {
                if (rank == 0) fprintf(stderr, "Invalid value for -%c: %s\n", option, argv[i]);
                MPI_Finalize();
                return 1;
            }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterationCount = value;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
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
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxSize / ny || nx * ny > maxSize / nz) {
        if (rank == 0) fprintf(stderr, "Invalid or too large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localZ = base + (static_cast<size_t>(rank) < remainder);
    const size_t zStart = static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), remainder);
    if (localZ && (maxSize / plane < 2 || localZ > maxSize / plane - 2)) {
        if (rank == 0) fprintf(stderr, "Local grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %zu\n", iterationCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    // Both buffers start with the same fixed boundaries. Halo planes are overwritten
    // before use, while owned interior cells are overwritten on every iteration.
    const size_t localSize = localZ ? (localZ + 2) * plane : 0;
    std::vector<Real> grid1(localSize), grid2(localSize);
    for (size_t z = 0; z < localZ; ++z) {
        const size_t globalOffset = (zStart + z) * plane;
        const size_t localOffset = (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i)
            grid1[localOffset + i] = static_cast<Real>((globalOffset + i) % 19);
    }
    grid2 = grid1;
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const int previous = rank > 0 && localZ ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
    std::vector<MPI_Request> requests;
    for (size_t iter = 0; iter < iterationCount; ++iter) {
        if (localZ && nx > 2 && ny > 2 && nz > 2) {
            requests.clear();
            postHalo(input, plane, previous, 1, true, requests);
            postHalo(input + (localZ + 1) * plane, plane, next, 0, true, requests);
            postHalo(input + plane, plane, previous, 0, false, requests);
            postHalo(input + localZ * plane, plane, next, 1, false, requests);
            if (localZ > 2) updatePlanes(input, output, nx, ny, nz, zStart, 2, localZ);
            if (!requests.empty())
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            updatePlanes(input, output, nx, ny, nz, zStart, 1, 2);
            if (localZ > 1) updatePlanes(input, output, nx, ny, nz, zStart, localZ, localZ + 1);
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterationCount;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> result(plane * nz);
            std::copy_n(input + plane, localZ * plane, result.data());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t countZ = base + (static_cast<size_t>(source) < remainder);
                const size_t startZ = static_cast<size_t>(source) * base +
                                      std::min(static_cast<size_t>(source), remainder);
                transferResult(result.data() + startZ * plane, countZ * plane, source, true);
            }
            print_results(result, "Grid");
        } else if (localZ) {
            transferResult(input + plane, localZ * plane, 0, false);
        }
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int localValid = 1;
        for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
            if (!std::isfinite(input[i])) localValid = 0;
            localMin = std::min(localMin, input[i]);
            localMax = std::max(localMax, input[i]);
        }
        Real globalMin, globalMax;
        MPI_Reduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6) {
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
