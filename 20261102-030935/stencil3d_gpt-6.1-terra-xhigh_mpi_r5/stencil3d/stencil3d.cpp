#include <algorithm>
#include <cerrno>
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

static bool parseSize(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long n = std::strtoull(text, &end, 10);
    if (errno || *end != '\0' || n > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(n);
    return true;
}

// Compute a range of local planes; the first owned plane is plane 1.
static void updatePlanes(const Real* __restrict input, Real* __restrict output,
                         size_t nx, size_t ny, size_t first, size_t last) {
    const size_t plane = nx * ny;
    for (size_t z = first; z < last; ++z) {
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i-1] + input[i+1] + input[i-nx] +
                             input[i+nx] + input[i-plane] + input[i+plane]) / 7.0;
            }
        }
    }
}

// MPI counts are int, so split very large contiguous planes into chunks.
static void postChunks(Real* data, size_t count, int peer, int tag, bool receive,
                       MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        requests.emplace_back();
        if (receive) MPI_Irecv(data, chunk, MPI_DOUBLE, peer, tag, comm, &requests.back());
        else MPI_Isend(data, chunk, MPI_DOUBLE, peer, tag, comm, &requests.back());
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void transferChunks(Real* data, size_t count, int peer, bool receive, MPI_Comm comm) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (receive) MPI_Recv(data, chunk, MPI_DOUBLE, peer, 2, comm, MPI_STATUS_IGNORE);
        else MPI_Send(data, chunk, MPI_DOUBLE, peer, 2, comm);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0, iterValue = 10;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value;
            if (!parseSize(argv[++i], value)) { status = 1; break; }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterValue = value;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            status = 1;
            break;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || iterValue > INT_MAX ||
        nx > limit / ny || nx * ny > limit / nz ||
        nx * ny * nz > limit / sizeof(Real)) status = 1;
    if (status) {
        if (rank == 0) {
            fprintf(stderr, "Invalid grid size, iteration count, or option.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    const int iterations = static_cast<int>(iterValue);
    const size_t plane = nx * ny;
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t base = nz / activeRanks, extra = nz % activeRanks;
    const size_t localNz = rank < activeRanks ? base + (static_cast<size_t>(rank) < extra) : 0;
    const size_t zStart = rank < activeRanks ? static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), extra) : nz;

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        printf("Validation: %s\nInitializing grid...\n", validate ? "enabled" : "disabled");
    }

    // Two extra planes hold halos. Both buffers start with the fixed boundaries.
    if (localNz && (localNz > limit - 2 ||
                    localNz + 2 > limit / sizeof(Real) / plane)) {
        fprintf(stderr, "Local grid is too large.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Real> grid1(localNz ? (localNz + 2) * plane : 0);
    std::vector<Real> grid2(grid1.size());
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalOffset = (zStart + z) * plane;
        const size_t localOffset = (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            grid1[localOffset + i] = grid2[localOffset + i] = static_cast<Real>((globalOffset + i) % 19);
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    const int prev = rank > 0 && rank < activeRanks ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    for (int iter = 0; iter < iterations; ++iter) {
        if (localNz && nx > 2 && ny > 2 && nz > 2) {
            requests.clear();
            postChunks(input, plane, prev, 1, true, MPI_COMM_WORLD, requests);
            postChunks(input + (localNz + 1) * plane, plane, next, 0, true, MPI_COMM_WORLD, requests);
            postChunks(input + plane, plane, prev, 0, false, MPI_COMM_WORLD, requests);
            postChunks(input + localNz * plane, plane, next, 1, false, MPI_COMM_WORLD, requests);

            // Interior planes do not depend on the incoming halos.
            if (localNz > 2) updatePlanes(input, output, nx, ny, 2, localNz);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (zStart > 0 && zStart + 1 < nz) updatePlanes(input, output, nx, ny, 1, 2);
            if (localNz > 1 && zStart + localNz < nz)
                updatePlanes(input, output, nx, ny, localNz, localNz + 1);
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> fullGrid(plane * nz);
            std::copy(input + plane, input + (localNz + 1) * plane, fullGrid.begin());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t sourceStart = static_cast<size_t>(source) * base + std::min(static_cast<size_t>(source), extra);
                const size_t sourceNz = base + (static_cast<size_t>(source) < extra);
                transferChunks(fullGrid.data() + sourceStart * plane, sourceNz * plane, source, true, MPI_COMM_WORLD);
            }
            print_results(fullGrid, "Grid");
        } else if (localNz) {
            transferChunks(input + plane, localNz * plane, 0, false, MPI_COMM_WORLD);
        }
    }

    if (validate) {
        int finite = 1;
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        for (size_t i = 0; i < localNz * plane; ++i) {
            const Real value = input[plane + i];
            if (!std::isfinite(value)) finite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int allFinite = 0;
        Real globalMin, globalMax;
        MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6)
                    printf("Validation failed: values out of expected range\n");
            }
            status = !allFinite || globalMax > 1e6 || globalMin < -1e6;
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
