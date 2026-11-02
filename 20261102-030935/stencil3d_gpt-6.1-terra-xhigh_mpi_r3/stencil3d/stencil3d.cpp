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

static bool parseSize(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long n = std::strtoull(text, &end, 10);
    if (errno || *end != '\0' || n > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(n);
    return true;
}

// Transfer arbitrarily large planes without exceeding MPI's int count limit.
static void postPlane(Real* data, size_t count, int peer, int tag, bool receive,
                      std::vector<MPI_Request>& requests) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive) MPI_Irecv(data, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else MPI_Isend(data, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void transferBlock(Real* data, size_t count, int peer, bool receive) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (receive) MPI_Recv(data, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else MPI_Send(data, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

// The two buffers already contain the fixed boundary values. Only update interior cells.
static void updatePlane(const Real* __restrict input, Real* __restrict output,
                        size_t nx, size_t ny, size_t plane, size_t z) {
    const size_t base = z * plane;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] + input[i - plane] +
                         input[i + plane]) / 7.0;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0, iterationCount = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        size_t* target = nullptr;
        if (strcmp(argv[i], "-x") == 0) target = &nx;
        else if (strcmp(argv[i], "-y") == 0) target = &ny;
        else if (strcmp(argv[i], "-z") == 0) target = &nz;
        else if (strcmp(argv[i], "-i") == 0) target = &iterationCount;
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
        if (target && (i + 1 >= argc || !parseSize(argv[++i], *target))) {
            if (rank == 0) fprintf(stderr, "Invalid or missing numeric argument\n");
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit / ny || nx * ny > limit / nz ||
        iterationCount > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t nRanks = static_cast<size_t>(ranks);
    const size_t myRank = static_cast<size_t>(rank);
    const size_t localNz = nz / nRanks + (myRank < nz % nRanks);
    const size_t firstZ = myRank * (nz / nRanks) + std::min(myRank, nz % nRanks);
    if (localNz && (limit / plane < 2 || localNz > limit / plane - 2)) {
        if (rank == 0) fprintf(stderr, "Local grid size is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %zu\nValidation: %s\n",
               nx, ny, nz, iterationCount, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localNz ? (localNz + 2) * plane : 0);
    std::vector<Real> grid2(grid1.size());
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalBase = (firstZ + z) * plane;
        const size_t localBase = (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            grid1[localBase + i] = static_cast<Real>((globalBase + i) % 19);
        }
    }
    grid2 = grid1;
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    for (size_t iter = 0; iter < iterationCount; ++iter) {
        if (localNz && hasInterior) {
            requests.clear();
            const bool lower = firstZ > 0;
            const bool upper = firstZ + localNz < nz;
            if (lower) {
                postPlane(input, plane, rank - 1, 1, true, requests);
                postPlane(input + plane, plane, rank - 1, 0, false, requests);
            }
            if (upper) {
                postPlane(input + (localNz + 1) * plane, plane, rank + 1, 0, true, requests);
                postPlane(input + localNz * plane, plane, rank + 1, 1, false, requests);
            }
            for (size_t z = 2; z < localNz; ++z)
                updatePlane(input, output, nx, ny, plane, z);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (firstZ > 0 && firstZ + 1 < nz)
                updatePlane(input, output, nx, ny, plane, 1);
            if (localNz > 1 && firstZ + localNz - 1 > 0 && firstZ + localNz < nz)
                updatePlane(input, output, nx, ny, plane, localNz);
        }
        std::swap(input, output);
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                             static_cast<double>(ny > 2 ? ny - 2 : 0) *
                             static_cast<double>(nz > 2 ? nz - 2 : 0) * iterationCount;
        printf("Computation time: %ld ms\n", milliseconds);
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / maxElapsed / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> globalGrid(nx * ny * nz);
            std::copy_n(input + plane, localNz * plane, globalGrid.data());
            for (int r = 1; r < ranks; ++r) {
                size_t rRank = static_cast<size_t>(r);
                size_t rNz = nz / nRanks + (rRank < nz % nRanks);
                size_t rFirst = rRank * (nz / nRanks) + std::min(rRank, nz % nRanks);
                if (rNz) transferBlock(globalGrid.data() + rFirst * plane, rNz * plane, r, true);
            }
            print_results(globalGrid, "Grid");
        } else if (localNz) {
            transferBlock(input + plane, localNz * plane, 0, false);
        }
    }
    int valid = 1;
    if (validate) {
        int localFinite = 1;
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
            Real value = input[i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int finite;
        Real minValue, maxValue;
        MPI_Reduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
                if (maxValue > 1e6 || minValue < -1e6) {
                    printf("Validation failed: values out of expected range\n"); valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
