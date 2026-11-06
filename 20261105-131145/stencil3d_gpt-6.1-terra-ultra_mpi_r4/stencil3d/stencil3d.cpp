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

// The order of operations is the same as in the original stencil.
static void updatePlanes(const Real* input, Real* output, size_t nx, size_t ny,
                         size_t begin, size_t end) {
    const size_t plane = nx * ny;
    for (size_t z = begin; z < end; ++z) {
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

// Split large messages to keep the count representable by the MPI int API.
static void exchangePlane(Real* data, size_t plane, int peer, int tag,
                          bool receive, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < plane;) {
        const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive)
            MPI_Irecv(data + offset, count, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else
            MPI_Isend(data + offset, count, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        offset += count;
    }
}

// The reporting hash and Kahan sum depend on the global element order. Stream
// slabs to rank zero in that order without allocating another global grid.
static void reportResults(const std::vector<Real>& grid, size_t plane, size_t localZ,
                          size_t nz, int rank, int activeRanks) {
    constexpr size_t chunkSize = 1024 * 1024;
    const size_t localElements = plane * localZ;
    if (rank != 0) {
        if (rank < activeRanks) {
            for (size_t offset = 0; offset < localElements;) {
                const int count = static_cast<int>(std::min(chunkSize, localElements - offset));
                MPI_Send(grid.data() + plane + offset, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                offset += count;
            }
        }
        return;
    }

    const size_t total = plane * nz;
    const size_t indices[5] = {0, total / 4, total / 2, 3 * (total / 4) + (3 * (total % 4)) / 4, total - 1};
    Real samples[5] = {};
    Real sum = 0, correction = 0, minVal = grid[plane], maxVal = grid[plane];
    hash h;
    size_t position = 0;
    auto consume = [&](const Real* data, size_t count) {
        for (size_t i = 0; i < count; ++i, ++position) {
            const Real val = data[i];
            const Real y = val - correction;
            const Real t = sum + y;
            correction = (t - sum) - y;
            sum = t;
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
            h.add(val);
            for (int s = 0; s < 5; ++s) if (position == indices[s]) samples[s] = val;
        }
    };
    consume(grid.data() + plane, localElements);
    std::vector<Real> buffer(std::min(chunkSize, total));
    for (int source = 1; source < activeRanks; ++source) {
        const size_t sourceZ = nz / activeRanks + (static_cast<size_t>(source) < nz % activeRanks);
        size_t remaining = plane * sourceZ;
        while (remaining) {
            const int count = static_cast<int>(std::min(chunkSize, remaining));
            MPI_Recv(buffer.data(), count, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            consume(buffer.data(), count);
            remaining -= count;
        }
    }
    printf("=== RESULTS ===\nName: Grid\nElements: %zu\n", total);
    printf("Sum: %.17e\nMin: %.17e\nMax: %.17e\n", sum, minVal, maxVal);
    for (int s = 0; s < 5; ++s) printf("Sample[%zu]: %.17e\n", indices[s], samples[s]);
    printf("Hash: %016" PRIx64 "\n=== END RESULTS ===\n", h.get());
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
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
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
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx > maxSize / ny ||
        nx * ny > maxSize / nz || nz > maxSize - 2) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t localZ = rank < activeRanks ? nz / activeRanks + (static_cast<size_t>(rank) < nz % activeRanks) : 0;
    const size_t startZ = rank < activeRanks ? static_cast<size_t>(rank) * (nz / activeRanks) +
                          std::min(static_cast<size_t>(rank), nz % activeRanks) : nz;
    if (plane > maxSize / (localZ + 2)) {
        if (rank == 0) fprintf(stderr, "Local grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localZ ? (localZ + 2) * plane : 0);
    for (size_t z = 0; z < localZ; ++z)
        for (size_t i = 0; i < plane; ++i)
            grid1[(z + 1) * plane + i] = static_cast<Real>(((startZ + z) * plane + i) % 19);
    // Fixed boundary values are present in both buffers for all timesteps.
    std::vector<Real> grid2 = grid1;
    Real* input = grid1.data();
    Real* output = grid2.data();
    const int lower = rank > 0 && rank < activeRanks ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
    std::vector<MPI_Request> requests;
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (localZ && nx > 2 && ny > 2 && nz > 2) {
            requests.clear();
            exchangePlane(input, plane, lower, 1, true, requests);
            exchangePlane(input + (localZ + 1) * plane, plane, upper, 0, true, requests);
            exchangePlane(input + plane, plane, lower, 0, false, requests);
            exchangePlane(input + localZ * plane, plane, upper, 1, false, requests);
            const size_t first = startZ == 0 ? 2 : 1;
            const size_t end = startZ + localZ == nz ? localZ : localZ + 1;
            const size_t innerBegin = std::max(first, size_t{2});
            const size_t innerEnd = std::min(end, localZ);
            if (innerBegin < innerEnd) updatePlanes(input, output, nx, ny, innerBegin, innerEnd);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (first < end && first == 1) updatePlanes(input, output, nx, ny, 1, 2);
            if (localZ > 1 && first <= localZ && localZ < end)
                updatePlanes(input, output, nx, ny, localZ, localZ + 1);
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
            static_cast<double>(ny > 2 ? ny - 2 : 0) * static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / maxElapsed / 1e6);
    }
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) reportResults(finalGrid, plane, localZ, nz, rank, activeRanks);
    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int localBad = 0;
        for (size_t i = plane; localZ && i < (localZ + 1) * plane; ++i) {
            const Real val = finalGrid[i];
            if (!std::isfinite(val)) localBad = 1;
            localMin = std::min(localMin, val);
            localMax = std::max(localMax, val);
        }
        Real minVal, maxVal;
        int bad;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (bad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    bad = 1;
                }
            }
            printf("Validation: %s\n", bad ? "FAILED" : "PASSED");
            status = bad;
        }
    }
    MPI_Finalize();
    return status;
}
