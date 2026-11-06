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

// Compute one owned plane. Global boundary planes and the x/y edges are
// initialized in both buffers and never need to be copied again.
static inline void stencilPlane(const Real* input, Real* output, size_t z,
                                size_t nx, size_t ny, size_t plane) {
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = z * plane + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t idx = row + x;
            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                           input[idx - nx] + input[idx + nx] +
                           input[idx - plane] + input[idx + plane]) / 7.0;
        }
    }
}

// MPI's element count is an int, so split even very large planes into chunks.
static void postPlane(Real* data, size_t count, int peer, int tag, bool receive,
                      std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive)
            MPI_Irecv(data + offset, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else
            MPI_Isend(data + offset, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        offset += static_cast<size_t>(chunk);
    }
}

static void transferGrid(Real* data, size_t count, int peer, bool receive) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        if (receive)
            MPI_Recv(data + offset, chunk, MPI_DOUBLE, peer, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Send(data + offset, chunk, MPI_DOUBLE, peer, 20, MPI_COMM_WORLD);
        offset += static_cast<size_t>(chunk);
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
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit / ny || nx * ny > limit / nz) {
        if (rank == 0) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t active = std::min(nz, static_cast<size_t>(ranks));
    const size_t r = static_cast<size_t>(rank);
    const size_t depth = r < active ? nz / active + (r < nz % active) : 0;
    const size_t zStart = r < active ? r * (nz / active) + std::min(r, nz % active) : nz;
    const int prev = depth && rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = depth && r + 1 < active ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    if (depth && (depth > limit - 2 || depth + 2 > limit / plane)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Real> grid1(depth ? (depth + 2) * plane : 0);
    std::vector<Real> grid2(grid1.size());
    for (size_t z = 0; z < depth; ++z) {
        const size_t globalOffset = (zStart + z) * plane;
        const size_t localOffset = (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>((globalOffset + i) % 19);
            grid1[localOffset + i] = value;
            // With negative odd iterations the original selects its zero-filled second buffer.
            if (iterations >= 0) grid2[localOffset + i] = value;
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    std::vector<MPI_Request> requests;
    for (int iter = 0; iter < iterations; ++iter) {
        if (depth && nx > 2 && ny > 2 && nz > 2) {
            requests.clear();
            postPlane(input, plane, prev, 11, true, requests);
            postPlane(input + (depth + 1) * plane, plane, next, 10, true, requests);
            postPlane(input + plane, plane, prev, 10, false, requests);
            postPlane(input + depth * plane, plane, next, 11, false, requests);
            for (size_t z = 2; z < depth; ++z) {
                const size_t globalZ = zStart + z - 1;
                if (globalZ > 0 && globalZ + 1 < nz)
                    stencilPlane(input, output, z, nx, ny, plane);
            }
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (zStart > 0 && zStart + 1 < nz)
                stencilPlane(input, output, 1, nx, ny, plane);
            if (depth > 1 && zStart + depth - 1 > 0 && zStart + depth < nz)
                stencilPlane(input, output, depth, nx, ny, plane);
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    // Match the original buffer selection, including negative odd iteration counts.
    const Real* finalData = (iterations % 2 == 0 ? grid1.data() : grid2.data());
    if (printResults) {
        if (rank == 0) {
            std::vector<Real> result(plane * nz);
            std::copy_n(finalData + plane, depth * plane, result.data());
            for (size_t source = 1; source < active; ++source) {
                const size_t sourceDepth = nz / active + (source < nz % active);
                const size_t sourceStart = source * (nz / active) + std::min(source, nz % active);
                transferGrid(result.data() + sourceStart * plane, sourceDepth * plane,
                             static_cast<int>(source), true);
            }
            print_results(result, "Grid");
        } else if (depth) {
            transferGrid(const_cast<Real*>(finalData + plane), depth * plane, 0, false);
        }
    }
    int status = 0;
    if (validate) {
        int bad = 0;
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        for (size_t i = 0; i < depth * plane; ++i) {
            const Real value = finalData[plane + i];
            if (!std::isfinite(value)) bad = 1;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int anyBad = 0;
        Real globalMin = 0, globalMax = 0;
        MPI_Reduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (anyBad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6)
                    printf("Validation failed: values out of expected range\n");
            }
            status = anyBad || globalMax > 1e6 || globalMin < -1e6;
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Finalize();
    return status;
}
