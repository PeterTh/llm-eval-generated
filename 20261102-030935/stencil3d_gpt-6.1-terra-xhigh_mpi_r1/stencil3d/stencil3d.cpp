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

// Compute local planes [begin, end), numbered from 1; planes 0 and
// localNz+1 are halos. Global boundary planes are left untouched.
static void updatePlanes(const Real* input, Real* output, size_t nx, size_t ny,
                         size_t nz, size_t zStart, size_t begin, size_t end) {
    if (nx < 3 || ny < 3 || nz < 3) return;
    const size_t plane = nx * ny;
    for (size_t z = begin; z < end; ++z) {
        const size_t globalZ = zStart + z - 1;
        if (globalZ == 0 || globalZ == nz - 1) continue;
        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - nx] + input[i + nx] +
                             input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

// MPI counts are int, so split large planes and result slabs into chunks.
static void postHalo(Real* buffer, size_t count, int peer, int tag, bool receive,
                     std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        if (receive)
            MPI_Irecv(buffer, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        else
            MPI_Isend(buffer, chunk, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &request);
        requests.push_back(request);
        buffer += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void transferSlab(Real* buffer, size_t count, int peer, bool receive) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (receive)
            MPI_Recv(buffer, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Send(buffer, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD);
        buffer += chunk;
        count -= static_cast<size_t>(chunk);
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
    if (nx == 0 || iterations < 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t base = nz / static_cast<size_t>(active);
    const size_t remainder = nz % static_cast<size_t>(active);
    const size_t localNz = rank < active ? base + (static_cast<size_t>(rank) < remainder) : 0;
    const size_t zStart = rank < active ? static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), remainder) : nz;
    if (localNz > SIZE_MAX / plane - std::min(static_cast<size_t>(2), SIZE_MAX / plane)) {
        if (rank == 0) fprintf(stderr, "Grid dimensions are too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localNz ? (localNz + 2) * plane : 0);
    std::vector<Real> grid2(grid1.size());
    for (size_t z = 0; z < localNz; ++z) {
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>(((zStart + z) * plane + i) % 19);
            grid1[(z + 1) * plane + i] = value;
            grid2[(z + 1) * plane + i] = value;
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int lower = rank > 0 && rank < active ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
    Real* input = grid1.data();
    Real* output = grid2.data();
    std::vector<MPI_Request> requests;
    for (int iter = 0; iter < iterations; ++iter) {
        if (localNz && nx >= 3 && ny >= 3 && nz >= 3) {
            requests.clear();
            postHalo(input, plane, lower, 1, true, requests);
            postHalo(input + (localNz + 1) * plane, plane, upper, 0, true, requests);
            postHalo(input + plane, plane, lower, 0, false, requests);
            postHalo(input + localNz * plane, plane, upper, 1, false, requests);
            if (localNz > 2) updatePlanes(input, output, nx, ny, nz, zStart, 2, localNz);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            updatePlanes(input, output, nx, ny, nz, zStart, 1, 2);
            if (localNz > 1) updatePlanes(input, output, nx, ny, nz, zStart, localNz, localNz + 1);
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                                   static_cast<double>(ny > 2 ? ny - 2 : 0) *
                                   static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / maxElapsed / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> fullGrid(plane * nz);
            std::copy(input + plane, input + (localNz + 1) * plane, fullGrid.begin());
            for (int r = 1; r < active; ++r) {
                const size_t count = base + (static_cast<size_t>(r) < remainder);
                const size_t offset = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder);
                transferSlab(fullGrid.data() + offset * plane, count * plane, r, true);
            }
            print_results(fullGrid, "Grid");
        } else if (localNz) {
            transferSlab(input + plane, localNz * plane, 0, false);
        }
    }

    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int finite = 1;
        for (size_t i = 0; i < localNz * plane; ++i) {
            const Real value = input[plane + i];
            if (!std::isfinite(value)) finite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real globalMin, globalMax;
        int allFinite;
        MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        valid = allFinite && globalMax <= 1e6 && globalMin >= -1e6;
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
