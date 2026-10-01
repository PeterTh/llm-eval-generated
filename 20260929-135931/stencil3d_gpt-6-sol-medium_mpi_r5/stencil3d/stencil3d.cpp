#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// A Z plane is contiguous in the original row-major layout.
static void updatePlane(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t z) {
    const size_t plane = nx * ny;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = z * plane + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx]
                    + in[i-plane] + in[i+plane]) / 7.0;
        }
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            parseStatus = 2;
            break;
        }
        if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (i + 1 < argc && (strcmp(argv[i], "-x") == 0 ||
                 strcmp(argv[i], "-y") == 0 || strcmp(argv[i], "-z") == 0 ||
                 strcmp(argv[i], "-i") == 0)) {
            const char* option = argv[i++];
            char* end = nullptr;
            if (strcmp(option, "-i") == 0) {
                long value = strtol(argv[i], &end, 10);
                if (!*argv[i] || *end || value < 0 || value > INT_MAX) {
                    parseStatus = 1;
                    break;
                }
                iterations = static_cast<int>(value);
            } else {
                if (argv[i][0] == '-') {
                    parseStatus = 1;
                    break;
                }
                unsigned long long value = strtoull(argv[i], &end, 10);
                if (!*argv[i] || *end || value > std::numeric_limits<size_t>::max()) {
                    parseStatus = 1;
                    break;
                }
                if (strcmp(option, "-x") == 0) nx = static_cast<size_t>(value);
                if (strcmp(option, "-y") == 0) ny = static_cast<size_t>(value);
                if (strcmp(option, "-z") == 0) nz = static_cast<size_t>(value);
            }
        } else {
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus == 2) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(INT_MAX) ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny)) parseStatus = 1;
    if (parseStatus) {
        if (rank == 0) {
            printf("Invalid option or grid size. Dimensions must be at least 2 and an XY plane must fit in an MPI count.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const int activeSize = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeSize ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const size_t localZ = nz / activeSize + (static_cast<size_t>(rank) < nz % activeSize);
    const size_t firstZ = static_cast<size_t>(rank) * (nz / activeSize)
                        + std::min(static_cast<size_t>(rank), nz % activeSize);
    if (localZ > std::numeric_limits<size_t>::max() / plane - 2) {
        if (rank == 0) printf("Grid is too large.\n");
        MPI_Abort(comm, 1);
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Both buffers keep the original fixed boundary values throughout all iterations.
    std::vector<Real> grid1((localZ + 2) * plane);
    std::vector<Real> grid2((localZ + 2) * plane);
    for (size_t z = 1; z <= localZ; ++z) {
        const size_t globalBase = (firstZ + z - 1) * plane;
        const size_t localBase = z * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>((globalBase + i) % 19);
            grid1[localBase + i] = value;
            grid2[localBase + i] = value;
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == activeSize ? MPI_PROC_NULL : rank + 1;
    const int count = static_cast<int>(plane);
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = grid1.data();
        Real* out = grid2.data();
        MPI_Request requests[4];
        int nrequests = 0;
        if (lower != MPI_PROC_NULL) {
            MPI_Irecv(in, count, MPI_DOUBLE, lower, 1, comm, &requests[nrequests++]);
            MPI_Isend(in + plane, count, MPI_DOUBLE, lower, 0, comm, &requests[nrequests++]);
        }
        if (upper != MPI_PROC_NULL) {
            MPI_Irecv(in + (localZ + 1) * plane, count, MPI_DOUBLE,
                      upper, 0, comm, &requests[nrequests++]);
            MPI_Isend(in + localZ * plane, count, MPI_DOUBLE,
                      upper, 1, comm, &requests[nrequests++]);
        }
        for (size_t z = 2; z < localZ; ++z) updatePlane(in, out, nx, ny, z);
        if (nrequests) MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        if (firstZ > 0 && firstZ < nz - 1) updatePlane(in, out, nx, ny, 1);
        if (localZ > 1 && firstZ + localZ - 1 < nz - 1)
            updatePlane(in, out, nx, ny, localZ);
        std::swap(grid1, grid2);
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / seconds / 1e6);
    }

    if (printResults) {
        // Gather in global order to retain the serial sum, samples, and hash.
        if (rank == 0) {
            std::vector<Real> fullGrid(nx * ny * nz);
            std::memcpy(fullGrid.data(), grid1.data() + plane, localZ * plane * sizeof(Real));
            for (int source = 1; source < activeSize; ++source) {
                const size_t sourceZ = nz / activeSize + (static_cast<size_t>(source) < nz % activeSize);
                const size_t sourceFirst = static_cast<size_t>(source) * (nz / activeSize)
                                         + std::min(static_cast<size_t>(source), nz % activeSize);
                size_t remaining = sourceZ * plane;
                Real* destination = fullGrid.data() + sourceFirst * plane;
                while (remaining) {
                    const int chunk = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(destination, chunk, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                    destination += chunk;
                    remaining -= chunk;
                }
            }
            print_results(fullGrid, "Grid");
        } else {
            size_t remaining = localZ * plane;
            Real* source = grid1.data() + plane;
            while (remaining) {
                const int chunk = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                MPI_Send(source, chunk, MPI_DOUBLE, 0, 2, comm);
                source += chunk;
                remaining -= chunk;
            }
        }
    }

    int result = 0;
    if (validate) {
        int localInvalid = 0;
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        const Real* owned = grid1.data() + plane;
        for (size_t i = 0; i < localZ * plane; ++i) {
            const Real value = owned[i];
            localInvalid |= !std::isfinite(value);
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int invalid = 0;
        Real minVal, maxVal;
        MPI_Reduce(&localInvalid, &invalid, 1, MPI_INT, MPI_MAX, 0, comm);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (invalid) printf("Validation failed: found NaN or Inf value\n");
            else printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
            if (!invalid && (maxVal > 1e6 || minVal < -1e6))
                printf("Validation failed: values out of expected range\n");
            result = invalid || maxVal > 1e6 || minVal < -1e6;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return result;
}
