#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Each rank owns whole z planes. The extra plane on either side is a halo.
static void updatePlanes(const Real* __restrict__ input, Real* __restrict__ output,
                         size_t nx, size_t ny, size_t first, size_t last) {
    const size_t plane = nx * ny;
    for (size_t z = first; z < last; ++z) {
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - nx] + input[i + nx] + input[i - plane] +
                             input[i + plane]) / 7.0;
            }
        }
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
    if (nx == 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const size_t plane = nx * ny;
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localZ = base + (static_cast<size_t>(rank) < remainder);
    const size_t zStart = base * static_cast<size_t>(rank) +
                          std::min(static_cast<size_t>(rank), remainder);
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const int lower = rank > 0 && localZ ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;
    std::vector<Real> grid1((localZ + 2) * plane);
    std::vector<Real> grid2((localZ + 2) * plane);
    for (size_t z = 0; z < localZ; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local = (z + 1) * plane + y * nx + x;
                const size_t global = (zStart + z) * plane + y * nx + x;
                grid1[local] = (global % 19) * 1.0;
                // Boundaries never change; initialize them in both buffers once.
                if (iterations > 0 && (x == 0 || x + 1 == nx || y == 0 ||
                    y + 1 == ny || zStart + z == 0 || zStart + z + 1 == nz)) {
                    grid2[local] = grid1[local];
                }
            }
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* input = grid1.data();
    Real* output = grid2.data();
    std::vector<MPI_Request> requests;
    for (int iter = 0; iter < iterations; ++iter) {
        if (localZ && nx > 2 && ny > 2 && nz > 2) {
            requests.clear();
            // Chunk messages to stay within the MPI int count limit.
            for (size_t offset = 0; offset < plane;) {
                const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
                MPI_Request request;
                if (lower != MPI_PROC_NULL) {
                    MPI_Irecv(input + offset, count, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &request);
                    requests.push_back(request);
                    MPI_Isend(input + plane + offset, count, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &request);
                    requests.push_back(request);
                }
                if (upper != MPI_PROC_NULL) {
                    MPI_Irecv(input + (localZ + 1) * plane + offset, count, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &request);
                    requests.push_back(request);
                    MPI_Isend(input + localZ * plane + offset, count, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &request);
                    requests.push_back(request);
                }
                offset += static_cast<size_t>(count);
            }
            // Local planes that do not depend on a halo can overlap communication.
            if (localZ > 2) updatePlanes(input, output, nx, ny, 2, localZ);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (zStart > 0 && zStart + 1 < nz) updatePlanes(input, output, nx, ny, 1, 2);
            if (localZ > 1 && zStart + localZ - 1 > 0 && zStart + localZ < nz)
                updatePlanes(input, output, nx, ny, localZ, localZ + 1);
        }
        std::swap(input, output);
    }
    // The original selects its zero-initialized second buffer for negative odd counts.
    if (iterations < 0 && iterations % 2 != 0) input = grid2.data();
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / (ms / 1000.0) / 1e6);
    }

    if (printResults) {
        // Preserve the original global order for the sequential Kahan sum and hash.
        std::vector<Real> global;
        if (rank == 0) {
            global.resize(plane * nz);
            std::copy(input + plane, input + plane + localZ * plane, global.begin());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t countZ = base + (static_cast<size_t>(source) < remainder);
                const size_t startZ = base * static_cast<size_t>(source) +
                                      std::min(static_cast<size_t>(source), remainder);
                const size_t count = countZ * plane;
                for (size_t offset = 0; offset < count;) {
                    const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(global.data() + startZ * plane + offset, chunk, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(chunk);
                }
            }
            print_results(global, "Grid");
        } else if (localZ) {
            const size_t count = localZ * plane;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                MPI_Send(input + plane + offset, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                offset += static_cast<size_t>(chunk);
            }
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int localFinite = 1;
        for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
            const Real value = input[i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real globalMin, globalMax;
        int finite;
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!finite) {
            if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else {
            if (rank == 0) printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (globalMax > 1e6 || globalMin < -1e6) {
                if (rank == 0) printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
