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

// Each participating rank owns a contiguous slab of global Z planes. The
// planes before and after the slab are halos, never part of the result.
static void stencilPlane(const Real* input, Real* output, size_t z,
                         size_t nx, size_t ny, size_t plane) {
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

// Chunk messages so that even a plane or slab larger than MPI's int count
// limit can be transferred with ordinary MPI implementations.
static void transferSlab(Real* data, size_t count, int peer, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t base = nz / static_cast<size_t>(active);
    const size_t extra = nz % static_cast<size_t>(active);
    const size_t localZ = rank < active ? base + (static_cast<size_t>(rank) < extra) : 0;
    const size_t firstZ = rank < active ? static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), extra) : nz;
    if (localZ && (std::numeric_limits<size_t>::max() / plane < 2 ||
                   localZ > std::numeric_limits<size_t>::max() / plane - 2)) {
        if (rank == 0) fprintf(stderr, "Grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localZ ? (localZ + 2) * plane : 0);
    std::vector<Real> grid2(grid1.size());
    for (size_t z = 0; z < localZ; ++z) {
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>(((firstZ + z) * plane + i) % 19);
            grid1[(z + 1) * plane + i] = value;
            // Boundaries never change; initializing both buffers avoids copying
            // them on every iteration. Interior values are overwritten.
            grid2[(z + 1) * plane + i] = value;
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    std::vector<MPI_Request> requests;
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0 ? grid1 : grid2).data();
        Real* output = (iter % 2 == 0 ? grid2 : grid1).data();
        if (localZ == 0) continue;
        requests.clear();
        const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
        const int upper = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
        for (size_t offset = 0; offset < plane;) {
            const int chunk = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
            MPI_Request request;
            if (lower != MPI_PROC_NULL) {
                MPI_Irecv(input + offset, chunk, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &request);
                requests.push_back(request);
                MPI_Isend(input + plane + offset, chunk, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &request);
                requests.push_back(request);
            }
            if (upper != MPI_PROC_NULL) {
                MPI_Irecv(input + (localZ + 1) * plane + offset, chunk, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &request);
                requests.push_back(request);
                MPI_Isend(input + localZ * plane + offset, chunk, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &request);
                requests.push_back(request);
            }
            offset += static_cast<size_t>(chunk);
        }
        for (size_t z = 2; z < localZ; ++z) {
            stencilPlane(input, output, z, nx, ny, plane);
        }
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        if (firstZ > 0 && firstZ + 1 < nz) stencilPlane(input, output, 1, nx, ny, plane);
        if (localZ > 1 && firstZ + localZ - 1 > 0 && firstZ + localZ < nz)
            stencilPlane(input, output, localZ, nx, ny, plane);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                                   static_cast<double>(ny > 2 ? ny - 2 : 0) *
                                   static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / maxElapsed / 1e6);
    }

    std::vector<Real>& finalGrid = iterations % 2 == 0 ? grid1 : grid2;
    if (printResults) {
        if (rank == 0) {
            std::vector<Real> fullGrid(nx * ny * nz);
            std::copy_n(finalGrid.data() + plane, localZ * plane, fullGrid.data());
            for (int source = 1; source < active; ++source) {
                const size_t sourceZ = static_cast<size_t>(source) * base + std::min(static_cast<size_t>(source), extra);
                const size_t sourceCount = (base + (static_cast<size_t>(source) < extra)) * plane;
                transferSlab(fullGrid.data() + sourceZ * plane, sourceCount, source, false);
            }
            print_results(fullGrid, "Grid");
        } else if (localZ) {
            transferSlab(finalGrid.data() + plane, localZ * plane, 0, true);
        }
    }
    int valid = 1;
    if (validate) {
        int localFinite = 1;
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        for (size_t i = 0; i < localZ * plane; ++i) {
            const Real value = finalGrid[plane + i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int finite = 0;
        Real minVal = 0, maxVal = 0;
        MPI_Reduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) {
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
