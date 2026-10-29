#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Update a range of local planes. Global boundary planes and the edges of
// every plane are already present in both buffers and never change.
static void updatePlanes(const Real* in, Real* out, size_t nx, size_t ny,
                         size_t first, size_t last) {
    const size_t plane = nx * ny;
    for (size_t z = first; z < last; ++z) {
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = row + x;
                out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] +
                          in[i+nx] + in[i-plane] + in[i+plane]) / 7.0;
            }
        }
    }
}

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or MPI plane count too large\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t active = std::min(nz, static_cast<size_t>(ranks));
    const size_t r = static_cast<size_t>(rank);
    const size_t local = r < active ? nz / active + (r < nz % active) : 0;
    const size_t offset = r < active ? r * (nz / active) + std::min(r, nz % active) : nz;
    const int prev = local && rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = local && r + 1 < active ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1((local + 2) * plane), grid2((local + 2) * plane);
    for (size_t z = 0; z < local; ++z) {
        for (size_t i = 0; i < plane; ++i) {
            Real value = static_cast<Real>(((offset + z) * plane + i) % 19);
            grid1[(z + 1) * plane + i] = value;
            grid2[(z + 1) * plane + i] = value;
        }
    }
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* in = grid1.data();
    Real* out = grid2.data();
    for (int iter = 0; iter < iterations; ++iter) {
        if (local) {
            MPI_Request requests[4];
            const int count = static_cast<int>(plane);
            MPI_Irecv(in, count, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(in + (local + 1) * plane, count, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(in + plane, count, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(in + local * plane, count, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[3]);
            // Interior planes can be computed while the two halos are in flight.
            if (local > 2) updatePlanes(in, out, nx, ny, 2, local);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            if (offset > 0 && offset + 1 < nz) updatePlanes(in, out, nx, ny, 1, 2);
            if (local > 1 && offset + local - 1 > 0 && offset + local < nz)
                updatePlanes(in, out, nx, ny, local, local + 1);
        }
        std::swap(in, out);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
            static_cast<double>(ny > 2 ? ny - 2 : 0) *
            static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    if (printResults) {
        // Preserve the original serial ordering and exact reporting algorithm.
        std::vector<Real> full;
        if (rank == 0) full.resize(nx * ny * nz);
        for (int source = 0; source < static_cast<int>(active); ++source) {
            const size_t sr = static_cast<size_t>(source);
            const size_t n = (nz / active + (sr < nz % active)) * plane;
            const size_t pos = (sr * (nz / active) + std::min(sr, nz % active)) * plane;
            for (size_t done = 0; done < n;) {
                int count = static_cast<int>(std::min(n - done, static_cast<size_t>(INT_MAX)));
                if (source == 0 && rank == 0)
                    std::copy_n(in + plane + done, count, full.data() + pos + done);
                else if (rank == source)
                    MPI_Send(in + plane + done, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                else if (rank == 0)
                    MPI_Recv(full.data() + pos + done, count, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                done += count;
            }
        }
        if (rank == 0) print_results(full, "Grid");
    }
    int valid = 1;
    if (validate) {
        Real localMin = INFINITY, localMax = -INFINITY;
        int localValid = 1;
        for (size_t i = 0; i < local * plane; ++i) {
            const Real value = in[plane + i];
            if (!std::isfinite(value)) localValid = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        Real minVal, maxVal;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
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
