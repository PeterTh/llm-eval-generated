#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
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
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > INT_MAX || nz > SIZE_MAX / (nx * ny) || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count (an MPI plane must fit in an int count).\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const size_t count = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t first = (nz / ranks) * rank + std::min(static_cast<size_t>(rank), nz % ranks);
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const int lower = rank > 0 && count ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < active && count ? rank + 1 : MPI_PROC_NULL;
    std::vector<Real> a((count + 2) * plane), b((count + 2) * plane);
    for (size_t z = 0; z < count; ++z)
        for (size_t j = 0; j < plane; ++j)
            a[(z + 1) * plane + j] = static_cast<Real>(((first + z) * plane + j) % 19);
    b = a; // Both buffers retain the fixed global boundary values.
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    auto compute = [&](size_t z) {
        const size_t global_z = first + z - 1;
        if (global_z == 0 || global_z + 1 == nz || nx < 3 || ny < 3) return;
        const Real* in = a.data(); Real* out = b.data();
        for (size_t y = 1; y + 1 < ny; ++y) {
            size_t idx = z * plane + y * nx + 1;
            for (size_t x = 1; x + 1 < nx; ++x, ++idx)
                out[idx] = (in[idx] + in[idx-1] + in[idx+1] + in[idx-nx] + in[idx+nx] + in[idx-plane] + in[idx+plane]) / 7.0;
        }
    };
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[4];
        int n = 0;
        if (count) {
            MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[n++]);
            MPI_Irecv(a.data() + (count + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[n++]);
            MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[n++]);
            MPI_Isend(a.data() + count * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[n++]);
            for (size_t z = 2; z < count; ++z) compute(z);
            MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
            compute(1);
            if (count > 1) compute(count);
        }
        a.swap(b);
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        long ms = static_cast<long>(maxElapsed * 1000);
        double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms, updates / maxElapsed / 1e6);
    }
    if (printResults) {
        // Transfer in chunks so neither the total grid nor a slab is limited by MPI's int count.
        std::vector<Real> grid;
        if (rank == 0) grid.resize(nx * ny * nz);
        for (int r = 0; r < active; ++r) {
            size_t slabs = nz / ranks + (static_cast<size_t>(r) < nz % ranks);
            size_t offset = ((nz / ranks) * r + std::min(static_cast<size_t>(r), nz % ranks)) * plane;
            size_t total = slabs * plane;
            for (size_t pos = 0; pos < total;) {
                int chunk = static_cast<int>(std::min(total - pos, static_cast<size_t>(INT_MAX)));
                if (r == 0 && rank == 0) std::copy_n(a.data() + plane + pos, chunk, grid.data() + offset + pos);
                else if (rank == r) MPI_Send(a.data() + plane + pos, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                else if (rank == 0) MPI_Recv(grid.data() + offset + pos, chunk, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pos += chunk;
            }
        }
        if (rank == 0) print_results(grid, "Grid");
    }
    int valid = 1;
    if (validate) {
        Real localMin = INFINITY, localMax = -INFINITY;
        int localValid = 1;
        for (size_t i = plane; i < (count + 1) * plane; ++i) {
            Real value = a[i];
            if (!std::isfinite(value)) localValid = 0;
            localMin = std::min(localMin, value); localMax = std::max(localMax, value);
        }
        Real minVal, maxVal;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) { printf("Validation failed: values out of expected range\n"); valid = 0; }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
