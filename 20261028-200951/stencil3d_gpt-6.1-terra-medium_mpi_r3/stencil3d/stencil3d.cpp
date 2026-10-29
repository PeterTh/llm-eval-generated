#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <cstdint>
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
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz || nx * ny > INT_MAX || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t count = rank < active ? nz / active + (static_cast<size_t>(rank) < nz % active) : 0;
    const size_t first = rank < active ? static_cast<size_t>(rank) * (nz / active) + std::min(static_cast<size_t>(rank), nz % active) : nz;
    std::vector<Real> a((count + 2) * plane), b((count + 2) * plane);
    for (size_t z = 0; z < count; ++z)
        for (size_t p = 0; p < plane; ++p)
            a[(z + 1) * plane + p] = b[(z + 1) * plane + p] = ((first + z) * plane + p) % 19;

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    auto compute = [&](size_t z) {
        if (first + z - 1 == 0 || first + z - 1 == nz - 1 || nx < 3 || ny < 3) return;
        const size_t base = z * plane;
        const Real* in = a.data();
        Real* out = b.data();
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = base + y * nx;
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = row + x;
                out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] + in[i-plane] + in[i+plane]) / 7.0;
            }
        }
    };
    for (int iter = 0; iter < iterations; ++iter) {
        if (count) {
            MPI_Request req[4];
            int nreq = 0;
            if (rank > 0) {
                MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[nreq++]);
                MPI_Isend(a.data()+plane, static_cast<int>(plane), MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[nreq++]);
            }
            if (rank + 1 < active) {
                MPI_Irecv(a.data()+(count+1)*plane, static_cast<int>(plane), MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[nreq++]);
                MPI_Isend(a.data()+count*plane, static_cast<int>(plane), MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[nreq++]);
            }
            for (size_t z = 2; z < count; ++z) compute(z);
            MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
            compute(1);
            if (count > 1) compute(count);
            a.swap(b);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const auto ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double updates = static_cast<double>(nx > 2 ? nx-2 : 0) * (ny > 2 ? ny-2 : 0) * (nz > 2 ? nz-2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    if (printResults) {
        std::vector<Real> full;
        if (rank == 0) {
            full.resize(plane * nz);
            std::copy_n(a.data() + plane, count * plane, full.data());
            for (int r = 1; r < active; ++r) {
                const size_t n = nz / active + (static_cast<size_t>(r) < nz % active);
                const size_t offset = (static_cast<size_t>(r) * (nz / active) + std::min(static_cast<size_t>(r), nz % active)) * plane;
                size_t remaining = n * plane, pos = offset;
                while (remaining) {
                    int chunk = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(full.data()+pos, chunk, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += chunk; remaining -= chunk;
                }
            }
            print_results(full, "Grid");
        } else if (count) {
            size_t remaining = count * plane, pos = plane;
            while (remaining) {
                int chunk = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                MPI_Send(a.data()+pos, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                pos += chunk; remaining -= chunk;
            }
        }
    }
    int valid = 1;
    if (validate) {
        Real localMin = INFINITY, localMax = -INFINITY;
        int localValid = 1;
        for (size_t i = plane; i < (count+1)*plane; ++i) {
            Real v = a[i];
            if (!std::isfinite(v)) localValid = 0;
            localMin = std::min(localMin, v); localMax = std::max(localMax, v);
        }
        Real globalMin, globalMax;
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6) { printf("Validation failed: values out of expected range\n"); valid = 0; }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
