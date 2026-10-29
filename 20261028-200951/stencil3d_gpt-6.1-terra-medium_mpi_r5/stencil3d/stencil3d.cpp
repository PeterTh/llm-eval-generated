#include <algorithm>
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

// Compute a contiguous range of local planes. Boundary cells in both buffers
// are initialized once and remain unchanged throughout the computation.
static void compute(const Real* in, Real* out, size_t nx, size_t ny,
                    size_t begin, size_t end) {
    const size_t plane = nx * ny;
    for (size_t z = begin; z < end; ++z) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || value < (option == 'i' ? 0 : 1) ||
                value > std::numeric_limits<int>::max()) { status = 1; break; }
            if (option == 'x') nx = static_cast<size_t>(value);
            else if (option == 'y') ny = static_cast<size_t>(value);
            else if (option == 'z') nz = static_cast<size_t>(value);
            else iterations = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { status = 1; break; }
    }
    if (status) {
        if (!rank) { printf("Invalid command line arguments\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (!rank) fprintf(stderr, "Grid dimensions are too large\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const size_t active = std::min(nz, static_cast<size_t>(ranks));
    const size_t base = nz / active, extra = nz % active;
    const size_t local = static_cast<size_t>(rank) < active ? base + (static_cast<size_t>(rank) < extra) : 0;
    const size_t first = static_cast<size_t>(rank) < active ? static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra) : nz;
    const int prev = rank > 0 && local ? rank - 1 : MPI_PROC_NULL;
    const int next = local && static_cast<size_t>(rank + 1) < active ? rank + 1 : MPI_PROC_NULL;
    std::vector<Real> grid1((local + 2) * plane), grid2((local + 2) * plane);
    for (size_t z = 0; z < local; ++z)
        for (size_t i = 0; i < plane; ++i)
            grid1[(z + 1) * plane + i] = static_cast<Real>(((first + z) * plane + i) % 19);
    grid2 = grid1;
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* in = grid1.data();
    Real* out = grid2.data();
    for (int iter = 0; iter < iterations; ++iter) {
        if (local && nx > 2 && ny > 2 && nz > 2) {
            MPI_Request requests[4];
            MPI_Irecv(in, static_cast<int>(plane), MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(in + (local + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(in + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(in + local * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[3]);
            if (local > 2) compute(in, out, nx, ny, 2, local);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            if (first > 0 && first + 1 < nz) compute(in, out, nx, ny, 1, 2);
            if (local > 1 && first + local - 1 > 0 && first + local < nz)
                compute(in, out, nx, ny, local, local + 1);
        }
        std::swap(in, out);
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    if (printResults) {
        std::vector<Real> global;
        if (!rank) {
            global.resize(plane * nz);
            std::copy(in + plane, in + (local + 1) * plane, global.begin());
            for (size_t r = 1; r < active; ++r) {
                size_t count = (base + (r < extra)) * plane;
                size_t offset = (r * base + std::min(r, extra)) * plane;
                while (count) {
                    int chunk = static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Recv(global.data() + offset, chunk, MPI_DOUBLE, static_cast<int>(r), 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk; count -= chunk;
                }
            }
            print_results(global, "Grid");
        } else if (local) {
            size_t count = local * plane, offset = plane;
            while (count) {
                int chunk = static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Send(in + offset, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                offset += chunk; count -= chunk;
            }
        }
    }
    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
        int bad = 0;
        for (size_t i = plane; i < (local + 1) * plane; ++i) {
            if (!std::isfinite(in[i])) bad = 1;
            lo = std::min(lo, in[i]); hi = std::max(hi, in[i]);
        }
        Real globalLo, globalHi; int globalBad;
        MPI_Reduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!rank) {
            printf("Validating result...\n");
            if (globalBad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalLo, globalHi);
                if (globalHi > 1e6 || globalLo < -1e6) { printf("Validation failed: values out of expected range\n"); globalBad = 1; }
            }
            printf("Validation: %s\n", globalBad ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&globalBad, 1, MPI_INT, 0, MPI_COMM_WORLD);
        status = globalBad;
    }
    MPI_Finalize();
    return status;
}
