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

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of iterations (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

// Exchange arbitrarily large planes without exceeding MPI's int count limit.
static void postPlane(Real* data, size_t count, int peer, int tag, bool send,
                      std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    while (count) {
        int n = static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max())));
        requests.emplace_back();
        if (send) MPI_Isend(data, n, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &requests.back());
        else MPI_Irecv(data, n, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &requests.back());
        data += n;
        count -= n;
    }
}

static void computePlanes(const Real* in, Real* out, size_t nx, size_t ny,
                          size_t first, size_t last) {
    const size_t plane = nx * ny;
    if (nx < 3 || ny < 3) return;
    for (size_t z = first; z < last; ++z) {
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = row + x;
                out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx]
                          + in[i-plane] + in[i+plane]) / 7.0;
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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const size_t base = nz / ranks, extra = nz % ranks;
    const size_t local = base + (static_cast<size_t>(rank) < extra);
    const size_t offset = base * rank + std::min(static_cast<size_t>(rank), extra);
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const int prev = rank > 0 && local ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active && local ? rank + 1 : MPI_PROC_NULL;
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> a((local + 2) * plane), b((local + 2) * plane);
    for (size_t z = 0; z < local; ++z)
        for (size_t i = 0; i < plane; ++i)
            a[(z + 1) * plane + i] = static_cast<Real>(((offset + z) * plane + i) % 19);
    b = a; // Boundary values are invariant across iterations.
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = (iter % 2 == 0 ? a.data() : b.data());
        Real* out = (iter % 2 == 0 ? b.data() : a.data());
        requests.clear();
        if (local) {
            postPlane(in, plane, prev, 1, false, requests);
            postPlane(in + (local + 1) * plane, plane, next, 0, false, requests);
            postPlane(in + plane, plane, prev, 0, true, requests);
            postPlane(in + local * plane, plane, next, 1, true, requests);
            // Compute planes independent of incoming halos while messages travel.
            const size_t begin = std::max<size_t>(2, offset == 0 ? 2 : 1);
            const size_t end = std::min(local, nz - offset); // exclusive
            if (begin < end) computePlanes(in, out, nx, ny, begin, end);
            if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (offset > 0 && offset + 1 < nz) computePlanes(in, out, nx, ny, 1, 2);
            if (local > 1 && offset + local - 1 > 0 && offset + local < nz)
                computePlanes(in, out, nx, ny, local, local + 1);
        }
    }
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        double updates = static_cast<double>(nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    const std::vector<Real>& finalGrid = iterations % 2 == 0 ? a : b;
    if (printResults) {
        std::vector<Real> global;
        if (rank == 0) {
            global.resize(plane * nz);
            std::copy_n(finalGrid.data() + plane, local * plane, global.data());
            for (int r = 1; r < active; ++r) {
                size_t count = (base + (static_cast<size_t>(r) < extra)) * plane;
                size_t pos = (base * r + std::min(static_cast<size_t>(r), extra)) * plane;
                while (count) {
                    int n = static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Recv(global.data() + pos, n, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += n; count -= n;
                }
            }
            print_results(global, "Grid");
        } else if (local) {
            size_t count = local * plane, pos = plane;
            while (count) {
                int n = static_cast<int>(std::min(count, static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Send(finalGrid.data() + pos, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                pos += n; count -= n;
            }
        }
    }
    int status = 0;
    if (validate) {
        Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
        int bad = 0, globalBad;
        for (size_t i = plane; i < (local + 1) * plane; ++i) {
            Real v = finalGrid[i];
            if (!std::isfinite(v)) bad = 1;
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
        Real globalLo, globalHi;
        MPI_Reduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (globalBad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalLo, globalHi);
                if (globalHi > 1e6 || globalLo < -1e6) {
                    printf("Validation failed: values out of expected range\n"); globalBad = 1;
                }
            }
            printf("Validation: %s\n", globalBad ? "FAILED" : "PASSED");
            status = globalBad;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
