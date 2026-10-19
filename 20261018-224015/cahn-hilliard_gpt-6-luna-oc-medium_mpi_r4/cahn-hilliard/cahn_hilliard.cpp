#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

void exchangePlanes(std::vector<double>& a, size_t plane, size_t localZ, int prev, int next, int rank, int size) {
    if (prev == MPI_PROC_NULL) std::copy_n(a.begin() + static_cast<ptrdiff_t>(plane), plane, a.begin());
    if (next == MPI_PROC_NULL) std::copy_n(a.begin() + static_cast<ptrdiff_t>(localZ * plane), plane,
                                           a.begin() + static_cast<ptrdiff_t>((localZ + 1) * plane));
    MPI_Sendrecv(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                 a.data() + (localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + localZ * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                 a.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    (void)rank; (void)size;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); bad = true; }
    }
    if (help || bad) {
        if (rank == 0) printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", argv[0]);
        MPI_Finalize(); return bad ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nz < static_cast<size_t>(nranks) || nx == 0 || ny == 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive and Z must be >= MPI process count\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny;
    const size_t base = nz / nranks, rem = nz % nranks;
    const size_t localZ = base + (static_cast<size_t>(rank) < rem);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t volume = nx * ny * nz;
    std::vector<double> c((localZ + 2) * plane), nextC((localZ + 2) * plane), mu((localZ + 2) * plane);
    for (size_t lz = 1; lz <= localZ; ++lz) {
        size_t z = zStart + lz - 1;
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            size_t id = z * plane + y * nx + x;
            c[idx3(x, y, lz, nx, ny)] = -1.0 + 2.0 * ((((id + 1) * 1299709) % volume) / static_cast<double>(volume));
        }
    }
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank == nranks - 1 ? MPI_PROC_NULL : rank + 1;
    const double dt = 0.01, gamma = 0.5, D = 1.0, dx = 1.0, dy = 1.0, dz = 1.0;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        exchangePlanes(c, plane, localZ, prev, next, rank, nranks);
        for (size_t z = 1; z <= localZ; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            const size_t id = idx3(x,y,z,nx,ny);
            const size_t xp = x + (x + 1 < nx), xn = x - (x > 0), yp = y + (y + 1 < ny), yn = y - (y > 0);
            const double cv = c[id];
            const double lap = (c[idx3(xp,y,z,nx,ny)] + c[idx3(xn,y,z,nx,ny)] - 2*cv)/(dx*dx)
                + (c[idx3(x,yp,z,nx,ny)] + c[idx3(x,yn,z,nx,ny)] - 2*cv)/(dy*dy)
                + (c[idx3(x,y,z+1,nx,ny)] + c[idx3(x,y,z-1,nx,ny)] - 2*cv)/(dz*dz);
            mu[id] = 4.5*((cv+1)*(-2.0/9.0)+(cv-1)*(-2.0/9.0)-2*cv*(2.0/9.0)) + 3*cv + cv*cv*cv - gamma*lap;
        }
        exchangePlanes(mu, plane, localZ, prev, next, rank, nranks);
        for (size_t z = 1; z <= localZ; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            const size_t id = idx3(x,y,z,nx,ny);
            const size_t xp = x + (x + 1 < nx), xn = x - (x > 0), yp = y + (y + 1 < ny), yn = y - (y > 0);
            const double lap = (mu[idx3(xp,y,z,nx,ny)] + mu[idx3(xn,y,z,nx,ny)] - 2*mu[id])
                + (mu[idx3(x,yp,z,nx,ny)] + mu[idx3(x,yn,z,nx,ny)] - 2*mu[id])
                + (mu[idx3(x,y,z+1,nx,ny)] + mu[idx3(x,y,z-1,nx,ny)] - 2*mu[id]);
            nextC[id] = c[id] + dt * D * lap;
        }
        c.swap(nextC);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end-start).count(), maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        size_t rz = nz / nranks + (static_cast<size_t>(r) < nz % nranks);
        size_t begin = static_cast<size_t>(r) * (nz / nranks) + std::min(static_cast<size_t>(r), nz % nranks);
        counts[r] = static_cast<int>(rz * plane); displs[r] = static_cast<int>(begin * plane);
    }
    std::vector<double> global(rank == 0 ? volume : 0);
    std::vector<double> owned(localZ * plane);
    std::copy(c.begin() + static_cast<ptrdiff_t>(plane), c.begin() + static_cast<ptrdiff_t>((localZ+1)*plane), owned.begin());
    MPI_Gatherv(owned.data(), static_cast<int>(owned.size()), MPI_DOUBLE, global.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", (static_cast<double>(volume) * iterations / maxElapsed) / 1e6);
        if (printResults) print_results(global, "Concentration");
        if (validate) {
            double minVal = *std::min_element(global.begin(), global.end()), maxVal = *std::max_element(global.begin(), global.end());
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n", minVal, maxVal);
            bad = !std::isfinite(minVal) || !std::isfinite(maxVal) || maxVal > 10 || minVal < -10;
            printf("Validation: %s\n", bad ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&bad, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validate && bad ? 1 : 0;
}
