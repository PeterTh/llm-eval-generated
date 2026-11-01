#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -x <num>     Grid size in X dimension (default: 128)\n");
                printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
                printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
                printf("  -i <num>     Number of iterations (default: 10)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Decompose z
    size_t base = nz / nprocs;
    size_t rem = nz % nprocs;
    size_t z_start = rank * base + std::min((size_t)rank, rem);
    size_t z_count = base + (rank < (int)rem ? 1 : 0);

    size_t planeSize = nx * ny;
    size_t localPlanes = z_count + 2; // +2 halos
    std::vector<Real> grid1(localPlanes * planeSize);
    std::vector<Real> grid2(localPlanes * planeSize);

    // Initialize owned core
    for (size_t lz = 0; lz < z_count; ++lz) {
        size_t gz = z_start + lz;
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = idx3(x, y, lz + 1, nx, ny); // local
                size_t gidx = gz * planeSize + y * nx + x;
                grid1[idx] = (gidx % 19) * 1.0;
            }
    }

    auto exchange = [&](std::vector<Real>& g) {
        // send first core plane to prev, recv into bottom halo
        if (rank > 0) {
            MPI_Sendrecv(g.data() + 1 * planeSize, planeSize, MPI_DOUBLE, rank - 1, 0,
                         g.data() + 0 * planeSize, planeSize, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < nprocs - 1) {
            MPI_Sendrecv(g.data() + z_count * planeSize, planeSize, MPI_DOUBLE, rank + 1, 1,
                         g.data() + (z_count + 1) * planeSize, planeSize, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    };

    auto stencil = [&](const std::vector<Real>& in, std::vector<Real>& out) {
        exchange(const_cast<std::vector<Real>&>(in));
        // interior points
        for (size_t lz = 1; lz <= z_count; ++lz) {
            size_t gz = z_start + (lz - 1);
            if (gz == 0 || gz == nz - 1) continue;
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    size_t idx = idx3(x, y, lz, nx, ny);
                    Real center = in[idx];
                    Real left = in[idx3(x-1, y, lz, nx, ny)];
                    Real right = in[idx3(x+1, y, lz, nx, ny)];
                    Real front = in[idx3(x, y-1, lz, nx, ny)];
                    Real back = in[idx3(x, y+1, lz, nx, ny)];
                    Real bottom = in[idx3(x, y, lz-1, nx, ny)];
                    Real top = in[idx3(x, y, lz+1, nx, ny)];
                    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        // copy boundary values (global boundaries or x/y boundaries)
        for (size_t lz = 1; lz <= z_count; ++lz) {
            size_t gz = z_start + (lz - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || gz == 0 || gz == nz-1) {
                        size_t idx = idx3(x, y, lz, nx, ny);
                        out[idx] = in[idx];
                    }
                }
            }
        }
    };

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) stencil(grid1, grid2);
        else stencil(grid2, grid1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_ms);
        double cellUpdates = (double)((nx-2)*(ny-2)*(nz-2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // Gather
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t bs = nz / nprocs;
        size_t rr = nz % nprocs;
        size_t zc = bs + (r < (int)rr ? 1 : 0);
        size_t zs = r * bs + std::min((size_t)r, rr);
        counts[r] = (int)(zc * planeSize);
        displs[r] = (int)(zs * planeSize);
    }
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    MPI_Gatherv(finalLocal.data() + 1 * planeSize, (int)(z_count * planeSize), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            printf("Validating result...\n");
            // inline validate
            bool valid = true;
            for (auto v : finalGrid) if (std::isnan(v) || std::isinf(v)) { valid = false; break; }
            Real mn = finalGrid[0], mx = finalGrid[0];
            for (auto v : finalGrid) { mn = std::min(mn, v); mx = std::max(mx, v); }
            printf("Value range: [%.6f, %.6f]\n", mn, mx);
            if (mx > 1e6 || mn < -1e6) valid = false;
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
