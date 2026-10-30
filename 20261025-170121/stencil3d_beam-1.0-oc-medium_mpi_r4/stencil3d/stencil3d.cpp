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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t z0, const size_t nzl) {
    for (size_t lz = 1; lz <= nzl - 2; ++lz) {
        size_t gz = z0 + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t gidx = gz * (nx * ny) + y * nx + x;
                grid[idx3(x, y, lz, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nzl,
                      const size_t z0, const size_t nz_global,
                      int rank, int size) {
    // interior owned planes are lz=1..nzl-2
    for (size_t lz = 1; lz <= nzl - 2; ++lz) {
        size_t gz = z0 + (lz - 1);
        bool zBoundary = (gz == 0 || gz == nz_global - 1);
        for (size_t y = 0; y < ny; ++y) {
            bool yBoundary = (y == 0 || y == ny - 1);
            for (size_t x = 0; x < nx; ++x) {
                bool xBoundary = (x == 0 || x == nx - 1);
                size_t idx = idx3(x, y, lz, nx, ny);
                if (xBoundary || yBoundary || zBoundary) {
                    output[idx] = input[idx];
                } else {
                    const Real center = input[idx];
                    const Real left = input[idx3(x-1, y, lz, nx, ny)];
                    const Real right = input[idx3(x+1, y, lz, nx, ny)];
                    const Real front = input[idx3(x, y-1, lz, nx, ny)];
                    const Real back = input[idx3(x, y+1, lz, nx, ny)];
                    const Real bottom = input[idx3(x, y, lz-1, nx, ny)];
                    const Real top = input[idx3(x, y, lz+1, nx, ny)];
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }
    // copy halo planes? not needed
    // exchange halos in output for next iteration
    size_t planeSize = nx * ny;
    int up = rank - 1;
    int down = rank + 1;
    // low halo (index 0) from up's last owned plane; send our first owned plane to up
    if (up >= 0) {
        MPI_Sendrecv(&output[idx3(0,0,1,nx,ny)], (int)planeSize, MPI_DOUBLE, up, 0,
                     &output[idx3(0,0,0,nx,ny)], (int)planeSize, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (down < size) {
        MPI_Sendrecv(&output[idx3(0,0,nzl-2,nx,ny)], (int)planeSize, MPI_DOUBLE, down, 1,
                     &output[idx3(0,0,nzl-1,nx,ny)], (int)planeSize, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t /*nx*/, const size_t /*ny*/, const size_t /*nz*/) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // partition z
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t z0 = rank * base + std::min((size_t)rank, rem);
    size_t z1 = z0 + base + (rank < (int)rem ? 1 : 0);
    size_t nz_local = z1 - z0;
    size_t nzl = nz_local + 2;

    size_t gridSizeLocal = nx * ny * nzl;
    std::vector<Real> grid1(gridSizeLocal);
    std::vector<Real> grid2(gridSizeLocal);

    initializeGrid(grid1, nx, ny, z0, nzl);
    // grid2 not needed init

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nzl, z0, nz, rank, size);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nzl, z0, nz, rank, size);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGridLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // gather to rank 0
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t rb = nz / size;
        size_t rr = nz % size;
        size_t rz0 = r * rb + std::min((size_t)r, rr);
        size_t rz1 = rz0 + rb + (r < (int)rr ? 1 : 0);
        counts[r] = (int)((rz1 - rz0) * nx * ny);
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];

    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    // send owned planes (lz=1..nzl-2)
    std::vector<Real> sendbuf(nz_local * nx * ny);
    for (size_t lz = 1; lz <= nzl - 2; ++lz) {
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                sendbuf[(lz-1)*nx*ny + y*nx + x] = finalGridLocal[idx3(x,y,lz,nx,ny)];
    }
    MPI_Gatherv(sendbuf.data(), (int)sendbuf.size(), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
