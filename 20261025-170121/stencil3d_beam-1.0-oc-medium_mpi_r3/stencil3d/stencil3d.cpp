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

void initializeLocal(std::vector<Real>& grid, size_t nx, size_t ny, size_t zStart, size_t localNz, size_t nz) {
    for (size_t lz = 0; lz < localNz; ++lz) {
        size_t gz = zStart + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = idx3(x, y, lz + 1, nx, ny); // +1 halo offset
                grid[idx] = ((gz * (nx * ny) + y * nx + x) % 19) * 1.0; // hmm original idx%19 where idx global linear
            }
        }
    }
}

// original initialize uses global idx = z*(nx*ny)+y*nx+x
void initLocal(std::vector<Real>& grid, size_t nx, size_t ny, size_t zStart, size_t localNz) {
    for (size_t lz = 0; lz < localNz; ++lz) {
        size_t gz = zStart + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t gidx = gz * (nx * ny) + y * nx + x;
                grid[idx3(x, y, lz + 1, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      size_t nx, size_t ny, size_t nz,
                      int rank, int nprocs,
                      size_t zStart, size_t localNz) {    int planeSize = (int)(nx * ny);
    int prev = rank - 1;
    int next = rank + 1;

    // exchange halos
    if (prev >= 0) {
        MPI_Sendrecv(input.data() + idx3(0,0,1,nx,ny), planeSize, MPI_DOUBLE, prev, 0,
                     input.data() + idx3(0,0,0,nx,ny), planeSize, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (next < nprocs) {
        MPI_Sendrecv(input.data() + idx3(0,0,localNz,nx,ny), planeSize, MPI_DOUBLE, next, 1,
                     input.data() + idx3(0,0,localNz+1,nx,ny), planeSize, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // compute
    for (size_t lz = 0; lz < localNz; ++lz) {
        size_t gz = zStart + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t li = idx3(x, y, lz + 1, nx, ny);
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || gz == 0 || gz == nz-1) {
                    output[li] = input[li];
                } else {
                    const Real center = input[li];
                    const Real left = input[idx3(x-1, y, lz+1, nx, ny)];
                    const Real right = input[idx3(x+1, y, lz+1, nx, ny)];
                    const Real front = input[idx3(x, y-1, lz+1, nx, ny)];
                    const Real back = input[idx3(x, y+1, lz+1, nx, ny)];
                    const Real bottom = input[idx3(x, y, lz, nx, ny)];
                    const Real top = input[idx3(x, y, lz+2, nx, ny)];
                    output[li] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }
    // halo planes copy (not needed but keep)
}

bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny, size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) { printf("Validation failed: found NaN or Inf value\n"); return false; }
    }
    Real minVal = grid[0], maxVal = grid[0];
    for (const auto& val : grid) { minVal = std::min(minVal, val); maxVal = std::max(maxVal, val); }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) { printf("Validation failed: values out of expected range\n"); return false; }
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0){ printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // decomposition along z
    size_t base = nz / nprocs;
    size_t rem = nz % nprocs;
    size_t zStart = 0;
    for (int r = 0; r < rank; ++r) zStart += base + (r < (int)rem ? 1 : 0);
    size_t localNz = base + (rank < (int)rem ? 1 : 0);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t localTotal = (localNz + 2) * nx * ny;
    std::vector<Real> grid1(localTotal, 0.0), grid2(localTotal, 0.0);
    initLocal(grid1, nx, ny, zStart, localNz);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, rank, nprocs, zStart, localNz);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, rank, nprocs, zStart, localNz);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // gather
    std::vector<int> counts(nprocs), displs(nprocs);
    size_t off = 0;
    for (int r = 0; r < nprocs; ++r) {
        size_t cn = base + (r < (int)rem ? 1 : 0);
        counts[r] = (int)(cn * nx * ny);
        displs[r] = (int)(off * nx * ny);
        off += cn;
    }
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx*ny*nz);
    const std::vector<Real>& lastLocal = (iterations % 2 == 0) ? grid1 : grid2;
    // gather owned planes (skip halo): pointer to first owned plane (index 1)
    MPI_Gatherv(lastLocal.data() + idx3(0,0,1,nx,ny), (int)(localNz*nx*ny), MPI_DOUBLE,
                finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_ms);
        double cellUpdates = (double)((nx-2)*(ny-2)*(nz-2)) * iterations;
        double mcups = cellUpdates / (global_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
