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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz,
                    const size_t z0, const size_t /*local_nz*/) {
    // grid has (local_nz+2) in z dimension with halo; interior z indices 1..local_nz correspond global z0..z0+local_nz-1
    for (size_t lz = 1; lz <= nz; ++lz) { // nz here is local_nz
        size_t gz = z0 + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny + 0); // placeholder not used
                (void)idx;
                const size_t gidx = gz * (nx * ny) + y * nx + x;
                grid[lz * (nx * ny) + y * nx + x] = (gidx % 19) * 1.0;
            }
        }
    }
}

// We'll implement inline in main for clarity.
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
            if (rank == 0) {
                printf("Usage: %s [options]\\n", argv[0]);
            }
            MPI_Finalize();
            return 0;
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

    // Partition z dimension
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t local_nz = base + (rank < (int)rem ? 1 : 0);
    size_t z0 = rank * base + (rank < (int)rem ? rank : rem);

    size_t plane = nx * ny;
    size_t local_total = plane * (local_nz + 2);
    std::vector<Real> grid1(local_total);
    std::vector<Real> grid2(local_total);

    // Initialize interior (halo rows remain 0, will be exchanged)
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        size_t gz = z0 + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t gidx = gz * plane + y * nx + x;
                grid1[lz * plane + y * nx + x] = (gidx % 19) * 1.0;
            }
        }
    }
    // boundary copy in x,y for local planes? handled in stencil.

    auto exchange = [&](std::vector<Real>& g) {
        // send bottom interior (lz=1) to prev rank halo (lz=0); send top interior (lz=local_nz) to next rank halo (lz=local_nz+1)
        if (rank > 0) {
            MPI_Sendrecv(g.data() + 1 * plane, plane, MPI_DOUBLE, rank - 1, 0,
                         g.data() + 0 * plane, plane, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(g.data() + local_nz * plane, plane, MPI_DOUBLE, rank + 1, 1,
                         g.data() + (local_nz + 1) * plane, plane, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    };

    auto stencil = [&](const std::vector<Real>& in, std::vector<Real>& out) {
        // halo exchange on input
        // We'll exchange in/out? input halos needed.
        // Make mutable copy? in is const; need exchange on in. But in may be grid1/2. We'll const_cast? Instead exchange before call.
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            size_t gz = z0 + (lz - 1);
            bool global_boundary_z = (gz == 0 || gz == nz - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t idx = lz * plane + y * nx + x;
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || global_boundary_z) {
                        out[idx] = in[idx];
                        continue;
                    }
                    // z neighbors: if lz==1 and rank>0 halo valid; if lz==local_nz and rank<size-1 halo valid; global boundaries handled by boundary copy (gz not boundary here)
                    Real center = in[idx];
                    Real left = in[lz * plane + y * nx + (x - 1)];
                    Real right = in[lz * plane + y * nx + (x + 1)];
                    Real front = in[lz * plane + (y - 1) * nx + x];
                    Real back = in[lz * plane + (y + 1) * nx + x];
                    Real bottom = in[(lz - 1) * plane + y * nx + x];
                    Real top = in[(lz + 1) * plane + y * nx + x];
                    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        // copy boundary planes? x/y boundaries already copied above; z global boundaries copied (gz==0/nz-1). 
        // halo rows in output not needed.
    };

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchange(grid1);
            stencil(grid1, grid2);
        } else {
            exchange(grid2);
            stencil(grid2, grid1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long global_ms;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (global_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    int validFlag = 1;
    // Gather to rank 0
    if (printResults || validate) {
        // gather counts
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            size_t lnz = base + (r < (int)rem ? 1 : 0);
            counts[r] = (int)(lnz * plane);
            displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
        }
        std::vector<Real> globalGrid;
        if (rank == 0) globalGrid.resize(nx * ny * nz);
        // finalLocal interior starts at plane offset 1
        std::vector<Real> sendbuf(local_nz * plane);
        for (size_t lz = 0; lz < local_nz; ++lz) {
            memcpy(sendbuf.data() + lz * plane, finalLocal.data() + (lz + 1) * plane, plane * sizeof(Real));
        }
        MPI_Gatherv(sendbuf.data(), (int)(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? globalGrid.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(globalGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\\n");
                bool valid = true;
                for (const auto& v : globalGrid) {
                    if (std::isnan(v) || std::isinf(v)) { printf("Validation failed: NaN/Inf\n"); valid = false; validFlag = 0; break; }
                }
                if (valid) {
                    Real minVal = globalGrid[0], maxVal = globalGrid[0];
                    for (const auto& v : globalGrid) { minVal = std::min(minVal, v); maxVal = std::max(maxVal, v); }
                    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                    if (maxVal > 1e6 || minVal < -1e6) { printf("Validation failed: range\n"); valid = false; validFlag = 0; }
                }
                if (valid) { printf("Validation: PASSED\n"); }
                else { printf("Validation: FAILED\n"); }
            }
        }

    }

    if (validate && printResults == false && true) {}
    if (validate) {
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    if (validate && !validFlag) return 1;
    return 0;
}
