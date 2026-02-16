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

// 3D index calculation (row-major: z * (nx*ny) + y * nx + x)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

bool validateResultLocal(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            return false;
        }
    }

    Real minVal = grid.empty() ? 0.0 : grid[0];
    Real maxVal = grid.empty() ? 0.0 : grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (maxVal > 1e12 || minVal < -1e12) return false;
    return true;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Use local validator and provide similar reporting as original
    if (!validateResultLocal(grid, nx, ny, nz)) {
        printf("Validation failed: found NaN/Inf or extreme values\n");
        return false;
    }
    if (grid.empty()) {
        printf("Value range: [0.000000, 0.000000]\n");
        return true;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints help/errors)
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Domain decomposition in Z dimension (simple block distribution)
    const size_t base = nz / size;
    const size_t rem = nz % size;
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t start_z = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t end_z = (local_nz == 0) ? start_z - 1 : (start_z + local_nz - 1);

    // local storage includes 2 ghost layers in Z
    const size_t local_with_ghosts = (local_nz == 0) ? 2 : (local_nz + 2);
    const size_t localPlane = nx * ny;
    const size_t localSize = localPlane * local_with_ghosts;

    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize local grid values to match original global initialization
    // Ghost layers initialized to zero; real layers set based on global index
    if (local_nz > 0) {
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            size_t gz = start_z + (lz - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t global_idx = gz * (nx * ny) + y * nx + x;
                    const size_t local_idx = idx3(x, y, lz, nx, ny);
                    grid1[local_idx] = static_cast<Real>((global_idx % 19) * 1.0);
                }
            }
        }
        // For boundaries at global edges, set ghost layers equal to boundary values so copy semantics hold
        // Bottom ghost (z=0) if this rank owns global z=0
        if (start_z == 0) {
            // copy global z=0 plane into ghost z=0
            size_t lz = 1; // real plane for global z=0
            memcpy(&grid1[idx3(0,0,0,nx,ny)], &grid1[idx3(0,0,lz,nx,ny)], sizeof(Real)*localPlane);
        }
        // Top ghost if this rank owns global z=nz-1
        if (end_z + 1 == nz) {
            size_t lz = local_nz; // real plane for global z=nz-1
            memcpy(&grid1[idx3(0,0,local_nz+1,nx,ny)], &grid1[idx3(0,0,lz,nx,ny)], sizeof(Real)*localPlane);
        }
    }

    // timing and computation
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // helper lambda for performing one stencil iteration on local data
    auto do_local_iteration = [&](const std::vector<Real>& in, std::vector<Real>& out) {
        // Assumes halos (ghost layers) are up-to-date in in
        if (local_nz == 0) return;

        // compute interior that does not need halo planes (z=2 .. local_nz-1)
        if (local_nz >= 3) {
            for (size_t lz = 2; lz <= local_nz - 1; ++lz) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t i = idx3(x,y,lz,nx,ny);
                        const Real center = in[i];
                        const Real left = in[idx3(x-1,y,lz,nx,ny)];
                        const Real right = in[idx3(x+1,y,lz,nx,ny)];
                        const Real front = in[idx3(x,y-1,lz,nx,ny)];
                        const Real back = in[idx3(x,y+1,lz,nx,ny)];
                        const Real bottom = in[idx3(x,y,lz-1,nx,ny)];
                        const Real top = in[idx3(x,y,lz+1,nx,ny)];
                        out[i] = (center + left + right + front + back + bottom + top) / 7.0;
                    }
                }
            }
        }

        // compute planes that depend on halos (lz=1 and lz=local_nz)
        for (size_t lz : {size_t(1), local_nz}) {
            if (local_nz == 1 && lz == local_nz && lz == 1) {
                // single plane case handled once
            }
            if (lz < 1 || lz > local_nz) continue;
            size_t gz = start_z + (lz - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t i = idx3(x,y,lz,nx,ny);
                    // Boundary check: if any global coordinate is on global boundary, copy
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || gz == 0 || gz == nz-1) {
                        out[i] = in[i];
                    } else if (x >= 1 && x < nx-1 && y >= 1 && y < ny-1) {
                        const Real center = in[i];
                        const Real left = in[idx3(x-1,y,lz,nx,ny)];
                        const Real right = in[idx3(x+1,y,lz,nx,ny)];
                        const Real front = in[idx3(x,y-1,lz,nx,ny)];
                        const Real back = in[idx3(x,y+1,lz,nx,ny)];
                        const Real bottom = in[idx3(x,y,lz-1,nx,ny)];
                        const Real top = in[idx3(x,y,lz+1,nx,ny)];
                        out[i] = (center + left + right + front + back + bottom + top) / 7.0;
                    } else {
                        // edges in x/y handled by copying boundary
                        out[i] = in[i];
                    }
                }
            }
        }

        // copy ghost layers for completeness (not strictly necessary)
        // copy global Z boundaries into ghosts so semantics match original
        if (start_z == 0) {
            memcpy(&out[idx3(0,0,0,nx,ny)], &out[idx3(0,0,1,nx,ny)], sizeof(Real)*localPlane);
        }
        if (end_z + 1 == nz) {
            memcpy(&out[idx3(0,0,local_nz+1,nx,ny)], &out[idx3(0,0,local_nz,nx,ny)], sizeof(Real)*localPlane);
        }
    };

    // Prepare MPI datatypes/requests
    MPI_Request reqs[4];

    for (int iter = 0; iter < iterations; ++iter) {
        // start non-blocking halo exchange of current input
        std::vector<MPI_Request> reqs_list;
        reqs_list.reserve(4);

        // send lower real plane (lz=1) to rank-1, recv into ghost 0
        if (local_nz > 0) {
            if (rank > 0) {
                MPI_Irecv(&grid1[idx3(0,0,0,nx,ny)], (int)localPlane, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[0]);
                MPI_Isend(&grid1[idx3(0,0,1,nx,ny)], (int)localPlane, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[1]);
            } else {
                // rank 0: ghost is boundary -> copy from real plane
                if (local_nz > 0) memcpy(&grid1[idx3(0,0,0,nx,ny)], &grid1[idx3(0,0,1,nx,ny)], sizeof(Real)*localPlane);
                reqs[0] = MPI_REQUEST_NULL; reqs[1] = MPI_REQUEST_NULL;
            }

            // send upper real plane (lz=local_nz) to rank+1, recv into ghost local_nz+1
            if (rank < size - 1) {
                MPI_Irecv(&grid1[idx3(0,0,local_nz+1,nx,ny)], (int)localPlane, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[2]);
                MPI_Isend(&grid1[idx3(0,0,local_nz,nx,ny)], (int)localPlane, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[3]);
            } else {
                if (local_nz > 0) memcpy(&grid1[idx3(0,0,local_nz+1,nx,ny)], &grid1[idx3(0,0,local_nz,nx,ny)], sizeof(Real)*localPlane);
                reqs[2] = MPI_REQUEST_NULL; reqs[3] = MPI_REQUEST_NULL;
            }

            // compute interior that doesn't need halo (overlap computation)
            if (local_nz >= 3) {
                for (size_t lz = 2; lz <= local_nz - 1; ++lz) {
                    for (size_t y = 1; y < ny - 1; ++y) {
                        for (size_t x = 1; x < nx - 1; ++x) {
                            const size_t i = idx3(x,y,lz,nx,ny);
                            const Real center = grid1[i];
                            const Real left = grid1[idx3(x-1,y,lz,nx,ny)];
                            const Real right = grid1[idx3(x+1,y,lz,nx,ny)];
                            const Real front = grid1[idx3(x,y-1,lz,nx,ny)];
                            const Real back = grid1[idx3(x,y+1,lz,nx,ny)];
                            const Real bottom = grid1[idx3(x,y,lz-1,nx,ny)];
                            const Real top = grid1[idx3(x,y,lz+1,nx,ny)];
                            grid2[i] = (center + left + right + front + back + bottom + top) / 7.0;
                        }
                    }
                }
            }

            // wait for halo communication to complete
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

            // compute boundary-adjacent planes (lz=1 and lz=local_nz)
            if (local_nz > 0) {
                for (size_t lz : {size_t(1), local_nz}) {
                    if (lz < 1 || lz > local_nz) continue;
                    size_t gz = start_z + (lz - 1);
                    for (size_t y = 0; y < ny; ++y) {
                        for (size_t x = 0; x < nx; ++x) {
                            const size_t i = idx3(x,y,lz,nx,ny);
                            if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || gz == 0 || gz == nz-1) {
                                grid2[i] = grid1[i];
                            } else if (x >= 1 && x < nx-1 && y >= 1 && y < ny-1) {
                                const Real center = grid1[i];
                                const Real left = grid1[idx3(x-1,y,lz,nx,ny)];
                                const Real right = grid1[idx3(x+1,y,lz,nx,ny)];
                                const Real front = grid1[idx3(x,y-1,lz,nx,ny)];
                                const Real back = grid1[idx3(x,y+1,lz,nx,ny)];
                                const Real bottom = grid1[idx3(x,y,lz-1,nx,ny)];
                                const Real top = grid1[idx3(x,y,lz+1,nx,ny)];
                                grid2[i] = (center + left + right + front + back + bottom + top) / 7.0;
                            } else {
                                grid2[i] = grid1[i];
                            }
                        }
                    }
                }
            }

            // swap buffers for next iteration
            std::swap(grid1, grid2);
        } else {
            // local_nz == 0: nothing to do
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_duration = t_end - t_start;
    double max_duration = 0.0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // calculate performance metrics on rank 0
    if (rank == 0) {
        printf("Computation time: %.3f s\n", max_duration);
        double cellUpdates = static_cast<double>((nx > 2 && ny > 2 && nz > 2) ? (nx-2) * (ny-2) * (nz-2) : 0) * iterations;
        double mcups = (max_duration > 0.0) ? (cellUpdates / max_duration / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // determine which buffer holds the final data
    std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2; // because we swapped at end of each iter

    // Gather results to rank 0 if needed for printing/validation
    std::vector<int> recvcounts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<Real> globalGrid;
    if (rank == 0) {
        // compute recvcounts and displacements (in number of doubles)
        for (int r = 0; r < size; ++r) {
            size_t r_local_nz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            recvcounts[r] = static_cast<int>(r_local_nz * localPlane);
        }
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = offset;
            offset += recvcounts[r];
        }
        globalGrid.resize((size_t)offset);
    }

    // send only real data (exclude ghost planes)
    int sendcount = static_cast<int>(local_nz * localPlane);
    Real* sendbuf = (local_nz > 0) ? &finalLocal[idx3(0,0,1,nx,ny)] : nullptr;

    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                rank == 0 ? globalGrid.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // If printResults or validate, perform operations on rank 0
    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
                ret = 0;
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    // broadcast return code to all ranks so everyone exits consistently
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return ret;
}
