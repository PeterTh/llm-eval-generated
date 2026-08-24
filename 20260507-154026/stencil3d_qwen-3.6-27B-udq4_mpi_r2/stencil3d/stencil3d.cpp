#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (local indices within a rank's subdomain)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Domain decomposition along Z dimension with halo layers
struct Domain {
    size_t nx, ny;          // local X, Y dimensions (same as global)
    size_t nz_local;        // local Z dimension (includes halo layers)
    size_t nz_interior;     // interior Z dimension (excludes halo layers)
    size_t z_start_global;  // global Z start of interior (no halo)
    size_t z_start_local;   // local Z start of interior (no halo)
    int rank;               // MPI rank
    int num_ranks;          // total MPI ranks
    size_t global_nz;       // global Z dimension
};

void computeDomain(const size_t global_nx, const size_t global_ny, const size_t global_nz,
                   const int rank, const int num_ranks, Domain& dom) {
    dom.nx = global_nx;
    dom.ny = global_ny;
    dom.global_nz = global_nz;
    dom.rank = rank;
    dom.num_ranks = num_ranks;

    // Distribute interior Z slices (exclude global boundary z=0 and z=nz-1)
    const size_t interior_nz = std::max(global_nz - 2, static_cast<size_t>(0));
    const size_t base = interior_nz / num_ranks;
    const size_t remainder = interior_nz % num_ranks;
    const size_t local_interior = base + (rank < static_cast<int>(remainder) ? 1 : 0);

    // Compute global Z start of this rank's interior
    size_t global_offset = 0;
    for (int r = 0; r < rank; ++r) {
        global_offset += base + (r < static_cast<int>(remainder) ? 1 : 0);
    }

    dom.nz_interior = local_interior;
    dom.z_start_global = global_offset + 1;  // +1 because global z=0 is boundary
    dom.z_start_local = 1;                   // local index 0 is lower halo
    dom.nz_local = local_interior + 2;       // +2 for halo layers
}

void initializeGrid(std::vector<Real>& grid, const Domain& dom, [[maybe_unused]] const size_t global_nz) {
    const size_t nx = dom.nx;
    const size_t ny = dom.ny;

    for (size_t lz = 0; lz < dom.nz_local; ++lz) {
        const size_t gz = dom.z_start_global - 1 + lz;  // map local z to global z
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = gz * (nx * ny) + y * nx + x;
                const size_t local_idx = idx3(x, y, lz, nx, ny);
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Non-blocking halo exchange: send/receive 1-cell halo layers in Z direction
void exchangeHalos(const std::vector<Real>& src, std::vector<Real>& dst, const Domain& dom) {
    const size_t slice_size = dom.nx * dom.ny;

    // Initialize dst with src
    std::copy(src.begin(), src.end(), dst.begin());

    // Lower halo (z=0): receive from rank-1, send to rank-1
    // Upper halo (z=nz_local-1): receive from rank+1, send to rank+1
    MPI_Request requests[4];
    int request_count = 0;

    // Send lower halo to rank-1 (my interior bottom = their halo top)
    if (dom.rank > 0) {
        MPI_Isend(&dst[idx3(0, 0, dom.z_start_local, dom.nx, dom.ny)],
                  static_cast<int>(slice_size), MPI_DOUBLE,
                  dom.rank - 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
    }

    // Send upper halo to rank+1 (my interior top = their halo bottom)
    if (dom.rank < dom.num_ranks - 1) {
        MPI_Isend(&dst[idx3(0, 0, dom.z_start_local + dom.nz_interior - 1, dom.nx, dom.ny)],
                  static_cast<int>(slice_size), MPI_DOUBLE,
                  dom.rank + 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
    }

    // Receive into lower halo from rank-1
    if (dom.rank > 0) {
        MPI_Irecv(&dst[idx3(0, 0, 0, dom.nx, dom.ny)],
                  static_cast<int>(slice_size), MPI_DOUBLE,
                  dom.rank - 1, 1, MPI_COMM_WORLD, &requests[request_count++]);
    }

    // Receive into upper halo from rank+1
    if (dom.rank < dom.num_ranks - 1) {
        MPI_Irecv(&dst[idx3(0, 0, dom.nz_local - 1, dom.nx, dom.ny)],
                  static_cast<int>(slice_size), MPI_DOUBLE,
                  dom.rank + 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
    }

    if (request_count > 0) {
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    }
}

// 7-point stencil computation on local subdomain with halo
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const Domain& dom) {
    const size_t nx = dom.nx;
    const size_t ny = dom.ny;

    // Process interior points (not on x/y boundaries, not on halo z boundaries)
    for (size_t lz = dom.z_start_local; lz < dom.z_start_local + dom.nz_interior; ++lz) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);

                const Real center = input[idx];
                const Real left   = input[idx - 1];
                const Real right  = input[idx + 1];
                const Real front  = input[idx - nx];
                const Real back   = input[idx + nx];
                const Real bottom = input[idx - nx * ny];
                const Real top    = input[idx + nx * ny];

                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Copy boundary values (x=0, x=nx-1, y=0, y=ny-1, and halo z layers)
    const size_t nz_local = dom.nz_local;
    for (size_t lz = 0; lz < nz_local; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                    lz == 0 || lz == nz_local - 1) {
                    const size_t idx = idx3(x, y, lz, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same)
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
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute local domain decomposition
    Domain dom;
    computeDomain(nx, ny, nz, rank, num_ranks, dom);

    const size_t local_size = nx * ny * dom.nz_local;

    // Allocate local grids (double buffering)
    std::vector<Real> grid1(local_size);
    std::vector<Real> grid2(local_size);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    initializeGrid(grid1, dom, nz);

    // Exchange initial halos so boundary values are correct
    exchangeHalos(grid1, grid1, dom);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, dom);
            exchangeHalos(grid2, grid2, dom);
        } else {
            stencilIteration(grid2, grid1, dom);
            exchangeHalos(grid1, grid1, dom);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long duration_ms = 0;

    // Aggregate timing across ranks
    MPI_Allreduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Calculate performance metrics (use global grid size)
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 for validation/results
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

   // Each rank contributes its interior slices + at most one global boundary slice.
    // Rank 0 contributes global z=0 (boundary) + its interior.
    // Last rank contributes its interior + global z=nz-1 (boundary).
    // Middle ranks contribute only their interior.
    // When rank 0 == last rank (num_ranks==1), it contributes all nz slices.
    const size_t slice_size = nx * ny;
    const bool is_first = (dom.rank == 0);
    const bool is_last  = (dom.rank == num_ranks - 1);

    int send_count;
    if (is_first && is_last) {
        send_count = static_cast<int>(slice_size * (1 + dom.nz_interior + 1));
    } else if (is_first) {
        send_count = static_cast<int>(slice_size * (1 + dom.nz_interior));
    } else if (is_last) {
        send_count = static_cast<int>(slice_size * (dom.nz_interior + 1));
    } else {
        send_count = static_cast<int>(slice_size * dom.nz_interior);
    }

    // Pack the data to send (contiguous buffer)
    std::vector<Real> sendBuf(send_count);
    int offset = 0;
    if (is_first) {
        std::copy(finalGrid.begin(), finalGrid.begin() + slice_size, sendBuf.begin() + offset);
        offset += static_cast<int>(slice_size);
    }
    // Copy interior
    std::copy(finalGrid.begin() + dom.z_start_local * slice_size,
              finalGrid.begin() + (dom.z_start_local + dom.nz_interior) * slice_size,
              sendBuf.begin() + offset);
    offset += static_cast<int>(slice_size * dom.nz_interior);
    if (is_last) {
        std::copy(finalGrid.begin() + (dom.nz_local - 1) * slice_size,
                  finalGrid.begin() + dom.nz_local * slice_size,
                  sendBuf.begin() + offset);
    }

    std::vector<Real> globalGrid;
    if (rank == 0) {
        globalGrid.resize(nx * ny * nz);
    }

    // Build receive counts and displacements
    std::vector<int> recvcounts(num_ranks);
    std::vector<int> displs(num_ranks);
    size_t disp = 0;
    for (int r = 0; r < num_ranks; ++r) {
        Domain tempDom;
        computeDomain(nx, ny, nz, r, num_ranks, tempDom);
        const bool r_first = (r == 0);
        const bool r_last  = (r == num_ranks - 1);
        int cnt;
        if (r_first && r_last) {
            cnt = static_cast<int>(slice_size * (1 + tempDom.nz_interior + 1));
        } else if (r_first) {
            cnt = static_cast<int>(slice_size * (1 + tempDom.nz_interior));
        } else if (r_last) {
            cnt = static_cast<int>(slice_size * (tempDom.nz_interior + 1));
        } else {
            cnt = static_cast<int>(slice_size * tempDom.nz_interior);
        }
        recvcounts[r] = cnt;
        displs[r] = static_cast<int>(disp);
        disp += cnt;
    }

    MPI_Gatherv(sendBuf.data(), send_count, MPI_DOUBLE,
                globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Reassemble global grid in proper order
    std::vector<Real> orderedGrid(nx * ny * nz);
    if (rank == 0) {
        size_t gpos = 0;  // global z position
        for (int r = 0; r < num_ranks; ++r) {
            Domain tempDom;
            computeDomain(nx, ny, nz, r, num_ranks, tempDom);
            const bool r_first = (r == 0);
            const bool r_last  = (r == num_ranks - 1);
            size_t src_offset = displs[r];
            if (r_first) {
                std::copy(globalGrid.begin() + src_offset,
                          globalGrid.begin() + src_offset + slice_size,
                          orderedGrid.begin() + gpos * slice_size);
                gpos++;
                src_offset += slice_size;
            }
            for (size_t i = 0; i < tempDom.nz_interior; ++i) {
                std::copy(globalGrid.begin() + src_offset,
                          globalGrid.begin() + src_offset + slice_size,
                          orderedGrid.begin() + gpos * slice_size);
                src_offset += slice_size;
                gpos++;
            }
            if (r_last) {
                std::copy(globalGrid.begin() + src_offset,
                          globalGrid.begin() + src_offset + slice_size,
                          orderedGrid.begin() + gpos * slice_size);
                gpos++;
            }
        }

        // Print results for external validation
        if (printResults) {
            print_results(orderedGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");

            // 1. No NaN or Inf values
            bool valid = true;
            for (const auto& val : orderedGrid) {
                if (std::isnan(val) || std::isinf(val)) {
                    printf("Validation failed: found NaN or Inf value\n");
                    valid = false;
                    break;
                }
            }

            if (valid) {
                // 2. Values should be reasonable (bounded)
                Real minVal = orderedGrid[0];
                Real maxVal = orderedGrid[0];
                for (const auto& val : orderedGrid) {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }

                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
