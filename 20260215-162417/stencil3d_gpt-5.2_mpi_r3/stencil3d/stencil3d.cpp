#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (row-major, x fastest)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline size_t block_z_start(const size_t nz, const int size, const int rank) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz / p;
    const size_t rem = nz % p;
    return r * base + std::min(r, rem);
}

static inline size_t block_z_count(const size_t nz, const int size, const int rank) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz / p;
    const size_t rem = nz % p;
    return base + (r < rem ? 1u : 0u);
}

static inline void initializeGridLocal(std::vector<Real>& grid_with_halo,
                                      const size_t nx, const size_t ny,
                                      const size_t local_nz, const size_t z_start,
                                      const size_t plane) {
    // Fill owned planes [1..local_nz] with the same values as the serial initialization.
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + (lz - 1);
        const size_t z_off = lz * plane;
        const size_t gz_off = gz * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t row_off = y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = gz_off + row_off + x;
                grid_with_halo[z_off + row_off + x] = (gidx % 19) * 1.0;
            }
        }
    }
}

static inline void exchangeHalosZ(const std::vector<Real>& in,
                                 const size_t nx, const size_t ny,
                                 const size_t local_nz,
                                 const int prev, const int next,
                                 MPI_Comm comm) {
    if (local_nz == 0) return;

    const size_t plane = nx * ny;
    Real* base = const_cast<Real*>(in.data()); // MPI API requires non-const buffer for some implementations.

    MPI_Request reqs[4];
    int nreq = 0;

    // Receive halos into z=0 (from prev's last plane) and z=local_nz+1 (from next's first plane).
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(base + 0 * plane, (int)plane, MPI_DOUBLE, prev, 1, comm, &reqs[nreq++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(base + (local_nz + 1) * plane, (int)plane, MPI_DOUBLE, next, 2, comm, &reqs[nreq++]);
    }

    // Send our boundary planes.
    if (next != MPI_PROC_NULL) {
        MPI_Isend(base + (local_nz) * plane, (int)plane, MPI_DOUBLE, next, 1, comm, &reqs[nreq++]);
    }
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(base + 1 * plane, (int)plane, MPI_DOUBLE, prev, 2, comm, &reqs[nreq++]);
    }

    if (nreq) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

static inline void copyBoundaryLocal(const std::vector<Real>& in,
                                    std::vector<Real>& out,
                                    const size_t nx, const size_t ny,
                                    const size_t local_nz, const size_t z_start, const size_t nz,
                                    const size_t plane) {
    if (local_nz == 0) return;

    const size_t last_x = (nx > 0) ? (nx - 1) : 0;
    const size_t last_y = (ny > 0) ? (ny - 1) : 0;

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + (lz - 1);
        const size_t zoff = lz * plane;

        if (gz == 0 || gz + 1 == nz) {
            // Global Z boundary plane: copy everything.
            std::memcpy(out.data() + zoff, in.data() + zoff, plane * sizeof(Real));
            continue;
        }

        if (nx == 0 || ny == 0) continue;

        // Copy y=0 and y=ny-1 rows.
        std::memcpy(out.data() + zoff + 0 * nx, in.data() + zoff + 0 * nx, nx * sizeof(Real));
        if (last_y != 0) {
            std::memcpy(out.data() + zoff + last_y * nx, in.data() + zoff + last_y * nx, nx * sizeof(Real));
        }

        // Copy x=0 and x=nx-1 columns for interior rows.
        if (last_x != 0 && last_y > 1) {
            for (size_t y = 1; y < last_y; ++y) {
                const size_t row = zoff + y * nx;
                out[row + 0] = in[row + 0];
                out[row + last_x] = in[row + last_x];
            }
        }
    }
}

static inline void stencilComputeRange(const std::vector<Real>& in,
                                      std::vector<Real>& out,
                                      const size_t nx, const size_t ny,
                                      const size_t lz_begin, const size_t lz_end,
                                      const size_t plane) {
    if (nx < 3 || ny < 3) return;

    for (size_t lz = lz_begin; lz < lz_end; ++lz) {
        const size_t zoff = lz * plane;
        const size_t zoff_m = (lz - 1) * plane;
        const size_t zoff_p = (lz + 1) * plane;

        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = y * nx;
            const size_t row_m = (y - 1) * nx;
            const size_t row_p = (y + 1) * nx;

            Real* __restrict outp = out.data() + zoff + row;
            const Real* __restrict inp = in.data() + zoff + row;
            const Real* __restrict inp_m = in.data() + zoff + row_m;
            const Real* __restrict inp_p = in.data() + zoff + row_p;
            const Real* __restrict inz_m = in.data() + zoff_m + row;
            const Real* __restrict inz_p = in.data() + zoff_p + row;

            for (size_t x = 1; x < nx - 1; ++x) {
                outp[x] = (inp[x] + inp[x - 1] + inp[x + 1] + inp_m[x] + inp_p[x] + inz_m[x] + inz_p[x]) / 7.0;
            }
        }
    }
}

static inline void stencilIterationLocal(const std::vector<Real>& input,
                                        std::vector<Real>& output,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const size_t local_nz, const size_t z_start,
                                        const int prev, const int next,
                                        MPI_Comm comm) {
    const size_t plane = nx * ny;
    if (local_nz == 0) return;

    // Start halo exchange.
    MPI_Request reqs[4];
    int nreq = 0;

    Real* in = const_cast<Real*>(input.data());

    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(in + 0 * plane, (int)plane, MPI_DOUBLE, prev, 1, comm, &reqs[nreq++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(in + (local_nz + 1) * plane, (int)plane, MPI_DOUBLE, next, 2, comm, &reqs[nreq++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(in + (local_nz) * plane, (int)plane, MPI_DOUBLE, next, 1, comm, &reqs[nreq++]);
    }
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(in + 1 * plane, (int)plane, MPI_DOUBLE, prev, 2, comm, &reqs[nreq++]);
    }

    // Copy global boundaries (and x/y boundaries) without needing halos.
    copyBoundaryLocal(input, output, nx, ny, local_nz, z_start, nz, plane);

    // Compute interior z-planes that do not require halo values.
    if (local_nz > 2) {
        const size_t lz0 = 2;
        const size_t lz1 = local_nz; // exclusive of local_nz

        // Clip to global compute range [1, nz-2].
        size_t begin = lz0;
        while (begin < lz1) {
            const size_t gz = z_start + (begin - 1);
            if (gz >= 1 && gz + 1 < nz) break;
            ++begin;
        }
        size_t end = lz1;
        while (end > begin) {
            const size_t gz = z_start + ((end - 1) - 1);
            if (gz >= 1 && gz + 1 < nz) break;
            --end;
        }
        if (begin < end) {
            stencilComputeRange(input, output, nx, ny, begin, end, plane);
        }
    }

    if (nreq) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Compute boundary z-planes that depend on halo values.
    // lz = 1 plane
    {
        const size_t lz = 1;
        const size_t gz = z_start;
        if (gz >= 1 && gz + 1 < nz) {
            stencilComputeRange(input, output, nx, ny, lz, lz + 1, plane);
        }
    }
    // lz = local_nz plane (if distinct)
    if (local_nz >= 2) {
        const size_t lz = local_nz;
        const size_t gz = z_start + (local_nz - 1);
        if (gz >= 1 && gz + 1 < nz) {
            stencilComputeRange(input, output, nx, ny, lz, lz + 1, plane);
        }
    }
}

static inline bool validateResultMPI(const std::vector<Real>& local_final,
                                    const size_t nx, const size_t ny,
                                    const size_t local_nz,
                                    MPI_Comm comm,
                                    int rank) {
    const size_t plane = nx * ny;
    bool local_ok = true;

    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const Real* p = local_final.data() + lz * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real v = p[i];
            if (std::isnan(v) || std::isinf(v)) {
                local_ok = false;
                break;
            }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }
        if (!local_ok) break;
    }

    int local_ok_i = local_ok ? 1 : 0;
    int global_ok_i = 0;
    MPI_Allreduce(&local_ok_i, &global_ok_i, 1, MPI_INT, MPI_LAND, comm);

    Real global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (!global_ok_i) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);

        if (global_max > 1e6 || global_min < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    return global_ok_i != 0;
}

static void printUsage(const char* progName) {
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    int validate = 0;
    int printResults = 0;

    int parse_rc = 0;
    int show_help = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                show_help = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parse_rc = 1;
                break;
            }
        }

        if (show_help) {
            printUsage(argv[0]);
        } else if (parse_rc) {
            printUsage(argv[0]);
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    MPI_Bcast(&parse_rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&show_help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (show_help || parse_rc) {
        MPI_Finalize();
        return parse_rc ? 1 : 0;
    }

    uint64_t nx64 = 0, ny64 = 0, nz64 = 0;
    if (rank == 0) {
        nx64 = (uint64_t)nx;
        ny64 = (uint64_t)ny;
        nz64 = (uint64_t)nz;
    }

    MPI_Bcast(&nx64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    nx = (size_t)nx64;
    ny = (size_t)ny64;
    nz = (size_t)nz64;

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const size_t plane = nx * ny;
    const size_t z_start = block_z_start(nz, size, rank);
    const size_t local_nz = block_z_count(nz, size, rank);
    const size_t z_end = z_start + local_nz;

    const int prev = (local_nz > 0 && z_start > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int next = (local_nz > 0 && z_end < nz) ? (rank + 1) : MPI_PROC_NULL;

    // Allocate local grids with 1-plane halo on each side in Z.
    std::vector<Real> grid1((local_nz + 2) * plane);
    std::vector<Real> grid2((local_nz + 2) * plane);

    if (local_nz > 0) {
        initializeGridLocal(grid1, nx, ny, local_nz, z_start, plane);
    }

    // Ensure halos are valid for the first iteration.
    exchangeHalosZ(grid1, nx, ny, local_nz, prev, next, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIterationLocal(grid1, grid2, nx, ny, nz, local_nz, z_start, prev, next, MPI_COMM_WORLD);
        } else {
            stencilIterationLocal(grid2, grid1, nx, ny, nz, local_nz, z_start, prev, next, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double local_dt = t1 - t0;

    double max_dt = 0.0;
    MPI_Reduce(&local_dt, &max_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)(max_dt * 1000.0));

        const double interior_x = (nx > 2) ? (double)(nx - 2) : 0.0;
        const double interior_y = (ny > 2) ? (double)(ny - 2) : 0.0;
        const double interior_z = (nz > 2) ? (double)(nz - 2) : 0.0;
        const double cellUpdates = interior_x * interior_y * interior_z * (double)iterations;
        const double mcups = (max_dt > 0.0) ? (cellUpdates / max_dt / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // Print results for external validation (gather to rank 0).
    if (printResults) {
        std::vector<int> counts(size, 0);
        std::vector<int> displs(size, 0);
        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                const size_t rs = block_z_start(nz, size, r);
                const size_t rc = block_z_count(nz, size, r);
                const size_t c = rc * plane;
                counts[r] = (int)c;
                displs[r] = (int)(rs * plane);
            }
        }

        std::vector<Real> globalGrid;
        if (rank == 0) globalGrid.resize(nx * ny * nz);

        MPI_Gatherv(local_nz ? (void*)(finalLocal.data() + 1 * plane) : nullptr,
                    (int)(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? (void*)globalGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(globalGrid, "Grid");
        }
    }

    // Validation (distributed reduction; no full gather required).
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResultMPI(finalLocal, nx, ny, local_nz, MPI_COMM_WORLD, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return (rank == 0) ? (valid ? 0 : 1) : 0;
    }

    MPI_Finalize();
    return 0;
}
