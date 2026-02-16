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

inline constexpr size_t idx_local(const size_t x, const size_t y, const size_t z_local, const size_t nx, const size_t ny) noexcept {
    return z_local * (nx * ny) + y * nx + x;
}

// Initialize local owned slices (z_local from 1..local_nz), global_z_start is the global z of local z=1
void initializeLocal(std::vector<Real>& local, const size_t nx, const size_t ny, const size_t local_nz, const size_t global_z_start, const size_t global_nx, const size_t global_ny) {
    const size_t slice = nx * ny;
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        size_t gz = global_z_start + (zl - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_idx = gz * (global_nx * global_ny) + y * global_nx + x;
                local[idx_local(x, y, zl, nx, ny)] = static_cast<Real>(global_idx % 19);
            }
        }
    }
    // For halos, initialize to 0
    if (local.size() >= slice) {
        std::fill(local.begin(), local.begin() + slice, 0.0); // z_local=0 halo
        std::fill(local.end() - slice, local.end(), 0.0); // z_local=local_nz+1 halo
    }
}

// Perform one stencil iteration on local domain (input/output include halos), respect global boundaries
void stencilIterationLocal(const std::vector<Real>& input, std::vector<Real>& output,
                           const size_t nx, const size_t ny, const size_t local_nz,
                           const size_t global_z_start, const size_t global_nz) {
    // Iterate over owned slices (1..local_nz)
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        size_t gz = global_z_start + (zl - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = idx_local(x, y, zl, nx, ny);
                // Check global boundaries
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == global_nz - 1) {
                    output[idx] = input[idx];
                } else {
                    Real center = input[idx];
                    Real left = input[idx_local(x - 1, y, zl, nx, ny)];
                    Real right = input[idx_local(x + 1, y, zl, nx, ny)];
                    Real front = input[idx_local(x, y - 1, zl, nx, ny)];
                    Real back = input[idx_local(x, y + 1, zl, nx, ny)];
                    Real bottom = input[idx_local(x, y, zl - 1, nx, ny)];
                    Real top = input[idx_local(x, y, zl + 1, nx, ny)];
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printf("Usage: %s [options]\n", argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Use at most nz ranks in z-dimension
    int used_procs = std::min(world_size, static_cast<int>(nz));
    if (world_rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI world size: %d, used procs: %d\n", world_size, used_procs);
    }

    if (world_rank >= used_procs) {
        // Unused ranks finalize early
        MPI_Finalize();
        return 0;
    }

    // Compute local z partitioning
    int base = nz / used_procs;
    int rem = nz % used_procs;
    int local_nz = base + (world_rank < rem ? 1 : 0);
    // global z start for this rank
    int z_start = world_rank * base + std::min(world_rank, rem);

    const size_t slice = nx * ny;
    // local buffer includes 2 halos: z_local in [0 .. local_nz+1]
    std::vector<Real> local1((local_nz + 2) * slice);
    std::vector<Real> local2((local_nz + 2) * slice);

    // Initialize owned region
    initializeLocal(local1, nx, ny, local_nz, z_start, nx, ny);
    // local2 can be left uninitialized

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int prev = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
    int next = (world_rank == used_procs - 1) ? MPI_PROC_NULL : world_rank + 1;

    for (int iter = 0; iter < iterations; ++iter) {
        // exchange halos for input buffer (which toggles)
        std::vector<Real>& in = (iter % 2 == 0) ? local1 : local2;
        std::vector<Real>& out = (iter % 2 == 0) ? local2 : local1;

        // send first owned slice to prev (to become their top halo), receive prev into z_local=0
        MPI_Sendrecv(in.data() + idx_local(0, 0, 1, nx, ny), slice, MPI_DOUBLE, prev, 0,
                     in.data() + idx_local(0, 0, 0, nx, ny), slice, MPI_DOUBLE, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // send last owned slice to next, receive next into z_local=local_nz+1
        MPI_Sendrecv(in.data() + idx_local(0, 0, local_nz, nx, ny), slice, MPI_DOUBLE, next, 1,
                     in.data() + idx_local(0, 0, local_nz + 1, nx, ny), slice, MPI_DOUBLE, next, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform computation on owned slices
        stencilIterationLocal(in, out, nx, ny, local_nz, z_start, nz);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double duration = t1 - t0;
    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Prepare final local pointer
    std::vector<Real>& final_local = (iterations % 2 == 0) ? local1 : local2;

    // Validation: check local for NaN/Inf and compute local min/max
    bool local_bad = false;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();
    for (size_t zl = 1; zl <= static_cast<size_t>(local_nz); ++zl) {
        for (size_t i = 0; i < slice; ++i) {
            Real v = final_local[idx_local(0, 0, zl, nx, ny) + i];
            if (std::isnan(v) || std::isinf(v)) local_bad = true;
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }
    }

    int any_bad = 0;
    Real global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (world_rank == 0) {
        if (any_bad) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        if (any_bad || global_max > 1e6 || global_min < -1e6) {
            if (any_bad) printf("Validation: FAILED\n");
        }
    }

    // Gather final full grid on rank 0 if requested
    int used = used_procs;
    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<Real> global_grid;
    int sendcount = static_cast<int>(local_nz * slice);

    if (world_rank == 0) {
        recvcounts.resize(used);
        displs.resize(used);
    }

    // compute recvcounts on root
    int local_nz_int = local_nz;
    MPI_Gather(&local_nz_int, 1, MPI_INT, recvcounts.empty() ? nullptr : recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        int offset = 0;
        for (int r = 0; r < used; ++r) {
            displs[r] = offset;
            recvcounts[r] = recvcounts[r] * slice;
            offset += recvcounts[r];
        }
        global_grid.resize(static_cast<size_t>(offset));
    }

    // send owned data (skip halos)
    Real* sendbuf = (sendcount > 0) ? &final_local[idx_local(0, 0, 1, nx, ny)] : nullptr;
    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                world_rank == 0 ? global_grid.data() : nullptr,
                world_rank == 0 ? recvcounts.data() : nullptr,
                world_rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (printResults && world_rank == 0) {
        // global_grid currently contains concatenated slabs in rank order; matches flattening by z
        print_results(global_grid, "Grid");
    }

    MPI_Finalize();
    return 0;
}
