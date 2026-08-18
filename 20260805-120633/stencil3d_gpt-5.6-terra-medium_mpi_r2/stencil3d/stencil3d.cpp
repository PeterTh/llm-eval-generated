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

struct Slab {
    size_t first_z;
    size_t planes;
};

static Slab slab_for_rank(const int rank, const int ranks, const size_t nz) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t planes = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t first = static_cast<size_t>(rank) * base +
                         std::min(static_cast<size_t>(rank), extra);
    return {first, planes};
}

static void initialize_grid(std::vector<Real>& grid, const Slab& slab,
                            const size_t nx, const size_t ny) {
    const size_t plane = nx * ny;
    for (size_t local_z = 0; local_z < slab.planes; ++local_z) {
        const size_t global_z = slab.first_z + local_z;
        Real* const dst = grid.data() + (local_z + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = ((global_z * plane + i) % 19) * 1.0;
        }
    }
}

static inline void update_plane(const Real* __restrict input, Real* __restrict output,
                                const size_t local_z, const size_t global_z,
                                const size_t nx, const size_t ny, const size_t nz) {
    if (global_z == 0 || global_z + 1 == nz || nx < 3 || ny < 3) return;
    const size_t plane = nx * ny;
    const size_t zoffset = local_z * plane;
    constexpr Real seventh = 1.0 / 7.0;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = zoffset + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - plane] + input[i + plane]) * seventh;
        }
    }
}

static void stencil_iteration(std::vector<Real>& input, std::vector<Real>& output,
                              const Slab& slab, const size_t nx, const size_t ny,
                              const size_t nz, const int rank, const int ranks,
                              MPI_Comm comm) {
    const size_t plane = nx * ny;
    // Copying first preserves every global boundary and the local x/y boundaries.
    std::memcpy(output.data() + plane, input.data() + plane,
                slab.planes * plane * sizeof(Real));

    MPI_Request requests[4];
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE,
              lower, 1, comm, &requests[0]);
    MPI_Irecv(input.data() + (slab.planes + 1) * plane,
              static_cast<int>(plane), MPI_DOUBLE, upper, 0, comm, &requests[1]);
    MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
              lower, 0, comm, &requests[2]);
    MPI_Isend(input.data() + slab.planes * plane, static_cast<int>(plane), MPI_DOUBLE,
              upper, 1, comm, &requests[3]);

    // These planes cannot depend on either incoming halo, so do them while messages fly.
    for (size_t local_z = 2; local_z < slab.planes; ++local_z) {
        update_plane(input.data(), output.data(), local_z, slab.first_z + local_z - 1,
                     nx, ny, nz);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    update_plane(input.data(), output.data(), 1, slab.first_z, nx, ny, nz);
    if (slab.planes > 1) {
        update_plane(input.data(), output.data(), slab.planes, slab.first_z + slab.planes - 1,
                     nx, ny, nz);
    }
}

static bool validate_result(const std::vector<Real>& grid, const Slab& slab,
                            const size_t nx, const size_t ny, MPI_Comm comm, int rank) {
    const size_t count = slab.planes * nx * ny;
    const Real* const values = grid.data() + nx * ny;
    int local_finite = 1;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) local_finite = 0;
        local_min = std::min(local_min, values[i]);
        local_max = std::max(local_max, values[i]);
    }
    int finite = 0;
    Real min_value = 0.0, max_value = 0.0;
    MPI_Allreduce(&local_finite, &finite, 1, MPI_INT, MPI_LAND, comm);
    MPI_Allreduce(&local_min, &min_value, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &max_value, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        if (!finite) printf("Validation failed: found NaN or Inf value\n");
        printf("Value range: [%.6f, %.6f]\n", min_value, max_value);
        if (max_value > 1e6 || min_value < -1e6) {
            printf("Validation failed: values out of expected range\n");
        }
    }
    return finite && max_value <= 1e6 && min_value >= -1e6;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of iterations (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, print_results_requested = false;
    int parse_ok = 1, show_help = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0) show_help = 1;
        else parse_ok = 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (show_help || !parse_ok || nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (world_rank == 0) {
            if (!parse_ok) printf("Invalid command-line option\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return (show_help && parse_ok) ? 0 : 1;
    }

    // More ranks than Z planes cannot contribute to a slab decomposition.  They remain
    // outside the work communicator but still participate in final program shutdown.
    const int active_size = std::min(world_size, static_cast<int>(nz));
    MPI_Comm work_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                   world_rank, &work_comm);
    int result = 0;
    if (world_rank < active_size) {
        int rank = 0, ranks = 1;
        MPI_Comm_rank(work_comm, &rank);
        MPI_Comm_size(work_comm, &ranks);
        const Slab slab = slab_for_rank(rank, ranks, nz);
        const size_t plane = nx * ny;
        if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) printf("Grid plane is too large for MPI count arguments\n");
            result = 1;
        } else {
            std::vector<Real> grid_a((slab.planes + 2) * plane);
            std::vector<Real> grid_b((slab.planes + 2) * plane);
            initialize_grid(grid_a, slab, nx, ny);
            if (rank == 0) {
                printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                       nx, ny, nz, iterations, validate ? "enabled" : "disabled");
                printf("MPI ranks: %d\nInitializing grid...\nRunning stencil computation...\n", ranks);
            }
            MPI_Barrier(work_comm);
            const double start = MPI_Wtime();
            for (int iter = 0; iter < iterations; ++iter) {
                stencil_iteration(grid_a, grid_b, slab, nx, ny, nz, rank, ranks, work_comm);
                grid_a.swap(grid_b);
            }
            double local_seconds = MPI_Wtime() - start, seconds = 0.0;
            MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, work_comm);
            if (rank == 0) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                const double updates = (nx > 2 && ny > 2 && nz > 2)
                    ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations : 0.0;
                printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
            }
            if (print_results_requested) {
                std::vector<int> counts, displacements;
                std::vector<Real> global;
                if (rank == 0) {
                    counts.resize(ranks); displacements.resize(ranks);
                    for (int r = 0; r < ranks; ++r) {
                        const Slab other = slab_for_rank(r, ranks, nz);
                        counts[r] = static_cast<int>(other.planes * plane);
                        displacements[r] = static_cast<int>(other.first_z * plane);
                    }
                    global.resize(nx * ny * nz);
                }
                MPI_Gatherv(grid_a.data() + plane, static_cast<int>(slab.planes * plane), MPI_DOUBLE,
                            rank == 0 ? global.data() : nullptr,
                            rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                            MPI_DOUBLE, 0, work_comm);
                if (rank == 0) print_results(global, "Grid");
            }
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                result = validate_result(grid_a, slab, nx, ny, work_comm, rank) ? 0 : 1;
                if (rank == 0) printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
            }
        }
        MPI_Comm_free(&work_comm);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
