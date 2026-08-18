#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

struct Decomposition {
    size_t first_z;
    size_t local_nz;
    int lower;
    int upper;
};

static Decomposition decompose(const size_t nz, const int rank, const int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t local_nz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t first_z = static_cast<size_t>(rank) * base +
                           std::min(static_cast<size_t>(rank), extra);
    const int active_ranks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    return {first_z, local_nz,
            local_nz != 0 && rank > 0 ? rank - 1 : MPI_PROC_NULL,
            local_nz != 0 && rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL};
}

static bool multiply_overflows(const size_t a, const size_t b) {
    return a != 0 && b > std::numeric_limits<size_t>::max() / a;
}

static bool parse_size(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || text[0] == '-' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

static bool parse_int(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static void initialize_grids(std::vector<Real>& grid1, std::vector<Real>& grid2,
                             const size_t nx, const size_t ny,
                             const Decomposition& d) {
    const size_t plane = nx * ny;
    for (size_t local_z = 1; local_z <= d.local_nz; ++local_z) {
        const size_t global_offset = (d.first_z + local_z - 1) * plane;
        const size_t local_offset = local_z * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real value = static_cast<Real>((global_offset + i) % 19);
            grid1[local_offset + i] = value;
            // All global boundary values are invariant.  Initializing the
            // second buffer also avoids copying them on every iteration.
            grid2[local_offset + i] = value;
        }
    }
}

static inline void update_plane(const Real* __restrict input,
                                Real* __restrict output,
                                const size_t z, const size_t nx,
                                const size_t ny, const size_t plane) {
    const size_t z_offset = z * plane;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = z_offset + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - plane] + input[i + plane]) / 7.0;
        }
    }
}

static void stencil_iteration(std::vector<Real>& input,
                              std::vector<Real>& output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const Decomposition& d, MPI_Comm comm) {
    if (d.local_nz == 0) {
        return;
    }

    const size_t plane = nx * ny;
    const int plane_count = static_cast<int>(plane);
    MPI_Request requests[4];
    int request_count = 0;

    // Receive first so eager buffering is not required.  The owned planes are
    // contiguous, so no packing or derived datatypes are needed.
    if (d.lower != MPI_PROC_NULL) {
        MPI_Irecv(input.data(), plane_count, MPI_DOUBLE,
                  d.lower, 1, comm, &requests[request_count++]);
        MPI_Isend(input.data() + plane, plane_count, MPI_DOUBLE,
                  d.lower, 0, comm, &requests[request_count++]);
    }
    if (d.upper != MPI_PROC_NULL) {
        MPI_Irecv(input.data() + (d.local_nz + 1) * plane,
                  plane_count, MPI_DOUBLE, d.upper, 0, comm,
                  &requests[request_count++]);
        MPI_Isend(input.data() + d.local_nz * plane, plane_count, MPI_DOUBLE,
                  d.upper, 1, comm, &requests[request_count++]);
    }

    // Planes not adjacent to a rank boundary overlap the halo transfer.
    for (size_t local_z = 2; local_z < d.local_nz; ++local_z) {
        const size_t global_z = d.first_z + local_z - 1;
        if (global_z != 0 && global_z + 1 < nz) {
            update_plane(input.data(), output.data(), local_z, nx, ny, plane);
        }
    }

    if (request_count != 0) {
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    }

    if (d.first_z != 0 && d.first_z + 1 < nz) {
        update_plane(input.data(), output.data(), 1, nx, ny, plane);
    }
    if (d.local_nz > 1) {
        const size_t last_global_z = d.first_z + d.local_nz - 1;
        if (last_global_z != 0 && last_global_z + 1 < nz) {
            update_plane(input.data(), output.data(), d.local_nz, nx, ny, plane);
        }
    }
}

static bool validate_result(const std::vector<Real>& grid, const size_t plane,
                            const Decomposition& d, MPI_Comm comm, const int rank) {
    int local_bad = 0;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();
    const Real* values = grid.data() + plane;
    const size_t count = d.local_nz * plane;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            local_bad = 1;
        }
        local_min = std::min(local_min, values[i]);
        local_max = std::max(local_max, values[i]);
    }

    int any_bad = 0;
    Real global_min = 0;
    Real global_max = 0;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (any_bad) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 1e6 || global_min < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return !any_bad && global_max <= 1e6 && global_min >= -1e6;
}

static void gather_grid(const std::vector<Real>& local_grid, std::vector<Real>& global_grid,
                        const size_t plane, const size_t nz, const Decomposition& d,
                        const int rank, const int ranks, MPI_Comm comm) {
    // Chunked point-to-point collection avoids MPI's int-count limit and keeps
    // rank order identical to the original global z-major array.
    constexpr size_t max_chunk = static_cast<size_t>(INT_MAX);
    if (rank == 0) {
        const size_t own_count = d.local_nz * plane;
        std::copy_n(local_grid.data() + plane, own_count, global_grid.data());
        for (int source = 1; source < ranks; ++source) {
            const Decomposition source_d = decompose(nz, source, ranks);
            size_t received = 0;
            const size_t total = source_d.local_nz * plane;
            Real* destination = global_grid.data() + source_d.first_z * plane;
            while (received < total) {
                const int chunk = static_cast<int>(std::min(max_chunk, total - received));
                MPI_Recv(destination + received, chunk, MPI_DOUBLE, source, 2, comm,
                         MPI_STATUS_IGNORE);
                received += static_cast<size_t>(chunk);
            }
        }
    } else {
        size_t sent = 0;
        const size_t total = d.local_nz * plane;
        const Real* source = local_grid.data() + plane;
        while (sent < total) {
            const int chunk = static_cast<int>(std::min(max_chunk, total - sent));
            MPI_Send(source + sent, chunk, MPI_DOUBLE, 0, 2, comm);
            sent += static_cast<size_t>(chunk);
        }
    }
}

static void print_usage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool help = false;
    bool arguments_ok = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            arguments_ok = parse_size(argv[++i], nx) && arguments_ok;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            arguments_ok = parse_size(argv[++i], ny) && arguments_ok;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            arguments_ok = parse_size(argv[++i], nz) && arguments_ok;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            arguments_ok = parse_int(argv[++i], iterations) && arguments_ok;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            arguments_ok = false;
            if (rank == 0) std::printf("Unknown or incomplete option: %s\n", argv[i]);
        }
    }

    if (help) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx < 3 || ny < 3 || nz < 3 || multiply_overflows(nx, ny) ||
        (!multiply_overflows(nx, ny) && multiply_overflows(nx * ny, nz)) ||
        nx * ny > static_cast<size_t>(INT_MAX)) {
        arguments_ok = false;
    }
    if (!arguments_ok) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions must be integers of at least 3; an XY plane must fit in an MPI count.\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const Decomposition d = decompose(nz, rank, ranks);
    if (multiply_overflows(d.local_nz + 2, plane)) {
        if (rank == 0) std::fprintf(stderr, "Local grid allocation is too large.\n");
        MPI_Abort(comm, 1);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> grid1((d.local_nz + 2) * plane);
    std::vector<Real> grid2((d.local_nz + 2) * plane);
    initialize_grids(grid1, grid2, nx, ny, d);

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if ((iteration & 1) == 0) {
            stencil_iteration(grid1, grid2, nx, ny, nz, d, comm);
        } else {
            stencil_iteration(grid2, grid1, nx, ny, nz, d, comm);
        }
    }
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const double cell_updates = static_cast<double>(nx - 2) *
                                    static_cast<double>(ny - 2) *
                                    static_cast<double>(nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    seconds > 0.0 ? cell_updates / seconds / 1e6 : 0.0);
    }

    const std::vector<Real>& final_grid = (iterations & 1) == 0 ? grid1 : grid2;
    if (print_results_requested) {
        std::vector<Real> gathered;
        if (rank == 0) gathered.resize(plane * nz);
        gather_grid(final_grid, gathered, plane, nz, d, rank, ranks, comm);
        if (rank == 0) print_results(gathered, "Grid");
    }

    int exit_code = 0;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validate_result(final_grid, plane, d, comm, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exit_code = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exit_code;
}
