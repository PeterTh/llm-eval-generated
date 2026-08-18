#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// The local array contains a one-cell halo on every side.  Keeping x as the
// unit-stride dimension makes the innermost stencil loop contiguous.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Block {
    size_t start = 0;
    size_t extent = 0;
};

Block splitBlock(const size_t global_size, const int parts, const int coordinate) {
    const size_t base = global_size / static_cast<size_t>(parts);
    const size_t remainder = global_size % static_cast<size_t>(parts);
    const size_t before = static_cast<size_t>(coordinate);

    Block block;
    block.start = before * base + std::min(before, remainder);
    block.extent = base + (before < remainder ? 1 : 0);
    return block;
}

bool checkedProduct(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool checkedProduct(const size_t a, const size_t b, const size_t c, size_t& result) {
    size_t ab = 0;
    return checkedProduct(a, b, ab) && checkedProduct(ab, c, result);
}

void initializeGrid(std::vector<Real>& grid,
                    const size_t global_nx, const size_t global_ny,
                    const size_t global_x, const size_t global_y,
                    const size_t global_z, const size_t local_nx,
                    const size_t local_ny, const size_t local_nz) {
    const size_t local_pitch_y = local_nx + 2;
    const size_t local_plane = local_pitch_y * (local_ny + 2);
    const size_t global_plane = global_nx * global_ny;

    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t gz = global_z + z - 1;
        for (size_t y = 1; y <= local_ny; ++y) {
            const size_t gy = global_y + y - 1;
            const size_t global_row = gz * global_plane + gy * global_nx + global_x;
            const size_t local_row = z * local_plane + y * local_pitch_y + 1;
            for (size_t x = 0; x < local_nx; ++x) {
                grid[local_row + x] = static_cast<Real>((global_row + x) % 19);
            }
        }
    }
}

inline void stencilRegion(const std::vector<Real>& input,
                          std::vector<Real>& output,
                          const size_t local_nx, const size_t local_ny,
                          const size_t x_begin, const size_t x_end,
                          const size_t y_begin, const size_t y_end,
                          const size_t z_begin, const size_t z_end) {
    if (x_begin >= x_end || y_begin >= y_end || z_begin >= z_end) {
        return;
    }

    const size_t pitch_y = local_nx + 2;
    const size_t plane = pitch_y * (local_ny + 2);

    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = y_begin; y < y_end; ++y) {
            const size_t row = z * plane + y * pitch_y;
            for (size_t x = x_begin; x < x_end; ++x) {
                const size_t index = row + x;
                const Real center = input[index];
                const Real left = input[index - 1];
                const Real right = input[index + 1];
                const Real front = input[index - pitch_y];
                const Real back = input[index + pitch_y];
                const Real bottom = input[index - plane];
                const Real top = input[index + plane];
                output[index] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

struct Interval {
    size_t begin;
    size_t end;
    bool core;
};

std::array<Interval, 3> splitIntervals(const size_t begin, const size_t end,
                                        const size_t local_extent) {
    // The core is [2, local_extent), i.e. all points whose six neighbors are
    // owned by this rank.  If a block is too thin, the middle interval is
    // empty and the whole update is done after halo completion.
    const size_t core_begin = 2;
    const size_t core_end = local_extent > 2 ? local_extent : core_begin;
    return {{{begin, std::min(end, core_begin), false},
             {std::max(begin, core_begin), std::min(end, core_end), true},
             {std::max(begin, core_end), end, false}}};
}

void copyGlobalBoundaries(const std::vector<Real>& input,
                          std::vector<Real>& output,
                          const size_t nx, const size_t ny, const size_t nz,
                          const size_t global_x, const size_t global_y,
                          const size_t global_z, const size_t local_nx,
                          const size_t local_ny, const size_t local_nz) {
    const size_t pitch_y = local_nx + 2;
    const size_t plane = pitch_y * (local_ny + 2);

    // Copy only the owned global boundary surfaces.  Interior points are
    // written by the stencil, so scanning and branching on every cell here
    // would add avoidable work to every iteration.
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 1; y <= local_ny; ++y) {
            const size_t row = z * plane + y * pitch_y;
            if (global_x == 0) {
                output[row + 1] = input[row + 1];
            }
            if (global_x + local_nx == nx) {
                output[row + local_nx] = input[row + local_nx];
            }
        }
    }

    for (size_t z = 1; z <= local_nz; ++z) {
        if (global_y == 0) {
            const size_t row = z * plane + pitch_y + 1;
            std::copy_n(input.data() + row, local_nx, output.data() + row);
        }
        if (global_y + local_ny == ny) {
            const size_t row = z * plane + local_ny * pitch_y + 1;
            std::copy_n(input.data() + row, local_nx, output.data() + row);
        }
    }

    if (global_z == 0) {
        for (size_t y = 1; y <= local_ny; ++y) {
            const size_t row = plane + y * pitch_y + 1;
            std::copy_n(input.data() + row, local_nx, output.data() + row);
        }
    }
    if (global_z + local_nz == nz) {
        const size_t z_plane = local_nz * plane;
        for (size_t y = 1; y <= local_ny; ++y) {
            const size_t row = z_plane + y * pitch_y + 1;
            std::copy_n(input.data() + row, local_nx, output.data() + row);
        }
    }
}

struct HaloTypes {
    MPI_Datatype x_face = MPI_DATATYPE_NULL;
    MPI_Datatype y_face = MPI_DATATYPE_NULL;
    MPI_Datatype z_face = MPI_DATATYPE_NULL;
    MPI_Datatype x_line = MPI_DATATYPE_NULL;

    void create(const size_t local_nx, const size_t local_ny, const size_t local_nz) {
        const size_t pitch_y = local_nx + 2;
        const size_t plane = pitch_y * (local_ny + 2);

        MPI_Type_create_hvector(static_cast<int>(local_ny), 1,
                                static_cast<MPI_Aint>(pitch_y * sizeof(Real)),
                                MPI_DOUBLE, &x_line);
        MPI_Type_create_hvector(static_cast<int>(local_nz), 1,
                                static_cast<MPI_Aint>(plane * sizeof(Real)),
                                x_line, &x_face);
        MPI_Type_create_hvector(static_cast<int>(local_nz), static_cast<int>(local_nx),
                                static_cast<MPI_Aint>(plane * sizeof(Real)),
                                MPI_DOUBLE, &y_face);
        // A z face consists of local_ny contiguous x-rows separated by the
        // halo-inclusive row pitch; it is not one contiguous plane.
        MPI_Type_create_hvector(static_cast<int>(local_ny), static_cast<int>(local_nx),
                                static_cast<MPI_Aint>(pitch_y * sizeof(Real)),
                                MPI_DOUBLE, &z_face);

        MPI_Type_commit(&x_face);
        MPI_Type_commit(&y_face);
        MPI_Type_commit(&z_face);
    }

    void destroy() {
        if (x_face != MPI_DATATYPE_NULL) MPI_Type_free(&x_face);
        if (y_face != MPI_DATATYPE_NULL) MPI_Type_free(&y_face);
        if (z_face != MPI_DATATYPE_NULL) MPI_Type_free(&z_face);
        if (x_line != MPI_DATATYPE_NULL) MPI_Type_free(&x_line);
    }
};

void postHaloExchange(std::vector<Real>& grid,
                      const size_t local_nx, const size_t local_ny,
                      const size_t local_nz, const HaloTypes& types,
                      const int x_minus, const int x_plus,
                      const int y_minus, const int y_plus,
                      const int z_minus, const int z_plus,
                      MPI_Comm communicator,
                      std::array<MPI_Request, 12>& requests, int& request_count) {
    const size_t pitch_y = local_nx + 2;
    request_count = 0;

    const size_t x_low = idx3(1, 1, 1, pitch_y, local_ny + 2);
    const size_t x_high = idx3(local_nx, 1, 1, pitch_y, local_ny + 2);
    const size_t y_low = idx3(1, 1, 1, pitch_y, local_ny + 2);
    const size_t y_high = idx3(1, local_ny, 1, pitch_y, local_ny + 2);
    const size_t z_low = idx3(1, 1, 1, pitch_y, local_ny + 2);
    const size_t z_high = idx3(1, 1, local_nz, pitch_y, local_ny + 2);
    const size_t x_halo_low = idx3(0, 1, 1, pitch_y, local_ny + 2);
    const size_t x_halo_high = idx3(local_nx + 1, 1, 1, pitch_y, local_ny + 2);
    const size_t y_halo_low = idx3(1, 0, 1, pitch_y, local_ny + 2);
    const size_t y_halo_high = idx3(1, local_ny + 1, 1, pitch_y, local_ny + 2);
    const size_t z_halo_low = idx3(1, 1, 0, pitch_y, local_ny + 2);
    const size_t z_halo_high = idx3(1, 1, local_nz + 1, pitch_y, local_ny + 2);

    auto receive = [&](const size_t index, const int source, const MPI_Datatype datatype) {
        if (source != MPI_PROC_NULL) {
            MPI_Irecv(grid.data() + index, 1, datatype, source, 0, communicator,
                      &requests[request_count++]);
        }
    };
    auto send = [&](const size_t index, const int destination, const MPI_Datatype datatype) {
        if (destination != MPI_PROC_NULL) {
            MPI_Isend(grid.data() + index, 1, datatype, destination, 0, communicator,
                      &requests[request_count++]);
        }
    };

    // Post every receive first, then every send, so all six directions can
    // make progress concurrently and no direction can serialize the exchange.
    receive(x_halo_low, x_minus, types.x_face);
    receive(x_halo_high, x_plus, types.x_face);
    receive(y_halo_low, y_minus, types.y_face);
    receive(y_halo_high, y_plus, types.y_face);
    receive(z_halo_low, z_minus, types.z_face);
    receive(z_halo_high, z_plus, types.z_face);

    send(x_low, x_minus, types.x_face);
    send(x_high, x_plus, types.x_face);
    send(y_low, y_minus, types.y_face);
    send(y_high, y_plus, types.y_face);
    send(z_low, z_minus, types.z_face);
    send(z_high, z_plus, types.z_face);
}

void computeStencilIteration(std::vector<Real>& input,
                             std::vector<Real>& output,
                             const size_t nx, const size_t ny, const size_t nz,
                             const size_t global_x, const size_t global_y,
                             const size_t global_z, const size_t local_nx,
                             const size_t local_ny, const size_t local_nz,
                             const HaloTypes& halo_types,
                             const int x_minus, const int x_plus,
                             const int y_minus, const int y_plus,
                             const int z_minus, const int z_plus,
                             MPI_Comm communicator) {
    const size_t x_begin = global_x == 0 ? 2 : 1;
    const size_t x_end = global_x + local_nx == nx ? local_nx : local_nx + 1;
    const size_t y_begin = global_y == 0 ? 2 : 1;
    const size_t y_end = global_y + local_ny == ny ? local_ny : local_ny + 1;
    const size_t z_begin = global_z == 0 ? 2 : 1;
    const size_t z_end = global_z + local_nz == nz ? local_nz : local_nz + 1;

    const auto x_parts = splitIntervals(x_begin, x_end, local_nx);
    const auto y_parts = splitIntervals(y_begin, y_end, local_ny);
    const auto z_parts = splitIntervals(z_begin, z_end, local_nz);

    std::array<MPI_Request, 12> requests{};
    int request_count = 0;
    postHaloExchange(input, local_nx, local_ny, local_nz, halo_types,
                     x_minus, x_plus, y_minus, y_plus, z_minus, z_plus,
                     communicator, requests, request_count);

    // Compute the block whose stencil does not touch a halo while MPI moves
    // the six boundary faces.
    stencilRegion(input, output, local_nx, local_ny,
                  x_parts[1].begin, x_parts[1].end,
                  y_parts[1].begin, y_parts[1].end,
                  z_parts[1].begin, z_parts[1].end);

    if (request_count != 0) {
        MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE);
    }

    // Finish the shell around the core.  Splitting the three dimensions into
    // low/core/high intervals avoids a branch in the hot stencil loop and
    // handles thin subdomains without special cases.
    for (const Interval& z : z_parts) {
        for (const Interval& y : y_parts) {
            for (const Interval& x : x_parts) {
                if (x.core && y.core && z.core) {
                    continue;
                }
                stencilRegion(input, output, local_nx, local_ny,
                              x.begin, x.end, y.begin, y.end, z.begin, z.end);
            }
        }
    }

    copyGlobalBoundaries(input, output, nx, ny, nz,
                         global_x, global_y, global_z,
                         local_nx, local_ny, local_nz);
}

bool validateLocal(const std::vector<Real>& grid,
                   const size_t local_nx, const size_t local_ny,
                   const size_t local_nz, Real& local_min, Real& local_max) {
    const size_t pitch_y = local_nx + 2;
    const size_t plane = pitch_y * (local_ny + 2);
    bool valid = true;
    local_min = std::numeric_limits<Real>::max();
    local_max = std::numeric_limits<Real>::lowest();

    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 1; y <= local_ny; ++y) {
            const size_t row = z * plane + y * pitch_y;
            for (size_t x = 1; x <= local_nx; ++x) {
                const Real value = grid[row + x];
                if (std::isnan(value) || std::isinf(value)) {
                    valid = false;
                }
                local_min = std::min(local_min, value);
                local_max = std::max(local_max, value);
            }
        }
    }
    return valid;
}

void packOwned(const std::vector<Real>& grid, std::vector<Real>& packed,
               const size_t local_nx, const size_t local_ny, const size_t local_nz) {
    const size_t pitch_y = local_nx + 2;
    const size_t plane = pitch_y * (local_ny + 2);
    size_t destination = 0;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 1; y <= local_ny; ++y) {
            const size_t source = z * plane + y * pitch_y + 1;
            std::copy_n(grid.data() + source, local_nx, packed.data() + destination);
            destination += local_nx;
        }
    }
}

void unpackBlock(const std::vector<Real>& packed, const size_t packed_offset,
                 std::vector<Real>& global_grid,
                 const size_t nx, const size_t ny,
                 const size_t global_x, const size_t global_y, const size_t global_z,
                 const size_t local_nx, const size_t local_ny, const size_t local_nz) {
    size_t source = packed_offset;
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < local_ny; ++y) {
            const size_t destination = idx3(global_x, global_y + y, global_z + z, nx, ny);
            std::copy_n(packed.data() + source, local_nx, global_grid.data() + destination);
            source += local_nx;
        }
    }
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

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseFailed = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseFailed = true;
            if (world_rank == 0) {
                printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            break;
        }
    }

    if (showHelp) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (parseFailed) {
        MPI_Finalize();
        return 1;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0) {
        if (world_rank == 0) {
            printf("Grid dimensions must be at least 2 and iterations must be non-negative.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dims_raw[3] = {0, 0, 0};
    MPI_Dims_create(world_size, 3, dims_raw);
    const std::array<std::array<int, 3>, 6> permutations = {{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}
    }};
    std::array<int, 3> process_dims = {{0, 0, 0}};
    long double best_surface = std::numeric_limits<long double>::max();

    for (const auto& permutation : permutations) {
        std::array<int, 3> candidate = {{
            dims_raw[permutation[0]], dims_raw[permutation[1]], dims_raw[permutation[2]]
        }};
        if (static_cast<size_t>(candidate[0]) > nx ||
            static_cast<size_t>(candidate[1]) > ny ||
            static_cast<size_t>(candidate[2]) > nz) {
            continue;
        }

        const long double local_x = static_cast<long double>(nx) / candidate[0];
        const long double local_y = static_cast<long double>(ny) / candidate[1];
        const long double local_z = static_cast<long double>(nz) / candidate[2];
        const long double surface = local_y * local_z + local_x * local_z + local_x * local_y;
        if (surface < best_surface) {
            best_surface = surface;
            process_dims = candidate;
        }
    }

    if (process_dims[0] == 0) {
        if (world_rank == 0) {
            printf("The MPI process count cannot be mapped to non-empty blocks for this grid.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cartesian = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, process_dims.data(), periods, 1, &cartesian);

    int rank = 0;
    MPI_Comm_rank(cartesian, &rank);
    int coordinates[3] = {0, 0, 0};
    MPI_Cart_coords(cartesian, rank, 3, coordinates);

    const Block x_block = splitBlock(nx, process_dims[0], coordinates[0]);
    const Block y_block = splitBlock(ny, process_dims[1], coordinates[1]);
    const Block z_block = splitBlock(nz, process_dims[2], coordinates[2]);
    const size_t local_nx = x_block.extent;
    const size_t local_ny = y_block.extent;
    const size_t local_nz = z_block.extent;

    size_t local_xy = 0;
    size_t local_size = 0;
    size_t global_size = 0;
    size_t local_owned_xy = 0;
    const bool sizes_ok = checkedProduct(local_nx + 2, local_ny + 2, local_xy) &&
                          checkedProduct(local_xy, local_nz + 2, local_size) &&
                          checkedProduct(local_nx, local_ny, local_owned_xy) &&
                          checkedProduct(nx, ny, nz, global_size) &&
                          local_nx <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                          local_ny <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                          local_nz <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                          local_owned_xy <= static_cast<size_t>(std::numeric_limits<int>::max());
    int all_sizes_ok = 0;
    const int this_sizes_ok = sizes_ok ? 1 : 0;
    MPI_Allreduce(&this_sizes_ok, &all_sizes_ok, 1, MPI_INT, MPI_MIN, cartesian);
    if (!all_sizes_ok) {
        if (rank == 0) {
            printf("Grid is too large for the available MPI count/index limits.\n");
        }
        MPI_Comm_free(&cartesian);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", world_size);
        printf("Process grid: %d x %d x %d\n", process_dims[0], process_dims[1], process_dims[2]);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    std::vector<Real> grid1(local_size, 0.0);
    std::vector<Real> grid2(local_size, 0.0);
    initializeGrid(grid1, nx, ny,
                   x_block.start, y_block.start, z_block.start,
                   local_nx, local_ny, local_nz);

    int x_minus = MPI_PROC_NULL;
    int x_plus = MPI_PROC_NULL;
    int y_minus = MPI_PROC_NULL;
    int y_plus = MPI_PROC_NULL;
    int z_minus = MPI_PROC_NULL;
    int z_plus = MPI_PROC_NULL;
    MPI_Cart_shift(cartesian, 0, 1, &x_minus, &x_plus);
    MPI_Cart_shift(cartesian, 1, 1, &y_minus, &y_plus);
    MPI_Cart_shift(cartesian, 2, 1, &z_minus, &z_plus);

    HaloTypes halo_types;
    halo_types.create(local_nx, local_ny, local_nz);

    MPI_Barrier(cartesian);
    if (rank == 0) printf("Running stencil computation...\n");
    const double start_time = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            computeStencilIteration(grid1, grid2, nx, ny, nz,
                                    x_block.start, y_block.start, z_block.start,
                                    local_nx, local_ny, local_nz,
                                    halo_types, x_minus, x_plus, y_minus, y_plus,
                                    z_minus, z_plus, cartesian);
        } else {
            computeStencilIteration(grid2, grid1, nx, ny, nz,
                                    x_block.start, y_block.start, z_block.start,
                                    local_nx, local_ny, local_nz,
                                    halo_types, x_minus, x_plus, y_minus, y_plus,
                                    z_minus, z_plus, cartesian);
        }
    }

    const double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const long double interior_cells = static_cast<long double>(nx - 2) *
                                           static_cast<long double>(ny - 2) *
                                           static_cast<long double>(nz - 2);
        const double mcups = elapsed > 0.0
                                 ? static_cast<double>(interior_cells * iterations / elapsed / 1.0e6L)
                                 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& final_grid = (iterations & 1) == 0 ? grid1 : grid2;
    if (printResults) {
        size_t packed_size = 0;
        const bool local_gather_size_ok =
            checkedProduct(local_nx, local_ny, local_nz, packed_size) &&
            packed_size <= static_cast<size_t>(std::numeric_limits<int>::max());
        int all_gather_size_ok = 0;
        const int this_gather_size_ok = local_gather_size_ok ? 1 : 0;
        MPI_Allreduce(&this_gather_size_ok, &all_gather_size_ok, 1,
                      MPI_INT, MPI_MIN, cartesian);
        if (!all_gather_size_ok) {
            if (rank == 0) printf("Result gathering exceeds MPI count limits.\n");
            halo_types.destroy();
            MPI_Comm_free(&cartesian);
            MPI_Finalize();
            return 1;
        }

        std::vector<Real> packed(packed_size);
        packOwned(final_grid, packed, local_nx, local_ny, local_nz);

        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<Real> all_packed;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(world_size));
            displacements.resize(static_cast<size_t>(world_size));
            size_t total = 0;
            for (int source_rank = 0; source_rank < world_size; ++source_rank) {
                int source_coordinates[3] = {0, 0, 0};
                MPI_Cart_coords(cartesian, source_rank, 3, source_coordinates);
                const Block source_x = splitBlock(nx, process_dims[0], source_coordinates[0]);
                const Block source_y = splitBlock(ny, process_dims[1], source_coordinates[1]);
                const Block source_z = splitBlock(nz, process_dims[2], source_coordinates[2]);
                size_t source_count = 0;
                checkedProduct(source_x.extent, source_y.extent, source_z.extent, source_count);
                if (source_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    total > static_cast<size_t>(std::numeric_limits<int>::max()) - source_count) {
                    counts.clear();
                    break;
                }
                counts[static_cast<size_t>(source_rank)] = static_cast<int>(source_count);
                displacements[static_cast<size_t>(source_rank)] = static_cast<int>(total);
                total += source_count;
            }
            if (counts.empty()) {
                printf("Result gathering exceeds MPI count limits.\n");
            } else {
                all_packed.resize(total);
            }
        }

        const int send_count = static_cast<int>(packed.size());
        const bool gather_ok = rank != 0 || !counts.empty();
        int all_gather_ok = 0;
        const int this_gather_ok = gather_ok ? 1 : 0;
        MPI_Allreduce(&this_gather_ok, &all_gather_ok, 1, MPI_INT, MPI_MIN, cartesian);
        if (!all_gather_ok) {
            halo_types.destroy();
            MPI_Comm_free(&cartesian);
            MPI_Finalize();
            return 1;
        }

        MPI_Gatherv(packed.data(), send_count, MPI_DOUBLE,
                    rank == 0 ? all_packed.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, cartesian);

        if (rank == 0) {
            std::vector<Real> global_grid(global_size);
            for (int source_rank = 0; source_rank < world_size; ++source_rank) {
                int source_coordinates[3] = {0, 0, 0};
                MPI_Cart_coords(cartesian, source_rank, 3, source_coordinates);
                const Block source_x = splitBlock(nx, process_dims[0], source_coordinates[0]);
                const Block source_y = splitBlock(ny, process_dims[1], source_coordinates[1]);
                const Block source_z = splitBlock(nz, process_dims[2], source_coordinates[2]);
                unpackBlock(all_packed, static_cast<size_t>(displacements[static_cast<size_t>(source_rank)]),
                            global_grid, nx, ny,
                            source_x.start, source_y.start, source_z.start,
                            source_x.extent, source_y.extent, source_z.extent);
            }
            print_results(global_grid, "Grid");
        }
    }

    int validation_result = 0;
    if (validate) {
        Real local_min = 0.0;
        Real local_max = 0.0;
        const bool local_valid = validateLocal(final_grid, local_nx, local_ny, local_nz,
                                               local_min, local_max);
        const int local_valid_int = local_valid ? 1 : 0;
        int global_valid = 0;
        MPI_Allreduce(&local_valid_int, &global_valid, 1, MPI_INT, MPI_MIN, cartesian);

        Real global_min = 0.0;
        Real global_max = 0.0;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, cartesian);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian);

        if (rank == 0) {
            printf("Validating result...\n");
            if (!global_valid) {
                printf("Validation failed: found NaN or Inf value\n");
                validation_result = 1;
            } else {
                printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 1e6 || global_min < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    validation_result = 1;
                } else {
                    printf("Validation: PASSED\n");
                }
            }
        }
        MPI_Bcast(&validation_result, 1, MPI_INT, 0, cartesian);
    }

    halo_types.destroy();
    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return validation_result;
}
