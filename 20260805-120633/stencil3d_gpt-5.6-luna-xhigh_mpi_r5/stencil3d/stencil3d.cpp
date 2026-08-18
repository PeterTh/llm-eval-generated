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

// 3D index calculation. The x coordinate is contiguous in memory.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Block {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t start_x;
    size_t start_y;
    size_t start_z;
};

struct Range {
    size_t begin;
    size_t end;
};

inline size_t blockExtent(const size_t global_size, const int process_count,
                          const int coordinate) noexcept {
    const size_t base = global_size / static_cast<size_t>(process_count);
    const size_t remainder = global_size % static_cast<size_t>(process_count);
    return base + (static_cast<size_t>(coordinate) < remainder ? 1 : 0);
}

inline size_t blockStart(const size_t global_size, const int process_count,
                         const int coordinate) noexcept {
    const size_t base = global_size / static_cast<size_t>(process_count);
    const size_t remainder = global_size % static_cast<size_t>(process_count);
    return static_cast<size_t>(coordinate) * base +
           std::min(static_cast<size_t>(coordinate), remainder);
}

Block makeBlock(const size_t global_nx, const size_t global_ny,
                const size_t global_nz, const int dims[3],
                const int coordinates[3]) noexcept {
    return {
        blockExtent(global_nx, dims[0], coordinates[0]),
        blockExtent(global_ny, dims[1], coordinates[1]),
        blockExtent(global_nz, dims[2], coordinates[2]),
        blockStart(global_nx, dims[0], coordinates[0]),
        blockStart(global_ny, dims[1], coordinates[1]),
        blockStart(global_nz, dims[2], coordinates[2])
    };
}

// MPI_Dims_create gives a good factorization. Try all axis permutations so
// that a short global dimension is not assigned too many processes. The
// fallback handles cases where the first factorization cannot fit the grid.
bool chooseProcessGrid(const int process_count, const size_t global_nx,
                       const size_t global_ny, const size_t global_nz,
                       int dims[3]) {
    int generated[3] = {0, 0, 0};
    if (MPI_Dims_create(process_count, 3, generated) != MPI_SUCCESS) {
        return false;
    }

    const size_t global_sizes[3] = {global_nx, global_ny, global_nz};
    const int permutations[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
        {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
    };

    long double best_score = std::numeric_limits<long double>::max();
    bool found = false;
    for (const auto& permutation : permutations) {
        const int candidate[3] = {
            generated[permutation[0]], generated[permutation[1]],
            generated[permutation[2]]
        };
        if (static_cast<size_t>(candidate[0]) > global_sizes[0] ||
            static_cast<size_t>(candidate[1]) > global_sizes[1] ||
            static_cast<size_t>(candidate[2]) > global_sizes[2]) {
            continue;
        }

        // Minimize the total area of partition cuts. This also chooses a
        // process grid whose shape follows the shape of the global grid.
        const long double score =
            static_cast<long double>(candidate[0] - 1) * global_ny * global_nz +
            static_cast<long double>(candidate[1] - 1) * global_nx * global_nz +
            static_cast<long double>(candidate[2] - 1) * global_nx * global_ny;
        if (!found || score < best_score) {
            found = true;
            best_score = score;
            dims[0] = candidate[0];
            dims[1] = candidate[1];
            dims[2] = candidate[2];
        }
    }
    if (found) {
        return true;
    }

    // Find a valid factorization when MPI_Dims_create's balanced one has a
    // factor larger than a grid dimension (for example, 32 ranks on a
    // 3 x 3 x 100 grid).
    best_score = std::numeric_limits<long double>::max();
    for (int px = 1; px <= process_count; ++px) {
        if (process_count % px != 0 || static_cast<size_t>(px) > global_nx) {
            continue;
        }
        const int remaining = process_count / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<size_t>(py) > global_ny) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<size_t>(pz) > global_nz) {
                continue;
            }
            const long double score =
                static_cast<long double>(px - 1) * global_ny * global_nz +
                static_cast<long double>(py - 1) * global_nx * global_nz +
                static_cast<long double>(pz - 1) * global_nx * global_ny;
            if (!found || score < best_score) {
                found = true;
                best_score = score;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
    return found;
}

void initializeGrid(std::vector<Real>& grid, const Block& block,
                    const size_t global_nx, const size_t global_ny) {
    const size_t local_nx = block.nx + 2;
    const size_t local_ny = block.ny + 2;

    for (size_t z = 0; z < block.nz; ++z) {
        for (size_t y = 0; y < block.ny; ++y) {
            const size_t global_row_start = idx3(
                block.start_x, block.start_y + y, block.start_z + z,
                global_nx, global_ny);
            const size_t local_row_start = idx3(1, y + 1, z + 1,
                                                local_nx, local_ny);
            for (size_t x = 0; x < block.nx; ++x) {
                grid[local_row_start + x] =
                    static_cast<Real>((global_row_start + x) % 19);
            }
        }
    }
}

// Compute a rectangular region of owned cells. The one-cell halo is included
// in the local allocation, so all neighbor accesses are unit-stride or use a
// fixed row/plane stride.
void stencilRegion(const std::vector<Real>& input, std::vector<Real>& output,
                   const size_t local_nx, const size_t local_ny,
                   const Range x_range, const Range y_range,
                   const Range z_range) {
    if (x_range.begin > x_range.end || y_range.begin > y_range.end ||
        z_range.begin > z_range.end) {
        return;
    }

    const size_t row_stride = local_nx + 2;
    const size_t plane_stride = row_stride * (local_ny + 2);

    for (size_t z = z_range.begin; z <= z_range.end; ++z) {
        for (size_t y = y_range.begin; y <= y_range.end; ++y) {
            size_t index = z * plane_stride + y * row_stride + x_range.begin;
            for (size_t x = x_range.begin; x <= x_range.end; ++x, ++index) {
                output[index] =
                    (input[index] + input[index - 1] + input[index + 1] +
                     input[index - row_stride] + input[index + row_stride] +
                     input[index - plane_stride] + input[index + plane_stride]) /
                    7.0;
            }
        }
    }
}

Range globalInteriorRange(const size_t local_size, const size_t global_start,
                          const size_t global_size) noexcept {
    const size_t begin = global_start == 0 ? 2 : 1;
    const size_t end = global_start + local_size == global_size
                           ? local_size - 1
                           : local_size;
    return {begin, end};
}

void copyPhysicalBoundaries(const std::vector<Real>& input,
                            std::vector<Real>& output, const Block& block,
                            const size_t global_nx, const size_t global_ny,
                            const size_t global_nz) {
    const size_t local_nx = block.nx + 2;
    const size_t local_ny = block.ny + 2;

    if (block.start_x == 0) {
        for (size_t z = 1; z <= block.nz; ++z) {
            for (size_t y = 1; y <= block.ny; ++y) {
                const size_t index = idx3(1, y, z, local_nx, local_ny);
                output[index] = input[index];
            }
        }
    }
    if (block.start_x + block.nx == global_nx) {
        for (size_t z = 1; z <= block.nz; ++z) {
            for (size_t y = 1; y <= block.ny; ++y) {
                const size_t index = idx3(block.nx, y, z, local_nx, local_ny);
                output[index] = input[index];
            }
        }
    }
    if (block.start_y == 0) {
        for (size_t z = 1; z <= block.nz; ++z) {
            const size_t index = idx3(1, 1, z, local_nx, local_ny);
            std::copy_n(input.data() + index, block.nx, output.data() + index);
        }
    }
    if (block.start_y + block.ny == global_ny) {
        for (size_t z = 1; z <= block.nz; ++z) {
            const size_t index = idx3(1, block.ny, z, local_nx, local_ny);
            std::copy_n(input.data() + index, block.nx, output.data() + index);
        }
    }
    if (block.start_z == 0) {
        for (size_t y = 1; y <= block.ny; ++y) {
            const size_t index = idx3(1, y, 1, local_nx, local_ny);
            std::copy_n(input.data() + index, block.nx, output.data() + index);
        }
    }
    if (block.start_z + block.nz == global_nz) {
        for (size_t y = 1; y <= block.ny; ++y) {
            const size_t index = idx3(1, y, block.nz, local_nx, local_ny);
            std::copy_n(input.data() + index, block.nx, output.data() + index);
        }
    }
}

struct FaceTypes {
    MPI_Datatype x;
    MPI_Datatype y;
    MPI_Datatype z;
};

FaceTypes createFaceTypes(const Block& block) {
    const int sizes[3] = {
        static_cast<int>(block.nz + 2), static_cast<int>(block.ny + 2),
        static_cast<int>(block.nx + 2)
    };
    const int starts[3] = {0, 0, 0};
    const int x_subsizes[3] = {
        static_cast<int>(block.nz), static_cast<int>(block.ny), 1
    };
    const int y_subsizes[3] = {
        static_cast<int>(block.nz), 1, static_cast<int>(block.nx)
    };
    const int z_subsizes[3] = {
        1, static_cast<int>(block.ny), static_cast<int>(block.nx)
    };

    FaceTypes types = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                       MPI_DATATYPE_NULL};
    MPI_Type_create_subarray(3, sizes, x_subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &types.x);
    MPI_Type_create_subarray(3, sizes, y_subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &types.y);
    MPI_Type_create_subarray(3, sizes, z_subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &types.z);
    MPI_Type_commit(&types.x);
    MPI_Type_commit(&types.y);
    MPI_Type_commit(&types.z);
    return types;
}

void destroyFaceTypes(FaceTypes& types) {
    MPI_Type_free(&types.x);
    MPI_Type_free(&types.y);
    MPI_Type_free(&types.z);
}

enum HaloTag : int {
    X_MINUS_TAG = 0,
    X_PLUS_TAG = 1,
    Y_MINUS_TAG = 2,
    Y_PLUS_TAG = 3,
    Z_MINUS_TAG = 4,
    Z_PLUS_TAG = 5
};

struct HaloExchange {
    MPI_Request requests[12];
    int request_count;
};

HaloExchange beginHaloExchange(std::vector<Real>& grid, const Block& block,
                               const int x_minus, const int x_plus,
                               const int y_minus, const int y_plus,
                               const int z_minus, const int z_plus,
                               MPI_Comm communicator, const FaceTypes& types) {
    const size_t local_nx = block.nx + 2;
    const size_t local_ny = block.ny + 2;
    auto cell = [&](const size_t x, const size_t y, const size_t z) {
        return grid.data() + idx3(x, y, z, local_nx, local_ny);
    };

    HaloExchange exchange = {{}, 0};
    if (x_minus != MPI_PROC_NULL) {
        MPI_Irecv(cell(0, 1, 1), 1, types.x, x_minus, X_PLUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (x_plus != MPI_PROC_NULL) {
        MPI_Irecv(cell(block.nx + 1, 1, 1), 1, types.x, x_plus,
                  X_MINUS_TAG, communicator,
                  &exchange.requests[exchange.request_count++]);
    }
    if (y_minus != MPI_PROC_NULL) {
        MPI_Irecv(cell(1, 0, 1), 1, types.y, y_minus, Y_PLUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (y_plus != MPI_PROC_NULL) {
        MPI_Irecv(cell(1, block.ny + 1, 1), 1, types.y, y_plus,
                  Y_MINUS_TAG, communicator,
                  &exchange.requests[exchange.request_count++]);
    }
    if (z_minus != MPI_PROC_NULL) {
        MPI_Irecv(cell(1, 1, 0), 1, types.z, z_minus, Z_PLUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (z_plus != MPI_PROC_NULL) {
        MPI_Irecv(cell(1, 1, block.nz + 1), 1, types.z, z_plus,
                  Z_MINUS_TAG, communicator,
                  &exchange.requests[exchange.request_count++]);
    }

    if (x_minus != MPI_PROC_NULL) {
        MPI_Isend(cell(1, 1, 1), 1, types.x, x_minus, X_MINUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (x_plus != MPI_PROC_NULL) {
        MPI_Isend(cell(block.nx, 1, 1), 1, types.x, x_plus, X_PLUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (y_minus != MPI_PROC_NULL) {
        MPI_Isend(cell(1, 1, 1), 1, types.y, y_minus, Y_MINUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (y_plus != MPI_PROC_NULL) {
        MPI_Isend(cell(1, block.ny, 1), 1, types.y, y_plus, Y_PLUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (z_minus != MPI_PROC_NULL) {
        MPI_Isend(cell(1, 1, 1), 1, types.z, z_minus, Z_MINUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }
    if (z_plus != MPI_PROC_NULL) {
        MPI_Isend(cell(1, 1, block.nz), 1, types.z, z_plus, Z_PLUS_TAG,
                  communicator, &exchange.requests[exchange.request_count++]);
    }

    return exchange;
}

void finishHaloExchange(HaloExchange& exchange) {
    if (exchange.request_count > 0) {
        MPI_Waitall(exchange.request_count, exchange.requests,
                    MPI_STATUSES_IGNORE);
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const Block& block, const size_t global_nx,
                      const size_t global_ny, const size_t global_nz,
                      const int neighbors[6], MPI_Comm communicator,
                      const FaceTypes& face_types) {
    const Range x_interior = globalInteriorRange(block.nx, block.start_x,
                                                  global_nx);
    const Range y_interior = globalInteriorRange(block.ny, block.start_y,
                                                  global_ny);
    const Range z_interior = globalInteriorRange(block.nz, block.start_z,
                                                  global_nz);
    const Range x_core = {2, block.nx >= 3 ? block.nx - 1 : 0};
    const Range y_core = {2, block.ny >= 3 ? block.ny - 1 : 0};
    const Range z_core = {2, block.nz >= 3 ? block.nz - 1 : 0};

    HaloExchange halo = beginHaloExchange(
        input, block, neighbors[0], neighbors[1], neighbors[2], neighbors[3],
        neighbors[4], neighbors[5], communicator, face_types);

    // The bulk does not read any halo cells and can execute while the six
    // nonblocking face exchanges make progress.
    stencilRegion(input, output, block.nx, block.ny, x_core, y_core, z_core);

    finishHaloExchange(halo);

    // x low/high fringes, including all y/z interior cells.
    if (x_interior.begin == 1) {
        stencilRegion(input, output, block.nx, block.ny, {1, 1}, y_interior,
                      z_interior);
    }
    if (x_interior.end == block.nx &&
        !(x_interior.begin == 1 && block.nx == 1)) {
        stencilRegion(input, output, block.nx, block.ny,
                      {block.nx, block.nx}, y_interior, z_interior);
    }

    // y fringes for x-core cells.
    if (y_interior.begin == 1) {
        stencilRegion(input, output, block.nx, block.ny, x_core, {1, 1},
                      z_interior);
    }
    if (y_interior.end == block.ny &&
        !(y_interior.begin == 1 && block.ny == 1)) {
        stencilRegion(input, output, block.nx, block.ny, x_core,
                      {block.ny, block.ny}, z_interior);
    }

    // z fringes for x/y-core cells.
    if (z_interior.begin == 1) {
        stencilRegion(input, output, block.nx, block.ny, x_core, y_core,
                      {1, 1});
    }
    if (z_interior.end == block.nz &&
        !(z_interior.begin == 1 && block.nz == 1)) {
        stencilRegion(input, output, block.nx, block.ny, x_core, y_core,
                      {block.nz, block.nz});
    }

    copyPhysicalBoundaries(input, output, block, global_nx, global_ny,
                           global_nz);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

bool validateDistributed(const std::vector<Real>& grid, const Block& block,
                         MPI_Comm communicator, const int rank) {
    const size_t local_nx = block.nx + 2;
    const size_t local_ny = block.ny + 2;
    bool local_finite = true;
    Real local_min = std::numeric_limits<Real>::max();
    Real local_max = std::numeric_limits<Real>::lowest();

    for (size_t z = 1; z <= block.nz; ++z) {
        for (size_t y = 1; y <= block.ny; ++y) {
            const size_t row = idx3(1, y, z, local_nx, local_ny);
            for (size_t x = 0; x < block.nx; ++x) {
                const Real value = grid[row + x];
                if (std::isnan(value) || std::isinf(value)) {
                    local_finite = false;
                } else {
                    local_min = std::min(local_min, value);
                    local_max = std::max(local_max, value);
                }
            }
        }
    }

    const int local_bad = local_finite ? 0 : 1;
    int global_bad = 0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_SUM,
                  communicator);

    Real global_min = 0.0;
    Real global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                  communicator);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  communicator);

    if (rank == 0) {
        if (global_bad != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        std::printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 1e6 || global_min < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            return false;
        }
    }
    return global_bad == 0 && global_max <= 1e6 && global_min >= -1e6;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Pack only owned cells for the optional root-side result reconstruction.
std::vector<Real> packOwned(const std::vector<Real>& grid, const Block& block) {
    const size_t local_nx = block.nx + 2;
    const size_t local_ny = block.ny + 2;
    std::vector<Real> packed(block.nx * block.ny * block.nz);
    size_t destination = 0;
    for (size_t z = 1; z <= block.nz; ++z) {
        for (size_t y = 1; y <= block.ny; ++y) {
            const size_t source = idx3(1, y, z, local_nx, local_ny);
            std::copy_n(grid.data() + source, block.nx,
                        packed.data() + destination);
            destination += block.nx;
        }
    }
    return packed;
}

bool gatherGlobalGrid(const std::vector<Real>& local_grid, const Block& block,
                      const size_t global_nx, const size_t global_ny,
                      const size_t global_nz, const int dims[3],
                      MPI_Comm communicator, const int rank, const int size,
                      std::vector<Real>& global_grid) {
    const std::vector<Real> packed = packOwned(local_grid, block);
    int local_count = 0;
    bool gatherable = packed.size() <=
                      static_cast<size_t>(std::numeric_limits<int>::max());
    std::vector<int> receive_counts;
    std::vector<int> displacements;
    std::vector<Real> gathered_data;

    if (rank == 0) {
        receive_counts.resize(static_cast<size_t>(size));
        displacements.resize(static_cast<size_t>(size));
        size_t displacement = 0;
        for (int process = 0; process < size; ++process) {
            int coordinates[3] = {0, 0, 0};
            MPI_Cart_coords(communicator, process, 3, coordinates);
            const Block process_block = makeBlock(
                global_nx, global_ny, global_nz, dims, coordinates);
            const size_t count = process_block.nx * process_block.ny *
                                 process_block.nz;
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                displacement > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                count > static_cast<size_t>(std::numeric_limits<int>::max()) -
                            displacement) {
                gatherable = false;
                break;
            }
            receive_counts[static_cast<size_t>(process)] =
                static_cast<int>(count);
            displacements[static_cast<size_t>(process)] =
                static_cast<int>(displacement);
            displacement += count;
        }
        if (gatherable) {
            gathered_data.resize(global_nx * global_ny * global_nz);
            global_grid.resize(global_nx * global_ny * global_nz);
        }
    }

    if (gatherable) {
        local_count = static_cast<int>(packed.size());
    }
    int all_gatherable = gatherable ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_gatherable, 1, MPI_INT, MPI_MIN,
                  communicator);
    if (all_gatherable == 0) {
        if (rank == 0) {
            std::printf("Result output is too large for MPI_Gatherv counts\n");
        }
        return false;
    }

    MPI_Gatherv(packed.data(), local_count, MPI_DOUBLE,
                rank == 0 ? gathered_data.data() : nullptr,
                rank == 0 ? receive_counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                communicator);

    if (rank == 0) {
        for (int process = 0; process < size; ++process) {
            int coordinates[3] = {0, 0, 0};
            MPI_Cart_coords(communicator, process, 3, coordinates);
            const Block process_block = makeBlock(
                global_nx, global_ny, global_nz, dims, coordinates);
            const Real* source = gathered_data.data() +
                                 displacements[static_cast<size_t>(process)];
            size_t source_offset = 0;
            for (size_t z = 0; z < process_block.nz; ++z) {
                for (size_t y = 0; y < process_block.ny; ++y) {
                    const size_t destination = idx3(
                        process_block.start_x,
                        process_block.start_y + y,
                        process_block.start_z + z, global_nx, global_ny);
                    std::copy_n(source + source_offset, process_block.nx,
                                global_grid.begin() +
                                    static_cast<std::ptrdiff_t>(destination));
                    source_offset += process_block.nx;
                }
            }
        }
    }
    return true;
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

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        if (world_rank == 0) {
            std::printf("Grid dimensions must be at least 3 and iterations "
                        "must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {0, 0, 0};
    if (!chooseProcessGrid(world_size, nx, ny, nz, dims)) {
        if (world_rank == 0) {
            std::printf("Cannot distribute %zu x %zu x %zu over %d MPI "
                        "processes\n", nx, ny, nz, world_size);
        }
        MPI_Finalize();
        return 1;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Comm cartesian;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cartesian);

    int rank = 0;
    MPI_Comm_rank(cartesian, &rank);
    int coordinates[3] = {0, 0, 0};
    MPI_Cart_coords(cartesian, rank, 3, coordinates);
    const Block block = makeBlock(nx, ny, nz, dims, coordinates);

    const bool local_mpi_sizes_valid =
        block.nx <= static_cast<size_t>(std::numeric_limits<int>::max() - 2) &&
        block.ny <= static_cast<size_t>(std::numeric_limits<int>::max() - 2) &&
        block.nz <= static_cast<size_t>(std::numeric_limits<int>::max() - 2);
    int mpi_sizes_valid = local_mpi_sizes_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &mpi_sizes_valid, 1, MPI_INT, MPI_MIN,
                  cartesian);
    if (mpi_sizes_valid == 0) {
        if (rank == 0) {
            std::printf("A local grid dimension exceeds MPI datatype limits\n");
        }
        MPI_Comm_free(&cartesian);
        MPI_Finalize();
        return 1;
    }

    const size_t local_nx_with_halo = block.nx + 2;
    const size_t local_ny_with_halo = block.ny + 2;
    const size_t local_nz_with_halo = block.nz + 2;
    const size_t local_grid_size = local_nx_with_halo * local_ny_with_halo *
                                   local_nz_with_halo;
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", world_size);
        std::printf("MPI process grid: %d x %d x %d\n", dims[0], dims[1],
                    dims[2]);
        std::printf("Initializing grid...\n");
    }
    initializeGrid(grid1, block, nx, ny);

    int x_minus = MPI_PROC_NULL;
    int x_plus = MPI_PROC_NULL;
    int y_minus = MPI_PROC_NULL;
    int y_plus = MPI_PROC_NULL;
    int z_minus = MPI_PROC_NULL;
    int z_plus = MPI_PROC_NULL;
    MPI_Cart_shift(cartesian, 0, 1, &x_minus, &x_plus);
    MPI_Cart_shift(cartesian, 1, 1, &y_minus, &y_plus);
    MPI_Cart_shift(cartesian, 2, 1, &z_minus, &z_plus);
    const int neighbors[6] = {x_minus, x_plus, y_minus,
                               y_plus, z_minus, z_plus};
    FaceTypes face_types = createFaceTypes(block);

    if (rank == 0) {
        std::printf("Running stencil computation...\n");
    }
    MPI_Barrier(cartesian);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIteration(grid1, grid2, block, nx, ny, nz, neighbors,
                              cartesian, face_types);
        } else {
            stencilIteration(grid2, grid1, block, nx, ny, nz, neighbors,
                              cartesian, face_types);
        }
    }
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               cartesian);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", duration_ms);
        const double cell_updates =
            static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
            static_cast<double>(nz - 2) * static_cast<double>(iterations);
        const double mcups = elapsed > 0.0
                                 ? cell_updates / elapsed / 1e6
                                 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& local_final_grid =
        (iterations & 1) == 0 ? grid1 : grid2;
    std::vector<Real> global_final_grid;
    bool result_output_ok = true;
    if (printResults) {
        result_output_ok = gatherGlobalGrid(
            local_final_grid, block, nx, ny, nz, dims, cartesian, rank,
            world_size, global_final_grid);
        if (result_output_ok && rank == 0) {
            print_results(global_final_grid, "Grid");
        }
    }

    bool valid = true;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
    }
    if (validate && result_output_ok) {
        if (printResults) {
            if (rank == 0) {
                valid = validateResult(global_final_grid, nx, ny, nz);
            }
            int valid_int = valid ? 1 : 0;
            MPI_Bcast(&valid_int, 1, MPI_INT, 0, cartesian);
            valid = valid_int != 0;
        } else {
            valid = validateDistributed(local_final_grid, block, cartesian,
                                         rank);
        }
    }
    if (validate && rank == 0) {
        std::printf("Validation: %s\n",
                    valid && result_output_ok ? "PASSED" : "FAILED");
    }

    destroyFaceTypes(face_types);
    MPI_Comm_free(&cartesian);
    MPI_Finalize();

    return (validate && (!valid || !result_output_ok)) ? 1 : 0;
}
