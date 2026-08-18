#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr val_t TRANSFER_COEFFICIENT = 0.8;
constexpr val_t INFLOW = 0.5;
constexpr val_t OUTFLOW = -0.5;

enum MessageTag : int {
    TO_UP = 0,
    TO_DOWN = 1,
    TO_LEFT = 2,
    TO_RIGHT = 3,
};

struct Block {
    int begin = 0;
    int size = 0;
};

struct Decomposition {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;
    int size = 0;
    int dimensions[2] = {1, 1};
    int coordinates[2] = {0, 0};
    int up = MPI_PROC_NULL;
    int down = MPI_PROC_NULL;
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;
    Block rows;
    Block columns;
    bool active = false;
};

struct World {
    int n = 0;
    int row_begin = 0;
    int column_begin = 0;
    int row_count = 0;
    int column_count = 0;

    // Structure of arrays keeps the hot energy stream contiguous.  Only
    // current_energy is communicated; accumulated_flux is strictly local.
    std::vector<val_t> current_energy;
    std::vector<val_t> next_energy;
    std::vector<val_t> accumulated_flux;
    std::vector<val_t> next_accumulated_flux;
    std::vector<val_t> send_left;
    std::vector<val_t> send_right;
    std::vector<val_t> halo_up;
    std::vector<val_t> halo_down;
    std::vector<val_t> halo_left;
    std::vector<val_t> halo_right;

    // Two persistent request sets correspond to the two alternating energy
    // buffers.  Reusing them avoids rebuilding eight MPI requests per step.
    MPI_Request halo_requests[2][8] = {};
};

struct Options {
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results = false;
};

Block blockForCoordinate(const int extent, const int parts, const int coordinate) {
    const int base = extent / parts;
    const int remainder = extent % parts;
    return Block{coordinate * base + std::min(coordinate, remainder),
                 base + (coordinate < remainder ? 1 : 0)};
}

// Select the largest usable rank count and its most square factorization.
// Constraining each Cartesian dimension to the mesh extent guarantees that
// every active rank owns at least one element.
void chooseProcessGrid(const int available_ranks, const int n, int& active_ranks,
                       int dimensions[2]) {
    const int64_t element_count = static_cast<int64_t>(n) * n;
    active_ranks = static_cast<int>(
        std::min<int64_t>(available_ranks, element_count));

    for (; active_ranks > 0; --active_ranks) {
        const int square_root = static_cast<int>(std::sqrt(active_ranks));
        for (int columns = square_root; columns >= 1; --columns) {
            if (active_ranks % columns != 0) {
                continue;
            }
            const int rows = active_ranks / columns;
            if (rows <= n && columns <= n) {
                dimensions[0] = rows;
                dimensions[1] = columns;
                return;
            }
        }
    }
}

Decomposition makeDecomposition(const int n, const int world_rank,
                                const int world_size) {
    Decomposition decomposition;
    int active_ranks = 0;
    chooseProcessGrid(world_size, n, active_ranks, decomposition.dimensions);

    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD,
                   world_rank < active_ranks ? 0 : MPI_UNDEFINED, world_rank,
                   &active_comm);

    if (active_comm == MPI_COMM_NULL) {
        return decomposition;
    }

    constexpr int periods[2] = {0, 0};
    MPI_Cart_create(active_comm, 2, decomposition.dimensions, periods, 0,
                    &decomposition.comm);
    MPI_Comm_free(&active_comm);

    decomposition.active = true;
    MPI_Comm_rank(decomposition.comm, &decomposition.rank);
    MPI_Comm_size(decomposition.comm, &decomposition.size);
    MPI_Cart_coords(decomposition.comm, decomposition.rank, 2,
                    decomposition.coordinates);
    MPI_Cart_shift(decomposition.comm, 0, 1, &decomposition.up,
                   &decomposition.down);
    MPI_Cart_shift(decomposition.comm, 1, 1, &decomposition.left,
                   &decomposition.right);

    decomposition.rows = blockForCoordinate(
        n, decomposition.dimensions[0], decomposition.coordinates[0]);
    decomposition.columns = blockForCoordinate(
        n, decomposition.dimensions[1], decomposition.coordinates[1]);
    return decomposition;
}

World buildSquare2D(const int n, const Decomposition& decomposition) {
    World world;
    world.n = n;
    world.row_begin = decomposition.rows.begin;
    world.column_begin = decomposition.columns.begin;
    world.row_count = decomposition.rows.size;
    world.column_count = decomposition.columns.size;

    const size_t local_elements =
        static_cast<size_t>(world.row_count) * world.column_count;
    world.current_energy.assign(local_elements, 0.0);
    world.next_energy.assign(local_elements, 0.0);
    world.accumulated_flux.assign(local_elements, 0.0);
    world.next_accumulated_flux.assign(local_elements, 0.0);

    world.send_left.resize(world.row_count);
    world.send_right.resize(world.row_count);
    world.halo_up.resize(world.column_count);
    world.halo_down.resize(world.column_count);
    world.halo_left.resize(world.row_count);
    world.halo_right.resize(world.row_count);

    return world;
}

inline val_t computeFlux(const val_t this_energy, const val_t other_energy) {
    return (other_energy - this_energy) * TRANSFER_COEFFICIENT * 1.0 * 0.25;
}

void initializeHaloExchange(World& world,
                            const Decomposition& decomposition) {
    const int rows = world.row_count;
    const int columns = world.column_count;
    val_t* energy_buffers[2] = {world.current_energy.data(),
                                world.next_energy.data()};

    for (int parity = 0; parity < 2; ++parity) {
        MPI_Request* requests = world.halo_requests[parity];
        MPI_Recv_init(world.halo_up.data(), columns, MPI_DOUBLE,
                      decomposition.up, TO_DOWN, decomposition.comm,
                      &requests[0]);
        MPI_Recv_init(world.halo_down.data(), columns, MPI_DOUBLE,
                      decomposition.down, TO_UP, decomposition.comm,
                      &requests[1]);
        MPI_Recv_init(world.halo_left.data(), rows, MPI_DOUBLE,
                      decomposition.left, TO_RIGHT, decomposition.comm,
                      &requests[2]);
        MPI_Recv_init(world.halo_right.data(), rows, MPI_DOUBLE,
                      decomposition.right, TO_LEFT, decomposition.comm,
                      &requests[3]);

        MPI_Send_init(energy_buffers[parity], columns, MPI_DOUBLE,
                      decomposition.up, TO_UP, decomposition.comm,
                      &requests[4]);
        MPI_Send_init(energy_buffers[parity] +
                          static_cast<size_t>(rows - 1) * columns,
                      columns, MPI_DOUBLE, decomposition.down, TO_DOWN,
                      decomposition.comm, &requests[5]);
        MPI_Send_init(world.send_left.data(), rows, MPI_DOUBLE,
                      decomposition.left, TO_LEFT, decomposition.comm,
                      &requests[6]);
        MPI_Send_init(world.send_right.data(), rows, MPI_DOUBLE,
                      decomposition.right, TO_RIGHT, decomposition.comm,
                      &requests[7]);
    }
}

void beginHaloExchange(World& world, const int parity) {
    const int rows = world.row_count;
    const int columns = world.column_count;
    for (int row = 0; row < rows; ++row) {
        const size_t offset = static_cast<size_t>(row) * columns;
        world.send_left[row] = world.current_energy[offset];
        world.send_right[row] = world.current_energy[offset + columns - 1];
    }
    MPI_Startall(8, world.halo_requests[parity]);
}

void destroyHaloExchange(World& world) {
    for (auto& request_set : world.halo_requests) {
        for (MPI_Request& request : request_set) {
            if (request != MPI_REQUEST_NULL) {
                MPI_Request_free(&request);
            }
        }
    }
}

// The local interior is independent of incoming halos.  Restrict-qualified
// streams let the compiler vectorize across columns while MPI transfers are in
// flight.
void updateInterior(World& world) {
    const int rows = world.row_count;
    const int columns = world.column_count;
    const val_t* __restrict current = world.current_energy.data();
    const val_t* __restrict accumulated = world.accumulated_flux.data();
    val_t* __restrict next = world.next_energy.data();
    val_t* __restrict next_accumulated =
        world.next_accumulated_flux.data();

    for (int row = 1; row < rows - 1; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * columns;
#pragma GCC ivdep
        for (int column = 1; column < columns - 1; ++column) {
            const size_t index = row_offset + column;
            const val_t energy = current[index];
            val_t total_flux = 0.0;
            // Match the original connection order: down, up, right, left.
            total_flux += computeFlux(energy, current[index + columns]);
            total_flux += computeFlux(energy, current[index - columns]);
            total_flux += computeFlux(energy, current[index + 1]);
            total_flux += computeFlux(energy, current[index - 1]);
            next[index] = energy + total_flux;
            next_accumulated[index] =
                accumulated[index] + std::abs(total_flux);
        }
    }
}

inline val_t externalFlowAt(const int n, const int global_row,
                            const int global_column) {
    val_t flow = 0.0;
    const int last = n - 1;
    // Preserve the original assignment order.  This matters for a 1x1 mesh.
    if (global_row == 0 && global_column == 0) {
        flow = INFLOW;
    }
    if (global_row == 0 && global_column == last) {
        flow = OUTFLOW;
    }
    if (global_row == last && global_column == 0) {
        flow = OUTFLOW;
    }
    if (global_row == last && global_column == last) {
        flow = INFLOW;
    }
    return flow;
}

inline void updateBoundaryElement(World& world, const int row,
                                  const int column) {
    const int columns = world.column_count;
    const size_t index = static_cast<size_t>(row) * columns + column;
    const int global_row = world.row_begin + row;
    const int global_column = world.column_begin + column;
    const val_t energy = world.current_energy[index];
    val_t total_flux = externalFlowAt(world.n, global_row, global_column);

    // Match the original connection order: down, up, right, left.
    if (global_row + 1 < world.n) {
        const val_t neighbor = row + 1 < world.row_count
                                   ? world.current_energy[index + columns]
                                   : world.halo_down[column];
        total_flux += computeFlux(energy, neighbor);
    }
    if (global_row > 0) {
        const val_t neighbor = row > 0
                                   ? world.current_energy[index - columns]
                                   : world.halo_up[column];
        total_flux += computeFlux(energy, neighbor);
    }
    if (global_column + 1 < world.n) {
        const val_t neighbor = column + 1 < world.column_count
                                   ? world.current_energy[index + 1]
                                   : world.halo_right[row];
        total_flux += computeFlux(energy, neighbor);
    }
    if (global_column > 0) {
        const val_t neighbor = column > 0
                                   ? world.current_energy[index - 1]
                                   : world.halo_left[row];
        total_flux += computeFlux(energy, neighbor);
    }

    world.next_energy[index] = energy + total_flux;
    world.next_accumulated_flux[index] =
        world.accumulated_flux[index] + std::abs(total_flux);
}

void updateBoundary(World& world) {
    const int rows = world.row_count;
    const int columns = world.column_count;

    for (int column = 0; column < columns; ++column) {
        updateBoundaryElement(world, 0, column);
    }
    if (rows > 1) {
        for (int column = 0; column < columns; ++column) {
            updateBoundaryElement(world, rows - 1, column);
        }
    }
    for (int row = 1; row < rows - 1; ++row) {
        updateBoundaryElement(world, row, 0);
        if (columns > 1) {
            updateBoundaryElement(world, row, columns - 1);
        }
    }
}

void runSimulation(World& world, const int n_iters) {
    for (int iteration = 0; iteration < n_iters; ++iteration) {
        const int parity = iteration & 1;
        beginHaloExchange(world, parity);
        updateInterior(world);
        MPI_Waitall(8, world.halo_requests[parity], MPI_STATUSES_IGNORE);
        updateBoundary(world);
        std::swap(world.current_energy, world.next_energy);
        std::swap(world.accumulated_flux, world.next_accumulated_flux);
    }
}

uint64_t doubleBits(const val_t value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t computeLocalHash(const World& world) {
    uint64_t result = 0;
    for (int row = 0; row < world.row_count; ++row) {
        const idx_t global_row = static_cast<idx_t>(world.row_begin + row);
        for (int column = 0; column < world.column_count; ++column) {
            const size_t local_index =
                static_cast<size_t>(row) * world.column_count + column;
            const idx_t global_index =
                global_row * static_cast<idx_t>(world.n) +
                static_cast<idx_t>(world.column_begin + column);
            result ^=
                (doubleBits(world.current_energy[local_index]) + global_index) *
                0x9e3779b97f4a7c15ULL;
            result ^=
                (doubleBits(world.accumulated_flux[local_index]) +
                 global_index) *
                0xbf58476d1ce4e5b9ULL;
        }
    }
    return result;
}

std::vector<val_t> gatherField(const std::vector<val_t>& local_field,
                               const int n,
                               const Decomposition& decomposition) {
    const int local_count = static_cast<int>(local_field.size());
    std::vector<int> receive_counts;
    if (decomposition.rank == 0) {
        receive_counts.resize(decomposition.size);
    }
    MPI_Gather(&local_count, 1, MPI_INT, receive_counts.data(), 1, MPI_INT, 0,
               decomposition.comm);

    std::vector<int> displacements;
    std::vector<val_t> packed;
    if (decomposition.rank == 0) {
        displacements.resize(decomposition.size);
        int displacement = 0;
        for (int rank = 0; rank < decomposition.size; ++rank) {
            displacements[rank] = displacement;
            displacement += receive_counts[rank];
        }
        packed.resize(displacement);
    }

    MPI_Gatherv(local_field.data(), local_count, MPI_DOUBLE, packed.data(),
                receive_counts.data(), displacements.data(), MPI_DOUBLE, 0,
                decomposition.comm);

    std::vector<val_t> global_field;
    if (decomposition.rank != 0) {
        return global_field;
    }

    global_field.resize(static_cast<size_t>(n) * n);
    for (int rank = 0; rank < decomposition.size; ++rank) {
        int coordinates[2];
        MPI_Cart_coords(decomposition.comm, rank, 2, coordinates);
        const Block rows =
            blockForCoordinate(n, decomposition.dimensions[0], coordinates[0]);
        const Block columns = blockForCoordinate(
            n, decomposition.dimensions[1], coordinates[1]);
        const val_t* source = packed.data() + displacements[rank];
        for (int row = 0; row < rows.size; ++row) {
            std::copy_n(source + static_cast<size_t>(row) * columns.size,
                        columns.size,
                        global_field.data() +
                            static_cast<size_t>(rows.begin + row) * n +
                            columns.begin);
        }
    }
    return global_field;
}

bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& accumulated_flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t index = 0; index < energy.size(); ++index) {
        energy_sum += energy[index];
        flux_sum += accumulated_flux[index];
        energy_max = std::max(energy[index], energy_max);
        energy_min = std::min(energy[index], energy_min);
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        std::printf(
            "  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    std::printf("  Validation: PASSED\n");
    return true;
}

uint64_t localMemoryBytes(const World& world) {
    return static_cast<uint64_t>(
               world.current_energy.size() + world.next_energy.size() +
               world.accumulated_flux.size() +
               world.next_accumulated_flux.size() +
               world.send_left.size() + world.send_right.size() +
               world.halo_up.size() +
               world.halo_down.size() + world.halo_left.size() +
               world.halo_right.size()) *
           sizeof(val_t);
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Returns 0 to run, 1 for help, and 2 for an argument error.
int parseOptions(const int argc, char** argv, Options& options) {
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            options.n_elems_root = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-i") == 0 &&
                   argument + 1 < argc) {
            options.n_iters = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            options.print_results = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            printUsage(argv[0]);
            return 1;
        } else {
            std::printf("Unknown option: %s\n", argv[argument]);
            printUsage(argv[0]);
            return 2;
        }
    }

    const int64_t element_count =
        static_cast<int64_t>(options.n_elems_root) * options.n_elems_root;
    if (options.n_elems_root <= 0 || options.n_iters < 0 ||
        element_count > std::numeric_limits<int>::max()) {
        std::printf(
            "Grid size must be positive, iterations non-negative, and N*N "
            "must fit in a 32-bit MPI count.\n");
        return 2;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Options options;
    int parse_status = 0;
    if (world_rank == 0) {
        parse_status = parseOptions(argc, argv, options);
    }

    int option_data[5] = {options.n_elems_root, options.n_iters,
                          options.validate ? 1 : 0,
                          options.print_results ? 1 : 0, parse_status};
    MPI_Bcast(option_data, 5, MPI_INT, 0, MPI_COMM_WORLD);
    options.n_elems_root = option_data[0];
    options.n_iters = option_data[1];
    options.validate = option_data[2] != 0;
    options.print_results = option_data[3] != 0;
    parse_status = option_data[4];

    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 1 ? 0 : 1;
    }

    Decomposition decomposition =
        makeDecomposition(options.n_elems_root, world_rank, world_size);
    int exit_code = 0;

    if (decomposition.active) {
        const int64_t n_elems =
            static_cast<int64_t>(options.n_elems_root) * options.n_elems_root;
        if (decomposition.rank == 0) {
            std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
            std::printf("============================================\n");
            std::printf("Grid size: %d x %d = %" PRId64 " elements\n",
                        options.n_elems_root, options.n_elems_root, n_elems);
            std::printf("Iterations: %d\n", options.n_iters);
            std::printf("Validation: %s\n",
                        options.validate ? "enabled" : "disabled");
            std::printf("MPI ranks: %d active of %d (%d x %d process grid)\n\n",
                        decomposition.size, world_size,
                        decomposition.dimensions[0],
                        decomposition.dimensions[1]);
            if (decomposition.size != world_size) {
                std::printf(
                    "Note: %d rank(s) are idle because a non-empty Cartesian "
                    "partition was not possible.\n",
                    world_size - decomposition.size);
            }
            std::printf("Building distributed unstructured mesh...\n");
        }

        World world = buildSquare2D(options.n_elems_root, decomposition);
        initializeHaloExchange(world, decomposition);
        const uint64_t local_memory = localMemoryBytes(world);
        uint64_t aggregate_memory = 0;
        uint64_t maximum_memory = 0;
        MPI_Reduce(&local_memory, &aggregate_memory, 1, MPI_UINT64_T, MPI_SUM, 0,
                   decomposition.comm);
        MPI_Reduce(&local_memory, &maximum_memory, 1, MPI_UINT64_T, MPI_MAX, 0,
                   decomposition.comm);
        if (decomposition.rank == 0) {
            constexpr double mib = 1024.0 * 1024.0;
            std::printf(
                "Memory usage (distributed): %.2f MB aggregate, %.2f MB "
                "maximum/rank\n\n",
                aggregate_memory / mib, maximum_memory / mib);
            std::printf("Running simulation...\n");
        }

        MPI_Barrier(decomposition.comm);
        const double start = MPI_Wtime();
        runSimulation(world, options.n_iters);
        const double local_seconds = MPI_Wtime() - start;
        destroyHaloExchange(world);
        double elapsed_seconds = 0.0;
        MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                   decomposition.comm);

        const uint64_t local_hash = computeLocalHash(world);
        uint64_t global_hash = 0;
        MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
                   decomposition.comm);

        if (decomposition.rank == 0) {
            const int measured_iterations = std::max(options.n_iters, 1);
            const double time_per_iteration_ms =
                elapsed_seconds * 1000.0 / measured_iterations;
            const double giga_elements_per_second =
                elapsed_seconds > 0.0
                    ? (static_cast<double>(options.n_iters) * n_elems) /
                          elapsed_seconds / 1.0e9
                    : 0.0;
            const double gflops = giga_elements_per_second * 22.0;

            std::printf("Computation time: %.3f ms\n",
                        elapsed_seconds * 1000.0);
            std::printf("Performance:\n");
            std::printf("  Time per iteration: %.4f ms\n",
                        time_per_iteration_ms);
            std::printf("  Elements/sec: %.4f GigaElements/s\n",
                        giga_elements_per_second);
            std::printf("  Performance: %.4f GFLOPS\n", gflops);
            std::printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
        }

        std::vector<val_t> global_energy;
        if (options.print_results || options.validate) {
            global_energy = gatherField(world.current_energy,
                                        options.n_elems_root, decomposition);
        }
        if (options.print_results && decomposition.rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }

        if (options.validate) {
            std::vector<val_t> global_flux =
                gatherField(world.accumulated_flux, options.n_elems_root,
                            decomposition);
            if (decomposition.rank == 0 &&
                !validateResults(global_energy, global_flux)) {
                exit_code = 1;
            }
        }

        MPI_Comm_free(&decomposition.comm);
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
