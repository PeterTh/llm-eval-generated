// Open MPI and MPICH otherwise expose their removed C++ bindings when this
// translation unit uses only the leaner, portable C API.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

#if defined(__GNUC__) || defined(__clang__)
#define UNSTRUCTURED_RESTRICT __restrict__
#else
#define UNSTRUCTURED_RESTRICT
#endif

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t INFLOW = 0.5;
constexpr val_t OUTFLOW = -0.5;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct Block {
    int begin;
    int size;
};

struct ProcessGrid {
    int active_ranks;
    int dims[2];
};

struct World {
    int n_elems_root = 0;
    int x_begin = 0;
    int y_begin = 0;
    int local_nx = 0;
    int local_ny = 0;
    int minus_x = MPI_PROC_NULL;
    int plus_x = MPI_PROC_NULL;
    int minus_y = MPI_PROC_NULL;
    int plus_y = MPI_PROC_NULL;

    // A structure-of-arrays layout keeps the stencil streams contiguous and
    // avoids communicating the accumulated flux, which is purely local state.
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> accumulated_flux;
    std::vector<val_t> accumulated_flux_swap;

    std::vector<val_t> halo_minus_x;
    std::vector<val_t> halo_plus_x;
    std::vector<val_t> halo_minus_y;
    std::vector<val_t> halo_plus_y;
    std::vector<val_t> send_minus_y;
    std::vector<val_t> send_plus_y;
};

Block decomposeBlock(const int extent, const int coordinate, const int parts) {
    const int base = extent / parts;
    const int remainder = extent % parts;
    const int size = base + (coordinate < remainder ? 1 : 0);
    const int begin = coordinate * base + std::min(coordinate, remainder);
    return Block{begin, size};
}

// Use as many ranks as possible while retaining a nonempty, nearly square 2D
// process grid. Extra ranks can only occur for grids smaller than the launch.
ProcessGrid chooseProcessGrid(const int world_size, const int n_elems_root) {
    ProcessGrid best{1, {1, 1}};
    int best_imbalance = 0;
    const int max_x = std::min(world_size, n_elems_root);

    for (int px = 1; px <= max_x; ++px) {
        const int py = std::min(n_elems_root, world_size / px);
        const int active = px * py;
        const int imbalance = std::abs(px - py);
        if (active > best.active_ranks ||
            (active == best.active_ranks && imbalance < best_imbalance)) {
            best.active_ranks = active;
            best.dims[0] = px;
            best.dims[1] = py;
            best_imbalance = imbalance;
        }
    }
    return best;
}

val_t externalFlowAt(const int x, const int y, const int n_elems_root) {
    const int last = n_elems_root - 1;
    // This ordering also preserves the original n=1 behavior, where all four
    // corner assignments target the same cell and the final material is inflow.
    if ((x == 0 && y == 0) || (x == last && y == last)) {
        return INFLOW;
    }
    if ((x == 0 && y == last) || (x == last && y == 0)) {
        return OUTFLOW;
    }
    return 0.0;
}

World buildSquare2D(const int n_elems_root, MPI_Comm cart_comm) {
    int cart_rank = 0;
    int coords[2] = {0, 0};
    int dims[2] = {1, 1};
    int periods[2] = {0, 0};
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);

    const Block x_block = decomposeBlock(n_elems_root, coords[0], dims[0]);
    const Block y_block = decomposeBlock(n_elems_root, coords[1], dims[1]);

    World world;
    world.n_elems_root = n_elems_root;
    world.x_begin = x_block.begin;
    world.y_begin = y_block.begin;
    world.local_nx = x_block.size;
    world.local_ny = y_block.size;
    MPI_Cart_shift(cart_comm, 0, 1, &world.minus_x, &world.plus_x);
    MPI_Cart_shift(cart_comm, 1, 1, &world.minus_y, &world.plus_y);

    const size_t local_elements =
        static_cast<size_t>(world.local_nx) * world.local_ny;
    world.energy.assign(local_elements, 0.0);
    world.energy_swap.resize(local_elements);
    world.accumulated_flux.assign(local_elements, 0.0);
    world.accumulated_flux_swap.resize(local_elements);

    world.halo_minus_x.resize(world.local_ny);
    world.halo_plus_x.resize(world.local_ny);
    world.halo_minus_y.resize(world.local_nx);
    world.halo_plus_y.resize(world.local_nx);
    world.send_minus_y.resize(world.local_nx);
    world.send_plus_y.resize(world.local_nx);
    return world;
}

inline val_t computeFlux(const val_t this_energy, const val_t other_energy) {
    // Keep the operation ordering of the original connectivity kernel.
    return (other_energy - this_energy) * TRANSFER_COEFF * 1.0 * 0.25;
}

inline void updateBoundaryCell(World& world, const int x, const int y) {
    const int ny = world.local_ny;
    const size_t i = static_cast<size_t>(x) * ny + y;
    const val_t this_energy = world.energy[i];
    const int global_x = world.x_begin + x;
    const int global_y = world.y_begin + y;
    val_t total_flux =
        externalFlowAt(global_x, global_y, world.n_elems_root);

    // The addition order is +x, -x, +y, -y, exactly as in the serial mesh.
    if (global_x + 1 < world.n_elems_root) {
        const val_t other = x + 1 < world.local_nx
                                ? world.energy[i + ny]
                                : world.halo_plus_x[y];
        total_flux += computeFlux(this_energy, other);
    }
    if (global_x > 0) {
        const val_t other = x > 0 ? world.energy[i - ny]
                                  : world.halo_minus_x[y];
        total_flux += computeFlux(this_energy, other);
    }
    if (global_y + 1 < world.n_elems_root) {
        const val_t other = y + 1 < world.local_ny
                                ? world.energy[i + 1]
                                : world.halo_plus_y[x];
        total_flux += computeFlux(this_energy, other);
    }
    if (global_y > 0) {
        const val_t other = y > 0 ? world.energy[i - 1]
                                  : world.halo_minus_y[x];
        total_flux += computeFlux(this_energy, other);
    }

    world.energy_swap[i] = this_energy + total_flux;
    world.accumulated_flux_swap[i] =
        world.accumulated_flux[i] + std::abs(total_flux);
}

void updateInterior(World& world) {
    const int nx = world.local_nx;
    const int ny = world.local_ny;
    const val_t* UNSTRUCTURED_RESTRICT current = world.energy.data();
    const val_t* UNSTRUCTURED_RESTRICT flux = world.accumulated_flux.data();
    val_t* UNSTRUCTURED_RESTRICT next = world.energy_swap.data();
    val_t* UNSTRUCTURED_RESTRICT next_flux =
        world.accumulated_flux_swap.data();

    // Interior cells cannot touch a remote halo, so this work overlaps all MPI
    // transfers. Keeping it as a separate, branch-free loop also vectorizes well.
    for (int x = 1; x + 1 < nx; ++x) {
        for (int y = 1; y + 1 < ny; ++y) {
            const size_t i = static_cast<size_t>(x) * ny + y;
            const val_t this_energy = current[i];
            val_t total_flux = 0.0;
            total_flux += computeFlux(this_energy, current[i + ny]);
            total_flux += computeFlux(this_energy, current[i - ny]);
            total_flux += computeFlux(this_energy, current[i + 1]);
            total_flux += computeFlux(this_energy, current[i - 1]);
            next[i] = this_energy + total_flux;
            next_flux[i] = flux[i] + std::abs(total_flux);
        }
    }
}

void updateBoundary(World& world) {
    for (int x = 0; x < world.local_nx; ++x) {
        for (int y = 0; y < world.local_ny; ++y) {
            if (x == 0 || x + 1 == world.local_nx ||
                y == 0 || y + 1 == world.local_ny) {
                updateBoundaryCell(world, x, y);
            }
        }
    }
}

void runSimulation(World& world, const int n_iters, MPI_Comm cart_comm) {
    constexpr int TO_MINUS_X = 10;
    constexpr int TO_PLUS_X = 11;
    constexpr int TO_MINUS_Y = 12;
    constexpr int TO_PLUS_Y = 13;

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int request_count = 0;

        if (world.minus_x != MPI_PROC_NULL) {
            MPI_Irecv(world.halo_minus_x.data(), world.local_ny, MPI_DOUBLE,
                      world.minus_x, TO_PLUS_X, cart_comm,
                      &requests[request_count++]);
        }
        if (world.plus_x != MPI_PROC_NULL) {
            MPI_Irecv(world.halo_plus_x.data(), world.local_ny, MPI_DOUBLE,
                      world.plus_x, TO_MINUS_X, cart_comm,
                      &requests[request_count++]);
        }
        if (world.minus_y != MPI_PROC_NULL) {
            MPI_Irecv(world.halo_minus_y.data(), world.local_nx, MPI_DOUBLE,
                      world.minus_y, TO_PLUS_Y, cart_comm,
                      &requests[request_count++]);
        }
        if (world.plus_y != MPI_PROC_NULL) {
            MPI_Irecv(world.halo_plus_y.data(), world.local_nx, MPI_DOUBLE,
                      world.plus_y, TO_MINUS_Y, cart_comm,
                      &requests[request_count++]);
        }

        // X faces are contiguous. Pack only the strided Y faces.
        if (world.minus_y != MPI_PROC_NULL || world.plus_y != MPI_PROC_NULL) {
            for (int x = 0; x < world.local_nx; ++x) {
                const size_t row = static_cast<size_t>(x) * world.local_ny;
                world.send_minus_y[x] = world.energy[row];
                world.send_plus_y[x] = world.energy[row + world.local_ny - 1];
            }
        }

        if (world.minus_x != MPI_PROC_NULL) {
            MPI_Isend(world.energy.data(), world.local_ny, MPI_DOUBLE,
                      world.minus_x, TO_MINUS_X, cart_comm,
                      &requests[request_count++]);
        }
        if (world.plus_x != MPI_PROC_NULL) {
            const size_t last_row =
                static_cast<size_t>(world.local_nx - 1) * world.local_ny;
            MPI_Isend(world.energy.data() + last_row, world.local_ny, MPI_DOUBLE,
                      world.plus_x, TO_PLUS_X, cart_comm,
                      &requests[request_count++]);
        }
        if (world.minus_y != MPI_PROC_NULL) {
            MPI_Isend(world.send_minus_y.data(), world.local_nx, MPI_DOUBLE,
                      world.minus_y, TO_MINUS_Y, cart_comm,
                      &requests[request_count++]);
        }
        if (world.plus_y != MPI_PROC_NULL) {
            MPI_Isend(world.send_plus_y.data(), world.local_nx, MPI_DOUBLE,
                      world.plus_y, TO_PLUS_Y, cart_comm,
                      &requests[request_count++]);
        }

        updateInterior(world);
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        updateBoundary(world);

        std::swap(world.energy, world.energy_swap);
        std::swap(world.accumulated_flux, world.accumulated_flux_swap);
    }
}

uint64_t bitsOf(const val_t value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t computeDistributedHash(const World& world, MPI_Comm cart_comm) {
    uint64_t local_hash = 0;
    for (int x = 0; x < world.local_nx; ++x) {
        const idx_t global_x = static_cast<idx_t>(world.x_begin + x);
        for (int y = 0; y < world.local_ny; ++y) {
            const size_t local_i = static_cast<size_t>(x) * world.local_ny + y;
            const idx_t global_i =
                global_x * static_cast<idx_t>(world.n_elems_root) +
                static_cast<idx_t>(world.y_begin + y);
            local_hash ^=
                (bitsOf(world.energy[local_i]) + global_i) *
                0x9e3779b97f4a7c15ULL;
            local_hash ^=
                (bitsOf(world.accumulated_flux[local_i]) + global_i) *
                0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               cart_comm);
    return global_hash;
}

void sendLarge(const val_t* data, size_t count, const int destination,
               const int tag, MPI_Comm comm) {
    constexpr size_t max_count = static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, max_count));
        MPI_Send(data, chunk, MPI_DOUBLE, destination, tag, comm);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void recvLarge(val_t* data, size_t count, const int source, const int tag,
               MPI_Comm comm) {
    constexpr size_t max_count = static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, max_count));
        MPI_Recv(data, chunk, MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void unpackBlock(std::vector<ElementDynamic>& global,
                 const std::vector<val_t>& packed, const Block x_block,
                 const Block y_block, const int n_elems_root) {
    for (int x = 0; x < x_block.size; ++x) {
        for (int y = 0; y < y_block.size; ++y) {
            const size_t local_i = static_cast<size_t>(x) * y_block.size + y;
            const size_t global_i =
                static_cast<size_t>(x_block.begin + x) * n_elems_root +
                y_block.begin + y;
            global[global_i] =
                ElementDynamic{packed[2 * local_i], packed[2 * local_i + 1]};
        }
    }
}

// Full global ordering is only materialized for validation or -r output. The
// timed path and its result hash remain completely distributed.
std::vector<ElementDynamic> gatherResults(const World& world, MPI_Comm cart_comm,
                                          const int cart_rank,
                                          const int cart_size) {
    constexpr int RESULTS_TAG = 20;
    const size_t local_elements =
        static_cast<size_t>(world.local_nx) * world.local_ny;
    std::vector<val_t> packed(2 * local_elements);
    for (size_t i = 0; i < local_elements; ++i) {
        packed[2 * i] = world.energy[i];
        packed[2 * i + 1] = world.accumulated_flux[i];
    }

    if (cart_rank != 0) {
        sendLarge(packed.data(), packed.size(), 0, RESULTS_TAG, cart_comm);
        return {};
    }

    int dims[2] = {1, 1};
    int periods[2] = {0, 0};
    int root_coords[2] = {0, 0};
    MPI_Cart_get(cart_comm, 2, dims, periods, root_coords);
    std::vector<ElementDynamic> global(
        static_cast<size_t>(world.n_elems_root) * world.n_elems_root);

    for (int rank = 0; rank < cart_size; ++rank) {
        int coords[2] = {0, 0};
        MPI_Cart_coords(cart_comm, rank, 2, coords);
        const Block x_block =
            decomposeBlock(world.n_elems_root, coords[0], dims[0]);
        const Block y_block =
            decomposeBlock(world.n_elems_root, coords[1], dims[1]);
        const size_t block_elements =
            static_cast<size_t>(x_block.size) * y_block.size;

        if (rank == 0) {
            unpackBlock(global, packed, x_block, y_block, world.n_elems_root);
        } else {
            std::vector<val_t> received(2 * block_elements);
            recvLarge(received.data(), received.size(), rank, RESULTS_TAG,
                      cart_comm);
            unpackBlock(global, received, x_block, y_block, world.n_elems_root);
        }
    }
    return global;
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

size_t localAllocatedBytes(const World& world) {
    return (world.energy.size() + world.energy_swap.size() +
            world.accumulated_flux.size() +
            world.accumulated_flux_swap.size() +
            world.halo_minus_x.size() + world.halo_plus_x.size() +
            world.halo_minus_y.size() + world.halo_plus_y.size() +
            world.send_minus_y.size() + world.send_plus_y.size()) *
           sizeof(val_t);
}

void printUsage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
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

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
            break;
        }
    }

    if (n_elems_root <= 0 || n_iters < 0) {
        if (world_rank == 0) {
            printf("Grid size must be positive and iterations nonnegative.\n");
        }
        arguments_valid = false;
    }
    if (show_help || !arguments_valid) {
        if (world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }

    const ProcessGrid process_grid =
        chooseProcessGrid(world_size, n_elems_root);
    MPI_Comm active_comm = MPI_COMM_NULL;
    const int color =
        world_rank < process_grid.active_ranks ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active_comm);

    // A launch can have more processes than cells. Such ranks participate in
    // MPI initialization/finalization but own no domain and allocate no mesh.
    if (color == MPI_UNDEFINED) {
        MPI_Finalize();
        return 0;
    }

    int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    // Permit the MPI runtime to map Cartesian neighbors onto the physical
    // network topology when it has placement information available.
    MPI_Cart_create(active_comm, 2, process_grid.dims, periods, 1, &cart_comm);
    MPI_Comm_free(&active_comm);

    int cart_rank = 0;
    int cart_size = 1;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Comm_size(cart_comm, &cart_size);
    const idx_t n_elems =
        static_cast<idx_t>(n_elems_root) * n_elems_root;

    if (cart_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n",
               n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI decomposition: %d x %d = %d active ranks (%d launched)\n",
               process_grid.dims[0], process_grid.dims[1], cart_size,
               world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\nBuilding unstructured mesh...\n");
    }

    World world = buildSquare2D(n_elems_root, cart_comm);

    const uint64_t local_memory = localAllocatedBytes(world);
    uint64_t aggregate_memory = 0;
    uint64_t maximum_memory = 0;
    MPI_Reduce(&local_memory, &aggregate_memory, 1, MPI_UINT64_T, MPI_SUM, 0,
               cart_comm);
    MPI_Reduce(&local_memory, &maximum_memory, 1, MPI_UINT64_T, MPI_MAX, 0,
               cart_comm);
    if (cart_rank == 0) {
        constexpr double mib = 1024.0 * 1024.0;
        printf("Memory usage: %.2f MB aggregate (%.2f MB max per rank)\n\n",
               aggregate_memory / mib, maximum_memory / mib);
        printf("Running simulation...\n");
    }

    MPI_Barrier(cart_comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, cart_comm);
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               cart_comm);

    const uint64_t result_hash = computeDistributedHash(world, cart_comm);
    if (cart_rank == 0) {
        const double duration_ms = elapsed_seconds * 1000.0;
        printf("Computation time: %.3f ms\n", duration_ms);

        const int measured_iters = std::max(n_iters, 1);
        const double time_per_iter = duration_ms / measured_iters;
        const double giga_elems_per_sec =
            elapsed_seconds > 0.0
                ? (static_cast<double>(n_iters) * static_cast<double>(n_elems)) /
                      elapsed_seconds / 1e9
                : 0.0;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }

    std::vector<ElementDynamic> global_results;
    if (print_results_requested || validate) {
        global_results = gatherResults(world, cart_comm, cart_rank, cart_size);
    }

    if (cart_rank == 0 && print_results_requested) {
        std::vector<double> energy_data;
        energy_data.reserve(global_results.size());
        for (const auto& elem : global_results) {
            energy_data.push_back(elem.current_energy);
        }
        print_results(energy_data, "ElementEnergy");
    }

    int exit_code = 0;
    if (cart_rank == 0 && validate && !validateResults(global_results)) {
        exit_code = 1;
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, cart_comm);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return exit_code;
}
