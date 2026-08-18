#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

constexpr Material DEFAULT_MATERIAL{0.8, 0.0};
constexpr val_t INFLOW = 0.5;
constexpr val_t OUTFLOW = -0.5;

struct Block {
    int begin;
    int size;
};

Block partition1D(const int global_size, const int parts, const int coordinate) {
    const int quotient = global_size / parts;
    const int remainder = global_size % parts;
    return Block{coordinate * quotient + std::min(coordinate, remainder),
                 quotient + (coordinate < remainder ? 1 : 0)};
}

// Each rank owns one rectangular part of the mesh. Energy has a one-cell halo;
// total_flux does not need to be communicated but uses the same layout so that
// both update streams have identical, cache-friendly addressing.
struct World {
    int global_n = 0;
    int local_nx = 0;
    int local_ny = 0;
    int x_begin = 0;
    int y_begin = 0;
    int pitch = 0;
    int x_minus = MPI_PROC_NULL;
    int x_plus = MPI_PROC_NULL;
    int y_minus = MPI_PROC_NULL;
    int y_plus = MPI_PROC_NULL;
    int current_buffer = 0;
    std::array<std::vector<val_t>, 2> energy;
    std::array<std::vector<val_t>, 2> total_flux;

    bool active() const { return local_nx != 0 && local_ny != 0; }

    size_t offset(const int local_x, const int local_y) const {
        return static_cast<size_t>(local_x) * static_cast<size_t>(pitch) +
               static_cast<size_t>(local_y);
    }
};

void buildSquare2D(World& world, const int n_elems_root, MPI_Comm cart_comm,
                   const int coordinates[2], const int dimensions[2]) {
    world.global_n = n_elems_root;
    const Block x_block = partition1D(n_elems_root, dimensions[0], coordinates[0]);
    const Block y_block = partition1D(n_elems_root, dimensions[1], coordinates[1]);
    world.x_begin = x_block.begin;
    world.y_begin = y_block.begin;
    world.local_nx = x_block.size;
    world.local_ny = y_block.size;
    world.pitch = world.local_ny + 2;

    const size_t storage_size = static_cast<size_t>(world.local_nx + 2) *
                                static_cast<size_t>(world.pitch);
    for (int buffer = 0; buffer < 2; ++buffer) {
        world.energy[buffer].assign(storage_size, 0.0);
        world.total_flux[buffer].assign(storage_size, 0.0);
    }

    int raw_x_minus = MPI_PROC_NULL;
    int raw_x_plus = MPI_PROC_NULL;
    int raw_y_minus = MPI_PROC_NULL;
    int raw_y_plus = MPI_PROC_NULL;
    MPI_Cart_shift(cart_comm, 0, 1, &raw_x_minus, &raw_x_plus);
    MPI_Cart_shift(cart_comm, 1, 1, &raw_y_minus, &raw_y_plus);

    if (world.active()) {
        // Empty blocks can occur when there are more process coordinates than
        // grid rows or columns. Nonempty blocks are a contiguous prefix, so an
        // empty Cartesian neighbor is always beyond a physical mesh boundary.
        world.x_minus = world.x_begin > 0 ? raw_x_minus : MPI_PROC_NULL;
        world.x_plus = world.x_begin + world.local_nx < n_elems_root
                           ? raw_x_plus
                           : MPI_PROC_NULL;
        world.y_minus = world.y_begin > 0 ? raw_y_minus : MPI_PROC_NULL;
        world.y_plus = world.y_begin + world.local_ny < n_elems_root
                           ? raw_y_plus
                           : MPI_PROC_NULL;
    }
}

enum MessageTag : int {
    TAG_X_NEGATIVE = 100,
    TAG_X_POSITIVE = 101,
    TAG_Y_NEGATIVE = 102,
    TAG_Y_POSITIVE = 103,
};

// Persistent requests remove per-iteration setup overhead. There is one set
// for each Jacobi buffer because persistent request buffer addresses are fixed.
struct HaloExchange {
    static constexpr int REQUEST_COUNT = 8;
    std::array<std::array<MPI_Request, REQUEST_COUNT>, 2> requests{};
    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    bool initialized = false;

    void initialize(World& world, MPI_Comm communicator) {
        const bool has_remote_neighbor =
            world.x_minus != MPI_PROC_NULL || world.x_plus != MPI_PROC_NULL ||
            world.y_minus != MPI_PROC_NULL || world.y_plus != MPI_PROC_NULL;
        if (!world.active() || !has_remote_neighbor) {
            return;
        }

        MPI_Type_vector(world.local_nx, 1, world.pitch, MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);

        for (int buffer = 0; buffer < 2; ++buffer) {
            val_t* const data = world.energy[buffer].data();
            auto& request = requests[buffer];
            int q = 0;

            // Receives precede sends so MPI can post all matching buffers as
            // soon as MPI_Startall enters the progress engine.
            MPI_Recv_init(data + world.offset(0, 1), world.local_ny, MPI_DOUBLE,
                          world.x_minus, TAG_X_POSITIVE, communicator, &request[q++]);
            MPI_Recv_init(data + world.offset(world.local_nx + 1, 1),
                          world.local_ny, MPI_DOUBLE, world.x_plus, TAG_X_NEGATIVE,
                          communicator, &request[q++]);
            MPI_Recv_init(data + world.offset(1, 0), 1, column_type, world.y_minus,
                          TAG_Y_POSITIVE, communicator, &request[q++]);
            MPI_Recv_init(data + world.offset(1, world.local_ny + 1), 1,
                          column_type, world.y_plus, TAG_Y_NEGATIVE, communicator,
                          &request[q++]);

            MPI_Send_init(data + world.offset(1, 1), world.local_ny, MPI_DOUBLE,
                          world.x_minus, TAG_X_NEGATIVE, communicator, &request[q++]);
            MPI_Send_init(data + world.offset(world.local_nx, 1), world.local_ny,
                          MPI_DOUBLE, world.x_plus, TAG_X_POSITIVE, communicator,
                          &request[q++]);
            MPI_Send_init(data + world.offset(1, 1), 1, column_type, world.y_minus,
                          TAG_Y_NEGATIVE, communicator, &request[q++]);
            MPI_Send_init(data + world.offset(1, world.local_ny), 1, column_type,
                          world.y_plus, TAG_Y_POSITIVE, communicator, &request[q++]);
        }
        initialized = true;
    }

    void start(const int buffer) {
        if (initialized) {
            MPI_Startall(REQUEST_COUNT, requests[buffer].data());
        }
    }

    void wait(const int buffer) {
        if (initialized) {
            MPI_Waitall(REQUEST_COUNT, requests[buffer].data(), MPI_STATUSES_IGNORE);
        }
    }

    void destroy() {
        if (!initialized) {
            return;
        }
        for (auto& buffer_requests : requests) {
            for (MPI_Request& request : buffer_requests) {
                if (request != MPI_REQUEST_NULL) {
                    MPI_Request_free(&request);
                }
            }
        }
        MPI_Type_free(&column_type);
        initialized = false;
    }
};

inline val_t computeFlux(const Material& material, const val_t this_energy,
                         const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) * material.transfer_coeff * connection_flux *
           0.25;
}

inline void updateElement(const World& world, const int local_x, const int local_y,
                          const val_t external_flow, const val_t* const current_energy,
                          const val_t* const current_total_flux,
                          val_t* const next_energy, val_t* const next_total_flux) {
    const size_t element = world.offset(local_x, local_y);
    const val_t energy = current_energy[element];
    val_t flux = external_flow;

    // This is the original connectivity order: +x, -x, +y, -y. Physical
    // boundary halos mirror their adjacent cell, which contributes exact zero
    // for an absent connection.
    flux += computeFlux(DEFAULT_MATERIAL, energy, 1.0,
                        current_energy[element + world.pitch]);
    flux += computeFlux(DEFAULT_MATERIAL, energy, 1.0,
                        current_energy[element - world.pitch]);
    flux += computeFlux(DEFAULT_MATERIAL, energy, 1.0, current_energy[element + 1]);
    flux += computeFlux(DEFAULT_MATERIAL, energy, 1.0, current_energy[element - 1]);

    next_energy[element] = energy + flux;
    next_total_flux[element] = current_total_flux[element] + std::abs(flux);
}

void updateRegion(const World& world, const int x_first, const int x_last,
                  const int y_first, const int y_last, const val_t* current_energy,
                  const val_t* current_total_flux, val_t* next_energy,
                  val_t* next_total_flux) {
    if (x_first > x_last || y_first > y_last) {
        return;
    }
    for (int local_x = x_first; local_x <= x_last; ++local_x) {
        for (int local_y = y_first; local_y <= y_last; ++local_y) {
            updateElement(world, local_x, local_y, DEFAULT_MATERIAL.external_flow,
                          current_energy, current_total_flux, next_energy,
                          next_total_flux);
        }
    }
}

void mirrorPhysicalBoundaries(const World& world, val_t* const current_energy) {
    if (!world.active()) {
        return;
    }
    if (world.x_minus == MPI_PROC_NULL) {
        std::copy_n(current_energy + world.offset(1, 1), world.local_ny,
                    current_energy + world.offset(0, 1));
    }
    if (world.x_plus == MPI_PROC_NULL) {
        std::copy_n(current_energy + world.offset(world.local_nx, 1), world.local_ny,
                    current_energy + world.offset(world.local_nx + 1, 1));
    }
    if (world.y_minus == MPI_PROC_NULL) {
        for (int local_x = 1; local_x <= world.local_nx; ++local_x) {
            current_energy[world.offset(local_x, 0)] =
                current_energy[world.offset(local_x, 1)];
        }
    }
    if (world.y_plus == MPI_PROC_NULL) {
        for (int local_x = 1; local_x <= world.local_nx; ++local_x) {
            current_energy[world.offset(local_x, world.local_ny + 1)] =
                current_energy[world.offset(local_x, world.local_ny)];
        }
    }
}

void updateGlobalCorner(const World& world, const int global_x, const int global_y,
                        const val_t external_flow, const val_t* current_energy,
                        const val_t* current_total_flux, val_t* next_energy,
                        val_t* next_total_flux) {
    if (global_x < world.x_begin || global_x >= world.x_begin + world.local_nx ||
        global_y < world.y_begin || global_y >= world.y_begin + world.local_ny) {
        return;
    }
    updateElement(world, global_x - world.x_begin + 1,
                  global_y - world.y_begin + 1, external_flow, current_energy,
                  current_total_flux, next_energy, next_total_flux);
}

void runSimulation(World& world, HaloExchange& halo_exchange, const int n_iters) {
    for (int iteration = 0; iteration < n_iters; ++iteration) {
        const int current_buffer = world.current_buffer;
        const int next_buffer = current_buffer ^ 1;
        val_t* const current_energy = world.energy[current_buffer].data();
        const val_t* const current_total_flux =
            world.total_flux[current_buffer].data();
        val_t* const next_energy = world.energy[next_buffer].data();
        val_t* const next_total_flux = world.total_flux[next_buffer].data();

        mirrorPhysicalBoundaries(world, current_energy);
        halo_exchange.start(current_buffer);

        // Cells strictly away from communicated faces are independent of the
        // incoming halos and overlap useful work with network progress.
        const int core_x_first = world.x_minus == MPI_PROC_NULL ? 1 : 2;
        const int core_x_last =
            world.x_plus == MPI_PROC_NULL ? world.local_nx : world.local_nx - 1;
        const int core_y_first = world.y_minus == MPI_PROC_NULL ? 1 : 2;
        const int core_y_last =
            world.y_plus == MPI_PROC_NULL ? world.local_ny : world.local_ny - 1;
        const bool has_core = core_x_first <= core_x_last &&
                              core_y_first <= core_y_last;

        if (has_core) {
            updateRegion(world, core_x_first, core_x_last, core_y_first, core_y_last,
                         current_energy, current_total_flux, next_energy,
                         next_total_flux);
        }

        halo_exchange.wait(current_buffer);

        if (!has_core) {
            updateRegion(world, 1, world.local_nx, 1, world.local_ny, current_energy,
                         current_total_flux, next_energy, next_total_flux);
        } else {
            // Four disjoint rectangles cover the communicated faces around the
            // already-computed core.
            updateRegion(world, 1, core_x_first - 1, 1, world.local_ny,
                         current_energy, current_total_flux, next_energy,
                         next_total_flux);
            updateRegion(world, core_x_last + 1, world.local_nx, 1, world.local_ny,
                         current_energy, current_total_flux, next_energy,
                         next_total_flux);
            updateRegion(world, core_x_first, core_x_last, 1, core_y_first - 1,
                         current_energy, current_total_flux, next_energy,
                         next_total_flux);
            updateRegion(world, core_x_first, core_x_last, core_y_last + 1,
                         world.local_ny, current_energy, current_total_flux,
                         next_energy, next_total_flux);
        }

        // The generated mesh has only four non-default materials. Recomputing
        // those cells keeps the hot stencil branch-free. Assignment order is
        // intentionally identical to the serial builder (important for N=1).
        const int last = world.global_n - 1;
        updateGlobalCorner(world, 0, 0, INFLOW, current_energy, current_total_flux,
                           next_energy, next_total_flux);
        updateGlobalCorner(world, 0, last, OUTFLOW, current_energy,
                           current_total_flux, next_energy, next_total_flux);
        updateGlobalCorner(world, last, 0, OUTFLOW, current_energy,
                           current_total_flux, next_energy, next_total_flux);
        updateGlobalCorner(world, last, last, INFLOW, current_energy,
                           current_total_flux, next_energy, next_total_flux);

        world.current_buffer = next_buffer;
    }
}

uint64_t computeLocalHash(const World& world) {
    const auto& energy = world.energy[world.current_buffer];
    const auto& total_flux = world.total_flux[world.current_buffer];
    uint64_t hash = 0;
    for (int local_x = 1; local_x <= world.local_nx; ++local_x) {
        const uint64_t global_x =
            static_cast<uint64_t>(world.x_begin + local_x - 1);
        for (int local_y = 1; local_y <= world.local_ny; ++local_y) {
            const size_t local_index = world.offset(local_x, local_y);
            const uint64_t global_index =
                global_x * static_cast<uint64_t>(world.global_n) +
                static_cast<uint64_t>(world.y_begin + local_y - 1);
            const uint64_t energy_bits = std::bit_cast<uint64_t>(energy[local_index]);
            const uint64_t flux_bits = std::bit_cast<uint64_t>(total_flux[local_index]);
            hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

std::vector<val_t> gatherField(const World& world,
                               const std::vector<val_t>& local_field,
                               MPI_Comm cart_comm, const int cart_rank,
                               const int cart_size, const int dimensions[2],
                               bool& success) {
    const uint64_t global_count = static_cast<uint64_t>(world.global_n) *
                                  static_cast<uint64_t>(world.global_n);
    success = global_count <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (!success) {
        if (cart_rank == 0) {
            std::fprintf(stderr,
                         "Diagnostic output requires no more than INT_MAX elements\n");
        }
        return {};
    }

    const int local_count = world.local_nx * world.local_ny;
    std::vector<val_t> packed(static_cast<size_t>(local_count));
    for (int local_x = 1; local_x <= world.local_nx; ++local_x) {
        std::copy_n(local_field.data() + world.offset(local_x, 1), world.local_ny,
                    packed.data() +
                        static_cast<size_t>(local_x - 1) * world.local_ny);
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<val_t> gathered;
    if (cart_rank == 0) {
        counts.resize(cart_size);
        displacements.resize(cart_size);
        int displacement = 0;
        for (int rank = 0; rank < cart_size; ++rank) {
            int coordinates[2] = {0, 0};
            MPI_Cart_coords(cart_comm, rank, 2, coordinates);
            const Block x_block =
                partition1D(world.global_n, dimensions[0], coordinates[0]);
            const Block y_block =
                partition1D(world.global_n, dimensions[1], coordinates[1]);
            counts[rank] = x_block.size * y_block.size;
            displacements[rank] = displacement;
            displacement += counts[rank];
        }
        gathered.resize(static_cast<size_t>(global_count));
    }

    val_t ignored = 0.0;
    MPI_Gatherv(packed.empty() ? &ignored : packed.data(), local_count, MPI_DOUBLE,
                cart_rank == 0 ? gathered.data() : nullptr,
                cart_rank == 0 ? counts.data() : nullptr,
                cart_rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                cart_comm);

    if (cart_rank != 0) {
        return {};
    }

    std::vector<val_t> global_field(static_cast<size_t>(global_count));
    for (int rank = 0; rank < cart_size; ++rank) {
        int coordinates[2] = {0, 0};
        MPI_Cart_coords(cart_comm, rank, 2, coordinates);
        const Block x_block =
            partition1D(world.global_n, dimensions[0], coordinates[0]);
        const Block y_block =
            partition1D(world.global_n, dimensions[1], coordinates[1]);
        const val_t* const rank_data = gathered.data() + displacements[rank];
        for (int local_x = 0; local_x < x_block.size; ++local_x) {
            const size_t destination =
                static_cast<size_t>(x_block.begin + local_x) * world.global_n +
                y_block.begin;
            std::copy_n(rank_data + static_cast<size_t>(local_x) * y_block.size,
                        y_block.size, global_field.data() + destination);
        }
    }
    return global_field;
}

bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& total_flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += total_flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
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
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
            break;
        }
    }

    if (show_help || !arguments_valid) {
        if (world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }
    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() - 2 ||
        n_iters < 0) {
        if (world_rank == 0) {
            std::fprintf(stderr,
                         "Grid size must be positive and iterations must be nonnegative\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dimensions);
    const int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &cart_comm);

    int cart_rank = 0;
    int cart_size = 1;
    int coordinates[2] = {0, 0};
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Comm_size(cart_comm, &cart_size);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coordinates);

    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) *
                             static_cast<uint64_t>(n_elems_root);
    if (cart_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root,
                    n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("MPI ranks: %d (%d x %d Cartesian grid)\n", cart_size,
                    dimensions[0], dimensions[1]);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building distributed unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, cart_comm, coordinates, dimensions);
    HaloExchange halo_exchange;
    halo_exchange.initialize(world, cart_comm);

    const uint64_t local_dynamic_bytes =
        4ULL * world.energy[0].size() * sizeof(val_t);
    uint64_t aggregate_dynamic_bytes = 0;
    uint64_t maximum_dynamic_bytes = 0;
    MPI_Reduce(&local_dynamic_bytes, &aggregate_dynamic_bytes, 1, MPI_UINT64_T,
               MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_dynamic_bytes, &maximum_dynamic_bytes, 1, MPI_UINT64_T,
               MPI_MAX, 0, cart_comm);
    if (cart_rank == 0) {
        constexpr double MEBIBYTE = 1024.0 * 1024.0;
        std::printf("Memory usage: %.2f MB aggregate (max %.2f MB/rank)\n\n",
                    aggregate_dynamic_bytes / MEBIBYTE,
                    maximum_dynamic_bytes / MEBIBYTE);
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(cart_comm);
    const double start_time = MPI_Wtime();
    runSimulation(world, halo_exchange, n_iters);
    const double local_duration = MPI_Wtime() - start_time;
    double duration = 0.0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    const uint64_t local_hash = computeLocalHash(world);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart_comm);

    if (cart_rank == 0) {
        const long duration_ms = static_cast<long>(duration * 1000.0);
        const int measured_iterations = std::max(n_iters - 1, 1);
        const double time_per_iteration_ms =
            duration * 1000.0 / measured_iterations;
        const double giga_elements_per_second =
            duration > 0.0
                ? (static_cast<double>(measured_iterations) *
                   static_cast<double>(n_elems)) /
                      duration / 1.0e9
                : std::numeric_limits<double>::infinity();
        const double gflops = giga_elements_per_second * 22.0;

        std::printf("Computation time: %ld ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration_ms);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
    }

    int exit_status = 0;
    if (print_results_requested || validate) {
        bool gather_succeeded = false;
        std::vector<val_t> global_energy =
            gatherField(world, world.energy[world.current_buffer], cart_comm,
                        cart_rank, cart_size, dimensions, gather_succeeded);
        if (!gather_succeeded) {
            exit_status = 1;
        } else {
            if (print_results_requested && cart_rank == 0) {
                print_results(global_energy, "ElementEnergy");
            }
            if (validate) {
                bool flux_gather_succeeded = false;
                std::vector<val_t> global_total_flux =
                    gatherField(world, world.total_flux[world.current_buffer],
                                cart_comm, cart_rank, cart_size, dimensions,
                                flux_gather_succeeded);
                if (!flux_gather_succeeded) {
                    exit_status = 1;
                } else if (cart_rank == 0 &&
                           !validateResults(global_energy, global_total_flux)) {
                    exit_status = 1;
                }
            }
        }
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, cart_comm);
    halo_exchange.destroy();
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return exit_status;
}
