#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <array>
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

#if defined(__GNUC__) || defined(__clang__)
#define UNSTRUCTURED_RESTRICT __restrict__
#else
#define UNSTRUCTURED_RESTRICT
#endif

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Each rank owns one rectangular block of the global mesh.  State is stored in
// structure-of-arrays form so halo rows are contiguous and the stencil streams
// through substantially less memory than the original replicated graph.
struct World {
    int n_elems_root = 0;
    int rank = 0;
    int size = 1;
    int dims[2] = {1, 1};
    int coords[2] = {0, 0};
    int x_begin = 0;
    int y_begin = 0;
    int local_nx = 0;
    int local_ny = 0;
    int x_minus = MPI_PROC_NULL;
    int x_plus = MPI_PROC_NULL;
    int y_minus = MPI_PROC_NULL;
    int y_plus = MPI_PROC_NULL;
    int current_buffer = 0;
    MPI_Comm cart_comm = MPI_COMM_NULL;

    std::array<Material, 3> materials{{
        {0.8, 0.0},
        {0.8, 0.5},
        {0.8, -0.5},
    }};
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> total_flux;
    std::vector<val_t> x_minus_halo;
    std::vector<val_t> x_plus_halo;
    std::vector<val_t> y_minus_halo;
    std::vector<val_t> y_plus_halo;
};

struct HaloExchange {
    std::array<std::vector<MPI_Request>, 2> requests;
    MPI_Datatype column_type = MPI_DATATYPE_NULL;
};

void decompose1D(const int n, const int parts, const int coordinate,
                 int& begin, int& count) {
    const int base = n / parts;
    const int remainder = n % parts;
    count = base + (coordinate < remainder ? 1 : 0);
    begin = coordinate * base + std::min(coordinate, remainder);
}

// Build the local portion of the square mesh and its Cartesian process
// topology.  No rank constructs or stores the global mesh.
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    MPI_Comm_size(MPI_COMM_WORLD, &world.size);

    world.dims[0] = 0;
    world.dims[1] = 0;
    MPI_Dims_create(world.size, 2, world.dims);
    const int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, world.dims, periods, 0, &world.cart_comm);
    MPI_Comm_rank(world.cart_comm, &world.rank);
    MPI_Cart_coords(world.cart_comm, world.rank, 2, world.coords);

    decompose1D(n_elems_root, world.dims[0], world.coords[0],
                world.x_begin, world.local_nx);
    decompose1D(n_elems_root, world.dims[1], world.coords[1],
                world.y_begin, world.local_ny);

    MPI_Cart_shift(world.cart_comm, 0, 1, &world.x_minus, &world.x_plus);
    MPI_Cart_shift(world.cart_comm, 1, 1, &world.y_minus, &world.y_plus);

    // Empty blocks are possible only when more ranks than elements are used.
    // Physical boundaries also have no neighbor even if an adjacent Cartesian
    // coordinate happens to own an empty block.
    if (world.local_nx == 0 || world.local_ny == 0) {
        world.x_minus = world.x_plus = MPI_PROC_NULL;
        world.y_minus = world.y_plus = MPI_PROC_NULL;
    } else {
        if (world.x_begin == 0) {
            world.x_minus = MPI_PROC_NULL;
        }
        if (world.x_begin + world.local_nx == n_elems_root) {
            world.x_plus = MPI_PROC_NULL;
        }
        if (world.y_begin == 0) {
            world.y_minus = MPI_PROC_NULL;
        }
        if (world.y_begin + world.local_ny == n_elems_root) {
            world.y_plus = MPI_PROC_NULL;
        }
    }

    const size_t local_size = static_cast<size_t>(world.local_nx) * world.local_ny;
    for (int buffer = 0; buffer < 2; ++buffer) {
        world.energy[buffer].assign(local_size, 0.0);
    }
    world.total_flux.assign(local_size, 0.0);
    world.x_minus_halo.resize(world.local_ny);
    world.x_plus_halo.resize(world.local_ny);
    world.y_minus_halo.resize(world.local_nx);
    world.y_plus_halo.resize(world.local_nx);
}

enum HaloTag : int {
    LOW_X = 100,
    HIGH_X = 101,
    LOW_Y = 102,
    HIGH_Y = 103,
};

// Persistent requests remove per-iteration request construction overhead.  A
// request set is created for each ping-pong energy buffer.
HaloExchange createHaloExchange(World& world) {
    HaloExchange exchange;
    if (world.local_nx == 0 || world.local_ny == 0) {
        return exchange;
    }

    MPI_Type_vector(world.local_nx, 1, world.local_ny, MPI_DOUBLE,
                    &exchange.column_type);
    MPI_Type_commit(&exchange.column_type);

    for (int buffer = 0; buffer < 2; ++buffer) {
        auto& requests = exchange.requests[buffer];
        requests.reserve(8);

        auto add_recv = [&](val_t* data, int count, MPI_Datatype datatype,
                            int source, int tag) {
            if (source != MPI_PROC_NULL) {
                MPI_Request request = MPI_REQUEST_NULL;
                MPI_Recv_init(data, count, datatype, source, tag,
                              world.cart_comm, &request);
                requests.push_back(request);
            }
        };
        auto add_send = [&](const val_t* data, int count, MPI_Datatype datatype,
                            int destination, int tag) {
            if (destination != MPI_PROC_NULL) {
                MPI_Request request = MPI_REQUEST_NULL;
                MPI_Send_init(data, count, datatype, destination, tag,
                              world.cart_comm, &request);
                requests.push_back(request);
            }
        };

        val_t* const data = world.energy[buffer].data();
        const size_t last_row = static_cast<size_t>(world.local_nx - 1) *
                                world.local_ny;

        add_recv(world.x_minus_halo.data(), world.local_ny, MPI_DOUBLE,
                 world.x_minus, HIGH_X);
        add_recv(world.x_plus_halo.data(), world.local_ny, MPI_DOUBLE,
                 world.x_plus, LOW_X);
        add_recv(world.y_minus_halo.data(), world.local_nx, MPI_DOUBLE,
                 world.y_minus, HIGH_Y);
        add_recv(world.y_plus_halo.data(), world.local_nx, MPI_DOUBLE,
                 world.y_plus, LOW_Y);

        add_send(data, world.local_ny, MPI_DOUBLE, world.x_minus, LOW_X);
        add_send(data + last_row, world.local_ny, MPI_DOUBLE,
                 world.x_plus, HIGH_X);
        add_send(data, 1, exchange.column_type, world.y_minus, LOW_Y);
        add_send(data + world.local_ny - 1, 1, exchange.column_type,
                 world.y_plus, HIGH_Y);
    }
    return exchange;
}

void destroyHaloExchange(HaloExchange& exchange) {
    for (auto& request_set : exchange.requests) {
        for (MPI_Request& request : request_set) {
            MPI_Request_free(&request);
        }
    }
    if (exchange.column_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&exchange.column_type);
    }
}

inline val_t computeFlux(const Material& material, const val_t this_energy,
                         const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) * material.transfer_coeff *
           connection_flux * 0.25;
}

inline idx_t materialIndex(const int global_x, const int global_y, const int n) {
    const bool low_x = global_x == 0;
    const bool high_x = global_x == n - 1;
    const bool low_y = global_y == 0;
    const bool high_y = global_y == n - 1;
    if ((low_x && low_y) || (high_x && high_y)) {
        return INFLOW_MAT_ID;
    }
    if ((low_x && high_y) || (high_x && low_y)) {
        return OUTFLOW_MAT_ID;
    }
    return DEFAULT_MAT_ID;
}

// Update a block-edge element after its required halos have arrived.  Fluxes
// are accumulated in the same x+, x-, y+, y- order as the original code.
inline void updateBoundaryElement(const World& world,
                                  const std::vector<val_t>& source_energy,
                                  std::vector<val_t>& destination_energy,
                                  std::vector<val_t>& total_flux,
                                  const int local_x, const int local_y) {
    const int n = world.n_elems_root;
    const int global_x = world.x_begin + local_x;
    const int global_y = world.y_begin + local_y;
    const size_t index = static_cast<size_t>(local_x) * world.local_ny + local_y;
    const val_t this_energy = source_energy[index];
    const Material& material = world.materials[materialIndex(global_x, global_y, n)];
    val_t flux = material.external_flow;

    if (global_x + 1 < n) {
        const val_t neighbor = local_x + 1 < world.local_nx
                                   ? source_energy[index + world.local_ny]
                                   : world.x_plus_halo[local_y];
        flux += computeFlux(material, this_energy, 1.0, neighbor);
    }
    if (global_x > 0) {
        const val_t neighbor = local_x > 0
                                   ? source_energy[index - world.local_ny]
                                   : world.x_minus_halo[local_y];
        flux += computeFlux(material, this_energy, 1.0, neighbor);
    }
    if (global_y + 1 < n) {
        const val_t neighbor = local_y + 1 < world.local_ny
                                   ? source_energy[index + 1]
                                   : world.y_plus_halo[local_x];
        flux += computeFlux(material, this_energy, 1.0, neighbor);
    }
    if (global_y > 0) {
        const val_t neighbor = local_y > 0
                                   ? source_energy[index - 1]
                                   : world.y_minus_halo[local_x];
        flux += computeFlux(material, this_energy, 1.0, neighbor);
    }

    destination_energy[index] = this_energy + flux;
    total_flux[index] += std::abs(flux);
}

void runSimulation(World& world, HaloExchange& exchange, const int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        const int source_buffer = world.current_buffer;
        const int destination_buffer = 1 - source_buffer;
        const auto& source_energy = world.energy[source_buffer];
        auto& destination_energy = world.energy[destination_buffer];
        auto& requests = exchange.requests[source_buffer];

        if (!requests.empty()) {
            MPI_Startall(static_cast<int>(requests.size()), requests.data());
        }

        // The strict block interior is independent of incoming halos and is
        // computed while all four edge transfers progress.
        const Material& material = world.materials[DEFAULT_MAT_ID];
        const val_t* UNSTRUCTURED_RESTRICT source = source_energy.data();
        val_t* UNSTRUCTURED_RESTRICT destination = destination_energy.data();
        val_t* UNSTRUCTURED_RESTRICT accumulated_flux = world.total_flux.data();
        const size_t row_stride = static_cast<size_t>(world.local_ny);
        for (int x = 1; x + 1 < world.local_nx; ++x) {
            const size_t row = static_cast<size_t>(x) * row_stride;
            for (int y = 1; y + 1 < world.local_ny; ++y) {
                const size_t index = row + y;
                const val_t this_energy = source[index];
                val_t flux = material.external_flow;
                flux += computeFlux(material, this_energy, 1.0,
                                    source[index + row_stride]);
                flux += computeFlux(material, this_energy, 1.0,
                                    source[index - row_stride]);
                flux += computeFlux(material, this_energy, 1.0,
                                    source[index + 1]);
                flux += computeFlux(material, this_energy, 1.0,
                                    source[index - 1]);
                destination[index] = this_energy + flux;
                accumulated_flux[index] += std::abs(flux);
            }
        }

        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                        MPI_STATUSES_IGNORE);
        }

        if (world.local_nx > 0 && world.local_ny > 0) {
            for (int y = 0; y < world.local_ny; ++y) {
                updateBoundaryElement(world, source_energy, destination_energy,
                                      world.total_flux, 0, y);
            }
            if (world.local_nx > 1) {
                const int last_x = world.local_nx - 1;
                for (int y = 0; y < world.local_ny; ++y) {
                    updateBoundaryElement(world, source_energy, destination_energy,
                                          world.total_flux, last_x, y);
                }
            }
            for (int x = 1; x + 1 < world.local_nx; ++x) {
                updateBoundaryElement(world, source_energy, destination_energy,
                                      world.total_flux, x, 0);
                if (world.local_ny > 1) {
                    updateBoundaryElement(world, source_energy, destination_energy,
                                          world.total_flux,
                                          x, world.local_ny - 1);
                }
            }
        }
        world.current_buffer = destination_buffer;
    }
}

// Gather a distributed block field in original global row-major order.  This
// is used only by explicitly requested reporting/validation, never in timing.
std::vector<val_t> gatherField(const World& world,
                               const std::vector<val_t>& local_field) {
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<val_t> packed;
    if (world.rank == 0) {
        counts.resize(world.size);
        displacements.resize(world.size);
        int total = 0;
        for (int rank = 0; rank < world.size; ++rank) {
            int coordinates[2];
            MPI_Cart_coords(world.cart_comm, rank, 2, coordinates);
            int begin = 0;
            int nx = 0;
            int ny = 0;
            decompose1D(world.n_elems_root, world.dims[0], coordinates[0], begin, nx);
            decompose1D(world.n_elems_root, world.dims[1], coordinates[1], begin, ny);
            counts[rank] = nx * ny;
            displacements[rank] = total;
            total += counts[rank];
        }
        packed.resize(total);
    }

    const int local_count = world.local_nx * world.local_ny;
    MPI_Gatherv(local_field.data(), local_count, MPI_DOUBLE,
                world.rank == 0 ? packed.data() : nullptr,
                world.rank == 0 ? counts.data() : nullptr,
                world.rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, world.cart_comm);

    std::vector<val_t> global;
    if (world.rank == 0) {
        const size_t global_size = static_cast<size_t>(world.n_elems_root) *
                                   world.n_elems_root;
        global.resize(global_size);
        for (int rank = 0; rank < world.size; ++rank) {
            int coordinates[2];
            MPI_Cart_coords(world.cart_comm, rank, 2, coordinates);
            int x_begin = 0;
            int y_begin = 0;
            int nx = 0;
            int ny = 0;
            decompose1D(world.n_elems_root, world.dims[0], coordinates[0],
                        x_begin, nx);
            decompose1D(world.n_elems_root, world.dims[1], coordinates[1],
                        y_begin, ny);
            const val_t* source = packed.data() + displacements[rank];
            for (int x = 0; x < nx; ++x) {
                val_t* destination = global.data() +
                    static_cast<size_t>(x_begin + x) * world.n_elems_root + y_begin;
                std::copy_n(source + static_cast<size_t>(x) * ny, ny, destination);
            }
        }
    }
    return global;
}

uint64_t computeHash(const World& world) {
    const auto& energy = world.energy[world.current_buffer];
    const auto& flux = world.total_flux;
    uint64_t local_hash = 0;
    for (int x = 0; x < world.local_nx; ++x) {
        for (int y = 0; y < world.local_ny; ++y) {
            const size_t local_index = static_cast<size_t>(x) * world.local_ny + y;
            const uint64_t global_index =
                static_cast<uint64_t>(world.x_begin + x) * world.n_elems_root +
                world.y_begin + y;
            uint64_t energy_bits = 0;
            uint64_t flux_bits = 0;
            std::memcpy(&energy_bits, &energy[local_index], sizeof(energy_bits));
            std::memcpy(&flux_bits, &flux[local_index], sizeof(flux_bits));
            local_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR,
               0, world.cart_comm);
    return global_hash;
}

bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
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

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
            break;
        }
    }

    if (show_help || !arguments_valid || n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) {
            if (n_elems_root <= 0 || n_iters < 0) {
                std::printf("Grid size must be positive and iterations non-negative.\n");
            }
            printUsage(argv[0]);
        }
        const int exit_code = (show_help && arguments_valid &&
                               n_elems_root > 0 && n_iters >= 0) ? 0 : 1;
        MPI_Finalize();
        return exit_code;
    }

    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %" PRIu64 " elements\n",
                    n_elems_root, n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("\nBuilding distributed unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root);
    HaloExchange exchange = createHaloExchange(world);

    const uint64_t local_memory =
        static_cast<uint64_t>(world.local_nx) * world.local_ny *
            (3 * sizeof(val_t)) +
        static_cast<uint64_t>(2 * (world.local_nx + world.local_ny)) * sizeof(val_t);
    uint64_t aggregate_memory = 0;
    uint64_t maximum_memory = 0;
    MPI_Reduce(&local_memory, &aggregate_memory, 1, MPI_UINT64_T, MPI_SUM,
               0, world.cart_comm);
    MPI_Reduce(&local_memory, &maximum_memory, 1, MPI_UINT64_T, MPI_MAX,
               0, world.cart_comm);

    if (world.rank == 0) {
        std::printf("MPI topology: %d ranks (%d x %d)\n",
                    world.size, world.dims[0], world.dims[1]);
        std::printf("Memory usage: %.2f MB aggregate (maximum rank: %.2f MB)\n\n",
                    aggregate_memory / (1024.0 * 1024.0),
                    maximum_memory / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(world.cart_comm);
    const double start = MPI_Wtime();
    runSimulation(world, exchange, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX,
               0, world.cart_comm);

    destroyHaloExchange(exchange);
    const uint64_t result_hash = computeHash(world);

    if (world.rank == 0) {
        const double duration_ms = elapsed_seconds * 1000.0;
        std::printf("Computation time: %.3f ms\n", duration_ms);
        const int measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / measured_iters;
        const double giga_elements_per_second = elapsed_seconds > 0.0
            ? (static_cast<double>(measured_iters) * static_cast<double>(n_elems)) /
                  elapsed_seconds / 1e9
            : 0.0;
        const double gflops = giga_elements_per_second * 22.0;
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }

    std::vector<val_t> global_energy;
    if (print_results_requested || validate) {
        global_energy = gatherField(world, world.energy[world.current_buffer]);
    }
    if (print_results_requested && world.rank == 0) {
        print_results(global_energy, "ElementEnergy");
    }

    bool valid = true;
    if (validate) {
        std::vector<val_t> global_flux =
            gatherField(world, world.total_flux);
        if (world.rank == 0) {
            valid = validateResults(global_energy, global_flux);
        }
    }

    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, world.cart_comm);
    MPI_Comm_free(&world.cart_comm);
    MPI_Finalize();
    return valid_int ? 0 : 1;
}
