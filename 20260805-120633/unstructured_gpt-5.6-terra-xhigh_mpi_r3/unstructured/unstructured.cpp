#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr val_t EXTERNAL_FLOW = 0.5;

// Each rank owns a rectangular subdomain.  Energy is the only state required
// by a neighbour, so total_flux remains private and is not communicated.
struct World {
    int global_n = 0;
    int x_begin = 0;
    int y_begin = 0;
    int nx = 0;
    int ny = 0;

    int north = MPI_PROC_NULL;
    int south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL;
    int east = MPI_PROC_NULL;

    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> total_flux;
    std::vector<val_t> total_flux_swap;

    std::vector<val_t> north_halo;
    std::vector<val_t> south_halo;
    std::vector<val_t> west_halo;
    std::vector<val_t> east_halo;

    // A column of the row-major local energy array.  It avoids a pack/copy on
    // the two vertical faces and is committed once during mesh construction.
    MPI_Datatype column_type = MPI_DATATYPE_NULL;

    // The two energy buffers alternate each iteration.  Persistent requests
    // remove per-step request construction from the timed stencil loop.
    std::array<std::array<MPI_Request, 8>, 2> halo_requests{};
    std::array<int, 2> halo_request_counts = {0, 0};
};

inline int blockBegin(const int extent, const int coordinate, const int dimensions) {
    const int base = extent / dimensions;
    const int remainder = extent % dimensions;
    return coordinate * base + std::min(coordinate, remainder);
}

inline int blockSize(const int extent, const int coordinate, const int dimensions) {
    return extent / dimensions + (coordinate < extent % dimensions ? 1 : 0);
}

inline bool isActive(const World& world) {
    return world.nx != 0 && world.ny != 0;
}

// Build the local portion of the same square mesh.  The connectivity is
// implicit because the original mesh has only the four regular neighbours.
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm cart_comm,
                   const std::array<int, 2>& coords,
                   const std::array<int, 2>& cart_dims) {
    world.global_n = n_elems_root;
    world.x_begin = blockBegin(n_elems_root, coords[0], cart_dims[0]);
    world.y_begin = blockBegin(n_elems_root, coords[1], cart_dims[1]);
    world.nx = blockSize(n_elems_root, coords[0], cart_dims[0]);
    world.ny = blockSize(n_elems_root, coords[1], cart_dims[1]);

    if (!isActive(world)) {
        return;
    }

    const size_t local_elements = static_cast<size_t>(world.nx) * world.ny;
    world.energy.assign(local_elements, 0.0);
    world.energy_swap.resize(local_elements);
    world.total_flux.assign(local_elements, 0.0);
    world.total_flux_swap.resize(local_elements);

    world.north_halo.resize(world.ny);
    world.south_halo.resize(world.ny);
    world.west_halo.resize(world.nx);
    world.east_halo.resize(world.nx);

    // A Cartesian communicator is used only to map physical neighbours.  A
    // process grid can be wider than a tiny test mesh; ranks with an empty
    // block simply remain idle while active ranks skip over no data.
    std::array<int, 2> neighbour_coords = coords;
    if (world.x_begin > 0) {
        --neighbour_coords[0];
        MPI_Cart_rank(cart_comm, neighbour_coords.data(), &world.north);
        ++neighbour_coords[0];
    }
    if (world.x_begin + world.nx < n_elems_root) {
        ++neighbour_coords[0];
        MPI_Cart_rank(cart_comm, neighbour_coords.data(), &world.south);
        --neighbour_coords[0];
    }
    if (world.y_begin > 0) {
        --neighbour_coords[1];
        MPI_Cart_rank(cart_comm, neighbour_coords.data(), &world.west);
        ++neighbour_coords[1];
    }
    if (world.y_begin + world.ny < n_elems_root) {
        ++neighbour_coords[1];
        MPI_Cart_rank(cart_comm, neighbour_coords.data(), &world.east);
    }

    MPI_Type_vector(world.nx, 1, world.ny, MPI_DOUBLE, &world.column_type);
    MPI_Type_commit(&world.column_type);
}

void destroyWorld(World& world) {
    for (int buffer = 0; buffer < 2; ++buffer) {
        for (int request = 0; request < world.halo_request_counts[buffer]; ++request) {
            MPI_Request_free(&world.halo_requests[buffer][request]);
        }
    }
    if (world.column_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&world.column_type);
    }
}

inline val_t computeFlux(const val_t this_energy, const val_t other_energy) {
    // Keep the arithmetic order used by the original ElementStatic traversal.
    return (other_energy - this_energy) * TRANSFER_COEFF * CONNECTION_FLUX * 0.25;
}

inline val_t externalFlow(const World& world, const int local_x, const int local_y) {
    const int x = world.x_begin + local_x;
    const int y = world.y_begin + local_y;
    const int last = world.global_n - 1;

    // This is equivalent to the original four assignments, including n == 1
    // where the final corner assignment leaves an inflow material.
    if ((x == 0 && y == 0) || (x == last && y == last)) {
        return EXTERNAL_FLOW;
    }
    if ((x == 0 && y == last) || (x == last && y == 0)) {
        return -EXTERNAL_FLOW;
    }
    return 0.0;
}

void initializePersistentHaloExchange(World& world, MPI_Comm comm) {
    if (!isActive(world)) {
        return;
    }

    constexpr int TAG_TO_NORTH = 0;
    constexpr int TAG_TO_SOUTH = 1;
    constexpr int TAG_TO_WEST = 2;
    constexpr int TAG_TO_EAST = 3;

    for (int buffer = 0; buffer < 2; ++buffer) {
        val_t* const energy = buffer == 0 ? world.energy.data() : world.energy_swap.data();
        int& request_count = world.halo_request_counts[buffer];

        // Receive first so every boundary can make progress immediately.
        // North and south are contiguous rows; west and east use the
        // prebuilt column datatype on the send side and contiguous buffers on
        // receive.
        if (world.north != MPI_PROC_NULL) {
            MPI_Recv_init(world.north_halo.data(), world.ny, MPI_DOUBLE, world.north,
                          TAG_TO_SOUTH, comm, &world.halo_requests[buffer][request_count++]);
        }
        if (world.south != MPI_PROC_NULL) {
            MPI_Recv_init(world.south_halo.data(), world.ny, MPI_DOUBLE, world.south,
                          TAG_TO_NORTH, comm, &world.halo_requests[buffer][request_count++]);
        }
        if (world.west != MPI_PROC_NULL) {
            MPI_Recv_init(world.west_halo.data(), world.nx, MPI_DOUBLE, world.west,
                          TAG_TO_EAST, comm, &world.halo_requests[buffer][request_count++]);
        }
        if (world.east != MPI_PROC_NULL) {
            MPI_Recv_init(world.east_halo.data(), world.nx, MPI_DOUBLE, world.east,
                          TAG_TO_WEST, comm, &world.halo_requests[buffer][request_count++]);
        }

        if (world.north != MPI_PROC_NULL) {
            MPI_Send_init(energy, world.ny, MPI_DOUBLE, world.north, TAG_TO_NORTH, comm,
                          &world.halo_requests[buffer][request_count++]);
        }
        if (world.south != MPI_PROC_NULL) {
            MPI_Send_init(energy + static_cast<size_t>(world.nx - 1) * world.ny,
                          world.ny, MPI_DOUBLE, world.south, TAG_TO_SOUTH, comm,
                          &world.halo_requests[buffer][request_count++]);
        }
        if (world.west != MPI_PROC_NULL) {
            MPI_Send_init(energy, 1, world.column_type, world.west, TAG_TO_WEST, comm,
                          &world.halo_requests[buffer][request_count++]);
        }
        if (world.east != MPI_PROC_NULL) {
            MPI_Send_init(energy + (world.ny - 1), 1, world.column_type, world.east,
                          TAG_TO_EAST, comm, &world.halo_requests[buffer][request_count++]);
        }
    }
}

inline void updateInteriorCell(World& world, const int x, const int y) {
    const size_t index = static_cast<size_t>(x) * world.ny + y;
    const val_t current_energy = world.energy[index];
    val_t flux = computeFlux(current_energy, world.energy[index + world.ny]);
    flux += computeFlux(current_energy, world.energy[index - world.ny]);
    flux += computeFlux(current_energy, world.energy[index + 1]);
    flux += computeFlux(current_energy, world.energy[index - 1]);

    world.energy_swap[index] = current_energy + flux;
    world.total_flux_swap[index] = world.total_flux[index] + std::abs(flux);
}

inline void updateBoundaryCell(World& world, const int x, const int y) {
    const size_t index = static_cast<size_t>(x) * world.ny + y;
    const val_t current_energy = world.energy[index];
    val_t flux = externalFlow(world, x, y);

    // Addition order is down, up, right, left, matching buildSquare2D().
    if (x + 1 < world.nx) {
        flux += computeFlux(current_energy, world.energy[index + world.ny]);
    } else if (world.x_begin + x + 1 < world.global_n) {
        flux += computeFlux(current_energy, world.south_halo[y]);
    }
    if (x > 0) {
        flux += computeFlux(current_energy, world.energy[index - world.ny]);
    } else if (world.x_begin > 0) {
        flux += computeFlux(current_energy, world.north_halo[y]);
    }
    if (y + 1 < world.ny) {
        flux += computeFlux(current_energy, world.energy[index + 1]);
    } else if (world.y_begin + y + 1 < world.global_n) {
        flux += computeFlux(current_energy, world.east_halo[x]);
    }
    if (y > 0) {
        flux += computeFlux(current_energy, world.energy[index - 1]);
    } else if (world.y_begin > 0) {
        flux += computeFlux(current_energy, world.west_halo[x]);
    }

    world.energy_swap[index] = current_energy + flux;
    world.total_flux_swap[index] = world.total_flux[index] + std::abs(flux);
}

void runSimulation(World& world, const int n_iters) {
    int energy_buffer = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int request_count = world.halo_request_counts[energy_buffer];
        if (request_count != 0) {
            MPI_Startall(request_count, world.halo_requests[energy_buffer].data());
        }

        // These cells never consume halos, so their work overlaps all four
        // nonblocking face exchanges.
        for (int x = 1; x + 1 < world.nx; ++x) {
            for (int y = 1; y + 1 < world.ny; ++y) {
                updateInteriorCell(world, x, y);
            }
        }

        if (request_count != 0) {
            MPI_Waitall(request_count, world.halo_requests[energy_buffer].data(),
                        MPI_STATUSES_IGNORE);
        }

        // Complete only the one-cell local perimeter after halo data is
        // ready.  Avoiding a second full-domain traversal matters on large
        // subdomains where the perimeter is small relative to the interior.
        if (isActive(world)) {
            for (int y = 0; y < world.ny; ++y) {
                updateBoundaryCell(world, 0, y);
            }
            if (world.nx > 1) {
                for (int y = 0; y < world.ny; ++y) {
                    updateBoundaryCell(world, world.nx - 1, y);
                }
            }
            for (int x = 1; x + 1 < world.nx; ++x) {
                updateBoundaryCell(world, x, 0);
                if (world.ny > 1) {
                    updateBoundaryCell(world, x, world.ny - 1);
                }
            }
        }

        std::swap(world.energy, world.energy_swap);
        std::swap(world.total_flux, world.total_flux_swap);
        energy_buffer ^= 1;
    }
}

uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t local_index = static_cast<size_t>(x) * world.ny + y;
            const idx_t global_index = static_cast<idx_t>(world.x_begin + x) *
                                           world.global_n + world.y_begin + y;
            uint64_t energy_bits;
            uint64_t flux_bits;
            std::memcpy(&energy_bits, &world.energy[local_index], sizeof(energy_bits));
            std::memcpy(&flux_bits, &world.total_flux[local_index], sizeof(flux_bits));
            hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

bool validateResults(const World& world, MPI_Comm comm, const int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    int local_finite = 1;

    for (size_t i = 0; i < world.energy.size(); ++i) {
        local_energy_sum += world.energy[i];
        local_flux_sum += world.total_flux[i];
        local_energy_max = std::max(world.energy[i], local_energy_max);
        local_energy_min = std::min(world.energy[i], local_energy_min);
        local_finite = local_finite && std::isfinite(world.energy[i]) &&
                       std::isfinite(world.total_flux[i]);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    int all_finite = 0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_finite, &all_finite, 1, MPI_INT, MPI_LAND, 0, comm);

    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!all_finite || !std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!all_finite || !std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!all_finite || !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

std::vector<val_t> gatherEnergy(const World& world, MPI_Comm comm, const int rank,
                                const int nranks) {
    const int local_count = static_cast<int>(world.energy.size());
    std::vector<int> counts(rank == 0 ? nranks : 0);
    std::array<int, 4> local_metadata = {world.x_begin, world.y_begin, world.nx, world.ny};
    std::vector<int> metadata(rank == 0 ? 4 * nranks : 0);

    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1,
               MPI_INT, 0, comm);
    MPI_Gather(local_metadata.data(), 4, MPI_INT,
               rank == 0 ? metadata.data() : nullptr, 4, MPI_INT, 0, comm);

    std::vector<int> displacements(rank == 0 ? nranks : 0);
    int gathered_count = 0;
    if (rank == 0) {
        for (int process = 0; process < nranks; ++process) {
            displacements[process] = gathered_count;
            gathered_count += counts[process];
        }
    }

    std::vector<val_t> packed_energy(rank == 0 ? gathered_count : 0);
    MPI_Gatherv(world.energy.data(), local_count, MPI_DOUBLE,
                rank == 0 ? packed_energy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);

    if (rank != 0) {
        return {};
    }

    const size_t global_elements = static_cast<size_t>(world.global_n) * world.global_n;
    std::vector<val_t> global_energy(global_elements);
    for (int process = 0; process < nranks; ++process) {
        const int x_begin = metadata[4 * process];
        const int y_begin = metadata[4 * process + 1];
        const int nx = metadata[4 * process + 2];
        const int ny = metadata[4 * process + 3];
        const int packed_begin = displacements[process];
        for (int x = 0; x < nx; ++x) {
            const size_t global_offset = static_cast<size_t>(x_begin + x) * world.global_n + y_begin;
            const size_t packed_offset = static_cast<size_t>(packed_begin) + static_cast<size_t>(x) * ny;
            std::copy_n(packed_energy.data() + packed_offset, ny,
                        global_energy.data() + global_offset);
        }
    }
    return global_energy;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool parse_ok = true;

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
            parse_ok = false;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }
    if (n_elems_root <= 0 || n_iters < 0) {
        parse_ok = false;
        if (rank == 0) {
            printf("Grid size must be positive and iterations must be non-negative\n");
        }
    }
    if (show_help || !parse_ok) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_ok ? 0 : 1;
    }

    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;

    int cart_dims[2] = {0, 0};
    MPI_Dims_create(nranks, 2, cart_dims);
    int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, cart_dims, periods, 0, &cart_comm);

    int raw_coords[2] = {0, 0};
    MPI_Cart_coords(cart_comm, rank, 2, raw_coords);
    const std::array<int, 2> coords = {raw_coords[0], raw_coords[1]};
    const std::array<int, 2> dims = {cart_dims[0], cart_dims[1]};

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d (Cartesian grid: %d x %d)\n", nranks, cart_dims[0], cart_dims[1]);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, cart_comm, coords, dims);
    initializePersistentHaloExchange(world, cart_comm);

    const uint64_t local_memory =
        static_cast<uint64_t>(world.energy.size() + world.energy_swap.size() +
                              world.total_flux.size() + world.total_flux_swap.size() +
                              world.north_halo.size() + world.south_halo.size() +
                              world.west_halo.size() + world.east_halo.size()) * sizeof(val_t);
    uint64_t aggregate_memory = 0;
    uint64_t max_rank_memory = 0;
    MPI_Reduce(&local_memory, &aggregate_memory, 1, MPI_UINT64_T, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_memory, &max_rank_memory, 1, MPI_UINT64_T, MPI_MAX, 0, cart_comm);
    if (rank == 0) {
        printf("Memory usage: %.2f MB distributed (max rank: %.2f MB)\n\n",
               aggregate_memory / (1024.0 * 1024.0), max_rank_memory / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(cart_comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_duration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    const uint64_t local_hash = computeLocalHash(world);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart_comm);

    if (rank == 0) {
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double duration_ms = duration * 1000.0;
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = duration > 0.0
            ? (static_cast<double>(n_measured_iters) * n_elems) / duration / 1e9
            : std::numeric_limits<double>::infinity();
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
    }

    if (print_results_requested) {
        std::vector<val_t> energy_data = gatherEnergy(world, cart_comm, rank, nranks);
        if (rank == 0) {
            print_results(energy_data, "ElementEnergy");
        }
    }

    bool valid = true;
    if (validate) {
        valid = validateResults(world, cart_comm, rank);
    }

    destroyWorld(world);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
