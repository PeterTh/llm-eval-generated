#include <mpi.h>

#include <algorithm>
#include <array>
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

constexpr int MAX_CONNECTIONS = 8;
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Neighbor indices are offsets into the local energy array, which includes one
// ghost row on either side of the owned rows. This preserves the original
// indirect, unstructured-mesh access pattern without a branch for ghost data.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::array<std::vector<val_t>, 2> energy;
    std::array<std::vector<val_t>, 2> total_flux;
    idx_t global_begin = 0;
    size_t local_elements = 0;
    size_t halo_elements = 0;
    int grid_size = 0;
    int first_row = 0;
    int local_rows = 0;

    val_t* ownedEnergy(const int buffer) {
        return energy[buffer].data() + halo_elements;
    }

    const val_t* ownedEnergy(const int buffer) const {
        return energy[buffer].data() + halo_elements;
    }
};

struct RowPartition {
    int first_row;
    int row_count;
};

RowPartition partitionRows(const int n_rows, const int rank, const int n_ranks) {
    const int base = n_rows / n_ranks;
    const int remainder = n_rows % n_ranks;
    const int row_count = base + (rank < remainder ? 1 : 0);
    const int first_row = rank * base + std::min(rank, remainder);
    return {first_row, row_count};
}

// Each rank constructs only its local portion. Since construction is
// deterministic, no global mesh distribution or replicated global state is
// required.
void buildSquare2D(World& world, const int n_elems_root, const int rank,
                   const int n_ranks) {
    const RowPartition partition = partitionRows(n_elems_root, rank, n_ranks);
    world.grid_size = n_elems_root;
    world.first_row = partition.first_row;
    world.local_rows = partition.row_count;
    world.halo_elements = partition.row_count == 0
                              ? 0
                              : static_cast<size_t>(n_elems_root);
    world.local_elements = static_cast<size_t>(partition.row_count) *
                           static_cast<size_t>(n_elems_root);
    world.global_begin = static_cast<idx_t>(partition.first_row) *
                         static_cast<idx_t>(n_elems_root);

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(world.local_elements);
    for (int buffer = 0; buffer < 2; ++buffer) {
        if (world.local_elements != 0) {
            world.energy[buffer].assign(
                world.local_elements + 2 * world.halo_elements, 0.0);
        }
        world.total_flux[buffer].assign(world.local_elements, 0.0);
    }

    constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    const int last = n_elems_root - 1;

    for (int local_x = 0; local_x < partition.row_count; ++local_x) {
        const int global_x = partition.first_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t local_idx = static_cast<size_t>(local_x) *
                                         static_cast<size_t>(n_elems_root) +
                                     static_cast<size_t>(y);
            ElementStatic& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            for (const auto& offset : offsets) {
                const int nx = global_x + offset[0];
                const int ny = y + offset[1];
                if (nx < 0 || nx >= n_elems_root || ny < 0 ||
                    ny >= n_elems_root) {
                    continue;
                }

                const int64_t global_neighbor =
                    static_cast<int64_t>(nx) * n_elems_root + ny;
                const int64_t local_neighbor =
                    global_neighbor - static_cast<int64_t>(world.global_begin) +
                    static_cast<int64_t>(world.halo_elements);
                const idx_t connection = elem.num_connections++;
                elem.connected_idx[connection] =
                    static_cast<idx_t>(local_neighbor);
                elem.connected_flux[connection] = 1.0;
            }

            // This ordering is equivalent to the four assignments in the
            // original implementation, including the 1x1-grid corner case.
            if ((global_x == 0 && y == 0) ||
                (global_x == last && y == last)) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if ((global_x == 0 && y == last) ||
                       (global_x == last && y == 0)) {
                elem.material_idx = OUTFLOW_MAT_ID;
            }
        }
    }
}

inline val_t computeFlux(const Material& mat, const val_t this_energy,
                         const val_t connection_flux,
                         const val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff *
           connection_flux * 0.25;
}

void updateElements(World& world, const int read_buffer,
                    const int write_buffer, const size_t begin,
                    const size_t end) {
    const std::vector<val_t>& energy_read = world.energy[read_buffer];
    std::vector<val_t>& energy_write = world.energy[write_buffer];
    const std::vector<val_t>& flux_read = world.total_flux[read_buffer];
    std::vector<val_t>& flux_write = world.total_flux[write_buffer];

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const Material& mat = world.materials[elem_static.material_idx];
        const size_t energy_idx = world.halo_elements + i;
        const val_t current_energy = energy_read[energy_idx];
        val_t total = mat.external_flow;

        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            total += computeFlux(mat, current_energy,
                                 elem_static.connected_flux[j],
                                 energy_read[elem_static.connected_idx[j]]);
        }

        energy_write[energy_idx] = current_energy + total;
        flux_write[i] = flux_read[i] + std::abs(total);
    }
}

// Exchange only the current-energy field. total_flux is purely local history
// and is never read by neighboring elements. Persistent requests remove MPI
// setup overhead from the iteration loop, while interior rows are computed
// concurrently with halo transfers.
int runSimulation(World& world, const int n_iters, const int rank) {
    if (world.local_elements == 0 || n_iters == 0) {
        return 0;
    }

    const int previous_rank = world.first_row == 0 ? MPI_PROC_NULL : rank - 1;
    const int next_rank =
        world.first_row + world.local_rows == world.grid_size ? MPI_PROC_NULL
                                                              : rank + 1;

    // With no neighboring rank, retain a zero-communication fast path while
    // still running under the mandatory MPI execution model.
    if (previous_rank == MPI_PROC_NULL && next_rank == MPI_PROC_NULL) {
        for (int iter = 0; iter < n_iters; ++iter) {
            const int read_buffer = iter & 1;
            updateElements(world, read_buffer, read_buffer ^ 1, 0,
                           world.local_elements);
        }
        return n_iters & 1;
    }

    constexpr int FIRST_ROW_TAG = 100;
    constexpr int LAST_ROW_TAG = 101;
    std::array<std::array<MPI_Request, 4>, 2> requests{};

    for (int buffer = 0; buffer < 2; ++buffer) {
        val_t* const data = world.energy[buffer].data();
        val_t* const owned = world.ownedEnergy(buffer);
        val_t* const bottom_halo =
            owned + world.local_elements;
        val_t* const last_owned_row =
            owned + world.local_elements - world.halo_elements;

        MPI_Recv_init(data, world.grid_size, MPI_DOUBLE, previous_rank,
                      LAST_ROW_TAG, MPI_COMM_WORLD,
                      &requests[buffer][0]);
        MPI_Recv_init(bottom_halo, world.grid_size, MPI_DOUBLE, next_rank,
                      FIRST_ROW_TAG, MPI_COMM_WORLD,
                      &requests[buffer][1]);
        MPI_Send_init(owned, world.grid_size, MPI_DOUBLE, previous_rank,
                      FIRST_ROW_TAG, MPI_COMM_WORLD,
                      &requests[buffer][2]);
        MPI_Send_init(last_owned_row, world.grid_size, MPI_DOUBLE, next_rank,
                      LAST_ROW_TAG, MPI_COMM_WORLD,
                      &requests[buffer][3]);
    }

    const size_t row_width = static_cast<size_t>(world.grid_size);
    for (int iter = 0; iter < n_iters; ++iter) {
        const int read_buffer = iter & 1;
        const int write_buffer = read_buffer ^ 1;
        MPI_Startall(static_cast<int>(requests[read_buffer].size()),
                     requests[read_buffer].data());

        if (world.local_rows > 2) {
            updateElements(world, read_buffer, write_buffer, row_width,
                           world.local_elements - row_width);
        }

        MPI_Waitall(static_cast<int>(requests[read_buffer].size()),
                    requests[read_buffer].data(), MPI_STATUSES_IGNORE);

        updateElements(world, read_buffer, write_buffer, 0, row_width);
        if (world.local_rows > 1) {
            updateElements(world, read_buffer, write_buffer,
                           world.local_elements - row_width,
                           world.local_elements);
        }
    }

    for (auto& buffer_requests : requests) {
        for (MPI_Request& request : buffer_requests) {
            MPI_Request_free(&request);
        }
    }
    return n_iters & 1;
}

bool validateResults(const World& world, const int active_buffer,
                     const int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    const val_t* const energy = world.local_elements == 0
                                    ? nullptr
                                    : world.ownedEnergy(active_buffer);
    for (size_t i = 0; i < world.local_elements; ++i) {
        local_energy_sum += energy[i];
        local_flux_sum += world.total_flux[active_buffer][i];
        local_energy_max = std::max(energy[i], local_energy_max);
        local_energy_min = std::min(energy[i], local_energy_min);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min,
                    energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        if (std::abs(energy_sum) > energy_epsilon) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected "
                        "conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) {
            std::printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

uint64_t computeLocalHash(const World& world, const int active_buffer) {
    uint64_t hash = 0;
    const val_t* const energy = world.local_elements == 0
                                    ? nullptr
                                    : world.ownedEnergy(active_buffer);
    for (size_t i = 0; i < world.local_elements; ++i) {
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits, &energy[i], sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.total_flux[active_buffer][i],
                    sizeof(flux_bits));
        const uint64_t global_idx = world.global_begin + i;
        hash ^= (energy_bits + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printDistributedResults(const World& world, const int active_buffer,
                             const int rank, const int n_ranks,
                             const int n_elems_root) {
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<val_t> global_energy;
    if (rank == 0) {
        counts.resize(n_ranks);
        displacements.resize(n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            const RowPartition partition =
                partitionRows(n_elems_root, r, n_ranks);
            counts[r] = partition.row_count * n_elems_root;
            displacements[r] = partition.first_row * n_elems_root;
        }
        global_energy.resize(static_cast<size_t>(n_elems_root) *
                             static_cast<size_t>(n_elems_root));
    }

    const val_t* const local_energy =
        world.local_elements == 0 ? nullptr : world.ownedEnergy(active_buffer);
    MPI_Gatherv(local_energy, static_cast<int>(world.local_elements), MPI_DOUBLE,
                rank == 0 ? global_energy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
        print_results(global_energy, "ElementEnergy");
    }
}

void printUsage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseInteger(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            arguments_valid = parseInteger(argv[++i], n_elems_root);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            arguments_valid = parseInteger(argv[++i], n_iters);
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
        }
        if (!arguments_valid) {
            break;
        }
    }

    if (show_help) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (n_elems_root <= 0 || n_iters < 0 ||
        (n_elems_root > 0 &&
         n_elems_root > std::numeric_limits<int>::max() / n_elems_root)) {
        arguments_valid = false;
        if (rank == 0) {
            std::fprintf(stderr,
                         "Grid size must be positive, its square must fit in "
                         "an int, and iterations must be non-negative.\n");
        }
    }
    if (!arguments_valid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %d elements\n", n_elems_root,
                    n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("MPI ranks: %d\n", n_ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("\nBuilding distributed unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    const uint64_t local_static_mem =
        world.elements_static.size() * sizeof(ElementStatic);
    const uint64_t local_dynamic_mem =
        (world.energy[0].size() + world.energy[1].size() +
         world.total_flux[0].size() + world.total_flux[1].size()) *
        sizeof(val_t);
    const uint64_t local_total_mem = local_static_mem + local_dynamic_mem;
    uint64_t aggregate_static_mem = 0;
    uint64_t aggregate_dynamic_mem = 0;
    uint64_t aggregate_total_mem = 0;
    uint64_t maximum_rank_mem = 0;
    MPI_Reduce(&local_static_mem, &aggregate_static_mem, 1, MPI_UINT64_T,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &aggregate_dynamic_mem, 1, MPI_UINT64_T,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_mem, &aggregate_total_mem, 1, MPI_UINT64_T, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_mem, &maximum_rank_mem, 1, MPI_UINT64_T, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        constexpr double bytes_per_mb = 1024.0 * 1024.0;
        std::printf("Memory usage: %.2f MB aggregate (static: %.2f MB, "
                    "dynamic: %.2f MB; max/rank: %.2f MB)\n\n",
                    aggregate_total_mem / bytes_per_mb,
                    aggregate_static_mem / bytes_per_mb,
                    aggregate_dynamic_mem / bytes_per_mb,
                    maximum_rank_mem / bytes_per_mb);
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int active_buffer = runSimulation(world, n_iters, rank);
    const double local_duration = MPI_Wtime() - start;
    double duration_seconds = 0.0;
    MPI_Reduce(&local_duration, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const uint64_t local_hash = computeLocalHash(world, active_buffer);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const double duration_ms = duration_seconds * 1000.0;
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            duration_seconds > 0.0
                ? (static_cast<double>(n_measured_iters) * n_elems) /
                      duration_seconds / 1e9
                : 0.0;
        const double gflops = giga_elems_per_sec * 22.0;

        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
    }

    if (print_results_requested) {
        printDistributedResults(world, active_buffer, rank, n_ranks,
                                n_elems_root);
    }

    int exit_code = 0;
    if (validate && !validateResults(world, active_buffer, rank)) {
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
