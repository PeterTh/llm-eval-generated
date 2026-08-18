#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements.
using idx_t = uint64_t;
using val_t = double;

// Material properties for energy transfer.
struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Material type IDs.
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Every rank owns one rectangular part of the global mesh.  The state is kept
// in structure-of-arrays form so that the north/south halo rows are contiguous
// and the update loop has unit-stride accesses.
struct World {
    std::vector<Material> materials;
    int n_elems_root = 0;
    int row_begin = 0;
    int col_begin = 0;
    int local_rows = 0;
    int local_cols = 0;

    std::vector<val_t> energy;
    std::vector<val_t> total_flux;
    std::vector<val_t> next_energy;
    std::vector<val_t> next_total_flux;

    std::vector<val_t> top_halo;
    std::vector<val_t> bottom_halo;
    std::vector<val_t> left_halo;
    std::vector<val_t> right_halo;
    std::vector<val_t> left_send;
    std::vector<val_t> right_send;

    MPI_Comm cart_comm = MPI_COMM_NULL;
    int north = MPI_PROC_NULL;
    int south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL;
    int east = MPI_PROC_NULL;
};

constexpr int TAG_TOP = 100;
constexpr int TAG_BOTTOM = 101;
constexpr int TAG_LEFT = 102;
constexpr int TAG_RIGHT = 103;

int blockStart(const int total, const int blocks, const int block) {
    const int base = total / blocks;
    const int remainder = total % blocks;
    return block * base + std::min(block, remainder);
}

int blockSize(const int total, const int blocks, const int block) {
    const int base = total / blocks;
    return base + (block < total % blocks ? 1 : 0);
}

// Select the largest rectangular process grid whose dimensions do not exceed
// the mesh dimensions.  This keeps every active rank useful, including for
// small problem sizes or oversubscribed launches.
int activeProcessCount(const int process_count, const int n_elems_root) {
    int active = 1;
    const int max_rows = std::min(process_count, n_elems_root);
    for (int row_ranks = 1; row_ranks <= max_rows; ++row_ranks) {
        const int col_ranks = std::min(n_elems_root, process_count / row_ranks);
        active = std::max(active, row_ranks * col_ranks);
    }
    return active;
}

// Build the distributed 2-D square grid.  Connectivity is implicit: this is
// exactly the same up/down/right/left adjacency built by the original mesh.
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm cart_comm) {
    world.n_elems_root = n_elems_root;
    world.cart_comm = cart_comm;
    world.materials = {
        {0.8, 0.0},   // Default material
        {0.8, 0.5},   // Inflow material
        {0.8, -0.5},  // Outflow material
    };

    int coords[2] = {0, 0};
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);

    world.row_begin = blockStart(n_elems_root, dims[0], coords[0]);
    world.col_begin = blockStart(n_elems_root, dims[1], coords[1]);
    world.local_rows = blockSize(n_elems_root, dims[0], coords[0]);
    world.local_cols = blockSize(n_elems_root, dims[1], coords[1]);

    const size_t local_elements =
        static_cast<size_t>(world.local_rows) * static_cast<size_t>(world.local_cols);
    world.energy.assign(local_elements, 0.0);
    world.total_flux.assign(local_elements, 0.0);
    world.next_energy.resize(local_elements);
    world.next_total_flux.resize(local_elements);

    world.top_halo.resize(world.local_cols);
    world.bottom_halo.resize(world.local_cols);
    world.left_halo.resize(world.local_rows);
    world.right_halo.resize(world.local_rows);
    world.left_send.resize(world.local_rows);
    world.right_send.resize(world.local_rows);

    MPI_Cart_shift(cart_comm, 0, 1, &world.north, &world.south);
    MPI_Cart_shift(cart_comm, 1, 1, &world.west, &world.east);
}

inline const Material& materialAt(const World& world, const int global_row,
                                  const int global_col) {
    const int last = world.n_elems_root - 1;
    if ((global_row == 0 && global_col == 0) ||
        (global_row == last && global_col == last)) {
        return world.materials[INFLOW_MAT_ID];
    }
    if ((global_row == 0 && global_col == last) ||
        (global_row == last && global_col == 0)) {
        return world.materials[OUTFLOW_MAT_ID];
    }
    return world.materials[DEFAULT_MAT_ID];
}

inline val_t computeFlux(const Material& material, const val_t this_energy,
                         const val_t other_energy) {
    // Keep the arithmetic order used by the original computeFlux function.
    return (other_energy - this_energy) * material.transfer_coeff * 1.0 * 0.25;
}

inline void updateElement(World& world, const int local_row,
                          const int local_col) {
    const int cols = world.local_cols;
    const size_t index = static_cast<size_t>(local_row) * cols + local_col;
    const int global_row = world.row_begin + local_row;
    const int global_col = world.col_begin + local_col;
    const val_t current_energy = world.energy[index];
    const Material& material = materialAt(world, global_row, global_col);

    val_t flux = material.external_flow;

    // The order is down, up, right, left, matching buildSquare2D in the
    // original implementation.
    if (global_row + 1 < world.n_elems_root) {
        const val_t neighbor = local_row + 1 < world.local_rows
                                   ? world.energy[index + cols]
                                   : world.bottom_halo[local_col];
        flux += computeFlux(material, current_energy, neighbor);
    }
    if (global_row > 0) {
        const val_t neighbor = local_row > 0 ? world.energy[index - cols]
                                              : world.top_halo[local_col];
        flux += computeFlux(material, current_energy, neighbor);
    }
    if (global_col + 1 < world.n_elems_root) {
        const val_t neighbor = local_col + 1 < world.local_cols
                                   ? world.energy[index + 1]
                                   : world.right_halo[local_row];
        flux += computeFlux(material, current_energy, neighbor);
    }
    if (global_col > 0) {
        const val_t neighbor = local_col > 0 ? world.energy[index - 1]
                                              : world.left_halo[local_row];
        flux += computeFlux(material, current_energy, neighbor);
    }

    world.next_energy[index] = current_energy + flux;
    world.next_total_flux[index] = world.total_flux[index] + std::abs(flux);
}

// Run the communication-independent interior while halos are in flight.  An
// interior element is never a global boundary/corner and always has four local
// neighbours, allowing a leaner hot loop.
void updateInterior(World& world) {
    if (world.local_rows < 3 || world.local_cols < 3) {
        return;
    }

    const int cols = world.local_cols;
    const Material& material = world.materials[DEFAULT_MAT_ID];
    for (int local_row = 1; local_row + 1 < world.local_rows; ++local_row) {
        const size_t row_begin = static_cast<size_t>(local_row) * cols;
        for (int local_col = 1; local_col + 1 < world.local_cols; ++local_col) {
            const size_t index = row_begin + local_col;
            const val_t current_energy = world.energy[index];
            val_t flux = material.external_flow;
            flux += computeFlux(material, current_energy, world.energy[index + cols]);
            flux += computeFlux(material, current_energy, world.energy[index - cols]);
            flux += computeFlux(material, current_energy, world.energy[index + 1]);
            flux += computeFlux(material, current_energy, world.energy[index - 1]);
            world.next_energy[index] = current_energy + flux;
            world.next_total_flux[index] = world.total_flux[index] + std::abs(flux);
        }
    }
}

void exchangeHalos(World& world, MPI_Request* requests, int& request_count) {
    request_count = 0;
    const int rows = world.local_rows;
    const int cols = world.local_cols;

    if (world.north != MPI_PROC_NULL) {
        MPI_Irecv(world.top_halo.data(), cols, MPI_DOUBLE, world.north, TAG_TOP,
                  world.cart_comm, &requests[request_count++]);
    }
    if (world.south != MPI_PROC_NULL) {
        MPI_Irecv(world.bottom_halo.data(), cols, MPI_DOUBLE, world.south, TAG_BOTTOM,
                  world.cart_comm, &requests[request_count++]);
    }
    if (world.west != MPI_PROC_NULL) {
        MPI_Irecv(world.left_halo.data(), rows, MPI_DOUBLE, world.west, TAG_LEFT,
                  world.cart_comm, &requests[request_count++]);
    }
    if (world.east != MPI_PROC_NULL) {
        MPI_Irecv(world.right_halo.data(), rows, MPI_DOUBLE, world.east, TAG_RIGHT,
                  world.cart_comm, &requests[request_count++]);
    }

    for (int row = 0; row < rows; ++row) {
        world.left_send[row] = world.energy[static_cast<size_t>(row) * cols];
        world.right_send[row] = world.energy[static_cast<size_t>(row) * cols + cols - 1];
    }

    if (world.north != MPI_PROC_NULL) {
        MPI_Isend(world.energy.data(), cols, MPI_DOUBLE, world.north, TAG_BOTTOM,
                  world.cart_comm, &requests[request_count++]);
    }
    if (world.south != MPI_PROC_NULL) {
        MPI_Isend(world.energy.data() + static_cast<size_t>(rows - 1) * cols, cols,
                  MPI_DOUBLE, world.south, TAG_TOP, world.cart_comm,
                  &requests[request_count++]);
    }
    if (world.west != MPI_PROC_NULL) {
        MPI_Isend(world.left_send.data(), rows, MPI_DOUBLE, world.west, TAG_RIGHT,
                  world.cart_comm, &requests[request_count++]);
    }
    if (world.east != MPI_PROC_NULL) {
        MPI_Isend(world.right_send.data(), rows, MPI_DOUBLE, world.east, TAG_LEFT,
                  world.cart_comm, &requests[request_count++]);
    }
}

// Run the simulation for n_iters iterations.
void runSimulation(World& world, const int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int request_count = 0;
        exchangeHalos(world, requests, request_count);

        updateInterior(world);

        if (request_count != 0) {
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        }

        // Finish only the perimeter.  The interior was updated while the
        // boundary values were being exchanged.
        for (int local_row = 0; local_row < world.local_rows; ++local_row) {
            for (int local_col = 0; local_col < world.local_cols; ++local_col) {
                const bool is_interior = local_row > 0 && local_row + 1 < world.local_rows &&
                                         local_col > 0 && local_col + 1 < world.local_cols;
                if (!is_interior) {
                    updateElement(world, local_row, local_col);
                }
            }
        }

        world.energy.swap(world.next_energy);
        world.total_flux.swap(world.next_total_flux);
    }
}

// Collect a distributed field in global row-major order.  This is used only
// for optional validation and result printing; the normal benchmark path does
// not replicate the mesh on rank zero.
void gatherGlobalState(const World& world, const int rank, const int process_count,
                       const bool active, const int n_elems_root,
                       std::vector<val_t>& global_energy,
                       std::vector<val_t>& global_total_flux) {
    const int local_count = active ? static_cast<int>(world.energy.size()) : 0;
    const int local_metadata[4] = {
        active ? world.row_begin : 0,
        active ? world.col_begin : 0,
        active ? world.local_rows : 0,
        active ? world.local_cols : 0,
    };

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<int> metadata;
    if (rank == 0) {
        counts.resize(process_count);
        displacements.resize(process_count);
        metadata.resize(static_cast<size_t>(process_count) * 4);
    }

    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_metadata, 4, MPI_INT, rank == 0 ? metadata.data() : nullptr, 4, MPI_INT,
               0, MPI_COMM_WORLD);

    int gathered_count = 0;
    if (rank == 0) {
        for (int process = 0; process < process_count; ++process) {
            displacements[process] = gathered_count;
            gathered_count += counts[process];
        }
    }

    std::vector<val_t> gathered_energy;
    std::vector<val_t> gathered_flux;
    if (rank == 0) {
        gathered_energy.resize(gathered_count);
        gathered_flux.resize(gathered_count);
    }

    MPI_Gatherv(active ? world.energy.data() : nullptr, local_count, MPI_DOUBLE,
                rank == 0 ? gathered_energy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(active ? world.total_flux.data() : nullptr, local_count, MPI_DOUBLE,
                rank == 0 ? gathered_flux.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return;
    }

    const size_t global_count = static_cast<size_t>(n_elems_root) * n_elems_root;
    global_energy.assign(global_count, 0.0);
    global_total_flux.assign(global_count, 0.0);
    for (int process = 0; process < process_count; ++process) {
        const int row_begin = metadata[4 * process];
        const int col_begin = metadata[4 * process + 1];
        const int rows = metadata[4 * process + 2];
        const int cols = metadata[4 * process + 3];
        const int source_begin = displacements[process];
        for (int row = 0; row < rows; ++row) {
            const size_t destination =
                static_cast<size_t>(row_begin + row) * n_elems_root + col_begin;
            const size_t source = static_cast<size_t>(source_begin) +
                                  static_cast<size_t>(row) * cols;
            std::memcpy(global_energy.data() + destination, gathered_energy.data() + source,
                        static_cast<size_t>(cols) * sizeof(val_t));
            std::memcpy(global_total_flux.data() + destination, gathered_flux.data() + source,
                        static_cast<size_t>(cols) * sizeof(val_t));
        }
    }
}

// Validate simulation results.  The vectors are global row-major fields on
// rank zero, so this retains the original validation arithmetic and reporting.
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

// Compute the original order-independent verification hash over the global
// element indices, then combine the local contributions with MPI_BXOR.
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    for (int local_row = 0; local_row < world.local_rows; ++local_row) {
        const size_t local_row_begin = static_cast<size_t>(local_row) * world.local_cols;
        const uint64_t global_row_begin =
            static_cast<uint64_t>(world.row_begin + local_row) * world.n_elems_root +
            world.col_begin;
        for (int local_col = 0; local_col < world.local_cols; ++local_col) {
            const size_t local_index = local_row_begin + local_col;
            uint64_t energy_bits;
            uint64_t flux_bits;
            std::memcpy(&energy_bits, &world.energy[local_index], sizeof(energy_bits));
            std::memcpy(&flux_bits, &world.total_flux[local_index], sizeof(flux_bits));
            const uint64_t global_index = global_row_begin + local_col;
            hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

size_t localStateBytes(const World& world) {
    return (world.energy.size() + world.total_flux.size() + world.next_energy.size() +
            world.next_total_flux.size() + world.top_halo.size() + world.bottom_halo.size() +
            world.left_halo.size() + world.right_halo.size() + world.left_send.size() +
            world.right_send.size()) *
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

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_enabled = false;
    bool show_usage = false;
    bool bad_arguments = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            print_results_enabled = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            show_usage = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            bad_arguments = true;
        }
    }

    const int max_root = 46340;  // sqrt(INT_MAX), required by MPI count arguments.
    if (n_elems_root <= 0 || n_elems_root > max_root || n_iters < 0) {
        if (rank == 0) {
            printf("Grid size must be in [1, %d] and iterations must be non-negative.\n",
                   max_root);
        }
        bad_arguments = true;
    }
    if (show_usage || bad_arguments) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return bad_arguments ? 1 : 0;
    }

    const int active_processes = activeProcessCount(process_count, n_elems_root);
    const bool active = rank < active_processes;
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &active_comm);

    World world;
    if (active) {
        int dims[2] = {0, 0};
        MPI_Dims_create(active_processes, 2, dims);
        int periods[2] = {0, 0};
        MPI_Comm cart_comm = MPI_COMM_NULL;
        MPI_Cart_create(active_comm, 2, dims, periods, 0, &cart_comm);
        MPI_Comm_free(&active_comm);
        buildSquare2D(world, n_elems_root, cart_comm);
    }

    const size_t global_elements =
        static_cast<size_t>(n_elems_root) * static_cast<size_t>(n_elems_root);
    const unsigned long long local_memory =
        static_cast<unsigned long long>(active ? localStateBytes(world) : 0);
    unsigned long long total_memory = 0;
    unsigned long long max_rank_memory = 0;
    MPI_Reduce(&local_memory, &total_memory, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_memory, &max_rank_memory, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root,
               global_elements);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d (%d active)\n", process_count, active_processes);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building distributed unstructured mesh...\n");
        printf("Memory usage: %.2f MB total (max rank: %.2f MB)\n\n",
               total_memory / (1024.0 * 1024.0), max_rank_memory / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        runSimulation(world, n_iters);
    }
    const double local_duration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const uint64_t local_hash = active ? computeLocalHash(world) : 0;
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    const int n_measured_iters = std::max(n_iters - 1, 1);
    if (rank == 0) {
        const double duration_ms = duration * 1000.0;
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            duration > 0.0 ? (n_measured_iters * static_cast<double>(global_elements)) /
                                 duration / 1e9
                           : std::numeric_limits<double>::infinity();
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016llX\n\n",
               static_cast<unsigned long long>(global_hash));
    }

    bool valid = true;
    if (validate || print_results_enabled) {
        std::vector<val_t> global_energy;
        std::vector<val_t> global_total_flux;
        gatherGlobalState(world, rank, process_count, active, n_elems_root, global_energy,
                          global_total_flux);
        if (rank == 0) {
            if (print_results_enabled) {
                print_results(global_energy, "ElementEnergy");
            }
            if (validate) {
                valid = validateResults(global_energy, global_total_flux);
            }
        }
    }

    int global_valid = valid ? 1 : 0;
    MPI_Bcast(&global_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (active) {
        MPI_Comm_free(&world.cart_comm);
    }
    MPI_Finalize();
    return global_valid ? 0 : 1;
}
