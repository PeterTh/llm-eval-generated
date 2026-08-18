#include <mpi.h>

#include <algorithm>
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

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // Both dynamic arrays contain one halo row at each end. Owned entries start
    // at halo_width and are contiguous in global row-major order.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int grid_size = 0;
    int first_row = 0;
    int local_rows = 0;
    size_t halo_width = 0;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

static void decomposeRows(int n, int rank, int ranks, int& first, int& count) {
    const int base = n / ranks;
    const int extra = n % ranks;
    count = base + (rank < extra ? 1 : 0);
    first = rank * base + std::min(rank, extra);
}

// Build only this rank's portion of the mesh. Connectivity indices address the
// local dynamic array, including its two halo rows.
void buildSquare2D(World& world, int n_elems_root, int rank, int ranks) {
    world.grid_size = n_elems_root;
    decomposeRows(n_elems_root, rank, ranks, world.first_row, world.local_rows);
    world.halo_width = static_cast<size_t>(n_elems_root);

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};

    const size_t local_elems = static_cast<size_t>(world.local_rows) * n_elems_root;
    world.elements_static.resize(local_elems);
    const size_t dynamic_elems = world.local_rows == 0
        ? 0 : local_elems + 2 * world.halo_width;
    world.elements_dynamic.resize(dynamic_elems, {0.0, 0.0});
    world.elements_dynamic_swap.resize(dynamic_elems, {0.0, 0.0});

    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int global_x = world.first_row + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t owned_idx = static_cast<size_t>(local_x) * n_elems_root + y;
            const size_t dyn_idx = world.halo_width + owned_idx;
            ElementStatic& elem = world.elements_static[owned_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Preserve the reference program's neighbor and floating-point
            // accumulation order: +x, -x, +y, -y.
            if (global_x + 1 < n_elems_root) {
                elem.connected_idx[elem.num_connections] = dyn_idx + n_elems_root;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (global_x > 0) {
                elem.connected_idx[elem.num_connections] = dyn_idx - n_elems_root;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (y + 1 < n_elems_root) {
                elem.connected_idx[elem.num_connections] = dyn_idx + 1;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (y > 0) {
                elem.connected_idx[elem.num_connections] = dyn_idx - 1;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }

            const bool left_corner = y == 0;
            const bool right_corner = y == n_elems_root - 1;
            if ((global_x == 0 || global_x == n_elems_root - 1) &&
                (left_corner || right_corner)) {
                elem.material_idx = (global_x == 0) == left_corner
                                        ? INFLOW_MAT_ID
                                        : OUTFLOW_MAT_ID;
            }
        }
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

static inline void updateRange(World& world, size_t begin, size_t end) {
    const size_t halo = world.halo_width;
    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& stat = world.elements_static[i];
        const ElementDynamic& current = world.elements_dynamic[halo + i];
        const Material& mat = world.materials[stat.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < stat.num_connections; ++j) {
            total_flux += computeFlux(mat, current, stat.connected_flux[j],
                                      world.elements_dynamic[stat.connected_idx[j]]);
        }
        ElementDynamic& next = world.elements_dynamic_swap[halo + i];
        next.current_energy = current.current_energy + total_flux;
        next.total_flux = current.total_flux + std::abs(total_flux);
    }
}

void runSimulation(World& world, int n_iters, int rank, int ranks, MPI_Comm comm) {
    const int n = world.grid_size;
    const size_t local_elems = world.elements_static.size();
    if (world.local_rows == 0) {
        // Ranks beyond the number of rows deliberately remain in the collective
        // execution but have no point-to-point neighbors or computation.
        return;
    }

    const int active_ranks = std::min(ranks, n);
    const int previous = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    const size_t first_owned = world.halo_width;
    const size_t last_owned = world.halo_width + local_elems - static_cast<size_t>(n);

    // Transfer only current_energy from the array-of-structs representation.
    // The vector type avoids packing an extra row-sized application buffer.
    MPI_Datatype energy_row;
    MPI_Type_vector(n, 1, static_cast<int>(sizeof(ElementDynamic) / sizeof(val_t)),
                    MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;

        if (previous != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[0].current_energy, 1, energy_row,
                      previous, 1, comm, &requests[request_count++]);
            MPI_Isend(&world.elements_dynamic[first_owned].current_energy, 1,
                      energy_row, previous, 2, comm, &requests[request_count++]);
        }
        if (next != MPI_PROC_NULL) {
            MPI_Irecv(&world.elements_dynamic[world.halo_width + local_elems].current_energy,
                      1, energy_row, next, 2, comm, &requests[request_count++]);
            MPI_Isend(&world.elements_dynamic[last_owned].current_energy, 1,
                      energy_row, next, 1, comm, &requests[request_count++]);
        }

        // Halo exchange is overlapped with rows having no off-rank dependency.
        if (world.local_rows > 2) {
            updateRange(world, static_cast<size_t>(n),
                        local_elems - static_cast<size_t>(n));
        }
        if (request_count != 0) {
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        }

        updateRange(world, 0, std::min(local_elems, static_cast<size_t>(n)));
        if (world.local_rows > 1) {
            updateRange(world, local_elems - static_cast<size_t>(n), local_elems);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
}

bool validateResults(const World& world, int rank, MPI_Comm comm) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    const size_t halo = world.halo_width;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const ElementDynamic& elem = world.elements_dynamic[halo + i];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_max = std::max(local_max, elem.current_energy);
        local_min = std::min(local_min, elem.current_energy);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(energy_sum) > 1e-8) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) printf("  Validation: PASSED\n");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

uint64_t computeHash(const World& world, MPI_Comm comm) {
    uint64_t local_hash = 0;
    const size_t global_start = static_cast<size_t>(world.first_row) * world.grid_size;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const ElementDynamic& elem = world.elements_dynamic[world.halo_width + i];
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
        const uint64_t global_i = global_start + i;
        local_hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    return global_hash;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool help = false;
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
            help = true;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            arguments_valid = false;
        }
    }
    if (n_elems_root <= 0 || n_iters < 0 ||
        static_cast<uint64_t>(n_elems_root) * n_elems_root >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Grid size must be positive and iterations nonnegative.\n");
        arguments_valid = false;
    }
    if (help || !arguments_valid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }

    const int n_elems = n_elems_root * n_elems_root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", ranks);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, ranks);
    const uint64_t local_static = world.elements_static.size() * sizeof(ElementStatic);
    const uint64_t local_dynamic = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    uint64_t total_static = 0, total_dynamic = 0, max_rank_mem = 0;
    const uint64_t local_mem = local_static + local_dynamic;
    MPI_Reduce(&local_static, &total_static, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    MPI_Reduce(&local_dynamic, &total_dynamic, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    MPI_Reduce(&local_mem, &max_rank_mem, 1, MPI_UINT64_T, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Aggregate memory: %.2f MB (static: %.2f MB, dynamic incl. halos: %.2f MB)\n",
               (total_static + total_dynamic) / (1024.0 * 1024.0),
               total_static / (1024.0 * 1024.0), total_dynamic / (1024.0 * 1024.0));
        printf("Maximum memory per rank: %.2f MB\n\n", max_rank_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, rank, ranks, comm);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const uint64_t result_hash = computeHash(world, comm);
    if (rank == 0) {
        const double duration_ms = seconds * 1000.0;
        const int measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / measured_iters;
        const double giga_elems_per_sec = seconds > 0.0
            ? (static_cast<double>(measured_iters) * n_elems) / seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
    }

    if (print_results_requested) {
        const int local_count = static_cast<int>(world.elements_static.size());
        std::vector<double> local_energy(local_count);
        for (int i = 0; i < local_count; ++i)
            local_energy[i] = world.elements_dynamic[world.halo_width + i].current_energy;

        std::vector<int> counts, displacements;
        std::vector<double> all_energy;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            all_energy.resize(n_elems);
            for (int r = 0; r < ranks; ++r) {
                int first, rows;
                decomposeRows(n_elems_root, r, ranks, first, rows);
                counts[r] = rows * n_elems_root;
                displacements[r] = first * n_elems_root;
            }
        }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(all_energy, "ElementEnergy");
    }

    const bool valid = !validate || validateResults(world, rank, comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
