#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    // MPI row decomposition. The dynamic arrays contain one halo row on
    // either side of the owned rows; elements_static contains owned elements
    // only.  A static connectivity index addresses elements_dynamic directly.
    int n_elems_root = 0;
    int row_begin = 0;
    int local_rows = 0;
    int mpi_rank = 0;
    int mpi_size = 1;

    std::size_t owned_elements() const {
        return static_cast<std::size_t>(local_rows) *
               static_cast<std::size_t>(n_elems_root);
    }

    std::size_t dynamic_offset() const {
        return static_cast<std::size_t>(n_elems_root);
    }
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int row_begin,
                   const int local_rows, const int mpi_rank,
                   const int mpi_size) {
    world.n_elems_root = n_elems_root;
    world.row_begin = row_begin;
    world.local_rows = local_rows;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;

    const std::size_t n = static_cast<std::size_t>(n_elems_root);
    const std::size_t n_elems =
        static_cast<std::size_t>(local_rows) * n;
    const std::size_t dynamic_elems = n_elems + 2 * n;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(dynamic_elems);
    world.elements_dynamic_swap.resize(dynamic_elems);
    
    // Initialize all elements with default material and zero energy
    for (std::size_t i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    for (std::size_t i = n_elems; i < dynamic_elems; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    std::fill(world.elements_dynamic_swap.begin(),
              world.elements_dynamic_swap.end(), ElementDynamic{0.0, 0.0});
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    const int row_end = row_begin + local_rows;
    for (int local_x = 0; local_x < local_rows; ++local_x) {
        const int x = row_begin + local_x;
        for (int y = 0; y < n_elems_root; ++y) {
            const std::size_t idx = static_cast<std::size_t>(local_x) * n +
                                    static_cast<std::size_t>(y);
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int direction = 0; direction < 4; ++direction) {
                const int nx = x + offsets[direction][0];
                const int ny = y + offsets[direction][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    std::size_t neighbor_idx;
                    if (nx < row_begin) {
                        neighbor_idx = static_cast<std::size_t>(ny);
                    } else if (nx >= row_end) {
                        neighbor_idx = static_cast<std::size_t>(local_rows + 1) * n +
                                       static_cast<std::size_t>(ny);
                    } else {
                        neighbor_idx = static_cast<std::size_t>(nx - row_begin + 1) * n +
                                       static_cast<std::size_t>(ny);
                    }
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    if (row_begin <= 0 && 0 < row_end) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[static_cast<std::size_t>(last)].material_idx =
            OUTFLOW_MAT_ID;
    }
    if (row_begin <= last && last < row_end) {
        const std::size_t last_row = static_cast<std::size_t>(local_rows - 1) * n;
        world.elements_static[last_row].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last_row + static_cast<std::size_t>(last)].material_idx =
            INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

inline void updateRows(World& world, const int first_row, const int last_row) {
    const std::size_t n = static_cast<std::size_t>(world.n_elems_root);
    const std::size_t dynamic_offset = world.dynamic_offset();

    for (int local_row = first_row; local_row < last_row; ++local_row) {
        const std::size_t static_base = static_cast<std::size_t>(local_row) * n;
        const std::size_t dynamic_base = static_base + dynamic_offset;
        for (std::size_t y = 0; y < n; ++y) {
            const ElementStatic& elem_static = world.elements_static[static_base + y];
            const ElementDynamic& elem_dyn =
                world.elements_dynamic[dynamic_base + y];
            const Material& mat = world.materials[elem_static.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn =
                    world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn,
                                          elem_static.connected_flux[j],
                                          neighbor_dyn);
            }

            ElementDynamic& elem_write =
                world.elements_dynamic_swap[dynamic_base + y];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
    }
}

// Run simulation for n_iters iterations. Each rank owns a group of rows and
// exchanges only the two boundary rows needed by the unstructured connectivity.
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    MPI_Datatype halo_row_type;
    MPI_Type_create_hvector(
        world.n_elems_root, 1, static_cast<MPI_Aint>(sizeof(ElementDynamic)),
        MPI_DOUBLE, &halo_row_type);
    MPI_Type_commit(&halo_row_type);

    const std::size_t n = static_cast<std::size_t>(world.n_elems_root);
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;

        // The first rank receives a top halo from its predecessor; the last
        // rank receives a bottom halo from its successor.
        if (world.mpi_rank > 0) {
            MPI_Irecv(&world.elements_dynamic[0].current_energy, 1,
                      halo_row_type, world.mpi_rank - 1, 0, comm,
                      &requests[request_count++]);
        }
        if (world.mpi_rank + 1 < world.mpi_size) {
            const std::size_t bottom_halo =
                static_cast<std::size_t>(world.local_rows + 1) * n;
            MPI_Irecv(&world.elements_dynamic[bottom_halo].current_energy, 1,
                      halo_row_type, world.mpi_rank + 1, 1, comm,
                      &requests[request_count++]);
        }

        if (world.mpi_rank > 0) {
            const std::size_t first_owned = n;
            MPI_Isend(&world.elements_dynamic[first_owned].current_energy, 1,
                      halo_row_type, world.mpi_rank - 1, 1, comm,
                      &requests[request_count++]);
        }
        if (world.mpi_rank + 1 < world.mpi_size) {
            const std::size_t last_owned =
                static_cast<std::size_t>(world.local_rows) * n;
            MPI_Isend(&world.elements_dynamic[last_owned].current_energy, 1,
                      halo_row_type, world.mpi_rank + 1, 0, comm,
                      &requests[request_count++]);
        }

        // Interior rows have no remote dependencies and can be computed while
        // the boundary exchange progresses.
        if (world.local_rows > 2) {
            updateRows(world, 1, world.local_rows - 1);
        }

        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        updateRows(world, 0, std::min(world.local_rows, 1));
        if (world.local_rows > 1) {
            updateRows(world, world.local_rows - 1, world.local_rows);
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }

    MPI_Type_free(&halo_row_type);
}

// Validate simulation results. Reductions avoid gathering the full state for
// the normal validation path; -r gathers only when globally ordered samples
// are explicitly requested.
bool validateResults(const World& world, MPI_Comm comm) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    const std::size_t dynamic_offset = world.dynamic_offset();
    for (std::size_t i = 0; i < world.owned_elements(); ++i) {
        const ElementDynamic& elem = world.elements_dynamic[dynamic_offset + i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = std::numeric_limits<val_t>::lowest();
    val_t global_energy_min = std::numeric_limits<val_t>::max();
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    int local_finite = 1;
    for (std::size_t i = 0; i < world.owned_elements(); ++i) {
        const ElementDynamic& elem = world.elements_dynamic[dynamic_offset + i];
        if (!std::isfinite(elem.current_energy) ||
            !std::isfinite(elem.total_flux)) {
            local_finite = 0;
            break;
        }
    }
    int all_finite = 0;
    MPI_Reduce(&local_finite, &all_finite, 1, MPI_INT, MPI_MIN, 0, comm);

    int valid = 1;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min,
               global_energy_max);

        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;

        if (!all_finite || !std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }

        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }

        if (!std::isfinite(global_energy_max) ||
            !std::isfinite(global_energy_min)) {
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

// Compute a simple hash of the results for verification
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    const std::size_t dynamic_offset = world.dynamic_offset();
    const std::size_t global_offset =
        static_cast<std::size_t>(world.row_begin) *
        static_cast<std::size_t>(world.n_elems_root);
    for (std::size_t i = 0; i < world.owned_elements(); ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits,
                    &world.elements_dynamic[dynamic_offset + i].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits,
                    &world.elements_dynamic[dynamic_offset + i].total_flux,
                    sizeof(flux_bits));
        const uint64_t global_i = static_cast<uint64_t>(global_offset + i);
        hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printGlobalResults(const World& world, MPI_Comm comm) {
    int comm_size = 1;
    int comm_rank = 0;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    const std::size_t n = static_cast<std::size_t>(world.n_elems_root);
    std::vector<int> counts(static_cast<std::size_t>(comm_size));
    std::vector<int> displacements(static_cast<std::size_t>(comm_size));
    const int base_rows = world.n_elems_root / comm_size;
    const int extra_rows = world.n_elems_root % comm_size;
    for (int rank = 0; rank < comm_size; ++rank) {
        const int rows = base_rows + (rank < extra_rows ? 1 : 0);
        const int first_row = rank * base_rows + std::min(rank, extra_rows);
        counts[static_cast<std::size_t>(rank)] = rows * world.n_elems_root;
        displacements[static_cast<std::size_t>(rank)] = first_row * world.n_elems_root;
    }

    std::vector<double> energy_data;
    if (comm_rank == 0) {
        energy_data.resize(n * n);
    }

    MPI_Datatype owned_type;
    MPI_Type_create_hvector(static_cast<int>(world.owned_elements()), 1,
                            static_cast<MPI_Aint>(sizeof(ElementDynamic)),
                            MPI_DOUBLE, &owned_type);
    MPI_Type_commit(&owned_type);

    const std::size_t first_owned = world.dynamic_offset();
    MPI_Gatherv(&world.elements_dynamic[first_owned].current_energy, 1,
                owned_type,
                comm_rank == 0 ? energy_data.data() : nullptr,
                counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);

    MPI_Type_free(&owned_type);
    if (comm_rank == 0) {
        print_results(energy_data, "ElementEnergy");
    }
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

    int mpi_world_rank = 0;
    int mpi_world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    bool show_help = false;
    bool parse_error = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (mpi_world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_error = true;
        }
    }

    if (show_help) {
        if (mpi_world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parse_error || n_elems_root <= 0 || n_iters < 0) {
        if (mpi_world_rank == 0 && !parse_error) {
            printf("Grid size and iterations must be non-negative, with a positive grid size.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const std::size_t n_elems = static_cast<std::size_t>(n_elems_root) *
                                static_cast<std::size_t>(n_elems_root);

    // There can be more MPI processes than mesh rows. Keep all ranks in the
    // MPI job, but use only the ranks that can own at least one row.
    const int active_size = std::min(mpi_world_size, n_elems_root);
    MPI_Comm active_comm = MPI_COMM_NULL;
    const int color = mpi_world_rank < active_size ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_world_rank, &active_comm);
    if (color == MPI_UNDEFINED) {
        MPI_Finalize();
        return 0;
    }

    int mpi_rank = 0;
    MPI_Comm_rank(active_comm, &mpi_rank);

    const int base_rows = n_elems_root / active_size;
    const int extra_rows = n_elems_root % active_size;
    const int local_rows = base_rows + (mpi_rank < extra_rows ? 1 : 0);
    const int row_begin = mpi_rank * base_rows + std::min(mpi_rank, extra_rows);

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root,
               n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", active_size);
        printf("\n");
    }

    // Build the unstructured mesh
    if (mpi_rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, row_begin, local_rows, mpi_rank,
                  active_size);

    // Calculate memory usage
    const std::size_t local_static_mem =
        world.elements_static.size() * sizeof(ElementStatic);
    const std::size_t local_dynamic_mem =
        world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const unsigned long long local_total_mem =
        static_cast<unsigned long long>(local_static_mem + local_dynamic_mem);
    unsigned long long max_local_mem = 0;
    MPI_Reduce(&local_total_mem, &max_local_mem, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_MAX, 0, active_comm);
    if (mpi_rank == 0) {
        const std::size_t static_mem = n_elems * sizeof(ElementStatic);
        const std::size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const std::size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (global static: %.2f MB, global dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("Peak local memory: %.2f MB\n",
               max_local_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (mpi_rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(active_comm);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters, active_comm);

    MPI_Barrier(active_comm);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               active_comm);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    if (mpi_rank == 0) {
        const double duration_ms = elapsed * 1000.0;
        const double safe_elapsed = std::max(elapsed, std::numeric_limits<double>::min());
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) /
            safe_elapsed / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification
    const uint64_t local_hash = computeLocalHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active_comm);
    if (mpi_rank == 0) {
        printf("  Result hash: %016" PRIX64 "\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        printGlobalResults(world, active_comm);
    }

    // Validation
    int exit_code = 0;
    if (validate) {
        const bool valid = validateResults(world, active_comm);
        if (!valid) {
            exit_code = 1;
        }
    }

    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return exit_code;
}
