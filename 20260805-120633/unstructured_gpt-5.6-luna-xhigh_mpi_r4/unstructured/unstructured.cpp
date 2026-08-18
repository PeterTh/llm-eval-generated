#include <algorithm>
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

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t),
              "ElementDynamic must contain two contiguous val_t fields");

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // Owned elements are stored first.  The following two rows are top and
    // bottom halo rows, respectively, and contain only current_energy.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    size_t n_elems_root = 0;
    size_t row_begin = 0;
    size_t row_end = 0;
    size_t local_rows = 0;
    size_t local_elements = 0;
    int rank = 0;
    int n_ranks = 1;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build the local row block of a 2D square grid as an unstructured mesh.
// This represents computation on arbitrarily-shaped geometries while keeping
// only local connectivity and two rows of dynamic halo state per rank.
void buildSquare2D(World& world, const size_t n_elems_root,
                   const int rank, const int n_ranks) {
    world.n_elems_root = n_elems_root;
    world.rank = rank;
    world.n_ranks = n_ranks;

    const size_t base_rows = n_elems_root / static_cast<size_t>(n_ranks);
    const size_t extra_rows = n_elems_root % static_cast<size_t>(n_ranks);
    world.row_begin = static_cast<size_t>(rank) * base_rows +
                      std::min(static_cast<size_t>(rank), extra_rows);
    world.local_rows = base_rows +
                       (static_cast<size_t>(rank) < extra_rows ? 1 : 0);
    world.row_end = world.row_begin + world.local_rows;
    world.local_elements = world.local_rows * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate owned elements followed by one top and one bottom halo row.
    world.elements_static.resize(world.local_elements);
    const size_t dynamic_elements = world.local_elements + 2 * n_elems_root;
    world.elements_dynamic.resize(dynamic_elements);
    world.elements_dynamic_swap.resize(dynamic_elements);
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < world.local_elements; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in the grid.
    // Connected indices are local dynamic-state indices; halo indices refer to
    // the top and bottom rows appended after the owned elements.
    for (size_t x = world.row_begin; x < world.row_end; ++x) {
        for (size_t y = 0; y < n_elems_root; ++y) {
            const size_t local_idx = (x - world.row_begin) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int64_t nx = static_cast<int64_t>(x) + offsets[n][0];
                const int64_t ny = static_cast<int64_t>(y) + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < static_cast<int64_t>(n_elems_root) &&
                    ny >= 0 && ny < static_cast<int64_t>(n_elems_root)) {
                    size_t neighbor_idx;
                    if (static_cast<size_t>(nx) < world.row_begin) {
                        neighbor_idx = world.local_elements + static_cast<size_t>(ny);
                    } else if (static_cast<size_t>(nx) >= world.row_end) {
                        neighbor_idx = world.local_elements + n_elems_root +
                                       static_cast<size_t>(ny);
                    } else {
                        neighbor_idx = (static_cast<size_t>(nx) - world.row_begin) *
                                       n_elems_root + static_cast<size_t>(ny);
                    }
                    elem.connected_idx[elem.num_connections] =
                        static_cast<idx_t>(neighbor_idx);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const size_t last = n_elems_root - 1;
    if (world.row_begin == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    if (world.local_rows != 0 && world.row_end == n_elems_root) {
        const size_t last_row = (world.local_rows - 1) * n_elems_root;
        world.elements_static[last_row].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last_row + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

inline void updateRange(World& world, const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];

        // Start with external flow.
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements.  Connectivity is local and
        // includes halo indices, so this loop has no global-index lookup.
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn =
                world.elements_dynamic[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn,
                                      elem_static.connected_flux[j], neighbor_dyn);
        }

        ElementDynamic& elem_write = world.elements_dynamic_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations using row-block domain decomposition.
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const size_t n_elems = world.local_elements;
    const size_t row_length = world.n_elems_root;
    const int mpi_row_length = static_cast<int>(row_length);
    MPI_Datatype energy_row_type;
    MPI_Type_vector(mpi_row_length, 1, 2, MPI_DOUBLE, &energy_row_type);
    MPI_Type_commit(&energy_row_type);

    for (int iter = 0; iter < n_iters; ++iter) {
        constexpr size_t first_boundary_row = 0;
        const size_t first_interior = world.local_rows > 1 ? row_length : n_elems;
        const size_t last_interior = world.local_rows > 1
                                         ? n_elems - row_length
                                         : n_elems;

        // Exchange current boundary values, then update rows independent of
        // the exchange while the messages are in flight.
        if (world.local_rows != 0) {
            constexpr int TAG_TOP = 0;
            constexpr int TAG_BOTTOM = 1;
            ElementDynamic* current = world.elements_dynamic.data();
            const size_t halo_begin = world.local_elements;
            MPI_Request requests[4];
            int request_count = 0;

            if (world.row_begin > 0) {
                MPI_Irecv(&current[halo_begin].current_energy, 1,
                          energy_row_type, world.rank - 1, TAG_BOTTOM, comm,
                          &requests[request_count++]);
            }
            if (world.row_end < world.n_elems_root) {
                MPI_Irecv(&current[halo_begin + row_length].current_energy, 1,
                          energy_row_type, world.rank + 1, TAG_TOP, comm,
                          &requests[request_count++]);
            }
            if (world.row_begin > 0) {
                MPI_Isend(&current[0].current_energy, 1, energy_row_type,
                          world.rank - 1, TAG_TOP, comm,
                          &requests[request_count++]);
            }
            if (world.row_end < world.n_elems_root) {
                MPI_Isend(&current[(world.local_rows - 1) * row_length]
                               .current_energy,
                          1, energy_row_type, world.rank + 1, TAG_BOTTOM, comm,
                          &requests[request_count++]);
            }

            updateRange(world, first_interior, last_interior);

            if (request_count != 0) {
                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            }
            updateRange(world, first_boundary_row, first_interior);
            if (world.local_rows > 1) {
                updateRange(world, last_interior, n_elems);
            }
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }

    MPI_Type_free(&energy_row_type);
}

// Validate simulation results with reductions over the distributed ownership
// ranges.  Only rank zero emits the benchmark's user-facing validation text.
bool validateResults(const World& world, MPI_Comm comm) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < world.local_elements; ++i) {
        const ElementDynamic& elem = world.elements_dynamic[i];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               comm);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
               comm);

    int valid = 1;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        // Check for numerical issues.
        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows.
        }
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid != 0) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

// Compute the same per-element hash as the original program.  XOR makes the
// per-rank partial hashes exactly reducible without collecting the full state.
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t global_begin = world.row_begin * world.n_elems_root;
    for (size_t i = 0; i < world.local_elements; ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &world.elements_dynamic[i].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.elements_dynamic[i].total_flux,
                    sizeof(flux_bits));
        const uint64_t global_idx = static_cast<uint64_t>(global_begin + i);
        hash ^= (energy_bits + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) {
            printf("Grid size must be positive and iterations cannot be negative.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t grid_size = static_cast<size_t>(n_elems_root);
    const size_t n_elems = grid_size * grid_size;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root,
               n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", n_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, grid_size, rank, n_ranks);

    // Report aggregate distributed storage, including the two halo rows on
    // every active rank.
    const unsigned long long local_static_mem =
        static_cast<unsigned long long>(world.elements_static.size() *
                                        sizeof(ElementStatic));
    const unsigned long long local_dynamic_mem =
        static_cast<unsigned long long>(world.elements_dynamic.size() *
                                        sizeof(ElementDynamic) * 2);
    unsigned long long static_mem = 0;
    unsigned long long dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double total_mem = static_cast<double>(static_mem + dynamic_mem);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, MPI_COMM_WORLD);
    const double local_seconds = MPI_Wtime() - start;
    double duration_seconds = 0.0;
    MPI_Reduce(&local_seconds, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    if (rank == 0) {
        const double measured_seconds = std::max(duration_seconds, 1.0e-12);
        const double time_per_iter = measured_seconds * 1000.0 /
                                     n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) /
            measured_seconds / 1e9;
        // Approximate FLOPS: ~22 FLOPS per element per iteration.
        const double gflops = giga_elems_per_sec * 22.0;
        const long duration_ms = std::max(
            1L, static_cast<long>(std::llround(measured_seconds * 1000.0)));
        printf("Computation time: %ld ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute the verification hash without gathering the distributed state.
    const uint64_t local_hash = computeHash(world);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016" PRIX64 "\n", global_hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        const int local_count = static_cast<int>(world.local_elements);
        std::vector<double> local_energy(world.local_elements);
        for (size_t i = 0; i < world.local_elements; ++i) {
            local_energy[i] = world.elements_dynamic[i].current_energy;
        }
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> energy_data;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(n_ranks));
            displacements.resize(static_cast<size_t>(n_ranks));
            int displacement = 0;
            const size_t base_rows = grid_size / static_cast<size_t>(n_ranks);
            const size_t extra_rows = grid_size % static_cast<size_t>(n_ranks);
            for (int r = 0; r < n_ranks; ++r) {
                const size_t rows = base_rows +
                    (static_cast<size_t>(r) < extra_rows ? 1 : 0);
                counts[static_cast<size_t>(r)] =
                    static_cast<int>(rows * grid_size);
                displacements[static_cast<size_t>(r)] = displacement;
                displacement += counts[static_cast<size_t>(r)];
            }
            energy_data.resize(n_elems);
        }

        MPI_Gatherv(local_energy.data(), local_count,
                    MPI_DOUBLE, rank == 0 ? energy_data.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(energy_data, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        const bool valid = validateResults(world, MPI_COMM_WORLD);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
