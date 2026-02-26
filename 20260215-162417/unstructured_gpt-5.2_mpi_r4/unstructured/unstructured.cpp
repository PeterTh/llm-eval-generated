#include <algorithm>
#include <chrono>
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

// World state (distributed by contiguous X-rows across MPI ranks)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;        // local elements
    std::vector<ElementDynamic> elements_dynamic;      // local elements
    std::vector<ElementDynamic> elements_dynamic_swap; // local elements

    int n_elems_root = 0;
    idx_t global_n_elems = 0;

    int row_start = 0;        // global starting row owned by this rank
    int local_rows = 0;       // number of rows owned by this rank
    idx_t global_start_idx = 0; // global starting element index owned by this rank
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct RowDecomp {
    int row_start;
    int local_rows;
    int row_end;
    idx_t global_start_idx;
    idx_t local_n_elems;
};

static inline RowDecomp decomposeRows(const int n_root, const int rank, const int size) {
    const int base = n_root / size;
    const int rem = n_root % size;

    const int local_rows = base + ((rank < rem) ? 1 : 0);
    const int row_start = rank * base + std::min(rank, rem);

    RowDecomp d;
    d.row_start = row_start;
    d.local_rows = local_rows;
    d.row_end = row_start + local_rows;
    d.global_start_idx = static_cast<idx_t>(row_start) * static_cast<idx_t>(n_root);
    d.local_n_elems = static_cast<idx_t>(local_rows) * static_cast<idx_t>(n_root);
    return d;
}

// Build a 2D square grid as an unstructured mesh (distributed by rows)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    world.n_elems_root = n_elems_root;
    world.global_n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);

    const RowDecomp d = decomposeRows(n_elems_root, rank, size);
    world.row_start = d.row_start;
    world.local_rows = d.local_rows;
    world.global_start_idx = d.global_start_idx;

    // Initialize materials (replicated)
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});   // Default material
    world.materials.emplace_back(Material{0.8, 0.5});   // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});  // Outflow material

    // Allocate local elements
    world.elements_static.resize(d.local_n_elems);
    world.elements_dynamic.resize(d.local_n_elems);
    world.elements_dynamic_swap.resize(d.local_n_elems);

    // Initialize all local elements with default material and zero energy
    for (idx_t li = 0; li < d.local_n_elems; ++li) {
        const idx_t gi = d.global_start_idx + li;

        ElementStatic& es = world.elements_static[li];
        es.material_idx = DEFAULT_MAT_ID;
        es.num_connections = 0;

        ElementDynamic& ed = world.elements_dynamic[li];
        ed.current_energy = 0.0;
        ed.total_flux = 0.0;

        // Set corner elements as inflow/outflow to create interesting dynamics
        const idx_t last = static_cast<idx_t>(n_elems_root - 1);
        const idx_t c0 = 0;
        const idx_t c1 = last;
        const idx_t c2 = last * static_cast<idx_t>(n_elems_root);
        const idx_t c3 = c2 + last;
        if (gi == c0 || gi == c3) {
            es.material_idx = INFLOW_MAT_ID;
        } else if (gi == c1 || gi == c2) {
            es.material_idx = OUTFLOW_MAT_ID;
        }

        // Build connectivity for this element
        const int x = static_cast<int>(gi / static_cast<idx_t>(n_elems_root));
        const int y = static_cast<int>(gi % static_cast<idx_t>(n_elems_root));
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * static_cast<idx_t>(n_elems_root) +
                                           static_cast<idx_t>(ny);
                es.connected_idx[es.num_connections] = neighbor_idx;
                es.connected_flux[es.num_connections] = 1.0;
                es.num_connections++;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const val_t this_energy,
                         const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations (MPI row decomposition + halo exchange)
void runSimulation(World& world, const int n_iters, MPI_Comm comm, const int rank, const int size) {
    (void)size;

    const int n_root = world.n_elems_root;
    const int row_start = world.row_start;
    const int local_rows = world.local_rows;
    const int row_end = row_start + local_rows;
    const idx_t global_start = world.global_start_idx;
    const idx_t local_n_elems = static_cast<idx_t>(world.elements_static.size());

    if (local_rows <= 0 || local_n_elems == 0) {
        return;
    }

    const int up_rank = (row_start > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int down_rank = (row_end < n_root) ? (rank + 1) : MPI_PROC_NULL;

    std::vector<val_t> halo_up(static_cast<size_t>(n_root));
    std::vector<val_t> halo_down(static_cast<size_t>(n_root));
    std::vector<val_t> send_first(static_cast<size_t>(n_root));
    std::vector<val_t> send_last(static_cast<size_t>(n_root));

    const idx_t row_stride = static_cast<idx_t>(n_root);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int req_count = 0;

        // Post receives first
        if (up_rank != MPI_PROC_NULL) {
            MPI_Irecv(halo_up.data(), n_root, MPI_DOUBLE, up_rank, 1, comm, &reqs[req_count++]);
        }
        if (down_rank != MPI_PROC_NULL) {
            MPI_Irecv(halo_down.data(), n_root, MPI_DOUBLE, down_rank, 0, comm, &reqs[req_count++]);
        }

        // Pack + send boundary rows
        if (up_rank != MPI_PROC_NULL) {
            for (int y = 0; y < n_root; ++y) {
                send_first[static_cast<size_t>(y)] = world.elements_dynamic[static_cast<size_t>(y)].current_energy;
            }
            MPI_Isend(send_first.data(), n_root, MPI_DOUBLE, up_rank, 0, comm, &reqs[req_count++]);
        }
        if (down_rank != MPI_PROC_NULL) {
            const idx_t last_row_off = static_cast<idx_t>(local_rows - 1) * row_stride;
            for (int y = 0; y < n_root; ++y) {
                const idx_t li = last_row_off + static_cast<idx_t>(y);
                send_last[static_cast<size_t>(y)] = world.elements_dynamic[static_cast<size_t>(li)].current_energy;
            }
            MPI_Isend(send_last.data(), n_root, MPI_DOUBLE, down_rank, 1, comm, &reqs[req_count++]);
        }

        // Compute interior rows while halos are in-flight
        const int interior_begin = 1;
        const int interior_end = std::max(local_rows - 1, 1);
        for (int lr = interior_begin; lr < interior_end; ++lr) {
            const idx_t base_off = static_cast<idx_t>(lr) * row_stride;
            for (int y = 0; y < n_root; ++y) {
                const idx_t li = base_off + static_cast<idx_t>(y);
                const ElementStatic& elem_static = world.elements_static[static_cast<size_t>(li)];
                const ElementDynamic& elem_dyn = world.elements_dynamic[static_cast<size_t>(li)];
                const Material& mat = world.materials[elem_static.material_idx];

                const val_t this_energy = elem_dyn.current_energy;
                val_t total_flux = mat.external_flow;

                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const val_t nbr_energy = world.elements_dynamic[static_cast<size_t>(neighbor_idx - global_start)].current_energy;
                    total_flux += computeFlux(mat, this_energy, elem_static.connected_flux[j], nbr_energy);
                }

                ElementDynamic& elem_write = world.elements_dynamic_swap[static_cast<size_t>(li)];
                elem_write.current_energy = this_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Compute boundary rows (may need halos)
        for (int boundary = 0; boundary < 2; ++boundary) {
            const int lr = (boundary == 0) ? 0 : (local_rows - 1);
            if (lr < 0 || lr >= local_rows) {
                continue;
            }
            // Avoid double-computing the only row when local_rows==1
            if (local_rows == 1 && boundary == 1) {
                continue;
            }

            const idx_t base_off = static_cast<idx_t>(lr) * row_stride;
            for (int y = 0; y < n_root; ++y) {
                const idx_t li = base_off + static_cast<idx_t>(y);
                const ElementStatic& elem_static = world.elements_static[static_cast<size_t>(li)];
                const ElementDynamic& elem_dyn = world.elements_dynamic[static_cast<size_t>(li)];
                const Material& mat = world.materials[elem_static.material_idx];

                const val_t this_energy = elem_dyn.current_energy;
                val_t total_flux = mat.external_flow;

                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const int nbr_row = static_cast<int>(neighbor_idx / static_cast<idx_t>(n_root));
                    const int nbr_col = static_cast<int>(neighbor_idx % static_cast<idx_t>(n_root));

                    val_t nbr_energy;
                    if (nbr_row < row_start) {
                        nbr_energy = halo_up[static_cast<size_t>(nbr_col)];
                    } else if (nbr_row >= row_end) {
                        nbr_energy = halo_down[static_cast<size_t>(nbr_col)];
                    } else {
                        nbr_energy = world.elements_dynamic[static_cast<size_t>(neighbor_idx - global_start)].current_energy;
                    }

                    total_flux += computeFlux(mat, this_energy, elem_static.connected_flux[j], nbr_energy);
                }

                ElementDynamic& elem_write = world.elements_dynamic_swap[static_cast<size_t>(li)];
                elem_write.current_energy = this_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (global reductions)
bool validateResults(const World& world, MPI_Comm comm, const int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : world.elements_dynamic) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    int valid_i = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid_i = 0;
        }

        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }

        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid_i = 0;
        }

        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid_i = 0;
        }

        if (valid_i) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid_i, 1, MPI_INT, 0, comm);
    return valid_i != 0;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const idx_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t gi = static_cast<uint64_t>(global_offset + static_cast<idx_t>(i));
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
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

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    if (rank == 0) {
        // Calculate memory usage (global, for parity with the serial benchmark)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }

    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    runSimulation(world, n_iters, comm, rank, size);

    MPI_Barrier(comm);
    const double t1 = MPI_Wtime();

    const double local_sec = t1 - t0;
    double max_sec = 0.0;
    MPI_Reduce(&local_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    long duration_ms = 0;
    if (rank == 0) {
        duration_ms = static_cast<long>(max_sec * 1000.0); // truncate for parity with duration_cast
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (global XOR reduction)
    const uint64_t local_hash = computeHash(world.elements_dynamic, world.global_start_idx);
    unsigned long long local_hash_ull = static_cast<unsigned long long>(local_hash);
    unsigned long long hash_ull = 0;
    MPI_Reduce(&local_hash_ull, &hash_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, comm);

    if (rank == 0) {
        printf("  Result hash: %016llX\n", hash_ull);
        printf("\n");
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        const int local_count = world.local_rows * n_elems_root;
        std::vector<double> local_energy(static_cast<size_t>(local_count));
        for (size_t i = 0; i < local_energy.size(); ++i) {
            local_energy[i] = world.elements_dynamic[i].current_energy;
        }

        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> energyData;

        if (rank == 0) {
            counts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));

            const int base = n_elems_root / size;
            const int rem = n_elems_root % size;
            for (int r = 0; r < size; ++r) {
                const int rows_r = base + ((r < rem) ? 1 : 0);
                const int row_start_r = r * base + std::min(r, rem);
                counts[static_cast<size_t>(r)] = rows_r * n_elems_root;
                displs[static_cast<size_t>(r)] = row_start_r * n_elems_root;
            }

            energyData.resize(static_cast<size_t>(n_elems));
        }

        MPI_Gatherv(local_energy.empty() ? nullptr : local_energy.data(),
                    local_count,
                    MPI_DOUBLE,
                    (rank == 0) ? energyData.data() : nullptr,
                    (rank == 0) ? counts.data() : nullptr,
                    (rank == 0) ? displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    comm);

        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        const bool valid = validateResults(world, comm, rank);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
