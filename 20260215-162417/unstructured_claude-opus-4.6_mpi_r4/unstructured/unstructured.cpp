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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local/ghost)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (local portion per MPI rank)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;   // local elements only
    std::vector<ElementDynamic> elements_dynamic;  // local + ghost elements
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build local portion of a 2D square grid as an unstructured mesh.
// Each rank owns rows [x_start, x_end). Ghost rows from neighbors are
// stored after local elements for halo exchange.
void buildSquare2DLocal(World& world, const int n_elems_root,
                        int x_start, int x_end,
                        int n_ghost_above, int n_ghost_below) {
    const int local_rows = x_end - x_start;
    const int local_count = local_rows * n_elems_root;
    const int n_ghosts = n_ghost_above + n_ghost_below;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements + ghost slots
    world.elements_static.resize(local_count);
    world.elements_dynamic.resize(local_count + n_ghosts);
    world.elements_dynamic_swap.resize(local_count + n_ghosts);

    for (int i = 0; i < local_count + n_ghosts; ++i) {
        world.elements_dynamic[i] = {0.0, 0.0};
        world.elements_dynamic_swap[i] = {0.0, 0.0};
    }

    const int last = n_elems_root - 1;

    // Build connectivity for local elements with global-to-local index mapping:
    //   Local element (gx,y): local index = (gx - x_start) * n_elems_root + y
    //   Ghost above (row x_start-1): index = local_count + y
    //   Ghost below (row x_end):     index = local_count + n_ghost_above + y
    for (int lx = 0; lx < local_rows; ++lx) {
        const int gx = x_start + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            const int li = lx * n_elems_root + y;
            ElementStatic& elem = world.elements_static[li];

            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Set corner elements as inflow/outflow
            if (gx == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            else if (gx == 0 && y == last) elem.material_idx = OUTFLOW_MAT_ID;
            else if (gx == last && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            else if (gx == last && y == last) elem.material_idx = INFLOW_MAT_ID;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = y + offsets[n][1];

                if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root)
                    continue;

                idx_t mapped;
                if (nx >= x_start && nx < x_end) {
                    mapped = static_cast<idx_t>((nx - x_start) * n_elems_root + ny);
                } else if (nx == x_start - 1) {
                    mapped = static_cast<idx_t>(local_count + ny);
                } else if (nx == x_end) {
                    mapped = static_cast<idx_t>(local_count + n_ghost_above + ny);
                } else {
                    continue;
                }

                elem.connected_idx[elem.num_connections] = mapped;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with MPI ghost exchange.
// Overlaps communication with computation: interior elements (no ghost
// dependency) are computed while halo exchange is in flight.
void runSimulation(World& world, const int n_iters, const int n_elems_root,
                   const int local_count, const int local_rows,
                   const int n_ghost_above, const int n_ghost_below,
                   const int rank_above, const int rank_below) {
    if (local_count == 0) return;

    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double),
                  "ElementDynamic must be two contiguous doubles for MPI transfer");

    const ElementStatic* es = world.elements_static.data();
    const Material* mats = world.materials.data();

    // Interior rows [1, local_rows-2] don't touch ghost data
    const int interior_start = n_elems_root;
    const int interior_end = (local_rows - 1) * n_elems_root;
    const int last_row_start = (local_rows - 1) * n_elems_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* rd = world.elements_dynamic.data();
        ElementDynamic* wr = world.elements_dynamic_swap.data();

        // Initiate non-blocking ghost exchange
        MPI_Request reqs[4];
        int nreq = 0;

        // Send first local row up; receive ghost-above from rank above
        MPI_Isend(rd, n_elems_root * 2, MPI_DOUBLE,
                  rank_above, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rd + local_count, n_ghost_above * 2, MPI_DOUBLE,
                  rank_above, 1, MPI_COMM_WORLD, &reqs[nreq++]);

        // Send last local row down; receive ghost-below from rank below
        MPI_Isend(rd + last_row_start, n_elems_root * 2, MPI_DOUBLE,
                  rank_below, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(rd + local_count + n_ghost_above, n_ghost_below * 2, MPI_DOUBLE,
                  rank_below, 0, MPI_COMM_WORLD, &reqs[nreq++]);

        // Compute interior elements while communication is in flight
        for (int i = interior_start; i < interior_end; ++i) {
            const ElementStatic& s = es[i];
            const ElementDynamic& d = rd[i];
            const Material& m = mats[s.material_idx];
            val_t tf = m.external_flow;
            for (idx_t j = 0; j < s.num_connections; ++j) {
                tf += computeFlux(m, d, s.connected_flux[j], rd[s.connected_idx[j]]);
            }
            wr[i].current_energy = d.current_energy + tf;
            wr[i].total_flux = d.total_flux + std::abs(tf);
        }

        // Wait for ghost exchange to complete
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Compute boundary elements (first row, needs ghost-above)
        for (int i = 0; i < n_elems_root; ++i) {
            const ElementStatic& s = es[i];
            const ElementDynamic& d = rd[i];
            const Material& m = mats[s.material_idx];
            val_t tf = m.external_flow;
            for (idx_t j = 0; j < s.num_connections; ++j) {
                tf += computeFlux(m, d, s.connected_flux[j], rd[s.connected_idx[j]]);
            }
            wr[i].current_energy = d.current_energy + tf;
            wr[i].total_flux = d.total_flux + std::abs(tf);
        }

        // Compute boundary elements (last row, needs ghost-below)
        if (local_rows > 1) {
            for (int i = last_row_start; i < local_count; ++i) {
                const ElementStatic& s = es[i];
                const ElementDynamic& d = rd[i];
                const Material& m = mats[s.material_idx];
                val_t tf = m.external_flow;
                for (idx_t j = 0; j < s.num_connections; ++j) {
                    tf += computeFlux(m, d, s.connected_flux[j], rd[s.connected_idx[j]]);
                }
                wr[i].current_energy = d.current_energy + tf;
                wr[i].total_flux = d.total_flux + std::abs(tf);
            }
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
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
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks)
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
            if (rank == 0) printUsage(argv[0]);
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

    // Row-based domain decomposition
    const int rows_per_rank = n_elems_root / nprocs;
    const int remainder = n_elems_root % nprocs;
    int x_start, x_end;
    if (rank < remainder) {
        x_start = rank * (rows_per_rank + 1);
        x_end = x_start + rows_per_rank + 1;
    } else {
        x_start = remainder * (rows_per_rank + 1) + (rank - remainder) * rows_per_rank;
        x_end = x_start + rows_per_rank;
    }
    const int local_rows = x_end - x_start;
    const int local_count = local_rows * n_elems_root;

    // Determine ghost regions and neighbor ranks
    int n_ghost_above = 0, n_ghost_below = 0;
    int rank_above = MPI_PROC_NULL, rank_below = MPI_PROC_NULL;
    if (local_rows > 0) {
        if (x_start > 0) {
            n_ghost_above = n_elems_root;
            rank_above = rank - 1;
        }
        if (x_end < n_elems_root) {
            n_ghost_below = n_elems_root;
            rank_below = rank + 1;
        }
    }
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the local portion of the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2DLocal(world, n_elems_root, x_start, x_end,
                       n_ghost_above, n_ghost_below);
    
    // Calculate memory usage (total across all ranks, as in original)
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    
    runSimulation(world, n_iters, n_elems_root, local_count, local_rows,
                  n_ghost_above, n_ghost_below, rank_above, rank_below);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    const double duration_s = t1 - t0;
    double max_duration_s = 0.0;
    MPI_Reduce(&duration_s, &max_duration_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(max_duration_s * 1000.0);
    
    // Gather all element data on rank 0 for output and validation
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));

    const int send_count = local_count * 2;  // doubles per rank
    std::vector<int> recv_counts(nprocs, 0);
    std::vector<int> recv_displs(nprocs, 0);
    MPI_Gather(&send_count, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<ElementDynamic> all_elements;
    if (rank == 0) {
        recv_displs[0] = 0;
        for (int i = 1; i < nprocs; ++i) {
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        }
        all_elements.resize(n_elems);
    }

    MPI_Gatherv(world.elements_dynamic.data(), send_count, MPI_DOUBLE,
                rank == 0 ? reinterpret_cast<double*>(all_elements.data()) : nullptr,
                recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / max_duration_s / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        
        // Compute hash for verification
        const uint64_t hash = computeHash(all_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        
        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(all_elements.size());
            for (const auto& elem : all_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (validate) {
            bool valid = validateResults(all_elements);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
