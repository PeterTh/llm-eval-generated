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

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must be tightly packed");

// World state with MPI partitioning
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;   // local elements only
    std::vector<ElementDynamic> elements_dynamic;  // local + ghost elements
    std::vector<ElementDynamic> elements_dynamic_swap;

    // MPI partition info
    int rank, nprocs;
    int n_elems_root;
    int row_start, row_end;         // rows owned by this rank [row_start, row_end)
    int local_start, local_end;     // global element index range
    int local_n;                    // number of local elements
    int ghost_above_n, ghost_below_n;
    int total_n;                    // local_n + ghosts
    int rank_above, rank_below;     // neighbor ranks (-1 if none)
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh, partitioned across MPI ranks.
// Each rank owns a contiguous band of rows and stores ghost layers for neighbors.
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;

    // Initialize materials (replicated on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Partition rows across ranks
    world.row_start = world.rank * n_elems_root / world.nprocs;
    world.row_end = (world.rank + 1) * n_elems_root / world.nprocs;
    world.local_start = world.row_start * n_elems_root;
    world.local_end = world.row_end * n_elems_root;
    world.local_n = world.local_end - world.local_start;

    // Find neighbor ranks (skip empty ranks)
    world.rank_above = -1;
    world.rank_below = -1;
    if (world.local_n > 0) {
        if (world.row_start > 0) {
            for (int r = world.rank - 1; r >= 0; --r) {
                int rs = r * n_elems_root / world.nprocs;
                int re = (r + 1) * n_elems_root / world.nprocs;
                if (re > rs) { world.rank_above = r; break; }
            }
        }
        if (world.row_end < n_elems_root) {
            for (int r = world.rank + 1; r < world.nprocs; ++r) {
                int rs = r * n_elems_root / world.nprocs;
                int re = (r + 1) * n_elems_root / world.nprocs;
                if (re > rs) { world.rank_below = r; break; }
            }
        }
    }

    // Ghost layers: one row from each neighbor
    world.ghost_above_n = (world.rank_above >= 0) ? n_elems_root : 0;
    world.ghost_below_n = (world.rank_below >= 0) ? n_elems_root : 0;
    world.total_n = world.local_n + world.ghost_above_n + world.ghost_below_n;

    // Allocate arrays
    world.elements_static.resize(world.local_n);
    world.elements_dynamic.resize(world.total_n);
    world.elements_dynamic_swap.resize(world.total_n);

    for (int i = 0; i < world.total_n; ++i) {
        world.elements_dynamic[i] = {0.0, 0.0};
        world.elements_dynamic_swap[i] = {0.0, 0.0};
    }

    // Build connectivity with local/ghost index remapping
    // Memory layout: [0, local_n) = local, [local_n, local_n+ghost_above_n) = ghost above,
    //                [local_n+ghost_above_n, total_n) = ghost below
    for (int x = world.row_start; x < world.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int global_idx = x * n_elems_root + y;
            const int local_idx = global_idx - world.local_start;
            ElementStatic& elem = world.elements_static[local_idx];

            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int nb = 0; nb < 4; ++nb) {
                const int nx = x + offsets[nb][0];
                const int ny = y + offsets[nb][1];

                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_global = nx * n_elems_root + ny;
                    idx_t neighbor_local;

                    if (neighbor_global >= world.local_start && neighbor_global < world.local_end) {
                        neighbor_local = neighbor_global - world.local_start;
                    } else if (world.rank_above >= 0 && nx == world.row_start - 1) {
                        neighbor_local = world.local_n + ny;
                    } else if (world.rank_below >= 0 && nx == world.row_end) {
                        neighbor_local = world.local_n + world.ghost_above_n + ny;
                    } else {
                        continue;
                    }

                    elem.connected_idx[elem.num_connections] = neighbor_local;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow (only if owned by this rank)
    const int last = n_elems_root - 1;
    auto setMat = [&](int gx, int gy, idx_t mat_id) {
        int gidx = gx * n_elems_root + gy;
        if (gidx >= world.local_start && gidx < world.local_end) {
            world.elements_static[gidx - world.local_start].material_idx = mat_id;
        }
    };
    setMat(0, 0, INFLOW_MAT_ID);
    setMat(0, last, OUTFLOW_MAT_ID);
    setMat(last, 0, OUTFLOW_MAT_ID);
    setMat(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Compute element updates for a range [start, end)
inline void computeElements(World& world, int start, int end) {
    for (int i = start; i < end; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];

        val_t total_flux = mat.external_flow;

        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        world.elements_dynamic_swap[i].current_energy = elem_dyn.current_energy + total_flux;
        world.elements_dynamic_swap[i].total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation with communication/computation overlap
void runSimulation(World& world, const int n_iters) {
    if (world.local_n == 0) return;

    const int n = world.n_elems_root;

    // Interior elements don't depend on ghost data; boundary elements do
    int interior_start = (world.rank_above >= 0) ? n : 0;
    int interior_end = (world.rank_below >= 0) ? world.local_n - n : world.local_n;
    if (interior_start > interior_end) {
        interior_start = 0;
        interior_end = 0;
    }

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int nreqs = 0;

        // Post non-blocking ghost exchange
        if (world.rank_above >= 0) {
            MPI_Isend(&world.elements_dynamic[0],
                      n * sizeof(ElementDynamic), MPI_BYTE,
                      world.rank_above, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(&world.elements_dynamic[world.local_n],
                      n * sizeof(ElementDynamic), MPI_BYTE,
                      world.rank_above, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (world.rank_below >= 0) {
            MPI_Isend(&world.elements_dynamic[world.local_n - n],
                      n * sizeof(ElementDynamic), MPI_BYTE,
                      world.rank_below, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(&world.elements_dynamic[world.local_n + world.ghost_above_n],
                      n * sizeof(ElementDynamic), MPI_BYTE,
                      world.rank_below, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        // Compute interior elements while communication is in progress
        computeElements(world, interior_start, interior_end);

        // Wait for ghost data
        if (nreqs > 0) {
            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
        }

        // Compute boundary elements that depend on ghost data
        if (interior_start > 0) {
            computeElements(world, 0, interior_start);
        }
        if (interior_end < world.local_n) {
            computeElements(world, interior_end, world.local_n);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (called on rank 0 with gathered data)
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
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh (partitioned across ranks)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    world.rank = rank;
    world.nprocs = nprocs;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (report global totals from rank 0)
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
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Gather all element data to rank 0
    std::vector<ElementDynamic> all_elements;
    if (rank == 0) all_elements.resize(n_elems);

    std::vector<int> recv_counts(nprocs), recv_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int rs = r * n_elems_root / nprocs;
        int re = (r + 1) * n_elems_root / nprocs;
        recv_counts[r] = (re - rs) * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
        recv_displs[r] = rs * n_elems_root * static_cast<int>(sizeof(ElementDynamic));
    }

    MPI_Gatherv(world.elements_dynamic.data(),
                world.local_n * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                all_elements.data(),
                recv_counts.data(), recv_displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    int ret = 0;
    if (rank == 0) {
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
        
        // Compute hash for verification
        const uint64_t hash = computeHash(all_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        
        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(n_elems);
            for (const auto& elem : all_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (validate) {
            if (!validateResults(all_elements)) {
                ret = 1;
            }
        }
    }
    
    MPI_Finalize();
    return ret;
}
