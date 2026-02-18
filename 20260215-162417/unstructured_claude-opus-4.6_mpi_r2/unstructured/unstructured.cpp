#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <vector>

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
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Partition helpers
inline void getPartitionRange(int rank, int nprocs, idx_t n_elems, idx_t& start, idx_t& end) {
    idx_t chunk = n_elems / nprocs;
    idx_t rem = n_elems % nprocs;
    start = (idx_t)rank * chunk + std::min((idx_t)rank, rem);
    end = start + chunk + ((idx_t)rank < rem ? 1 : 0);
}

inline int getOwnerRank(idx_t g, idx_t n_elems, int nprocs) {
    idx_t chunk = n_elems / nprocs;
    idx_t rem = n_elems % nprocs;
    idx_t boundary = rem * (chunk + 1);
    if (g < boundary) return (int)(g / (chunk + 1));
    return (int)(rem + (g - boundary) / chunk);
}

// Ghost exchange communication pattern
struct GhostExchange {
    struct Neighbor {
        int rank;
        std::vector<idx_t> send_local;  // local indices of owned elements to send
        std::vector<idx_t> recv_local;  // local indices where ghost data goes
    };
    std::vector<Neighbor> neighbors;
    std::vector<std::vector<val_t>> send_bufs;
    std::vector<std::vector<val_t>> recv_bufs;

    void allocateBuffers() {
        send_bufs.resize(neighbors.size());
        recv_bufs.resize(neighbors.size());
        for (size_t i = 0; i < neighbors.size(); ++i) {
            send_bufs[i].resize(neighbors[i].send_local.size());
            recv_bufs[i].resize(neighbors[i].recv_local.size());
        }
    }
};

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation with MPI ghost exchange
void runSimulation(
    const std::vector<Material>& materials,
    const std::vector<ElementStatic>& local_static,
    std::vector<ElementDynamic>& local_dynamic,
    std::vector<ElementDynamic>& local_dynamic_swap,
    idx_t local_count,
    GhostExchange& ghost,
    const int n_iters)
{
    std::vector<MPI_Request> requests;

    for (int iter = 0; iter < n_iters; ++iter) {
        requests.clear();

        // Post non-blocking receives for ghost current_energy
        for (size_t n = 0; n < ghost.neighbors.size(); ++n) {
            if (ghost.recv_bufs[n].empty()) continue;
            MPI_Request req;
            MPI_Irecv(ghost.recv_bufs[n].data(), (int)ghost.recv_bufs[n].size(),
                      MPI_DOUBLE, ghost.neighbors[n].rank, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }

        // Pack and post non-blocking sends
        for (size_t n = 0; n < ghost.neighbors.size(); ++n) {
            const auto& ni = ghost.neighbors[n];
            for (size_t k = 0; k < ni.send_local.size(); ++k) {
                ghost.send_bufs[n][k] = local_dynamic[ni.send_local[k]].current_energy;
            }
            if (ghost.send_bufs[n].empty()) continue;
            MPI_Request req;
            MPI_Isend(ghost.send_bufs[n].data(), (int)ghost.send_bufs[n].size(),
                      MPI_DOUBLE, ni.rank, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }

        // Wait for all ghost communication
        if (!requests.empty()) {
            MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }

        // Unpack received ghost current_energy
        for (size_t n = 0; n < ghost.neighbors.size(); ++n) {
            const auto& ni = ghost.neighbors[n];
            for (size_t k = 0; k < ni.recv_local.size(); ++k) {
                local_dynamic[ni.recv_local[k]].current_energy = ghost.recv_bufs[n][k];
            }
        }

        // Compute all owned elements
        for (idx_t i = 0; i < local_count; ++i) {
            const ElementStatic& es = local_static[i];
            const ElementDynamic& ed = local_dynamic[i];
            const Material& mat = materials[es.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                total_flux += computeFlux(mat, ed, es.connected_flux[j],
                                          local_dynamic[es.connected_idx[j]]);
            }

            local_dynamic_swap[i].current_energy = ed.current_energy + total_flux;
            local_dynamic_swap[i].total_flux = ed.total_flux + std::abs(total_flux);
        }

        // Swap owned portion
        for (idx_t i = 0; i < local_count; ++i) {
            std::swap(local_dynamic[i], local_dynamic_swap[i]);
        }
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh (all ranks build full mesh for connectivity info)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    // Partition elements across MPI ranks
    idx_t local_start, local_end;
    getPartitionRange(rank, nprocs, (idx_t)n_elems, local_start, local_end);
    idx_t local_count = local_end - local_start;

    // Find ghost elements and build send/recv maps using mesh symmetry
    std::map<int, std::set<idx_t>> recv_map; // src_rank -> ghost global indices I need
    std::map<int, std::set<idx_t>> send_map; // dest_rank -> my global indices they need
    for (idx_t i = local_start; i < local_end; ++i) {
        for (idx_t j = 0; j < world.elements_static[i].num_connections; ++j) {
            idx_t nbr = world.elements_static[i].connected_idx[j];
            if (nbr < local_start || nbr >= local_end) {
                int owner = getOwnerRank(nbr, (idx_t)n_elems, nprocs);
                recv_map[owner].insert(nbr);
                send_map[owner].insert(i);
            }
        }
    }

    // Collect unique ghost global indices and build global-to-local mapping
    std::vector<idx_t> ghost_globals;
    for (auto& [r, gset] : recv_map) {
        for (idx_t g : gset) ghost_globals.push_back(g);
    }
    std::sort(ghost_globals.begin(), ghost_globals.end());
    ghost_globals.erase(std::unique(ghost_globals.begin(), ghost_globals.end()),
                        ghost_globals.end());
    idx_t n_ghosts = ghost_globals.size();

    std::map<idx_t, idx_t> g2l;
    for (idx_t i = 0; i < local_count; ++i) {
        g2l[local_start + i] = i;
    }
    for (idx_t i = 0; i < n_ghosts; ++i) {
        g2l[ghost_globals[i]] = local_count + i;
    }

    // Extract local static data with remapped connection indices
    std::vector<ElementStatic> local_static(local_count);
    for (idx_t i = 0; i < local_count; ++i) {
        local_static[i] = world.elements_static[local_start + i];
        for (idx_t j = 0; j < local_static[i].num_connections; ++j) {
            local_static[i].connected_idx[j] = g2l[local_static[i].connected_idx[j]];
        }
    }

    // Extract local dynamic data (owned + ghost slots)
    std::vector<ElementDynamic> local_dynamic(local_count + n_ghosts);
    std::vector<ElementDynamic> local_dynamic_swap(local_count + n_ghosts);
    for (idx_t i = 0; i < local_count; ++i) {
        local_dynamic[i] = world.elements_dynamic[local_start + i];
    }
    for (idx_t i = 0; i < n_ghosts; ++i) {
        local_dynamic[local_count + i] = world.elements_dynamic[ghost_globals[i]];
    }
    for (idx_t i = 0; i < local_count + n_ghosts; ++i) {
        local_dynamic_swap[i] = {0.0, 0.0};
    }

    // Build ghost exchange structure
    GhostExchange ghost;
    std::set<int> neighbor_ranks;
    for (auto& [r, _] : recv_map) neighbor_ranks.insert(r);
    for (auto& [r, _] : send_map) neighbor_ranks.insert(r);
    for (int r : neighbor_ranks) {
        GhostExchange::Neighbor ni;
        ni.rank = r;
        if (send_map.count(r)) {
            for (idx_t g : send_map[r]) {
                ni.send_local.push_back(g - local_start);
            }
        }
        if (recv_map.count(r)) {
            for (idx_t g : recv_map[r]) {
                ni.recv_local.push_back(g2l[g]);
            }
        }
        ghost.neighbors.push_back(std::move(ni));
    }
    ghost.allocateBuffers();

    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Free global mesh data (no longer needed)
    { std::vector<ElementStatic>().swap(world.elements_static); }
    { std::vector<ElementDynamic>().swap(world.elements_dynamic); }
    { std::vector<ElementDynamic>().swap(world.elements_dynamic_swap); }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world.materials, local_static, local_dynamic, local_dynamic_swap,
                  local_count, ghost, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather results on rank 0
    std::vector<int> gather_counts(nprocs), gather_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        idx_t rs, re;
        getPartitionRange(r, nprocs, (idx_t)n_elems, rs, re);
        gather_counts[r] = (int)(re - rs);
        gather_displs[r] = (int)rs;
    }

    std::vector<val_t> local_energy(local_count), local_flux(local_count);
    for (idx_t i = 0; i < local_count; ++i) {
        local_energy[i] = local_dynamic[i].current_energy;
        local_flux[i] = local_dynamic[i].total_flux;
    }

    std::vector<val_t> all_energy, all_flux;
    if (rank == 0) {
        all_energy.resize(n_elems);
        all_flux.resize(n_elems);
    }

    MPI_Gatherv(local_energy.data(), (int)local_count, MPI_DOUBLE,
                rank == 0 ? all_energy.data() : nullptr,
                gather_counts.data(), gather_displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), (int)local_count, MPI_DOUBLE,
                rank == 0 ? all_flux.data() : nullptr,
                gather_counts.data(), gather_displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Reconstruct full element state
        std::vector<ElementDynamic> all_dynamic(n_elems);
        for (int i = 0; i < n_elems; ++i) {
            all_dynamic[i].current_energy = all_energy[i];
            all_dynamic[i].total_flux = all_flux[i];
        }

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        const uint64_t hash = computeHash(all_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(n_elems);
            for (const auto& elem : all_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(all_dynamic);
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
