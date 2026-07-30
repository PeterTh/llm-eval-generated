#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <numeric>

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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local indices including ghost)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (local to each MPI process)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;       // Only owned elements
    std::vector<ElementDynamic> elements_dynamic;     // Owned + ghost rows
    std::vector<ElementDynamic> elements_dynamic_swap;// Owned + ghost rows
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Domain decomposition info for 1D row-based decomposition
struct Decomp {
    int rank;
    int nprocs;
    int n_elems_root;
    int x_start;          // First owned global row
    int x_end;            // One past last owned global row
    int local_x;          // Number of owned rows (x_end - x_start)
    int rank_above;       // Rank of process above (MPI_PROC_NULL if none)
    int rank_below;       // Rank of process below (MPI_PROC_NULL if none)

    // Local array layout:
    // Row 0: ghost from above (global row x_start - 1)
    // Rows 1..local_x: owned (global rows x_start..x_end-1)
    // Row local_x+1: ghost from below (global row x_end)
    static constexpr int owned_offset = 1;

    int local_array_rows() const { return local_x + 2; }
    int n_owned() const { return local_x * n_elems_root; }
    int n_local() const { return local_array_rows() * n_elems_root; }
    int halo_size() const { return n_elems_root; } // elements per halo row

    // Convert global (gx, gy) to local flat index
    int global_to_local(int gx, int gy) const {
        return (gx - x_start + owned_offset) * n_elems_root + gy;
    }

    // Offset in elements_dynamic array for send/recv buffers
    int send_top_offset() const { return owned_offset * n_elems_root; }
    int send_bottom_offset() const { return (owned_offset + local_x - 1) * n_elems_root; }
    int recv_top_offset() const { return 0; }
    int recv_bottom_offset() const { return (owned_offset + local_x) * n_elems_root; }
};

static Decomp computeDecomp(int rank, int nprocs, int n_elems_root) {
    Decomp d;
    d.rank = rank;
    d.nprocs = nprocs;
    d.n_elems_root = n_elems_root;

    int base_rows = n_elems_root / nprocs;
    int remainder = n_elems_root % nprocs;
    if (rank < remainder) {
        d.local_x = base_rows + 1;
        d.x_start = rank * (base_rows + 1);
    } else {
        d.local_x = base_rows;
        d.x_start = remainder * (base_rows + 1) + (rank - remainder) * base_rows;
    }
    d.x_end = d.x_start + d.local_x;

    d.rank_above = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    d.rank_below = (rank < nprocs - 1) ? (rank + 1) : MPI_PROC_NULL;

    return d;
}

// Build a 2D square grid as an unstructured mesh (distributed across MPI ranks)
void buildSquare2D(World& world, const Decomp& decomp) {
    const int n_elems_root = decomp.n_elems_root;
    const int x_start = decomp.x_start;
    const int x_end = decomp.x_end;

    // Initialize materials (same on all processes)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements_static for owned elements only
    const int n_owned = decomp.n_owned();
    world.elements_static.resize(n_owned);

    // Allocate elements_dynamic for owned + ghost rows
    const int n_local = decomp.n_local();
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all dynamic elements to zero
    for (int i = 0; i < n_local; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity for owned elements, remapping neighbor indices to local
    int static_idx = 0;
    for (int x = x_start; x < x_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            ElementStatic& elem = world.elements_static[static_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int local_neighbor_idx = decomp.global_to_local(nx, ny);
                    elem.connected_idx[elem.num_connections] = local_neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
            static_idx++;
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    auto set_material = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= x_start && gx < x_end) {
            int s_idx = (gx - x_start) * n_elems_root + gy;
            world.elements_static[s_idx].material_idx = mat_id;
        }
    };

    const int last = n_elems_root - 1;
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Exchange halo (ghost) rows between neighboring MPI processes
inline void exchangeHalos(World& world, const Decomp& decomp) {
    const int halo_bytes = decomp.halo_size() * static_cast<int>(sizeof(ElementDynamic));

    // Exchange with rank_above: send my top owned row, receive into my top ghost row
    MPI_Sendrecv(
        reinterpret_cast<char*>(&world.elements_dynamic[decomp.send_top_offset()]),
        halo_bytes, MPI_BYTE,
        decomp.rank_above, 0,
        reinterpret_cast<char*>(&world.elements_dynamic[decomp.recv_top_offset()]),
        halo_bytes, MPI_BYTE,
        decomp.rank_above, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Exchange with rank_below: send my bottom owned row, receive into my bottom ghost row
    MPI_Sendrecv(
        reinterpret_cast<char*>(&world.elements_dynamic[decomp.send_bottom_offset()]),
        halo_bytes, MPI_BYTE,
        decomp.rank_below, 1,
        reinterpret_cast<char*>(&world.elements_dynamic[decomp.recv_bottom_offset()]),
        halo_bytes, MPI_BYTE,
        decomp.rank_below, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}

// Run simulation for n_iters iterations (MPI-parallel)
void runSimulation(World& world, const int n_iters, const Decomp& decomp) {
    const int n_owned = decomp.n_owned();
    // Offset into elements_dynamic where owned elements start
    const int owned_start = decomp.owned_offset * decomp.n_elems_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange halo rows so ghost rows have up-to-date neighbor data
        exchangeHalos(world, decomp);

        // Update all owned elements
        for (int i = 0; i < n_owned; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const int local_dyn_idx = owned_start + i;
            const ElementDynamic& elem_dyn = world.elements_dynamic[local_dyn_idx];
            const Material& mat = world.materials[elem_static.material_idx];

            // Start with external flow
            val_t total_flux = mat.external_flow;

            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }

            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[local_dyn_idx];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (operates on globally gathered data)
bool validateResults(const std::vector<ElementDynamic>& all_elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : all_elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather all owned ElementDynamic data to rank 0 in global order
std::vector<ElementDynamic> gatherResults(const World& world, const Decomp& decomp) {
    const int n_elems = decomp.n_elems_root * decomp.n_elems_root;
    std::vector<ElementDynamic> all_elements;

    // Gather counts from all processes
    int local_owned = decomp.n_owned();
    std::vector<int> recv_counts(decomp.nprocs);
    MPI_Gather(&local_owned, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // Compute displacements (in number of ElementDynamic elements)
    std::vector<int> displacements(decomp.nprocs, 0);
    if (decomp.rank == 0) {
        for (int i = 1; i < decomp.nprocs; ++i) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
        all_elements.resize(n_elems);
    }

    // Send buffer: owned portion of elements_dynamic (contiguous block)
    const ElementDynamic* send_buf = world.elements_dynamic.data() + decomp.send_top_offset();

    // Use MPI_BYTE for portable struct transfer
    int send_bytes = local_owned * static_cast<int>(sizeof(ElementDynamic));
    std::vector<int> recv_bytes(decomp.nprocs);
    std::vector<int> byte_displacements(decomp.nprocs);
    if (decomp.rank == 0) {
        for (int i = 0; i < decomp.nprocs; ++i) {
            recv_bytes[i] = recv_counts[i] * static_cast<int>(sizeof(ElementDynamic));
            byte_displacements[i] = displacements[i] * static_cast<int>(sizeof(ElementDynamic));
        }
    }

    MPI_Gatherv(
        const_cast<ElementDynamic*>(send_buf), send_bytes, MPI_BYTE,
        all_elements.data(), recv_bytes.data(), byte_displacements.data(), MPI_BYTE,
        0, MPI_COMM_WORLD
    );

    return all_elements;
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

    // Parse command line arguments (all ranks parse identically)
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
            printUsage(argv[0]);
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
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("MPI processes: %d\n", nprocs);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Compute domain decomposition
    Decomp decomp = computeDecomp(rank, nprocs, n_elems_root);

    // Build the local portion of the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, decomp);

    // Calculate memory usage (per process and total)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t local_mem = static_mem + dynamic_mem;
    unsigned long long total_mem_ull = 0;
    unsigned long long local_mem_ull = static_cast<unsigned long long>(local_mem);
    MPI_Reduce(&local_mem_ull, &total_mem_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Memory usage: %.2f MB total (%.2f MB per process avg)\n",
               total_mem_ull / (1024.0 * 1024.0),
               (total_mem_ull / (double)nprocs) / (1024.0 * 1024.0));
        printf("\n");
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    double start_time = MPI_Wtime();

    runSimulation(world, n_iters, decomp);

    double end_time = MPI_Wtime();
    double duration_ms = (end_time - start_time) * 1000.0;

    // Get max time across all processes for conservative performance metrics
    double max_duration_ms = 0.0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_duration_ms));

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (max_duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Gather all results to rank 0 for hash, validation, and output
    std::vector<ElementDynamic> all_elements = gatherResults(world, decomp);

    if (rank == 0) {
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
