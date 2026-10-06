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

// World state (local partition of the global mesh owned by one MPI rank)
//
// The global n_root x n_root mesh is partitioned into contiguous blocks of
// grid rows (x). Each rank stores its owned rows plus one ghost row above and
// below. Local element index = (x - row_begin + 1) * n_root + y, so local
// index 0..n_root-1 is the upper ghost row and the last n_root entries are the
// lower ghost row. Connectivity indices refer to local indices.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;      // only owned elements are valid
    std::vector<ElementDynamic> elements_dynamic;    // owned + ghost rows
    std::vector<ElementDynamic> elements_dynamic_swap;

    int n_root = 0;          // global grid dimension
    int row_begin = 0;       // first owned global row
    int n_rows = 0;          // number of owned rows
    int rank_up = MPI_PROC_NULL;    // rank owning row_begin - 1
    int rank_down = MPI_PROC_NULL;  // rank owning row_begin + n_rows
    MPI_Datatype energy_row_type = MPI_DATATYPE_NULL;  // n_root strided energies

    size_t ownedBegin() const { return static_cast<size_t>(n_root); }
    size_t ownedEnd() const { return static_cast<size_t>(n_rows + 1) * n_root; }
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of rows over ranks
static void rowRange(int n_root, int n_parts, int part, int& begin, int& count) {
    const int base = n_root / n_parts;
    const int rem = n_root % n_parts;
    count = base + (part < rem ? 1 : 0);
    begin = part * base + std::min(part, rem);
}

// Build the local part of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    world.n_root = n_elems_root;
    rowRange(n_elems_root, size, rank, world.row_begin, world.n_rows);
    if (world.n_rows > 0) {
        if (world.row_begin > 0) world.rank_up = rank - 1;
        if (world.row_begin + world.n_rows < n_elems_root) world.rank_down = rank + 1;
    }

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (owned rows + 2 ghost rows)
    const size_t n_local = static_cast<size_t>(world.n_rows + 2) * n_elems_root;
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    const int x_off = world.row_begin - 1;
    for (int x = world.row_begin; x < world.row_begin + world.n_rows; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(x - x_off) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx = static_cast<size_t>(nx - x_off) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    // (same assignment order as the global construction)
    const int last = n_elems_root - 1;
    auto setMat = [&](int x, int y, idx_t mat) {
        if (x >= world.row_begin && x < world.row_begin + world.n_rows) {
            world.elements_static[static_cast<size_t>(x - x_off) * n_elems_root + y].material_idx = mat;
        }
    };
    setMat(0, 0, INFLOW_MAT_ID);
    setMat(0, last, OUTFLOW_MAT_ID);
    setMat(last, 0, OUTFLOW_MAT_ID);
    setMat(last, last, INFLOW_MAT_ID);

    // Datatype describing the energies of one grid row (strided in ElementDynamic)
    MPI_Datatype tmp;
    MPI_Type_vector(n_elems_root, 1, sizeof(ElementDynamic) / sizeof(val_t), MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(ElementDynamic), &world.energy_row_type);
    MPI_Type_commit(&world.energy_row_type);
    MPI_Type_free(&tmp);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update local elements in [begin, end)
static inline void updateRange(const World& world, const ElementDynamic* __restrict dyn,
                               ElementDynamic* __restrict out, size_t begin, size_t end) {
    const ElementStatic* __restrict stat = world.elements_static.data();
    const Material* __restrict mats = world.materials.data();
    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = stat[i];
        const ElementDynamic& elem_dyn = dyn[i];
        const Material& mat = mats[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = dyn[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = out[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    if (world.n_rows == 0) return;

    const size_t n = static_cast<size_t>(world.n_root);
    const size_t own_begin = world.ownedBegin();
    const size_t own_end = world.ownedEnd();
    const size_t first_row_end = own_begin + n;       // end of first owned row
    const size_t last_row_begin = own_end - n;        // start of last owned row

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* dyn = world.elements_dynamic.data();
        ElementDynamic* out = world.elements_dynamic_swap.data();

        // Halo exchange of energies: ghost rows <- neighbors' boundary rows
        MPI_Request reqs[4];
        MPI_Irecv(&dyn[0], 1, world.energy_row_type, world.rank_up, 0, comm, &reqs[0]);
        MPI_Irecv(&dyn[own_end], 1, world.energy_row_type, world.rank_down, 1, comm, &reqs[1]);
        MPI_Isend(&dyn[own_begin], 1, world.energy_row_type, world.rank_up, 1, comm, &reqs[2]);
        MPI_Isend(&dyn[last_row_begin], 1, world.energy_row_type, world.rank_down, 0, comm, &reqs[3]);

        // Interior rows don't depend on ghost data: overlap with communication
        if (world.n_rows > 2) {
            updateRange(world, dyn, out, first_row_end, last_row_begin);
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Boundary rows
        updateRange(world, dyn, out, own_begin, first_row_end);
        if (world.n_rows > 1) {
            updateRange(world, dyn, out, last_row_begin, own_end);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather all owned dynamic states to rank 0 in global order
std::vector<ElementDynamic> gatherDynamic(const World& world, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int vals_per_elem = sizeof(ElementDynamic) / sizeof(val_t);
    std::vector<int> counts, displs;
    std::vector<ElementDynamic> global;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            int b, c;
            rowRange(world.n_root, size, r, b, c);
            counts[r] = c * world.n_root * vals_per_elem;
            displs[r] = b * world.n_root * vals_per_elem;
        }
        global.resize(static_cast<size_t>(world.n_root) * world.n_root);
    }
    const int my_count = world.n_rows * world.n_root * vals_per_elem;
    const val_t* send = world.n_rows > 0
        ? &world.elements_dynamic[world.ownedBegin()].current_energy : nullptr;
    MPI_Gatherv(send, my_count, MPI_DOUBLE,
                rank == 0 ? &global.data()->current_energy : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    return global;
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
// (partial hash over elements [begin, end) whose global index starts at global_offset;
// partial hashes combine with XOR)
uint64_t computeHash(const ElementDynamic* elements, size_t count, size_t global_offset) {
    uint64_t hash = 0;
    for (size_t k = 0; k < count; ++k) {
        const size_t i = global_offset + k;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &elements[k].current_energy, sizeof(e_bits));
        std::memcpy(&f_bits, &elements[k].total_flux, sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (root) {
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
    buildSquare2D(world, n_elems_root, comm);
    
    // Calculate memory usage (global mesh)
    const size_t g_elems = static_cast<size_t>(n_elems);
    const size_t static_mem = g_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = g_elems * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    
        // Run simulation
        printf("Running simulation...\n");
        fflush(stdout);
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, comm);
    
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, comm);
    
    // Compute hash for verification (XOR-combinable partial hashes)
    const uint64_t local_hash = world.n_rows > 0
        ? computeHash(&world.elements_dynamic[world.ownedBegin()],
                      static_cast<size_t>(world.n_rows) * n_elems_root,
                      static_cast<size_t>(world.row_begin) * n_elems_root)
        : 0;
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    if (root) {
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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    int ret = 0;
    if (printResults || validate) {
        const std::vector<ElementDynamic> global = gatherDynamic(world, comm);
        if (root) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(global.size());
                for (const auto& elem : global) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }
    
            // Validation
            if (validate && !validateResults(global)) {
                ret = 1;
            }
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, comm);
    }
    
    MPI_Type_free(&world.energy_row_type);
    MPI_Finalize();
    return ret;
}
