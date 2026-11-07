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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local numbering)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// MPI domain decomposition: the global grid of n_root rows (x) is split into
// contiguous blocks of rows. Each rank owns rows [row_begin, row_end).
// Local dynamic arrays hold one ghost row on each side:
//   local row 0            -> ghost (global row row_begin - 1)
//   local rows 1..n_rows   -> owned rows
//   local row n_rows + 1   -> ghost (global row row_end)
// Static connectivity is stored only for owned elements and references
// local dynamic indices, so the simulation kernel stays fully unstructured.
struct Decomposition {
    int rank = 0;
    int size = 1;
    int n_root = 0;
    int row_begin = 0;
    int row_end = 0;
    int n_rows = 0;       // owned rows
    int prev_rank = MPI_PROC_NULL;  // owner of global row row_begin - 1
    int next_rank = MPI_PROC_NULL;  // owner of global row row_end
};

// World state
struct World {
    Decomposition decomp;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;        // owned elements only
    std::vector<ElementDynamic> elements_dynamic;      // owned + ghost rows
    std::vector<ElementDynamic> elements_dynamic_swap; // owned + ghost rows
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Row range owned by a given rank (balanced block distribution)
static inline void rowRangeForRank(const int n_root, const int size, const int rank,
                                   int& begin, int& end) {
    const int base = n_root / size;
    const int rem = n_root % size;
    begin = rank * base + std::min(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}

// Rank owning a given global row
static inline int ownerOfRow(const int n_root, const int size, const int row) {
    const int base = n_root / size;
    const int rem = n_root % size;
    const int big = rem * (base + 1);  // rows handled by ranks with an extra row
    if (row < big) return row / (base + 1);
    return rem + (row - big) / base;
}

static void setupDecomposition(Decomposition& d, const int n_root) {
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    d.n_root = n_root;
    rowRangeForRank(n_root, d.size, d.rank, d.row_begin, d.row_end);
    d.n_rows = d.row_end - d.row_begin;
    d.prev_rank = MPI_PROC_NULL;
    d.next_rank = MPI_PROC_NULL;
    if (d.n_rows > 0) {
        if (d.row_begin > 0) d.prev_rank = ownerOfRow(n_root, d.size, d.row_begin - 1);
        if (d.row_end < n_root) d.next_rank = ownerOfRow(n_root, d.size, d.row_end);
    }
}

// Build a 2D square grid as an unstructured mesh (local partition only)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const Decomposition& d = world.decomp;
    const size_t n_local = static_cast<size_t>(d.n_rows) * n_elems_root;
    const size_t n_local_with_ghost = static_cast<size_t>(d.n_rows + 2) * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local_with_ghost);
    world.elements_dynamic_swap.resize(n_local_with_ghost);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (size_t i = 0; i < n_local_with_ghost; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = d.row_begin; x < d.row_end; ++x) {
        const int lx = x - d.row_begin;  // local owned row index (0-based)
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(lx) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Neighbor lives in local dynamic array (ghost layout shifts rows by +1)
                    const int lnx = nx - d.row_begin + 1;
                    const idx_t neighbor_idx = static_cast<idx_t>(lnx) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto setMaterial = [&](int gx, int gy, idx_t mat) {
        if (gx >= d.row_begin && gx < d.row_end) {
            const size_t idx = static_cast<size_t>(gx - d.row_begin) * n_elems_root + gy;
            world.elements_static[idx].material_idx = mat;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update owned elements in local row range [lrow_begin, lrow_end) (0-based owned rows)
static inline void updateRows(World& world, const int lrow_begin, const int lrow_end) {
    const int n_root = world.decomp.n_root;
    const ElementStatic* __restrict es = world.elements_static.data();
    const ElementDynamic* __restrict ed = world.elements_dynamic.data();
    ElementDynamic* __restrict ew = world.elements_dynamic_swap.data();
    const Material* __restrict mats = world.materials.data();

    const size_t i_begin = static_cast<size_t>(lrow_begin) * n_root;
    const size_t i_end = static_cast<size_t>(lrow_end) * n_root;
    const size_t ghost_offset = static_cast<size_t>(n_root);  // owned i -> dynamic i + n_root

    for (size_t i = i_begin; i < i_end; ++i) {
        const ElementStatic& elem_static = es[i];
        const ElementDynamic& elem_dyn = ed[i + ghost_offset];
        const Material& mat = mats[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = ed[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = ew[i + ghost_offset];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const Decomposition& d = world.decomp;
    const int n_root = d.n_root;
    const int n_rows = d.n_rows;
    // Each row of ElementDynamic is contiguous: 2 doubles per element
    const int row_count = 2 * n_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (n_rows == 0) continue;  // nothing owned by this rank

        ElementDynamic* dyn = world.elements_dynamic.data();
        MPI_Request reqs[4];
        int n_req = 0;

        // Post halo exchange: receive neighbor boundary rows into ghost rows,
        // send our first and last owned rows.
        if (d.prev_rank != MPI_PROC_NULL) {
            MPI_Irecv(dyn, row_count, MPI_DOUBLE, d.prev_rank, 0, MPI_COMM_WORLD, &reqs[n_req++]);
            MPI_Isend(dyn + static_cast<size_t>(1) * n_root, row_count, MPI_DOUBLE,
                      d.prev_rank, 1, MPI_COMM_WORLD, &reqs[n_req++]);
        }
        if (d.next_rank != MPI_PROC_NULL) {
            MPI_Irecv(dyn + static_cast<size_t>(n_rows + 1) * n_root, row_count, MPI_DOUBLE,
                      d.next_rank, 1, MPI_COMM_WORLD, &reqs[n_req++]);
            MPI_Isend(dyn + static_cast<size_t>(n_rows) * n_root, row_count, MPI_DOUBLE,
                      d.next_rank, 0, MPI_COMM_WORLD, &reqs[n_req++]);
        }

        // Overlap: update interior rows (not touching ghost rows) while halos are in flight
        if (n_rows > 2) {
            updateRows(world, 1, n_rows - 1);
        }

        MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

        // Boundary rows that depend on ghost data
        if (n_rows > 2) {
            updateRows(world, 0, 1);
            updateRows(world, n_rows - 1, n_rows);
        } else {
            updateRows(world, 0, n_rows);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather owned ElementDynamic data from all ranks to rank 0 in global order
static std::vector<ElementDynamic> gatherDynamic(const World& world) {
    const Decomposition& d = world.decomp;
    const int n_root = d.n_root;
    std::vector<int> counts(d.size), displs(d.size);
    if (d.rank == 0) {
        for (int r = 0; r < d.size; ++r) {
            int b, e;
            rowRangeForRank(n_root, d.size, r, b, e);
            counts[r] = (e - b) * n_root * 2;
            displs[r] = b * n_root * 2;
        }
    }
    std::vector<ElementDynamic> global;
    if (d.rank == 0) global.resize(static_cast<size_t>(n_root) * n_root);
    const int send_count = d.n_rows * n_root * 2;
    const double* send_ptr = reinterpret_cast<const double*>(
        world.elements_dynamic.data() + static_cast<size_t>(n_root));
    MPI_Gatherv(send_ptr, send_count, MPI_DOUBLE,
                d.rank == 0 ? reinterpret_cast<double*>(global.data()) : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return global;
}

// Validate simulation results (called on rank 0 with the gathered global state)
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

// Compute a simple hash of the results for verification.
// The hash is an XOR of per-element terms keyed by the global index, so each
// rank hashes its owned elements and the partial hashes are combined with BXOR.
uint64_t computeHash(const World& world) {
    const Decomposition& d = world.decomp;
    const size_t n_root = static_cast<size_t>(d.n_root);
    const ElementDynamic* owned = world.elements_dynamic.data() + n_root;
    const size_t n_owned = static_cast<size_t>(d.n_rows) * n_root;
    const size_t global_offset = static_cast<size_t>(d.row_begin) * n_root;

    uint64_t local_hash = 0;
    for (size_t k = 0; k < n_owned; ++k) {
        const size_t i = global_offset + k;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &owned[k].current_energy, sizeof(uint64_t));
        std::memcpy(&f_bits, &owned[k].total_flux, sizeof(uint64_t));
        local_hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash = 0;
    MPI_Allreduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool is_root = (rank == 0);

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
            if (is_root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (is_root) printf("Building unstructured mesh...\n");
    World world;
    setupDecomposition(world.decomp, n_elems_root);
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (global totals, as in the original program)
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (is_root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (is_root) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const auto local_duration_ms = duration_ms;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (is_root) printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    // Compute hash for verification (collective)
    const uint64_t hash = computeHash(world);

    if (is_root) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    int exit_code = 0;
    if (printResults || validate) {
        // Gather the global state to rank 0 only when needed for output/validation
        std::vector<ElementDynamic> global = gatherDynamic(world);

        // Print results for external validation
        if (printResults && is_root) {
            std::vector<double> energyData;
            energyData.reserve(global.size());
            for (const auto& elem : global) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (validate) {
            if (is_root) {
                bool valid = validateResults(global);
                if (!valid) exit_code = 1;
            }
            MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return exit_code;
}
