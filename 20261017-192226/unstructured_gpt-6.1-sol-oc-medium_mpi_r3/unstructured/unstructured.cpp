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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "MPI halo strides require two adjacent doubles");

// World state
struct World {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int grid_size = 0, rows = 0, cols = 0, x_begin = 0, y_begin = 0;
    size_t pitch = 0;
    int neighbors[4]; // +x, -x, +y, -y
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    world.grid_size = n_elems_root;
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(world.comm, 2, dims, periods, coords);
    world.rows = n_elems_root / dims[0] + (coords[0] < n_elems_root % dims[0]);
    world.cols = n_elems_root / dims[1] + (coords[1] < n_elems_root % dims[1]);
    world.x_begin = coords[0] * (n_elems_root / dims[0]) + std::min(coords[0], n_elems_root % dims[0]);
    world.y_begin = coords[1] * (n_elems_root / dims[1]) + std::min(coords[1], n_elems_root % dims[1]);
    world.pitch = static_cast<size_t>(world.cols) + 2;
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    const size_t n_elems = static_cast<size_t>(world.rows) * world.cols;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    // One ghost layer holds only neighboring energies; flux remains rank-local.
    world.elements_dynamic.resize((static_cast<size_t>(world.rows) + 2) * world.pitch);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const size_t idx = static_cast<size_t>(x) * world.cols + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = world.x_begin + x + offsets[n][0];
                const int ny = world.y_begin + y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx = static_cast<size_t>(x + 1 + offsets[n][0]) * world.pitch
                                              + y + 1 + offsets[n][1];
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
            // Keep the original assignment order, including the overlapping N=1 corners.
            const int gx = world.x_begin + x, gy = world.y_begin + y;
            const int last = n_elems_root - 1;
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == last) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == last && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == last && gy == last) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    // Datatypes send only current_energy, skipping the accumulated local flux.
    MPI_Datatype row_type, col_type;
    MPI_Type_vector(world.cols, 1, 2, MPI_DOUBLE, &row_type);
    MPI_Type_vector(world.rows, 1, static_cast<int>(2 * world.pitch), MPI_DOUBLE, &col_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);

    const auto update = [&](int x, int y) {
        const size_t i = static_cast<size_t>(x + 1) * world.pitch + y + 1;
        const ElementStatic& elem_static = world.elements_static[static_cast<size_t>(x) * world.cols + y];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    };
    
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        const size_t p = world.pitch;
        const size_t sends[4] = {static_cast<size_t>(world.rows) * p + 1, p + 1,
                                p + world.cols, p + 1};
        const size_t recvs[4] = {(static_cast<size_t>(world.rows) + 1) * p + 1, 1,
                                p + world.cols + 1, p};
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            MPI_Irecv(&world.elements_dynamic[recvs[d]].current_energy, 1,
                      d < 2 ? row_type : col_type, world.neighbors[d], d ^ 1,
                      world.comm, &requests[count++]);
        }
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            MPI_Isend(&world.elements_dynamic[sends[d]].current_energy, 1,
                      d < 2 ? row_type : col_type, world.neighbors[d], d,
                      world.comm, &requests[count++]);
        }
        // Interior updates need no halo values and overlap boundary communication.
        for (int x = 1; x < world.rows - 1; ++x)
            for (int y = 1; y < world.cols - 1; ++y) update(x, y);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int y = 0; y < world.cols; ++y) {
            update(0, y);
            if (world.rows > 1) update(world.rows - 1, y);
        }
        for (int x = 1; x < world.rows - 1; ++x) {
            update(x, 0);
            if (world.cols > 1) update(x, world.cols - 1);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x + 1) * world.pitch + y + 1];
            energy_sum += elem.current_energy;
            flux_sum += elem.total_flux;
            energy_max = std::max(elem.current_energy, energy_max);
            energy_min = std::min(elem.current_energy, energy_min);
        }
    }

    val_t sums[2] = {energy_sum, flux_sum}, global_sums[2];
    MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    energy_sum = global_sums[0];
    flux_sum = global_sums[1];
    if (world.rank != 0) {
        return std::isfinite(energy_sum) && std::isfinite(flux_sum)
            && std::isfinite(energy_min) && std::isfinite(energy_max);
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
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x + 1) * world.pitch + y + 1];
            const uint64_t i = static_cast<uint64_t>(world.x_begin + x) * world.grid_size + world.y_begin + y;
            // Simple hash combining energy and flux values
            uint64_t energy_bits, flux_bits;
            memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
            memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
            hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global_hash;
}

// Gather only for explicitly requested external validation, restoring global row order.
void printDistributedResults(const World& world) {
    std::vector<double> local;
    local.reserve(world.elements_static.size());
    for (int x = 0; x < world.rows; ++x)
        for (int y = 0; y < world.cols; ++y)
            local.push_back(world.elements_dynamic[static_cast<size_t>(x + 1) * world.pitch + y + 1].current_energy);
    std::vector<int> counts(world.size), displs(world.size);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(world.comm, 2, dims, periods, coords);
    int total = 0;
    for (int r = 0; r < world.size; ++r) {
        MPI_Cart_coords(world.comm, r, 2, coords);
        const int rows = world.grid_size / dims[0] + (coords[0] < world.grid_size % dims[0]);
        const int cols = world.grid_size / dims[1] + (coords[1] < world.grid_size % dims[1]);
        counts[r] = rows * cols;
        displs[r] = total;
        total += counts[r];
    }
    std::vector<double> packed(world.rank == 0 ? total : 0);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, packed.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, world.comm);
    if (world.rank != 0) return;
    std::vector<double> energyData(total);
    for (int r = 0; r < world.size; ++r) {
        MPI_Cart_coords(world.comm, r, 2, coords);
        const int rows = world.grid_size / dims[0] + (coords[0] < world.grid_size % dims[0]);
        const int cols = world.grid_size / dims[1] + (coords[1] < world.grid_size % dims[1]);
        const int xb = coords[0] * (world.grid_size / dims[0]) + std::min(coords[0], world.grid_size % dims[0]);
        const int yb = coords[1] * (world.grid_size / dims[1]) + std::min(coords[1], world.grid_size % dims[1]);
        for (int x = 0; x < rows; ++x)
            std::copy_n(packed.data() + displs[r] + static_cast<size_t>(x) * cols, cols,
                        energyData.data() + static_cast<size_t>(xb + x) * world.grid_size + yb);
    }
    print_results(energyData, "ElementEnergy");
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
    int rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
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

    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() / n_elems_root) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and NxN must fit in an int.\n");
        MPI_Finalize();
        return 1;
    }
    
    const int n_elems = n_elems_root * n_elems_root;

    // Use the largest possible nonempty rectangular process grid, then minimize
    // its perimeter. Extra ranks (only possible for very small meshes) stay idle.
    int dims[2] = {1, 1};
    for (int a = 1; a <= std::min(n_elems_root, process_count); ++a) {
        const int b = std::min(n_elems_root, process_count / a);
        if (a * b > dims[0] * dims[1] ||
            (a * b == dims[0] * dims[1] && a + b < dims[0] + dims[1])) {
            dims[0] = a;
            dims[1] = b;
        }
    }
    const int active_count = dims[0] * dims[1];
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active_count ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    World world;
    const int periods[2] = {0, 0};
    MPI_Cart_create(active, 2, dims, periods, 0, &world.comm);
    MPI_Comm_free(&active);
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Comm_size(world.comm, &world.size);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d subdomains)\n", active_count, dims[0], dims[1]);
        printf("\n");

        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    uint64_t local_mem[2] = {world.elements_static.size() * sizeof(ElementStatic),
                            world.elements_dynamic.size() * sizeof(ElementDynamic) * 2};
    uint64_t global_mem[2] = {};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    if (rank == 0) {
        const uint64_t static_mem = global_mem[0], dynamic_mem = global_mem[1];
        const uint64_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }
    MPI_Barrier(world.comm);
    const double start = MPI_Wtime();
    
    runSimulation(world, n_iters);
    
    const double local_duration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    const double duration_ms = duration * 1000.0;
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / duration / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016" PRIX64 "\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        printDistributedResults(world);
    }
    
    // Validation
    const bool valid = !validate || validateResults(world);
    MPI_Comm_free(&world.comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
