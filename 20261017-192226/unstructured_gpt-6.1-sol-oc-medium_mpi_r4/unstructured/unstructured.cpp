#include <algorithm>
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

// World state
struct World {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int dims[2] = {1, 1};
    int coords[2] = {0, 0};
    int neighbors[4];
    int nx, ny, x0, y0, n;
    size_t pitch;
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
    world.n = n_elems_root;
    world.x0 = static_cast<int>(static_cast<int64_t>(world.coords[0]) * world.n / world.dims[0]);
    world.y0 = static_cast<int>(static_cast<int64_t>(world.coords[1]) * world.n / world.dims[1]);
    world.nx = static_cast<int>(static_cast<int64_t>(world.coords[0] + 1) * world.n / world.dims[0]) - world.x0;
    world.ny = static_cast<int>(static_cast<int64_t>(world.coords[1] + 1) * world.n / world.dims[1]) - world.y0;
    world.pitch = static_cast<size_t>(world.ny) + 2;
    const size_t n_elems = static_cast<size_t>(world.nx) * world.ny;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize((static_cast<size_t>(world.nx) + 2) * world.pitch);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    
    // resize value-initializes connectivity and both energy/flux buffers to zero.
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t idx = static_cast<size_t>(x) * world.ny + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            const int gx = world.x0 + x, gy = world.y0 + y;
            // Preserve the original corner-assignment order, including N=1.
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == world.n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == world.n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == world.n - 1 && gy == world.n - 1) elem.material_idx = INFLOW_MAT_ID;
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (gx + offsets[n][0] >= 0 && gx + offsets[n][0] < world.n &&
                    gy + offsets[n][1] >= 0 && gy + offsets[n][1] < world.n) {
                    const size_t neighbor_idx = static_cast<size_t>(nx + 1) * world.pitch + ny + 1;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
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

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    // Datatypes select only energy; accumulated flux never needs communication.
    MPI_Datatype cell, row, column;
    MPI_Type_create_resized(MPI_DOUBLE, 0, sizeof(ElementDynamic), &cell);
    MPI_Type_commit(&cell);
    MPI_Type_contiguous(world.ny, cell, &row);
    MPI_Type_commit(&row);
    MPI_Type_vector(world.nx, 1, static_cast<int>(world.pitch), cell, &column);
    MPI_Type_commit(&column);
    const size_t p = world.pitch;
    const size_t send_offsets[4] = {p + 1, static_cast<size_t>(world.nx) * p + 1,
                                   p + 1, p + world.ny};
    const size_t recv_offsets[4] = {1, (static_cast<size_t>(world.nx) + 1) * p + 1,
                                   p, p + world.ny + 1};
    // Persistent requests amortize setup and bind to both ping-pong buffers.
    MPI_Request requests[2][8];
    for (int b = 0; b < 2; ++b) {
        auto& buffer = b == 0 ? world.elements_dynamic : world.elements_dynamic_swap;
        for (int d = 0; d < 4; ++d) {
            MPI_Datatype type = d < 2 ? row : column;
            MPI_Recv_init(&buffer[recv_offsets[d]].current_energy, 1, type,
                          world.neighbors[d], d ^ 1, world.comm, &requests[b][d]);
            MPI_Send_init(&buffer[send_offsets[d]].current_energy, 1, type,
                          world.neighbors[d], d, world.comm, &requests[b][d + 4]);
        }
    }
    auto update = [&](int x, int y) {
        const size_t i = static_cast<size_t>(x) * p + y;
        const ElementStatic& elem_static = world.elements_static[static_cast<size_t>(x - 1) * world.ny + y - 1];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];

        // Preserve the original connection order and floating-point arithmetic.
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
        MPI_Startall(8, requests[iter & 1]);
        for (int x = 2; x < world.nx; ++x) {
            for (int y = 2; y < world.ny; ++y) update(x, y);
        }
        MPI_Waitall(8, requests[iter & 1], MPI_STATUSES_IGNORE);
        for (int y = 1; y <= world.ny; ++y) {
            update(1, y);
            if (world.nx > 1) update(world.nx, y);
        }
        for (int x = 2; x < world.nx; ++x) {
            update(x, 1);
            if (world.ny > 1) update(x, world.ny);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    for (auto& set : requests) for (auto& request : set) MPI_Request_free(&request);
    MPI_Type_free(&column);
    MPI_Type_free(&row);
    MPI_Type_free(&cell);
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (int x = 1; x <= world.nx; ++x) {
        for (int y = 1; y <= world.ny; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x) * world.pitch + y];
            energy_sum += elem.current_energy;
            flux_sum += elem.total_flux;
            energy_max = std::max(elem.current_energy, energy_max);
            energy_min = std::min(elem.current_energy, energy_min);
        }
    }
    val_t sums[2] = {energy_sum, flux_sum}, global_sums[2];
    MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    energy_sum = global_sums[0];
    flux_sum = global_sums[1];
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    if (world.rank != 0) return std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
                                std::isfinite(energy_max) && std::isfinite(energy_min);
    
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
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x + 1) * world.pitch + y + 1];
            const uint64_t i = static_cast<uint64_t>(world.x0 + x) * world.n + world.y0 + y;
            // Simple hash combining energy and flux values
            uint64_t e, f;
            std::memcpy(&e, &elem.current_energy, sizeof(e));
            std::memcpy(&f, &elem.total_flux, sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return world.rank == 0 ? global_hash : 0;
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
    struct FinalizeMPI {
        ~FinalizeMPI() { MPI_Finalize(); }
    } finalize_mpi;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() - 2 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        return 1;
    }
    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
    World world;
    // Maximize usable ranks, then minimize the perimeter of each square-grid tile.
    // For tiny meshes surplus ranks stay outside the simulation communicator.
    int active = 0;
    for (int x = 1; x <= std::min(size, n_elems_root); ++x) {
        const int y = std::min(size / x, n_elems_root);
        const int count = x * y;
        if (count > active || (count == active && x + y < world.dims[0] + world.dims[1])) {
            active = count;
            world.dims[0] = x;
            world.dims[1] = y;
        }
    }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (rank >= active) return 0;
    int periodic[2] = {0, 0};
    MPI_Cart_create(workers, 2, world.dims, periodic, 0, &world.comm);
    MPI_Comm_free(&workers);
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Cart_coords(world.comm, world.rank, 2, world.coords);
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[0], &world.neighbors[1]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[2], &world.neighbors[3]);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
        printf("MPI processes: %d (%d x %d blocks)\n", active, world.dims[0], world.dims[1]);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    uint64_t memory[2] = {world.elements_static.size() * sizeof(ElementStatic),
                          world.elements_dynamic.size() * sizeof(ElementDynamic) * 2};
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : memory, memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    if (rank == 0) {
        const uint64_t static_mem = memory[0], dynamic_mem = memory[1];
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
    
    double duration_ms = (MPI_Wtime() - start) * 1000.0;
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        // Report global throughput using the slowest rank's elapsed time.
        const double time_per_iter = duration_ms / std::max(n_iters, 1);
        const double giga_elems_per_sec = (static_cast<double>(n_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
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
        std::vector<double> energyData;
        if (rank == 0) energyData.resize(n_elems);
        std::vector<double> row(world.ny);
        // Gather only when requested, placing tiles in original global row order.
        // Row-sized messages avoid MPI's int count limit for large global meshes.
        for (int r = 0; r < active; ++r) {
            int coords[2];
            MPI_Cart_coords(world.comm, r, 2, coords);
            const int x0 = static_cast<int>(static_cast<int64_t>(coords[0]) * world.n / world.dims[0]);
            const int x1 = static_cast<int>(static_cast<int64_t>(coords[0] + 1) * world.n / world.dims[0]);
            const int y0 = static_cast<int>(static_cast<int64_t>(coords[1]) * world.n / world.dims[1]);
            const int y1 = static_cast<int>(static_cast<int64_t>(coords[1] + 1) * world.n / world.dims[1]);
            for (int x = x0; x < x1; ++x) {
                if (rank == r) {
                    for (int y = 0; y < world.ny; ++y)
                        row[y] = world.elements_dynamic[static_cast<size_t>(x - x0 + 1) * world.pitch + y + 1].current_energy;
                    if (r != 0) MPI_Send(row.data(), world.ny, MPI_DOUBLE, 0, 10, world.comm);
                    else std::copy(row.begin(), row.end(), energyData.begin() + static_cast<size_t>(x) * world.n + y0);
                }
                if (rank == 0 && r != 0)
                    MPI_Recv(energyData.data() + static_cast<size_t>(x) * world.n + y0,
                             y1 - y0, MPI_DOUBLE, r, 10, world.comm, MPI_STATUS_IGNORE);
            }
        }
        if (rank == 0) print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    const bool valid = !validate || validateResults(world);
    MPI_Comm_free(&world.comm);
    return valid ? 0 : 1;
}
