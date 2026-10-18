#include <algorithm>
#include <array>
#include <climits>
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
    int rank = 0, size = 1;
    int n = 0, nx = 0, ny = 0, x0 = 0, y0 = 0;
    int dims[2] = {1, 1};
    int neighbors[4];
    size_t halo[4];
    std::array<std::vector<double>, 4> send;
    std::array<std::vector<double>, 4> receive;
    std::vector<size_t> interior, boundary;
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
    int coords[2];
    MPI_Cart_coords(world.comm, world.rank, 2, coords);
    world.nx = world.n / world.dims[0] + (coords[0] < world.n % world.dims[0]);
    world.ny = world.n / world.dims[1] + (coords[1] < world.n % world.dims[1]);
    world.x0 = coords[0] * (world.n / world.dims[0]) + std::min(coords[0], world.n % world.dims[0]);
    world.y0 = coords[1] * (world.n / world.dims[1]) + std::min(coords[1], world.n % world.dims[1]);
    // Direction order matches the original connection accumulation order.
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    const size_t n_elems = size_t(world.nx) * world.ny;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    size_t storage = n_elems;
    for (int d = 0; d < 4; ++d) {
        world.halo[d] = storage;
        if (world.neighbors[d] != MPI_PROC_NULL) {
            const int count = d < 2 ? world.ny : world.nx;
            storage += count;
            world.send[d].resize(count);
            world.receive[d].resize(count);
        }
    }
    world.elements_dynamic.resize(storage);
    world.elements_dynamic_swap.resize(storage);
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t idx = size_t(x) * world.ny + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            const int gx = world.x0 + x, gy = world.y0 + y;
            if ((gx == 0 || gx == world.n - 1) && (gy == 0 || gy == world.n - 1))
                elem.material_idx = ((gx == 0 && gy == 0) ||
                                     (gx == world.n - 1 && gy == world.n - 1)) ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
            bool boundary = false;
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (gx + offsets[n][0] >= 0 && gx + offsets[n][0] < world.n &&
                    gy + offsets[n][1] >= 0 && gy + offsets[n][1] < world.n) {
                    size_t neighbor_idx;
                    if (nx >= 0 && nx < world.nx && ny >= 0 && ny < world.ny) {
                        neighbor_idx = size_t(nx) * world.ny + ny;
                    } else {
                        neighbor_idx = world.halo[n] + (n < 2 ? y : x);
                        boundary = true;
                    }
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
            (boundary ? world.boundary : world.interior).push_back(idx);
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
    auto update = [&](const std::vector<size_t>& indices) {
        for (size_t i : indices) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
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
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            const int length = static_cast<int>(world.send[d].size());
            for (int j = 0; j < length; ++j) {
                size_t i;
                if (d < 2) i = size_t(d == 0 ? world.nx - 1 : 0) * world.ny + j;
                else i = size_t(j) * world.ny + (d == 2 ? world.ny - 1 : 0);
                world.send[d][j] = world.elements_dynamic[i].current_energy;
            }
            MPI_Irecv(world.receive[d].data(), length, MPI_DOUBLE, world.neighbors[d],
                      d ^ 1, world.comm, &requests[count++]);
            MPI_Isend(world.send[d].data(), length, MPI_DOUBLE, world.neighbors[d],
                      d, world.comm, &requests[count++]);
        }
        update(world.interior);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d)
            for (size_t j = 0; j < world.receive[d].size(); ++j)
                world.elements_dynamic[world.halo[d] + j].current_energy = world.receive[d][j];
        update(world.boundary);
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    val_t sums[2] = {energy_sum, flux_sum}, global_sums[2];
    MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    energy_sum = global_sums[0];
    flux_sum = global_sums[1];
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    if (world.rank != 0)
        return std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
               std::isfinite(energy_min) && std::isfinite(energy_max);
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
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const size_t global_i = (world.x0 + i / world.ny) * size_t(world.n) + world.y0 + i % world.ny;
        // Simple hash combining energy and flux values
        uint64_t energy, flux;
        std::memcpy(&energy, &world.elements_dynamic[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &world.elements_dynamic[i].total_flux, sizeof(flux));
        hash ^= (energy + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + global_i) * 0xbf58476d1ce4e5b9ULL;
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

// Only explicit result output assembles the global array. Unpack each Cartesian
// block into the original row-major order for the order-sensitive output hash.
void printDistributedResults(const World& world) {
    std::vector<double> local(world.elements_static.size());
    for (size_t i = 0; i < local.size(); ++i)
        local[i] = world.elements_dynamic[i].current_energy;
    if (world.rank != 0) {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 4, world.comm);
        return;
    }
    std::vector<double> energyData(size_t(world.n) * world.n);
    for (int rank = 0; rank < world.size; ++rank) {
        int coords[2];
        MPI_Cart_coords(world.comm, rank, 2, coords);
        const int nx = world.n / world.dims[0] + (coords[0] < world.n % world.dims[0]);
        const int ny = world.n / world.dims[1] + (coords[1] < world.n % world.dims[1]);
        const int x0 = coords[0] * (world.n / world.dims[0]) + std::min(coords[0], world.n % world.dims[0]);
        const int y0 = coords[1] * (world.n / world.dims[1]) + std::min(coords[1], world.n % world.dims[1]);
        if (rank != 0) {
            local.resize(size_t(nx) * ny);
            MPI_Recv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank, 4,
                     world.comm, MPI_STATUS_IGNORE);
        }
        for (int x = 0; x < nx; ++x)
            std::copy_n(local.data() + size_t(x) * ny, ny,
                        energyData.data() + size_t(x0 + x) * world.n + y0);
    }
    print_results(energyData, "ElementEnergy");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // Every normal exit, including help and argument errors, finalizes MPI.
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize;
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
    
    if (n_elems_root <= 0 || int64_t(n_elems_root) * n_elems_root > INT_MAX || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive with N*N <= INT_MAX; iterations must be nonnegative.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    World world;
    // Maximize usable ranks, then minimize the perimeter of each 2D block.
    // Excess ranks on tiny meshes participate in startup/finalization only.
    int active = 0;
    double best_perimeter = std::numeric_limits<double>::max();
    for (int px = 1; px <= std::min(size, n_elems_root); ++px) {
        const int py = std::min(size / px, n_elems_root);
        const int used = px * py;
        const double perimeter = double(n_elems_root) / px + double(n_elems_root) / py;
        if (used > active || (used == active && perimeter < best_perimeter)) {
            active = used;
            best_perimeter = perimeter;
            world.dims[0] = px;
            world.dims[1] = py;
        }
    }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (rank >= active) return 0;
    int periods[2] = {0, 0};
    MPI_Cart_create(workers, 2, world.dims, periods, 0, &world.comm);
    MPI_Comm_free(&workers);
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Comm_size(world.comm, &world.size);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d partition)\n", active, world.dims[0], world.dims[1]);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    uint64_t memory[2] = {
        world.elements_static.size() * sizeof(ElementStatic),
        world.elements_dynamic.size() * sizeof(ElementDynamic) * 2
    };
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : memory, memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    const size_t static_mem = memory[0];
    const size_t dynamic_mem = memory[1];
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
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
    double duration;
    MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        const double duration_ms = duration * 1000.0;
        printf("Computation time: %.3f ms\n", duration_ms);

        const double time_per_iter = duration_ms / std::max(n_iters, 1);
        const double giga_elems_per_sec = duration > 0.0 ?
            (double(n_iters) * n_elems) / duration / 1e9 : 0.0;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
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
    return valid ? 0 : 1;
}
