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
constexpr int MAX_CONNECTIONS = 4;

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
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int grid_size = 0;
    int dims[2] = {1, 1};
    int begin[2] = {0, 0};
    int extent[2] = {0, 0};
    // Directions: +x, -x, +y, -y (same order as the original flux sum).
    int neighbors[4];
    size_t ghost[4];
    std::vector<size_t> boundary;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    world.grid_size = n_elems_root;
    int coords[2], periods[2];
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Cart_get(world.comm, 2, world.dims, periods, coords);
    for (int d = 0; d < 2; ++d) {
        world.begin[d] = coords[d] * (n_elems_root / world.dims[d]) +
                         std::min(coords[d], n_elems_root % world.dims[d]);
        world.extent[d] = n_elems_root / world.dims[d] +
                          (coords[d] < n_elems_root % world.dims[d]);
    }
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    const int nx = world.extent[0], ny = world.extent[1];
    const size_t n_elems = size_t(nx) * ny;
    world.ghost[0] = n_elems;
    world.ghost[1] = world.ghost[0] + ny;
    world.ghost[2] = world.ghost[1] + ny;
    world.ghost[3] = world.ghost[2] + nx;
    const size_t storage = world.ghost[3] + nx;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(storage);
    world.elements_dynamic_swap.resize(storage);
    
    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            const size_t idx = size_t(x) * ny + y;
            ElementStatic& elem = world.elements_static[idx];
            const int gx = world.begin[0] + x, gy = world.begin[1] + y;
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int lx = x + offsets[n][0];
                const int ly = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (gx + offsets[n][0] >= 0 && gx + offsets[n][0] < n_elems_root &&
                    gy + offsets[n][1] >= 0 && gy + offsets[n][1] < n_elems_root) {
                    const size_t neighbor_idx =
                        lx < 0 || lx >= nx ? world.ghost[n] + y :
                        ly < 0 || ly >= ny ? world.ghost[n] + x : size_t(lx) * ny + ly;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
            if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1)
                world.boundary.push_back(idx);

            // Preserve the original assignment order, including the 1x1 case.
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
    const int nx = world.extent[0], ny = world.extent[1];
    const int counts[4] = {ny, ny, nx, nx};
    std::vector<double> send[4], receive[4];
    MPI_Request requests[8];
    int n_requests = 0;
    for (int d = 0; d < 4; ++d) {
        if (world.neighbors[d] == MPI_PROC_NULL) continue;
        send[d].resize(counts[d]);
        receive[d].resize(counts[d]);
        // The buffers and neighbors are fixed: reuse the communication schedule.
        MPI_Recv_init(receive[d].data(), counts[d], MPI_DOUBLE, world.neighbors[d],
                      d ^ 1, world.comm, &requests[n_requests++]);
        MPI_Send_init(send[d].data(), counts[d], MPI_DOUBLE, world.neighbors[d],
                      d, world.comm, &requests[n_requests++]);
    }
    // Keep the per-element arithmetic and connection order identical to serial.
    auto update = [&](size_t i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const auto& neighbor = world.elements_dynamic[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor);
        }
        auto& output = world.elements_dynamic_swap[i];
        output.current_energy = elem_dyn.current_energy + total_flux;
        output.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            for (int k = 0; k < counts[d]; ++k) {
                const size_t i = d == 0 ? size_t(nx - 1) * ny + k :
                                 d == 1 ? size_t(k) :
                                 d == 2 ? size_t(k) * ny + ny - 1 : size_t(k) * ny;
                send[d][k] = world.elements_dynamic[i].current_energy;
            }
        }
        MPI_Startall(n_requests, requests);
        // Interior work does not depend on incoming halos.
        for (int x = 1; x < nx - 1; ++x)
            for (int y = 1; y < ny - 1; ++y)
                update(size_t(x) * ny + y);
        MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            for (int k = 0; k < counts[d]; ++k)
                world.elements_dynamic[world.ghost[d] + k].current_energy = receive[d][k];
        }
        for (size_t i : world.boundary) update(i);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    for (int r = 0; r < n_requests; ++r) MPI_Request_free(&requests[r]);
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
    val_t sums[2] = {energy_sum, flux_sum};
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    energy_sum = sums[0];
    flux_sum = sums[1];
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
        const uint64_t global_i = (world.begin[0] + i / world.extent[1]) *
                                  uint64_t(world.grid_size) + world.begin[1] + i % world.extent[1];
        // Simple hash combining energy and flux values
        uint64_t energy, flux;
        std::memcpy(&energy, &world.elements_dynamic[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &world.elements_dynamic[i].total_flux, sizeof(flux));
        hash ^= (energy + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, world.comm);
    return hash;
}

// Gather only on explicit request, restoring the original global row-major order.
void printDistributedResults(const World& world) {
    std::vector<double> local(world.elements_static.size());
    for (size_t i = 0; i < local.size(); ++i)
        local[i] = world.elements_dynamic[i].current_energy;
    if (world.rank != 0) {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 0, world.comm);
        return;
    }
    std::vector<double> global(size_t(world.grid_size) * world.grid_size);
    int size;
    MPI_Comm_size(world.comm, &size);
    for (int rank = 0; rank < size; ++rank) {
        int coords[2], begin[2], extent[2];
        MPI_Cart_coords(world.comm, rank, 2, coords);
        for (int d = 0; d < 2; ++d) {
            begin[d] = coords[d] * (world.grid_size / world.dims[d]) +
                       std::min(coords[d], world.grid_size % world.dims[d]);
            extent[d] = world.grid_size / world.dims[d] +
                        (coords[d] < world.grid_size % world.dims[d]);
        }
        double* dest = global.data() + size_t(begin[0]) * world.grid_size + begin[1];
        if (rank == 0) {
            for (int x = 0; x < extent[0]; ++x)
                std::copy_n(local.data() + size_t(x) * extent[1], extent[1],
                            dest + size_t(x) * world.grid_size);
        } else {
            MPI_Datatype block;
            MPI_Type_vector(extent[0], extent[1], world.grid_size, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            MPI_Recv(dest, 1, block, rank, 0, world.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
    }
    print_results(global, "ElementEnergy");
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
    
    if (n_elems_root <= 0 || int64_t(n_elems_root) * n_elems_root >
        std::numeric_limits<int>::max() || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive with N*N <= INT_MAX; iterations must be nonnegative.\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;

    // Choose the largest usable process grid, preferring square blocks. Excess
    // ranks (possible for tiny meshes) participate in shutdown, but own no cells.
    int dims[2] = {1, 1};
    for (int x = 1; x <= std::min(size, n_elems_root); ++x) {
        const int y = std::min(n_elems_root, size / x);
        if (x * y > dims[0] * dims[1] ||
            (x * y == dims[0] * dims[1] && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED,
                   rank, &active);
    int status = 0;
    if (active == MPI_COMM_NULL) {
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return status;
    }
    World world;
    int periods[2] = {0, 0};
    MPI_Cart_create(active, 2, dims, periods, 0, &world.comm);
    MPI_Comm_free(&active);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
        printf("MPI process grid: %d x %d (%d ranks)\n", dims[0], dims[1], dims[0] * dims[1]);
    }
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    uint64_t memory[2] = {static_mem, dynamic_mem};
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : memory, memory, 2, MPI_UINT64_T,
               MPI_SUM, 0, world.comm);
    if (rank == 0) {
        const uint64_t total_mem = memory[0] + memory[1];
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               memory[0] / (1024.0 * 1024.0),
               memory[1] / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }
    MPI_Barrier(world.comm);
    const double start = MPI_Wtime();
    
    runSimulation(world, n_iters);
    
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &elapsed, &elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, world.comm);
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        const double duration_ms = elapsed * 1000.0;

        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate performance metrics
        const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
        const double giga_elems_per_sec = elapsed > 0 ? (double(n_iters) * n_elems) / elapsed / 1e9 : 0.0;

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
    if (printResults) printDistributedResults(world);
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) status = 1;
    }
    MPI_Comm_free(&world.comm);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
