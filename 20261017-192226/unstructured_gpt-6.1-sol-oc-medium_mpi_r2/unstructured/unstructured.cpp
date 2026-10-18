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

// The generated 2D mesh has at most four neighbors. Avoid streaming unused
// connectivity slots through memory in the simulation's bandwidth-bound loop.
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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n, nx, ny, x_begin, y_begin;
    size_t pitch;
    MPI_Comm comm;
    int neighbors[4];
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm comm) {
    world.n = n_elems_root;
    world.comm = comm;
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    auto partition = [n_elems_root](int dim, int coord, int& begin, int& count) {
        count = n_elems_root / dim + (coord < n_elems_root % dim);
        begin = coord * (n_elems_root / dim) + std::min(coord, n_elems_root % dim);
    };
    partition(dims[0], coords[0], world.x_begin, world.nx);
    partition(dims[1], coords[1], world.y_begin, world.ny);
    world.pitch = static_cast<size_t>(world.ny) + 2;
    MPI_Cart_shift(comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    const size_t n_elems = static_cast<size_t>(world.nx) * world.ny;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize((static_cast<size_t>(world.nx) + 2) * world.pitch);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t idx = static_cast<size_t>(x) * world.ny + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            const int gx = world.x_begin + x;
            const int gy = world.y_begin + y;
            // Keep the original corner assignment order, including the 1x1 case.
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == world.n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == world.n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == world.n - 1 && gy == world.n - 1) elem.material_idx = INFLOW_MAT_ID;
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = gy + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx =
                        static_cast<size_t>(x + 1 + offsets[n][0]) * world.pitch +
                        y + 1 + offsets[n][1];
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
    // Transfer energy only: total_flux is accumulated locally and never read
    // by neighboring elements. Derived types avoid explicit pack/unpack passes.
    MPI_Datatype row, column;
    MPI_Type_vector(world.ny, 1, 2, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Type_vector(world.nx, 1, static_cast<int>(2 * world.pitch), MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    const size_t p = world.pitch;
    const size_t send_offsets[4] = {world.nx * p + 1, p + 1,
                                   p + world.ny, p + 1};
    const size_t recv_offsets[4] = {(world.nx + 1) * p + 1, 1,
                                   p + world.ny + 1, p};
    auto update = [&](int x, int y) {
        const size_t i = static_cast<size_t>(x) * p + y;
        const ElementStatic& elem_static = world.elements_static[
            static_cast<size_t>(x - 1) * world.ny + y - 1];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux in the original neighbor order for bitwise equivalence.
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
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            MPI_Irecv(&world.elements_dynamic[recv_offsets[d]].current_energy,
                      1, d < 2 ? row : column, world.neighbors[d], d ^ 1,
                      world.comm, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[send_offsets[d]].current_energy,
                      1, d < 2 ? row : column, world.neighbors[d], d,
                      world.comm, &requests[count++]);
        }
        // Strictly interior elements do not depend on incoming halos.
        for (int x = 2; x < world.nx; ++x)
            for (int y = 2; y < world.ny; ++y) update(x, y);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
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
    MPI_Type_free(&row);
    MPI_Type_free(&column);
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
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t i = static_cast<size_t>(world.x_begin + x) * world.n + world.y_begin + y;
            const auto& elem = world.elements_dynamic[(x + 1) * world.pitch + y + 1];
            // Simple hash combining energy and flux values
            uint64_t energy, flux;
            std::memcpy(&energy, &elem.current_energy, sizeof(energy));
            std::memcpy(&flux, &elem.total_flux, sizeof(flux));
            hash ^= (energy + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

// Gather only for requested serial-format diagnostics. Receive directly into
// global row-major order, preserving the original summation and FNV hash order.
std::vector<ElementDynamic> gatherResults(const World& world, int rank, int size) {
    std::vector<ElementDynamic> elements;
    if (rank == 0) {
        elements.resize(static_cast<size_t>(world.n) * world.n);
        for (int x = 0; x < world.nx; ++x)
            std::copy_n(world.elements_dynamic.begin() + (x + 1) * world.pitch + 1,
                        world.ny, elements.begin() +
                        static_cast<size_t>(world.x_begin + x) * world.n + world.y_begin);
        for (int src = 1; src < size; ++src) {
            int block[4];
            MPI_Recv(block, 4, MPI_INT, src, 0, world.comm, MPI_STATUS_IGNORE);
            MPI_Datatype target;
            MPI_Type_vector(block[2], 2 * block[3], 2 * world.n, MPI_DOUBLE, &target);
            MPI_Type_commit(&target);
            MPI_Recv(&elements[static_cast<size_t>(block[0]) * world.n + block[1]].current_energy,
                     1, target, src, 1, world.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&target);
        }
    } else {
        int block[4] = {world.x_begin, world.y_begin, world.nx, world.ny};
        MPI_Send(block, 4, MPI_INT, 0, 0, world.comm);
        MPI_Datatype source;
        MPI_Type_vector(world.nx, 2 * world.ny, static_cast<int>(2 * world.pitch), MPI_DOUBLE, &source);
        MPI_Type_commit(&source);
        MPI_Send(&world.elements_dynamic[world.pitch + 1].current_energy,
                 1, source, 0, 1, world.comm);
        MPI_Type_free(&source);
    }
    return elements;
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
    
    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() / 2 - 2 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and MPI-representable; iterations must be nonnegative.\n");
        MPI_Finalize();
        return 1;
    }
    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;

    // Maximize occupied ranks, then minimize the subdomain surface/volume ratio.
    // Excess ranks are excluded when the grid cannot supply a cell per rank.
    int dims[2] = {1, 1};
    int active = 1;
    for (int x = 1; x <= std::min(n_elems_root, size); ++x) {
        int y = std::min(n_elems_root, size / x);
        int used = x * y;
        if (used > active || (used == active && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            active = used;
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (workers == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    MPI_Comm comm;
    int periods[2] = {0, 0};
    MPI_Cart_create(workers, 2, dims, periods, 0, &comm);
    MPI_Comm_free(&workers);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("MPI processes: %d (%d x %d)\n", size, dims[0], dims[1]);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, comm);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    uint64_t local_mem[2] = {static_mem, dynamic_mem}, memory[2];
    MPI_Reduce(local_mem, memory, 2, MPI_UINT64_T, MPI_SUM, 0, comm);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (memory[0] + memory[1]) / (1024.0 * 1024.0),
               memory[0] / (1024.0 * 1024.0),
               memory[1] / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    
    runSimulation(world, n_iters);
    
    const double local_duration = (MPI_Wtime() - start) * 1000.0;
    double duration_ms;
    MPI_Reduce(&local_duration, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const uint64_t local_hash = computeHash(world);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        // Use the slowest rank's wall time and the actual timed iteration count.
        const double time_per_iter = duration_ms / std::max(n_iters, 1);
        const double giga_elems_per_sec = duration_ms > 0.0 ?
            (static_cast<double>(n_iters) * n_elems) / (duration_ms / 1000.0) / 1e9 : 0.0;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016llX\n", static_cast<unsigned long long>(result_hash));
        printf("\n");
    }

    std::vector<ElementDynamic> results;
    if (printResults || validate) results = gatherResults(world, rank, size);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(results.size());
        for (const auto& elem : results) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    int valid = 1;
    if (validate && rank == 0) {
        valid = validateResults(results);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
