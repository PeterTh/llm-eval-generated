#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
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
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, rows = 0, cols = 0, x0 = 0, y0 = 0, grid = 0;
    int neighbors[4];
    std::vector<size_t> interior, boundary;
    std::vector<double> send[4], recv[4];
    size_t ghost[4];
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
// Each rank builds only its own connectivity. Remote indices address compact
// ghost regions; only current_energy is communicated, never accumulated flux.
void buildSquare2D(World& world, const int n_elems_root) {
    world.grid = n_elems_root;
    int dims[2], periods[2], coords[2];
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Cart_get(world.comm, 2, dims, periods, coords);
    world.x0 = static_cast<int>(static_cast<int64_t>(n_elems_root) * coords[0] / dims[0]);
    world.y0 = static_cast<int>(static_cast<int64_t>(n_elems_root) * coords[1] / dims[1]);
    world.rows = static_cast<int>(static_cast<int64_t>(n_elems_root) * (coords[0] + 1) / dims[0]) - world.x0;
    world.cols = static_cast<int>(static_cast<int64_t>(n_elems_root) * (coords[1] + 1) / dims[1]) - world.y0;
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t count = static_cast<size_t>(world.rows) * world.cols;
    size_t allocated = count;
    for (int d = 0; d < 4; ++d) {
        world.ghost[d] = allocated;
        const int length = d < 2 ? world.cols : world.rows;
        if (world.neighbors[d] != MPI_PROC_NULL) {
            world.send[d].resize(length);
            world.recv[d].resize(length);
            allocated += length;
        }
    }
    world.elements_static.resize(count);
    world.elements_dynamic.resize(allocated);
    world.elements_dynamic_swap.resize(allocated);
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const size_t i = static_cast<size_t>(x) * world.cols + y;
            auto& elem = world.elements_static[i];
            const int gx = world.x0 + x, gy = world.y0 + y;
            elem.material_idx = DEFAULT_MAT_ID;
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == n_elems_root - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n_elems_root - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n_elems_root - 1 && gy == n_elems_root - 1) elem.material_idx = INFLOW_MAT_ID;
            const int offsets[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
            bool remote = false;
            for (int d = 0; d < 4; ++d) {
                const int nx = x + offsets[d][0], ny = y + offsets[d][1];
                if (gx + offsets[d][0] < 0 || gx + offsets[d][0] >= n_elems_root ||
                    gy + offsets[d][1] < 0 || gy + offsets[d][1] >= n_elems_root) continue;
                size_t neighbor;
                if (nx >= 0 && nx < world.rows && ny >= 0 && ny < world.cols) {
                    neighbor = static_cast<size_t>(nx) * world.cols + ny;
                } else {
                    neighbor = world.ghost[d] + (d < 2 ? y : x);
                    remote = true;
                }
                elem.connected_idx[elem.num_connections] = neighbor;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            (remote ? world.boundary : world.interior).push_back(i);
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
            const auto& elem_static = world.elements_static[i];
            const auto& elem_dyn = world.elements_dynamic[i];
            const auto& mat = world.materials[elem_static.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                    world.elements_dynamic[elem_static.connected_idx[j]]);
            }
            world.elements_dynamic_swap[i] = {
                elem_dyn.current_energy + total_flux,
                elem_dyn.total_flux + std::abs(total_flux)};
        }
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            const int length = static_cast<int>(world.send[d].size());
            MPI_Irecv(world.recv[d].data(), length, MPI_DOUBLE, world.neighbors[d],
                      d ^ 1, world.comm, &requests[count++]);
            for (int j = 0; j < length; ++j) {
                size_t i;
                if (d < 2) i = static_cast<size_t>(d == 0 ? world.rows - 1 : 0) * world.cols + j;
                else i = static_cast<size_t>(j) * world.cols + (d == 2 ? world.cols - 1 : 0);
                world.send[d][j] = world.elements_dynamic[i].current_energy;
            }
            MPI_Isend(world.send[d].data(), length, MPI_DOUBLE, world.neighbors[d],
                      d, world.comm, &requests[count++]);
        }
        update(world.interior);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d) {
            for (size_t j = 0; j < world.recv[d].size(); ++j)
                world.elements_dynamic[world.ghost[d] + j].current_energy = world.recv[d][j];
        }
        update(world.boundary);
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
    
    val_t sums[2] = {energy_sum, flux_sum}, global[2];
    MPI_Allreduce(sums, global, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    energy_sum = global[0];
    flux_sum = global[1];
    if (world.rank != 0) return std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
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
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x) * world.cols + y];
            const uint64_t i = static_cast<uint64_t>(world.x0 + x) * world.grid + world.y0 + y;
            uint64_t e, f;
            std::memcpy(&e, &elem.current_energy, sizeof(e));
            std::memcpy(&f, &elem.total_flux, sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global = 0;
    MPI_Reduce(&hash, &global, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global;
}

// Collect only on request, directly into global row-major order. Row messages
// avoid MPI's int limit on the total grid size and require no root staging copy.
void printGlobalResults(const World& world) {
    std::vector<double> energyData;
    if (world.rank == 0) energyData.resize(static_cast<size_t>(world.grid) * world.grid);
    std::vector<double> row(world.cols);
    int size;
    MPI_Comm_size(world.comm, &size);
    for (int rank = 0; rank < size; ++rank) {
        int bounds[4];
        if (world.rank == rank) {
            bounds[0] = world.x0; bounds[1] = world.y0;
            bounds[2] = world.rows; bounds[3] = world.cols;
            if (rank != 0) MPI_Send(bounds, 4, MPI_INT, 0, 10, world.comm);
            for (int x = 0; x < world.rows; ++x) {
                for (int y = 0; y < world.cols; ++y)
                    row[y] = world.elements_dynamic[static_cast<size_t>(x) * world.cols + y].current_energy;
                if (rank != 0) MPI_Send(row.data(), world.cols, MPI_DOUBLE, 0, 11, world.comm);
                else std::copy(row.begin(), row.end(), energyData.begin() +
                               static_cast<size_t>(world.x0 + x) * world.grid + world.y0);
            }
        } else if (world.rank == 0) {
            MPI_Recv(bounds, 4, MPI_INT, rank, 10, world.comm, MPI_STATUS_IGNORE);
            for (int x = 0; x < bounds[2]; ++x)
                MPI_Recv(energyData.data() + static_cast<size_t>(bounds[0] + x) * world.grid + bounds[1],
                         bounds[3], MPI_DOUBLE, rank, 11, world.comm, MPI_STATUS_IGNORE);
        }
    }
    if (world.rank == 0) print_results(energyData, "ElementEnergy");
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
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto printf = [rank](const char* format, auto... args) {
        if (rank == 0) {
            if constexpr (sizeof...(args) == 0) std::fputs(format, stdout);
            else std::printf(format, args...);
        }
    };
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
            printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (n_elems_root <= 0 || n_iters < 0) {
        printf("Grid size must be positive and iteration count nonnegative.\n");
        return 1;
    }
    const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
    // Maximize occupied ranks, then minimize the perimeter of each block.
    int dims[2] = {1, 1};
    for (int x = 1; x <= std::min(n_elems_root, size); ++x) {
        int y = std::min(n_elems_root, size / x);
        if (x * y > dims[0] * dims[1] ||
            (x * y == dims[0] * dims[1] && x + y < dims[0] + dims[1])) {
            dims[0] = x; dims[1] = y;
        }
    }
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) return 0;
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(active, 2, dims, periods, 0, &cart);
    MPI_Comm_free(&active);
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root, static_cast<unsigned long long>(n_elems));
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    World world;
    world.comm = cart;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic) +
        (world.interior.capacity() + world.boundary.capacity()) * sizeof(size_t);
    size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    for (int d = 0; d < 4; ++d)
        dynamic_mem += (world.send[d].size() + world.recv[d].size()) * sizeof(double);
    uint64_t local_mem[2] = {static_mem, dynamic_mem}, global_mem[2] = {};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, cart);
    const uint64_t total_mem = global_mem[0] + global_mem[1];
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           global_mem[0] / (1024.0 * 1024.0),
           global_mem[1] / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    const double duration_ms = duration * 1000.0;
    printf("Computation time: %.3f ms\n", duration_ms);

    // Calculate performance metrics
    const double time_per_iter = n_iters ? duration_ms / n_iters : 0.0;
    const double giga_elems_per_sec = duration > 0.0 ?
        (static_cast<double>(n_iters) * n_elems) / duration / 1e9 : 0.0;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    printf("  Result hash: %016" PRIX64 "\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) printGlobalResults(world);

    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Comm_free(&cart);
            return 1;
        }
    }
    MPI_Comm_free(&cart);
    return 0;
}
