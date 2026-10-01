#include <algorithm>
// Use the MPI C interface; deprecated C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, n = 0, nx = 0, ny = 0, x0 = 0, y0 = 0, pitch = 0;
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
// Every rank constructs only its owned connectivity. Neighbor indices refer to
// local state, including a one-cell halo; no global mesh is replicated.
void buildSquare2D(World& world, const int n, const int dims[2]) {
    world.n = n;
    MPI_Comm_rank(world.comm, &world.rank);
    int coords[2];
    MPI_Cart_coords(world.comm, world.rank, 2, coords);
    world.x0 = static_cast<int>(static_cast<int64_t>(n) * coords[0] / dims[0]);
    world.y0 = static_cast<int>(static_cast<int64_t>(n) * coords[1] / dims[1]);
    world.nx = static_cast<int>(static_cast<int64_t>(n) * (coords[0] + 1) / dims[0]) - world.x0;
    world.ny = static_cast<int>(static_cast<int64_t>(n) * (coords[1] + 1) / dims[1]) - world.y0;
    world.pitch = world.ny + 2;
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(world.nx) * world.ny);
    world.elements_dynamic.resize(static_cast<size_t>(world.nx + 2) * world.pitch);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = 1; x <= world.nx; ++x) {
        for (int y = 1; y <= world.ny; ++y) {
            const int gx = world.x0 + x - 1, gy = world.y0 + y - 1;
            auto& elem = world.elements_static[static_cast<size_t>(x - 1) * world.ny + y - 1];
            elem.material_idx = DEFAULT_MAT_ID;
            // Preserve the original assignment order, including the 1x1 case.
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = INFLOW_MAT_ID;
            for (const auto& offset : offsets) {
                if (gx + offset[0] >= 0 && gx + offset[0] < n &&
                    gy + offset[1] >= 0 && gy + offset[1] < n) {
                    elem.connected_idx[elem.num_connections] =
                        static_cast<size_t>(x + offset[0]) * world.pitch + y + offset[1];
                    elem.connected_flux[elem.num_connections++] = 1.0;
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
    // Derived datatypes transfer energy only, never the cumulative local flux.
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));
    MPI_Datatype row, column;
    MPI_Type_vector(world.ny, 1, 2, MPI_DOUBLE, &row);
    MPI_Type_vector(world.nx, 1, 2 * world.pitch, MPI_DOUBLE, &column);
    MPI_Type_commit(&row);
    MPI_Type_commit(&column);
    const size_t pitch = world.pitch;
    const size_t send_offsets[4] = {world.nx * pitch + 1, pitch + 1,
                                  pitch + world.ny, pitch + 1};
    const size_t recv_offsets[4] = {(world.nx + 1) * pitch + 1, 1,
                                  pitch + world.ny + 1, pitch};
    auto update = [&](int x, int y) {
        const size_t i = static_cast<size_t>(x) * pitch + y;
        const auto& elem = world.elements_static[static_cast<size_t>(x - 1) * world.ny + y - 1];
        const auto& state = world.elements_dynamic[i];
        const auto& mat = world.materials[elem.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem.num_connections; ++j)
            total_flux += computeFlux(mat, state, elem.connected_flux[j],
                                      world.elements_dynamic[elem.connected_idx[j]]);
        world.elements_dynamic_swap[i] = {state.current_energy + total_flux,
                                          state.total_flux + std::abs(total_flux)};
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        for (int d = 0; d < 4; ++d)
            MPI_Irecv(&world.elements_dynamic[recv_offsets[d]].current_energy, 1,
                      d < 2 ? row : column, world.neighbors[d], d ^ 1,
                      world.comm, &requests[d]);
        for (int d = 0; d < 4; ++d)
            MPI_Isend(&world.elements_dynamic[send_offsets[d]].current_energy, 1,
                      d < 2 ? row : column, world.neighbors[d], d,
                      world.comm, &requests[d + 4]);
        for (int x = 2; x < world.nx; ++x)
            for (int y = 2; y < world.ny; ++y) update(x, y);
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        for (int y = 1; y <= world.ny; ++y) {
            update(1, y);
            if (world.nx > 1) update(world.nx, y);
        }
        for (int x = 2; x < world.nx; ++x) {
            update(x, 1);
            if (world.ny > 1) update(x, world.ny);
        }
        world.elements_dynamic.swap(world.elements_dynamic_swap);
    }
    MPI_Type_free(&row);
    MPI_Type_free(&column);
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
    uint64_t hash = 0, global_hash = 0;
    for (int x = 1; x <= world.nx; ++x) {
        for (int y = 1; y <= world.ny; ++y) {
            const auto& elem = world.elements_dynamic[static_cast<size_t>(x) * world.pitch + y];
            const uint64_t i = static_cast<uint64_t>(world.x0 + x - 1) * world.n + world.y0 + y - 1;
            uint64_t e, f;
            std::memcpy(&e, &elem.current_energy, sizeof(e));
            std::memcpy(&f, &elem.total_flux, sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global_hash;
}

// Only the optional serial-order result formatter needs a global energy array.
void printDistributedResults(const World& world, const int dims[2]) {
    std::vector<double> local;
    local.reserve(world.elements_static.size());
    for (int x = 1; x <= world.nx; ++x)
        for (int y = 1; y <= world.ny; ++y)
            local.push_back(world.elements_dynamic[static_cast<size_t>(x) * world.pitch + y].current_energy);
    if (world.rank != 0) {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 0, world.comm);
        return;
    }
    std::vector<double> energy(static_cast<size_t>(world.n) * world.n);
    for (int x = 0; x < world.nx; ++x)
        std::copy_n(local.data() + static_cast<size_t>(x) * world.ny, world.ny,
                    energy.data() + static_cast<size_t>(world.x0 + x) * world.n + world.y0);
    int size;
    MPI_Comm_size(world.comm, &size);
    for (int rank = 1; rank < size; ++rank) {
        int coords[2], sizes[2] = {world.n, world.n}, starts[2], subsizes[2];
        MPI_Cart_coords(world.comm, rank, 2, coords);
        for (int d = 0; d < 2; ++d) {
            starts[d] = static_cast<int>(static_cast<int64_t>(world.n) * coords[d] / dims[d]);
            subsizes[d] = static_cast<int>(static_cast<int64_t>(world.n) * (coords[d] + 1) / dims[d]) - starts[d];
        }
        MPI_Datatype block;
        MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(energy.data(), 1, block, rank, 0, world.comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(energy, "ElementEnergy");
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
    int n_elems_root = 512, n_iters = 10;
    bool validate = false, printResults = false;
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
    if (n_elems_root <= 0 || static_cast<int64_t>(n_elems_root) * n_elems_root > INT_MAX || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive with N*N <= INT_MAX; iterations must be nonnegative.\n");
        MPI_Finalize();
        return 1;
    }
    // Maximize occupied ranks, then minimize the rectangular block perimeter.
    // Excess ranks remain idle when the grid cannot accommodate them.
    int dims[2] = {1, 1}, active = 1;
    for (int x = 1; x <= std::min(n_elems_root, size); ++x) {
        const int y = std::min(n_elems_root, size / x);
        if (x * y > active || (x * y == active && x + y < dims[0] + dims[1])) {
            dims[0] = x; dims[1] = y; active = x * y;
        }
    }
    MPI_Comm members;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &members);
    int valid = 1;
    if (rank < active) {
        World world;
        int periods[2] = {0, 0};
        MPI_Cart_create(members, 2, dims, periods, 0, &world.comm);
        MPI_Comm_free(&members);
        const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
            printf("Iterations: %d\nValidation: %s\n", n_iters, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (%d x %d)\n\nBuilding unstructured mesh...\n", active, dims[0], dims[1]);
        }
        buildSquare2D(world, n_elems_root, dims);
        uint64_t memory[2] = {world.elements_static.size() * sizeof(ElementStatic),
                             world.elements_dynamic.size() * sizeof(ElementDynamic) * 2};
        uint64_t total[2];
        MPI_Reduce(memory, total, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (total[0] + total[1]) / (1024.0 * 1024.0), total[0] / (1024.0 * 1024.0), total[1] / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(world.comm);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters);
        const double elapsed = MPI_Wtime() - start;
        double duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
        const uint64_t hash = computeHash(world);
        if (rank == 0) {
            const double rate = duration > 0 ? static_cast<double>(n_iters) * n_elems / duration / 1e9 : 0;
            printf("Computation time: %.3f ms\n", duration * 1000);
            printf("Performance:\n  Time per iteration: %.4f ms\n", duration * 1000 / std::max(n_iters, 1));
            printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        if (printResults) printDistributedResults(world, dims);
        if (validate) valid = validateResults(world);
        MPI_Comm_free(&world.comm);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
