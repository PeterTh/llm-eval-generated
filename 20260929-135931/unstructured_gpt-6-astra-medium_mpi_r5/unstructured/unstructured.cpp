#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <climits>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
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
    int neighbors[4]; // +x, -x, +y, -y
    size_t ghost[4];
    std::vector<size_t> boundary;
    std::vector<double> send[4], receive[4];
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
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    // Use as many nonempty rectangular blocks as possible, favoring squares.
    int dims[2] = {1, 1};
    for (int x = 1; x <= std::min(processes, world.n); ++x) {
        int y = std::min(processes / x, world.n);
        if (x * y > dims[0] * dims[1] ||
            (x * y == dims[0] * dims[1] &&
             std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            dims[0] = x;
            dims[1] = y;
        }
    }
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD,
                   global_rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED,
                   global_rank, &active);
    if (active == MPI_COMM_NULL) return;
    int periodic[2] = {0, 0};
    MPI_Cart_create(active, 2, dims, periodic, 0, &world.comm);
    MPI_Comm_free(&active);
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Comm_size(world.comm, &world.size);
    int coords[2];
    MPI_Cart_coords(world.comm, world.rank, 2, coords);
    world.x0 = int(int64_t(world.n) * coords[0] / dims[0]);
    world.y0 = int(int64_t(world.n) * coords[1] / dims[1]);
    world.nx = int(int64_t(world.n) * (coords[0] + 1) / dims[0]) - world.x0;
    world.ny = int(int64_t(world.n) * (coords[1] + 1) / dims[1]) - world.y0;
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t owned = size_t(world.nx) * world.ny;
    size_t storage = owned;
    for (int d = 0; d < 4; ++d) {
        world.ghost[d] = storage;
        const int length = d < 2 ? world.ny : world.nx;
        storage += length;
        world.send[d].resize(length);
        world.receive[d].resize(length);
    }
    world.elements_static.resize(owned);
    world.elements_dynamic.resize(storage);
    world.elements_dynamic_swap.resize(storage);
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t i = size_t(x) * world.ny + y;
            auto& elem = world.elements_static[i];
            int gx = world.x0 + x, gy = world.y0 + y;
            elem.material_idx = DEFAULT_MAT_ID;
            // Assignment order also preserves the original one-element case.
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == world.n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == world.n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == world.n - 1 && gy == world.n - 1) elem.material_idx = INFLOW_MAT_ID;
            for (int d = 0; d < 4; ++d) {
                int xx = x + offsets[d][0], yy = y + offsets[d][1];
                if (gx + offsets[d][0] < 0 || gx + offsets[d][0] >= world.n ||
                    gy + offsets[d][1] < 0 || gy + offsets[d][1] >= world.n) continue;
                size_t neighbor = (xx >= 0 && xx < world.nx && yy >= 0 && yy < world.ny)
                    ? size_t(xx) * world.ny + yy
                    : world.ghost[d] + (d < 2 ? y : x);
                elem.connected_idx[elem.num_connections] = neighbor;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (x == 0 || x == world.nx - 1 || y == 0 || y == world.ny - 1)
                world.boundary.push_back(i);
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
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            const int length = static_cast<int>(world.send[d].size());
            MPI_Irecv(world.receive[d].data(), length, MPI_DOUBLE, world.neighbors[d],
                      d ^ 1, world.comm, &requests[count++]);
            for (int k = 0; k < length; ++k) {
                size_t i;
                if (d < 2) i = size_t(d == 0 ? world.nx - 1 : 0) * world.ny + k;
                else i = size_t(k) * world.ny + (d == 2 ? world.ny - 1 : 0);
                world.send[d][k] = world.elements_dynamic[i].current_energy;
            }
            MPI_Isend(world.send[d].data(), length, MPI_DOUBLE, world.neighbors[d],
                      d, world.comm, &requests[count++]);
        }
        auto update = [&](size_t i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                    world.elements_dynamic[elem_static.connected_idx[j]]);
            }
            auto& next = world.elements_dynamic_swap[i];
            next.current_energy = elem_dyn.current_energy + total_flux;
            next.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        };
        for (int x = 1; x < world.nx - 1; ++x)
            for (int y = 1; y < world.ny - 1; ++y)
                update(size_t(x) * world.ny + y);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            for (size_t k = 0; k < world.receive[d].size(); ++k)
                world.elements_dynamic[world.ghost[d] + k].current_energy = world.receive[d][k];
        }
        for (size_t i : world.boundary) update(i);
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

    double local_sums[2] = {energy_sum, flux_sum}, sums[2];
    MPI_Allreduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
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
        uint64_t energy, flux;
        std::memcpy(&energy, &world.elements_dynamic[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &world.elements_dynamic[i].total_flux, sizeof(flux));
        const uint64_t global = (world.x0 + i / world.ny) * world.n + world.y0 + i % world.ny;
        hash ^= (energy + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + global) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t result = 0;
    MPI_Reduce(&hash, &result, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return result;
}

// Gather only when ordered external output is requested. Receive each block
// directly into its global location, avoiding a second full-sized copy.
void printDistributedResults(const World& world) {
    std::vector<double> local(world.elements_static.size());
    for (size_t i = 0; i < local.size(); ++i)
        local[i] = world.elements_dynamic[i].current_energy;
    if (world.rank != 0) {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 0, world.comm);
        return;
    }
    std::vector<double> energy(size_t(world.n) * world.n);
    for (int x = 0; x < world.nx; ++x)
        std::copy_n(local.data() + size_t(x) * world.ny, world.ny,
                    energy.data() + size_t(world.x0 + x) * world.n + world.y0);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(world.comm, 2, dims, periods, coords);
    for (int rank = 1; rank < world.size; ++rank) {
        MPI_Cart_coords(world.comm, rank, 2, coords);
        int x0 = int(int64_t(world.n) * coords[0] / dims[0]);
        int y0 = int(int64_t(world.n) * coords[1] / dims[1]);
        int nx = int(int64_t(world.n) * (coords[0] + 1) / dims[0]) - x0;
        int ny = int(int64_t(world.n) * (coords[1] + 1) / dims[1]) - y0;
        MPI_Datatype block;
        MPI_Type_vector(nx, ny, world.n, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(energy.data() + size_t(x0) * world.n + y0, 1, block,
                 rank, 0, world.comm, MPI_STATUS_IGNORE);
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
    // Finalize on every normal exit, including help and invalid arguments.
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    if (n_elems_root <= 0 || int64_t(n_elems_root) * n_elems_root > INT_MAX || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive with N*N <= INT_MAX; iterations must be nonnegative.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
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
    buildSquare2D(world, n_elems_root);
    if (world.comm == MPI_COMM_NULL) return 0;

    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    uint64_t local_mem[2] = {static_mem, dynamic_mem}, global_mem[2];
    MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    if (rank == 0) {
        const uint64_t total_mem = global_mem[0] + global_mem[1];
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               global_mem[0] / (1024.0 * 1024.0),
               global_mem[1] / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }
    MPI_Barrier(world.comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    if (rank == 0) {
        const double duration_ms = duration * 1000.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        const double time_per_iter = duration_ms / std::max(n_iters, 1);
        const double giga_elems_per_sec = duration > 0 ?
            (double(n_iters) * n_elems) / duration / 1e9 : 0;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    if (printResults) printDistributedResults(world);
    bool valid = !validate || validateResults(world);
    MPI_Comm_free(&world.comm);
    return valid ? 0 : 1;
}
