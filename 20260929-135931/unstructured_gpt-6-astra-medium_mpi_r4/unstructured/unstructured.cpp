#include <algorithm>
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
    int rank = 0, size = 1, n = 0;
    int nx = 0, ny = 0, x0 = 0, y0 = 0;
    size_t stride = 0;
    int neighbors[4];
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
// Each rank owns a rectangular block, with one layer of ghost energies.
// Connectivity retains the original neighbor order and uses local indices.
void buildSquare2D(World& world, const int n, MPI_Comm comm) {
    world.comm = comm;
    world.n = n;
    MPI_Comm_rank(comm, &world.rank);
    MPI_Comm_size(comm, &world.size);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    world.x0 = static_cast<int>(static_cast<int64_t>(n) * coords[0] / dims[0]);
    world.y0 = static_cast<int>(static_cast<int64_t>(n) * coords[1] / dims[1]);
    world.nx = static_cast<int>(static_cast<int64_t>(n) * (coords[0] + 1) / dims[0]) - world.x0;
    world.ny = static_cast<int>(static_cast<int64_t>(n) * (coords[1] + 1) / dims[1]) - world.y0;
    world.stride = static_cast<size_t>(world.ny) + 2;
    MPI_Cart_shift(comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(world.nx) * world.ny);
    world.elements_dynamic.resize((static_cast<size_t>(world.nx) + 2) * world.stride);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            auto& elem = world.elements_static[static_cast<size_t>(x) * world.ny + y];
            const int gx = world.x0 + x, gy = world.y0 + y;
            elem.material_idx = DEFAULT_MAT_ID;
            // Assignment order also preserves the original N=1 behavior.
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = INFLOW_MAT_ID;
            for (const auto& offset : offsets) {
                const int xx = gx + offset[0], yy = gy + offset[1];
                if (xx >= 0 && xx < n && yy >= 0 && yy < n) {
                    const idx_t j = elem.num_connections++;
                    elem.connected_idx[j] = static_cast<size_t>(x + 1 + offset[0]) * world.stride
                                            + y + 1 + offset[1];
                    elem.connected_flux[j] = 1.0;
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
    const int nx = world.nx, ny = world.ny;
    const size_t stride = world.stride;
    std::vector<double> send[4], recv[4];
    for (int d = 0; d < 4; ++d) {
        send[d].resize(d < 2 ? ny : nx);
        recv[d].resize(send[d].size());
    }
    auto update = [&](int x, int y) {
        const auto& elem_static = world.elements_static[static_cast<size_t>(x) * ny + y];
        const size_t i = static_cast<size_t>(x + 1) * stride + y + 1;
        const auto& elem_dyn = world.elements_dynamic[i];
        const auto& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                                     world.elements_dynamic[elem_static.connected_idx[j]]);
        }
        world.elements_dynamic_swap[i] = {elem_dyn.current_energy + total_flux,
                                         elem_dyn.total_flux + std::abs(total_flux)};
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            MPI_Irecv(recv[d].data(), static_cast<int>(recv[d].size()), MPI_DOUBLE,
                      world.neighbors[d], d ^ 1, world.comm, &requests[count++]);
            for (size_t k = 0; k < send[d].size(); ++k) {
                const size_t i = d == 0 ? static_cast<size_t>(nx) * stride + k + 1 :
                                 d == 1 ? stride + k + 1 :
                                 (k + 1) * stride + (d == 2 ? ny : 1);
                send[d][k] = world.elements_dynamic[i].current_energy;
            }
            MPI_Isend(send[d].data(), static_cast<int>(send[d].size()), MPI_DOUBLE,
                      world.neighbors[d], d, world.comm, &requests[count++]);
        }
        // All reads here are local, so communication may proceed concurrently.
        for (int x = 1; x < nx - 1; ++x)
            for (int y = 1; y < ny - 1; ++y) update(x, y);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            for (size_t k = 0; k < recv[d].size(); ++k) {
                const size_t i = d == 0 ? (static_cast<size_t>(nx) + 1) * stride + k + 1 :
                                 d == 1 ? k + 1 :
                                 (k + 1) * stride + (d == 2 ? ny + 1 : 0);
                world.elements_dynamic[i].current_energy = recv[d][k];
            }
        }
        for (int y = 0; y < ny; ++y) update(0, y);
        if (nx > 1) for (int y = 0; y < ny; ++y) update(nx - 1, y);
        for (int x = 1; x < nx - 1; ++x) {
            update(x, 0);
            if (ny > 1) update(x, ny - 1);
        }
        world.elements_dynamic.swap(world.elements_dynamic_swap);
    }
}

// Only optional serial-format diagnostics need the complete state. Derived
// datatypes place each block directly into its original global row order.
std::vector<ElementDynamic> gatherResults(const World& world) {
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));
    MPI_Datatype element, block;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element);
    MPI_Type_commit(&element);
    std::vector<ElementDynamic> result;
    if (world.rank == 0) {
        result.resize(static_cast<size_t>(world.n) * world.n);
        int dims[2], periods[2], coords[2];
        MPI_Cart_get(world.comm, 2, dims, periods, coords);
        for (int rank = 0; rank < world.size; ++rank) {
            MPI_Cart_coords(world.comm, rank, 2, coords);
            const int x0 = static_cast<int>(static_cast<int64_t>(world.n) * coords[0] / dims[0]);
            const int y0 = static_cast<int>(static_cast<int64_t>(world.n) * coords[1] / dims[1]);
            const int nx = static_cast<int>(static_cast<int64_t>(world.n) * (coords[0] + 1) / dims[0]) - x0;
            const int ny = static_cast<int>(static_cast<int64_t>(world.n) * (coords[1] + 1) / dims[1]) - y0;
            auto* dest = result.data() + static_cast<size_t>(x0) * world.n + y0;
            if (rank == 0) {
                for (int x = 0; x < nx; ++x)
                    std::copy_n(world.elements_dynamic.data() + (x + 1) * world.stride + 1,
                                ny, dest + static_cast<size_t>(x) * world.n);
            } else {
                MPI_Type_vector(nx, ny, world.n, element, &block);
                MPI_Type_commit(&block);
                MPI_Recv(dest, 1, block, rank, 10, world.comm, MPI_STATUS_IGNORE);
                MPI_Type_free(&block);
            }
        }
    } else {
        MPI_Type_vector(world.nx, world.ny, static_cast<int>(world.stride), element, &block);
        MPI_Type_commit(&block);
        MPI_Send(world.elements_dynamic.data() + world.stride + 1, 1, block, 0, 10, world.comm);
        MPI_Type_free(&block);
    }
    MPI_Type_free(&element);
    return result;
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
    uint64_t local_hash = 0, global_hash = 0;
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const auto& elem = world.elements_dynamic[(x + 1) * world.stride + y + 1];
            const uint64_t i = static_cast<uint64_t>(world.x0 + x) * world.n + world.y0 + y;
            uint64_t energy, flux;
            std::memcpy(&energy, &elem.current_energy, sizeof(energy));
            std::memcpy(&flux, &elem.total_flux, sizeof(flux));
            local_hash ^= (energy + i) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (flux + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global_hash;
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
    if (n_elems_root <= 0 || n_elems_root > INT_MAX - 2 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        MPI_Finalize();
        return 1;
    }
    // Maximize occupied ranks, then minimize the boundary length. For tiny
    // grids or unsuitable prime process counts, spare ranks remain idle.
    int dims[2] = {1, 1};
    for (int x = 1; x <= std::min(n_elems_root, size); ++x) {
        const int y = std::min(n_elems_root, size / x);
        if (x * y > dims[0] * dims[1] ||
            (x * y == dims[0] * dims[1] && x + y < dims[0] + dims[1])) {
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm active, cart;
    MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED, rank, &active);
    int status = 0;
    if (active != MPI_COMM_NULL) {
        const int periods[2] = {0, 0};
        MPI_Cart_create(active, 2, dims, periods, 0, &cart);
        const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
            printf("Iterations: %d\nValidation: %s\n", n_iters, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (%d x %d blocks)\n\n", dims[0] * dims[1], dims[0], dims[1]);
            printf("Building unstructured mesh...\n");
        }
        World world;
        buildSquare2D(world, n_elems_root, cart);
        uint64_t local_mem[2] = {world.elements_static.size() * sizeof(ElementStatic),
                                world.elements_dynamic.size() * sizeof(ElementDynamic) * 2};
        uint64_t global_mem[2];
        MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, cart);
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (global_mem[0] + global_mem[1]) / (1024.0 * 1024.0),
                   global_mem[0] / (1024.0 * 1024.0), global_mem[1] / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters);
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        const uint64_t hash = computeHash(world);
        if (rank == 0) {
            const double rate = seconds > 0 ? static_cast<double>(n_iters) * n_elems / seconds / 1e9 : 0;
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", n_iters ? seconds * 1000.0 / n_iters : 0);
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        if (printResults || validate) {
            const auto elements = gatherResults(world);
            if (rank == 0) {
                if (printResults) {
                    std::vector<double> energyData;
                    energyData.reserve(elements.size());
                    for (const auto& elem : elements) energyData.push_back(elem.current_energy);
                    print_results(energyData, "ElementEnergy");
                }
                if (validate && !validateResults(elements)) status = 1;
            }
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
