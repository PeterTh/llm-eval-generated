#include <algorithm>
#include <mpi.h>
#include <cinttypes>
#include <exception>
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

// Each rank owns a rectangle; dynamic buffers include one layer of ghost cells.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype row_type, column_type;
    int rank = 0, size = 1;
    int n, rows, columns, x_start, y_start;
    size_t pitch;
    int neighbors[4];
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Choose a nearly square process grid, allowing surplus ranks to be idle on
// tiny meshes rather than constructing empty subdomains.
bool buildSquare2D(World& world, const int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int dims[2] = {1, 1};
    for (int x = 1; x <= std::min(n, size); ++x) {
        const int y = std::min(n, size / x);
        const int used = x * y;
        const int best = dims[0] * dims[1];
        if (used > best || (used == best && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED,
                   rank, &active);
    if (active == MPI_COMM_NULL) return false;
    int periods[2] = {0, 0};
    MPI_Cart_create(active, 2, dims, periods, 0, &world.comm);
    MPI_Comm_free(&active);
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Comm_size(world.comm, &world.size);
    int coords[2];
    MPI_Cart_coords(world.comm, world.rank, 2, coords);
    world.n = n;
    world.x_start = coords[0] * (n / dims[0]) + std::min(coords[0], n % dims[0]);
    world.y_start = coords[1] * (n / dims[1]) + std::min(coords[1], n % dims[1]);
    world.rows = n / dims[0] + (coords[0] < n % dims[0]);
    world.columns = n / dims[1] + (coords[1] < n % dims[1]);
    world.pitch = static_cast<size_t>(world.columns) + 2;
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(world.rows) * world.columns);
    world.elements_dynamic.resize((static_cast<size_t>(world.rows) + 2) * world.pitch);
    world.elements_dynamic_swap.resize(world.elements_dynamic.size());
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.columns; ++y) {
            const int gx = world.x_start + x, gy = world.y_start + y;
            auto& elem = world.elements_static[static_cast<size_t>(x) * world.columns + y];
            elem.material_idx = DEFAULT_MAT_ID;
            // Preserve the assignment order even for the single-element mesh.
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = INFLOW_MAT_ID;
            const size_t i = (static_cast<size_t>(x) + 1) * world.pitch + y + 1;
            const size_t indices[4] = {i + world.pitch, i - world.pitch, i + 1, i - 1};
            const bool exists[4] = {gx + 1 < n, gx > 0, gy + 1 < n, gy > 0};
            for (int j = 0; j < 4; ++j) {
                if (exists[j]) {
                    elem.connected_idx[elem.num_connections] = indices[j];
                    elem.connected_flux[elem.num_connections++] = 1.0;
                }
            }
        }
    }
    // Communicate only current_energy, never the accumulated flux.
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));
    MPI_Type_vector(world.columns, 1, 2, MPI_DOUBLE, &world.row_type);
    MPI_Type_commit(&world.row_type);
    MPI_Type_vector(world.rows, 1, static_cast<int>(2 * world.pitch), MPI_DOUBLE,
                    &world.column_type);
    MPI_Type_commit(&world.column_type);
    return true;
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void runSimulation(World& world, const int n_iters) {
    const size_t pitch = world.pitch;
    const int rows = world.rows, columns = world.columns;
    const auto update = [&](int x, int first, int last) {
        for (int y = first; y < last; ++y) {
            const auto& elem = world.elements_static[static_cast<size_t>(x) * columns + y];
            const size_t i = (static_cast<size_t>(x) + 1) * pitch + y + 1;
            const auto& current = world.elements_dynamic[i];
            const auto& mat = world.materials[elem.material_idx];
            val_t flux = mat.external_flow;
            for (idx_t j = 0; j < elem.num_connections; ++j) {
                flux += computeFlux(mat, current, elem.connected_flux[j],
                                    world.elements_dynamic[elem.connected_idx[j]]);
            }
            world.elements_dynamic_swap[i] = {
                current.current_energy + flux, current.total_flux + std::abs(flux)};
        }
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        const size_t send_indices[4] = {static_cast<size_t>(rows) * pitch + 1,
                                       pitch + 1, pitch + columns, pitch + 1};
        const size_t recv_indices[4] = {(static_cast<size_t>(rows) + 1) * pitch + 1,
                                       1, pitch + columns + 1, pitch};
        for (int d = 0; d < 4; ++d) {
            if (world.neighbors[d] == MPI_PROC_NULL) continue;
            const MPI_Datatype type = d < 2 ? world.row_type : world.column_type;
            MPI_Irecv(&world.elements_dynamic[recv_indices[d]].current_energy, 1, type,
                      world.neighbors[d], d ^ 1, world.comm, &requests[count++]);
            MPI_Isend(&world.elements_dynamic[send_indices[d]].current_energy, 1, type,
                      world.neighbors[d], d, world.comm, &requests[count++]);
        }
        for (int x = 1; x < rows - 1; ++x) update(x, 1, columns - 1);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        update(0, 0, columns);
        if (rows > 1) update(rows - 1, 0, columns);
        for (int x = 1; x < rows - 1; ++x) {
            update(x, 0, 1);
            if (columns > 1) update(x, columns - 1, columns);
        }
        world.elements_dynamic.swap(world.elements_dynamic_swap);
    }
}

// Diagnostic output requires the original global row-major ordering. Derived
// datatypes gather rectangles directly, with no packing or per-step gather.
std::vector<ElementDynamic> gatherResults(const World& world) {
    std::vector<ElementDynamic> result;
    MPI_Datatype element, local;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element);
    MPI_Type_commit(&element);
    MPI_Type_vector(world.rows, world.columns, static_cast<int>(world.pitch), element, &local);
    MPI_Type_commit(&local);
    const int metadata[4] = {world.x_start, world.y_start, world.rows, world.columns};
    if (world.rank == 0) {
        result.resize(static_cast<size_t>(world.n) * world.n);
        for (int x = 0; x < world.rows; ++x) {
            std::copy_n(world.elements_dynamic.begin() + (static_cast<size_t>(x) + 1) * world.pitch + 1,
                        world.columns, result.begin() + static_cast<size_t>(world.x_start + x) * world.n + world.y_start);
        }
        for (int rank = 1; rank < world.size; ++rank) {
            int m[4];
            MPI_Recv(m, 4, MPI_INT, rank, 10, world.comm, MPI_STATUS_IGNORE);
            const int sizes[2] = {world.n, world.n};
            const int subsizes[2] = {m[2], m[3]}, starts[2] = {m[0], m[1]};
            MPI_Datatype rectangle;
            MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, element, &rectangle);
            MPI_Type_commit(&rectangle);
            MPI_Recv(result.data(), 1, rectangle, rank, 11, world.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&rectangle);
        }
    } else {
        MPI_Send(metadata, 4, MPI_INT, 0, 10, world.comm);
        MPI_Send(&world.elements_dynamic[world.pitch + 1], 1, local, 0, 11, world.comm);
    }
    MPI_Type_free(&local);
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
    uint64_t hash = 0;
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.columns; ++y) {
            const auto& elem = world.elements_dynamic[(static_cast<size_t>(x) + 1) * world.pitch + y + 1];
            const uint64_t i = static_cast<uint64_t>(world.x_start + x) * world.n + world.y_start + y;
            uint64_t energy_bits, flux_bits;
            std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
            std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
            hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto finish = [](int status) { MPI_Finalize(); return status; };
    int n_elems_root = 512;
    int n_iters = 10;
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
            return finish(0);
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }
    if (n_elems_root <= 0 || n_elems_root > 46340 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be in [1, 46340] and iterations must be nonnegative.\n");
        return finish(1);
    }
    try {
        World world;
        if (!buildSquare2D(world, n_elems_root)) return finish(0);
        const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
        const uint64_t local_memory[2] = {
            world.elements_static.size() * sizeof(ElementStatic),
            world.elements_dynamic.size() * sizeof(ElementDynamic) * 2};
        uint64_t memory[2] = {};
        MPI_Reduce(local_memory, memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
            printf("Iterations: %d\nValidation: %s\n", n_iters, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d\n\n", world.size);
            printf("Building unstructured mesh...\n");
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (memory[0] + memory[1]) / (1024.0 * 1024.0),
                   memory[0] / (1024.0 * 1024.0), memory[1] / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(world.comm);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters);
        const double elapsed = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
        const uint64_t hash = computeHash(world);
        if (rank == 0) {
            const double duration_ms = duration * 1000.0;
            const double giga_elems_per_sec = duration > 0.0 ?
                (static_cast<double>(n_iters) * n_elems) / duration / 1e9 : 0.0;
            printf("Computation time: %.3f ms\n", duration_ms);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", n_iters > 0 ? duration_ms / n_iters : 0.0);
            printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
            printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        bool valid = true;
        if (printResults || validate) {
            const auto elements = gatherResults(world);
            if (rank == 0) {
                if (printResults) {
                    std::vector<double> energyData;
                    energyData.reserve(elements.size());
                    for (const auto& elem : elements) energyData.push_back(elem.current_energy);
                    print_results(energyData, "ElementEnergy");
                }
                if (validate) valid = validateResults(elements);
            }
            MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, world.comm);
        }
        MPI_Type_free(&world.row_type);
        MPI_Type_free(&world.column_type);
        MPI_Comm_free(&world.comm);
        return finish(valid ? 0 : 1);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
