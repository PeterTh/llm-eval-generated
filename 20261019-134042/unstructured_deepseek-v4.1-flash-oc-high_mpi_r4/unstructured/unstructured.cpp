#include <algorithm>
#include <chrono>
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
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n_elems_root = 0;
    int start_row = 0;
    int local_rows = 0;
    int rank = 0;
    int size = 1;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Split the global row range into contiguous blocks, one per rank.
static void rowRange(int n_rows, int size, int rank, int& start, int& end) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    start = rank * base + std::min(rank, rem);
    end = start + base + (rank < rem ? 1 : 0);
}

// Build the locally owned part of a 2D square grid as an unstructured mesh.
// The grid is decomposed by rows (x direction); each rank stores one halo row
// on each side of its owned block so that all neighbor accesses stay local.
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    const int n = n_elems_root;
    world.n_elems_root = n;
    world.rank = rank;
    world.size = size;

    int end_row = 0;
    rowRange(n, size, rank, world.start_row, end_row);
    world.local_rows = end_row - world.start_row;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const int local_rows = world.local_rows;
    const size_t local_elems = static_cast<size_t>(local_rows) * static_cast<size_t>(n);
    const size_t extended_elems = (static_cast<size_t>(local_rows) + 2) * static_cast<size_t>(n);

    // Allocate locally owned static data and extended dynamic data (owned rows
    // plus one halo row above and below).
    world.elements_static.resize(local_elems);
    world.elements_dynamic.assign(extended_elems, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(extended_elems, ElementDynamic{0.0, 0.0});

    if (n <= 0) {
        return;
    }

    // Build connectivity for owned elements. Neighbor indices are expressed in
    // the extended (halo-inclusive) local dynamic array.
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int x = world.start_row; x < end_row; ++x) {
        for (int y = 0; y < n; ++y) {
            const size_t local_idx = static_cast<size_t>(x - world.start_row) * n + y;
            ElementStatic& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            for (int k = 0; k < 4; ++k) {
                const int nx = x + offsets[k][0];
                const int ny = y + offsets[k][1];

                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    const int lnx = nx - world.start_row + 1;
                    const size_t neighbor_idx = static_cast<size_t>(lnx) * n + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n - 1;
    if (world.start_row == 0) {
        world.elements_static[static_cast<size_t>(0) * n + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[static_cast<size_t>(0) * n + last].material_idx = OUTFLOW_MAT_ID;
    }
    if (last >= world.start_row && last < end_row) {
        const size_t base = static_cast<size_t>(last - world.start_row) * n;
        world.elements_static[base + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[base + last].material_idx = INFLOW_MAT_ID;
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
    const int n = world.n_elems_root;
    if (n <= 0) {
        return;
    }

    const int local_rows = world.local_rows;
    if (local_rows <= 0) {
        return;
    }

    const size_t first_owned = static_cast<size_t>(n);
    const size_t n_owned = static_cast<size_t>(local_rows) * static_cast<size_t>(n);

    int up_rows_start = 0, up_rows_end = 0;
    int down_rows_start = 0, down_rows_end = 0;
    if (world.rank > 0) {
        rowRange(n, world.size, world.rank - 1, up_rows_start, up_rows_end);
    }
    if (world.rank + 1 < world.size) {
        rowRange(n, world.size, world.rank + 1, down_rows_start, down_rows_end);
    }
    const int up = (world.rank > 0 && up_rows_end > up_rows_start)
                       ? world.rank - 1 : MPI_PROC_NULL;
    const int down = (world.rank + 1 < world.size && down_rows_end > down_rows_start)
                         ? world.rank + 1 : MPI_PROC_NULL;

    const size_t below_halo = (static_cast<size_t>(local_rows) + 1) * n;
    const size_t bottom_owned = first_owned + static_cast<size_t>(local_rows - 1) * n;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all locally owned elements.
        for (size_t e = first_owned; e < first_owned + n_owned; ++e) {
            const ElementStatic& elem_static = world.elements_static[e - first_owned];
            const ElementDynamic& elem_dyn = world.elements_dynamic[e];
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
            ElementDynamic& elem_write = world.elements_dynamic_swap[e];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Exchange the newly computed boundary rows between neighboring ranks.
        ElementDynamic* sw = world.elements_dynamic_swap.data();
        double* top_send = reinterpret_cast<double*>(sw + first_owned);
        double* bottom_send = reinterpret_cast<double*>(sw + bottom_owned);
        double* above_recv = reinterpret_cast<double*>(sw);
        double* below_recv = reinterpret_cast<double*>(sw + below_halo);
        const int exchange_count = 2 * n;
        MPI_Sendrecv(top_send, exchange_count, MPI_DOUBLE, up, 0,
                     below_recv, exchange_count, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(bottom_send, exchange_count, MPI_DOUBLE, down, 1,
                     above_recv, exchange_count, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned dynamic state from every rank into global index order on rank 0.
static std::vector<ElementDynamic> gatherResults(const World& world) {
    const int n = world.n_elems_root;
    const int size = world.size;
    const int rank = world.rank;
    const int local_count = world.local_rows * n;

    std::vector<ElementDynamic> global;
    std::vector<int> counts;
    std::vector<int> displs;

    if (rank == 0) {
        global.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            int start = 0, end = 0;
            rowRange(n, size, r, start, end);
            counts[r] = (end - start) * n * static_cast<int>(sizeof(ElementDynamic));
            displs[r] = start * n * static_cast<int>(sizeof(ElementDynamic));
        }
    }

    const ElementDynamic* sendbuf =
        (n > 0) ? (world.elements_dynamic.data() + static_cast<size_t>(n)) : nullptr;
    const int sendcount = local_count * static_cast<int>(sizeof(ElementDynamic));

    MPI_Gatherv(sendbuf, sendcount, MPI_BYTE,
                (rank == 0) ? static_cast<void*>(global.data()) : nullptr,
                (rank == 0) ? counts.data() : nullptr,
                (rank == 0) ? displs.data() : nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);

    return global;
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            showHelp = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseError = true;
            break;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    const long long n_elems =
        static_cast<long long>(n_elems_root) * static_cast<long long>(n_elems_root);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    if (rank == 0) {
        // Calculate memory usage
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Collect the distributed state on rank 0 for hashing, output, validation.
    std::vector<ElementDynamic> global = gatherResults(world);

    uint64_t hash = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) /
            (static_cast<double>(duration_ms) / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        hash = computeHash(global);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global.size());
            for (const auto& elem : global) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
    }

    int rc = 0;

    // Validation
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            valid = validateResults(global) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) {
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
