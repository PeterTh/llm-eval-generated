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

// World state: each rank holds a contiguous block of grid rows plus halo rows.
// Dynamic arrays are indexed locally with halo rows included; static
// connectivity is stored for owned elements only and references local indices.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;      // owned elements
    std::vector<ElementDynamic> elements_dynamic;    // halo_lo + owned + halo_hi
    std::vector<ElementDynamic> elements_dynamic_swap;

    // Decomposition info
    int n_root = 0;        // global grid dimension
    int row_begin = 0;     // first owned global row
    int row_end = 0;       // one past last owned global row
    int halo_lo = 0;       // 1 if a lower halo row exists
    int halo_hi = 0;       // 1 if an upper halo row exists
    int rank = 0;
    int nprocs = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of grid rows across ranks
static void rowRange(int n_root, int nprocs, int rank, int& begin, int& end) {
    const int base = n_root / nprocs;
    const int rem = n_root % nprocs;
    begin = rank * base + std::min(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}

// Build the local portion of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.n_root = n_elems_root;
    rowRange(n_elems_root, world.nprocs, world.rank, world.row_begin, world.row_end);

    const int n = n_elems_root;
    const int rows_owned = world.row_end - world.row_begin;
    world.halo_lo = (rows_owned > 0 && world.row_begin > 0) ? 1 : 0;
    world.halo_hi = (rows_owned > 0 && world.row_end < n) ? 1 : 0;

    const size_t n_owned = static_cast<size_t>(rows_owned) * n;
    const size_t n_local = static_cast<size_t>(rows_owned + world.halo_lo + world.halo_hi) * n;

    // Allocate elements
    world.elements_static.resize(n_owned);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_owned; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each owned element connects to its neighbors in the
    // 2D grid, using local indices into the halo-extended dynamic arrays.
    for (int x = world.row_begin; x < world.row_end; ++x) {
        for (int y = 0; y < n; ++y) {
            const int local_row = x - world.row_begin;
            const int idx = local_row * n + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int m = 0; m < 4; ++m) {
                const int nx = x + offsets[m][0];
                const int ny = y + offsets[m][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    const int neighbor_idx =
                        (nx - world.row_begin + world.halo_lo) * n + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto setCorner = [&](int x, int y, idx_t mat) {
        if (x >= world.row_begin && x < world.row_end) {
            world.elements_static[(x - world.row_begin) * n + y].material_idx = mat;
        }
    };
    setCorner(0, 0, INFLOW_MAT_ID);
    setCorner(0, last, OUTFLOW_MAT_ID);
    setCorner(last, 0, OUTFLOW_MAT_ID);
    setCorner(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update owned elements in the local row range [row_lo, row_hi)
static void updateRows(World& world, int row_lo, int row_hi) {
    const int n = world.n_root;
    const size_t dyn_offset = static_cast<size_t>(world.halo_lo) * n;

    for (int r = row_lo; r < row_hi; ++r) {
        const size_t begin = static_cast<size_t>(r) * n;
        for (size_t i = begin; i < begin + n; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[dyn_offset + i];
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
            ElementDynamic& elem_write = world.elements_dynamic_swap[dyn_offset + i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
    }
}

// Run simulation for n_iters iterations with halo exchange each step,
// overlapping boundary communication with interior computation.
void runSimulation(World& world, const int n_iters) {
    const int n = world.n_root;
    const int rows_owned = world.row_end - world.row_begin;

    std::vector<val_t> send_lo(world.halo_lo ? n : 0), recv_lo(world.halo_lo ? n : 0);
    std::vector<val_t> send_hi(world.halo_hi ? n : 0), recv_hi(world.halo_hi ? n : 0);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int n_reqs = 0;

        // Exchange boundary row energies with neighboring ranks
        if (world.halo_lo) {
            const size_t first_owned = static_cast<size_t>(world.halo_lo) * n;
            for (int y = 0; y < n; ++y) {
                send_lo[y] = world.elements_dynamic[first_owned + y].current_energy;
            }
            MPI_Irecv(recv_lo.data(), n, MPI_DOUBLE, world.rank - 1, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(send_lo.data(), n, MPI_DOUBLE, world.rank - 1, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }
        if (world.halo_hi) {
            const size_t last_owned =
                static_cast<size_t>(world.halo_lo + rows_owned - 1) * n;
            for (int y = 0; y < n; ++y) {
                send_hi[y] = world.elements_dynamic[last_owned + y].current_energy;
            }
            MPI_Irecv(recv_hi.data(), n, MPI_DOUBLE, world.rank + 1, 1,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(send_hi.data(), n, MPI_DOUBLE, world.rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        // Compute interior rows while boundary data is in flight
        const int interior_lo = world.halo_lo ? 1 : 0;
        const int interior_hi = world.halo_hi ? rows_owned - 1 : rows_owned;
        if (interior_hi > interior_lo) {
            updateRows(world, interior_lo, interior_hi);
        }

        if (n_reqs > 0) {
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
        }

        // Unpack halos and compute the remaining boundary rows
        if (world.halo_lo) {
            for (int y = 0; y < n; ++y) {
                world.elements_dynamic[y].current_energy = recv_lo[y];
            }
        }
        if (world.halo_hi) {
            const size_t halo_hi_off =
                static_cast<size_t>(world.halo_lo + rows_owned) * n;
            for (int y = 0; y < n; ++y) {
                world.elements_dynamic[halo_hi_off + y].current_energy = recv_hi[y];
            }
        }
        updateRows(world, 0, std::min(interior_lo, rows_owned));
        updateRows(world, std::max(interior_hi, std::min(interior_lo, rows_owned)),
                   rows_owned);

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results (sequential semantics on gathered global data)
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

// Compute a simple hash of the results for verification.
// The per-element contributions are XOR-combined, so each rank hashes its
// owned elements with their global indices and the partial hashes are
// combined with a bitwise-XOR reduction — bit-identical to the serial hash.
uint64_t computeLocalHash(const World& world) {
    const int n = world.n_root;
    const size_t dyn_offset = static_cast<size_t>(world.halo_lo) * n;
    const size_t n_owned = world.elements_static.size();
    const size_t global_offset = static_cast<size_t>(world.row_begin) * n;

    uint64_t hash = 0;
    for (size_t i = 0; i < n_owned; ++i) {
        const ElementDynamic& elem = world.elements_dynamic[dyn_offset + i];
        const size_t gi = global_offset + i;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool is_root = (rank == 0);

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
            if (is_root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh (each rank builds its local block of rows)
    if (is_root) printf("Building unstructured mesh...\n");
    World world;
    world.rank = rank;
    world.nprocs = nprocs;
    buildSquare2D(world, n_elems_root);

    // Calculate memory usage (aggregated over all ranks)
    size_t local_mem[2] = {
        world.elements_static.size() * sizeof(ElementStatic),
        world.elements_dynamic.size() * sizeof(ElementDynamic) * 2
    };
    size_t global_mem[2] = {0, 0};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (is_root) {
        const size_t static_mem = global_mem[0];
        const size_t dynamic_mem = global_mem[1];
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (is_root) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long duration_ms = static_cast<long>((end - start) * 1000.0);

    // Compute hash for verification (XOR-combine partial hashes)
    const uint64_t local_hash = computeLocalHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (is_root) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather full dynamic state to rank 0 when needed for output/validation
    if (printResults || validate) {
        const int n = world.n_root;
        const size_t dyn_offset = static_cast<size_t>(world.halo_lo) * n;
        const int local_count = static_cast<int>(world.elements_static.size()) * 2;

        std::vector<int> counts(is_root ? nprocs : 0);
        std::vector<int> displs(is_root ? nprocs : 0);
        if (is_root) {
            for (int r = 0; r < nprocs; ++r) {
                int rb, re;
                rowRange(n_elems_root, nprocs, r, rb, re);
                counts[r] = (re - rb) * n * 2;
                displs[r] = rb * n * 2;
            }
        }

        std::vector<ElementDynamic> global_dynamic(is_root ? n_elems : 0);
        static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));
        MPI_Gatherv(reinterpret_cast<const double*>(world.elements_dynamic.data() + dyn_offset),
                    local_count, MPI_DOUBLE,
                    reinterpret_cast<double*>(global_dynamic.data()),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (is_root) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(global_dynamic.size());
                for (const auto& elem : global_dynamic) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }

            // Validation
            if (validate && !validateResults(global_dynamic)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
