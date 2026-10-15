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

// World state (local row block plus one halo row on each side)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n_root = 0;      // global grid edge
    int row_begin = 0;   // first owned global row
    int n_rows = 0;      // number of owned rows
    int up = MPI_PROC_NULL;    // rank owning row_begin-1
    int down = MPI_PROC_NULL;  // rank owning row_begin+n_rows
    MPI_Comm comm = MPI_COMM_NULL;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build the local block of a 2D square grid as an unstructured mesh.
// Local element index = global index - (row_begin - 1) * n, so row 0 is the halo above.
void buildSquare2D(World& world, const int n_elems_root) {
    const int n = n_elems_root;
    const size_t local_total = static_cast<size_t>(world.n_rows + 2) * n;

    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elements_static.resize(local_total);
    world.elements_dynamic.assign(local_total, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(local_total, ElementDynamic{0.0, 0.0});

    const int last = n - 1;
    const long base = static_cast<long>(world.row_begin - 1) * n;
    for (int r = 0; r < world.n_rows; ++r) {
        const int x = world.row_begin + r;
        for (int y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(r + 1) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int k = 0; k < 4; ++k) {
                const int nx = x + offsets[k][0];
                const int ny = y + offsets[k][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    elem.connected_idx[elem.num_connections] = static_cast<long>(nx) * n + ny - base;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
            if ((x == 0 && y == 0) || (x == last && y == last)) elem.material_idx = INFLOW_MAT_ID;
            else if ((x == 0 && y == last) || (x == last && y == 0)) elem.material_idx = OUTFLOW_MAT_ID;
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update local rows [r0, r1) (0-based owned-row indices)
static void updateRows(World& world, const int r0, const int r1) {
    const size_t n = world.n_root;
    const ElementStatic* __restrict es = world.elements_static.data();
    const ElementDynamic* __restrict ed = world.elements_dynamic.data();
    ElementDynamic* __restrict ew = world.elements_dynamic_swap.data();
    const Material* mats = world.materials.data();
    for (size_t i = (r0 + 1) * n; i < static_cast<size_t>(r1 + 1) * n; ++i) {
        const ElementStatic& elem_static = es[i];
        const ElementDynamic& elem_dyn = ed[i];
        const Material& mat = mats[elem_static.material_idx];

        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j],
                                      ed[elem_static.connected_idx[j]]);
        }
        ew[i].current_energy = elem_dyn.current_energy + total_flux;
        ew[i].total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int n = world.n_root;
    const int rows = world.n_rows;
    const int count = n * static_cast<int>(sizeof(ElementDynamic) / sizeof(double));

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* d = world.elements_dynamic.data();
        MPI_Request req[4];
        int nreq = 0;
        // Halo exchange of the boundary rows (tag 0: moving down, tag 1: moving up)
        if (world.up != MPI_PROC_NULL) {
            MPI_Irecv(d, count, MPI_DOUBLE, world.up, 0, world.comm, &req[nreq++]);
            MPI_Isend(d + static_cast<size_t>(n), count, MPI_DOUBLE, world.up, 1, world.comm, &req[nreq++]);
        }
        if (world.down != MPI_PROC_NULL) {
            MPI_Irecv(d + static_cast<size_t>(rows + 1) * n, count, MPI_DOUBLE, world.down, 1, world.comm, &req[nreq++]);
            MPI_Isend(d + static_cast<size_t>(rows) * n, count, MPI_DOUBLE, world.down, 0, world.comm, &req[nreq++]);
        }

        // Interior rows overlap with communication
        if (rows > 2) updateRows(world, 1, rows - 1);
        if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        updateRows(world, 0, std::min(1, rows));
        if (rows > 1) updateRows(world, rows - 1, rows);

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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

// Partial hash over the owned rows (XOR-combinable across ranks)
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const size_t n = world.n_root;
    const size_t first = static_cast<size_t>(world.row_begin) * n;
    for (size_t k = 0; k < static_cast<size_t>(world.n_rows) * n; ++k) {
        const size_t i = first + k;
        const ElementDynamic& e = world.elements_dynamic[n + k];
        uint64_t eb, fb;
        memcpy(&eb, &e.current_energy, sizeof(eb));
        memcpy(&fb, &e.total_flux, sizeof(fb));
        hash ^= (eb + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (fb + i) * 0xbf58476d1ce4e5b9ULL;
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
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    const bool root = (wrank == 0);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Row-block decomposition; ranks beyond the number of rows sit out
    const int active = std::max(1, std::min(wsize, n_elems_root));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < active ? 0 : MPI_UNDEFINED, wrank, &comm);

    World world;
    world.n_root = n_elems_root;
    if (comm != MPI_COMM_NULL) {
        world.comm = comm;
        const int base = n_elems_root / active, rem = n_elems_root % active;
        world.n_rows = base + (wrank < rem ? 1 : 0);
        world.row_begin = wrank * base + std::min(wrank, rem);
        if (wrank > 0) world.up = wrank - 1;
        if (wrank < active - 1) world.down = wrank + 1;
        buildSquare2D(world, n_elems_root);
    }

    if (root) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (comm != MPI_COMM_NULL) runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    uint64_t hash = 0;
    if (comm != MPI_COMM_NULL) hash = computeHash(world);
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    // Gather the full state on root only when needed
    std::vector<val_t> energyData, fluxData;
    if (validate || printResults) {
        const int nloc = world.n_rows * n_elems_root;
        std::vector<val_t> le(nloc), lf(nloc);
        for (int k = 0; k < nloc; ++k) {
            le[k] = world.elements_dynamic[n_elems_root + k].current_energy;
            lf[k] = world.elements_dynamic[n_elems_root + k].total_flux;
        }
        std::vector<int> counts(wsize, 0), displs(wsize, 0);
        if (root) {
            const int base = n_elems_root / active, rem = n_elems_root % active;
            int off = 0;
            for (int r = 0; r < active; ++r) {
                counts[r] = (base + (r < rem ? 1 : 0)) * n_elems_root;
                displs[r] = off;
                off += counts[r];
            }
            energyData.resize(n_elems);
            fluxData.resize(n_elems);
        }
        MPI_Gatherv(le.data(), nloc, MPI_DOUBLE, energyData.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(lf.data(), nloc, MPI_DOUBLE, fluxData.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int ret = 0;
    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");

        if (printResults) print_results(energyData, "ElementEnergy");

        if (validate && !validateResults(energyData, fluxData)) ret = 1;
    }

    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
