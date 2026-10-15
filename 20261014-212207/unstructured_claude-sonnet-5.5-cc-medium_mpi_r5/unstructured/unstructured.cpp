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
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build the locally owned rows [row0, row0+nrows) of a 2D square grid as an
// unstructured mesh. Dynamic arrays carry one ghost row on each side; connected
// indices refer to positions in the padded dynamic arrays.
void buildSquare2D(World& world, const int n_elems_root, const int row0, const int nrows) {
    const size_t n = n_elems_root;
    const size_t n_local = static_cast<size_t>(nrows) * n;

    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elements_static.resize(n_local);
    world.elements_dynamic.assign(n_local + 2 * n, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(n_local + 2 * n, ElementDynamic{0.0, 0.0});

    const idx_t pad_base = static_cast<idx_t>(row0) * n_elems_root - n_elems_root;  // may wrap; only used modularly

    for (int lx = 0; lx < nrows; ++lx) {
        const int x = row0 + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(lx) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int k = 0; k < 4; ++k) {
                const int nx = x + offsets[k][0];
                const int ny = y + offsets[k][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t g = static_cast<idx_t>(nx) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = g - pad_base;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    auto setMat = [&](int x, int y, idx_t m) {
        if (x >= row0 && x < row0 + nrows)
            world.elements_static[static_cast<size_t>(x - row0) * n + y].material_idx = m;
    };
    setMat(0, 0, INFLOW_MAT_ID);
    setMat(0, last, OUTFLOW_MAT_ID);
    setMat(last, 0, OUTFLOW_MAT_ID);
    setMat(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update owned elements [a, b) (owned-local indices) reading from the padded array
static inline void updateRange(const World& world, size_t a, size_t b, size_t pad) {
    const ElementStatic* __restrict es = world.elements_static.data();
    const ElementDynamic* __restrict cur = world.elements_dynamic.data();
    ElementDynamic* __restrict nxt = const_cast<ElementDynamic*>(world.elements_dynamic_swap.data());
    const Material* mats = world.materials.data();
    for (size_t i = a; i < b; ++i) {
        const ElementStatic& elem_static = es[i];
        const ElementDynamic& elem_dyn = cur[i + pad];
        const Material& mat = mats[elem_static.material_idx];

        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const ElementDynamic& neighbor_dyn = cur[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        ElementDynamic& elem_write = nxt[i + pad];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations with halo exchange of boundary rows
void runSimulation(World& world, const int n_iters, const int n_root, const int nrows,
                   const int rank, const int nprocs, MPI_Comm comm) {
    const size_t n = n_root;
    const size_t n_local = static_cast<size_t>(nrows) * n;
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank < nprocs - 1 ? rank + 1 : MPI_PROC_NULL;
    const int cnt = static_cast<int>(n * 2);  // doubles per row

    for (int iter = 0; iter < n_iters; ++iter) {
        double* cur = reinterpret_cast<double*>(world.elements_dynamic.data());
        MPI_Request req[4];
        // Receive into ghost rows, send first/last owned rows
        MPI_Irecv(cur, cnt, MPI_DOUBLE, prev, 0, comm, &req[0]);
        MPI_Irecv(cur + 2 * (n_local + n), cnt, MPI_DOUBLE, next, 1, comm, &req[1]);
        MPI_Isend(cur + 2 * n, cnt, MPI_DOUBLE, prev, 1, comm, &req[2]);
        MPI_Isend(cur + 2 * n_local, cnt, MPI_DOUBLE, next, 0, comm, &req[3]);

        // Interior rows don't need ghosts: overlap with communication
        if (nrows >= 2) updateRange(world, n, n_local - n, n);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        updateRange(world, 0, std::min(n, n_local), n);
        if (nrows >= 2) updateRange(world, n_local - n, n_local, n);

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elems) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : elems) {
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

// Hash contribution of locally owned elements (XOR is order independent);
// first_idx is the global index of the first owned element
uint64_t computeHash(const ElementDynamic* elements, size_t count, size_t first_idx) {
    uint64_t hash = 0;
    for (size_t k = 0; k < count; ++k) {
        const size_t i = first_idx + k;
        uint64_t e, f;
        std::memcpy(&e, &elements[k].current_energy, sizeof(e));
        std::memcpy(&f, &elements[k].total_flux, sizeof(f));
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
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
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const bool root = world_rank == 0;

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

    // Use at most one rank per grid row
    const int nprocs = std::max(1, std::min(world_size, n_elems_root));
    const bool active = world_rank < nprocs;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (!active) {
        MPI_Finalize();
        return 0;
    }
    const int rank = world_rank;

    // 1D row decomposition
    const int base = n_elems_root / nprocs, rem = n_elems_root % nprocs;
    auto rowStart = [&](int r) { return r * base + std::min(r, rem); };
    const int row0 = rowStart(rank);
    const int nrows = rowStart(rank + 1) - row0;
    const size_t n_local = static_cast<size_t>(nrows) * n_elems_root;
    const size_t first_idx = static_cast<size_t>(row0) * n_elems_root;

    World world;
    buildSquare2D(world, n_elems_root, row0, nrows);

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

    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, n_elems_root, nrows, rank, nprocs, comm);

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long dur_ll = duration_ms, dur_max = 0;
    MPI_Reduce(&dur_ll, &dur_max, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    duration_ms = dur_max;

    // Global hash
    const ElementDynamic* owned = world.elements_dynamic.data() + n_elems_root;
    uint64_t local_hash = computeHash(owned, n_local, first_idx), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    // Gather full state on root only when needed
    std::vector<ElementDynamic> all;
    if (validate || printResults) {
        std::vector<int> counts(nprocs), displs(nprocs);
        if (root) {
            all.resize(n_elems);
            for (int r = 0; r < nprocs; ++r) {
                counts[r] = (rowStart(r + 1) - rowStart(r)) * n_elems_root * 2;
                displs[r] = rowStart(r) * n_elems_root * 2;
            }
        }
        MPI_Gatherv(owned, static_cast<int>(n_local * 2), MPI_DOUBLE, all.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    }

    int rc = 0;
    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(all.size());
            for (const auto& elem : all) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate && !validateResults(all)) rc = 1;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return rc;
}
