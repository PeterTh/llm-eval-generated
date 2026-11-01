#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"
#include <mpi.h>

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

// Build owned rows [x_start, x_end) of a 2D square grid
void buildLocal(World& world, const int n_elems_root, int x_start, int x_end) {
    const int n_local_rows = x_end - x_start;
    const int n_local = n_local_rows * n_elems_root;

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    for (int i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = x_start; x < x_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = (x - x_start) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    const int last = n_elems_root - 1;
    if (x_start == 0) {
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
        world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    // set corners by global index mapping
    auto setmat = [&](int gx, int gy, idx_t m) {
        if (gx >= x_start && gx < x_end) {
            world.elements_static[(gx-x_start)*n_elems_root + gy].material_idx = m;
        }
    };
    setmat(0, 0, INFLOW_MAT_ID);
    setmat(0, last, OUTFLOW_MAT_ID);
    setmat(last, 0, OUTFLOW_MAT_ID);
    setmat(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Map global element index to pointer in dynamic buffer with ghosts
static inline ElementDynamic* getDyn(std::vector<ElementDynamic>& buf, int n_elems_root, int x_start, int x_end, idx_t gidx) {
    int x = (int)(gidx / n_elems_root);
    int y = (int)(gidx % n_elems_root);
    if (x >= x_start && x < x_end) {
        return &buf[n_elems_root + (x - x_start) * n_elems_root + y];
    } else if (x == x_start - 1) {
        return &buf[y]; // top ghost
    } else if (x == x_end) {
        return &buf[n_elems_root + (x_end - x_start) * n_elems_root + y]; // bottom ghost
    }
    return nullptr;
}

void runSimulation(World& world, const int n_elems_root, int x_start, int x_end, int rank, int size, const int n_iters) {
    const int n_local = (x_end - x_start) * n_elems_root;
    const int N = n_elems_root;
    // buffers: top ghost N, owned n_local, bottom ghost N
    std::vector<ElementDynamic> cur(N + n_local + N), nxt(N + n_local + N);
    for (int i = 0; i < n_local; ++i) {
        cur[N + i] = world.elements_dynamic[i];
        nxt[N + i] = world.elements_dynamic[i];
    }
    // zero ghosts
    for (int i = 0; i < N; ++i) { cur[i].current_energy=0; cur[i].total_flux=0; cur[N+n_local+i].current_energy=0; cur[N+n_local+i].total_flux=0; }

    for (int iter = 0; iter < n_iters; ++iter) {
        // exchange boundaries
        if (rank > 0) {
            MPI_Sendrecv(&cur[N], N*sizeof(ElementDynamic), MPI_BYTE, rank-1, 0,
                         &cur[0], N*sizeof(ElementDynamic), MPI_BYTE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size-1) {
            MPI_Sendrecv(&cur[N + n_local - N], N*sizeof(ElementDynamic), MPI_BYTE, rank+1, 1,
                         &cur[N + n_local], N*sizeof(ElementDynamic), MPI_BYTE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // compute
        for (int i = 0; i < n_local; ++i) {
            const ElementStatic& es = world.elements_static[i];
            ElementDynamic& ed = cur[N + i];
            const Material& mat = world.materials[es.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                idx_t nid = es.connected_idx[j];
                ElementDynamic* nb = getDyn(cur, N, x_start, x_end, nid);
                total_flux += computeFlux(mat, ed, es.connected_flux[j], *nb);
            }
            nxt[N + i].current_energy = ed.current_energy + total_flux;
            nxt[N + i].total_flux = ed.total_flux + std::abs(total_flux);
        }
        std::swap(cur, nxt);
    }
    // copy owned back
    for (int i = 0; i < n_local; ++i) world.elements_dynamic[i] = cur[N + i];
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Partition rows
    int base = n_elems_root / size;
    int rem = n_elems_root % size;
    int x_start = rank * base + std::min(rank, rem);
    int x_end = x_start + base + (rank < rem ? 1 : 0);

    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildLocal(world, n_elems_root, x_start, x_end);

    if (rank == 0) {
        const size_t static_mem = (size_t)n_elems_root*n_elems_root*sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems_root*n_elems_root*sizeof(ElementDynamic)*2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (static_mem+dynamic_mem)/(1024.0*1024.0),
               static_mem/(1024.0*1024.0),
               dynamic_mem/(1024.0*1024.0));
        printf("\nRunning simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_elems_root, x_start, x_end, rank, size, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Gather results to rank 0
    int n_local = world.elements_dynamic.size();
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];
    }
    // convert to bytes
    std::vector<int> bcounts(size), bdispls(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) { bcounts[i] = counts[i] * (int)sizeof(ElementDynamic); bdispls[i] = displs[i] * (int)sizeof(ElementDynamic); }
    }
    World global;
    if (rank == 0) {
        int total = n_elems_root * n_elems_root;
        global.elements_dynamic.resize(total);
    }
    MPI_Gatherv(world.elements_dynamic.data(), n_local * (int)sizeof(ElementDynamic), MPI_BYTE,
                rank==0 ? global.elements_dynamic.data() : nullptr,
                rank==0 ? bcounts.data() : nullptr, rank==0 ? bdispls.data() : nullptr, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (long long)n_elems_root * n_elems_root) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        const uint64_t hash = computeHash(global.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global.elements_dynamic.size());
            for (const auto& elem : global.elements_dynamic) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }
        if (validate) {
            World w; w.elements_dynamic = global.elements_dynamic;
            // validateResults expects world; set minimal
            bool valid = validateResults(w);
            if (!valid) { MPI_Finalize(); return 1; }
        }
    }
    MPI_Finalize();
    return 0;
}
