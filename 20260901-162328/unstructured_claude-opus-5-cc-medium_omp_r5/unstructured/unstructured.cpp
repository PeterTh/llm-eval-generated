#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// ---------------------------------------------------------------------------
// Thread placement
//
// The kernel is memory bound, so on a multi-socket machine the threads must stay
// on the core whose NUMA node holds the data they first touched. The OpenMP
// runtime only binds threads when OMP_PROC_BIND is set in the environment (the
// proc_bind clause is ignored otherwise), which cannot be fixed up from within
// the program because the runtime parses its environment before main() runs.
// Therefore the threads are pinned explicitly, unless the user asked the OpenMP
// runtime itself to do the binding.
// ---------------------------------------------------------------------------

// Number of distinct physical cores at the front of the cpuOrder() list
static size_t g_num_cores = 0;

// Logical CPU ids ordered by SMT level: one CPU per physical core first (in system
// order, i.e. socket by socket), SMT siblings only afterwards.
static const std::vector<int>& cpuOrder() {
    static const std::vector<int> order = [] {
        std::vector<int> allowed;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &mask)) allowed.push_back(c);
            }
        }

        // Group the CPUs by physical core, identified by the lowest id among its
        // SMT siblings (as reported by sysfs)
        std::vector<int> core_keys;                // lowest sibling id per core
        std::vector<std::vector<int>> core_cpus;   // CPUs of that core
        for (const int c : allowed) {
            int key = c;
            char path[128];
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            if (FILE* f = fopen(path, "r")) {
                int first = -1;
                if (fscanf(f, "%d", &first) == 1 && first >= 0) key = first;
                fclose(f);
            }
            const auto it = std::find(core_keys.begin(), core_keys.end(), key);
            if (it == core_keys.end()) {
                core_keys.push_back(key);
                core_cpus.push_back({c});
            } else {
                core_cpus[it - core_keys.begin()].push_back(c);
            }
        }

        // Round-robin over the cores: all first siblings, then all second ones, ...
        std::vector<int> result;
        result.reserve(allowed.size());
        for (size_t level = 0; result.size() < allowed.size(); ++level) {
            for (const auto& cpus : core_cpus) {
                if (level < cpus.size()) result.push_back(cpus[level]);
            }
        }
        g_num_cores = core_cpus.size();
        return result;
    }();
    return order;
}

// Pin the calling OpenMP thread to a dedicated logical CPU
static void pinThread() {
    const std::vector<int>& order = cpuOrder();
    if (order.empty() || omp_get_proc_bind() != omp_proc_bind_false) return;

    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    const size_t nthreads = static_cast<size_t>(omp_get_num_threads());

    // Teams that fit on the physical cores are spread evenly over them, so that a
    // partially loaded machine still uses the memory controllers of all sockets;
    // larger teams additionally fill the SMT siblings.
    const int cpu = (nthreads <= g_num_cores && nthreads > 0)
                        ? order[tid * g_num_cores / nthreads]
                        : order[tid % order.size()];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// Allocator that leaves trivially-constructible elements uninitialized on resize().
// This keeps the freshly mapped pages untouched so that the parallel initialization
// loops below perform the first touch, placing every page on the NUMA node of the
// thread that will later work on it.
template <typename T>
struct default_init_allocator : std::allocator<T> {
    using std::allocator<T>::allocator;

    template <typename U>
    struct rebind {
        using other = default_init_allocator<U>;
    };

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new(static_cast<void*>(p)) U;
    }
};

// Ask for transparent huge pages on a large, still untouched buffer. The kernel
// may be configured for "madvise" only, in which case streaming the mesh with 4 KiB
// pages would waste a significant part of the TLB.
static void adviseHugePages(void* ptr, size_t bytes) {
    const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t begin = (base + page - 1) & ~(page - 1);
    const uintptr_t end = (base + bytes) & ~(page - 1);
    if (end > begin) {
        madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
    }
}

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

// Element containers use the default-init allocator so that their pages are
// first-touched by the OpenMP worker threads (NUMA locality), see above.
using StaticVec = std::vector<ElementStatic, default_init_allocator<ElementStatic>>;
using DynamicVec = std::vector<ElementDynamic, default_init_allocator<ElementDynamic>>;

// World state
struct World {
    std::vector<Material> materials;
    StaticVec elements_static;
    DynamicVec elements_dynamic;
    DynamicVec elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    adviseHugePages(world.elements_static.data(), n_elems * sizeof(ElementStatic));
    adviseHugePages(world.elements_dynamic.data(), n_elems * sizeof(ElementDynamic));
    adviseHugePages(world.elements_dynamic_swap.data(), n_elems * sizeof(ElementDynamic));

    // Initialize all elements and build connectivity: each element connects to its
    // neighbors in the 2D grid. The loop is flattened over the element index and
    // distributed with the same static schedule as the simulation kernel, so each
    // thread first-touches (and later works on) the very same elements.
    #pragma omp parallel proc_bind(spread)
    {
        pinThread();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < static_cast<size_t>(n_elems); ++i) {
            const int x = static_cast<int>(i) / n_elems_root;
            const int y = static_cast<int>(i) % n_elems_root;

            ElementStatic& elem = world.elements_static[i];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            world.elements_dynamic[i].current_energy = 0.0;
            world.elements_dynamic[i].total_flux = 0.0;
            world.elements_dynamic_swap[i].current_energy = 0.0;
            world.elements_dynamic_swap[i].total_flux = 0.0;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const ElementStatic* __restrict const statics = world.elements_static.data();
    const Material* __restrict const materials = world.materials.data();

    // A single parallel region spans the whole time loop: the implicit barrier of
    // the worksharing construct separates the iterations, so no region is
    // re-created per step. The read/write buffers are swapped redundantly by every
    // thread through private pointers, which avoids an extra synchronization point.
    #pragma omp parallel proc_bind(spread)
    {
        pinThread();

        ElementDynamic* cur_buf = world.elements_dynamic.data();
        ElementDynamic* next_buf = world.elements_dynamic_swap.data();

        for (int iter = 0; iter < n_iters; ++iter) {
            const ElementDynamic* __restrict const read_buf = cur_buf;
            ElementDynamic* __restrict const write_buf = next_buf;

            // Update all elements
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n_elems; ++i) {
                const ElementStatic& elem_static = statics[i];
                const ElementDynamic& elem_dyn = read_buf[i];
                const Material& mat = materials[elem_static.material_idx];

                // Start with external flow
                val_t total_flux = mat.external_flow;

                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = read_buf[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }

                // Update element state
                ElementDynamic& elem_write = write_buf[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }

            // Swap buffers (private view; the containers are swapped after the loop)
            std::swap(cur_buf, next_buf);
        }
    }

    // Make the container holding the final state the "current" one again
    if (n_iters % 2 != 0) {
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
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
uint64_t computeHash(const DynamicVec& elements) {
    uint64_t hash = 0;
    // XOR is associative and commutative, so the reduction is bit-exact
    #pragma omp parallel for schedule(static) reduction(^:hash)
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
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
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
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
