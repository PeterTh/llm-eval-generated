#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Cache line size used for alignment and false-sharing avoidance
constexpr size_t CACHE_LINE = 64;

// Allocator that leaves default-constructed (trivial) elements uninitialized.
// This keeps std::vector::resize() from touching every page from the calling
// thread, so that the parallel initialization loops below perform the NUMA
// first touch and every thread ends up owning the pages it later works on.
template <typename T>
struct NoInitAllocator {
    using value_type = T;

    NoInitAllocator() noexcept = default;
    template <typename U>
    constexpr NoInitAllocator(const NoInitAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(CACHE_LINE)));
    }
    void deallocate(T* p, size_t) noexcept {
        ::operator delete(p, std::align_val_t(CACHE_LINE));
    }

    // Skip value-initialization; the elements are filled in explicitly.
    template <typename U>
    void construct(U*) noexcept {
        static_assert(std::is_trivially_default_constructible_v<U>);
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    bool operator==(const NoInitAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const NoInitAllocator<U>&) const noexcept { return false; }
};

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

template <typename T>
using ElemVector = std::vector<T, NoInitAllocator<T>>;

using DynamicArray = ElemVector<ElementDynamic>;

// World state
struct World {
    std::vector<Material> materials;
    ElemVector<ElementStatic> elements_static;
    DynamicArray elements_dynamic;
    DynamicArray elements_dynamic_swap;
};

// Split [0, n) into one contiguous chunk per thread of the enclosing parallel
// region. Chunk boundaries are cache line aligned with respect to
// ElementDynamic so that neighbouring threads never write the same line, and
// the partition depends only on the thread count, which lets the mesh setup
// and the simulation use exactly the same distribution (NUMA first touch).
inline void threadRange(size_t n, size_t& begin, size_t& end) {
    constexpr size_t grain = CACHE_LINE / sizeof(ElementDynamic);
    const size_t n_threads = static_cast<size_t>(omp_get_num_threads());
    const size_t tid = static_cast<size_t>(omp_get_thread_num());

    const size_t n_blocks = (n + grain - 1) / grain;
    const size_t per_thread = n_blocks / n_threads;
    const size_t remainder = n_blocks % n_threads;
    const size_t first_block = tid * per_thread + std::min(tid, remainder);
    const size_t n_own = per_thread + (tid < remainder ? 1 : 0);

    begin = std::min(first_block * grain, n);
    end = std::min((first_block + n_own) * grain, n);
}

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
    
    // Initialize elements and build connectivity in parallel. Each thread owns
    // exactly the index range it will later update in the simulation, so this
    // loop performs the NUMA first touch of the pages it will be working on.
    #pragma omp parallel
    {
        size_t begin, end;
        threadRange(static_cast<size_t>(n_elems), begin, end);

        const size_t row_len = static_cast<size_t>(n_elems_root);
        for (size_t i = begin; i < end; ++i) {
            const int x = static_cast<int>(i / row_len);
            const int y = static_cast<int>(i % row_len);

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
    const ElementStatic* const __restrict elems_static = world.elements_static.data();
    const Material* const __restrict materials = world.materials.data();

    // Buffers are flipped by pointer instead of by swapping the vectors, so the
    // parallel region can stay open across all iterations.
    ElementDynamic* read_buf = world.elements_dynamic.data();
    ElementDynamic* write_buf = world.elements_dynamic_swap.data();

    #pragma omp parallel
    {
        // Same partitioning as the mesh setup: every thread updates the range
        // of elements whose pages it first touched.
        size_t begin, end;
        threadRange(n_elems, begin, end);

        for (int iter = 0; iter < n_iters; ++iter) {
            const ElementDynamic* const __restrict src = read_buf;
            ElementDynamic* const __restrict dst = write_buf;

            // Update all elements owned by this thread
            for (size_t i = begin; i < end; ++i) {
                const ElementStatic& elem_static = elems_static[i];
                const ElementDynamic& elem_dyn = src[i];
                const Material& mat = materials[elem_static.material_idx];

                // Start with external flow
                val_t total_flux = mat.external_flow;

                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = src[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }

                // Update element state
                ElementDynamic& elem_write = dst[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }

            // All updates must be visible before the buffers are flipped
            #pragma omp barrier
            #pragma omp single
            {
                std::swap(read_buf, write_buf);
            }
            // implicit barrier at the end of 'single'
        }
    }

    // Make sure the freshest state lives in world.elements_dynamic
    if (read_buf != world.elements_dynamic.data()) {
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
uint64_t computeHash(const DynamicArray& elements) {
    uint64_t hash = 0;
    // XOR is associative and commutative, so the reduction is bit-exact
    #pragma omp parallel for reduction(^ : hash) schedule(static)
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

#ifdef __linux__
// Read a "0-3,8" style CPU list from sysfs
std::vector<int> parseCpuList(const char* path) {
    std::vector<int> cpus;
    FILE* f = fopen(path, "r");
    if (!f) {
        return cpus;
    }
    char buf[1024];
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) {
        return cpus;
    }

    const char* p = buf;
    while (*p) {
        char* endp = nullptr;
        long first = strtol(p, &endp, 10);
        if (endp == p) {
            break;
        }
        p = endp;
        long last = first;
        if (*p == '-') {
            last = strtol(p + 1, &endp, 10);
            p = endp;
        }
        for (long c = first; c <= last; ++c) {
            cpus.push_back(static_cast<int>(c));
        }
        if (*p == ',') {
            ++p;
        } else {
            break;
        }
    }
    return cpus;
}

// Usable CPUs grouped by socket and by SMT level, each group sorted by CPU id
struct CpuTopology {
    // cpus[package][smt_rank] -> cpu ids
    std::vector<std::vector<std::vector<int>>> cpus;
    size_t n_cores = 0;  // number of usable physical cores
};

CpuTopology queryTopology() {
    CpuTopology topo;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return topo;
    }

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }

        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        const std::vector<int> siblings = parseCpuList(path);
        int smt_rank = 0;
        for (const int sib : siblings) {
            if (sib == cpu) {
                break;
            }
            if (CPU_ISSET(sib, &allowed)) {
                ++smt_rank;
            }
        }

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        int package = 0;
        if (FILE* f = fopen(path, "r")) {
            if (fscanf(f, "%d", &package) != 1) {
                package = 0;
            }
            fclose(f);
        }
        package = std::max(package, 0);

        const size_t pkg = static_cast<size_t>(package);
        const size_t smt = static_cast<size_t>(smt_rank);
        if (topo.cpus.size() <= pkg) {
            topo.cpus.resize(pkg + 1);
        }
        if (topo.cpus[pkg].size() <= smt) {
            topo.cpus[pkg].resize(smt + 1);
        }
        topo.cpus[pkg][smt].push_back(cpu);
        if (smt == 0) {
            ++topo.n_cores;
        }
    }

    // Drop packages without any usable CPU (e.g. under a restricted cpuset)
    topo.cpus.erase(std::remove_if(topo.cpus.begin(), topo.cpus.end(),
                                   [](const std::vector<std::vector<int>>& pkg) {
                                       for (const auto& level : pkg) {
                                           if (!level.empty()) return false;
                                       }
                                       return true;
                                   }),
                    topo.cpus.end());

    return topo;
}

// Pick 'count' CPUs out of a list, spread as evenly as possible over it. The
// list is walked with a stride so that, on chiplet based CPUs, a partially
// filled socket still uses cores from all of its chiplets (and therefore all
// of the available fabric/memory bandwidth) instead of crowding onto a few.
void pickSpread(const std::vector<int>& cpus, size_t count, std::vector<int>& out) {
    if (cpus.empty() || count == 0) {
        return;
    }
    if (count >= cpus.size()) {
        out.insert(out.end(), cpus.begin(), cpus.end());
        return;
    }
    for (size_t k = 0; k < count; ++k) {
        out.push_back(cpus[(k * cpus.size()) / count]);
    }
}

// Compute the CPU each OpenMP thread should be pinned to. The threads are
// split into one contiguous block per socket, so that each socket owns one
// contiguous half of the mesh (the threads of a block work on neighbouring
// elements, and only the two block borders cross sockets). Within a socket the
// threads fill the physical cores, spread over the chiplets, before any SMT
// sibling is used. Together with the first-touch initialization this keeps
// every memory controller busy, which is what this bandwidth-bound kernel
// needs.
std::vector<int> buildPlacement(const CpuTopology& topo, size_t n_threads) {
    const size_t n_packages = topo.cpus.size();

    std::vector<int> placement;
    placement.reserve(n_threads);

    for (size_t pkg = 0; pkg < n_packages; ++pkg) {
        // Threads assigned to this socket
        size_t remaining = n_threads / n_packages + (pkg < n_threads % n_packages ? 1 : 0);
        const size_t wanted = remaining;
        const size_t before = placement.size();

        for (const std::vector<int>& level : topo.cpus[pkg]) {
            if (remaining == 0) {
                break;
            }
            const size_t take = std::min(remaining, level.size());
            pickSpread(level, take, placement);
            remaining -= take;
        }

        // More threads than CPUs on this socket: reuse them round-robin
        const size_t available = placement.size() - before;
        if (available == 0) {
            placement.clear();
            return placement;
        }
        for (size_t k = available; k < wanted; ++k) {
            placement.push_back(placement[before + (k % available)]);
        }
    }
    return placement;
}
#endif  // __linux__

// Pin the OpenMP threads unless the user asked for a specific placement.
void configureThreadPlacement() {
    // Fixed team sizes: the mesh setup and the simulation must partition the
    // elements the same way for the first-touch placement to pay off.
    omp_set_dynamic(0);

#ifdef __linux__
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) {
        return;  // respect the user's placement policy
    }

    const CpuTopology topo = queryTopology();
    if (topo.cpus.empty() || topo.n_cores == 0) {
        return;
    }

    // This kernel is memory bandwidth bound, so SMT siblings add contention
    // rather than throughput: default to one thread per physical core.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(static_cast<int>(topo.n_cores));
    }

    const std::vector<int> placement =
        buildPlacement(topo, static_cast<size_t>(omp_get_max_threads()));
    if (placement.empty()) {
        return;
    }

    #pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        if (tid < placement.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(placement[tid], &set);
            sched_setaffinity(0, sizeof(set), &set);
        }
    }
#endif
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
    
    configureThreadPlacement();

    const int n_elems = n_elems_root * n_elems_root;

    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
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
