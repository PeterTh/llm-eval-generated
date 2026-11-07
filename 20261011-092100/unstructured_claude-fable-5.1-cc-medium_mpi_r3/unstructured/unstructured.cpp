#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#if __has_include(<linux/mempolicy.h>)
#include <linux/mempolicy.h>
#endif
#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif

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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local numbering)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Allocator backing large arrays with 2 MiB-aligned anonymous mappings that are
// advised for transparent huge pages and pinned to the local NUMA node via an
// explicit memory policy. The explicit policy matters: the MPI runtime starts
// helper threads, which makes the kernel's automatic NUMA balancing scan the
// process and inject millions of hint page faults into the (otherwise
// fault-free) time-stepping loop. VMAs with an explicit policy are skipped by
// that scanner, and each rank's data is local to the core it runs on anyway.
template <typename T>
struct HugePageAllocator {
    using value_type = T;
    static constexpr size_t kHugePage = 2u << 20;

    HugePageAllocator() = default;
    template <typename U>
    HugePageAllocator(const HugePageAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        const size_t bytes = (n * sizeof(T) + kHugePage - 1) / kHugePage * kHugePage;
        if (bytes == 0) return nullptr;
        // Over-allocate by one huge page so that the region can be aligned, then trim.
        char* raw = static_cast<char*>(mmap(nullptr, bytes + kHugePage, PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (raw == MAP_FAILED) throw std::bad_alloc();
        char* aligned = reinterpret_cast<char*>(
            (reinterpret_cast<uintptr_t>(raw) + kHugePage - 1) & ~(uintptr_t)(kHugePage - 1));
        if (aligned != raw) munmap(raw, aligned - raw);
        const size_t tail = (raw + bytes + kHugePage) - (aligned + bytes);
        if (tail > 0) munmap(aligned + bytes, tail);
        madvise(aligned, bytes, MADV_HUGEPAGE);
#ifdef SYS_mbind
        // MPOL_PREFERRED with an empty node mask == allocate on the local node.
        // Failure (no NUMA support, seccomp, ...) is harmless, so ignore it.
        syscall(SYS_mbind, aligned, bytes, MPOL_PREFERRED, nullptr, 0UL, 0U);
#endif
        return reinterpret_cast<T*>(aligned);
    }
    void deallocate(T* p, size_t n) noexcept {
        if (p == nullptr) return;
        const size_t bytes = (n * sizeof(T) + kHugePage - 1) / kHugePage * kHugePage;
        munmap(p, bytes);
    }
    template <typename U>
    bool operator==(const HugePageAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const HugePageAllocator<U>&) const noexcept { return false; }
};

// World state (per-rank partition of the global mesh)
//
// The global mesh is partitioned by contiguous blocks of grid rows (x). Each rank
// stores its owned rows plus one halo row before and one after. The local
// element numbering is
//     local_idx = (x - x_begin + 1) * n_elems_root + y
// so that halo rows occupy local rows 0 and n_local_rows + 1. Because the halo
// rows are contiguous in memory at both ends of the dynamic buffers, halo
// exchange is done with zero packing directly out of / into those buffers.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic, HugePageAllocator<ElementStatic>> elements_static;
    std::vector<ElementDynamic, HugePageAllocator<ElementDynamic>> elements_dynamic;
    std::vector<ElementDynamic, HugePageAllocator<ElementDynamic>> elements_dynamic_swap;

    // Decomposition info
    int n_elems_root = 0;   // Global grid side length
    int x_begin = 0;        // First owned global row
    int n_local_rows = 0;   // Number of owned rows
    int prev_rank = MPI_PROC_NULL;  // Rank owning row x_begin - 1
    int next_rank = MPI_PROC_NULL;  // Rank owning row x_begin + n_local_rows

    size_t n_owned() const { return static_cast<size_t>(n_local_rows) * n_elems_root; }
    size_t owned_begin() const { return static_cast<size_t>(n_elems_root); }  // local index of first owned element
};

// MPI datatype matching ElementDynamic (two contiguous doubles)
inline MPI_Datatype elementDynamicType() {
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "ElementDynamic layout");
        MPI_Type_contiguous(2, MPI_DOUBLE, &type);
        MPI_Type_commit(&type);
    }
    return type;
}

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Number of rows owned by a given rank for an even block distribution
inline int rowsForRank(const int n_elems_root, const int rank, const int size) {
    return n_elems_root / size + (rank < n_elems_root % size ? 1 : 0);
}

inline int rowBeginForRank(const int n_elems_root, const int rank, const int size) {
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    return rank * base + std::min(rank, rem);
}

// Build (the local partition of) a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int size) {
    world.n_elems_root = n_elems_root;
    world.n_local_rows = rowsForRank(n_elems_root, rank, size);
    world.x_begin = rowBeginForRank(n_elems_root, rank, size);

    // Neighbouring ranks (only if they own at least one row)
    world.prev_rank = MPI_PROC_NULL;
    world.next_rank = MPI_PROC_NULL;
    if (world.n_local_rows > 0) {
        if (rank > 0 && rowsForRank(n_elems_root, rank - 1, size) > 0) world.prev_rank = rank - 1;
        if (rank + 1 < size && rowsForRank(n_elems_root, rank + 1, size) > 0) world.next_rank = rank + 1;
    }

    const size_t n_local = static_cast<size_t>(world.n_local_rows + 2) * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements (owned + 2 halo rows)
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    // Initialize all elements with default material and zero energy
    for (size_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux = 0.0;
    }

    // Build connectivity: each owned element connects to its neighbors in 2D grid
    for (int lx = 0; lx < world.n_local_rows; ++lx) {
        const int x = world.x_begin + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(lx + 1) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx =
                        static_cast<size_t>(nx - world.x_begin + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto setMaterial = [&](const int x, const int y, const idx_t mat) {
        if (x >= world.x_begin && x < world.x_begin + world.n_local_rows) {
            const size_t idx = static_cast<size_t>(x - world.x_begin + 1) * n_elems_root + y;
            world.elements_static[idx].material_idx = mat;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update a contiguous range [begin, end) of local elements
inline void updateRange(const World& world, const size_t begin, const size_t end) {
    const ElementStatic* __restrict es = world.elements_static.data();
    const ElementDynamic* __restrict ed = world.elements_dynamic.data();
    ElementDynamic* __restrict ew = const_cast<ElementDynamic*>(world.elements_dynamic_swap.data());
    const Material* __restrict mats = world.materials.data();

    for (size_t i = begin; i < end; ++i) {
        const ElementStatic& elem_static = es[i];
        const ElementDynamic& elem_dyn = ed[i];
        const Material& mat = mats[elem_static.material_idx];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = ed[neighbor_idx];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }

        // Update element state
        ElementDynamic& elem_write = ew[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n = world.n_elems_root;
    const size_t n_rows = world.n_local_rows;
    const size_t first = world.owned_begin();        // first owned local index
    const size_t last = first + n_rows * n;          // one past last owned local index

    // Only current_energy of halo elements is needed by neighbours, but exchanging
    // whole ElementDynamic rows (16 B/elem) keeps the exchange zero-copy.
    const int halo_count = static_cast<int>(n);
    const MPI_Datatype elem_t = elementDynamicType();

    for (int iter = 0; iter < n_iters; ++iter) {
        ElementDynamic* dyn = world.elements_dynamic.data();
        MPI_Request reqs[4];
        int n_reqs = 0;

        if (n_rows > 0) {
            // Receive halos from neighbours into the halo rows
            MPI_Irecv(dyn, halo_count, elem_t, world.prev_rank, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Irecv(dyn + last, halo_count, elem_t, world.next_rank, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
            // Send our first/last owned rows
            MPI_Isend(dyn + first, halo_count, elem_t, world.prev_rank, 1, MPI_COMM_WORLD, &reqs[n_reqs++]);
            MPI_Isend(dyn + last - n, halo_count, elem_t, world.next_rank, 0, MPI_COMM_WORLD, &reqs[n_reqs++]);
        }

        if (n_rows > 2) {
            // Interior rows do not depend on the halo: overlap with communication
            updateRange(world, first + n, last - n);
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
            updateRange(world, first, first + n);
            updateRange(world, last - n, last);
        } else {
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
            updateRange(world, first, last);
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Gather the owned dynamic state of all ranks onto rank 0 in global order.
// Returns the full array on rank 0, an empty vector elsewhere.
std::vector<ElementDynamic> gatherDynamic(const World& world, const int rank, const int size) {
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(static_cast<size_t>(rowsForRank(world.n_elems_root, r, size)) * world.n_elems_root);
        displs[r] = static_cast<int>(static_cast<size_t>(rowBeginForRank(world.n_elems_root, r, size)) * world.n_elems_root);
    }
    std::vector<ElementDynamic> all;
    if (rank == 0) all.resize(static_cast<size_t>(world.n_elems_root) * world.n_elems_root);
    MPI_Gatherv(world.elements_dynamic.data() + world.owned_begin(),
                static_cast<int>(world.n_owned()), elementDynamicType(),
                all.data(), counts.data(), displs.data(), elementDynamicType(), 0, MPI_COMM_WORLD);
    return all;
}

// Validate simulation results (expects the globally gathered element array; rank 0 only)
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
// Each rank hashes its owned elements using their global indices; since the
// combination is XOR, the partial hashes are reduced with MPI_BXOR to obtain
// exactly the same value as the sequential hash over the global array.
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const ElementDynamic* elements = world.elements_dynamic.data() + world.owned_begin();
    const size_t n_owned = world.n_owned();
    const size_t global_offset = static_cast<size_t>(world.x_begin) * world.n_elems_root;
    for (size_t k = 0; k < n_owned; ++k) {
        const size_t i = global_offset + k;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[k].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[k].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Allreduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    if (n_elems_root < 0) n_elems_root = 0;

    const int n_elems = n_elems_root * n_elems_root;

    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (global, summed over all ranks)
    size_t local_mem[2] = {world.elements_static.size() * sizeof(ElementStatic),
                           world.elements_dynamic.size() * sizeof(ElementDynamic) * 2};
    size_t global_mem[2] = {0, 0};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (is_root) {
        const size_t static_mem = global_mem[0];
        const size_t dynamic_mem = global_mem[1];
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Compute hash for verification (collective)
    const uint64_t hash = computeHash(world);

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

    int exit_code = 0;
    if (printResults || validate) {
        // Gather the full state on rank 0 so that output is identical to the sequential version
        const std::vector<ElementDynamic> all = gatherDynamic(world, rank, size);

        // Print results for external validation
        if (printResults && is_root) {
            std::vector<double> energyData;
            energyData.reserve(all.size());
            for (const auto& elem : all) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate && is_root) {
            bool valid = validateResults(all);
            if (!valid) {
                exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Datatype elem_t = elementDynamicType();
    MPI_Type_free(&elem_t);
    MPI_Finalize();
    return exit_code;
}
