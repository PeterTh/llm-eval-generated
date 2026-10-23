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

// World state (distributed across MPI ranks)
struct World {
    std::vector<Material> materials;
    std::vector<idx_t> material_idx;
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux;
    std::vector<val_t> flux_swap;
    int n_root = 0;
    int local_nx = 0;
    int start_x = 0;
    int end_x = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr int TAG_TO_PREV = 10;
constexpr int TAG_TO_NEXT = 20;

// Build the local part of a 2D square grid as an unstructured mesh.
// Each rank owns a contiguous block of rows in the x direction, while all
// ranks store the full y dimension. One halo row is kept on each x side.
void buildSquare2D(World& world, const int n_elems_root, const int start_x,
                   const int end_x) {
    const int N = n_elems_root;
    const int local_nx = end_x - start_x;

    world.n_root = N;
    world.local_nx = local_nx;
    world.start_x = start_x;
    world.end_x = end_x;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const size_t ext_rows = static_cast<size_t>(local_nx + 2);
    world.energy.assign(ext_rows * N, 0.0);
    world.energy_swap.assign(ext_rows * N, 0.0);
    world.flux.assign(ext_rows * N, 0.0);
    world.flux_swap.assign(ext_rows * N, 0.0);

    const size_t local_elems = static_cast<size_t>(local_nx) * N;
    world.material_idx.assign(local_elems, DEFAULT_MAT_ID);

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = N - 1;
    for (int lx = 0; lx < local_nx; ++lx) {
        const int x = start_x + lx;
        for (int y = 0; y < N; ++y) {
            idx_t& material =
                world.material_idx[static_cast<size_t>(lx) * N + y];
            if (x == 0 && y == 0) {
                material = INFLOW_MAT_ID;
            }
            if (x == 0 && y == last) {
                material = OUTFLOW_MAT_ID;
            }
            if (x == last && y == 0) {
                material = OUTFLOW_MAT_ID;
            }
            if (x == last && y == last) {
                material = INFLOW_MAT_ID;
            }
        }
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, const int prev_rank,
                   const int next_rank) {
    const int N = world.n_root;
    const int local_nx = world.local_nx;
    const size_t row_stride = static_cast<size_t>(N);
    const bool has_neighbors =
        (prev_rank != MPI_PROC_NULL) || (next_rank != MPI_PROC_NULL);

    auto compute_row = [&](const int lx) {
        const size_t mat_base = static_cast<size_t>(lx) * row_stride;
        const size_t dyn_base = static_cast<size_t>(lx + 1) * row_stride;
        const int x = world.start_x + lx;
        const bool has_up = (x + 1 < N);
        const bool has_down = (x > 0);

        const val_t* __restrict__ energy = world.energy.data();
        val_t* __restrict__ energy_swap = world.energy_swap.data();
        const val_t* __restrict__ flux = world.flux.data();
        val_t* __restrict__ flux_swap = world.flux_swap.data();
        const idx_t* __restrict__ material_idx = world.material_idx.data();
        const Material* __restrict__ materials = world.materials.data();

        for (int y = 0; y < N; ++y) {
            const size_t i = mat_base + y;
            const size_t dyn_idx = dyn_base + y;
            const val_t this_energy = energy[dyn_idx];
            const Material& mat = materials[material_idx[i]];

            // Start with external flow
            val_t total_flux = mat.external_flow;

            // Add flux from all connected elements, in the original order
            if (has_up) {
                total_flux += computeFlux(mat, this_energy, 1.0,
                                          energy[dyn_idx + row_stride]);
            }
            if (has_down) {
                total_flux += computeFlux(mat, this_energy, 1.0,
                                          energy[dyn_idx - row_stride]);
            }
            if (y + 1 < N) {
                total_flux += computeFlux(mat, this_energy, 1.0,
                                          energy[dyn_idx + 1]);
            }
            if (y > 0) {
                total_flux += computeFlux(mat, this_energy, 1.0,
                                          energy[dyn_idx - 1]);
            }

            // Update element state
            energy_swap[dyn_idx] = this_energy + total_flux;
            flux_swap[dyn_idx] = flux[dyn_idx] + std::abs(total_flux);
        }
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        if (iter + 1 < n_iters && has_neighbors) {
            // Compute the boundary rows first so their exchange can overlap
            // with the computation of the interior rows.
            if (local_nx > 0) {
                compute_row(0);
                if (local_nx > 1) {
                    compute_row(local_nx - 1);
                }
            }

            val_t* const data = world.energy_swap.data();
            val_t* const halo_left = data;
            val_t* const first_owned = data + row_stride;
            val_t* const last_owned =
                data + static_cast<size_t>(local_nx) * row_stride;
            val_t* const halo_right =
                data + static_cast<size_t>(local_nx + 1) * row_stride;

            MPI_Request requests[4];
            int n_requests = 0;

            if (prev_rank != MPI_PROC_NULL) {
                MPI_Irecv(halo_left, N, MPI_DOUBLE, prev_rank, TAG_TO_NEXT,
                          MPI_COMM_WORLD, &requests[n_requests++]);
                MPI_Isend(first_owned, N, MPI_DOUBLE, prev_rank, TAG_TO_PREV,
                          MPI_COMM_WORLD, &requests[n_requests++]);
            }

            if (next_rank != MPI_PROC_NULL) {
                MPI_Irecv(halo_right, N, MPI_DOUBLE, next_rank, TAG_TO_PREV,
                          MPI_COMM_WORLD, &requests[n_requests++]);
                MPI_Isend(last_owned, N, MPI_DOUBLE, next_rank, TAG_TO_NEXT,
                          MPI_COMM_WORLD, &requests[n_requests++]);
            }

            // Interior rows overlap with the halo communication
            for (int lx = 1; lx < local_nx - 1; ++lx) {
                compute_row(lx);
            }

            if (n_requests > 0) {
                MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);
            }
        } else {
            // Last iteration (or no halo needed): update all local elements
            for (int lx = 0; lx < local_nx; ++lx) {
                compute_row(lx);
            }
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
        std::swap(world.flux, world.flux_swap);
    }
}

// Gather all owned element data onto rank 0 in global (x, y) order.
void gatherResults(const World& world, std::vector<val_t>& all_energy,
                   std::vector<val_t>& all_flux, const int rank,
                   const int size) {
    const int N = world.n_root;
    const int local_count = world.local_nx * N;

    std::vector<int> counts(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        const int s = static_cast<int>(static_cast<long long>(r) * N / size);
        const int e =
            static_cast<int>(static_cast<long long>(r + 1) * N / size);
        counts[r] = (e - s) * N;
        displs[r] = offset;
        offset += counts[r];
    }

    if (rank == 0) {
        all_energy.resize(static_cast<size_t>(offset));
        all_flux.resize(static_cast<size_t>(offset));
    }

    const val_t* send_energy = world.energy.data() + N;
    const val_t* send_flux = world.flux.data() + N;
    val_t* recv_energy = (rank == 0) ? all_energy.data() : nullptr;
    val_t* recv_flux = (rank == 0) ? all_flux.data() : nullptr;

    MPI_Gatherv(send_energy, local_count, MPI_DOUBLE, recv_energy,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(send_flux, local_count, MPI_DOUBLE, recv_flux, counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
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

// Compute the local contribution to the verification hash. The hash is an
// XOR of independent per-element terms, so local contributions can be
// combined across ranks with a bitwise XOR reduction.
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    const size_t n_elems = world.material_idx.size();
    const size_t row_stride = static_cast<size_t>(world.n_root);
    const size_t global_base =
        static_cast<size_t>(world.start_x) * row_stride;

    for (size_t i = 0; i < n_elems; ++i) {
        const size_t dyn_idx = i + row_stride;
        const size_t global_idx = global_base + i;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr =
            reinterpret_cast<const uint64_t*>(&world.energy[dyn_idx]);
        const uint64_t* f_ptr =
            reinterpret_cast<const uint64_t*>(&world.flux[dyn_idx]);
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
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

    // Parse command line arguments (all ranks parse identically)
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
            parseError = true;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            break;
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return showHelp ? 0 : 1;
    }

    const int N = n_elems_root;
    const int n_elems = N * N;

    // Partition rows of the grid across ranks
    const int start_x =
        static_cast<int>(static_cast<long long>(rank) * N / size);
    const int end_x =
        static_cast<int>(static_cast<long long>(rank + 1) * N / size);

    // Determine neighboring ranks that own adjacent rows (skipping empty ranks)
    std::vector<int> owner(N);
    for (int r = 0; r < size; ++r) {
        const int s = static_cast<int>(static_cast<long long>(r) * N / size);
        const int e =
            static_cast<int>(static_cast<long long>(r + 1) * N / size);
        for (int x = s; x < e; ++x) {
            owner[x] = r;
        }
    }

    const bool has_rows = (start_x < end_x);
    const int prev_rank =
        (has_rows && start_x > 0) ? owner[start_x - 1] : MPI_PROC_NULL;
    const int next_rank =
        (has_rows && end_x < N) ? owner[end_x] : MPI_PROC_NULL;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root,
               n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, start_x, end_x);

    if (rank == 0) {
        // Calculate global memory usage (same logical problem as serial)
        const size_t static_mem =
            static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem =
            static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
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

    runSimulation(world, n_iters, prev_rank, next_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    const uint64_t local_hash = computeLocalHash(world);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               MPI_COMM_WORLD);

    std::vector<val_t> all_energy;
    std::vector<val_t> all_flux;
    if (printResults || validate) {
        gatherResults(world, all_energy, all_flux, rank, size);
    }

    int local_rc = 0;

    if (rank == 0) {
        auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter =
            static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Report hash for verification
        printf("  Result hash: %016lX\n", result_hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(all_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(all_energy, all_flux);
            if (!valid) {
                local_rc = 1;
            }
        }
    }

    int global_rc = 0;
    MPI_Allreduce(&local_rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_rc;
}
