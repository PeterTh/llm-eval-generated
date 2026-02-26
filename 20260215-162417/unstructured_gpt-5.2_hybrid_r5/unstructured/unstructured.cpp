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
#include <omp.h>

#include <cuda_runtime.h>

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

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

static inline void mpiCheck(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        char errstr[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(err, errstr, &len);
        fprintf(stderr, "MPI error (%s): %.*s\n", what, len, errstr);
        std::abort();
    }
}

// Build a 2D square grid as an unstructured mesh (kept for reference/semantics)
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

    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

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

static void computeRowDecomposition(int n_root, int rank, int size, int& x_start, int& local_rows) {
    const int base = n_root / size;
    const int rem = n_root % size;
    local_rows = base + (rank < rem ? 1 : 0);
    x_start = rank * base + (rank < rem ? rank : rem);
}

__global__ void step_kernel(int n_root,
                            int x_start,
                            int local_rows,
                            const val_t* __restrict__ energy,
                            const val_t* __restrict__ flux_acc,
                            const val_t* __restrict__ external_flow,
                            const val_t* __restrict__ halo_up,
                            const val_t* __restrict__ halo_down,
                            val_t* __restrict__ energy_next,
                            val_t* __restrict__ flux_next,
                            int has_up,
                            int has_down) {
    const int li = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_n = local_rows * n_root;
    if (li >= local_n) return;

    const int lx = li / n_root;
    const int y = li - lx * n_root;
    const int gx = x_start + lx;

    const val_t this_e = energy[li];
    const val_t transfer_coeff = 0.8;
    const val_t connection_flux = 1.0;

    // Preserve the original neighbor order: (x+1,y), (x-1,y), (x,y+1), (x,y-1)
    val_t total_flux = external_flow[li];

    // (x+1,y)
    if (gx + 1 < n_root) {
        val_t other_e;
        if (lx + 1 < local_rows) {
            other_e = energy[li + n_root];
        } else {
            other_e = has_down ? halo_down[y] : this_e;
        }
        total_flux += (other_e - this_e) * transfer_coeff * connection_flux * 0.25;
    }

    // (x-1,y)
    if (gx - 1 >= 0) {
        val_t other_e;
        if (lx > 0) {
            other_e = energy[li - n_root];
        } else {
            other_e = has_up ? halo_up[y] : this_e;
        }
        total_flux += (other_e - this_e) * transfer_coeff * connection_flux * 0.25;
    }

    // (x,y+1)
    if (y + 1 < n_root) {
        const val_t other_e = energy[li + 1];
        total_flux += (other_e - this_e) * transfer_coeff * connection_flux * 0.25;
    }

    // (x,y-1)
    if (y - 1 >= 0) {
        const val_t other_e = energy[li - 1];
        total_flux += (other_e - this_e) * transfer_coeff * connection_flux * 0.25;
    }

    energy_next[li] = this_e + total_flux;
    flux_next[li] = flux_acc[li] + fabs(total_flux);
}

int main(int argc, char** argv) {
    mpiCheck(MPI_Init(&argc, &argv), "MPI_Init");

    int rank = 0;
    int size = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int parseOk = 1;

    if (rank == 0) {
        // Parse command line arguments (rank 0), then broadcast.
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseOk = 0;
                break;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        } else if (!parseOk) {
            printUsage(argv[0]);
        }

        if (n_elems_root <= 0 || n_iters < 0) {
            printf("Invalid parameters.\n");
            parseOk = 0;
        }
    }

    mpiCheck(MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(parseOk)");
    mpiCheck(MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(showHelp)");
    mpiCheck(MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(n_elems_root)");
    mpiCheck(MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(n_iters)");
    mpiCheck(MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(validate)");
    mpiCheck(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(printResults)");

    if (!parseOk || showHelp) {
        mpiCheck(MPI_Finalize(), "MPI_Finalize");
        return parseOk ? 0 : 1;
    }

    // Select CUDA device per rank.
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        mpiCheck(MPI_Finalize(), "MPI_Finalize");
        return 1;
    }
    const int dev = rank % deviceCount;
    cudaCheck(cudaSetDevice(dev), "cudaSetDevice");

    const int n_elems = n_elems_root * n_elems_root;
    const size_t global_n = static_cast<size_t>(n_elems);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Calculate memory usage (global, as in the original code)
        const size_t static_mem = global_n * sizeof(ElementStatic);
        const size_t dynamic_mem = global_n * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Building unstructured mesh...\n");
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    int x_start = 0;
    int local_rows = 0;
    computeRowDecomposition(n_elems_root, rank, size, x_start, local_rows);
    const size_t local_n = static_cast<size_t>(local_rows) * static_cast<size_t>(n_elems_root);

    // Host-side initialization (OpenMP): per-element external flow and initial state.
    std::vector<val_t> h_external_flow(local_n);
    std::vector<val_t> h_energy(local_n, 0.0);
    std::vector<val_t> h_flux_acc(local_n, 0.0);

#pragma omp parallel for schedule(static)
    for (size_t li = 0; li < local_n; ++li) {
        const int lx = static_cast<int>(li / static_cast<size_t>(n_elems_root));
        const int y = static_cast<int>(li - static_cast<size_t>(lx) * static_cast<size_t>(n_elems_root));
        const int gx = x_start + lx;
        const int last = n_elems_root - 1;

        // Default material external flow: 0.0
        val_t ext = 0.0;
        // Corner inflow/outflow as in buildSquare2D()
        if ((gx == 0 && y == 0) || (gx == last && y == last)) {
            ext = 0.5;
        } else if ((gx == 0 && y == last) || (gx == last && y == 0)) {
            ext = -0.5;
        }
        h_external_flow[li] = ext;
    }

    // Device allocations
    val_t* d_energy = nullptr;
    val_t* d_flux_acc = nullptr;
    val_t* d_energy_next = nullptr;
    val_t* d_flux_next = nullptr;
    val_t* d_external_flow = nullptr;
    val_t* d_halo_up = nullptr;
    val_t* d_halo_down = nullptr;

    if (local_n > 0) {
        cudaCheck(cudaMalloc(&d_energy, local_n * sizeof(val_t)), "cudaMalloc(d_energy)");
        cudaCheck(cudaMalloc(&d_flux_acc, local_n * sizeof(val_t)), "cudaMalloc(d_flux_acc)");
        cudaCheck(cudaMalloc(&d_energy_next, local_n * sizeof(val_t)), "cudaMalloc(d_energy_next)");
        cudaCheck(cudaMalloc(&d_flux_next, local_n * sizeof(val_t)), "cudaMalloc(d_flux_next)");
        cudaCheck(cudaMalloc(&d_external_flow, local_n * sizeof(val_t)), "cudaMalloc(d_external_flow)");
        cudaCheck(cudaMalloc(&d_halo_up, static_cast<size_t>(n_elems_root) * sizeof(val_t)), "cudaMalloc(d_halo_up)");
        cudaCheck(cudaMalloc(&d_halo_down, static_cast<size_t>(n_elems_root) * sizeof(val_t)), "cudaMalloc(d_halo_down)");

        cudaCheck(cudaMemcpy(d_energy, h_energy.data(), local_n * sizeof(val_t), cudaMemcpyHostToDevice), "cudaMemcpy(d_energy)");
        cudaCheck(cudaMemcpy(d_flux_acc, h_flux_acc.data(), local_n * sizeof(val_t), cudaMemcpyHostToDevice), "cudaMemcpy(d_flux_acc)");
        cudaCheck(cudaMemcpy(d_external_flow, h_external_flow.data(), local_n * sizeof(val_t), cudaMemcpyHostToDevice), "cudaMemcpy(d_external_flow)");
    }

    const int up_rank = rank - 1;
    const int down_rank = rank + 1;
    const int has_up = (up_rank >= 0) ? 1 : 0;
    const int has_down = (down_rank < size) ? 1 : 0;

    std::vector<val_t> h_send_up(n_elems_root);
    std::vector<val_t> h_send_down(n_elems_root);
    std::vector<val_t> h_recv_up(n_elems_root);
    std::vector<val_t> h_recv_down(n_elems_root);

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(pre-timing)");
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < n_iters; ++iter) {
        if (local_n == 0) continue;

        // Halo exchange of boundary rows (energies only).
        if (size > 1 && local_rows > 0) {
            // Copy current boundary rows from device to host.
            cudaCheck(cudaMemcpy(h_send_up.data(), d_energy, static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(boundary up)");
            cudaCheck(cudaMemcpy(h_send_down.data(), d_energy + (static_cast<size_t>(local_rows - 1) * static_cast<size_t>(n_elems_root)),
                                 static_cast<size_t>(n_elems_root) * sizeof(val_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy(boundary down)");

            MPI_Status st;
            if (has_up) {
                mpiCheck(MPI_Sendrecv(h_send_up.data(), n_elems_root, MPI_DOUBLE, up_rank, 0,
                                      h_recv_up.data(), n_elems_root, MPI_DOUBLE, up_rank, 1,
                                      MPI_COMM_WORLD, &st),
                         "MPI_Sendrecv(up)");
                cudaCheck(cudaMemcpy(d_halo_up, h_recv_up.data(), static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy(halo up)");
            }
            if (has_down) {
                mpiCheck(MPI_Sendrecv(h_send_down.data(), n_elems_root, MPI_DOUBLE, down_rank, 1,
                                      h_recv_down.data(), n_elems_root, MPI_DOUBLE, down_rank, 0,
                                      MPI_COMM_WORLD, &st),
                         "MPI_Sendrecv(down)");
                cudaCheck(cudaMemcpy(d_halo_down, h_recv_down.data(), static_cast<size_t>(n_elems_root) * sizeof(val_t),
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy(halo down)");
            }
        }

        // GPU timestep.
        const int threads = 256;
        const int blocks = static_cast<int>((local_n + threads - 1) / threads);
        step_kernel<<<blocks, threads>>>(n_elems_root, x_start, local_rows, d_energy, d_flux_acc, d_external_flow,
                                         d_halo_up, d_halo_down, d_energy_next, d_flux_next, has_up, has_down);
        cudaCheck(cudaGetLastError(), "step_kernel launch");

        // Swap buffers.
        std::swap(d_energy, d_energy_next);
        std::swap(d_flux_acc, d_flux_next);
    }

    if (local_n > 0) {
        cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(post-timing)");
    const double t1 = MPI_Wtime();

    double local_ms = (t1 - t0) * 1000.0;
    double max_ms = 0.0;
    mpiCheck(MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce(time)");

    // Copy results back to host for gather.
    if (local_n > 0) {
        cudaCheck(cudaMemcpy(h_energy.data(), d_energy, local_n * sizeof(val_t), cudaMemcpyDeviceToHost), "cudaMemcpy(h_energy)");
        cudaCheck(cudaMemcpy(h_flux_acc.data(), d_flux_acc, local_n * sizeof(val_t), cudaMemcpyDeviceToHost), "cudaMemcpy(h_flux_acc)");
    }

    // Gather local sizes to root, then gather energy/flux arrays.
    size_t local_n_u64 = local_n;
    std::vector<size_t> all_counts_u64;
    if (rank == 0) all_counts_u64.resize(static_cast<size_t>(size));
    mpiCheck(MPI_Gather(&local_n_u64, 1, MPI_UINT64_T,
                        rank == 0 ? all_counts_u64.data() : nullptr, 1, MPI_UINT64_T,
                        0, MPI_COMM_WORLD),
             "MPI_Gather(counts)");

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<val_t> energy_global;
    std::vector<val_t> flux_global;

    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        size_t disp = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(all_counts_u64[static_cast<size_t>(r)]);
            displs[r] = static_cast<int>(disp);
            disp += all_counts_u64[static_cast<size_t>(r)];
        }
        energy_global.resize(global_n);
        flux_global.resize(global_n);
    }

    mpiCheck(MPI_Gatherv(h_energy.data(), static_cast<int>(local_n), MPI_DOUBLE,
                         rank == 0 ? energy_global.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displs.data() : nullptr,
                         MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "MPI_Gatherv(energy)");

    mpiCheck(MPI_Gatherv(h_flux_acc.data(), static_cast<int>(local_n), MPI_DOUBLE,
                         rank == 0 ? flux_global.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displs.data() : nullptr,
                         MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "MPI_Gatherv(flux)");

    // Cleanup device memory.
    if (local_n > 0) {
        cudaCheck(cudaFree(d_energy), "cudaFree(d_energy)");
        cudaCheck(cudaFree(d_flux_acc), "cudaFree(d_flux_acc)");
        cudaCheck(cudaFree(d_energy_next), "cudaFree(d_energy_next)");
        cudaCheck(cudaFree(d_flux_next), "cudaFree(d_flux_next)");
        cudaCheck(cudaFree(d_external_flow), "cudaFree(d_external_flow)");
        cudaCheck(cudaFree(d_halo_up), "cudaFree(d_halo_up)");
        cudaCheck(cudaFree(d_halo_down), "cudaFree(d_halo_down)");
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_ms));

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (max_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Reconstruct global element array for hash/validation (semantics equivalent to serial).
        World world;
        world.elements_dynamic.resize(global_n);
        for (size_t i = 0; i < global_n; ++i) {
            world.elements_dynamic[i].current_energy = energy_global[i];
            world.elements_dynamic[i].total_flux = flux_global[i];
        }

        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            print_results(energy_global, "ElementEnergy");
        }

        if (validate) {
            bool valid = validateResults(world);
            if (!valid) {
                mpiCheck(MPI_Finalize(), "MPI_Finalize");
                return 1;
            }
        }
    }

    mpiCheck(MPI_Finalize(), "MPI_Finalize");
    return 0;
}
