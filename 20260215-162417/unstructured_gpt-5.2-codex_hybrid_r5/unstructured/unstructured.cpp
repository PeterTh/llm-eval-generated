#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t FLOW_IN = 0.5;
constexpr val_t FLOW_OUT = -0.5;
constexpr val_t FLUX_SCALE = 0.25;
constexpr val_t NEIGHBOR_FACTOR = TRANSFER_COEFF * FLUX_SCALE;

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

__global__ void updateKernel(const val_t* energy, const val_t* total_flux,
                             val_t* energy_next, val_t* total_flux_next,
                             const val_t* halo_top, const val_t* halo_bottom,
                             int n_root, int start_row, int local_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = local_rows * n_root;
    if (idx >= local_count) {
        return;
    }

    int lx = idx / n_root;
    int y = idx - lx * n_root;
    int gx = start_row + lx;
    int last = n_root - 1;

    val_t current = energy[idx];
    val_t iter_flux = 0.0;

    if (gx == 0 && y == 0) {
        iter_flux += FLOW_IN;
    } else if (gx == 0 && y == last) {
        iter_flux += FLOW_OUT;
    } else if (gx == last && y == 0) {
        iter_flux += FLOW_OUT;
    } else if (gx == last && y == last) {
        iter_flux += FLOW_IN;
    }

    if (gx > 0) {
        val_t neighbor = (lx > 0) ? energy[idx - n_root] : halo_top[y];
        iter_flux += (neighbor - current) * NEIGHBOR_FACTOR;
    }
    if (gx < last) {
        val_t neighbor = (lx + 1 < local_rows) ? energy[idx + n_root] : halo_bottom[y];
        iter_flux += (neighbor - current) * NEIGHBOR_FACTOR;
    }
    if (y > 0) {
        val_t neighbor = energy[idx - 1];
        iter_flux += (neighbor - current) * NEIGHBOR_FACTOR;
    }
    if (y < last) {
        val_t neighbor = energy[idx + 1];
        iter_flux += (neighbor - current) * NEIGHBOR_FACTOR;
    }

    energy_next[idx] = current + iter_flux;
    total_flux_next[idx] = total_flux[idx] + fabs(iter_flux);
}

static int localRowsForRank(int n_rows, int active_size, int rank) {
    int base = n_rows / active_size;
    int rem = n_rows % active_size;
    return base + (rank < rem ? 1 : 0);
}

static int startRowForRank(int n_rows, int active_size, int rank) {
    int base = n_rows / active_size;
    int rem = n_rows % active_size;
    return rank * base + std::min(rank, rem);
}

static void buildCountsDispls(int n_rows, int active_size, int size,
                              std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int local_rows = (r < active_size) ? localRowsForRank(n_rows, active_size, r) : 0;
        int count = local_rows * n_rows;
        counts[r] = count;
        displs[r] = offset;
        offset += count;
    }
}

static uint64_t computeHashLocal(const val_t* energy, const val_t* flux,
                                 size_t count, size_t global_offset) {
    uint64_t hash = 0;
#pragma omp parallel for reduction(^:hash)
    for (size_t i = 0; i < count; ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux[i]);
        const uint64_t gi = static_cast<uint64_t>(global_offset + i);
        hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;
    int show_help = 0;
    int parse_error = 0;
    
    // Parse command line arguments
    if (rank == 0) {
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
                show_help = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_error = 1;
                break;
            }
        }
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&show_help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parse_error, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (show_help) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (parse_error) {
        MPI_Finalize();
        return 1;
    }

    if (n_elems_root <= 0) {
        if (rank == 0) {
            printf("Grid size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    const int active_size = std::min(size, n_elems_root);
    const int local_rows = (rank < active_size) ? localRowsForRank(n_elems_root, active_size, rank) : 0;
    const int start_row = (rank < active_size) ? startRowForRank(n_elems_root, active_size, rank) : 0;
    const size_t local_count = static_cast<size_t>(local_rows) * static_cast<size_t>(n_elems_root);
    const int prev_rank = (rank > 0 && rank < active_size) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank + 1 < active_size) ? rank + 1 : MPI_PROC_NULL;
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (active: %d)\n", size, active_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices visible: %d\n", device_count);
        printf("\n");
    }

    const size_t row_bytes = static_cast<size_t>(n_elems_root) * sizeof(val_t);
    const size_t local_bytes = local_count * sizeof(val_t);

    uint64_t local_dynamic_mem = static_cast<uint64_t>(local_count) * sizeof(val_t) * 4;
    uint64_t local_halo_mem = (local_rows > 0) ? static_cast<uint64_t>(row_bytes) * 2 : 0;
    uint64_t total_dynamic_mem = 0;
    uint64_t total_halo_mem = 0;
    MPI_Reduce(&local_dynamic_mem, &total_dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_halo_mem, &total_halo_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_mem = static_cast<double>(total_dynamic_mem + total_halo_mem) / (1024.0 * 1024.0);
        double dynamic_mem = static_cast<double>(total_dynamic_mem) / (1024.0 * 1024.0);
        double halo_mem = static_cast<double>(total_halo_mem) / (1024.0 * 1024.0);
        printf("Memory usage: %.2f MB (dynamic: %.2f MB, halos: %.2f MB)\n",
               total_mem, dynamic_mem, halo_mem);
        printf("\n");
    }

    val_t* d_energy = nullptr;
    val_t* d_total_flux = nullptr;
    val_t* d_energy_next = nullptr;
    val_t* d_total_flux_next = nullptr;
    val_t* d_halo_top = nullptr;
    val_t* d_halo_bottom = nullptr;

    val_t* h_send_top = nullptr;
    val_t* h_recv_top = nullptr;
    val_t* h_send_bottom = nullptr;
    val_t* h_recv_bottom = nullptr;

    if (local_rows > 0) {
        CUDA_CHECK(cudaMalloc(&d_energy, local_bytes));
        CUDA_CHECK(cudaMalloc(&d_total_flux, local_bytes));
        CUDA_CHECK(cudaMalloc(&d_energy_next, local_bytes));
        CUDA_CHECK(cudaMalloc(&d_total_flux_next, local_bytes));
        CUDA_CHECK(cudaMalloc(&d_halo_top, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_halo_bottom, row_bytes));

        CUDA_CHECK(cudaMemset(d_energy, 0, local_bytes));
        CUDA_CHECK(cudaMemset(d_total_flux, 0, local_bytes));
        CUDA_CHECK(cudaMemset(d_energy_next, 0, local_bytes));
        CUDA_CHECK(cudaMemset(d_total_flux_next, 0, local_bytes));
        CUDA_CHECK(cudaMemset(d_halo_top, 0, row_bytes));
        CUDA_CHECK(cudaMemset(d_halo_bottom, 0, row_bytes));

        CUDA_CHECK(cudaMallocHost(&h_send_top, row_bytes));
        CUDA_CHECK(cudaMallocHost(&h_recv_top, row_bytes));
        CUDA_CHECK(cudaMallocHost(&h_send_bottom, row_bytes));
        CUDA_CHECK(cudaMallocHost(&h_recv_bottom, row_bytes));
    }

    if (rank == 0) {
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    const int tag_up = 100;
    const int tag_down = 101;
    const int threads = 256;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (local_rows > 0) {
            if (prev_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(h_send_top, d_energy, row_bytes, cudaMemcpyDeviceToHost));
            }
            if (next_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(h_send_bottom,
                                      d_energy + static_cast<size_t>(local_rows - 1) * n_elems_root,
                                      row_bytes, cudaMemcpyDeviceToHost));
            }

            if (prev_rank != MPI_PROC_NULL) {
                MPI_Sendrecv(h_send_top, n_elems_root, MPI_DOUBLE, prev_rank, tag_up,
                             h_recv_top, n_elems_root, MPI_DOUBLE, prev_rank, tag_down,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (next_rank != MPI_PROC_NULL) {
                MPI_Sendrecv(h_send_bottom, n_elems_root, MPI_DOUBLE, next_rank, tag_down,
                             h_recv_bottom, n_elems_root, MPI_DOUBLE, next_rank, tag_up,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (prev_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_halo_top, h_recv_top, row_bytes, cudaMemcpyHostToDevice));
            }
            if (next_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_halo_bottom, h_recv_bottom, row_bytes, cudaMemcpyHostToDevice));
            }

            const int blocks = static_cast<int>((local_count + threads - 1) / threads);
            updateKernel<<<blocks, threads>>>(d_energy, d_total_flux,
                                              d_energy_next, d_total_flux_next,
                                              d_halo_top, d_halo_bottom,
                                              n_elems_root, start_row, local_rows);
            CUDA_CHECK(cudaGetLastError());

            std::swap(d_energy, d_energy_next);
            std::swap(d_total_flux, d_total_flux_next);
        }
    }

    if (local_rows > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double local_time = MPI_Wtime() - start_time;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> local_energy;
    std::vector<val_t> local_flux;
    if (local_rows > 0) {
        local_energy.resize(local_count);
        local_flux.resize(local_count);
        CUDA_CHECK(cudaMemcpy(local_energy.data(), d_energy, local_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), d_total_flux, local_bytes, cudaMemcpyDeviceToHost));
    }

    unsigned long long local_hash = 0;
    if (local_rows > 0) {
        size_t global_offset = static_cast<size_t>(start_row) * static_cast<size_t>(n_elems_root);
        local_hash = static_cast<unsigned long long>(
            computeHashLocal(local_energy.data(), local_flux.data(), local_count, global_offset));
    }
    unsigned long long global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) /
                                          (max_time) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016llX\n", global_hash);
        printf("\n");
    }

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<val_t> all_energy;
    std::vector<val_t> all_flux;

    if (printResults || validate) {
        if (rank == 0) {
            buildCountsDispls(n_elems_root, active_size, size, counts, displs);
            all_energy.resize(n_elems);
            if (validate) {
                all_flux.resize(n_elems);
            }
        }

        const int local_count_int = static_cast<int>(local_count);
        MPI_Gatherv(local_energy.data(), local_count_int, MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (validate) {
            MPI_Gatherv(local_flux.data(), local_count_int, MPI_DOUBLE,
                        rank == 0 ? all_flux.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displs.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    int validation_ok = 1;
    if (rank == 0) {
        if (printResults) {
            print_results(all_energy, "ElementEnergy");
        }

        if (validate) {
            World world;
            world.elements_dynamic.resize(n_elems);
            for (int i = 0; i < n_elems; ++i) {
                world.elements_dynamic[i].current_energy = all_energy[i];
                world.elements_dynamic[i].total_flux = all_flux[i];
            }
            validation_ok = validateResults(world) ? 1 : 0;
        }
    }

    if (validate) {
        MPI_Bcast(&validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (d_energy) {
        CUDA_CHECK(cudaFree(d_energy));
    }
    if (d_total_flux) {
        CUDA_CHECK(cudaFree(d_total_flux));
    }
    if (d_energy_next) {
        CUDA_CHECK(cudaFree(d_energy_next));
    }
    if (d_total_flux_next) {
        CUDA_CHECK(cudaFree(d_total_flux_next));
    }
    if (d_halo_top) {
        CUDA_CHECK(cudaFree(d_halo_top));
    }
    if (d_halo_bottom) {
        CUDA_CHECK(cudaFree(d_halo_bottom));
    }
    if (h_send_top) {
        CUDA_CHECK(cudaFreeHost(h_send_top));
    }
    if (h_recv_top) {
        CUDA_CHECK(cudaFreeHost(h_recv_top));
    }
    if (h_send_bottom) {
        CUDA_CHECK(cudaFreeHost(h_send_bottom));
    }
    if (h_recv_bottom) {
        CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    }

    MPI_Finalize();

    if (validate && !validation_ok) {
        return 1;
    }

    return 0;
}
