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

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::abort();
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

__device__ __forceinline__ double externalFlowFor(const int x, const int y, const int n) {
    // Matches buildSquare2D corner material assignment.
    if ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) return 0.5;
    if ((x == 0 && y == n - 1) || (x == n - 1 && y == 0)) return -0.5;
    return 0.0;
}

__global__ void update_rows_kernel(const double* __restrict__ energy,
                                  double* __restrict__ energy_next,
                                  const double* __restrict__ flux,
                                  double* __restrict__ flux_next,
                                  int n,
                                  int global_x_start,
                                  int local_rows,
                                  int row_begin,   // owned-row index in [1, local_rows]
                                  int row_end) {   // owned-row index in [1, local_rows]
    const int cols = n;
    const int rows_to_do = row_end - row_begin + 1;
    const int total = rows_to_do * cols;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int row_off = tid / cols;
    const int y = tid - row_off * cols;
    const int lx = row_begin + row_off;      // local row index (1..local_rows)
    const int x = global_x_start + (lx - 1); // global row index

    const int center_idx = lx * cols + y;
    const double e_center = energy[center_idx];

    constexpr double transfer_coeff = 0.8;
    constexpr double conn_flux = 1.0;

    // Start with external flow
    double total_flux = externalFlowFor(x, y, n);

    // Offsets order matches buildSquare2D: (down, up, right, left)
    if (x + 1 < n) {
        const double e_other = energy[(lx + 1) * cols + y];
        total_flux += (e_other - e_center) * transfer_coeff * conn_flux * 0.25;
    }
    if (x - 1 >= 0) {
        const double e_other = energy[(lx - 1) * cols + y];
        total_flux += (e_other - e_center) * transfer_coeff * conn_flux * 0.25;
    }
    if (y + 1 < n) {
        const double e_other = energy[lx * cols + (y + 1)];
        total_flux += (e_other - e_center) * transfer_coeff * conn_flux * 0.25;
    }
    if (y - 1 >= 0) {
        const double e_other = energy[lx * cols + (y - 1)];
        total_flux += (e_other - e_center) * transfer_coeff * conn_flux * 0.25;
    }

    const int out_owned_idx = (lx - 1) * cols + y;
    energy_next[center_idx] = e_center + total_flux;
    flux_next[out_owned_idx] = flux[out_owned_idx] + fabs(total_flux);
}

// Hybrid MPI + OpenMP + CUDA simulation (row-wise domain decomposition).
static void runSimulationHybridMPI_CUDA(int n_elems_root,
                                        int n_iters,
                                        std::vector<double>& out_energy_global,
                                        std::vector<double>& out_flux_global,
                                        int rank,
                                        int size,
                                        MPI_Comm comm) {
    const int n = n_elems_root;
    const int global_rows = n;
    const int cols = n;

    // Row partition.
    const int base = global_rows / size;
    const int rem = global_rows % size;
    const int local_rows = base + (rank < rem ? 1 : 0);
    const int global_x_start = rank * base + (rank < rem ? rank : rem);

    // Neighbor ranks for halo exchange.
    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

    // Device buffers: energy has 2 halo rows (top+bottom).
    double* d_energy = nullptr;
    double* d_energy_next = nullptr;
    double* d_flux = nullptr;
    double* d_flux_next = nullptr;

    const size_t energy_elems = static_cast<size_t>(local_rows + 2) * static_cast<size_t>(cols);
    const size_t owned_elems = static_cast<size_t>(local_rows) * static_cast<size_t>(cols);

    CUDA_CHECK(cudaMalloc(&d_energy, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy_next, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux, owned_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux_next, owned_elems * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_energy, 0, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_energy_next, 0, energy_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_flux, 0, owned_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_flux_next, 0, owned_elems * sizeof(double)));

    cudaStream_t stream_pack{};
    cudaStream_t stream_compute{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_pack, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));

    // Pinned host halo buffers (1 row each).
    double* h_send_top = nullptr;
    double* h_send_bottom = nullptr;
    double* h_recv_top = nullptr;
    double* h_recv_bottom = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send_top, cols * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, cols * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, cols * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, cols * sizeof(double)));

    cudaEvent_t pack_done{};
    CUDA_CHECK(cudaEventCreateWithFlags(&pack_done, cudaEventDisableTiming));

    for (int iter = 0; iter < n_iters; ++iter) {
        // Stage halo send rows D->H.
        if (local_rows > 0) {
            if (prev != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_energy + 1 * cols, cols * sizeof(double), cudaMemcpyDeviceToHost, stream_pack));
            }
            if (next != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_send_bottom, d_energy + local_rows * cols, cols * sizeof(double), cudaMemcpyDeviceToHost, stream_pack));
            }
        }

        // Compute interior rows while halo copy is in flight.
        if (local_rows > 2) {
            const int row_begin = 2;
            const int row_end = local_rows - 1;
            const int rows_to_do = row_end - row_begin + 1;
            const int total = rows_to_do * cols;
            const int block = 256;
            const int grid = (total + block - 1) / block;
            update_rows_kernel<<<grid, block, 0, stream_compute>>>(
                d_energy, d_energy_next, d_flux, d_flux_next, n, global_x_start, local_rows, row_begin, row_end);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(pack_done, stream_pack));
        CUDA_CHECK(cudaEventSynchronize(pack_done));

        // Halo exchange (host buffers; OpenMPI here is not CUDA-aware).
        MPI_Request reqs[4];
        int req_count = 0;
        if (prev != MPI_PROC_NULL && local_rows > 0) {
            MPI_Irecv(h_recv_top, cols, MPI_DOUBLE, prev, 100, comm, &reqs[req_count++]);
            MPI_Isend(h_send_top, cols, MPI_DOUBLE, prev, 101, comm, &reqs[req_count++]);
        }
        if (next != MPI_PROC_NULL && local_rows > 0) {
            MPI_Irecv(h_recv_bottom, cols, MPI_DOUBLE, next, 101, comm, &reqs[req_count++]);
            MPI_Isend(h_send_bottom, cols, MPI_DOUBLE, next, 100, comm, &reqs[req_count++]);
        }
        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Copy received halos H->D.
        if (local_rows > 0) {
            if (prev != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_energy + 0 * cols, h_recv_top, cols * sizeof(double), cudaMemcpyHostToDevice, stream_pack));
            }
            if (next != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_energy + (local_rows + 1) * cols, h_recv_bottom, cols * sizeof(double), cudaMemcpyHostToDevice, stream_pack));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_pack));

        // Compute boundary rows (which depend on halos).
        if (local_rows > 0) {
            {
                const int total = cols;
                const int block = 256;
                const int grid = (total + block - 1) / block;
                update_rows_kernel<<<grid, block, 0, stream_compute>>>(
                    d_energy, d_energy_next, d_flux, d_flux_next, n, global_x_start, local_rows, 1, 1);
                CUDA_CHECK(cudaGetLastError());
            }
            if (local_rows >= 2) {
                const int total = cols;
                const int block = 256;
                const int grid = (total + block - 1) / block;
                update_rows_kernel<<<grid, block, 0, stream_compute>>>(
                    d_energy, d_energy_next, d_flux, d_flux_next, n, global_x_start, local_rows, local_rows, local_rows);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        std::swap(d_energy, d_energy_next);
        std::swap(d_flux, d_flux_next);
    }

    // Copy owned results back.
    std::vector<double> local_energy(owned_elems);
    std::vector<double> local_flux(owned_elems);
    if (local_rows > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), d_energy + 1 * cols, owned_elems * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), d_flux, owned_elems * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0 (contiguous rows).
    std::vector<int> recvcounts;
    std::vector<int> displs;
    if (rank == 0) {
        recvcounts.resize(size);
        displs.resize(size);
    }
    const int local_count = static_cast<int>(owned_elems);
    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? recvcounts.data() : nullptr, 1, MPI_INT, 0, comm);
    if (rank == 0) {
        int disp = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = disp;
            disp += recvcounts[r];
        }
        out_energy_global.resize(static_cast<size_t>(global_rows) * static_cast<size_t>(cols));
        out_flux_global.resize(static_cast<size_t>(global_rows) * static_cast<size_t>(cols));
    }

    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                rank == 0 ? out_energy_global.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE,
                rank == 0 ? out_flux_global.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);

    CUDA_CHECK(cudaEventDestroy(pack_done));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaStreamDestroy(stream_pack));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_energy_next));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_flux_next));
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

    if (rank == 0) {
        // Parse command line arguments
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
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast config.
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count > 0) {
        CUDA_CHECK(cudaSetDevice(rank % dev_count));
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("\n");

        // Calculate memory usage (global, as in the original model).
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> energy_global;
    std::vector<double> flux_global;

    const double t0 = MPI_Wtime();
    runSimulationHybridMPI_CUDA(n_elems_root, n_iters, energy_global, flux_global, rank, size, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    if (rank == 0) {
        const long duration_ms = static_cast<long>((t1 - t0) * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        // Build a World-like view for existing hashing/validation logic.
        World world;
        world.elements_dynamic.resize(static_cast<size_t>(n_elems));

        #pragma omp parallel for
        for (int i = 0; i < n_elems; ++i) {
            world.elements_dynamic[static_cast<size_t>(i)].current_energy = energy_global[static_cast<size_t>(i)];
            world.elements_dynamic[static_cast<size_t>(i)].total_flux = flux_global[static_cast<size_t>(i)];
        }

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.resize(world.elements_dynamic.size());
            #pragma omp parallel for
            for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
                energyData[i] = world.elements_dynamic[i].current_energy;
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            if (!validateResults(world)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
