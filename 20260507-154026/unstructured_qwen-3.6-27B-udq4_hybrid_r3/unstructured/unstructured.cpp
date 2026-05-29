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

// ---------------------------------------------------------------------------
// MPI domain decomposition helpers
// ---------------------------------------------------------------------------
struct DomainDecomp {
    int rank;
    int size;
    int n_root;            // global grid size (NxN)
    int local_x_start;     // inclusive
    int local_x_end;       // exclusive
    int local_y_start;     // inclusive
    int local_y_end;       // exclusive
    int halo_width;        // always 1 for 4-neighbor stencil
};

// Split n_root into `size` blocks as evenly as possible along the X axis.
// Each block gets either floor(n_root/size) or ceil(n_root/size) columns.
static void compute_decomposition(DomainDecomp& decomp, int n_root, int rank, int size) {
    decomp.rank = rank;
    decomp.size = size;
    decomp.n_root = n_root;
    decomp.halo_width = 1;

    int base_width = n_root / size;
    int remainder = n_root % size;
    int x_start = rank * base_width + std::min(rank, remainder);
    int x_end = (rank + 1) * base_width + std::min(rank + 1, remainder);

    decomp.local_x_start = x_start;
    decomp.local_x_end = x_end;
    decomp.local_y_start = 0;
    decomp.local_y_end = n_root;
}

// ---------------------------------------------------------------------------
// Build a 2D square grid as an unstructured mesh (OpenMP parallel)
// ---------------------------------------------------------------------------
void buildSquare2D(World& world, const int n_elems_root, const DomainDecomp& decomp) {

    // All ranks need materials (CUDA kernel reads from device copy)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Compute local element count: owned cells + halo cells
    int lx_start = decomp.local_x_start;
    int lx_end   = decomp.local_x_end;
    int ly_start = decomp.local_y_start;
    int ly_end   = decomp.local_y_end;

    // Expand by halo_width for ghost cells
    int lx_start_halo = std::max(0, lx_start - decomp.halo_width);
    int lx_end_halo   = std::min(n_elems_root, lx_end + decomp.halo_width);
    int ly_start_halo = std::max(0, ly_start - decomp.halo_width);
    int ly_end_halo   = std::min(n_elems_root, ly_end + decomp.halo_width);

    int local_nx = lx_end_halo - lx_start_halo;
    int local_ny = ly_end_halo - ly_start_halo;
    int local_n  = local_nx * local_ny;

    world.elements_static.resize(local_n);
    world.elements_dynamic.resize(local_n);
    world.elements_dynamic_swap.resize(local_n);

    // Local index from global (x,y) within halo-expanded region
    auto lidx = [&](int x, int y) {
        return (x - lx_start_halo) * local_ny + (y - ly_start_halo);
    };

    // Initialize all elements with default material and zero energy
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_n; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each owned element connects to its neighbors
    // (including halo neighbors — they are present in our local arrays)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = lx_start; x < lx_end; ++x) {
        for (int y = ly_start; y < ly_end; ++y) {
            int li = lidx(x, y);
            ElementStatic& elem = world.elements_static[li];

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                int nx = x + offsets[n][0];
                int ny = y + offsets[n][1];
                if (nx >= lx_start_halo && nx < lx_end_halo &&
                    ny >= ly_start_halo && ny < ly_end_halo) {
                    elem.connected_idx[elem.num_connections] = lidx(nx, ny);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow (only on ranks that own them)
    const int last = n_elems_root - 1;
    auto set_material = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= lx_start && gx < lx_end && gy >= ly_start && gy < ly_end) {
            world.elements_static[lidx(gx, gy)].material_idx = mat_id;
        }
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);

 }

// ---------------------------------------------------------------------------
// CUDA kernel – energy transfer step
// ---------------------------------------------------------------------------
// We pass flat arrays; each thread handles one owned element.
// The kernel reads from elements_dynamic (source) and writes to elements_dynamic_swap (dest).
// Halo cells are NOT updated by the kernel (they are filled by MPI exchange).
// ---------------------------------------------------------------------------
__device__ inline val_t d_computeFlux(val_t transfer_coeff, val_t this_energy,
                                      val_t conn_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * conn_flux * 0.25;
}

__global__ void simulationStepKernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const Material* __restrict__ materials,
    int num_owned,
    int owned_offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_owned) return;

    int li = i + owned_offset;

    const ElementStatic& es = elements_static[li];
    const ElementDynamic& ed = elements_dynamic[li];
    const Material& mat = materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    for (int j = 0; j < es.num_connections; ++j) {
        int nj = es.connected_idx[j];
        const ElementDynamic& neighbor = elements_dynamic[nj];
        total_flux += d_computeFlux(mat.transfer_coeff, ed.current_energy,
                                    es.connected_flux[j], neighbor.current_energy);
    }

    ElementDynamic& ew = elements_dynamic_swap[li];
    ew.current_energy = ed.current_energy + total_flux;
    ew.total_flux     = ed.total_flux + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// MPI halo exchange
// ---------------------------------------------------------------------------
static void exchange_halos(World& world, const DomainDecomp& decomp) {
    int n_root = decomp.n_root;
    int lx_start = decomp.local_x_start;
    int lx_end   = decomp.local_x_end;
    int ly_start = decomp.local_y_start;
    int ly_end   = decomp.local_y_end;

    int lx_start_halo = std::max(0, lx_start - 1);
    int lx_end_halo   = std::min(n_root, lx_end + 1);
    int ly_start_halo = std::max(0, ly_start - 1);
    int ly_end_halo   = std::min(n_root, ly_end + 1);

    int local_ny = ly_end_halo - ly_start_halo;

    auto lidx = [&](int x, int y) {
        return (x - lx_start_halo) * local_ny + (y - ly_start_halo);
    };

    // Determine neighbor ranks: we decompose along X, so neighbors are rank-1 and rank+1.
    // Communication is along the Y-direction of the local array (columns at boundaries).

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = decomp.rank;

    // Tags for send/recv
    const int TAG_SEND = 100;
    const int TAG_RECV = 200;

    // Left halo (column lx_start_halo if it exists and is a halo column)
    // Right halo (column lx_end_halo - 1 if it exists and is a halo column)
    // Each halo column has local_ny elements = 2 doubles each

    bool has_left_halo  = (lx_start > 0);
    bool has_right_halo = (lx_end < n_root);

    int halo_elems = local_ny;  // number of elements in each halo column
    int halo_bytes = halo_elems * sizeof(ElementDynamic);

    // Prepare send/recv buffers
    std::vector<ElementDynamic> send_left_buf(halo_elems);
    std::vector<ElementDynamic> send_right_buf(halo_elems);
    std::vector<ElementDynamic> recv_left_buf(halo_elems);
    std::vector<ElementDynamic> recv_right_buf(halo_elems);

    MPI_Request requests[4];
    int request_count = 0;

    // --- Right boundary: send our right-most owned column to rank+1, recv right halo ---
    if (has_right_halo) {
        // Send: our right-most owned column (x = lx_end - 1) → rank+1
        int right_rank = rank + 1;
        for (int k = 0; k < halo_elems; ++k) {
            send_right_buf[k] = world.elements_dynamic[lidx(lx_end - 1, ly_start_halo + k)];
        }
        MPI_Isend(send_right_buf.data(), halo_bytes, MPI_BYTE, right_rank, TAG_SEND, comm, &requests[request_count++]);

        // Recv: into our right halo column (x = lx_end)
        MPI_Irecv(recv_right_buf.data(), halo_bytes, MPI_BYTE, right_rank, TAG_RECV, comm, &requests[request_count++]);
    }

    // --- Left boundary: send our left-most owned column to rank-1, recv left halo ---
    if (has_left_halo) {
        // Send: our left-most owned column (x = lx_start) → rank-1
        int left_rank = rank - 1;
        for (int k = 0; k < halo_elems; ++k) {
            send_left_buf[k] = world.elements_dynamic[lidx(lx_start, ly_start_halo + k)];
        }
        MPI_Isend(send_left_buf.data(), halo_bytes, MPI_BYTE, left_rank, TAG_RECV, comm, &requests[request_count++]);

        // Recv: into our left halo column (x = lx_start - 1)
        MPI_Irecv(recv_left_buf.data(), halo_bytes, MPI_BYTE, left_rank, TAG_SEND, comm, &requests[request_count++]);
    }

    // Wait for all communications
    if (request_count > 0) {
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    }

    // Scatter received data into halo columns
    if (has_right_halo) {
        for (int k = 0; k < halo_elems; ++k) {
            world.elements_dynamic[lidx(lx_end, ly_start_halo + k)] = recv_right_buf[k];
        }
    }
    if (has_left_halo) {
        for (int k = 0; k < halo_elems; ++k) {
            world.elements_dynamic[lidx(lx_start_halo, ly_start_halo + k)] = recv_left_buf[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Run simulation for n_iters iterations (CUDA + MPI)
// ---------------------------------------------------------------------------
void runSimulation(World& world, const int n_iters, const DomainDecomp& decomp) {
    int n_root = decomp.n_root;
    int lx_start = decomp.local_x_start;
    int lx_end   = decomp.local_x_end;
    int ly_start = decomp.local_y_start;
    int ly_end   = decomp.local_y_end;

    int lx_start_halo = std::max(0, lx_start - 1);
    int lx_end_halo   = std::min(n_root, lx_end + 1);

    int local_nx = lx_end_halo - lx_start_halo;
    int local_ny = ly_end - ly_start;  // = n_root for this decomposition
    int local_n  = local_nx * local_ny;

    // Number of owned (non-halo) elements
    int owned_nx = lx_end - lx_start;
    int num_owned = owned_nx * local_ny;

    // --- Pin host memory for faster PCIe transfers ---
    ElementStatic*  h_static  = world.elements_static.data();
    ElementDynamic* h_dyn     = world.elements_dynamic.data();
    ElementDynamic* h_dyn_sw  = world.elements_dynamic_swap.data();

    cudaHostRegister(h_static,  local_n  * sizeof(ElementStatic),  cudaHostRegisterDefault);
    cudaHostRegister(h_dyn,     local_n  * sizeof(ElementDynamic), cudaHostRegisterDefault);
    cudaHostRegister(h_dyn_sw,  local_n  * sizeof(ElementDynamic), cudaHostRegisterDefault);

    // --- Allocate device memory ---
    ElementStatic*  d_static  = nullptr;
    ElementDynamic* d_dyn     = nullptr;
    ElementDynamic* d_dyn_sw  = nullptr;
    Material*       d_mat     = nullptr;

    cudaMalloc(&d_static,  local_n  * sizeof(ElementStatic));
    cudaMalloc(&d_dyn,     local_n  * sizeof(ElementDynamic));
    cudaMalloc(&d_dyn_sw,  local_n  * sizeof(ElementDynamic));
    cudaMalloc(&d_mat,     3 * sizeof(Material));

    // Upload static data (connectivity doesn't change)
    cudaMemcpy(d_static, h_static, local_n * sizeof(ElementStatic), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat,    world.materials.data(), 3 * sizeof(Material), cudaMemcpyHostToDevice);

    // Upload initial dynamic state
    cudaMemcpy(d_dyn, h_dyn, local_n * sizeof(ElementDynamic), cudaMemcpyHostToDevice);

    // CUDA stream for async transfers
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Kernel launch config
    int block_size = 256;
    int grid_size  = (num_owned + block_size - 1) / block_size;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch CUDA kernel for owned elements
        // owned_offset = number of halo columns on the left * local_ny
        int owned_offset = (lx_start - lx_start_halo) * local_ny;
        simulationStepKernel<<<grid_size, block_size, 0, stream>>>(
            d_static, d_dyn, d_dyn_sw, d_mat, num_owned, owned_offset);

        // Synchronize after kernel
        cudaStreamSynchronize(stream);

        // Swap device pointers (avoids data copy)
        std::swap(d_dyn, d_dyn_sw);

        // Download current state to pinned host memory for halo exchange
        cudaMemcpyAsync(h_dyn, d_dyn, local_n * sizeof(ElementDynamic),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // MPI halo exchange: communicate new boundary values so neighbors
        // have fresh halo data for the next iteration
        exchange_halos(world, decomp);

        // Upload updated host data (with fresh halos) back to device
        cudaMemcpyAsync(d_dyn, h_dyn, local_n * sizeof(ElementDynamic),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    // Final download
    cudaMemcpy(h_dyn, d_dyn, local_n * sizeof(ElementDynamic), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_static);
    cudaFree(d_dyn);
    cudaFree(d_dyn_sw);
    cudaFree(d_mat);

    cudaHostUnregister(h_static);
    cudaHostUnregister(h_dyn);
    cudaHostUnregister(h_dyn_sw);
}

// ---------------------------------------------------------------------------
// Validate simulation results (OpenMP parallel reduction)
// ---------------------------------------------------------------------------
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum   = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:local_energy_sum,local_flux_sum) \
                             reduction(max:local_energy_max) \
                             reduction(min:local_energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        local_energy_sum += world.elements_dynamic[i].current_energy;
        local_flux_sum   += world.elements_dynamic[i].total_flux;
        local_energy_max  = std::max(world.elements_dynamic[i].current_energy, local_energy_max);
        local_energy_min  = std::min(world.elements_dynamic[i].current_energy, local_energy_min);
    }

    // Reduce across MPI ranks
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum,   &flux_sum,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Compute a simple hash of the results for verification (MPI gather + hash)
// ---------------------------------------------------------------------------
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
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
    // Initialize MPI first
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    // Select CUDA device per MPI rank (round-robin across available GPUs)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int gpu_id = rank % std::max(num_gpus, 1);
    cudaSetDevice(gpu_id);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d | GPUs detected: %d | OpenMP threads: %d\n",
               size, num_gpus, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Domain decomposition
    DomainDecomp decomp;
    compute_decomposition(decomp, n_elems_root, rank, size);

    // Build the unstructured mesh (OpenMP parallelized)
    World world;
    buildSquare2D(world, n_elems_root, decomp);

    // Calculate memory usage (per rank)
    const size_t local_n = world.elements_static.size();
    const size_t static_mem = local_n * sizeof(ElementStatic);
    const size_t dynamic_mem = local_n * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;

    if (rank == 0) {
        printf("Memory per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, decomp);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather timing info (use Allreduce so rank 0 gets the max)
    long long duration_ms_all;
    duration_ms_all = duration_ms;
    MPI_Allreduce(&duration_ms_all, &duration_ms_all, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
    duration_ms = duration_ms_all;

    // Calculate performance metrics (on rank 0)
    if (rank == 0) {
        printf("Computation time: %lld ms\n", (long long)duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Gather all element data to rank 0 for hash computation
    // We need global-ordered data for a deterministic hash.
    std::vector<ElementDynamic> global_elements;
    if (rank == 0) {
        global_elements.resize(n_elems);
    }

    // Each rank sends its owned (non-halo) elements to rank 0 in global order
    int lx_start = decomp.local_x_start;
    int lx_end   = decomp.local_x_end;
    int n_root   = decomp.n_root;
    int lx_start_halo = std::max(0, lx_start - 1);
    int local_ny = n_root;
    int owned_nx = lx_end - lx_start;
    int num_owned = owned_nx * local_ny;

    auto lidx = [&](int x, int y) {
        return (x - lx_start_halo) * local_ny + (y - 0);
    };

    // Pack owned elements in global order for this rank
    std::vector<ElementDynamic> owned_data(num_owned);
    for (int x = lx_start; x < lx_end; ++x) {
        for (int y = 0; y < n_root; ++y) {
            int global_idx = x * n_root + y;
            int local_idx  = lidx(x, y);
            owned_data[global_idx - lx_start * n_root] = world.elements_dynamic[local_idx];
        }
    }

    // Compute displacements and counts for MPI_Gatherv
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    {
        int base_width = n_elems_root / size;
        int remainder  = n_elems_root % size;
        int cur_disp   = 0;
        for (int r = 0; r < size; ++r) {
            int r_width = base_width + (r < remainder ? 1 : 0);
            counts[r] = r_width * n_root;
            displs[r] = cur_disp;
            cur_disp += counts[r];
        }
    }

    // Use MPI_Type_create_contiguous for ElementDynamic-sized datatype
    MPI_Datatype type_elem;
    MPI_Type_contiguous(sizeof(ElementDynamic), MPI_BYTE, &type_elem);
    MPI_Type_commit(&type_elem);

    MPI_Gatherv(owned_data.data(), num_owned, type_elem,
                global_elements.data(),
                counts.data(), displs.data(),
                type_elem,
                0, MPI_COMM_WORLD);

    MPI_Type_free(&type_elem);

    // Compute hash on rank 0
    if (rank == 0) {
        const uint64_t hash = computeHash(global_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_elements.size());
            for (const auto& elem : global_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            printf("Validation results:\n");
            val_t energy_sum = 0.0;
            val_t flux_sum   = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();

            #pragma omp parallel for reduction(+:energy_sum,flux_sum) \
                                     reduction(max:energy_max) \
                                     reduction(min:energy_min)
            for (size_t i = 0; i < global_elements.size(); ++i) {
                energy_sum += global_elements[i].current_energy;
                flux_sum   += global_elements[i].total_flux;
                energy_max  = std::max(global_elements[i].current_energy, energy_max);
                energy_min  = std::min(global_elements[i].current_energy, energy_min);
            }

            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;

            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }

            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }

            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }

            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }

            if (valid) {
                printf("  Validation: PASSED\n");
            }

            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
