#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Parallelization: MPI, OpenMP, CUDA
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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// World state with MPI domain decomposition, GPU acceleration, and OpenMP parallelism
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    // MPI domain decomposition
    int rank, size;
    int dims[2], coords[2];
    int up_rank, down_rank, left_rank, right_rank;
    MPI_Comm cart_comm;
    int nx_global, ny_global;
    int nx_loc, ny_loc, gx_off, gy_off;
    int stride;        // ny_loc + 2 (with ghost columns)
    int total_local;

    // CUDA device pointers
    val_t*  d_energy;        // ghost-padded: (nx_loc+2) x (ny_loc+2)
    val_t*  d_energy_swap;   // ghost-padded swap buffer
    val_t*  d_total_flux;    // local only: nx_loc x ny_loc
    idx_t*  d_material_idx;  // local only: nx_loc x ny_loc
    Material* d_materials;
    val_t*  d_col_buf;       // device buffer for column pack/unpack
    val_t*  d_col_buf2;

    // Pinned host buffers for MPI ghost exchange
    val_t* h_send_up;    val_t* h_recv_up;
    val_t* h_send_down;  val_t* h_recv_down;
    val_t* h_send_left;  val_t* h_recv_left;
    val_t* h_send_right; val_t* h_recv_right;
};

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n",                \
                world.rank, __FILE__, __LINE__, cudaGetErrorString(err));      \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while(0)

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Flux computation: one thread per local element
//   energy / energy_next are ghost-padded arrays (stride = ny_loc+2)
//   Ghost cells at row 0, row nx_loc+1, col 0, col ny_loc+1
//   Local elements  : row 1..nx_loc, col 1..ny_loc
__global__ void computeFluxKernel(
    const val_t* __restrict__ energy,
    val_t* __restrict__ energy_next,
    val_t* __restrict__ total_flux,
    const idx_t* __restrict__ mat_idx,
    const Material* __restrict__ materials,
    int nx, int ny, int stride,
    int gx_off, int gy_off, int nx_global)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int ly = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (lx > nx || ly > ny) return;

    int gx = gx_off + (lx - 1);
    int gy = gy_off + (ly - 1);
    int idx = lx * stride + ly;
    int lidx = (lx - 1) * ny + (ly - 1);

    int mat_id = mat_idx[lidx];
    val_t cur = energy[idx];
    val_t flux = materials[mat_id].external_flow;
    val_t tc  = materials[mat_id].transfer_coeff;

    // Match original order: offsets {{1,0}, {-1,0}, {0,1}, {0,-1}} = down, up, right, left
    // Down (gx+1, gy)
    if (gx < nx_global - 1)
        flux += (energy[(lx+1)*stride + ly] - cur) * tc * 0.25;
    // Up   (gx-1, gy)
    if (gx > 0)
        flux += (energy[(lx-1)*stride + ly] - cur) * tc * 0.25;
    // Right (gx, gy+1)
    if (gy < nx_global - 1)
        flux += (energy[lx*stride + (ly+1)] - cur) * tc * 0.25;
    // Left  (gx, gy-1)
    if (gy > 0)
        flux += (energy[lx*stride + (ly-1)] - cur) * tc * 0.25;

    energy_next[idx] = cur + flux;
    total_flux[lidx] += fabs(flux);
}

// Pack local left  boundary (column 1)   -> contiguous buffer
__global__ void packLeft(const val_t* __restrict__ energy, val_t* __restrict__ buf,
                          int nx, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) buf[i] = energy[(i+1) * stride + 1];
}

// Pack local right boundary (column ny) -> contiguous buffer
__global__ void packRight(const val_t* __restrict__ energy, val_t* __restrict__ buf,
                           int nx, int ny, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) buf[i] = energy[(i+1) * stride + ny];
}

// Unpack contiguous buffer -> left ghost column (column 0)
__global__ void unpackLeft(val_t* __restrict__ energy, const val_t* __restrict__ buf,
                            int nx, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) energy[(i+1) * stride + 0] = buf[i];
}

// Unpack contiguous buffer -> right ghost column (column ny+1)
__global__ void unpackRight(val_t* __restrict__ energy, const val_t* __restrict__ buf,
                             int nx, int ny, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) energy[(i+1) * stride + (ny+1)] = buf[i];
}

// Copy initialized element data into ghost-padded energy array on device
__global__ void initEnergyKernel(val_t* __restrict__ energy,
                                  const val_t* __restrict__ src,
                                  int nx, int ny, int stride) {
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= nx || ly >= ny) return;
    energy[(lx+1) * stride + (ly+1)] = src[lx * ny + ly];
}

// ============================================================================
// HOST HELPER FUNCTIONS
// ============================================================================

// Build the local subdomain for this MPI rank
void buildLocalGrid(World& world, const int n_elems_root) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.size);
    world.nx_global = n_elems_root;
    world.ny_global = n_elems_root;

    // 2D Cartesian topology
    world.dims[0] = 0; world.dims[1] = 0;
    MPI_Dims_create(world.size, 2, world.dims);
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, world.dims, periods, 0, &world.cart_comm);
    MPI_Cart_coords(world.cart_comm, world.rank, 2, world.coords);
    MPI_Cart_shift(world.cart_comm, 0, 1, &world.up_rank, &world.down_rank);
    MPI_Cart_shift(world.cart_comm, 1, 1, &world.left_rank, &world.right_rank);

    // Partition the grid (handle uneven division by distributing remainder)
    int base_nx = n_elems_root / world.dims[0];
    int base_ny = n_elems_root / world.dims[1];
    int rem_nx  = n_elems_root % world.dims[0];
    int rem_ny  = n_elems_root % world.dims[1];

    world.nx_loc = base_nx + (world.coords[0] < rem_nx ? 1 : 0);
    world.ny_loc = base_ny + (world.coords[1] < rem_ny ? 1 : 0);

    // Global offset of the first local element
    world.gx_off = 0;
    for (int i = 0; i < world.coords[0]; ++i)
        world.gx_off += base_nx + (i < rem_nx ? 1 : 0);
    world.gy_off = 0;
    for (int i = 0; i < world.coords[1]; ++i)
        world.gy_off += base_ny + (i < rem_ny ? 1 : 0);

    world.stride      = world.ny_loc + 2;
    world.total_local = world.nx_loc * world.ny_loc;

    // Initialize materials
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});   // DEFAULT_MAT_ID = 0
    world.materials.emplace_back(Material{0.8, 0.5});   // INFLOW_MAT_ID  = 1
    world.materials.emplace_back(Material{0.8, -0.5});  // OUTFLOW_MAT_ID = 2

    // Allocate local storage
    world.elements_static.resize(world.total_local);
    world.elements_dynamic.resize(world.total_local);
    world.elements_dynamic_swap.resize(world.total_local);

    // Build local elements (global grid semantics preserved)
    for (int lx = 0; lx < world.nx_loc; ++lx) {
        for (int ly = 0; ly < world.ny_loc; ++ly) {
            int gx = world.gx_off + lx;
            int gy = world.gy_off + ly;
            int lidx = lx * world.ny_loc + ly;

            auto& es = world.elements_static[lidx];
            es.material_idx  = DEFAULT_MAT_ID;
            es.num_connections = 0;
            world.elements_dynamic[lidx].current_energy = 0.0;
            world.elements_dynamic[lidx].total_flux     = 0.0;

            // Build connectivity with global indices (for host-side bookkeeping)
            int conn = 0;
            if (gx > 0) {
                es.connected_idx[conn] = (gx-1)*n_elems_root + gy;
                es.connected_flux[conn] = 1.0; conn++;
            }
            if (gx < n_elems_root-1) {
                es.connected_idx[conn] = (gx+1)*n_elems_root + gy;
                es.connected_flux[conn] = 1.0; conn++;
            }
            if (gy > 0) {
                es.connected_idx[conn] = gx*n_elems_root + (gy-1);
                es.connected_flux[conn] = 1.0; conn++;
            }
            if (gy < n_elems_root-1) {
                es.connected_idx[conn] = gx*n_elems_root + (gy+1);
                es.connected_flux[conn] = 1.0; conn++;
            }
            es.num_connections = conn;
        }
    }

    // Set corner materials (inflow/outflow at global corners)
    auto setMat = [&](int gx, int gy, int mat_id) {
        if (gx >= world.gx_off && gx < world.gx_off + world.nx_loc &&
            gy >= world.gy_off && gy < world.gy_off + world.ny_loc) {
            int lidx = (gx - world.gx_off) * world.ny_loc + (gy - world.gy_off);
            world.elements_static[lidx].material_idx = mat_id;
        }
    };
    int last = n_elems_root - 1;
    setMat(0,    0,    INFLOW_MAT_ID);
    setMat(0,    last, OUTFLOW_MAT_ID);
    setMat(last, 0,    OUTFLOW_MAT_ID);
    setMat(last, last, INFLOW_MAT_ID);
}

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------
void allocateGPU(World& world) {
    int ghost_size = (world.nx_loc + 2) * world.stride;
    CUDA_CHECK(cudaMalloc(&world.d_energy,       ghost_size * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_energy_swap,  ghost_size * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_total_flux,   world.total_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_material_idx, world.total_local * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&world.d_materials,    world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&world.d_col_buf,      world.nx_loc * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_col_buf2,     world.nx_loc * sizeof(val_t)));

    // Pinned host memory for MPI staging
    int max_row = world.ny_loc;
    int max_col = world.nx_loc;
    CUDA_CHECK(cudaMallocHost(&world.h_send_up,    max_row * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_recv_up,    max_row * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_send_down,  max_row * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_recv_down,  max_row * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_send_left,  max_col * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_recv_left,  max_col * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_send_right, max_col * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_recv_right, max_col * sizeof(val_t)));
}

void freeGPU(World& world) {
    CUDA_CHECK(cudaFree(world.d_energy));
    CUDA_CHECK(cudaFree(world.d_energy_swap));
    CUDA_CHECK(cudaFree(world.d_total_flux));
    CUDA_CHECK(cudaFree(world.d_material_idx));
    CUDA_CHECK(cudaFree(world.d_materials));
    CUDA_CHECK(cudaFree(world.d_col_buf));
    CUDA_CHECK(cudaFree(world.d_col_buf2));
    CUDA_CHECK(cudaFreeHost(world.h_send_up));
    CUDA_CHECK(cudaFreeHost(world.h_recv_up));
    CUDA_CHECK(cudaFreeHost(world.h_send_down));
    CUDA_CHECK(cudaFreeHost(world.h_recv_down));
    CUDA_CHECK(cudaFreeHost(world.h_send_left));
    CUDA_CHECK(cudaFreeHost(world.h_recv_left));
    CUDA_CHECK(cudaFreeHost(world.h_send_right));
    CUDA_CHECK(cudaFreeHost(world.h_recv_right));
}

void copyToGPU(World& world) {
    int ghost_size = (world.nx_loc + 2) * world.stride;

    // Zero out ghost-padded arrays (ghost cells = 0 initially)
    CUDA_CHECK(cudaMemset(world.d_energy,      0, ghost_size * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(world.d_energy_swap, 0, ghost_size * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(world.d_total_flux,  0, world.total_local * sizeof(val_t)));

    // Copy material indices to device
    std::vector<idx_t> mat_host(world.total_local);
    #pragma omp parallel for
    for (int i = 0; i < world.total_local; ++i)
        mat_host[i] = world.elements_static[i].material_idx;
    CUDA_CHECK(cudaMemcpy(world.d_material_idx, mat_host.data(),
                           world.total_local * sizeof(idx_t), cudaMemcpyHostToDevice));

    // Copy materials
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(),
                           world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
}

void copyFromGPU(World& world) {
    int ghost_size = (world.nx_loc + 2) * world.stride;
    std::vector<val_t> h_energy(ghost_size);
    std::vector<val_t> h_tflux(world.total_local);

    CUDA_CHECK(cudaMemcpy(h_energy.data(), world.d_energy,
                           ghost_size * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tflux.data(), world.d_total_flux,
                           world.total_local * sizeof(val_t), cudaMemcpyDeviceToHost));

    #pragma omp parallel for
    for (int lx = 0; lx < world.nx_loc; ++lx) {
        for (int ly = 0; ly < world.ny_loc; ++ly) {
            int lidx = lx * world.ny_loc + ly;
            int gidx = (lx+1) * world.stride + (ly+1);
            world.elements_dynamic[lidx].current_energy = h_energy[gidx];
            world.elements_dynamic[lidx].total_flux     = h_tflux[lidx];
        }
    }
}

// ---------------------------------------------------------------------------
// Ghost cell exchange: MPI neighbour communication + CUDA pack/unpack
// ---------------------------------------------------------------------------
void exchangeGhosts(World& world) {
    int nx = world.nx_loc, ny = world.ny_loc, s = world.stride;

    // ----- Up / Down (contiguous rows) ------------------------------------
    // Top    boundary at row 1    -> send up,   receive into row 0
    // Bottom boundary at row nx   -> send down, receive into row nx+1
    CUDA_CHECK(cudaMemcpy(world.h_send_up,   world.d_energy + 1*s + 1,
                           ny * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.h_send_down, world.d_energy + nx*s + 1,
                           ny * sizeof(val_t), cudaMemcpyDeviceToHost));

    MPI_Sendrecv(world.h_send_up,   ny, MPI_DOUBLE, world.up_rank,   0,
                 world.h_recv_up,   ny, MPI_DOUBLE, world.down_rank, 0,
                 world.cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(world.h_send_down, ny, MPI_DOUBLE, world.down_rank, 1,
                 world.h_recv_down, ny, MPI_DOUBLE, world.up_rank,   1,
                 world.cart_comm, MPI_STATUS_IGNORE);

    CUDA_CHECK(cudaMemcpy(world.d_energy + 0*s + 1,   world.h_recv_up,
                           ny * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_energy + (nx+1)*s + 1, world.h_recv_down,
                           ny * sizeof(val_t), cudaMemcpyHostToDevice));

    // ----- Left / Right (non-contiguous columns, pack via kernels) --------
    int threads = 256;
    int blocks  = (nx + threads - 1) / threads;

    // Pack left boundary (col 1) and right boundary (col ny) on device
    packLeft<<<blocks, threads>>>(world.d_energy, world.d_col_buf, nx, s);
    packRight<<<blocks, threads>>>(world.d_energy, world.d_col_buf2, nx, ny, s);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Transfer packed columns to pinned host
    CUDA_CHECK(cudaMemcpy(world.h_send_left,  world.d_col_buf,
                           nx * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.h_send_right, world.d_col_buf2,
                           nx * sizeof(val_t), cudaMemcpyDeviceToHost));

    MPI_Sendrecv(world.h_send_left,  nx, MPI_DOUBLE, world.left_rank,  2,
                 world.h_recv_left,  nx, MPI_DOUBLE, world.right_rank, 2,
                 world.cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(world.h_send_right, nx, MPI_DOUBLE, world.right_rank, 3,
                 world.h_recv_right, nx, MPI_DOUBLE, world.left_rank,  3,
                 world.cart_comm, MPI_STATUS_IGNORE);

    // Transfer received data back to device
    CUDA_CHECK(cudaMemcpy(world.d_col_buf,  world.h_recv_left,
                           nx * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_col_buf2, world.h_recv_right,
                           nx * sizeof(val_t), cudaMemcpyHostToDevice));

    // Unpack into ghost columns (col 0 and col ny+1)
    unpackLeft<<<blocks, threads>>>(world.d_energy, world.d_col_buf,  nx, s);
    unpackRight<<<blocks, threads>>>(world.d_energy, world.d_col_buf2, nx, ny, s);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// PARALLEL SIMULATION (MPI + CUDA + OpenMP)
// ============================================================================
void runSimulation(World& world, const int n_iters) {
    // GPU configuration
    dim3 block2D(16, 16);
    dim3 grid2D((world.nx_loc + 15) / 16, (world.ny_loc + 15) / 16);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch CUDA kernel: read d_energy (with ghost cells) -> write d_energy_swap
        computeFluxKernel<<<grid2D, block2D>>>(
            world.d_energy, world.d_energy_swap, world.d_total_flux,
            world.d_material_idx, world.d_materials,
            world.nx_loc, world.ny_loc, world.stride,
            world.gx_off, world.gy_off, world.nx_global);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap device buffers (energy <-> energy_swap)
        std::swap(world.d_energy, world.d_energy_swap);

        // Exchange ghost cells for next iteration (skip after last)
        if (iter < n_iters - 1) {
            exchangeGhosts(world);
        }
    }
}

// ============================================================================
// HASH (MPI bitwise-XOR reduction for distributed consistency)
// ============================================================================
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    // Each rank computes contribution using global element indices
    for (int lx = 0; lx < world.nx_loc; ++lx) {
        for (int ly = 0; ly < world.ny_loc; ++ly) {
            size_t i = static_cast<size_t>(world.gx_off + lx) * world.nx_global +
                       (world.gy_off + ly);
            const auto& elem = world.elements_dynamic[lx * world.ny_loc + ly];
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
            hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return global_hash;
}

// ============================================================================
// VALIDATION (global sum/min/max on rank 0)
// ============================================================================
bool validateResults(const World& world) {
    // Local reductions
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum   = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
                                 reduction(max:local_energy_max)             \
                                 reduction(min:local_energy_min)
    for (int i = 0; i < world.total_local; ++i) {
        const auto& elem = world.elements_dynamic[i];
        local_energy_sum += elem.current_energy;
        local_flux_sum   += elem.total_flux;
        if (elem.current_energy > local_energy_max) local_energy_max = elem.current_energy;
        if (elem.current_energy < local_energy_min) local_energy_min = elem.current_energy;
    }

    // MPI reductions to rank 0
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = 0.0, energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum,   &flux_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            return false;
        }
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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
    return true;
}

// ============================================================================
// COLLECT RESULTS to rank 0
// ============================================================================
void gatherResults(const World& world, std::vector<val_t>& global_energy) {
    if (world.rank == 0) {
        global_energy.resize(world.nx_global * world.ny_global);
        // Copy local portion
        for (int lx = 0; lx < world.nx_loc; ++lx)
            for (int ly = 0; ly < world.ny_loc; ++ly)
                global_energy[(world.gx_off + lx) * world.nx_global + (world.gy_off + ly)] =
                    world.elements_dynamic[lx * world.ny_loc + ly].current_energy;

        // Receive from each other rank
        for (int r = 1; r < world.size; ++r) {
            int meta[4];
            MPI_Recv(meta, 4, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int gx0 = meta[0], gy0 = meta[1], nxl = meta[2], nyl = meta[3];
            std::vector<val_t> buf(nxl * nyl);
            MPI_Recv(buf.data(), nxl * nyl, MPI_DOUBLE, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int lx = 0; lx < nxl; ++lx)
                for (int ly = 0; ly < nyl; ++ly)
                    global_energy[(gx0 + lx) * world.nx_global + (gy0 + ly)] =
                        buf[lx * nyl + ly];
        }
    } else {
        int meta[4] = {world.gx_off, world.gy_off, world.nx_loc, world.ny_loc};
        MPI_Send(meta, 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::vector<val_t> local_energy(world.total_local);
        for (int i = 0; i < world.total_local; ++i)
            local_energy[i] = world.elements_dynamic[i].current_energy;
        MPI_Send(local_energy.data(), world.total_local, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
}

// ============================================================================
// COMMAND-LINE HELP
// ============================================================================
void printUsage(const char* progName) {
    printf("Usage: mpirun -np <N> %s [options]\n", progName);
    printf("Hybrid MPI+OpenMP+CUDA unstructured mesh benchmark\n");
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int n_elems_root = 512;
    int n_iters      = 10;
    bool validate    = false;
    bool printRes    = false;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments on rank 0, broadcast to all
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printRes = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printRes,     1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    const int n_elems = n_elems_root * n_elems_root;
    World world;

    // Build local subdomain
    buildLocalGrid(world, n_elems_root);

    // Report header from rank 0
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d  (grid %dx%d)\n", size, world.dims[0], world.dims[1]);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        int cuda_dev_count = 0;
        cudaGetDeviceCount(&cuda_dev_count);
        printf("CUDA devices available: %d\n", cuda_dev_count);
        printf("\n");
    }

    // Set CUDA device (round-robin assignment across MPI ranks)
    int cuda_dev_count = 0;
    cudaGetDeviceCount(&cuda_dev_count);
    int cuda_dev_id = (cuda_dev_count > 0) ? (rank % cuda_dev_count) : 0;
    cudaSetDevice(cuda_dev_id);
    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cuda_dev_id);
        printf("Building unstructured mesh...\n");
    }

    // Allocate and initialize GPU memory
    allocateGPU(world);
    copyToGPU(world);

    // Calculate memory usage (global problem)
    if (rank == 0) {
        const size_t static_mem  = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem   = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation – timed section
    if (rank == 0) printf("Running simulation (MPI+OpenMP+CUDA)...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    runSimulation(world, n_iters);

    double t_end = MPI_Wtime();
    double local_time = (t_end - t_start) * 1000.0; // ms
    double max_time_ms = 0.0;
    MPI_Reduce(&local_time, &max_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results from GPU back to host
    copyFromGPU(world);

    // Report timing & performance from rank 0
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_time_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (max_time_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Computation time: %ld ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(world);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    } else {
        // Non-root ranks still need to call computeHash for the MPI_Reduce
        computeHash(world);
    }

    // Print results for external validation
    if (printRes) {
        std::vector<double> global_energy;
        gatherResults(world, global_energy);
        if (rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        bool valid = validateResults(world);
        bool global_valid = false;
        MPI_Reduce(&valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
        if (rank == 0 && !global_valid) {
            MPI_Finalize();
            return 1;
        }
    }

    // Cleanup
    freeGPU(world);
    MPI_Comm_free(&world.cart_comm);
    MPI_Finalize();
    return 0;
}
