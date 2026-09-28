#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include <cuda.h>
#include <nvrtc.h>

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

// Structure-of-arrays layout of the static mesh connectivity, used to
// stage data efficiently onto the GPU.
struct MeshSoA {
    std::vector<idx_t> material_idx;
    std::vector<idx_t> num_connections;
    std::vector<idx_t> connected_idx;   // size n_elems * MAX_CONNECTIONS
    std::vector<val_t> connected_flux;  // size n_elems * MAX_CONNECTIONS
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ---------------------------------------------------------------------------
// CUDA driver / NVRTC error checking helpers
// ---------------------------------------------------------------------------

#define CU_CHECK(call)                                                          \
    do {                                                                        \
        CUresult _cu_res = (call);                                              \
        if (_cu_res != CUDA_SUCCESS) {                                          \
            const char* _cu_err = nullptr;                                      \
            cuGetErrorString(_cu_res, &_cu_err);                                \
            fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__,       \
                    __LINE__, _cu_err ? _cu_err : "unknown error");             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                        \
    } while (0)

#define NVRTC_CHECK(call)                                                        \
    do {                                                                        \
        nvrtcResult _nv_res = (call);                                           \
        if (_nv_res != NVRTC_SUCCESS) {                                        \
            fprintf(stderr, "NVRTC error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    nvrtcGetErrorString(_nv_res));                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                        \
    } while (0)

// CUDA kernel source, compiled at runtime via NVRTC. Implements exactly the
// same per-element energy-flux update as the original sequential code:
// for each owned element, sum flux contributions from connected neighbors
// and update energy / accumulated flux magnitude.
static const char* kKernelSource = R"CUDA(
extern "C" __global__
void update_kernel(
    const unsigned long long* __restrict__ material_idx,
    const unsigned long long* __restrict__ num_connections,
    const unsigned long long* __restrict__ connected_idx,
    const double* __restrict__ connected_flux,
    const double* __restrict__ mat_transfer_coeff,
    const double* __restrict__ mat_external_flow,
    const double* __restrict__ energy,
    const double* __restrict__ flux_state,
    double* __restrict__ new_energy,
    double* __restrict__ new_flux,
    unsigned long long start,
    unsigned long long end,
    int max_connections)
{
    unsigned long long i = start + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;

    unsigned long long mat = material_idx[i];
    double transfer_coeff = mat_transfer_coeff[mat];
    double total_flux = mat_external_flow[mat];
    double e_i = energy[i];

    unsigned long long nconn = num_connections[i];
    unsigned long long base = i * (unsigned long long)max_connections;
    for (unsigned long long j = 0; j < nconn; ++j) {
        unsigned long long neighbor = connected_idx[base + j];
        double cflux = connected_flux[base + j];
        total_flux += (energy[neighbor] - e_i) * transfer_coeff * cflux * 0.25;
    }

    new_energy[i] = e_i + total_flux;
    double abs_flux = total_flux < 0.0 ? -total_flux : total_flux;
    new_flux[i] = flux_state[i] + abs_flux;
}

// Scatters received halo values (energy/flux for elements owned by other
// ranks) into the full-sized energy/flux arrays at their global indices, so
// that the next iteration's update_kernel can read them like any other
// element.
extern "C" __global__
void scatter_halo_kernel(
    const unsigned long long* __restrict__ halo_indices,
    const double* __restrict__ halo_energy,
    const double* __restrict__ halo_flux,
    double* __restrict__ energy,
    double* __restrict__ flux_state,
    unsigned long long halo_count)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= halo_count) return;
    unsigned long long idx = halo_indices[k];
    energy[idx] = halo_energy[k];
    flux_state[idx] = halo_flux[k];
}
)CUDA";

// Holds the GPU context, compiled kernel, and device buffers for a single
// MPI rank's local GPU.
struct GpuContext {
    CUcontext ctx = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUfunction scatterKernel = nullptr;

    CUdeviceptr d_material_idx = 0;
    CUdeviceptr d_num_connections = 0;
    CUdeviceptr d_connected_idx = 0;
    CUdeviceptr d_connected_flux = 0;
    CUdeviceptr d_mat_transfer_coeff = 0;
    CUdeviceptr d_mat_external_flow = 0;
    CUdeviceptr d_energy = 0;
    CUdeviceptr d_flux_state = 0;
    CUdeviceptr d_new_energy = 0;
    CUdeviceptr d_new_flux = 0;

    // Halo support: indices (global) of elements owned by other ranks that
    // this rank's local elements depend on, and staging buffers for the
    // values received for them each iteration.
    CUdeviceptr d_halo_indices = 0;
    CUdeviceptr d_halo_energy_in = 0;
    CUdeviceptr d_halo_flux_in = 0;
};

// Compile the kernel source for the given compute capability and load it.
void compileKernel(GpuContext& gpu, int cc_major, int cc_minor) {
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, kKernelSource, "update_kernel.cu", 0, nullptr, nullptr));

    char archOpt[48];
    snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
    // Disable FMA contraction so GPU floating-point rounding stays as close
    // as possible to the straightforward sequential arithmetic it replaces.
    const char* opts[] = {archOpt, "--fmad=false"};

    nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);

    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        if (compileResult != NVRTC_SUCCESS) {
            fprintf(stderr, "NVRTC compile log:\n%s\n", log.data());
        }
    }
    if (compileResult != NVRTC_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t ptxSize = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    CU_CHECK(cuModuleLoadDataEx(&gpu.module, ptx.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&gpu.kernel, gpu.module, "update_kernel"));
    CU_CHECK(cuModuleGetFunction(&gpu.scatterKernel, gpu.module, "scatter_halo_kernel"));
}

void allocateGpuBuffers(GpuContext& gpu, size_t n_elems, size_t n_materials, size_t halo_count) {
    CU_CHECK(cuMemAlloc(&gpu.d_material_idx, n_elems * sizeof(idx_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_num_connections, n_elems * sizeof(idx_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_connected_idx, n_elems * MAX_CONNECTIONS * sizeof(idx_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_connected_flux, n_elems * MAX_CONNECTIONS * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_mat_transfer_coeff, n_materials * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_mat_external_flow, n_materials * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_energy, n_elems * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_flux_state, n_elems * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_new_energy, n_elems * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_new_flux, n_elems * sizeof(val_t)));

    const size_t halo_alloc = std::max<size_t>(halo_count, 1);
    CU_CHECK(cuMemAlloc(&gpu.d_halo_indices, halo_alloc * sizeof(idx_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_halo_energy_in, halo_alloc * sizeof(val_t)));
    CU_CHECK(cuMemAlloc(&gpu.d_halo_flux_in, halo_alloc * sizeof(val_t)));
}

void freeGpuBuffers(GpuContext& gpu) {
    cuMemFree(gpu.d_material_idx);
    cuMemFree(gpu.d_num_connections);
    cuMemFree(gpu.d_connected_idx);
    cuMemFree(gpu.d_connected_flux);
    cuMemFree(gpu.d_mat_transfer_coeff);
    cuMemFree(gpu.d_mat_external_flow);
    cuMemFree(gpu.d_energy);
    cuMemFree(gpu.d_flux_state);
    cuMemFree(gpu.d_new_energy);
    cuMemFree(gpu.d_new_flux);
    cuMemFree(gpu.d_halo_indices);
    cuMemFree(gpu.d_halo_energy_in);
    cuMemFree(gpu.d_halo_flux_in);
    if (gpu.module) cuModuleUnload(gpu.module);
    if (gpu.ctx) cuCtxDestroy(gpu.ctx);
}

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
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static) collapse(2)
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            idx_t n_conn = 0;
            idx_t local_idx[4];
            val_t local_flux[4];
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    local_idx[n_conn] = neighbor_idx;
                    local_flux[n_conn] = 1.0;
                    n_conn++;
                }
            }
            elem.num_connections = n_conn;
            for (idx_t n = 0; n < n_conn; ++n) {
                elem.connected_idx[n] = local_idx[n];
                elem.connected_flux[n] = local_flux[n];
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

// Flatten the AoS static mesh connectivity into GPU-friendly SoA arrays.
void buildSoA(const World& world, MeshSoA& soa) {
    const size_t n = world.elements_static.size();
    soa.material_idx.resize(n);
    soa.num_connections.resize(n);
    soa.connected_idx.resize(n * MAX_CONNECTIONS);
    soa.connected_flux.resize(n * MAX_CONNECTIONS);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& es = world.elements_static[i];
        soa.material_idx[i] = es.material_idx;
        soa.num_connections[i] = es.num_connections;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            soa.connected_idx[i * MAX_CONNECTIONS + j] = es.connected_idx[j];
            soa.connected_flux[i * MAX_CONNECTIONS + j] = es.connected_flux[j];
        }
    }
}

// Determine, for the given owned element range, the sorted set of distinct
// global element indices outside that range which are referenced as
// neighbors ("halo" elements). This is derived directly from the actual
// connectivity graph, so it works for arbitrary (not just grid-structured)
// unstructured meshes.
std::vector<idx_t> computeHaloIndices(const MeshSoA& soa, size_t n_elems, size_t local_start, size_t local_count) {
    const size_t local_end = local_start + local_count;
    std::vector<uint8_t> is_halo(n_elems, 0);

    #pragma omp parallel for schedule(static)
    for (size_t i = local_start; i < local_end; ++i) {
        const idx_t nconn = soa.num_connections[i];
        const size_t base = i * MAX_CONNECTIONS;
        for (idx_t j = 0; j < nconn; ++j) {
            const idx_t nb = soa.connected_idx[base + j];
            if (nb < local_start || nb >= local_end) {
                #pragma omp atomic write
                is_halo[nb] = 1;
            }
        }
    }

    std::vector<idx_t> halo_indices;
    for (size_t j = 0; j < n_elems; ++j) {
        if (is_halo[j]) halo_indices.push_back(static_cast<idx_t>(j));
    }
    return halo_indices;
}

// Find the owning rank of a global element index given the partition's
// exclusive prefix-sum boundaries (range_ends[r] = displs[r] + counts[r]).
int findOwnerRank(const std::vector<int>& range_ends, idx_t idx) {
    auto it = std::upper_bound(range_ends.begin(), range_ends.end(), static_cast<int>(idx));
    return static_cast<int>(it - range_ends.begin());
}

// Run the simulation for n_iters iterations using a hybrid MPI + OpenMP +
// CUDA strategy:
//  - MPI: the element range is partitioned into contiguous blocks across
//    ranks; each rank owns and computes updates for its own block. Only the
//    "halo" elements actually referenced across rank boundaries (derived
//    from the real connectivity graph) are exchanged each iteration, so
//    communication volume scales with the boundary size, not the mesh size.
//  - CUDA: each rank offloads its per-element flux update, and the halo
//    scatter, to its assigned GPU via NVRTC-compiled kernels.
//  - OpenMP: host-side data preparation, staging, and reductions are
//    parallelized across CPU threads.
void runSimulationHybrid(World& world, const MeshSoA& soa, const int n_iters,
                          int mpi_rank, int mpi_size) {
    const size_t n_elems = world.elements_static.size();
    const size_t n_materials = world.materials.size();

    // Partition elements across MPI ranks into contiguous blocks.
    std::vector<int> counts(mpi_size), displs(mpi_size);
    {
        const size_t base = n_elems / static_cast<size_t>(mpi_size);
        const size_t rem = n_elems % static_cast<size_t>(mpi_size);
        size_t offset = 0;
        for (int r = 0; r < mpi_size; ++r) {
            const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(offset);
            offset += c;
        }
    }
    const size_t local_start = static_cast<size_t>(displs[mpi_rank]);
    const size_t local_count = static_cast<size_t>(counts[mpi_rank]);
    const size_t local_end = local_start + local_count;

    // Determine the halo-exchange communication pattern (derived from the
    // actual connectivity graph, once, before touching the GPU).
    std::vector<idx_t> halo_indices = computeHaloIndices(soa, n_elems, local_start, local_count);
    const size_t halo_count = halo_indices.size();

    std::vector<int> range_ends(mpi_size);
    for (int r = 0; r < mpi_size; ++r) range_ends[r] = displs[r] + counts[r];

    // request_counts[r]: how many halo values this rank needs from rank r.
    std::vector<int> request_counts(mpi_size, 0), request_displs(mpi_size, 0);
    for (idx_t idx : halo_indices) {
        request_counts[findOwnerRank(range_ends, idx)]++;
    }
    {
        int offset = 0;
        for (int r = 0; r < mpi_size; ++r) {
            request_displs[r] = offset;
            offset += request_counts[r];
        }
    }

    // provide_counts[r]: how many of *our* local values rank r needs from us.
    std::vector<int> provide_counts(mpi_size, 0), provide_displs(mpi_size, 0);
    MPI_Alltoall(request_counts.data(), 1, MPI_INT, provide_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    {
        int offset = 0;
        for (int r = 0; r < mpi_size; ++r) {
            provide_displs[r] = offset;
            offset += provide_counts[r];
        }
    }
    const size_t provide_count = static_cast<size_t>(provide_displs.back()) + provide_counts.back();

    std::vector<idx_t> provide_global_idx(std::max<size_t>(provide_count, 1));
    MPI_Alltoallv(halo_indices.data(), request_counts.data(), request_displs.data(), MPI_UINT64_T,
                  provide_global_idx.data(), provide_counts.data(), provide_displs.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);

    std::vector<size_t> provide_local_offsets(provide_count);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < provide_count; ++k) {
        provide_local_offsets[k] = static_cast<size_t>(provide_global_idx[k]) - local_start;
    }

    // Determine node-local rank to assign a distinct GPU per rank on a node.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    CU_CHECK(cuInit(0));
    int device_count = 0;
    CU_CHECK(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device_id = local_rank % device_count;

    CUdevice cuDevice;
    CU_CHECK(cuDeviceGet(&cuDevice, device_id));

    GpuContext gpu;
    CU_CHECK(cuCtxCreate(&gpu.ctx, 0, cuDevice));

    int cc_major = 0, cc_minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CU_CHECK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    if (mpi_rank == 0) {
        printf("Parallelization: %d MPI rank(s) x %d OpenMP thread(s)/rank, CUDA device per rank (sm_%d%d)\n",
               mpi_size, omp_get_max_threads(), cc_major, cc_minor);
    }

    compileKernel(gpu, cc_major, cc_minor);
    allocateGpuBuffers(gpu, n_elems, n_materials, halo_count);

    // Upload static (read-only for the whole run) connectivity data once.
    CU_CHECK(cuMemcpyHtoD(gpu.d_material_idx, soa.material_idx.data(), n_elems * sizeof(idx_t)));
    CU_CHECK(cuMemcpyHtoD(gpu.d_num_connections, soa.num_connections.data(), n_elems * sizeof(idx_t)));
    CU_CHECK(cuMemcpyHtoD(gpu.d_connected_idx, soa.connected_idx.data(), n_elems * MAX_CONNECTIONS * sizeof(idx_t)));
    CU_CHECK(cuMemcpyHtoD(gpu.d_connected_flux, soa.connected_flux.data(), n_elems * MAX_CONNECTIONS * sizeof(val_t)));

    std::vector<val_t> mat_transfer_coeff(n_materials), mat_external_flow(n_materials);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < n_materials; ++m) {
        mat_transfer_coeff[m] = world.materials[m].transfer_coeff;
        mat_external_flow[m] = world.materials[m].external_flow;
    }
    CU_CHECK(cuMemcpyHtoD(gpu.d_mat_transfer_coeff, mat_transfer_coeff.data(), n_materials * sizeof(val_t)));
    CU_CHECK(cuMemcpyHtoD(gpu.d_mat_external_flow, mat_external_flow.data(), n_materials * sizeof(val_t)));

    // Upload the static halo index list once.
    if (halo_count > 0) {
        CU_CHECK(cuMemcpyHtoD(gpu.d_halo_indices, halo_indices.data(), halo_count * sizeof(idx_t)));
    }

    // Pinned host staging buffers for fast H2D/D2H transfers.
    val_t *h_local_energy = nullptr, *h_local_flux = nullptr;
    val_t *h_provide_energy = nullptr, *h_provide_flux = nullptr;
    val_t *h_halo_energy = nullptr, *h_halo_flux = nullptr;
    CU_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&h_local_energy), std::max<size_t>(local_count, 1) * sizeof(val_t)));
    CU_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&h_local_flux), std::max<size_t>(local_count, 1) * sizeof(val_t)));
    CU_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&h_provide_energy), std::max<size_t>(provide_count, 1) * sizeof(val_t)));
    CU_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&h_provide_flux), std::max<size_t>(provide_count, 1) * sizeof(val_t)));
    CU_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&h_halo_energy), std::max<size_t>(halo_count, 1) * sizeof(val_t)));
    CU_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&h_halo_flux), std::max<size_t>(halo_count, 1) * sizeof(val_t)));

    // Initialize device energy/flux state from the (zero-initialized) World.
    {
        std::vector<val_t> init_energy(n_elems), init_flux(n_elems);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elems; ++i) {
            init_energy[i] = world.elements_dynamic[i].current_energy;
            init_flux[i] = world.elements_dynamic[i].total_flux;
        }
        CU_CHECK(cuMemcpyHtoD(gpu.d_energy, init_energy.data(), n_elems * sizeof(val_t)));
        CU_CHECK(cuMemcpyHtoD(gpu.d_flux_state, init_flux.data(), n_elems * sizeof(val_t)));
    }

    constexpr unsigned int BLOCK_SIZE = 256;
    const unsigned int grid_size =
        local_count > 0 ? static_cast<unsigned int>((local_count + BLOCK_SIZE - 1) / BLOCK_SIZE) : 1;
    const unsigned int halo_grid_size =
        halo_count > 0 ? static_cast<unsigned int>((halo_count + BLOCK_SIZE - 1) / BLOCK_SIZE) : 1;

    unsigned long long u_start = local_start;
    unsigned long long u_end = local_end;
    int max_conn = MAX_CONNECTIONS;
    unsigned long long u_halo_count = halo_count;

    void* kernelArgs[] = {&gpu.d_material_idx,    &gpu.d_num_connections, &gpu.d_connected_idx,
                          &gpu.d_connected_flux,   &gpu.d_mat_transfer_coeff, &gpu.d_mat_external_flow,
                          &gpu.d_energy,           &gpu.d_flux_state,      &gpu.d_new_energy,
                          &gpu.d_new_flux,         &u_start,               &u_end,
                          &max_conn};

    void* scatterArgs[] = {&gpu.d_halo_indices, &gpu.d_halo_energy_in, &gpu.d_halo_flux_in,
                           &gpu.d_energy,        &gpu.d_flux_state,     &u_halo_count};

    for (int iter = 0; iter < n_iters; ++iter) {
        // 1. Compute owned range on the GPU.
        if (local_count > 0) {
            CU_CHECK(cuLaunchKernel(gpu.kernel, grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, nullptr, kernelArgs, nullptr));
            // 2. Commit the freshly computed owned range back into the
            //    unified energy/flux arrays (device-to-device, no host trip).
            CU_CHECK(cuMemcpyDtoD(gpu.d_energy + local_start * sizeof(val_t),
                                   gpu.d_new_energy + local_start * sizeof(val_t), local_count * sizeof(val_t)));
            CU_CHECK(cuMemcpyDtoD(gpu.d_flux_state + local_start * sizeof(val_t),
                                   gpu.d_new_flux + local_start * sizeof(val_t), local_count * sizeof(val_t)));
            // 3. Pull our owned range to the host so we can share it with
            //    ranks that need it as halo data.
            CU_CHECK(cuMemcpyDtoH(h_local_energy, gpu.d_energy + local_start * sizeof(val_t),
                                   local_count * sizeof(val_t)));
            CU_CHECK(cuMemcpyDtoH(h_local_flux, gpu.d_flux_state + local_start * sizeof(val_t),
                                   local_count * sizeof(val_t)));
        }

        // 4. Gather the subset other ranks requested from us.
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < provide_count; ++k) {
            h_provide_energy[k] = h_local_energy[provide_local_offsets[k]];
            h_provide_flux[k] = h_local_flux[provide_local_offsets[k]];
        }

        // 5. Exchange halo values: send what others need from us, receive
        //    what we need from them (volume scales with boundary size).
        MPI_Alltoallv(h_provide_energy, provide_counts.data(), provide_displs.data(), MPI_DOUBLE, h_halo_energy,
                      request_counts.data(), request_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoallv(h_provide_flux, provide_counts.data(), provide_displs.data(), MPI_DOUBLE, h_halo_flux,
                      request_counts.data(), request_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // 6. Scatter received halo values into the device arrays.
        if (halo_count > 0) {
            CU_CHECK(cuMemcpyHtoD(gpu.d_halo_energy_in, h_halo_energy, halo_count * sizeof(val_t)));
            CU_CHECK(cuMemcpyHtoD(gpu.d_halo_flux_in, h_halo_flux, halo_count * sizeof(val_t)));
            CU_CHECK(cuLaunchKernel(gpu.scatterKernel, halo_grid_size, 1, 1, BLOCK_SIZE, 1, 1, 0, nullptr,
                                     scatterArgs, nullptr));
        }
        CU_CHECK(cuCtxSynchronize());
    }

    // Final one-time global gather (not per-iteration) so every rank has the
    // complete state for hashing/validation/output, matching the original
    // program's semantics.
    std::vector<val_t> full_energy(n_elems), full_flux(n_elems);
    if (local_count > 0) {
        CU_CHECK(cuMemcpyDtoH(h_local_energy, gpu.d_energy + local_start * sizeof(val_t), local_count * sizeof(val_t)));
        CU_CHECK(cuMemcpyDtoH(h_local_flux, gpu.d_flux_state + local_start * sizeof(val_t), local_count * sizeof(val_t)));
    }
    MPI_Allgatherv(h_local_energy, static_cast<int>(local_count), MPI_DOUBLE, full_energy.data(), counts.data(),
                   displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(h_local_flux, static_cast<int>(local_count), MPI_DOUBLE, full_flux.data(), counts.data(),
                   displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = full_energy[i];
        world.elements_dynamic[i].total_flux = full_flux[i];
    }

    cuMemFreeHost(h_local_energy);
    cuMemFreeHost(h_local_flux);
    cuMemFreeHost(h_provide_energy);
    cuMemFreeHost(h_provide_flux);
    cuMemFreeHost(h_halo_energy);
    cuMemFreeHost(h_halo_flux);

    freeGpuBuffers(gpu);
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    const size_t n = world.elements_dynamic.size();
    #pragma omp parallel for schedule(static) reduction(+:energy_sum, flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < n; ++i) {
        const auto& elem = world.elements_dynamic[i];
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
    const size_t n = elements.size();
    // XOR-combine is commutative/associative, so this reduction is safe
    // regardless of iteration order.
    #pragma omp parallel for schedule(static) reduction(^:hash)
    for (size_t i = 0; i < n; ++i) {
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
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh (replicated identically on every rank;
    // fully deterministic given n_elems_root, so no communication needed).
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    MeshSoA soa;
    buildSoA(world, soa);

    // Calculate memory usage
    if (mpi_rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n", total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulationHybrid(world, soa, n_iters, mpi_rank, mpi_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (mpi_rank == 0) {
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
    }

    // Compute hash for verification (deterministic, identical across ranks
    // since the full state was synchronized every iteration).
    const uint64_t hash = computeHash(world.elements_dynamic);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    int valid_flag = 1;
    if (validate) {
        if (mpi_rank == 0) {
            valid_flag = validateResults(world) ? 1 : 0;
        }
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid_flag) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
