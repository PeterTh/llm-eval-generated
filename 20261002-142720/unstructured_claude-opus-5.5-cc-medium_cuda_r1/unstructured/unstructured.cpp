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

// CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Accumulate one connection's flux into total_flux:
//   total_flux += p * 0.25,  with p = ((other - this) * transfer_coeff) * connection_flux
// Explicit round-to-nearest intrinsics pin the floating-point semantics so the
// results are bitwise identical to the reference CPU build. That build (GCC,
// -O3 -march=native, default -ffp-contract=fast) vectorizes the connection
// loop in groups of 4 and 2 without contraction and handles a single leftover
// connection in scalar code where the multiply-add is fused. So the last
// connection of an element with an odd connection count uses an FMA. The
// difference only shows when p * 0.25 is inexact (subnormal results).
__device__ __forceinline__ val_t accumulateFlux(val_t total_flux, val_t p, bool fused) {
    return fused ? __fma_rn(p, 0.25, total_flux) : __dadd_rn(total_flux, __dmul_rn(p, 0.25));
}

// Device-side material: adds transfer_coeff * 0.25 for the fast flux path.
// quarter_coeff is NaN if transfer_coeff * 0.25 is not exactly representable,
// which forces the exact path for every connection of that material.
struct DeviceMaterial {
    val_t transfer_coeff;
    val_t external_flow;
    val_t quarter_coeff;
};

// Flux for unit connection coefficient: ((other - this) * tc) * 0.25.
// Fast path computes d * (tc * 0.25) with a single rounding. Since scaling by
// a power of two commutes with rounding, this is bitwise identical whenever the
// result is a normal double with |r| < 2^1022 (biased exponent in [1, 2044]),
// or d is zero. In those cases p * 0.25 is exact, so fused and unfused
// accumulation agree. Otherwise (underflow, near overflow, inf/NaN) the exact
// reference sequence is used.
__device__ __forceinline__ val_t accumulateUnitFlux(val_t total_flux, val_t transfer_coeff,
                                                    val_t quarter_coeff, val_t d, bool fused) {
    const val_t r = __dmul_rn(d, quarter_coeff);
    const unsigned expo = ((unsigned)__double2hiint(r) >> 20) & 0x7ffu;
    // d == +-0 with finite quarter_coeff gives r == +-0 == p * 0.25 (same sign)
    const bool zero = ((__double_as_longlong(d) | __double_as_longlong(r)) << 1) == 0;
    if (expo - 1u < 2044u || zero) return __dadd_rn(total_flux, r);
    return accumulateFlux(total_flux, __dmul_rn(d, transfer_coeff), fused);
}

// Device-side mesh in structure-of-arrays layout (coalesced access).
// Connection arrays are stored slot-major: slot j of element i is at [j * n + i].
template <typename MatT>
struct DeviceMesh {
    size_t n;
    const uint8_t* num_conn;        // [n]
    const MatT* mat_idx;            // [n]
    const uint32_t* conn_idx;       // [MAX_CONNECTIONS * n]
    const val_t* conn_flux;         // [MAX_CONNECTIONS * n] (unused if UNIT_FLUX)
    const DeviceMaterial* materials; // [n_materials]
};

// One simulation step. Each thread updates one element.
// UNIT_FLUX: all connection flux coefficients are exactly 1.0, so multiplying
// by them is an exact identity and the loads can be skipped.
template <bool UNIT_FLUX, typename MatT>
__global__ void __launch_bounds__(256)
stepKernel(DeviceMesh<MatT> mesh, const val_t* __restrict__ energy_in,
           val_t* __restrict__ energy_out, val_t* __restrict__ total_flux_acc) {
    const size_t n = mesh.n;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    const uint32_t m = __ldg(&mesh.mat_idx[i]);
    const int nc = __ldg(&mesh.num_conn[i]);
    const val_t e = __ldg(&energy_in[i]);
    const val_t tf_old = __ldg(&total_flux_acc[i]);
    const val_t transfer_coeff = __ldg(&mesh.materials[m].transfer_coeff);
    const val_t quarter_coeff = __ldg(&mesh.materials[m].quarter_coeff);

    // Start with external flow
    val_t total_flux = __ldg(&mesh.materials[m].external_flow);

    // Add flux from all connected elements (same order as sequential code)
    const uint32_t* __restrict__ ci = mesh.conn_idx + i;
    const val_t* __restrict__ cf = UNIT_FLUX ? nullptr : mesh.conn_flux + i;
#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nc) {
            const uint32_t nb = __ldg(ci + j * n);
            const val_t d = __dsub_rn(__ldg(&energy_in[nb]), e);
            const bool fused = (j == nc - 1) && (nc & 1);
            if (UNIT_FLUX) {
                // Multiplication by exactly 1.0 is an identity; skip it
                total_flux = accumulateUnitFlux(total_flux, transfer_coeff, quarter_coeff, d, fused);
            } else {
                const val_t p = __dmul_rn(__dmul_rn(d, transfer_coeff), __ldg(cf + j * n));
                total_flux = accumulateFlux(total_flux, p, fused);
            }
        }
    }

    // Update element state
    energy_out[i] = __dadd_rn(e, total_flux);
    total_flux_acc[i] = __dadd_rn(tf_old, fabs(total_flux));
}

template <bool UNIT_FLUX, typename MatT>
static void launchSteps(const DeviceMesh<MatT>& mesh, val_t* d_e0, val_t* d_e1, val_t* d_tf,
                        int n_iters, cudaStream_t stream) {
    constexpr int block = 256;
    const unsigned grid = (unsigned)((mesh.n + block - 1) / block);
    for (int iter = 0; iter < n_iters; ++iter) {
        stepKernel<UNIT_FLUX, MatT><<<grid, block, 0, stream>>>(mesh, d_e0, d_e1, d_tf);
        // Swap buffers
        std::swap(d_e0, d_e1);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Pipelined host<->device staging through double-buffered pinned memory, so
// host-side (un)packing overlaps with PCIe transfers.
class Stager {
public:
    static constexpr size_t MAX_CHUNK_BYTES = size_t(2) << 20;
    // Largest staged per-element footprint of any pass (flux pass: 8 doubles)
    static constexpr size_t MAX_BYTES_PER_ELEM = MAX_CONNECTIONS * sizeof(val_t);
    Stager(cudaStream_t s, size_t n_elems)
        : stream_(s), chunk_bytes_(std::min(MAX_CHUNK_BYTES, std::max<size_t>(n_elems, 1) * MAX_BYTES_PER_ELEM)) {
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMallocHost(&buf_[b], chunk_bytes_));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_[b], cudaEventDisableTiming));
        }
    }
    ~Stager() {
        for (int b = 0; b < 2; ++b) {
            cudaEventSynchronize(ev_[b]);
            cudaFreeHost(buf_[b]);
            cudaEventDestroy(ev_[b]);
        }
    }
    // Number of elements per chunk given the per-element staged byte count
    size_t chunkElems(size_t bytes_per_elem) const { return chunk_bytes_ / bytes_per_elem; }

    // Get a free host buffer (waits until its previous transfer completed)
    char* acquire() {
        cur_ ^= 1;
        CUDA_CHECK(cudaEventSynchronize(ev_[cur_]));
        return buf_[cur_];
    }
    void h2d(void* dst, size_t offset_in_buf, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, buf_[cur_] + offset_in_buf, bytes, cudaMemcpyHostToDevice, stream_));
    }
    void d2h(size_t offset_in_buf, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(buf_[cur_] + offset_in_buf, src, bytes, cudaMemcpyDeviceToHost, stream_));
    }
    void release() { CUDA_CHECK(cudaEventRecord(ev_[cur_], stream_)); }
    void waitCurrent() { CUDA_CHECK(cudaEventSynchronize(ev_[cur_])); }

private:
    cudaStream_t stream_;
    size_t chunk_bytes_;
    char* buf_[2] = {nullptr, nullptr};
    cudaEvent_t ev_[2];
    int cur_ = 1;
};

template <typename MatT>
static void simulateOnDevice(World& world, const int n_iters, cudaStream_t stream, Stager& stager) {
    const size_t n = world.elements_static.size();
    const size_t n_mats = world.materials.size();

    uint8_t* d_num_conn = nullptr;
    MatT* d_mat = nullptr;
    uint32_t* d_conn_idx = nullptr;
    val_t *d_conn_flux = nullptr, *d_e0 = nullptr, *d_e1 = nullptr, *d_tf = nullptr;
    DeviceMaterial* d_mats = nullptr;

    CUDA_CHECK(cudaMalloc(&d_num_conn, n));
    CUDA_CHECK(cudaMalloc(&d_mat, n * sizeof(MatT)));
    CUDA_CHECK(cudaMalloc(&d_conn_idx, MAX_CONNECTIONS * n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_e0, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_e1, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_tf, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mats, std::max<size_t>(n_mats, 1) * sizeof(DeviceMaterial)));
    {
        std::vector<DeviceMaterial> h_mats(n_mats);
        for (size_t k = 0; k < n_mats; ++k) {
            const Material& mat = world.materials[k];
            const val_t q = mat.transfer_coeff * 0.25;
            const bool exact = std::isnormal(q) && q * 4.0 == mat.transfer_coeff;
            h_mats[k] = {mat.transfer_coeff, mat.external_flow,
                         exact ? q : std::numeric_limits<val_t>::quiet_NaN()};
        }
        CUDA_CHECK(cudaMemcpy(d_mats, h_mats.data(), n_mats * sizeof(DeviceMaterial), cudaMemcpyHostToDevice));
    }

    // Pass 1: connectivity, materials and initial dynamic state.
    // Chunk layout: [num_conn C][mat C][energy C][total_flux C][idx slot 0 C]...[idx slot 7 C]
    bool unit_flux = true;
    {
        const size_t bpe = 1 + sizeof(MatT) + 2 * sizeof(val_t) + MAX_CONNECTIONS * sizeof(uint32_t);
        const size_t C = stager.chunkElems(bpe);
        for (size_t base = 0; base < n; base += C) {
            const size_t cnt = std::min(C, n - base);
            char* hb = stager.acquire();
            uint8_t* h_nc = reinterpret_cast<uint8_t*>(hb);
            MatT* h_mat = reinterpret_cast<MatT*>(hb + C);
            val_t* h_e = reinterpret_cast<val_t*>(hb + C * (1 + sizeof(MatT)));
            val_t* h_tf = h_e + C;
            uint32_t* h_idx = reinterpret_cast<uint32_t*>(h_tf + C);
            idx_t max_nc = 0;
            for (size_t k = 0; k < cnt; ++k) {
                const ElementStatic& s = world.elements_static[base + k];
                const ElementDynamic& d = world.elements_dynamic[base + k];
                const idx_t nc = s.num_connections;
                h_nc[k] = (uint8_t)nc;
                h_mat[k] = (MatT)s.material_idx;
                h_e[k] = d.current_energy;
                h_tf[k] = d.total_flux;
                max_nc = std::max(max_nc, nc);
                for (idx_t j = 0; j < nc; ++j) {
                    h_idx[j * C + k] = (uint32_t)s.connected_idx[j];
                    unit_flux &= (s.connected_flux[j] == 1.0);
                }
            }
            stager.h2d(d_num_conn + base, 0, cnt);
            stager.h2d(d_mat + base, C, cnt * sizeof(MatT));
            stager.h2d(d_e0 + base, C * (1 + sizeof(MatT)), cnt * sizeof(val_t));
            stager.h2d(d_tf + base, C * (1 + sizeof(MatT)) + C * sizeof(val_t), cnt * sizeof(val_t));
            const size_t idx_off = reinterpret_cast<char*>(h_idx) - hb;
            for (idx_t j = 0; j < max_nc; ++j)
                stager.h2d(d_conn_idx + j * n + base, idx_off + j * C * sizeof(uint32_t),
                           cnt * sizeof(uint32_t));
            stager.release();
        }
    }

    // Pass 2 (only if needed): non-unit connection flux coefficients.
    if (!unit_flux) {
        CUDA_CHECK(cudaMalloc(&d_conn_flux, MAX_CONNECTIONS * n * sizeof(val_t)));
        const size_t C = stager.chunkElems(MAX_CONNECTIONS * sizeof(val_t));
        for (size_t base = 0; base < n; base += C) {
            const size_t cnt = std::min(C, n - base);
            val_t* h_f = reinterpret_cast<val_t*>(stager.acquire());
            idx_t max_nc = 0;
            for (size_t k = 0; k < cnt; ++k) {
                const ElementStatic& s = world.elements_static[base + k];
                max_nc = std::max(max_nc, s.num_connections);
                for (idx_t j = 0; j < MAX_CONNECTIONS; ++j)
                    h_f[j * C + k] = j < s.num_connections ? s.connected_flux[j] : 0.0;
            }
            for (idx_t j = 0; j < max_nc; ++j)
                stager.h2d(d_conn_flux + j * n + base, j * C * sizeof(val_t), cnt * sizeof(val_t));
            stager.release();
        }
    }

    DeviceMesh<MatT> mesh{n, d_num_conn, d_mat, d_conn_idx, d_conn_flux, d_mats};
    if (unit_flux) launchSteps<true, MatT>(mesh, d_e0, d_e1, d_tf, n_iters, stream);
    else launchSteps<false, MatT>(mesh, d_e0, d_e1, d_tf, n_iters, stream);

    // Final state lives in d_e0 after an even number of steps, else d_e1
    const val_t* d_result = (n_iters > 0 && (n_iters & 1)) ? d_e1 : d_e0;

    // Download results, unpacking each chunk while the next one transfers
    {
        const size_t C = stager.chunkElems(2 * sizeof(val_t));
        size_t prev_base = 0, prev_cnt = 0;
        val_t* prev_buf = nullptr;
        auto unpack = [&](const val_t* hb, size_t base, size_t cnt) {
            for (size_t k = 0; k < cnt; ++k) {
                world.elements_dynamic[base + k].current_energy = hb[k];
                world.elements_dynamic[base + k].total_flux = hb[C + k];
            }
        };
        for (size_t base = 0; base < n; base += C) {
            const size_t cnt = std::min(C, n - base);
            val_t* hb = reinterpret_cast<val_t*>(stager.acquire());
            stager.d2h(0, d_result + base, cnt * sizeof(val_t));
            stager.d2h(C * sizeof(val_t), d_tf + base, cnt * sizeof(val_t));
            stager.release();
            if (prev_buf) unpack(prev_buf, prev_base, prev_cnt);
            prev_buf = nullptr;
            // Wait for this chunk; the unpack of it overlaps the next transfer
            stager.waitCurrent();
            prev_buf = hb; prev_base = base; prev_cnt = cnt;
        }
        if (prev_buf) unpack(prev_buf, prev_base, prev_cnt);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_num_conn); cudaFree(d_mat); cudaFree(d_conn_idx); cudaFree(d_conn_flux);
    cudaFree(d_mats); cudaFree(d_e0); cudaFree(d_e1); cudaFree(d_tf);
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    if (world.elements_static.empty()) return;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        Stager stager(stream, world.elements_static.size());
        if (world.materials.size() <= 256) simulateOnDevice<uint8_t>(world, n_iters, stream, stager);
        else simulateOnDevice<uint32_t>(world, n_iters, stream, stager);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
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
    
    // Initialize the CUDA context up front so its one-time cost is not timed
    CUDA_CHECK(cudaFree(nullptr));

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
