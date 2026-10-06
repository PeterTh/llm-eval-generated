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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Initialize the material table (identical on all ranks)
void buildMaterials(World& world) {
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
}

// Build the part [begin, end) of a 2D square grid represented as an unstructured mesh.
// Element i of the result corresponds to global element begin + i; connected_idx holds
// global element indices. This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(std::vector<ElementStatic>& elems, const int64_t n_elems_root,
                   const int64_t begin, const int64_t end) {
    elems.resize(end - begin);
    const int64_t last = n_elems_root - 1;

    #pragma omp parallel for schedule(static)
    for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t x = idx / n_elems_root;
        const int64_t y = idx % n_elems_root;
        ElementStatic& elem = elems[idx - begin];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];
            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                elem.connected_idx[elem.num_connections] = nx * n_elems_root + ny;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        if ((x == 0 && y == 0) || (x == last && y == last)) elem.material_idx = INFLOW_MAT_ID;
        else if ((x == 0 && y == last) || (x == last && y == 0)) elem.material_idx = OUTFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

constexpr int MAX_MATERIALS = 64;
__constant__ val_t c_transfer_coeff[MAX_MATERIALS];
__constant__ val_t c_external_flow[MAX_MATERIALS];

// Device-side (SoA / ELL) representation of the local partition.
// Local ordering: [0, n_interior) interior elements, [n_interior, n_local) boundary
// elements (those with at least one ghost neighbor), [n_local, n_local + n_ghost) ghosts.
struct DeviceMesh {
    int n_local = 0, n_interior = 0, n_ghost = 0, width = 0, ld = 0;
    int* conn = nullptr;        // [width][ld] local neighbor index
    val_t* cflux = nullptr;     // [width][ld] connection flux coefficient
    uint8_t* ncon = nullptr;    // [n_local]
    uint8_t* mat = nullptr;     // [n_local]
    val_t* energy[2] = {nullptr, nullptr};  // [n_local + n_ghost]
    val_t* flux[2] = {nullptr, nullptr};    // [n_local]
    int* send_idx = nullptr;    // local indices of owned elements to send
    val_t* send_buf = nullptr;
};

template <int WIDTH>
__global__ void __launch_bounds__(256)
updateKernel(const int first, const int last, const int ld,
             const int* __restrict__ conn, const val_t* __restrict__ cflux,
             const uint8_t* __restrict__ ncon, const uint8_t* __restrict__ mat,
             const val_t* __restrict__ e_cur, const val_t* __restrict__ f_cur,
             val_t* __restrict__ e_new, val_t* __restrict__ f_new) {
    const int i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= last) return;

    const int m = mat[i];
    const int nc = ncon[i];
    const val_t coeff = c_transfer_coeff[m];
    const val_t my_e = e_cur[i];

    // Start with external flow
    val_t total_flux = c_external_flow[m];

    // Add flux from all connected elements (same order as the reference)
    #pragma unroll
    for (int j = 0; j < WIDTH; ++j) {
        if (j < nc) {
            const int nb = conn[j * ld + i];
            total_flux += computeFlux(coeff, my_e, cflux[j * ld + i], __ldg(&e_cur[nb]));
        }
    }

    // Update element state
    e_new[i] = my_e + total_flux;
    f_new[i] = f_cur[i] + fabs(total_flux);
}

__global__ void packKernel(const int n, const int* __restrict__ idx,
                           const val_t* __restrict__ e, val_t* __restrict__ out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) out[k] = e[idx[k]];
}

static void launchUpdate(const DeviceMesh& dm, int first, int last, int cur, cudaStream_t s) {
    if (last <= first) return;
    constexpr int BS = 256;
    const int blocks = (last - first + BS - 1) / BS;
    const int nxt = cur ^ 1;
#define LAUNCH_W(W)                                                                     \
    updateKernel<W><<<blocks, BS, 0, s>>>(first, last, dm.ld, dm.conn, dm.cflux, dm.ncon, \
                                          dm.mat, dm.energy[cur], dm.flux[cur],         \
                                          dm.energy[nxt], dm.flux[nxt])
    if (dm.width <= 4) LAUNCH_W(4);
    else LAUNCH_W(MAX_CONNECTIONS);
#undef LAUNCH_W
}

// Halo exchange plan (host side)
struct HaloPlan {
    std::vector<int> send_ranks, send_counts, send_displs;
    std::vector<int> recv_ranks, recv_counts, recv_displs;
    std::vector<int> send_local_idx;  // local indices of owned elements to send
    int n_send = 0, n_recv = 0;
};

// Distributed simulation state
struct Partition {
    int64_t begin = 0, end = 0;
    std::vector<int64_t> local_to_global;  // owned elements, local order
    HaloPlan halo;
    DeviceMesh dm;
    val_t* h_send = nullptr;  // pinned
    val_t* h_recv = nullptr;  // pinned
    cudaStream_t s_comp, s_comm;
    cudaEvent_t ev_comp, ev_comm;
};

// Set up the local partition: build local mesh, discover halo, upload to GPU.
void setupPartition(World& world, Partition& p, const int64_t n_elems_root,
                    const std::vector<int64_t>& offsets, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    p.begin = offsets[rank];
    p.end = offsets[rank + 1];
    const int64_t n_own = p.end - p.begin;

    buildMaterials(world);
    if (world.materials.size() > (size_t)MAX_MATERIALS) {
        fprintf(stderr, "Too many materials\n");
        MPI_Abort(comm, 1);
    }
    std::vector<ElementStatic>& es = world.elements_static;
    buildSquare2D(es, n_elems_root, p.begin, p.end);

    // Discover ghosts (neighbors owned by other ranks) and boundary elements
    std::vector<uint8_t> is_boundary(n_own, 0);
    std::vector<int64_t> ghosts;
    int width = 0;
    for (int64_t i = 0; i < n_own; ++i) {
        const ElementStatic& e = es[i];
        width = std::max(width, (int)e.num_connections);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const int64_t g = (int64_t)e.connected_idx[j];
            if (g < p.begin || g >= p.end) {
                is_boundary[i] = 1;
                ghosts.push_back(g);
            }
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    const int n_ghost = (int)ghosts.size();

    // Local ordering: interior first, then boundary (each in global order)
    std::vector<int> global_to_local(n_own);
    p.local_to_global.clear();
    p.local_to_global.reserve(n_own);
    for (int64_t i = 0; i < n_own; ++i)
        if (!is_boundary[i]) p.local_to_global.push_back(p.begin + i);
    const int n_interior = (int)p.local_to_global.size();
    for (int64_t i = 0; i < n_own; ++i)
        if (is_boundary[i]) p.local_to_global.push_back(p.begin + i);
    for (int64_t l = 0; l < n_own; ++l) global_to_local[p.local_to_global[l] - p.begin] = (int)l;

    auto ownerOf = [&](int64_t g) {
        return (int)(std::upper_bound(offsets.begin(), offsets.end(), g) - offsets.begin()) - 1;
    };
    auto localOf = [&](int64_t g) -> int {
        if (g >= p.begin && g < p.end) return global_to_local[g - p.begin];
        return (int)n_own + (int)(std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin());
    };

    // Requests: ghosts are sorted by global index, hence grouped by owner rank
    std::vector<int> req_counts(size, 0), offer_counts(size, 0);
    for (int64_t g : ghosts) req_counts[ownerOf(g)]++;
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, offer_counts.data(), 1, MPI_INT, comm);
    std::vector<int> req_displs(size, 0), offer_displs(size, 0);
    for (int r = 1; r < size; ++r) {
        req_displs[r] = req_displs[r - 1] + req_counts[r - 1];
        offer_displs[r] = offer_displs[r - 1] + offer_counts[r - 1];
    }
    const int n_offer = offer_displs[size - 1] + offer_counts[size - 1];
    std::vector<int64_t> offered(n_offer);
    MPI_Alltoallv(ghosts.data(), req_counts.data(), req_displs.data(), MPI_INT64_T,
                  offered.data(), offer_counts.data(), offer_displs.data(), MPI_INT64_T, comm);

    HaloPlan& h = p.halo;
    for (int r = 0; r < size; ++r) {
        if (req_counts[r] > 0) {
            h.recv_ranks.push_back(r);
            h.recv_counts.push_back(req_counts[r]);
            h.recv_displs.push_back(req_displs[r]);
        }
        if (offer_counts[r] > 0) {
            h.send_ranks.push_back(r);
            h.send_counts.push_back(offer_counts[r]);
            h.send_displs.push_back(offer_displs[r]);
        }
    }
    h.n_send = n_offer;
    h.n_recv = n_ghost;
    h.send_local_idx.resize(n_offer);
    for (int k = 0; k < n_offer; ++k) h.send_local_idx[k] = localOf(offered[k]);

    // Build SoA / ELL arrays in local ordering
    DeviceMesh& dm = p.dm;
    dm.n_local = (int)n_own;
    dm.n_interior = n_interior;
    dm.n_ghost = n_ghost;
    dm.width = std::max(width, 1);
    dm.ld = std::max<int>(((int)n_own + 31) / 32 * 32, 32);
    const size_t ell = (size_t)dm.width * dm.ld;
    std::vector<int> h_conn(ell, 0);
    std::vector<val_t> h_cflux(ell, 0.0);
    std::vector<uint8_t> h_ncon(n_own), h_mat(n_own);
    #pragma omp parallel for schedule(static)
    for (int64_t l = 0; l < n_own; ++l) {
        const ElementStatic& e = es[p.local_to_global[l] - p.begin];
        h_ncon[l] = (uint8_t)e.num_connections;
        h_mat[l] = (uint8_t)e.material_idx;
        for (idx_t j = 0; j < e.num_connections; ++j) {
            h_conn[j * dm.ld + l] = localOf((int64_t)e.connected_idx[j]);
            h_cflux[j * dm.ld + l] = e.connected_flux[j];
        }
    }

    std::vector<val_t> tc(world.materials.size()), ef(world.materials.size());
    for (size_t m = 0; m < world.materials.size(); ++m) {
        tc[m] = world.materials[m].transfer_coeff;
        ef[m] = world.materials[m].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_transfer_coeff, tc.data(), tc.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_external_flow, ef.data(), ef.size() * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&dm.conn, ell * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dm.cflux, ell * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dm.ncon, std::max<int64_t>(n_own, 1)));
    CUDA_CHECK(cudaMalloc(&dm.mat, std::max<int64_t>(n_own, 1)));
    CUDA_CHECK(cudaMemcpy(dm.conn, h_conn.data(), ell * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.cflux, h_cflux.data(), ell * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.ncon, h_ncon.data(), n_own, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.mat, h_mat.data(), n_own, cudaMemcpyHostToDevice));
    const size_t n_e = std::max<size_t>(n_own + n_ghost, 1);
    for (int b = 0; b < 2; ++b) {
        // All elements start with zero energy and zero accumulated flux
        CUDA_CHECK(cudaMalloc(&dm.energy[b], n_e * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dm.flux[b], n_e * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dm.energy[b], 0, n_e * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dm.flux[b], 0, n_e * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&dm.send_idx, std::max(n_offer, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dm.send_buf, std::max(n_offer, 1) * sizeof(val_t)));
    if (n_offer > 0)
        CUDA_CHECK(cudaMemcpy(dm.send_idx, h.send_local_idx.data(), n_offer * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&p.h_send, std::max(n_offer, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&p.h_recv, std::max(n_ghost, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&p.s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&p.s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&p.ev_comp, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&p.ev_comm, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Static data now lives on the device
    std::vector<ElementStatic>().swap(es);
}

void freePartition(Partition& p) {
    DeviceMesh& dm = p.dm;
    cudaFree(dm.conn); cudaFree(dm.cflux); cudaFree(dm.ncon); cudaFree(dm.mat);
    for (int b = 0; b < 2; ++b) { cudaFree(dm.energy[b]); cudaFree(dm.flux[b]); }
    cudaFree(dm.send_idx); cudaFree(dm.send_buf);
    cudaFreeHost(p.h_send); cudaFreeHost(p.h_recv);
    cudaStreamDestroy(p.s_comp); cudaStreamDestroy(p.s_comm);
    cudaEventDestroy(p.ev_comp); cudaEventDestroy(p.ev_comm);
}

// Run simulation for n_iters iterations; returns index of the buffer holding the result
int runSimulation(Partition& p, const int n_iters, MPI_Comm comm) {
    DeviceMesh& dm = p.dm;
    HaloPlan& h = p.halo;
    const bool has_halo = (h.n_send > 0 || h.n_recv > 0);
    std::vector<MPI_Request> reqs(h.send_ranks.size() + h.recv_ranks.size());
    int cur = 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (!has_halo) {
            launchUpdate(dm, 0, dm.n_local, cur, p.s_comp);
            cur ^= 1;
            continue;
        }

        // Gather outgoing boundary energies (written by previous boundary update on s_comm)
        if (h.n_send > 0) {
            packKernel<<<(h.n_send + 255) / 256, 256, 0, p.s_comm>>>(h.n_send, dm.send_idx,
                                                                   dm.energy[cur], dm.send_buf);
            CUDA_CHECK(cudaMemcpyAsync(p.h_send, dm.send_buf, h.n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, p.s_comm));
        }

        // Interior update overlaps with the halo exchange
        launchUpdate(dm, 0, dm.n_interior, cur, p.s_comp);
        CUDA_CHECK(cudaEventRecord(p.ev_comp, p.s_comp));

        int nr = 0;
        for (size_t k = 0; k < h.recv_ranks.size(); ++k)
            MPI_Irecv(p.h_recv + h.recv_displs[k], h.recv_counts[k], MPI_DOUBLE,
                      h.recv_ranks[k], 0, comm, &reqs[nr++]);
        CUDA_CHECK(cudaStreamSynchronize(p.s_comm));
        for (size_t k = 0; k < h.send_ranks.size(); ++k)
            MPI_Isend(p.h_send + h.send_displs[k], h.send_counts[k], MPI_DOUBLE,
                      h.send_ranks[k], 0, comm, &reqs[nr++]);
        MPI_Waitall(nr, reqs.data(), MPI_STATUSES_IGNORE);

        if (h.n_recv > 0)
            CUDA_CHECK(cudaMemcpyAsync(dm.energy[cur] + dm.n_local, p.h_recv,
                                       h.n_recv * sizeof(val_t), cudaMemcpyHostToDevice,
                                       p.s_comm));
        // Boundary update must not overwrite data still read by the interior update
        CUDA_CHECK(cudaStreamWaitEvent(p.s_comm, p.ev_comp, 0));
        launchUpdate(dm, dm.n_interior, dm.n_local, cur, p.s_comm);
        CUDA_CHECK(cudaEventRecord(p.ev_comm, p.s_comm));
        // Next interior update reads boundary values written here
        CUDA_CHECK(cudaStreamWaitEvent(p.s_comp, p.ev_comm, 0));

        // Swap buffers
        cur ^= 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(p.s_comp));
    CUDA_CHECK(cudaStreamSynchronize(p.s_comm));
    CUDA_CHECK(cudaGetLastError());
    return cur;
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

// Compute a simple hash of the results for verification.
// elements[k] is global element first_idx + k; XOR combination is order independent.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const uint64_t first_idx = 0) {
    uint64_t hash = 0;
    const int64_t n = (int64_t)elements.size();
    #pragma omp parallel for reduction(^ : hash) schedule(static)
    for (int64_t k = 0; k < n; ++k) {
        const uint64_t i = first_idx + (uint64_t)k;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &elements[k].current_energy, sizeof(e_bits));
        memcpy(&f_bits, &elements[k].total_flux, sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // One GPU per rank, assigned round-robin among the ranks of a node
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int local_rank;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev <= 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % n_dev));
        CUDA_CHECK(cudaFree(0));
    }

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
    const int64_t n_elems_total = (int64_t)n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    
    // Partition elements into contiguous blocks (one per rank)
    std::vector<int64_t> offsets(size + 1);
    for (int r = 0; r <= size; ++r) offsets[r] = n_elems_total * r / size;

    // Build the local part of the unstructured mesh and set up the halo exchange
    World world;
    Partition part;
    setupPartition(world, part, n_elems_root, offsets, MPI_COMM_WORLD);
    
    // Calculate memory usage (of the full mesh)
    const size_t static_mem = (size_t)n_elems_total * sizeof(ElementStatic);
    const size_t dynamic_mem = (size_t)n_elems_total * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
        fflush(stdout);
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    const int result_buf = runSimulation(part, n_iters, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Fetch local results and restore global element order
    const int n_local = part.dm.n_local;
    std::vector<val_t> h_energy(std::max(n_local, 1)), h_flux(std::max(n_local, 1));
    CUDA_CHECK(cudaMemcpy(h_energy.data(), part.dm.energy[result_buf], n_local * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), part.dm.flux[result_buf], n_local * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    world.elements_dynamic.resize(n_local);
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < n_local; ++l) {
        ElementDynamic& d = world.elements_dynamic[part.local_to_global[l] - part.begin];
        d.current_energy = h_energy[l];
        d.total_flux = h_flux[l];
    }
    
    // Compute hash for verification
    const uint64_t local_hash = computeHash(world.elements_dynamic, (uint64_t)part.begin);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
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
    
    // Gather the full state on rank 0 for result printing / validation
    bool valid = true;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (int)((offsets[r + 1] - offsets[r]) * 2);
            displs[r] = (int)(offsets[r] * 2);
        }
        World global;
        if (rank == 0) global.elements_dynamic.resize(n_elems_total);
        static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "unexpected layout");
        MPI_Gatherv(world.elements_dynamic.data(), n_local * 2, MPI_DOUBLE,
                    rank == 0 ? global.elements_dynamic.data() : nullptr, counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(global.elements_dynamic.size());
                for (const auto& elem : global.elements_dynamic) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }
            
            // Validation
            if (validate) {
                valid = validateResults(global);
            }
        }
    }
    
    freePartition(part);
    MPI_Finalize();
    return valid ? 0 : 1;
}
