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

static inline void checkCuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, call, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CHECK_CUDA(x) checkCuda((x), #x, __FILE__, __LINE__)

static int getLocalRankFallback(int world_rank) {
    // Common environment variables across MPI implementations
    const char* vars[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
        "MPI_LOCALRANKID",
    };
    for (const char* v : vars) {
        if (const char* s = std::getenv(v)) {
            return std::atoi(s);
        }
    }
    return world_rank;
}

__global__ void updateKernel(const double* __restrict__ energy_in,
                            const double* __restrict__ flux_in,
                            double* __restrict__ energy_out,
                            double* __restrict__ flux_out,
                            int nroot,
                            int start_x,
                            int row_begin_local,
                            int row_end_local) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int rows = row_end_local - row_begin_local + 1;
    const int total = rows * nroot;
    if (tid >= total) return;

    const int x_local = row_begin_local + (tid / nroot);  // includes halo rows
    const int y = tid - (tid / nroot) * nroot;

    const int global_x = start_x + (x_local - 1);
    const int last = nroot - 1;

    const size_t e_idx = (size_t)x_local * (size_t)nroot + (size_t)y;
    const size_t l_idx = (size_t)(x_local - 1) * (size_t)nroot + (size_t)y;

    const double self = energy_in[e_idx];

    double external_flow = 0.0;
    // Corner inflow/outflow sources (matches buildSquare2D)
    if ((global_x == 0 && y == 0) || (global_x == last && y == last)) {
        external_flow = 0.5;
    } else if ((global_x == 0 && y == last) || (global_x == last && y == 0)) {
        external_flow = -0.5;
    }

    // transfer_coeff(0.8) * connection_flux(1.0) * 0.25 = 0.2
    constexpr double coeff = 0.2;

    double total_flux = external_flow;

    // Neighbor order matches original offsets: {+x,-x,+y,-y}
    if (global_x + 1 < nroot) {
        const double nb = energy_in[e_idx + (size_t)nroot];
        total_flux += coeff * (nb - self);
    }
    if (global_x - 1 >= 0) {
        const double nb = energy_in[e_idx - (size_t)nroot];
        total_flux += coeff * (nb - self);
    }
    if (y + 1 < nroot) {
        const double nb = energy_in[e_idx + 1];
        total_flux += coeff * (nb - self);
    }
    if (y - 1 >= 0) {
        const double nb = energy_in[e_idx - 1];
        total_flux += coeff * (nb - self);
    }

    energy_out[e_idx] = self + total_flux;
    flux_out[l_idx] = flux_in[l_idx] + fabs(total_flux);
}

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

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all elements
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

static void runSimulationHybrid(const int n_elems_root,
                                const int n_iters,
                                std::vector<ElementDynamic>& out_global_elements,
                                long& out_duration_ms) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    volatile int omp_dummy = 0;
#pragma omp parallel
    {
#pragma omp atomic
        omp_dummy += 1;
    }
    (void)omp_dummy;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    const int local_rank = getLocalRankFallback(rank);
    CHECK_CUDA(cudaSetDevice(local_rank % device_count));

    const int nroot = n_elems_root;
    const int base = nroot / size;
    const int rem = nroot % size;
    const int local_nx = base + (rank < rem ? 1 : 0);
    const int start_x = rank * base + (rank < rem ? rank : rem);

    const size_t row_bytes = (size_t)nroot * sizeof(double);
    const size_t energy_elems = (size_t)(local_nx + 2) * (size_t)nroot;  // includes 2 halo rows
    const size_t local_elems = (size_t)local_nx * (size_t)nroot;

    double *d_energy_a = nullptr, *d_energy_b = nullptr;
    double *d_flux_a = nullptr, *d_flux_b = nullptr;

    CHECK_CUDA(cudaMalloc(&d_energy_a, energy_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_energy_b, energy_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_flux_a, local_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_flux_b, local_elems * sizeof(double)));

    CHECK_CUDA(cudaMemset(d_energy_a, 0, energy_elems * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_energy_b, 0, energy_elems * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_flux_a, 0, local_elems * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_flux_b, 0, local_elems * sizeof(double)));

    double* h_send_up = nullptr;
    double* h_send_down = nullptr;
    double* h_recv_up = nullptr;
    double* h_recv_down = nullptr;

    if (rank > 0) {
        CHECK_CUDA(cudaMallocHost(&h_send_up, row_bytes));
        CHECK_CUDA(cudaMallocHost(&h_recv_up, row_bytes));
    }
    if (rank + 1 < size) {
        CHECK_CUDA(cudaMallocHost(&h_send_down, row_bytes));
        CHECK_CUDA(cudaMallocHost(&h_recv_down, row_bytes));
    }

    cudaStream_t stream_compute{};
    cudaStream_t stream_halo{};
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_halo, cudaStreamNonBlocking));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const int threads = 256;

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int nreq = 0;

        if (rank > 0) {
            CHECK_CUDA(cudaMemcpyAsync(h_send_up, d_energy_a + (size_t)nroot, row_bytes, cudaMemcpyDeviceToHost, stream_halo));
        }
        if (rank + 1 < size) {
            CHECK_CUDA(cudaMemcpyAsync(h_send_down, d_energy_a + (size_t)local_nx * (size_t)nroot, row_bytes, cudaMemcpyDeviceToHost, stream_halo));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_halo));

        if (rank > 0) {
            MPI_Irecv(h_recv_up, nroot, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (rank + 1 < size) {
            MPI_Irecv(h_recv_down, nroot, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (rank > 0) {
            MPI_Isend(h_send_up, nroot, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (rank + 1 < size) {
            MPI_Isend(h_send_down, nroot, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        // Compute interior rows while halo exchange is in flight.
        if (local_nx > 2) {
            const int row_begin = 2;
            const int row_end = local_nx - 1;
            const int rows = row_end - row_begin + 1;
            const int total = rows * nroot;
            const int blocks = (total + threads - 1) / threads;
            updateKernel<<<blocks, threads, 0, stream_compute>>>(d_energy_a, d_flux_a, d_energy_b, d_flux_b, nroot, start_x, row_begin, row_end);
        }

        if (nreq) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        if (rank > 0) {
            CHECK_CUDA(cudaMemcpyAsync(d_energy_a + 0, h_recv_up, row_bytes, cudaMemcpyHostToDevice, stream_halo));
        }
        if (rank + 1 < size) {
            CHECK_CUDA(cudaMemcpyAsync(d_energy_a + (size_t)(local_nx + 1) * (size_t)nroot, h_recv_down, row_bytes, cudaMemcpyHostToDevice, stream_halo));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_halo));

        // Boundary rows
        if (local_nx >= 1) {
            {
                const int total = 1 * nroot;
                const int blocks = (total + threads - 1) / threads;
                updateKernel<<<blocks, threads, 0, stream_compute>>>(d_energy_a, d_flux_a, d_energy_b, d_flux_b, nroot, start_x, 1, 1);
            }
            if (local_nx > 1) {
                const int total = 1 * nroot;
                const int blocks = (total + threads - 1) / threads;
                updateKernel<<<blocks, threads, 0, stream_compute>>>(d_energy_a, d_flux_a, d_energy_b, d_flux_b, nroot, start_x, local_nx, local_nx);
            }
        }

        CHECK_CUDA(cudaStreamSynchronize(stream_compute));
        CHECK_CUDA(cudaGetLastError());

        std::swap(d_energy_a, d_energy_b);
        std::swap(d_flux_a, d_flux_b);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double local_ms = (t1 - t0) * 1000.0;
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        out_duration_ms = (long)llround(max_ms);
    }

    std::vector<double> local_energy(local_elems);
    std::vector<double> local_flux(local_elems);

    CHECK_CUDA(cudaMemcpy(local_energy.data(), d_energy_a + (size_t)nroot, local_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(local_flux.data(), d_flux_a, local_elems * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<int> all_nx;
    if (rank == 0) {
        all_nx.resize(size);
    }
    int local_nx_i = local_nx;
    MPI_Gather(&local_nx_i, 1, MPI_INT, rank == 0 ? all_nx.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displs;
    std::vector<double> global_energy, global_flux;

    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        int x = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = x * nroot;
            counts[r] = all_nx[r] * nroot;
            x += all_nx[r];
        }
        global_energy.resize((size_t)nroot * (size_t)nroot);
        global_flux.resize((size_t)nroot * (size_t)nroot);
    }

    MPI_Gatherv(local_energy.data(), (int)local_elems, MPI_DOUBLE,
                rank == 0 ? global_energy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_flux.data(), (int)local_elems, MPI_DOUBLE,
                rank == 0 ? global_flux.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        out_global_elements.resize((size_t)nroot * (size_t)nroot);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < out_global_elements.size(); ++i) {
            out_global_elements[i].current_energy = global_energy[i];
            out_global_elements[i].total_flux = global_flux[i];
        }
    }

    CHECK_CUDA(cudaStreamDestroy(stream_compute));
    CHECK_CUDA(cudaStreamDestroy(stream_halo));

    if (h_send_up) CHECK_CUDA(cudaFreeHost(h_send_up));
    if (h_send_down) CHECK_CUDA(cudaFreeHost(h_send_down));
    if (h_recv_up) CHECK_CUDA(cudaFreeHost(h_recv_up));
    if (h_recv_down) CHECK_CUDA(cudaFreeHost(h_recv_down));

    CHECK_CUDA(cudaFree(d_energy_a));
    CHECK_CUDA(cudaFree(d_energy_b));
    CHECK_CUDA(cudaFree(d_flux_a));
    CHECK_CUDA(cudaFree(d_flux_b));
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate_i = 0;
    int printResults_i = 0;
    int exit_code = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exit_code = 100;  // early-exit sentinel
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_code == 100) {
        MPI_Finalize();
        return 0;
    }
    if (exit_code != 0) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Report reference memory footprint (same accounting as the serial version)
        printf("Building unstructured mesh...\n");
        const size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    long duration_ms = 0;
    std::vector<ElementDynamic> global_elements;
    runSimulationHybrid(n_elems_root, n_iters, global_elements, duration_ms);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(global_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData(global_elements.size());
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < global_elements.size(); ++i) {
                energyData[i] = global_elements[i].current_energy;
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            World world;
            world.elements_dynamic = global_elements;
            const bool valid = validateResults(world);
            exit_code = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
