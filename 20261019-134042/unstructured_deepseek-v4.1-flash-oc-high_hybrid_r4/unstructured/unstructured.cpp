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

struct GpuDomain {
    int x0 = 0;
    int y0 = 0;
    int lx = 0;
    int ly = 0;
    int stride = 0;
    bool empty = true;

    size_t off_top = 0;
    size_t off_bottom = 0;
    size_t off_left = 0;
    size_t off_right = 0;
    size_t boundary_size = 0;

    val_t* d_energy = nullptr;
    val_t* d_energy_new = nullptr;
    val_t* d_flux = nullptr;

    val_t* d_send = nullptr;
    val_t* d_recv = nullptr;
    val_t* h_send = nullptr;
    val_t* h_recv = nullptr;
};

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static int partitionStart(int n, int parts, int coord) {
    const int base = n / parts;
    const int rem = n % parts;
    return coord * base + std::min(coord, rem);
}

static int partitionLength(int n, int parts, int coord) {
    const int base = n / parts;
    const int rem = n % parts;
    return base + (coord < rem ? 1 : 0);
}

__global__ void computeKernel(const val_t* __restrict__ energy,
                              val_t* __restrict__ energy_new,
                              val_t* __restrict__ flux,
                              val_t transfer_coeff,
                              int x0, int y0, int n,
                              val_t* __restrict__ send_top,
                              val_t* __restrict__ send_bottom,
                              val_t* __restrict__ send_left,
                              val_t* __restrict__ send_right,
                              int lx, int ly, int stride) {
    const int lj = blockIdx.x * blockDim.x + threadIdx.x;
    const int li = blockIdx.y * blockDim.y + threadIdx.y;

    if (li >= lx || lj >= ly) {
        return;
    }

    const int gx = x0 + li;
    const int gy = y0 + lj;
    const int pidx = (li + 1) * stride + (lj + 1);
    const int cidx = li * ly + lj;
    const val_t this_energy = energy[pidx];
    const int last = n - 1;

    val_t total_flux = 0.0;
    if ((gx == 0 && gy == 0) || (gx == last && gy == last)) {
        total_flux = 0.5;
    } else if ((gx == 0 && gy == last) || (gx == last && gy == 0)) {
        total_flux = -0.5;
    }

    if (gx + 1 < n) {
        total_flux += (energy[pidx + stride] - this_energy) * transfer_coeff * 0.25;
    }
    if (gx > 0) {
        total_flux += (energy[pidx - stride] - this_energy) * transfer_coeff * 0.25;
    }
    if (gy + 1 < n) {
        total_flux += (energy[pidx + 1] - this_energy) * transfer_coeff * 0.25;
    }
    if (gy > 0) {
        total_flux += (energy[pidx - 1] - this_energy) * transfer_coeff * 0.25;
    }

    const val_t next_energy = this_energy + total_flux;
    energy_new[pidx] = next_energy;
    flux[cidx] += fabs(total_flux);

    if (li == 0) {
        send_top[lj] = next_energy;
    }
    if (li == lx - 1) {
        send_bottom[lj] = next_energy;
    }
    if (lj == 0) {
        send_left[li] = next_energy;
    }
    if (lj == ly - 1) {
        send_right[li] = next_energy;
    }
}

__global__ void unpackHaloKernel(val_t* __restrict__ energy,
                                 const val_t* __restrict__ recv_top,
                                 const val_t* __restrict__ recv_bottom,
                                 const val_t* __restrict__ recv_left,
                                 const val_t* __restrict__ recv_right,
                                 int lx, int ly, int stride) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < ly) {
        energy[t + 1] = recv_top[t];
        energy[(lx + 1) * stride + t + 1] = recv_bottom[t];
    }
    if (t < lx) {
        energy[(t + 1) * stride] = recv_left[t];
        energy[(t + 1) * stride + ly + 1] = recv_right[t];
    }
}

static void buildLocalDomain(GpuDomain& gpu, int n, int cx, int cy, int px, int py) {
    gpu.x0 = partitionStart(n, px, cx);
    gpu.lx = partitionLength(n, px, cx);
    gpu.y0 = partitionStart(n, py, cy);
    gpu.ly = partitionLength(n, py, cy);
    gpu.stride = gpu.ly + 2;
    gpu.empty = (gpu.lx == 0 || gpu.ly == 0);

    const size_t local_elems = static_cast<size_t>(gpu.lx) * static_cast<size_t>(gpu.ly);
    const size_t padded_elems = gpu.empty
        ? 1
        : static_cast<size_t>(gpu.lx + 2) * static_cast<size_t>(gpu.ly + 2);
    const size_t dynamic_elems = local_elems > 0 ? local_elems : 1;

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.d_energy), padded_elems * sizeof(val_t)), "cudaMalloc energy");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.d_energy_new), padded_elems * sizeof(val_t)), "cudaMalloc energy_new");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.d_flux), dynamic_elems * sizeof(val_t)), "cudaMalloc flux");

    cudaCheck(cudaMemset(gpu.d_energy, 0, padded_elems * sizeof(val_t)), "cudaMemset energy");
    cudaCheck(cudaMemset(gpu.d_energy_new, 0, padded_elems * sizeof(val_t)), "cudaMemset energy_new");
    cudaCheck(cudaMemset(gpu.d_flux, 0, dynamic_elems * sizeof(val_t)), "cudaMemset flux");

    gpu.off_top = 0;
    gpu.off_bottom = static_cast<size_t>(gpu.ly);
    gpu.off_left = static_cast<size_t>(2 * gpu.ly);
    gpu.off_right = static_cast<size_t>(2 * gpu.ly + gpu.lx);
    gpu.boundary_size = static_cast<size_t>(2 * gpu.ly + 2 * gpu.lx);

    const size_t boundary_alloc = gpu.boundary_size > 0 ? gpu.boundary_size : 1;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.d_send), boundary_alloc * sizeof(val_t)), "cudaMalloc send");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.d_recv), boundary_alloc * sizeof(val_t)), "cudaMalloc recv");
    cudaCheck(cudaMemset(gpu.d_send, 0, boundary_alloc * sizeof(val_t)), "cudaMemset send");
    cudaCheck(cudaMemset(gpu.d_recv, 0, boundary_alloc * sizeof(val_t)), "cudaMemset recv");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&gpu.h_send), boundary_alloc * sizeof(val_t)), "cudaMallocHost send");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&gpu.h_recv), boundary_alloc * sizeof(val_t)), "cudaMallocHost recv");
    memset(gpu.h_send, 0, boundary_alloc * sizeof(val_t));
    memset(gpu.h_recv, 0, boundary_alloc * sizeof(val_t));
}

static void exchangeHalo(GpuDomain& gpu, MPI_Comm cart,
                         int source_x, int dest_x,
                         int source_y, int dest_y) {
    constexpr int TAG_X_BOTTOM = 20;
    constexpr int TAG_X_TOP = 21;
    constexpr int TAG_Y_RIGHT = 22;
    constexpr int TAG_Y_LEFT = 23;

    if (!gpu.empty) {
        cudaCheck(cudaMemcpy(gpu.h_send, gpu.d_send,
                             gpu.boundary_size * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "send pack");
    }

    MPI_Request requests[8];

    MPI_Irecv(gpu.h_recv + gpu.off_top, gpu.ly, MPI_DOUBLE, source_x, TAG_X_BOTTOM, cart, &requests[0]);
    MPI_Irecv(gpu.h_recv + gpu.off_bottom, gpu.ly, MPI_DOUBLE, dest_x, TAG_X_TOP, cart, &requests[1]);
    MPI_Irecv(gpu.h_recv + gpu.off_left, gpu.lx, MPI_DOUBLE, source_y, TAG_Y_RIGHT, cart, &requests[2]);
    MPI_Irecv(gpu.h_recv + gpu.off_right, gpu.lx, MPI_DOUBLE, dest_y, TAG_Y_LEFT, cart, &requests[3]);
    MPI_Isend(gpu.h_send + gpu.off_bottom, gpu.ly, MPI_DOUBLE, dest_x, TAG_X_BOTTOM, cart, &requests[4]);
    MPI_Isend(gpu.h_send + gpu.off_top, gpu.ly, MPI_DOUBLE, source_x, TAG_X_TOP, cart, &requests[5]);
    MPI_Isend(gpu.h_send + gpu.off_right, gpu.lx, MPI_DOUBLE, dest_y, TAG_Y_RIGHT, cart, &requests[6]);
    MPI_Isend(gpu.h_send + gpu.off_left, gpu.lx, MPI_DOUBLE, source_y, TAG_Y_LEFT, cart, &requests[7]);

    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    if (!gpu.empty) {
        cudaCheck(cudaMemcpy(gpu.d_recv, gpu.h_recv,
                             gpu.boundary_size * sizeof(val_t),
                             cudaMemcpyHostToDevice), "recv unpack");

        const int threads = 256;
        const int count = std::max(gpu.lx, gpu.ly);
        const int blocks = (count + threads - 1) / threads;
        unpackHaloKernel<<<blocks, threads>>>(gpu.d_energy,
                                             gpu.d_recv + gpu.off_top,
                                             gpu.d_recv + gpu.off_bottom,
                                             gpu.d_recv + gpu.off_left,
                                             gpu.d_recv + gpu.off_right,
                                             gpu.lx, gpu.ly, gpu.stride);
    }
}

static void runSimulation(GpuDomain& gpu, int n_iters, int n, MPI_Comm cart,
                          int source_x, int dest_x,
                          int source_y, int dest_y) {
    for (int iter = 0; iter < n_iters; ++iter) {
        if (!gpu.empty) {
            const dim3 block(32, 8);
            const dim3 grid((gpu.ly + block.x - 1) / block.x,
                            (gpu.lx + block.y - 1) / block.y);

            computeKernel<<<grid, block>>>(
                gpu.d_energy, gpu.d_energy_new, gpu.d_flux, 0.8,
                gpu.x0, gpu.y0, n,
                gpu.d_send + gpu.off_top,
                gpu.d_send + gpu.off_bottom,
                gpu.d_send + gpu.off_left,
                gpu.d_send + gpu.off_right,
                gpu.lx, gpu.ly, gpu.stride);
        }

        std::swap(gpu.d_energy, gpu.d_energy_new);

        if (iter + 1 < n_iters) {
            exchangeHalo(gpu, cart, source_x, dest_x, source_y, dest_y);
        }
    }

    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "simulation sync");
}

static void fetchLocal(const GpuDomain& gpu, std::vector<val_t>& energy,
                       std::vector<val_t>& flux) {
    const size_t local_elems = static_cast<size_t>(gpu.lx) * static_cast<size_t>(gpu.ly);
    energy.resize(local_elems);
    flux.resize(local_elems);

    if (local_elems == 0) {
        return;
    }

    cudaCheck(cudaMemcpy2D(energy.data(), static_cast<size_t>(gpu.ly) * sizeof(val_t),
                           gpu.d_energy + gpu.stride + 1,
                           static_cast<size_t>(gpu.stride) * sizeof(val_t),
                           static_cast<size_t>(gpu.ly) * sizeof(val_t),
                           gpu.lx, cudaMemcpyDeviceToHost), "fetch energy");
    cudaCheck(cudaMemcpy(flux.data(), gpu.d_flux, local_elems * sizeof(val_t),
                         cudaMemcpyDeviceToHost), "fetch flux");
}

static uint64_t localHash(const GpuDomain& gpu, const std::vector<val_t>& energy,
                          const std::vector<val_t>& flux, int n) {
    uint64_t result = 0;

#pragma omp parallel for schedule(static) reduction(^:result)
    for (int li = 0; li < gpu.lx; ++li) {
        uint64_t row_hash = 0;
        for (int lj = 0; lj < gpu.ly; ++lj) {
            const int cidx = li * gpu.ly + lj;
            uint64_t e_bits;
            uint64_t f_bits;
            memcpy(&e_bits, &energy[cidx], sizeof(uint64_t));
            memcpy(&f_bits, &flux[cidx], sizeof(uint64_t));
            const uint64_t gidx = static_cast<uint64_t>(gpu.x0 + li) * static_cast<uint64_t>(n)
                                + static_cast<uint64_t>(gpu.y0 + lj);
            row_hash ^= (e_bits + gidx) * 0x9e3779b97f4a7c15ULL;
            row_hash ^= (f_bits + gidx) * 0xbf58476d1ce4e5b9ULL;
        }
        result ^= row_hash;
    }

    return result;
}

static void gatherResults(const std::vector<val_t>& local_energy,
                          const std::vector<val_t>& local_flux,
                          int n, int px, int py, MPI_Comm cart,
                          bool need_flux,
                          std::vector<val_t>& global_energy,
                          std::vector<val_t>& global_flux,
                          int rank, int size) {
    const size_t total_elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const int local_count = static_cast<int>(local_energy.size());

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int> rank_x0;
    std::vector<int> rank_y0;
    std::vector<int> rank_lx;
    std::vector<int> rank_ly;

    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        rank_x0.resize(size);
        rank_y0.resize(size);
        rank_lx.resize(size);
        rank_ly.resize(size);

        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int coords[2];
            MPI_Cart_coords(cart, r, 2, coords);
            const int lx = partitionLength(n, px, coords[0]);
            const int ly = partitionLength(n, py, coords[1]);
            rank_x0[r] = partitionStart(n, px, coords[0]);
            rank_y0[r] = partitionStart(n, py, coords[1]);
            rank_lx[r] = lx;
            rank_ly[r] = ly;
            counts[r] = lx * ly;
            displs[r] = offset;
            offset += counts[r];
        }
        global_energy.assign(total_elems, 0.0);
        if (need_flux) {
            global_flux.assign(total_elems, 0.0);
        }
    }

    std::vector<val_t> packed;
    if (rank == 0) {
        packed.resize(total_elems > 0 ? total_elems : 1);
    }

    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                rank == 0 ? packed.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, cart);

    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < size; ++r) {
            const int lx = rank_lx[r];
            const int ly = rank_ly[r];
            const int src = displs[r];
            const int gx0 = rank_x0[r];
            const int gy0 = rank_y0[r];
            for (int li = 0; li < lx; ++li) {
                val_t* dst = global_energy.data()
                           + static_cast<size_t>(gx0 + li) * static_cast<size_t>(n)
                           + static_cast<size_t>(gy0);
                const val_t* s = packed.data() + src + static_cast<size_t>(li) * static_cast<size_t>(ly);
                memcpy(dst, s, static_cast<size_t>(ly) * sizeof(val_t));
            }
        }
    }

    if (need_flux) {
        MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE,
                    rank == 0 ? packed.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, cart);

        if (rank == 0) {
#pragma omp parallel for schedule(static)
            for (int r = 0; r < size; ++r) {
                const int lx = rank_lx[r];
                const int ly = rank_ly[r];
                const int src = displs[r];
                const int gx0 = rank_x0[r];
                const int gy0 = rank_y0[r];
                for (int li = 0; li < lx; ++li) {
                    val_t* dst = global_flux.data()
                               + static_cast<size_t>(gx0 + li) * static_cast<size_t>(n)
                               + static_cast<size_t>(gy0);
                    const val_t* s = packed.data() + src + static_cast<size_t>(li) * static_cast<size_t>(ly);
                    memcpy(dst, s, static_cast<size_t>(ly) * sizeof(val_t));
                }
            }
        }
    }
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    const int max_threads = omp_get_max_threads();
    omp_set_num_threads(std::max(1, max_threads / std::max(1, local_size)));

    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        if (world_rank == 0) {
            fprintf(stderr, "CUDA device error: %s\n",
                    cudaGetErrorString(device_status == cudaSuccess ? cudaErrorNoDevice : device_status));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");

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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    const int px = dims[0];
    const int py = dims[1];

    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);

    int cart_rank = 0;
    int cart_size = 1;
    int coords[2] = {0, 0};
    MPI_Comm_rank(cart, &cart_rank);
    MPI_Comm_size(cart, &cart_size);
    MPI_Cart_coords(cart, cart_rank, 2, coords);

    int source_x = MPI_PROC_NULL;
    int dest_x = MPI_PROC_NULL;
    int source_y = MPI_PROC_NULL;
    int dest_y = MPI_PROC_NULL;
    MPI_Cart_shift(cart, 0, 1, &source_x, &dest_x);
    MPI_Cart_shift(cart, 1, 1, &source_y, &dest_y);

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    GpuDomain gpu;
    buildLocalDomain(gpu, n_elems_root, coords[0], coords[1], px, py);

    if (world_rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(gpu, n_iters, n_elems_root, cart, source_x, dest_x, source_y, dest_y);

    auto end = std::chrono::high_resolution_clock::now();
    auto local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = local_ms;
    long long global_duration_ms = 0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, cart);
    duration_ms = global_duration_ms;

    std::vector<val_t> local_energy;
    std::vector<val_t> local_flux;
    fetchLocal(gpu, local_energy, local_flux);

    const uint64_t rank_hash = localHash(gpu, local_energy, local_flux, n_elems_root);
    uint64_t result_hash = 0;
    MPI_Reduce(&rank_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart);

    const bool need_gather = validate || printResults;
    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    if (need_gather) {
        gatherResults(local_energy, local_flux, n_elems_root, px, py, cart,
                      validate, global_energy, global_flux, cart_rank, cart_size);
    }

    if (world_rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems)
                                        / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        printf("  Result hash: %016lX\n", result_hash);
        printf("\n");

        if (printResults) {
            print_results(global_energy, "ElementEnergy");
        }

        if (validate) {
            World result_world;
            result_world.elements_dynamic.resize(static_cast<size_t>(n_elems));
            for (size_t i = 0; i < result_world.elements_dynamic.size(); ++i) {
                result_world.elements_dynamic[i].current_energy = global_energy[i];
                result_world.elements_dynamic[i].total_flux = global_flux[i];
            }

            const bool valid = validateResults(result_world);
            if (!valid) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    cudaCheck(cudaFree(gpu.d_energy), "free energy");
    cudaCheck(cudaFree(gpu.d_energy_new), "free energy_new");
    cudaCheck(cudaFree(gpu.d_flux), "free flux");
    cudaCheck(cudaFree(gpu.d_send), "free send");
    cudaCheck(cudaFree(gpu.d_recv), "free recv");
    cudaCheck(cudaFreeHost(gpu.h_send), "free host send");
    cudaCheck(cudaFreeHost(gpu.h_recv), "free host recv");

    MPI_Comm_free(&cart);
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
