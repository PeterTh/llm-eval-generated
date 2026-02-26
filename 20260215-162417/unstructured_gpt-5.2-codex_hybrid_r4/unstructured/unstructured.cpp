#include <algorithm>
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

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct World {
    int n_elems_root = 0;
    int n_cols = 0;
    int rank = 0;
    int size = 1;
    int local_rows = 0;
    int row_start = 0;
    int row_end = 0;
    int prev_rank = -1;
    int next_rank = -1;
    int padded_rows = 0;
    std::vector<Material> materials;
    std::vector<idx_t> material_idx;
};

struct DeviceBuffers {
    val_t* d_energy = nullptr;
    val_t* d_energy_swap = nullptr;
    val_t* d_flux = nullptr;
    val_t* d_flux_swap = nullptr;
    idx_t* d_material_idx = nullptr;
    Material* d_materials = nullptr;
    val_t* h_send_top = nullptr;
    val_t* h_send_bottom = nullptr;
    val_t* h_recv_top = nullptr;
    val_t* h_recv_bottom = nullptr;
};

inline void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int rowsForRank(const int n_rows, const int rank, const int size) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    return base + (rank < rem ? 1 : 0);
}

int rowStartForRank(const int n_rows, const int rank, const int size) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    return rank * base + std::min(rank, rem);
}

int ownerRankForRow(const int n_rows, const int row, const int size) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    if (base == 0) {
        return row;
    }
    const int cutoff = rem * (base + 1);
    if (row < cutoff) {
        return row / (base + 1);
    }
    return rem + (row - cutoff) / base;
}

int getLocalRankFromEnv() {
    const char* env_vars[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID"
    };
    for (const char* var : env_vars) {
        const char* value = std::getenv(var);
        if (value) {
            return std::atoi(value);
        }
    }
    return -1;
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    world.n_cols = n_elems_root;
    world.local_rows = rowsForRank(n_elems_root, world.rank, world.size);
    world.row_start = rowStartForRank(n_elems_root, world.rank, world.size);
    world.row_end = world.row_start + world.local_rows;
    world.padded_rows = world.local_rows + 2;

    if (world.local_rows > 0) {
        if (world.row_start > 0) {
            world.prev_rank = ownerRankForRow(n_elems_root, world.row_start - 1, world.size);
        }
        if (world.row_end < n_elems_root) {
            world.next_rank = ownerRankForRow(n_elems_root, world.row_end, world.size);
        }
    }

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});   // Default material
    world.materials.emplace_back(Material{0.8, 0.5});   // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});  // Outflow material

    const int local_elems = world.local_rows * world.n_cols;
    world.material_idx.assign(local_elems, DEFAULT_MAT_ID);

    if (local_elems == 0) {
        return;
    }

    const int last = n_elems_root - 1;

    #pragma omp parallel for
    for (int local_r = 0; local_r < world.local_rows; ++local_r) {
        const int global_r = world.row_start + local_r;
        for (int c = 0; c < world.n_cols; ++c) {
            idx_t mat_id = DEFAULT_MAT_ID;
            if (global_r == 0 && c == 0) {
                mat_id = INFLOW_MAT_ID;
            } else if (global_r == 0 && c == last) {
                mat_id = OUTFLOW_MAT_ID;
            } else if (global_r == last && c == 0) {
                mat_id = OUTFLOW_MAT_ID;
            } else if (global_r == last && c == last) {
                mat_id = INFLOW_MAT_ID;
            }
            world.material_idx[local_r * world.n_cols + c] = mat_id;
        }
    }
}

__device__ inline val_t computeFluxDevice(const val_t transfer_coeff, const val_t this_energy, const val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * 0.25;
}

__global__ void updateKernel(const val_t* energy, const val_t* flux, val_t* energy_out, val_t* flux_out,
                            const idx_t* material_idx, const Material* materials,
                            const int n_cols, const int local_rows, const int row_start, const int n_elems_root) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= n_cols || local_r >= local_rows) {
        return;
    }

    const int global_r = row_start + local_r;
    const int idx_local = (local_r + 1) * n_cols + c;
    const idx_t mat_id = material_idx[local_r * n_cols + c];
    const Material mat = materials[mat_id];

    const val_t this_energy = energy[idx_local];
    val_t total_flux = mat.external_flow;

    if (global_r > 0) {
        const int n_idx = local_r * n_cols + c;
        total_flux += computeFluxDevice(mat.transfer_coeff, this_energy, energy[n_idx]);
    }

    if (global_r + 1 < n_elems_root) {
        const int n_idx = (local_r + 2) * n_cols + c;
        total_flux += computeFluxDevice(mat.transfer_coeff, this_energy, energy[n_idx]);
    }

    if (c > 0) {
        const int n_idx = idx_local - 1;
        total_flux += computeFluxDevice(mat.transfer_coeff, this_energy, energy[n_idx]);
    }

    if (c + 1 < n_cols) {
        const int n_idx = idx_local + 1;
        total_flux += computeFluxDevice(mat.transfer_coeff, this_energy, energy[n_idx]);
    }

    energy_out[idx_local] = this_energy + total_flux;
    flux_out[idx_local] = flux[idx_local] + fabs(total_flux);
}

void allocateDeviceBuffers(const World& world, DeviceBuffers& dev) {
    const size_t padded_elems = static_cast<size_t>(world.padded_rows) * world.n_cols;
    const size_t local_elems = static_cast<size_t>(world.local_rows) * world.n_cols;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_energy), padded_elems * sizeof(val_t)), "cudaMalloc d_energy");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_energy_swap), padded_elems * sizeof(val_t)), "cudaMalloc d_energy_swap");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_flux), padded_elems * sizeof(val_t)), "cudaMalloc d_flux");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_flux_swap), padded_elems * sizeof(val_t)), "cudaMalloc d_flux_swap");

    if (local_elems > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_material_idx), local_elems * sizeof(idx_t)), "cudaMalloc d_material_idx");
        checkCuda(cudaMemcpy(dev.d_material_idx, world.material_idx.data(), local_elems * sizeof(idx_t), cudaMemcpyHostToDevice),
                  "cudaMemcpy material_idx");
    }

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev.d_materials), world.materials.size() * sizeof(Material)), "cudaMalloc d_materials");
    checkCuda(cudaMemcpy(dev.d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice),
              "cudaMemcpy materials");

    checkCuda(cudaMemset(dev.d_energy, 0, padded_elems * sizeof(val_t)), "cudaMemset d_energy");
    checkCuda(cudaMemset(dev.d_energy_swap, 0, padded_elems * sizeof(val_t)), "cudaMemset d_energy_swap");
    checkCuda(cudaMemset(dev.d_flux, 0, padded_elems * sizeof(val_t)), "cudaMemset d_flux");
    checkCuda(cudaMemset(dev.d_flux_swap, 0, padded_elems * sizeof(val_t)), "cudaMemset d_flux_swap");

    const size_t row_bytes = static_cast<size_t>(world.n_cols) * sizeof(val_t);
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&dev.h_send_top), row_bytes), "cudaMallocHost send_top");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&dev.h_send_bottom), row_bytes), "cudaMallocHost send_bottom");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&dev.h_recv_top), row_bytes), "cudaMallocHost recv_top");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&dev.h_recv_bottom), row_bytes), "cudaMallocHost recv_bottom");
}

void freeDeviceBuffers(DeviceBuffers& dev) {
    if (dev.d_energy) cudaFree(dev.d_energy);
    if (dev.d_energy_swap) cudaFree(dev.d_energy_swap);
    if (dev.d_flux) cudaFree(dev.d_flux);
    if (dev.d_flux_swap) cudaFree(dev.d_flux_swap);
    if (dev.d_material_idx) cudaFree(dev.d_material_idx);
    if (dev.d_materials) cudaFree(dev.d_materials);
    if (dev.h_send_top) cudaFreeHost(dev.h_send_top);
    if (dev.h_send_bottom) cudaFreeHost(dev.h_send_bottom);
    if (dev.h_recv_top) cudaFreeHost(dev.h_recv_top);
    if (dev.h_recv_bottom) cudaFreeHost(dev.h_recv_bottom);
}

void exchangeHalos(const World& world, DeviceBuffers& dev) {
    if (world.local_rows == 0) {
        return;
    }

    const size_t row_bytes = static_cast<size_t>(world.n_cols) * sizeof(val_t);

    if (world.prev_rank >= 0) {
        checkCuda(cudaMemcpy(dev.h_send_top, dev.d_energy + world.n_cols, row_bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy send_top");
    }
    if (world.next_rank >= 0) {
        checkCuda(cudaMemcpy(dev.h_send_bottom, dev.d_energy + world.local_rows * world.n_cols, row_bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy send_bottom");
    }

    MPI_Status status;
    if (world.prev_rank >= 0) {
        MPI_Sendrecv(dev.h_send_top, world.n_cols, MPI_DOUBLE, world.prev_rank, 100,
                     dev.h_recv_top, world.n_cols, MPI_DOUBLE, world.prev_rank, 101,
                     MPI_COMM_WORLD, &status);
    }
    if (world.next_rank >= 0) {
        MPI_Sendrecv(dev.h_send_bottom, world.n_cols, MPI_DOUBLE, world.next_rank, 101,
                     dev.h_recv_bottom, world.n_cols, MPI_DOUBLE, world.next_rank, 100,
                     MPI_COMM_WORLD, &status);
    }

    if (world.prev_rank >= 0) {
        checkCuda(cudaMemcpy(dev.d_energy, dev.h_recv_top, row_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy recv_top");
    }
    if (world.next_rank >= 0) {
        checkCuda(cudaMemcpy(dev.d_energy + (world.local_rows + 1) * world.n_cols, dev.h_recv_bottom, row_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy recv_bottom");
    }
}

void runSimulation(World& world, DeviceBuffers& dev, const int n_iters) {
    if (world.local_rows == 0) {
        return;
    }

    const dim3 block(16, 16);
    const dim3 grid((world.n_cols + block.x - 1) / block.x,
                    (world.local_rows + block.y - 1) / block.y);

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(world, dev);
        updateKernel<<<grid, block>>>(dev.d_energy, dev.d_flux, dev.d_energy_swap, dev.d_flux_swap,
                                      dev.d_material_idx, dev.d_materials, world.n_cols, world.local_rows,
                                      world.row_start, world.n_elems_root);
        checkCuda(cudaGetLastError(), "updateKernel launch");
        checkCuda(cudaDeviceSynchronize(), "updateKernel sync");
        std::swap(dev.d_energy, dev.d_energy_swap);
        std::swap(dev.d_flux, dev.d_flux_swap);
    }
}

void copyResultsToHost(const World& world, const DeviceBuffers& dev,
                       std::vector<val_t>& energy, std::vector<val_t>& flux) {
    const size_t local_elems = static_cast<size_t>(world.local_rows) * world.n_cols;
    energy.assign(local_elems, 0.0);
    flux.assign(local_elems, 0.0);
    if (local_elems == 0) {
        return;
    }

    const size_t bytes = local_elems * sizeof(val_t);
    checkCuda(cudaMemcpy(energy.data(), dev.d_energy + world.n_cols, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy energy results");
    checkCuda(cudaMemcpy(flux.data(), dev.d_flux + world.n_cols, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy flux results");
}

bool validateResults(const World& world, const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:energy_sum, flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;

    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    bool valid = true;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }

        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }

        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }

        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return valid;
}

uint64_t computeHashLocal(const World& world, const std::vector<val_t>& energy,
                          const std::vector<val_t>& flux) {
    uint64_t hash = 0;
    const int n_cols = world.n_cols;

    #pragma omp parallel for reduction(^:hash)
    for (size_t i = 0; i < energy.size(); ++i) {
        const int local_row = static_cast<int>(i / n_cols);
        const int col = static_cast<int>(i - static_cast<size_t>(local_row) * n_cols);
        const size_t global_index = static_cast<size_t>(world.row_start + local_row) * n_cols + col;

        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux[i]);

        hash ^= (*e_ptr + global_index) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_index) * 0xbf58476d1ce4e5b9ULL;
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

    World world;
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

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
            if (world.rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        if (world.rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rank = getLocalRankFromEnv();
    if (local_rank < 0) {
        local_rank = world.rank;
    }
    const int device_id = local_rank % device_count;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");

    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    const int n_elems = n_elems_root * n_elems_root;

    if (world.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI+OpenMP+CUDA)\n");
        printf("=======================================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", world.size);
        printf("OpenMP threads (max): %d\n", omp_get_max_threads());
        printf("CUDA devices detected: %d\n", device_count);
        printf("CUDA device (rank 0): %d - %s\n", device_id, prop.name);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    buildSquare2D(world, n_elems_root);

    DeviceBuffers dev;
    allocateDeviceBuffers(world, dev);

    const size_t local_static_mem = world.material_idx.size() * sizeof(idx_t) + world.materials.size() * sizeof(Material);
    const size_t local_dynamic_mem = static_cast<size_t>(world.padded_rows) * world.n_cols * sizeof(val_t) * 4;
    const unsigned long long local_total_mem = static_cast<unsigned long long>(local_static_mem + local_dynamic_mem);

    unsigned long long global_total_mem = 0;
    MPI_Reduce(&local_total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
        printf("Memory usage (approx): %.2f MB total across ranks\n",
               global_total_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    runSimulation(world, dev, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end_time = MPI_Wtime();

    double local_duration = end_time - start_time;
    double duration_sec = 0.0;
    MPI_Reduce(&local_duration, &duration_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> energy;
    std::vector<val_t> flux;
    copyResultsToHost(world, dev, energy, flux);

    const uint64_t local_hash = computeHashLocal(world, energy, flux);
    const unsigned long long local_hash_ull = static_cast<unsigned long long>(local_hash);
    unsigned long long global_hash_ull = 0;
    MPI_Reduce(&local_hash_ull, &global_hash_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
        const long duration_ms = static_cast<long>(duration_sec * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_sec) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016llX\n", global_hash_ull);
        printf("\n");
    }

    if (printResults) {
        std::vector<int> recv_counts;
        std::vector<int> displs;
        std::vector<double> energyData;
        const int n_cols = world.n_cols;
        const int local_count = world.local_rows * n_cols;

        if (world.rank == 0) {
            recv_counts.resize(world.size);
            displs.resize(world.size);
            for (int r = 0; r < world.size; ++r) {
                const int rows = rowsForRank(n_elems_root, r, world.size);
                recv_counts[r] = rows * n_cols;
                displs[r] = rowStartForRank(n_elems_root, r, world.size) * n_cols;
            }
            energyData.resize(static_cast<size_t>(n_elems));
        }

        MPI_Gatherv(energy.data(), local_count, MPI_DOUBLE,
                    energyData.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (world.rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    if (validate) {
        const bool valid = validateResults(world, energy, flux);
        if (!valid) {
            freeDeviceBuffers(dev);
            MPI_Finalize();
            return 1;
        }
    }

    freeDeviceBuffers(dev);
    MPI_Finalize();
    return 0;
}
