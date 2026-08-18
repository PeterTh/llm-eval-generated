#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
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

// Maximum number of connections per element (the generated mesh uses four).
constexpr int MAX_CONNECTIONS = 8;
constexpr int GRID_CONNECTIONS = 4;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Each MPI rank owns a contiguous band of rows. The connectivity indices are
// local indices into a two-row-halo array: row zero is the top halo and the
// last row is the bottom halo.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    int n_elems_root = 0;
    int row_start = 0;
    int local_rows = 0;
    int mpi_rank = 0;
    int mpi_size = 1;
    idx_t owned_offset = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

[[noreturn]] void cudaFailure(const cudaError_t status, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n", file, line,
                 expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                          \
        const cudaError_t cuda_status = (expression);                             \
        if (cuda_status != cudaSuccess) {                                         \
            cudaFailure(cuda_status, #expression, __FILE__, __LINE__);            \
        }                                                                         \
    } while (false)

void mpiCheck(const int status, const char* expression, const char* file,
              const int line) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]{};
        int message_length = 0;
        MPI_Error_string(status, message, &message_length);
        std::fprintf(stderr, "MPI failure at %s:%d (%s): %.*s\n", file, line,
                     expression, message_length, message);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define MPI_CHECK(expression)                                                     \
    do {                                                                          \
        mpiCheck((expression), #expression, __FILE__, __LINE__);                  \
    } while (false)

// CUDA kernel for one Jacobi-style timestep. The input state is read-only and
// the output state is written to a separate buffer, exactly as in the serial
// implementation. Every thread owns one element, so no atomics are needed.
__global__ void updateElementsKernel(
    const idx_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const unsigned char* __restrict__ num_connections,
    const val_t* __restrict__ transfer_coeff,
    const val_t* __restrict__ external_flow,
    const val_t* __restrict__ current_energy,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ accumulated_flux,
    const idx_t owned_offset,
    const size_t n_elements) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x;
    if (element >= n_elements) {
        return;
    }

    const idx_t current_idx = owned_offset + static_cast<idx_t>(element);
    const val_t element_energy = current_energy[current_idx];
    const val_t transfer = transfer_coeff[element];
    val_t total_flux = external_flow[element];
    const unsigned int connection_count = num_connections[element];

    for (unsigned int connection = 0; connection < connection_count;
         ++connection) {
        const size_t connection_idx =
            static_cast<size_t>(element) * MAX_CONNECTIONS + connection;
        const idx_t neighbor_idx = connected_idx[connection_idx];
        total_flux += (current_energy[neighbor_idx] - element_energy) *
                      transfer * connected_flux[connection_idx] * 0.25;
    }

    next_energy[current_idx] = element_energy + total_flux;
    accumulated_flux[element] += fabs(total_flux);
}

void selectCudaDevice(const World& world) {
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_comm));

    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "MPI rank %d found no CUDA devices\n",
                     world.mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // MPI ranks sharing a node are distributed over the node's GPUs. The
    // modulo also permits more ranks than GPUs while keeping the mapping
    // deterministic for oversubscribed development runs.
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_CHECK(MPI_Comm_free(&local_comm));
}

// Build the local part of the same square mesh generated by the reference
// code. Static connectivity is expressed in local-with-halo indices, while
// material selection still uses global coordinates.
void buildSquare2D(World& world, const int n_elems_root, const int mpi_rank,
                   const int mpi_size) {
    const int rows_per_rank = n_elems_root / mpi_size;
    const int extra_rows = n_elems_root % mpi_size;
    world.local_rows = rows_per_rank + (mpi_rank < extra_rows ? 1 : 0);
    world.row_start = mpi_rank * rows_per_rank +
                      std::min(mpi_rank, extra_rows);
    world.owned_offset = static_cast<idx_t>(n_elems_root);

    const size_t n_local_elements =
        static_cast<size_t>(world.local_rows) * n_elems_root;

    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(n_local_elements);
    world.elements_dynamic.resize(n_local_elements);
    world.elements_dynamic_swap.resize(n_local_elements);

    // The two initialization loops are independent and are deliberately
    // parallelized with OpenMP on every rank.
#pragma omp parallel for schedule(static)
    for (long long linear = 0;
         linear < static_cast<long long>(n_local_elements); ++linear) {
        ElementStatic& element = world.elements_static[linear];
        element.material_idx = DEFAULT_MAT_ID;
        element.num_connections = 0;
        for (int connection = 0; connection < MAX_CONNECTIONS; ++connection) {
            element.connected_idx[connection] = 0;
            element.connected_flux[connection] = 0.0;
        }
        world.elements_dynamic[linear] = ElementDynamic{0.0, 0.0};
        world.elements_dynamic_swap[linear] = ElementDynamic{0.0, 0.0};
    }

#pragma omp parallel for schedule(static)
    for (long long linear = 0;
         linear < static_cast<long long>(n_local_elements); ++linear) {
        const int local_row = static_cast<int>(linear / n_elems_root);
        const int y = static_cast<int>(linear % n_elems_root);
        const int x = world.row_start + local_row;
        ElementStatic& element = world.elements_static[linear];

        if ((x == 0 && y == 0) ||
            (x == n_elems_root - 1 && y == n_elems_root - 1)) {
            element.material_idx = INFLOW_MAT_ID;
        } else if ((x == 0 && y == n_elems_root - 1) ||
                   (x == n_elems_root - 1 && y == 0)) {
            element.material_idx = OUTFLOW_MAT_ID;
        }

        constexpr int offsets[GRID_CONNECTIONS][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int connection = 0; connection < GRID_CONNECTIONS; ++connection) {
            const int neighbor_x = x + offsets[connection][0];
            const int neighbor_y = y + offsets[connection][1];
            if (neighbor_x < 0 || neighbor_x >= n_elems_root ||
                neighbor_y < 0 || neighbor_y >= n_elems_root) {
                continue;
            }

            int local_neighbor_row = 0;
            if (neighbor_x < world.row_start) {
                local_neighbor_row = 0; // top halo
            } else if (neighbor_x >= world.row_start + world.local_rows) {
                local_neighbor_row = world.local_rows + 1; // bottom halo
            } else {
                local_neighbor_row = neighbor_x - world.row_start + 1;
            }

            const idx_t local_neighbor =
                static_cast<idx_t>(local_neighbor_row) * n_elems_root +
                static_cast<idx_t>(neighbor_y);
            element.connected_idx[element.num_connections] = local_neighbor;
            element.connected_flux[element.num_connections] = 1.0;
            ++element.num_connections;
        }
    }
}

// Run the distributed CUDA simulation. Host/device copies are limited to the
// two boundary rows per iteration; all interior work remains on the GPU.
void runSimulation(World& world, const int n_iters) {
    selectCudaDevice(world);

    const size_t n_elements = world.elements_static.size();
    const size_t n = static_cast<size_t>(world.n_elems_root);
    const size_t storage_elements =
        static_cast<size_t>(world.local_rows + 2) * n;
    const size_t flattened_connections = n_elements * MAX_CONNECTIONS;
    const size_t row_bytes = n * sizeof(val_t);

    // A structure-of-arrays representation reduces the global-memory traffic
    // in the stencil kernel versus transferring the padded host structs.
    std::vector<idx_t> host_connected_idx(flattened_connections, 0);
    std::vector<val_t> host_connected_flux(flattened_connections, 0.0);
    std::vector<unsigned char> host_num_connections(n_elements, 0);
    std::vector<val_t> host_transfer_coeff(n_elements, 0.0);
    std::vector<val_t> host_external_flow(n_elements, 0.0);

#pragma omp parallel for schedule(static)
    for (long long linear = 0;
         linear < static_cast<long long>(n_elements); ++linear) {
        const ElementStatic& element = world.elements_static[linear];
        host_num_connections[linear] =
            static_cast<unsigned char>(element.num_connections);
        const Material& material = world.materials[element.material_idx];
        host_transfer_coeff[linear] = material.transfer_coeff;
        host_external_flow[linear] = material.external_flow;
        for (int connection = 0; connection < MAX_CONNECTIONS; ++connection) {
            const size_t flattened =
                static_cast<size_t>(linear) * MAX_CONNECTIONS + connection;
            host_connected_idx[flattened] = element.connected_idx[connection];
            host_connected_flux[flattened] = element.connected_flux[connection];
        }
    }

    idx_t* device_connected_idx = nullptr;
    val_t* device_connected_flux = nullptr;
    unsigned char* device_num_connections = nullptr;
    val_t* device_transfer_coeff = nullptr;
    val_t* device_external_flow = nullptr;
    val_t* device_energy_a = nullptr;
    val_t* device_energy_b = nullptr;
    val_t* device_accumulated_flux = nullptr;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_connected_idx),
                          flattened_connections * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_connected_flux),
                          flattened_connections * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_num_connections),
                          n_elements * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_transfer_coeff),
                          n_elements * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_external_flow),
                          n_elements * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_energy_a),
                          storage_elements * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_energy_b),
                          storage_elements * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_accumulated_flux),
                          n_elements * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(device_connected_idx, host_connected_idx.data(),
                          flattened_connections * sizeof(idx_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_connected_flux, host_connected_flux.data(),
                          flattened_connections * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_num_connections, host_num_connections.data(),
                          n_elements * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_transfer_coeff, host_transfer_coeff.data(),
                          n_elements * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_external_flow, host_external_flow.data(),
                          n_elements * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_energy_a, 0,
                          storage_elements * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(device_energy_b, 0,
                          storage_elements * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(device_accumulated_flux, 0,
                          n_elements * sizeof(val_t)));

    val_t* send_top = nullptr;
    val_t* send_bottom = nullptr;
    val_t* receive_top = nullptr;
    val_t* receive_bottom = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send_top), row_bytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send_bottom), row_bytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&receive_top), row_bytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&receive_bottom),
                             row_bytes, cudaHostAllocPortable));

    val_t* current_energy = device_energy_a;
    val_t* next_energy = device_energy_b;
    const int top_rank = world.mpi_rank > 0 ? world.mpi_rank - 1 : MPI_PROC_NULL;
    const int bottom_rank = world.mpi_rank + 1 < world.mpi_size
                                ? world.mpi_rank + 1
                                : MPI_PROC_NULL;
    constexpr int threads_per_block = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (n_elements + threads_per_block - 1) / threads_per_block);

    for (int iter = 0; iter < n_iters; ++iter) {
        const idx_t first_owned = world.owned_offset;
        const idx_t last_owned =
            world.owned_offset + static_cast<idx_t>(world.local_rows - 1) * n;
        if (top_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(send_top, current_energy + first_owned,
                                       row_bytes, cudaMemcpyDeviceToHost,
                                       stream));
        }
        if (bottom_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(send_bottom, current_energy + last_owned,
                                       row_bytes, cudaMemcpyDeviceToHost,
                                       stream));
        }
        if (top_rank != MPI_PROC_NULL || bottom_rank != MPI_PROC_NULL) {
            // MPI sends cannot begin until the asynchronous device-to-host
            // copies have completed. With one rank this synchronization and
            // both boundary copies disappear entirely.
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_Request requests[4];
        int request_count = 0;
        if (top_rank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(receive_top, world.n_elems_root, MPI_DOUBLE,
                                top_rank, 2, MPI_COMM_WORLD,
                                &requests[request_count++]));
            MPI_CHECK(MPI_Isend(send_top, world.n_elems_root, MPI_DOUBLE,
                                top_rank, 1, MPI_COMM_WORLD,
                                &requests[request_count++]));
        }
        if (bottom_rank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(receive_bottom, world.n_elems_root, MPI_DOUBLE,
                                bottom_rank, 1, MPI_COMM_WORLD,
                                &requests[request_count++]));
            MPI_CHECK(MPI_Isend(send_bottom, world.n_elems_root, MPI_DOUBLE,
                                bottom_rank, 2, MPI_COMM_WORLD,
                                &requests[request_count++]));
        }
        if (request_count != 0) {
            MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
        }

        if (top_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(current_energy, receive_top, row_bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (bottom_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(
                current_energy + static_cast<idx_t>(world.local_rows + 1) * n,
                receive_bottom, row_bytes, cudaMemcpyHostToDevice, stream));
        }

        updateElementsKernel<<<blocks, threads_per_block, 0, stream>>>(
            device_connected_idx, device_connected_flux, device_num_connections,
            device_transfer_coeff, device_external_flow, current_energy,
            next_energy, device_accumulated_flux, world.owned_offset,
            n_elements);
        CUDA_CHECK(cudaGetLastError());
        std::swap(current_energy, next_energy);
    }

    std::vector<val_t> host_energy(n_elements, 0.0);
    std::vector<val_t> host_accumulated_flux(n_elements, 0.0);
    CUDA_CHECK(cudaMemcpyAsync(host_energy.data(),
                               current_energy + world.owned_offset,
                               n_elements * sizeof(val_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_accumulated_flux.data(),
                               device_accumulated_flux,
                               n_elements * sizeof(val_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

#pragma omp parallel for schedule(static)
    for (long long linear = 0;
         linear < static_cast<long long>(n_elements); ++linear) {
        world.elements_dynamic[linear] =
            ElementDynamic{host_energy[linear], host_accumulated_flux[linear]};
    }

    CUDA_CHECK(cudaFreeHost(send_top));
    CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(receive_top));
    CUDA_CHECK(cudaFreeHost(receive_bottom));
    CUDA_CHECK(cudaFree(device_connected_idx));
    CUDA_CHECK(cudaFree(device_connected_flux));
    CUDA_CHECK(cudaFree(device_num_connections));
    CUDA_CHECK(cudaFree(device_transfer_coeff));
    CUDA_CHECK(cudaFree(device_external_flow));
    CUDA_CHECK(cudaFree(device_energy_a));
    CUDA_CHECK(cudaFree(device_energy_b));
    CUDA_CHECK(cudaFree(device_accumulated_flux));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Validate simulation results using the same serial reduction order as the
// original program. Validation is performed only after rank zero has gathered
// the globally ordered state.
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& element : world.elements_dynamic) {
        energy_sum += element.current_energy;
        flux_sum += element.total_flux;
        energy_max = std::max(element.current_energy, energy_max);
        energy_min = std::min(element.current_energy, energy_min);
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        std::printf(
            "  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    std::printf("  Validation: PASSED\n");
    return true;
}

// Compute a simple hash of the results for verification.
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits, &elements[i].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather both dynamic fields to rank zero in global row-major order.
void gatherResults(World& world) {
    const int n = world.n_elems_root;
    const int rank_count = world.mpi_size;
    const int rows_per_rank = n / rank_count;
    const int extra_rows = n % rank_count;
    const size_t global_elements = static_cast<size_t>(n) * n;
    const size_t local_elements = world.elements_dynamic.size();

    std::vector<int> counts(rank_count);
    std::vector<int> displacements(rank_count);
    for (int rank = 0; rank < rank_count; ++rank) {
        const int rows = rows_per_rank + (rank < extra_rows ? 1 : 0);
        const int first_row = rank * rows_per_rank + std::min(rank, extra_rows);
        const size_t count = static_cast<size_t>(rows) * n;
        const size_t displacement = static_cast<size_t>(first_row) * n;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (world.mpi_rank == 0) {
                std::fprintf(stderr,
                             "MPI gather size exceeds MPI int count limits\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        counts[rank] = static_cast<int>(count);
        displacements[rank] = static_cast<int>(displacement);
    }

    std::vector<val_t> local_energy(local_elements);
    std::vector<val_t> local_flux(local_elements);
#pragma omp parallel for schedule(static)
    for (long long linear = 0;
         linear < static_cast<long long>(local_elements); ++linear) {
        local_energy[linear] = world.elements_dynamic[linear].current_energy;
        local_flux[linear] = world.elements_dynamic[linear].total_flux;
    }

    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    if (world.mpi_rank == 0) {
        global_energy.resize(global_elements);
        global_flux.resize(global_elements);
    }

    MPI_CHECK(MPI_Gatherv(
        local_energy.data(), static_cast<int>(local_elements), MPI_DOUBLE,
        world.mpi_rank == 0 ? global_energy.data() : nullptr, counts.data(),
        displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Gatherv(
        local_flux.data(), static_cast<int>(local_elements), MPI_DOUBLE,
        world.mpi_rank == 0 ? global_flux.data() : nullptr, counts.data(),
        displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (world.mpi_rank == 0) {
        world.elements_dynamic.resize(global_elements);
#pragma omp parallel for schedule(static)
        for (long long linear = 0;
             linear < static_cast<long long>(global_elements); ++linear) {
            world.elements_dynamic[linear] =
                ElementDynamic{global_energy[linear], global_flux[linear]};
        }
    }
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                              &provided_thread_level));

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (mpi_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (n_elems_root <= 0 || mpi_size > n_elems_root) {
        if (mpi_rank == 0) {
            std::fprintf(stderr,
                         "Grid size must be positive and at least the MPI rank count\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const long long n_elems = static_cast<long long>(n_elems_root) * n_elems_root;
    if (n_elems > std::numeric_limits<int>::max()) {
        if (mpi_rank == 0) {
            std::fprintf(stderr,
                         "Grid is too large for the benchmark's MPI interface\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (mpi_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\n", n_elems_root,
                    n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", mpi_size);
        std::printf("OpenMP threads/rank: %d\n", omp_get_max_threads());
        std::printf("CUDA execution: enabled\n\n");
    }

    if (mpi_rank == 0) {
        std::printf("Building unstructured mesh...\n");
    }
    World world;
    world.n_elems_root = n_elems_root;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;
    buildSquare2D(world, n_elems_root, mpi_rank, mpi_size);

    if (mpi_rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) *
                                  sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) *
                                   sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    total_mem / (1024.0 * 1024.0),
                    static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("\nRunning simulation...\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters);
    const auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms_max = 0.0;
    MPI_CHECK(MPI_Reduce(&local_duration_ms, &duration_ms_max, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD));

    gatherResults(world);

    if (mpi_rank == 0) {
        const long duration_ms = static_cast<long>(duration_ms_max);
        std::printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double elapsed_seconds = duration_ms_max / 1000.0;
        const double giga_elems_per_sec =
            elapsed_seconds > 0.0
                ? (static_cast<double>(n_measured_iters) * n_elems) /
                      elapsed_seconds / 1e9
                : 0.0;
        const double gflops = giga_elems_per_sec * 22.0;

        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n",
                    duration_ms_max / n_measured_iters);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(world.elements_dynamic);
        std::printf("  Result hash: %016" PRIX64 "\n", hash);
        std::printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world.elements_dynamic.size());
            for (const auto& element : world.elements_dynamic) {
                energyData.push_back(element.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
    }

    int valid = 1;
    if (validate && mpi_rank == 0) {
        valid = validateResults(world) ? 1 : 0;
    }
    if (validate) {
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    MPI_CHECK(MPI_Finalize());
    return valid == 1 ? 0 : 1;
}
