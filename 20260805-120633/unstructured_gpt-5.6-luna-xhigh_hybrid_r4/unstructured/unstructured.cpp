#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Local indices, including ghosts
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state held by one MPI rank. Elements are ordered by their original
// global index, so concatenating rank-local ranges reconstructs the reference
// result exactly.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    int n_elems_root = 0;
    int first_row = 0;
    int local_rows = 0;
    int active_ranks = 1;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

[[noreturn]] void abortWithMessage(const char* message, const int rank, const int error_code = 1) {
    if (rank == 0) {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, error_code);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line,
               const int rank) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error at %s:%d (%s): %s", file, line,
                      expression, cudaGetErrorString(error));
        abortWithMessage(message, rank);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__, mpi_rank)

struct RowPartition {
    int first_row = 0;
    int rows = 0;
};

RowPartition partitionRows(const int n_elems_root, const int active_ranks, const int rank) {
    if (rank >= active_ranks) {
        return {};
    }

    const int base_rows = n_elems_root / active_ranks;
    const int remainder = n_elems_root % active_ranks;
    const int rows_before = rank * base_rows + std::min(rank, remainder);
    return {rows_before, base_rows + (rank < remainder ? 1 : 0)};
}

// Build a 2D square grid as an unstructured mesh. Each MPI rank owns a
// contiguous range of rows and stores one ghost row on either side.
void buildSquare2D(World& world, const int n_elems_root, const int mpi_rank, const int mpi_size) {
    world.n_elems_root = n_elems_root;
    world.active_ranks = std::min(n_elems_root, mpi_size);
    const RowPartition partition = partitionRows(n_elems_root, world.active_ranks, mpi_rank);
    world.first_row = partition.first_row;
    world.local_rows = partition.rows;

    // Initialize materials on every rank. Keeping these replicated makes the
    // update kernel independent of MPI communication.
    world.materials = {
        Material{0.8, 0.0},    // Default material
        Material{0.8, 0.5},    // Inflow material
        Material{0.8, -0.5}    // Outflow material
    };

    const size_t local_elements = static_cast<size_t>(world.local_rows) * n_elems_root;
    world.elements_static.resize(local_elements);
    world.elements_dynamic.resize(local_elements);
    world.elements_dynamic_swap.resize(local_elements);

    // The original code visits neighbors in this order. Preserving that order
    // also preserves the floating-point summation order in each element.
    constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

#pragma omp parallel for schedule(static)
    for (long long local_index = 0; local_index < static_cast<long long>(local_elements); ++local_index) {
        const int local_row = static_cast<int>(local_index / n_elems_root);
        const int y = static_cast<int>(local_index % n_elems_root);
        const int x = world.first_row + local_row;
        ElementStatic& elem = world.elements_static[static_cast<size_t>(local_index)];

        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            elem.connected_idx[j] = 0;
            elem.connected_flux[j] = 0.0;
        }

        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root) {
                continue;
            }

            idx_t local_neighbor_idx;
            if (nx < world.first_row) {
                // Lower ghost row follows all owned elements.
                local_neighbor_idx = static_cast<idx_t>(local_elements + ny);
            } else if (nx >= world.first_row + world.local_rows) {
                // Upper ghost row follows the lower ghost row.
                local_neighbor_idx = static_cast<idx_t>(local_elements + n_elems_root + ny);
            } else {
                local_neighbor_idx = static_cast<idx_t>(
                    (nx - world.first_row) * n_elems_root + ny);
            }

            const idx_t connection = elem.num_connections++;
            elem.connected_idx[connection] = local_neighbor_idx;
            elem.connected_flux[connection] = 1.0;
        }

        const int last = n_elems_root - 1;
        if ((x == 0 && y == 0) || (x == last && y == last)) {
            elem.material_idx = INFLOW_MAT_ID;
        } else if ((x == 0 && y == last) || (x == last && y == 0)) {
            elem.material_idx = OUTFLOW_MAT_ID;
        }

        world.elements_dynamic[static_cast<size_t>(local_index)] = ElementDynamic{0.0, 0.0};
        world.elements_dynamic_swap[static_cast<size_t>(local_index)] = ElementDynamic{0.0, 0.0};
    }
}

// CUDA uses a structure-of-arrays view for the static mesh. This reduces the
// amount of unrelated connectivity data fetched by neighboring threads.
struct DeviceMesh {
    idx_t* material_idx = nullptr;
    idx_t* num_connections = nullptr;
    idx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    Material* materials = nullptr;
    size_t elements = 0;
    int mpi_rank = 0;

    void initialize(const World& world) {
        elements = world.elements_static.size();
        if (elements == 0) {
            return;
        }

        std::vector<idx_t> h_material_idx(elements);
        std::vector<idx_t> h_num_connections(elements);
        std::vector<idx_t> h_connected_idx(elements * MAX_CONNECTIONS);
        std::vector<val_t> h_connected_flux(elements * MAX_CONNECTIONS);

#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(elements); ++i) {
            const ElementStatic& source = world.elements_static[static_cast<size_t>(i)];
            h_material_idx[static_cast<size_t>(i)] = source.material_idx;
            h_num_connections[static_cast<size_t>(i)] = source.num_connections;
            for (int j = 0; j < MAX_CONNECTIONS; ++j) {
                h_connected_idx[static_cast<size_t>(i) * MAX_CONNECTIONS + j] = source.connected_idx[j];
                h_connected_flux[static_cast<size_t>(i) * MAX_CONNECTIONS + j] = source.connected_flux[j];
            }
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&material_idx), elements * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&num_connections), elements * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&connected_idx),
                              elements * MAX_CONNECTIONS * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&connected_flux),
                              elements * MAX_CONNECTIONS * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&materials),
                              world.materials.size() * sizeof(Material)));

        CUDA_CHECK(cudaMemcpy(material_idx, h_material_idx.data(), elements * sizeof(idx_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(num_connections, h_num_connections.data(), elements * sizeof(idx_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_idx, h_connected_idx.data(),
                              elements * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_flux, h_connected_flux.data(),
                              elements * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(materials, world.materials.data(),
                              world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    }

    void release() {
        if (material_idx != nullptr) cudaFree(material_idx);
        if (num_connections != nullptr) cudaFree(num_connections);
        if (connected_idx != nullptr) cudaFree(connected_idx);
        if (connected_flux != nullptr) cudaFree(connected_flux);
        if (materials != nullptr) cudaFree(materials);
        material_idx = nullptr;
        num_connections = nullptr;
        connected_idx = nullptr;
        connected_flux = nullptr;
        materials = nullptr;
    }
};

struct DeviceState {
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux[2] = {nullptr, nullptr};
    size_t storage_elements = 0;
    int mpi_rank = 0;

    void initialize(const size_t owned_elements, const size_t ghost_elements) {
        storage_elements = owned_elements + ghost_elements;
        const size_t bytes = std::max(storage_elements, size_t{1}) * sizeof(val_t);
        for (int buffer = 0; buffer < 2; ++buffer) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&energy[buffer]), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&total_flux[buffer]), bytes));
            CUDA_CHECK(cudaMemset(energy[buffer], 0, bytes));
            CUDA_CHECK(cudaMemset(total_flux[buffer], 0, bytes));
        }
    }

    void release() {
        for (int buffer = 0; buffer < 2; ++buffer) {
            if (energy[buffer] != nullptr) cudaFree(energy[buffer]);
            if (total_flux[buffer] != nullptr) cudaFree(total_flux[buffer]);
            energy[buffer] = nullptr;
            total_flux[buffer] = nullptr;
        }
    }
};

__global__ void updateKernel(const idx_t* __restrict__ material_idx,
                             const idx_t* __restrict__ num_connections,
                             const idx_t* __restrict__ connected_idx,
                             const val_t* __restrict__ connected_flux,
                             const Material* __restrict__ materials,
                             const val_t* __restrict__ current_energy,
                             const val_t* __restrict__ current_total_flux,
                             val_t* __restrict__ next_energy,
                             val_t* __restrict__ next_total_flux,
                             const size_t elements) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) {
        return;
    }

    const val_t this_energy = current_energy[i];
    const Material mat = materials[material_idx[i]];
    val_t total_flux = mat.external_flow;
    const size_t connection_base = i * MAX_CONNECTIONS;
    for (idx_t j = 0; j < num_connections[i]; ++j) {
        const idx_t neighbor_idx = connected_idx[connection_base + j];
        total_flux += (current_energy[neighbor_idx] - this_energy) *
                      mat.transfer_coeff * connected_flux[connection_base + j] * 0.25;
    }

    next_energy[i] = this_energy + total_flux;
    next_total_flux[i] = current_total_flux[i] + fabs(total_flux);
}

// Exchange one ghost row in each direction. The buffers are pinned so the
// device-to-host and host-to-device transfers remain efficient without
// requiring CUDA-aware MPI from the cluster's MPI installation.
void exchangeHalos(DeviceState& state, const int current_buffer, const World& world,
                   const int mpi_rank, const int mpi_size, cudaStream_t stream,
                   val_t* lower_send, val_t* lower_recv, val_t* upper_send, val_t* upper_recv) {
    const int lower_rank = (mpi_rank > 0 && mpi_rank < world.active_ranks) ? mpi_rank - 1 : MPI_PROC_NULL;
    const int upper_rank = (mpi_rank < world.active_ranks - 1) ? mpi_rank + 1 : MPI_PROC_NULL;
    const size_t row_bytes = static_cast<size_t>(world.n_elems_root) * sizeof(val_t);
    const size_t local_elements = world.elements_static.size();
    const size_t root = static_cast<size_t>(world.n_elems_root);

    if (world.local_rows == 0 || mpi_rank >= world.active_ranks) {
        return;
    }

    if (lower_rank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(lower_send, state.energy[current_buffer], row_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (upper_rank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(upper_send, state.energy[current_buffer] + local_elements - root,
                                   row_bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Request requests[4];
    int request_count = 0;
    if (lower_rank != MPI_PROC_NULL) {
        MPI_Irecv(lower_recv, world.n_elems_root, MPI_DOUBLE, lower_rank, 100,
                  MPI_COMM_WORLD, &requests[request_count++]);
        MPI_Isend(lower_send, world.n_elems_root, MPI_DOUBLE, lower_rank, 101,
                  MPI_COMM_WORLD, &requests[request_count++]);
    }
    if (upper_rank != MPI_PROC_NULL) {
        MPI_Irecv(upper_recv, world.n_elems_root, MPI_DOUBLE, upper_rank, 101,
                  MPI_COMM_WORLD, &requests[request_count++]);
        MPI_Isend(upper_send, world.n_elems_root, MPI_DOUBLE, upper_rank, 100,
                  MPI_COMM_WORLD, &requests[request_count++]);
    }
    if (request_count > 0) {
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    }

    if (lower_rank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(state.energy[current_buffer] + local_elements, lower_recv,
                                   row_bytes, cudaMemcpyHostToDevice, stream));
    }
    if (upper_rank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(state.energy[current_buffer] + local_elements + root, upper_recv,
                                   row_bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Run simulation for n_iters iterations. MPI owns row-to-row communication,
// while all element updates are issued as one CUDA grid per iteration.
void runSimulation(World& world, const int n_iters, const int mpi_rank, const int mpi_size) {
    const size_t local_elements = world.elements_static.size();
    const size_t ghost_elements = static_cast<size_t>(2) * world.n_elems_root;
    const int threads_per_block = 256;

    DeviceMesh mesh;
    mesh.mpi_rank = mpi_rank;
    mesh.initialize(world);

    DeviceState state;
    state.mpi_rank = mpi_rank;
    state.initialize(local_elements, ghost_elements);

    val_t* lower_send = nullptr;
    val_t* lower_recv = nullptr;
    val_t* upper_send = nullptr;
    val_t* upper_recv = nullptr;
    const size_t row_bytes = static_cast<size_t>(world.n_elems_root) * sizeof(val_t);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&lower_send), row_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&lower_recv), row_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&upper_send), row_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&upper_recv), row_bytes, cudaHostAllocDefault));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int current_buffer = 0;
    const size_t blocks = (local_elements + threads_per_block - 1) / threads_per_block;
    for (int iter = 0; iter < n_iters; ++iter) {
        if (local_elements > 0) {
            updateKernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0, stream>>>(
                mesh.material_idx, mesh.num_connections, mesh.connected_idx, mesh.connected_flux,
                mesh.materials, state.energy[current_buffer], state.total_flux[current_buffer],
                state.energy[1 - current_buffer], state.total_flux[1 - current_buffer], local_elements);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        current_buffer = 1 - current_buffer;
        if (iter + 1 < n_iters) {
            exchangeHalos(state, current_buffer, world, mpi_rank, mpi_size, stream,
                          lower_send, lower_recv, upper_send, upper_recv);
        }
    }

    // Copy the final SoA state into the original AoS representation. This is
    // also the only full-device-to-host copy, keeping iterative communication
    // limited to the two boundary energy rows.
    if (local_elements > 0) {
        std::vector<val_t> host_energy(local_elements);
        std::vector<val_t> host_flux(local_elements);
        CUDA_CHECK(cudaMemcpy(host_energy.data(), state.energy[current_buffer],
                              local_elements * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_flux.data(), state.total_flux[current_buffer],
                              local_elements * sizeof(val_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(local_elements); ++i) {
            world.elements_dynamic[static_cast<size_t>(i)] =
                ElementDynamic{host_energy[static_cast<size_t>(i)], host_flux[static_cast<size_t>(i)]};
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(lower_send));
    CUDA_CHECK(cudaFreeHost(lower_recv));
    CUDA_CHECK(cudaFreeHost(upper_send));
    CUDA_CHECK(cudaFreeHost(upper_recv));
    state.release();
    mesh.release();
}

// Validate distributed simulation results. Reductions are only used for
// diagnostics; element values themselves remain in their original order.
bool validateResults(const World& world, const int mpi_rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    int local_finite = 1;

#pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
    reduction(max:local_energy_max) reduction(min:local_energy_min) reduction(&:local_finite) schedule(static)
    for (long long i = 0; i < static_cast<long long>(world.elements_dynamic.size()); ++i) {
        const ElementDynamic& elem = world.elements_dynamic[static_cast<size_t>(i)];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(local_energy_max, elem.current_energy);
        local_energy_min = std::min(local_energy_min, elem.current_energy);
        local_finite &= std::isfinite(elem.current_energy) && std::isfinite(elem.total_flux);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    int all_finite = 0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_finite, &all_finite, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        return true;
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!all_finite || !std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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

// The reference hash is an XOR of independent per-element contributions, so
// it can be reduced exactly across MPI ranks when global indices are used.
uint64_t computeHash(const World& world, const int mpi_rank) {
    uint64_t local_hash = 0;
    const uint64_t first_global = static_cast<uint64_t>(world.first_row) * world.n_elems_root;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &world.elements_dynamic[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.elements_dynamic[i].total_flux, sizeof(flux_bits));
        const uint64_t global_index = first_global + i;
        local_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return mpi_rank == 0 ? global_hash : 0;
}

void gatherEnergyResults(const World& world, std::vector<double>& energy_data,
                         const int mpi_rank, const int mpi_size) {
    int local_count = static_cast<int>(world.elements_dynamic.size());
    std::vector<int> counts;
    std::vector<int> displacements;
    if (mpi_rank == 0) {
        counts.resize(mpi_size);
        displacements.resize(mpi_size);
        for (int rank = 0; rank < mpi_size; ++rank) {
            const RowPartition partition = partitionRows(world.n_elems_root, world.active_ranks, rank);
            counts[rank] = partition.rows * world.n_elems_root;
            displacements[rank] = partition.first_row * world.n_elems_root;
        }
        energy_data.resize(static_cast<size_t>(world.n_elems_root) * world.n_elems_root);
    }

    std::vector<double> local_energy(world.elements_dynamic.size());
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(world.elements_dynamic.size()); ++i) {
        local_energy[static_cast<size_t>(i)] = world.elements_dynamic[static_cast<size_t>(i)].current_energy;
    }

    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                mpi_rank == 0 ? energy_data.data() : nullptr,
                mpi_rank == 0 ? counts.data() : nullptr,
                mpi_rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI implementation does not provide MPI_THREAD_FUNNELED", mpi_rank);
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root <= 0 || n_iters < 0) {
        abortWithMessage("Grid size must be positive and iterations must be non-negative", mpi_rank);
    }

    const uint64_t n_elems_u64 = static_cast<uint64_t>(n_elems_root) * n_elems_root;
    if (n_elems_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        abortWithMessage("Grid is too large for MPI element counts", mpi_rank);
    }
    const size_t n_elems = static_cast<size_t>(n_elems_u64);

    // One MPI rank drives one GPU. On multi-GPU nodes, use the shared-memory
    // local rank so global MPI rank numbering does not create GPU collisions.
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        abortWithMessage("At least one CUDA device is required per node", mpi_rank);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    CUDA_CHECK(cudaFree(0));

    if (mpi_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI ranks, %d OpenMP threads/rank, CUDA device %d\n",
                    mpi_size, omp_get_max_threads(), local_rank % device_count);
        std::printf("\n");
    }

    if (mpi_rank == 0) std::printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, mpi_rank, mpi_size);

    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    size_t max_total_mem = 0;
    MPI_Reduce(&total_mem, &max_total_mem, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        std::printf("Memory usage/rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    max_total_mem / (1024.0 * 1024.0),
                    static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("\n");
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    runSimulation(world, n_iters, mpi_rank, mpi_size);
    const auto end = std::chrono::steady_clock::now();
    const long long local_duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    long long duration_ns = 0;
    MPI_Reduce(&local_duration_ns, &duration_ns, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        const double duration_ms = duration_ns / 1.0e6;
        std::printf("Computation time: %.3f ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double seconds = std::max(duration_ns / 1.0e9, std::numeric_limits<double>::min());
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / seconds / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    const uint64_t hash_value = computeHash(world, mpi_rank);
    if (mpi_rank == 0) {
        std::printf("  Result hash: %016" PRIX64 "\n", hash_value);
        std::printf("\n");
    }

    if (printResults) {
        std::vector<double> energy_data;
        gatherEnergyResults(world, energy_data, mpi_rank, mpi_size);
        if (mpi_rank == 0) {
            print_results(energy_data, "ElementEnergy");
        }
    }

    bool valid = true;
    if (validate) {
        valid = validateResults(world, mpi_rank);
    }

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
