#include <algorithm>
#include <array>
#include <cinttypes>
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

// World state for the row slab owned by one MPI rank.  The static connectivity
// uses two extra rows of indices for the top and bottom CUDA halo rows.
struct World {
    int n_elems_root = 0;
    int first_row = 0;
    int local_rows = 0;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct RowPartition {
    int first_row;
    int row_count;
};

RowPartition partitionRows(const int n_elems_root, const int rank, const int ranks) {
    const int rows_per_rank = n_elems_root / ranks;
    const int remainder = n_elems_root % ranks;
    const int first_row = rank * rows_per_rank + std::min(rank, remainder);
    return {first_row, rows_per_rank + (rank < remainder ? 1 : 0)};
}

[[noreturn]] void mpiFailure(const int error, const char* expression,
                             const char* file, const int line) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int message_length = 0;
    MPI_Error_string(error, message, &message_length);
    std::fprintf(stderr, "MPI failure at %s:%d (%s): %.*s\n",
                 file, line, expression, message_length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(call)                                                        \
    do {                                                                        \
        const int mpi_error__ = (call);                                        \
        if (mpi_error__ != MPI_SUCCESS) {                                      \
            mpiFailure(mpi_error__, #call, __FILE__, __LINE__);                \
        }                                                                       \
    } while (false)

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n",
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_error__ = (call);                               \
        if (cuda_error__ != cudaSuccess) {                                     \
            cudaFailure(cuda_error__, #call, __FILE__, __LINE__);              \
        }                                                                       \
    } while (false)

// Build the local portion of the 2D mesh.  The order of connections is kept
// identical to the original implementation, which preserves update semantics.
void buildSquare2D(World& world, const int n_elems_root, const int active_rank,
                   const int active_ranks) {
    const RowPartition partition = partitionRows(n_elems_root, active_rank, active_ranks);
    const size_t local_elements = static_cast<size_t>(partition.row_count) *
                                  static_cast<size_t>(n_elems_root);

    world.n_elems_root = n_elems_root;
    world.first_row = partition.first_row;
    world.local_rows = partition.row_count;

    world.materials = {
        Material{0.8, 0.0},    // Default material
        Material{0.8, 0.5},    // Inflow material
        Material{0.8, -0.5}    // Outflow material
    };
    world.elements_static.resize(local_elements);
    world.elements_dynamic.resize(local_elements);
    world.elements_dynamic_swap.resize(local_elements);

    const int last = n_elems_root - 1;
    const size_t top_halo_offset = local_elements;
    const size_t bottom_halo_offset = local_elements +
                                      static_cast<size_t>(n_elems_root);

#pragma omp parallel for schedule(static)
    for (long long local_index = 0;
         local_index < static_cast<long long>(local_elements); ++local_index) {
        const size_t index = static_cast<size_t>(local_index);
        const int local_x = static_cast<int>(index / static_cast<size_t>(n_elems_root));
        const int y = static_cast<int>(index % static_cast<size_t>(n_elems_root));
        const int x = partition.first_row + local_x;

        ElementStatic& elem = world.elements_static[index];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // These assignments intentionally remain independent and ordered as in
        // the reference code, including the n=1 corner case.
        if (x == 0 && y == 0) {
            elem.material_idx = INFLOW_MAT_ID;
        }
        if (x == 0 && y == last) {
            elem.material_idx = OUTFLOW_MAT_ID;
        }
        if (x == last && y == 0) {
            elem.material_idx = OUTFLOW_MAT_ID;
        }
        if (x == last && y == last) {
            elem.material_idx = INFLOW_MAT_ID;
        }

        constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int connection = 0; connection < 4; ++connection) {
            const int nx = x + offsets[connection][0];
            const int ny = y + offsets[connection][1];
            if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root) {
                continue;
            }

            idx_t neighbor_index = 0;
            if (nx < partition.first_row) {
                neighbor_index = static_cast<idx_t>(top_halo_offset +
                                                    static_cast<size_t>(ny));
            } else if (nx >= partition.first_row + partition.row_count) {
                neighbor_index = static_cast<idx_t>(bottom_halo_offset +
                                                    static_cast<size_t>(ny));
            } else {
                neighbor_index = static_cast<idx_t>(
                    static_cast<size_t>(nx - partition.first_row) *
                        static_cast<size_t>(n_elems_root) +
                    static_cast<size_t>(ny));
            }

            const idx_t connection_index = elem.num_connections++;
            elem.connected_idx[connection_index] = neighbor_index;
            elem.connected_flux[connection_index] = 1.0;
        }

        world.elements_dynamic[index] = ElementDynamic{0.0, 0.0};
        world.elements_dynamic_swap[index] = ElementDynamic{0.0, 0.0};
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    DeviceBuffer() = default;
    explicit DeviceBuffer(const size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(const size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
};

struct PinnedBuffer {
    val_t* data = nullptr;

    explicit PinnedBuffer(const size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data),
                                     count * sizeof(val_t), cudaHostAllocDefault));
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data != nullptr) {
            cudaFreeHost(data);
        }
    }
};

__device__ __forceinline__ val_t computeFluxDevice(
    const Material& mat, const val_t this_energy, const val_t connection_flux,
    const val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff *
           connection_flux * 0.25;
}

__global__ void updateRowsKernel(const ElementStatic* elements_static,
                                 const Material* materials,
                                 const ElementDynamic* current,
                                 ElementDynamic* next, const size_t first,
                                 const size_t count) {
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         static_cast<size_t>(threadIdx.x);
    if (local >= count) {
        return;
    }

    const size_t index = first + local;
    const ElementStatic& elem_static = elements_static[index];
    const val_t this_energy = current[index].current_energy;
    const Material& mat = materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t connection = 0; connection < elem_static.num_connections; ++connection) {
        const idx_t neighbor_index = elem_static.connected_idx[connection];
        total_flux += computeFluxDevice(
            mat, this_energy, elem_static.connected_flux[connection],
            current[neighbor_index].current_energy);
    }

    next[index].current_energy = this_energy + total_flux;
    next[index].total_flux = current[index].total_flux + fabs(total_flux);
}

__global__ void updateBoundaryKernel(const ElementStatic* elements_static,
                                     const Material* materials,
                                     const ElementDynamic* current,
                                     ElementDynamic* next, const int local_rows,
                                     const int n_elems_root) {
    const int boundary = static_cast<int>(blockIdx.x) * blockDim.x +
                         static_cast<int>(threadIdx.x);
    const int boundary_count = local_rows == 1 ? 1 : 2;
    if (boundary >= boundary_count) {
        return;
    }

    const int row = boundary == 0 ? 0 : local_rows - 1;
    const size_t column = static_cast<size_t>(blockIdx.y) *
                          static_cast<size_t>(blockDim.y) +
                          static_cast<size_t>(threadIdx.y);
    if (column >= static_cast<size_t>(n_elems_root)) {
        return;
    }
    const size_t index = static_cast<size_t>(row) *
                         static_cast<size_t>(n_elems_root) + column;

    const ElementStatic& elem_static = elements_static[index];
    const val_t this_energy = current[index].current_energy;
    const Material& mat = materials[elem_static.material_idx];
    val_t total_flux = mat.external_flow;
    for (idx_t connection = 0; connection < elem_static.num_connections; ++connection) {
        const idx_t neighbor_index = elem_static.connected_idx[connection];
        total_flux += computeFluxDevice(
            mat, this_energy, elem_static.connected_flux[connection],
            current[neighbor_index].current_energy);
    }

    next[index].current_energy = this_energy + total_flux;
    next[index].total_flux = current[index].total_flux + fabs(total_flux);
}

// The boundary kernel uses a two-dimensional block: x selects top/bottom and
// y selects an element in the row.  The interior kernel uses a regular 1D grid.
void launchInterior(const DeviceBuffer<ElementStatic>& d_static,
                    const DeviceBuffer<Material>& d_materials,
                    ElementDynamic* current, ElementDynamic* next,
                    const int n_elems_root, const int local_rows,
                    cudaStream_t stream) {
    if (local_rows <= 2) {
        return;
    }

    const size_t count = static_cast<size_t>(local_rows - 2) *
                         static_cast<size_t>(n_elems_root);
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (count + threads - 1) / threads);
    updateRowsKernel<<<blocks, threads, 0, stream>>>(
        d_static.data, d_materials.data, current, next,
        static_cast<size_t>(n_elems_root), count);
    CUDA_CHECK(cudaGetLastError());
}

void launchBoundaries(const DeviceBuffer<ElementStatic>& d_static,
                      const DeviceBuffer<Material>& d_materials,
                      ElementDynamic* current, ElementDynamic* next,
                      const int n_elems_root, const int local_rows,
                      cudaStream_t stream) {
    const dim3 threads(1, 256, 1);
    const unsigned int boundary_count = local_rows == 1 ? 1U : 2U;
    const unsigned int column_blocks = static_cast<unsigned int>(
        (n_elems_root + static_cast<int>(threads.y) - 1) / threads.y);
    // The x dimension selects the boundary.  Use one block in x and let the
    // y grid cover wide rows without imposing a grid-size limit.
    updateBoundaryKernel<<<dim3(boundary_count, column_blocks, 1), threads, 0, stream>>>(
        d_static.data, d_materials.data, current, next, local_rows,
        n_elems_root);
    CUDA_CHECK(cudaGetLastError());
}

void runSimulation(World& world, const int n_iters, MPI_Comm active_comm,
                    const int active_rank, const int active_ranks) {
    const int n_elems_root = world.n_elems_root;
    const int local_rows = world.local_rows;
    const size_t owned_elements = world.elements_static.size();
    const size_t halo_elements = static_cast<size_t>(2) *
                                 static_cast<size_t>(n_elems_root);
    const size_t device_elements = owned_elements + halo_elements;

    DeviceBuffer<Material> d_materials(world.materials.size());
    DeviceBuffer<ElementStatic> d_static(owned_elements);
    DeviceBuffer<ElementDynamic> d_state_a(device_elements);
    DeviceBuffer<ElementDynamic> d_state_b(device_elements);

    CUDA_CHECK(cudaMemcpy(d_materials.data, world.materials.data(),
                          world.materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_static.data, world.elements_static.data(),
                          owned_elements * sizeof(ElementStatic),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_state_a.data, 0,
                          device_elements * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemset(d_state_b.data, 0,
                          device_elements * sizeof(ElementDynamic)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    ElementDynamic* current = d_state_a.data;
    ElementDynamic* next = d_state_b.data;

    const int top_rank = active_rank > 0 ? active_rank - 1 : MPI_PROC_NULL;
    const int bottom_rank = active_rank + 1 < active_ranks
                                ? active_rank + 1
                                : MPI_PROC_NULL;
    PinnedBuffer send_top(static_cast<size_t>(n_elems_root));
    PinnedBuffer send_bottom(static_cast<size_t>(n_elems_root));
    PinnedBuffer recv_top(static_cast<size_t>(n_elems_root));
    PinnedBuffer recv_bottom(static_cast<size_t>(n_elems_root));

    constexpr int TAG_TOP = 2401;
    constexpr int TAG_BOTTOM = 2402;

    for (int iter = 0; iter < n_iters; ++iter) {
        const bool exchanges_halo = top_rank != MPI_PROC_NULL ||
                                    bottom_rank != MPI_PROC_NULL;
        // Pack the two rows directly from the GPU.  The stream synchronization
        // is limited to the small halo copies; it leaves the following interior
        // kernel free to overlap with MPI communication.
        if (top_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                send_top.data, sizeof(val_t),
                current, sizeof(ElementDynamic),
                sizeof(val_t), static_cast<size_t>(n_elems_root),
                cudaMemcpyDeviceToHost, stream));
        }
        if (bottom_rank != MPI_PROC_NULL) {
            const size_t bottom_offset = static_cast<size_t>(local_rows - 1) *
                                         static_cast<size_t>(n_elems_root);
            CUDA_CHECK(cudaMemcpy2DAsync(
                send_bottom.data, sizeof(val_t),
                current + bottom_offset, sizeof(ElementDynamic),
                sizeof(val_t), static_cast<size_t>(n_elems_root),
                cudaMemcpyDeviceToHost, stream));
        }
        if (exchanges_halo) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::array<MPI_Request, 4> requests{};
        int request_count = 0;
        if (top_rank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_top.data, n_elems_root, MPI_DOUBLE,
                                top_rank, TAG_BOTTOM, active_comm,
                                &requests[request_count++]));
            MPI_CHECK(MPI_Isend(send_top.data, n_elems_root, MPI_DOUBLE,
                                top_rank, TAG_TOP, active_comm,
                                &requests[request_count++]));
        }
        if (bottom_rank != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_bottom.data, n_elems_root, MPI_DOUBLE,
                                bottom_rank, TAG_TOP, active_comm,
                                &requests[request_count++]));
            MPI_CHECK(MPI_Isend(send_bottom.data, n_elems_root, MPI_DOUBLE,
                                bottom_rank, TAG_BOTTOM, active_comm,
                                &requests[request_count++]));
        }

        // Interior elements do not depend on either halo and can run while MPI
        // progresses on the host.
        launchInterior(d_static, d_materials, current, next, n_elems_root,
                       local_rows, stream);

        if (request_count != 0) {
            MPI_CHECK(MPI_Waitall(request_count, requests.data(),
                                  MPI_STATUSES_IGNORE));
        }

        // Install received rows into the current-state halo.  The default
        // stream ordering guarantees the interior update has completed before
        // the boundary update reads the same state buffer.
        if (top_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                current + owned_elements, sizeof(ElementDynamic),
                recv_top.data, sizeof(val_t), sizeof(val_t),
                static_cast<size_t>(n_elems_root),
                cudaMemcpyHostToDevice, stream));
        }
        if (bottom_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                current + owned_elements + static_cast<size_t>(n_elems_root),
                sizeof(ElementDynamic), recv_bottom.data, sizeof(val_t),
                sizeof(val_t), static_cast<size_t>(n_elems_root),
                cudaMemcpyHostToDevice, stream));
        }

        launchBoundaries(d_static, d_materials, current, next, n_elems_root,
                         local_rows, stream);
        std::swap(current, next);
    }

    CUDA_CHECK(cudaMemcpyAsync(
        world.elements_dynamic.data(), current,
        owned_elements * sizeof(ElementDynamic), cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Compute the same hash as the reference program, with the global element
// index used on every rank.  XOR makes the distributed reduction exact.
uint64_t computeHash(const std::vector<ElementDynamic>& elements,
                     const size_t global_offset) {
    uint64_t result_hash = 0;

#pragma omp parallel for schedule(static) reduction(^:result_hash)
    for (long long local_index = 0;
         local_index < static_cast<long long>(elements.size()); ++local_index) {
        const size_t index = static_cast<size_t>(local_index);
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits, &elements[index].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[index].total_flux, sizeof(flux_bits));
        const uint64_t global_index = static_cast<uint64_t>(global_offset + index);
        result_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        result_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return result_hash;
}

bool validateResults(const World& world, MPI_Comm comm, const int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

#pragma omp parallel for schedule(static) reduction(+:local_energy_sum,local_flux_sum) \
    reduction(max:local_energy_max) reduction(min:local_energy_min)
    for (long long index = 0;
         index < static_cast<long long>(world.elements_dynamic.size()); ++index) {
        const ElementDynamic& elem = world.elements_dynamic[static_cast<size_t>(index)];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(local_energy_max, elem.current_energy);
        local_energy_min = std::min(local_energy_min, elem.current_energy);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    MPI_CHECK(MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE,
                         MPI_SUM, 0, comm));
    MPI_CHECK(MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE,
                         MPI_SUM, 0, comm));
    MPI_CHECK(MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE,
                         MPI_MAX, 0, comm));
    MPI_CHECK(MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE,
                         MPI_MIN, 0, comm));

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        if (std::abs(energy_sum) > energy_epsilon) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid != 0) {
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, comm));
    return valid != 0;
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

int localRankFromEnvironment() {
    constexpr const char* names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID", "MPI_LOCALRANKID"
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return std::max(0, std::atoi(value));
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    bool parse_ok = true;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_ok = false;
        }
    }

    if (!parse_ok || n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0 && parse_ok) {
            std::printf("Grid size must be positive and iterations cannot be negative.\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    // There is at most one useful rank per mesh row.  Extra MPI ranks remain
    // collective participants but do no work, avoiding zero-sized CUDA grids.
    const int active_ranks = std::min(ranks, n_elems_root);
    const bool active = rank < active_ranks;
    MPI_Comm active_comm = MPI_COMM_NULL;
    int active_rank = -1;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED,
                             rank, &active_comm));

    World world;
    int device_count = 0;
    int selected_device = -1;
    int local_threads = 0;
    if (active) {
        MPI_CHECK(MPI_Comm_rank(active_comm, &active_rank));
        buildSquare2D(world, n_elems_root, active_rank, active_ranks);

        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            std::fprintf(stderr, "MPI rank %d found no CUDA device.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        selected_device = localRankFromEnvironment() % device_count;
        CUDA_CHECK(cudaSetDevice(selected_device));
        local_threads = omp_get_max_threads();
    }

    unsigned long long local_static_mem = active
                                            ? static_cast<unsigned long long>(
                                                  world.elements_static.size() *
                                                  sizeof(ElementStatic))
                                            : 0ULL;
    unsigned long long local_dynamic_mem = active
                                             ? static_cast<unsigned long long>(
                                                   world.elements_dynamic.size() *
                                                   sizeof(ElementDynamic) * 2)
                                             : 0ULL;
    unsigned long long global_static_mem = 0;
    unsigned long long global_dynamic_mem = 0;
    MPI_CHECK(MPI_Reduce(&local_static_mem, &global_static_mem, 1,
                         MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_dynamic_mem, &global_dynamic_mem, 1,
                         MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                         MPI_COMM_WORLD));

    const size_t n_elems = static_cast<size_t>(n_elems_root) *
                           static_cast<size_t>(n_elems_root);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root,
                    n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (active: %d), OpenMP threads/rank: %d, CUDA devices visible: %d\n",
                    ranks, active_ranks, local_threads, device_count);
        std::printf("\n");
        std::printf("Building unstructured mesh...\n");
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    (global_static_mem + global_dynamic_mem) / (1024.0 * 1024.0),
                    global_static_mem / (1024.0 * 1024.0),
                    global_dynamic_mem / (1024.0 * 1024.0));
        std::printf("\n");
        std::printf("Running simulation...\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    if (active) {
        runSimulation(world, n_iters, active_comm, active_rank, active_ranks);
    }
    const double local_duration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_CHECK(MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD));

    if (rank == 0) {
        const long long duration_ms = static_cast<long long>(duration * 1000.0);
        std::printf("Computation time: %lld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration * 1000.0 / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) /
            duration / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    uint64_t local_hash = 0;
    if (active) {
        local_hash = computeHash(world.elements_dynamic,
                                 static_cast<size_t>(world.first_row) *
                                     static_cast<size_t>(n_elems_root));
    }
    uint64_t global_hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR,
                         0, MPI_COMM_WORLD));
    if (rank == 0) {
        std::printf("  Result hash: %016" PRIX64 "\n", global_hash);
        std::printf("\n");
    }

    if (printResults) {
        const int local_count = active
                                    ? static_cast<int>(world.elements_dynamic.size())
                                    : 0;
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<val_t> local_energy(static_cast<size_t>(local_count));
#pragma omp parallel for schedule(static) if (active)
        for (int i = 0; i < local_count; ++i) {
            local_energy[static_cast<size_t>(i)] =
                world.elements_dynamic[static_cast<size_t>(i)].current_energy;
        }

        if (rank == 0) {
            counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
        }
        MPI_CHECK(MPI_Gather(&local_count, 1, MPI_INT,
                             rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
                             0, MPI_COMM_WORLD));

        std::vector<val_t> all_energy;
        if (rank == 0) {
            int displacement = 0;
            for (int r = 0; r < ranks; ++r) {
                displacements[static_cast<size_t>(r)] = displacement;
                displacement += counts[static_cast<size_t>(r)];
            }
            all_energy.resize(static_cast<size_t>(displacement));
        }
        MPI_CHECK(MPI_Gatherv(
            local_energy.data(), local_count, MPI_DOUBLE,
            rank == 0 ? all_energy.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
            MPI_COMM_WORLD));
        if (rank == 0) {
            print_results(all_energy, "ElementEnergy");
        }
    }

    bool valid = true;
    if (validate) {
        if (active) {
            valid = validateResults(world, active_comm, active_rank);
        }
        int valid_int = valid ? 1 : 0;
        MPI_CHECK(MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD));
        valid = valid_int != 0;
    }

    if (active) {
        MPI_CHECK(MPI_Comm_free(&active_comm));
    }
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
