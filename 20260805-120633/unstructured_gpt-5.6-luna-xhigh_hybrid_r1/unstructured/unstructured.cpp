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

// Types to represent unstructured mesh elements.
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (the generated mesh uses four).
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer.
struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Static connectivity information for each element.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

// Dynamic state for each element.  The two values are deliberately contiguous
// so that MPI can gather owned states as an array of MPI_DOUBLE values.
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t),
              "ElementDynamic must be a contiguous pair of values");

// Material type IDs.
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// A rank owns a contiguous set of rows.  Its dynamic arrays contain one ghost
// row on each side, so local row one is the first owned row.
struct World {
    int n_elems_root = 0;
    int first_row = 0;
    int local_rows = 0;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    size_t owned_count() const {
        return static_cast<size_t>(local_rows) * static_cast<size_t>(n_elems_root);
    }

    size_t owned_offset() const {
        return static_cast<size_t>(n_elems_root);
    }

    size_t dynamic_count() const {
        return static_cast<size_t>(local_rows + 2) * static_cast<size_t>(n_elems_root);
    }
};

struct RowPartition {
    int first_row;
    int rows;
};

RowPartition partition_for_rank(const int n_elems_root, const int rank,
                                const int active_ranks) {
    if (rank >= active_ranks) {
        return RowPartition{n_elems_root, 0};
    }

    const int base_rows = n_elems_root / active_ranks;
    const int remainder = n_elems_root % active_ranks;
    const int rows = base_rows + (rank < remainder ? 1 : 0);
    const int first_row = rank * base_rows + std::min(rank, remainder);
    return RowPartition{first_row, rows};
}

void mpi_check(const int error, const char* expression, const char* file,
               const int line) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char message[MPI_MAX_ERROR_STRING];
    int message_length = 0;
    MPI_Error_string(error, message, &message_length);
    std::fprintf(stderr, "MPI error at %s:%d in %s: %.*s\n", file, line,
                 expression, message_length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::exit(EXIT_FAILURE);
}

#define MPI_CHECK(expression) \
    mpi_check((expression), #expression, __FILE__, __LINE__)

void cuda_check(const cudaError_t error, const char* expression,
                const char* file, const int line) {
    if (error == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    cuda_check((expression), #expression, __FILE__, __LINE__)

// Material data is tiny and read by every CUDA thread, so keep it in the
// constant cache rather than making every element fetch it from global memory.
__constant__ Material device_materials[3];

__global__ void update_elements(const ElementStatic* elements_static,
                                const ElementDynamic* current,
                                ElementDynamic* next, const idx_t n_owned,
                                const idx_t owned_offset) {
    const idx_t local_index = static_cast<idx_t>(blockIdx.x) * blockDim.x +
                              static_cast<idx_t>(threadIdx.x);
    if (local_index >= n_owned) {
        return;
    }

    const ElementStatic& element = elements_static[local_index];
    const idx_t current_index = owned_offset + local_index;
    const val_t this_energy = current[current_index].current_energy;
    const Material& material = device_materials[element.material_idx];

    // Keep the same connection order and arithmetic as the original scalar
    // implementation.  All reads come from the previous iteration's buffer.
    val_t total_flux = material.external_flow;
    for (idx_t connection = 0; connection < element.num_connections;
         ++connection) {
        const idx_t neighbor_index = element.connected_idx[connection];
        const val_t neighbor_energy = current[neighbor_index].current_energy;
        total_flux += (neighbor_energy - this_energy) *
                      material.transfer_coeff *
                      element.connected_flux[connection] * 0.25;
    }

    next[current_index].current_energy = this_energy + total_flux;
    next[current_index].total_flux = current[current_index].total_flux +
                                     fabs(total_flux);
}

// Build the local portion of the mesh.  Connectivity indices are local device
// indices: owned rows are [1, local_rows], with ghost rows at 0 and local_rows+1.
void buildSquare2D(World& world, const int n_elems_root, const int rank,
                   const int world_size) {
    const int active_ranks = std::min(world_size, n_elems_root);
    const RowPartition partition =
        partition_for_rank(n_elems_root, rank, active_ranks);

    world.n_elems_root = n_elems_root;
    world.first_row = partition.first_row;
    world.local_rows = partition.rows;

    world.materials = {
        Material{0.8, 0.0},   // Default material
        Material{0.8, 0.5},   // Inflow material
        Material{0.8, -0.5},  // Outflow material
    };

    const size_t n_owned = world.owned_count();
    const size_t n_dynamic = world.dynamic_count();
    world.elements_static.resize(n_owned);
    world.elements_dynamic.resize(n_dynamic);
    world.elements_dynamic_swap.resize(n_dynamic);

    // The initialization is intentionally parallel even when the GPU handles
    // the iterative solve: it keeps large multi-rank setup costs off one CPU
    // thread and exercises the OpenMP part of the hybrid implementation.
#pragma omp parallel for schedule(static)
    for (int64_t local_index = 0;
         local_index < static_cast<int64_t>(n_owned); ++local_index) {
        world.elements_dynamic[static_cast<size_t>(local_index)] =
            ElementDynamic{0.0, 0.0};
        world.elements_dynamic_swap[static_cast<size_t>(local_index)] =
            ElementDynamic{0.0, 0.0};
    }

#pragma omp parallel for schedule(static)
    for (int64_t local_index = 0;
         local_index < static_cast<int64_t>(n_owned); ++local_index) {
        const int local_row = static_cast<int>(
            local_index / static_cast<int64_t>(n_elems_root));
        const int y = static_cast<int>(local_index % n_elems_root);
        const int x = partition.first_row + local_row;

        ElementStatic element{};
        element.material_idx = DEFAULT_MAT_ID;
        element.num_connections = 0;

        // Connect to neighbors (up, down, left, right), matching the original
        // unstructured connectivity construction and traversal order.
        constexpr int offsets[4][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int connection = 0; connection < 4; ++connection) {
            const int nx = x + offsets[connection][0];
            const int ny = y + offsets[connection][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 &&
                ny < n_elems_root) {
                const int local_neighbor_row = nx - partition.first_row + 1;
                const size_t local_neighbor_index =
                    static_cast<size_t>(local_neighbor_row) *
                        static_cast<size_t>(n_elems_root) +
                    static_cast<size_t>(ny);
                element.connected_idx[element.num_connections] =
                    static_cast<idx_t>(local_neighbor_index);
                element.connected_flux[element.num_connections] = 1.0;
                ++element.num_connections;
            }
        }

        world.elements_static[static_cast<size_t>(local_index)] = element;
    }

    // Set corner elements as inflow/outflow.  A corner belongs to exactly one
    // rank; the sequence below also preserves the original n=1 behavior.
    const int last = n_elems_root - 1;
    const auto set_material_if_owned = [&](const int row, const int column,
                                           const idx_t material) {
        if (row >= partition.first_row &&
            row < partition.first_row + partition.rows) {
            const size_t local_index =
                static_cast<size_t>(row - partition.first_row) *
                    static_cast<size_t>(n_elems_root) +
                static_cast<size_t>(column);
            world.elements_static[local_index].material_idx = material;
        }
    };
    set_material_if_owned(0, 0, INFLOW_MAT_ID);
    set_material_if_owned(0, last, OUTFLOW_MAT_ID);
    set_material_if_owned(last, 0, OUTFLOW_MAT_ID);
    set_material_if_owned(last, last, INFLOW_MAT_ID);

    // Ghost rows are zero-initialized as well.  They are overwritten by MPI
    // before they are read whenever the corresponding neighbor exists.
#pragma omp parallel for schedule(static)
    for (int64_t index = static_cast<int64_t>(n_owned);
         index < static_cast<int64_t>(n_dynamic); ++index) {
        world.elements_dynamic[static_cast<size_t>(index)] =
            ElementDynamic{0.0, 0.0};
        world.elements_dynamic_swap[static_cast<size_t>(index)] =
            ElementDynamic{0.0, 0.0};
    }
}

struct PinnedBuffer {
    val_t* data = nullptr;
    size_t count = 0;

    PinnedBuffer() = default;

    void allocate(const size_t new_count) {
        count = new_count;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data),
                                 count * sizeof(val_t), cudaHostAllocPortable));
    }

    ~PinnedBuffer() {
        if (data != nullptr) {
            cudaFreeHost(data);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

void runSimulation(World& world, const int n_iters, const int rank,
                   const int world_size, cudaStream_t stream) {
    const size_t n_owned = world.owned_count();
    const size_t n_dynamic = world.dynamic_count();
    const size_t n_root = static_cast<size_t>(world.n_elems_root);
    const int active_ranks = std::min(world_size, world.n_elems_root);
    const bool has_top_neighbor = rank > 0 && rank < active_ranks;
    const bool has_bottom_neighbor = rank + 1 < active_ranks;

    ElementStatic* device_static = nullptr;
    ElementDynamic* device_current = nullptr;
    ElementDynamic* device_next = nullptr;

    if (n_owned != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_static),
                              n_owned * sizeof(ElementStatic)));
        CUDA_CHECK(cudaMemcpy(device_static, world.elements_static.data(),
                              n_owned * sizeof(ElementStatic),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_current),
                          n_dynamic * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_next),
                          n_dynamic * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemsetAsync(device_current, 0,
                               n_dynamic * sizeof(ElementDynamic), stream));
    CUDA_CHECK(cudaMemsetAsync(device_next, 0,
                               n_dynamic * sizeof(ElementDynamic), stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        device_materials, world.materials.data(),
        world.materials.size() * sizeof(Material), 0, cudaMemcpyHostToDevice,
        stream));

    PinnedBuffer send_top;
    PinnedBuffer send_bottom;
    PinnedBuffer receive_top;
    PinnedBuffer receive_bottom;
    send_top.allocate(n_root);
    send_bottom.allocate(n_root);
    receive_top.allocate(n_root);
    receive_bottom.allocate(n_root);

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>((n_owned + block_size - 1) / block_size);

    for (int iteration = 0; iteration < n_iters; ++iteration) {
        if (n_owned != 0) {
            update_elements<<<grid_size, block_size, 0, stream>>>(
                device_static, device_current, device_next,
                static_cast<idx_t>(n_owned),
                static_cast<idx_t>(world.owned_offset()));
            CUDA_CHECK(cudaGetLastError());

            // Only the two newly computed boundary rows participate in MPI.
            // Pinned buffers make these copies asynchronous with respect to the
            // host, while the single stream preserves iteration ordering.
            if (has_top_neighbor) {
                CUDA_CHECK(cudaMemcpyAsync(
                    send_top.data,
                    device_next + world.owned_offset(),
                    n_root * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
            }
            if (has_bottom_neighbor) {
                const size_t bottom_owned_offset =
                    static_cast<size_t>(world.local_rows) * n_root;
                CUDA_CHECK(cudaMemcpyAsync(
                    send_bottom.data, device_next + bottom_owned_offset,
                    n_root * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_Request requests[4];
        int request_count = 0;
        constexpr int top_tag = 401;
        constexpr int bottom_tag = 402;

        if (has_top_neighbor) {
            // The rank above sends its bottom row with bottom_tag.
            MPI_CHECK(MPI_Irecv(receive_top.data,
                                static_cast<int>(n_root), MPI_DOUBLE,
                                rank - 1, bottom_tag, MPI_COMM_WORLD,
                                &requests[request_count++]));
            MPI_CHECK(MPI_Isend(send_top.data, static_cast<int>(n_root),
                                MPI_DOUBLE, rank - 1, top_tag,
                                MPI_COMM_WORLD, &requests[request_count++]));
        }
        if (has_bottom_neighbor) {
            // The rank below sends its top row with top_tag.
            MPI_CHECK(MPI_Irecv(receive_bottom.data,
                                static_cast<int>(n_root), MPI_DOUBLE,
                                rank + 1, top_tag, MPI_COMM_WORLD,
                                &requests[request_count++]));
            MPI_CHECK(MPI_Isend(send_bottom.data, static_cast<int>(n_root),
                                MPI_DOUBLE, rank + 1, bottom_tag,
                                MPI_COMM_WORLD, &requests[request_count++]));
        }
        if (request_count != 0) {
            MPI_CHECK(MPI_Waitall(request_count, requests,
                                  MPI_STATUSES_IGNORE));
        }

        std::swap(device_current, device_next);

        // Install received values into the newly active current buffer.  These
        // copies are ordered before the next kernel launch in the same stream.
        if (has_top_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(
                device_current, receive_top.data, n_root * sizeof(val_t),
                cudaMemcpyHostToDevice, stream));
        }
        if (has_bottom_neighbor) {
            const size_t bottom_ghost_offset =
                static_cast<size_t>(world.local_rows + 1) * n_root;
            CUDA_CHECK(cudaMemcpyAsync(
                device_current + bottom_ghost_offset, receive_bottom.data,
                n_root * sizeof(val_t), cudaMemcpyHostToDevice, stream));
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(world.elements_dynamic.data(), device_current,
                               n_dynamic * sizeof(ElementDynamic),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(device_next));
    CUDA_CHECK(cudaFree(device_current));
    if (device_static != nullptr) {
        CUDA_CHECK(cudaFree(device_static));
    }
}

// Validate simulation results.
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
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

// Compute the original XOR hash over a global result vector.
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t result = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits, &elements[i].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        result ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

uint64_t computeLocalHash(const World& world) {
    uint64_t result = 0;
    const size_t n_owned = world.owned_count();
    const size_t global_offset =
        static_cast<size_t>(world.first_row) *
        static_cast<size_t>(world.n_elems_root);

#pragma omp parallel for reduction(^ : result) schedule(static)
    for (int64_t local_index = 0;
         local_index < static_cast<int64_t>(n_owned); ++local_index) {
        const size_t local = static_cast<size_t>(local_index);
        const size_t global = global_offset + local;
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits,
                    &world.elements_dynamic[world.owned_offset() + local]
                         .current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits,
                    &world.elements_dynamic[world.owned_offset() + local]
                         .total_flux,
                    sizeof(flux_bits));
        result ^= (energy_bits + global) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + global) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

int checked_mpi_count(const size_t count) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "MPI message is too large for this MPI ABI\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(count);
}

void gatherResults(const World& world, const int rank, const int world_size,
                   std::vector<ElementDynamic>& global_elements) {
    const int active_ranks = std::min(world_size, world.n_elems_root);
    const size_t local_values = world.owned_count() * 2;
    const int send_count = checked_mpi_count(local_values);

    std::vector<int> receive_counts;
    std::vector<int> displacements;
    if (rank == 0) {
        const size_t global_count =
            static_cast<size_t>(world.n_elems_root) *
            static_cast<size_t>(world.n_elems_root);
        global_elements.resize(global_count);
        receive_counts.resize(world_size, 0);
        displacements.resize(world_size, 0);
        for (int source = 0; source < world_size; ++source) {
            const RowPartition partition = partition_for_rank(
                world.n_elems_root, source, active_ranks);
            const size_t first_value =
                static_cast<size_t>(partition.first_row) *
                static_cast<size_t>(world.n_elems_root) * 2;
            const size_t values = static_cast<size_t>(partition.rows) *
                                  static_cast<size_t>(world.n_elems_root) * 2;
            receive_counts[source] = checked_mpi_count(values);
            displacements[source] = checked_mpi_count(first_value);
        }
    }

    const val_t* send_buffer = reinterpret_cast<const val_t*>(
        world.elements_dynamic.data() + world.owned_offset());
    val_t* receive_buffer = nullptr;
    if (rank == 0) {
        receive_buffer = reinterpret_cast<val_t*>(global_elements.data());
    }
    MPI_CHECK(MPI_Gatherv(
        send_buffer, send_count, MPI_DOUBLE, receive_buffer,
        rank == 0 ? receive_counts.data() : nullptr,
        rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
        MPI_COMM_WORLD));
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

bool parseInteger(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                              &provided_thread_level));

    int rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation did not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    bool show_help = false;
    bool parse_error = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            parse_error = !parseInteger(argv[++i], n_elems_root) || parse_error;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parse_error = !parseInteger(argv[++i], n_iters) || parse_error;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            parse_error = true;
        }
    }

    if (show_help || parse_error || n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) {
            if (parse_error || n_elems_root <= 0 || n_iters < 0) {
                std::fprintf(stderr, "Invalid command line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return parse_error || n_elems_root <= 0 || n_iters < 0 ? 1 : 0;
    }

    // A local communicator gives stable one-rank-per-GPU assignment on every
    // node, independent of how MPI numbers ranks globally.
    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_communicator));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_communicator, &local_rank));

    int device_count = 0;
    const cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
    if (device_count_error != cudaSuccess || device_count == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No usable CUDA device was found: %s\n",
                         cudaGetErrorString(device_count_error));
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device_id = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device_id));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    omp_set_dynamic(0);

    if (rank == 0) {
        const size_t n_elems = static_cast<size_t>(n_elems_root) *
                              static_cast<size_t>(n_elems_root);
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root,
                    n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", world_size);
        std::printf("OpenMP threads/rank: %d\n", omp_get_max_threads());
        std::printf("CUDA device/rank: %s\n", device_properties.name);
        std::printf("\n");
        std::printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, world_size);

    if (rank == 0) {
        const size_t n_elems = static_cast<size_t>(n_elems_root) *
                              static_cast<size_t>(n_elems_root);
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    total_mem / (1024.0 * 1024.0),
                    static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("\n");
        std::printf("Running simulation...\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters, rank, world_size, stream);
    const auto end = std::chrono::high_resolution_clock::now();

    const long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    long long duration_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG,
                         MPI_MAX, 0, MPI_COMM_WORLD));

    const bool need_global_results = validate || printResults;
    std::vector<ElementDynamic> global_elements;
    if (need_global_results) {
        gatherResults(world, rank, world_size, global_elements);
    }

    uint64_t result_hash = 0;
    if (need_global_results) {
        if (rank == 0) {
            result_hash = computeHash(global_elements);
        }
    } else {
        const uint64_t local_hash = computeLocalHash(world);
        MPI_CHECK(MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T,
                             MPI_BXOR, 0, MPI_COMM_WORLD));
    }

    if (rank == 0) {
        const size_t n_elems = static_cast<size_t>(n_elems_root) *
                              static_cast<size_t>(n_elems_root);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double elapsed_seconds =
            std::max(static_cast<double>(duration_ms) / 1000.0, 1.0e-9);
        const double time_per_iter =
            static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) /
            elapsed_seconds / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        std::printf("Computation time: %lld ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016lX\n",
                    static_cast<unsigned long>(result_hash));
        std::printf("\n");

        if (printResults) {
            std::vector<double> energy_data(global_elements.size());
#pragma omp parallel for schedule(static)
            for (int64_t index = 0;
                 index < static_cast<int64_t>(global_elements.size()); ++index) {
                energy_data[static_cast<size_t>(index)] =
                    global_elements[static_cast<size_t>(index)].current_energy;
            }
            print_results(energy_data, "ElementEnergy");
        }
    }

    int validation_status = 0;
    if (validate && rank == 0) {
        validation_status = validateResults(global_elements) ? 0 : 1;
    }
    if (validate) {
        MPI_CHECK(MPI_Bcast(&validation_status, 1, MPI_INT, 0,
                            MPI_COMM_WORLD));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_CHECK(MPI_Comm_free(&local_communicator));
    MPI_CHECK(MPI_Finalize());
    return validation_status;
}
