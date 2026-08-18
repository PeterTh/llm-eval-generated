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

using idx_t = uint64_t;
using val_t = double;

// The mesh currently has four neighbors per element, but retaining spare
// connectivity slots keeps the representation suitable for irregular meshes.
constexpr int MAX_CONNECTIONS = 8;
constexpr int CUDA_THREADS = 256;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// A World is the portion of the global mesh owned by one MPI rank.  Its
// element indices are local, while row_start identifies their global rows.
struct World {
    int n_elems_root = 0;
    int row_start = 0;
    int local_rows = 0;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression)                                                     \
    do {                                                                            \
        const cudaError_t cuda_status__ = (expression);                            \
        if (cuda_status__ != cudaSuccess) {                                        \
            cudaFailure(cuda_status__, #expression, __FILE__, __LINE__);           \
        }                                                                           \
    } while (false)

[[noreturn]] void mpiFailure(const int error, const char* expression,
                             const char* file, const int line) {
    char error_string[MPI_MAX_ERROR_STRING] = {};
    int error_length = 0;
    MPI_Error_string(error, error_string, &error_length);
    std::fprintf(stderr, "MPI error at %s:%d for %s: %.*s\n", file, line,
                 expression, error_length, error_string);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define MPI_CHECK(expression)                                                       \
    do {                                                                            \
        const int mpi_status__ = (expression);                                      \
        if (mpi_status__ != MPI_SUCCESS) {                                          \
            mpiFailure(mpi_status__, #expression, __FILE__, __LINE__);              \
        }                                                                           \
    } while (false)

void buildSquare2D(World& world, const int n_elems_root, const int row_start,
                   const int local_rows) {
    world.n_elems_root = n_elems_root;
    world.row_start = row_start;
    world.local_rows = local_rows;
    world.materials = {
        Material{0.8, 0.0},
        Material{0.8, 0.5},
        Material{0.8, -0.5},
    };

    const size_t local_element_count =
        static_cast<size_t>(local_rows) * static_cast<size_t>(n_elems_root);
    world.elements_static.resize(local_element_count);
    world.elements_dynamic.resize(local_element_count);
    world.elements_dynamic_swap.resize(local_element_count);

    // Every element is independent during construction, so mesh generation
    // scales across the host threads without requiring synchronization.
#pragma omp parallel for schedule(static)
    for (int64_t local_index = 0;
         local_index < static_cast<int64_t>(local_element_count); ++local_index) {
        const int local_row = static_cast<int>(local_index / n_elems_root);
        const int y = static_cast<int>(local_index % n_elems_root);
        const int x = row_start + local_row;
        ElementStatic& element = world.elements_static[local_index];
        element.material_idx = DEFAULT_MAT_ID;
        element.num_connections = 0;

        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int neighbor = 0; neighbor < 4; ++neighbor) {
            const int nx = x + offsets[neighbor][0];
            const int ny = y + offsets[neighbor][1];
            if (nx < 0 || nx >= n_elems_root || ny < 0 || ny >= n_elems_root) {
                continue;
            }

            // Device state contains one halo row above and below the local
            // rows.  Thus a remote neighbor maps to row zero or local_rows+1.
            int state_row = 0;
            if (nx < row_start) {
                state_row = 0;
            } else if (nx >= row_start + local_rows) {
                state_row = local_rows + 1;
            } else {
                state_row = nx - row_start + 1;
            }
            element.connected_idx[element.num_connections] =
                static_cast<idx_t>(state_row * n_elems_root + ny);
            element.connected_flux[element.num_connections] = 1.0;
            ++element.num_connections;
        }

        world.elements_dynamic[local_index] = ElementDynamic{0.0, 0.0};
        world.elements_dynamic_swap[local_index] = ElementDynamic{0.0, 0.0};
    }

    const int last = n_elems_root - 1;
    auto set_material_if_local = [&](const int global_row, const int column,
                                     const idx_t material) {
        if (global_row >= row_start && global_row < row_start + local_rows) {
            const size_t local_index =
                static_cast<size_t>(global_row - row_start) * n_elems_root +
                static_cast<size_t>(column);
            world.elements_static[local_index].material_idx = material;
        }
    };
    set_material_if_local(0, 0, INFLOW_MAT_ID);
    set_material_if_local(0, last, OUTFLOW_MAT_ID);
    set_material_if_local(last, 0, OUTFLOW_MAT_ID);
    set_material_if_local(last, last, INFLOW_MAT_ID);
}

__global__ void updateKernel(const ElementStatic* __restrict__ elements,
                             const Material* __restrict__ materials,
                             const val_t* __restrict__ current,
                             val_t* __restrict__ next,
                             val_t* __restrict__ total_flux,
                             const size_t begin, const size_t end,
                             const int n_elems_root) {
    const size_t local_index = begin +
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_index >= end) {
        return;
    }

    const ElementStatic element = elements[local_index];
    const size_t local_row = local_index / static_cast<size_t>(n_elems_root);
    const size_t column = local_index % static_cast<size_t>(n_elems_root);
    const size_t state_index =
        (local_row + 1) * static_cast<size_t>(n_elems_root) + column;
    const val_t current_energy = current[state_index];
    const Material material = materials[element.material_idx];

    val_t flux = material.external_flow;
#pragma unroll
    for (idx_t connection = 0; connection < element.num_connections;
         ++connection) {
        const val_t neighbor_energy = current[element.connected_idx[connection]];
        // Keep the operation order of the reference implementation.  The
        // build disables CUDA FMA contraction to retain its numerical model.
        flux += (neighbor_energy - current_energy) * material.transfer_coeff *
                element.connected_flux[connection] * 0.25;
    }

    next[state_index] = current_energy + flux;
    total_flux[local_index] += fabs(flux);
}

class GpuSimulation {
public:
    GpuSimulation(World& world, MPI_Comm communicator, const int rank,
                   const int world_size)
        : world_(world), communicator_(communicator), rank_(rank),
          world_size_(world_size), n_elems_root_(world.n_elems_root),
          row_start_(world.row_start), local_rows_(world.local_rows) {
        int local_rank = 0;
        MPI_Comm node_communicator = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split_type(communicator_, MPI_COMM_TYPE_SHARED, rank_,
                                      MPI_INFO_NULL, &node_communicator));
        MPI_CHECK(MPI_Comm_rank(node_communicator, &local_rank));

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            std::fprintf(stderr, "MPI rank %d found no CUDA device\n", rank_);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        device_id_ = local_rank % device_count;
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaFree(nullptr)); // Force context creation before timing.

        MPI_CHECK(MPI_Comm_free(&node_communicator));

        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&state_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready_, cudaEventDisableTiming));

        if (local_rows_ == 0) {
            return;
        }

        const size_t local_element_count =
            static_cast<size_t>(local_rows_) * n_elems_root_;
        const size_t padded_count =
            static_cast<size_t>(local_rows_ + 2) * n_elems_root_;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_elements_),
                              local_element_count * sizeof(ElementStatic)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_current_),
                              padded_count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_),
                              padded_count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_total_flux_),
                              local_element_count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materials_),
                              world_.materials.size() * sizeof(Material)));

        CUDA_CHECK(cudaMemcpy(d_elements_, world_.elements_static.data(),
                              local_element_count * sizeof(ElementStatic),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_materials_, world_.materials.data(),
                              world_.materials.size() * sizeof(Material),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemsetAsync(d_current_, 0, padded_count * sizeof(val_t),
                                   compute_stream_));
        CUDA_CHECK(cudaMemsetAsync(d_next_, 0, padded_count * sizeof(val_t),
                                   compute_stream_));
        CUDA_CHECK(cudaMemsetAsync(d_total_flux_, 0,
                                   local_element_count * sizeof(val_t),
                                   compute_stream_));
        CUDA_CHECK(cudaEventRecord(state_ready_, compute_stream_));

        const size_t row_bytes =
            static_cast<size_t>(n_elems_root_) * sizeof(val_t);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send_top_), row_bytes,
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send_bottom_), row_bytes,
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recv_top_), row_bytes,
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recv_bottom_), row_bytes,
                                 cudaHostAllocPortable));
    }

    ~GpuSimulation() {
        if (compute_stream_ != nullptr) {
            cudaStreamSynchronize(compute_stream_);
        }
        if (send_top_ != nullptr) {
            cudaFreeHost(send_top_);
        }
        if (send_bottom_ != nullptr) {
            cudaFreeHost(send_bottom_);
        }
        if (recv_top_ != nullptr) {
            cudaFreeHost(recv_top_);
        }
        if (recv_bottom_ != nullptr) {
            cudaFreeHost(recv_bottom_);
        }
        if (d_elements_ != nullptr) {
            cudaFree(d_elements_);
        }
        if (d_current_ != nullptr) {
            cudaFree(d_current_);
        }
        if (d_next_ != nullptr) {
            cudaFree(d_next_);
        }
        if (d_total_flux_ != nullptr) {
            cudaFree(d_total_flux_);
        }
        if (d_materials_ != nullptr) {
            cudaFree(d_materials_);
        }
        if (state_ready_ != nullptr) {
            cudaEventDestroy(state_ready_);
        }
        if (halos_ready_ != nullptr) {
            cudaEventDestroy(halos_ready_);
        }
        if (compute_stream_ != nullptr) {
            cudaStreamDestroy(compute_stream_);
        }
        if (copy_stream_ != nullptr) {
            cudaStreamDestroy(copy_stream_);
        }
    }

    void run(const int n_iters) {
        if (local_rows_ == 0) {
            return;
        }

        const size_t row_width = static_cast<size_t>(n_elems_root_);
        const size_t local_element_count =
            static_cast<size_t>(local_rows_) * row_width;
        const int up = (row_start_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
        const int down = (row_start_ + local_rows_ < n_elems_root_)
                             ? rank_ + 1
                             : MPI_PROC_NULL;

        for (int iteration = 0; iteration < n_iters; ++iteration) {
            // Border copies use a separate stream so the GPU can update rows
            // that do not depend on MPI halos while messages are in flight.
            const size_t top_offset = row_width;
            const size_t bottom_offset =
                static_cast<size_t>(local_rows_) * row_width;
            const size_t row_bytes = row_width * sizeof(val_t);
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, state_ready_, 0));
            CUDA_CHECK(cudaMemcpyAsync(send_top_, d_current_ + top_offset,
                                       row_bytes, cudaMemcpyDeviceToHost,
                                       copy_stream_));
            CUDA_CHECK(cudaMemcpyAsync(send_bottom_, d_current_ + bottom_offset,
                                       row_bytes, cudaMemcpyDeviceToHost,
                                       copy_stream_));

            // Rows 1..local_rows-2 have no remote dependency.  They can be
            // processed concurrently with the MPI exchange.
            if (local_rows_ > 2) {
                launchKernel(row_width, local_element_count - row_width);
            }
            CUDA_CHECK(cudaStreamSynchronize(copy_stream_));

            MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                       MPI_REQUEST_NULL, MPI_REQUEST_NULL};
            MPI_CHECK(MPI_Irecv(recv_top_, static_cast<int>(row_width), MPI_DOUBLE,
                                up, 0, communicator_, &requests[0]));
            MPI_CHECK(MPI_Irecv(recv_bottom_, static_cast<int>(row_width), MPI_DOUBLE,
                                down, 1, communicator_, &requests[1]));
            MPI_CHECK(MPI_Isend(send_top_, static_cast<int>(row_width), MPI_DOUBLE,
                                up, 1, communicator_, &requests[2]));
            MPI_CHECK(MPI_Isend(send_bottom_, static_cast<int>(row_width), MPI_DOUBLE,
                                down, 0, communicator_, &requests[3]));
            MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));

            if (up != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_current_, recv_top_, row_bytes,
                                           cudaMemcpyHostToDevice, copy_stream_));
            }
            if (down != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_current_ + bottom_offset + row_width,
                                           recv_bottom_, row_bytes,
                                           cudaMemcpyHostToDevice, copy_stream_));
            }
            CUDA_CHECK(cudaEventRecord(halos_ready_, copy_stream_));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, halos_ready_, 0));

            if (local_rows_ == 1) {
                launchKernel(0, row_width);
            } else {
                launchKernel(0, row_width);
                launchKernel(local_element_count - row_width, local_element_count);
            }
            CUDA_CHECK(cudaEventRecord(state_ready_, compute_stream_));
            std::swap(d_current_, d_next_);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }

    void downloadResults() {
        if (local_rows_ == 0) {
            return;
        }
        const size_t row_width = static_cast<size_t>(n_elems_root_);
        const size_t local_element_count =
            static_cast<size_t>(local_rows_) * row_width;
        const size_t row_bytes = row_width * sizeof(val_t);
        std::vector<val_t> energies(local_element_count);
        std::vector<val_t> fluxes(local_element_count);

        CUDA_CHECK(cudaMemcpy2D(energies.data(), row_bytes, d_current_ + row_width,
                                row_bytes, row_bytes, local_rows_,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(fluxes.data(), d_total_flux_,
                              local_element_count * sizeof(val_t),
                              cudaMemcpyDeviceToHost));

#pragma omp parallel for schedule(static)
        for (int64_t index = 0;
             index < static_cast<int64_t>(local_element_count); ++index) {
            world_.elements_dynamic[index] =
                ElementDynamic{energies[index], fluxes[index]};
        }
    }

private:
    void launchKernel(const size_t begin, const size_t end) {
        if (begin >= end) {
            return;
        }
        const size_t count = end - begin;
        const unsigned int blocks = static_cast<unsigned int>(
            (count + CUDA_THREADS - 1) / CUDA_THREADS);
        updateKernel<<<blocks, CUDA_THREADS, 0, compute_stream_>>>(
            d_elements_, d_materials_, d_current_, d_next_, d_total_flux_,
            begin, end, n_elems_root_);
        CUDA_CHECK(cudaGetLastError());
    }

    World& world_;
    MPI_Comm communicator_;
    int rank_ = 0;
    int world_size_ = 1;
    int n_elems_root_ = 0;
    int row_start_ = 0;
    int local_rows_ = 0;
    int device_id_ = 0;

    ElementStatic* d_elements_ = nullptr;
    Material* d_materials_ = nullptr;
    val_t* d_current_ = nullptr;
    val_t* d_next_ = nullptr;
    val_t* d_total_flux_ = nullptr;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t state_ready_ = nullptr;
    cudaEvent_t halos_ready_ = nullptr;
    val_t* send_top_ = nullptr;
    val_t* send_bottom_ = nullptr;
    val_t* recv_top_ = nullptr;
    val_t* recv_bottom_ = nullptr;
};

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& element : elements) {
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

uint64_t computeHash(const std::vector<ElementDynamic>& elements,
                     const uint64_t global_offset) {
    uint64_t hash = 0;
    for (size_t index = 0; index < elements.size(); ++index) {
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits, &elements[index].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[index].total_flux, sizeof(flux_bits));
        const uint64_t global_index = global_offset + index;
        hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
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
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation did not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    omp_set_dynamic(0);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_requested = false;
    bool parse_ok = true;

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            n_elems_root = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-i") == 0 && argument + 1 < argc) {
            n_iters = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            print_requested = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[argument]);
                printUsage(argv[0]);
            }
            parse_ok = false;
        }
    }
    if (n_elems_root <= 0 || n_iters < 0) {
        if (rank == 0) {
            std::printf("Grid size must be positive and iterations must be non-negative\n");
        }
        parse_ok = false;
    }
    if (!parse_ok) {
        MPI_Finalize();
        return 1;
    }

    const int base_rows = n_elems_root / world_size;
    const int extra_rows = n_elems_root % world_size;
    const int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int row_start = rank * base_rows + std::min(rank, extra_rows);
    const size_t global_element_count =
        static_cast<size_t>(n_elems_root) * n_elems_root;
    const size_t local_element_count =
        static_cast<size_t>(local_rows) * n_elems_root;

    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root,
                    n_elems_root, global_element_count);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("MPI ranks: %d\n", world_size);
        std::printf("OpenMP threads/rank: %d\n", omp_get_max_threads());
        std::printf("CUDA: enabled on one local device per MPI rank\n");
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    if (rank == 0) {
        const size_t static_mem = global_element_count * sizeof(ElementStatic);
        const size_t dynamic_mem = global_element_count * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        std::printf("Building unstructured mesh...\n");
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    total_mem / (1024.0 * 1024.0),
                    static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, row_start, local_rows);
    GpuSimulation simulation(world, MPI_COMM_WORLD, rank, world_size);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        std::printf("Running simulation...\n");
    }
    const double start = MPI_Wtime();
    simulation.run(n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD));
    simulation.downloadResults();

    const uint64_t local_hash = computeHash(
        world.elements_dynamic,
        static_cast<uint64_t>(row_start) * n_elems_root);
    uint64_t global_hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR,
                         0, MPI_COMM_WORLD));

    const bool need_global_results = validate || print_requested;
    std::vector<int> receive_counts;
    std::vector<int> receive_displacements;
    std::vector<val_t> global_energies;
    std::vector<val_t> global_fluxes;
    if (rank == 0 && need_global_results) {
        receive_counts.resize(world_size);
        receive_displacements.resize(world_size);
        for (int receiver = 0; receiver < world_size; ++receiver) {
            const int receiver_rows = base_rows + (receiver < extra_rows ? 1 : 0);
            const int receiver_start =
                receiver * base_rows + std::min(receiver, extra_rows);
            receive_counts[receiver] = receiver_rows * n_elems_root;
            receive_displacements[receiver] = receiver_start * n_elems_root;
        }
        global_energies.resize(global_element_count);
        if (validate) {
            global_fluxes.resize(global_element_count);
        }
    }

    // Pack fields explicitly rather than relying on an ABI-dependent MPI
    // datatype for ElementDynamic.
    if (need_global_results) {
        std::vector<val_t> local_energies(local_element_count);
        std::vector<val_t> local_fluxes(validate ? local_element_count : 0);
#pragma omp parallel for schedule(static)
        for (int64_t index = 0;
             index < static_cast<int64_t>(local_element_count); ++index) {
            local_energies[index] = world.elements_dynamic[index].current_energy;
            if (validate) {
                local_fluxes[index] = world.elements_dynamic[index].total_flux;
            }
        }

        double dummy = 0.0;
        MPI_CHECK(MPI_Gatherv(local_element_count == 0 ? &dummy : local_energies.data(),
                              static_cast<int>(local_element_count), MPI_DOUBLE,
                              rank == 0 ? global_energies.data() : nullptr,
                              rank == 0 ? receive_counts.data() : nullptr,
                              rank == 0 ? receive_displacements.data() : nullptr,
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (validate) {
            MPI_CHECK(MPI_Gatherv(local_element_count == 0 ? &dummy : local_fluxes.data(),
                                  static_cast<int>(local_element_count), MPI_DOUBLE,
                                  rank == 0 ? global_fluxes.data() : nullptr,
                                  rank == 0 ? receive_counts.data() : nullptr,
                                  rank == 0 ? receive_displacements.data() : nullptr,
                                  MPI_DOUBLE, 0, MPI_COMM_WORLD));
        }
    }

    if (rank == 0) {
        const double duration_ms = elapsed_seconds * 1000.0;
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double safe_duration_seconds = std::max(elapsed_seconds, 1.0e-12);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elements_per_sec =
            (static_cast<double>(n_measured_iters) * global_element_count) /
            safe_duration_seconds / 1.0e9;
        const double gflops = giga_elements_per_sec * 22.0;

        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016lX\n\n",
                    static_cast<unsigned long>(global_hash));

        if (print_requested) {
            print_results(global_energies, "ElementEnergy");
        }
    }

    int valid = 1;
    if (rank == 0 && validate) {
        std::vector<ElementDynamic> global_dynamic(global_element_count);
#pragma omp parallel for schedule(static)
        for (int64_t index = 0;
             index < static_cast<int64_t>(global_element_count); ++index) {
            global_dynamic[index] =
                ElementDynamic{global_energies[index], global_fluxes[index]};
        }
        valid = validateResults(global_dynamic) ? 1 : 0;
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_Finalize();
    return valid == 1 ? 0 : 1;
}
