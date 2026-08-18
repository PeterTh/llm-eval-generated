#include <cuda_runtime.h>

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kHaloTagLower = 4101;
constexpr int kHaloTagUpper = 4102;

inline size_t checked_product(const size_t a, const size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        std::fprintf(stderr, "Size overflow while calculating %s\n", what);
        std::abort();
    }
    return a * b;
}

inline size_t checked_sum(const size_t a, const size_t b, const char* what) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        std::fprintf(stderr, "Size overflow while calculating %s\n", what);
        std::abort();
    }
    return a + b;
}

[[noreturn]] void abort_all(const MPI_Comm communicator, const char* operation, const int error_code) {
    std::fprintf(stderr, "stencil3d: %s failed (error %d)\n", operation, error_code);
    MPI_Abort(communicator, error_code == 0 ? 1 : error_code);
    std::abort();
}

void mpi_check(const int error, const char* operation, const MPI_Comm communicator = MPI_COMM_WORLD) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]{};
        int message_length = 0;
        MPI_Error_string(error, message, &message_length);
        std::fprintf(stderr, "stencil3d: MPI operation '%s' failed: %.*s\n",
                     operation, message_length, message);
        MPI_Abort(communicator, error);
        std::abort();
    }
}

void cuda_check(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "stencil3d: CUDA operation '%s' failed: %s\n",
                     operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
        std::abort();
    }
}

#define MPI_CHECK(call) mpi_check((call), #call)
#define MPI_CHECK_COMM(call, communicator) mpi_check((call), #call, (communicator))
#define CUDA_CHECK(call) cuda_check((call), #call)

// The same row-major layout is used on the host and device. Making this
// helper device-callable avoids extra address arithmetic in the kernel.
__host__ __device__ __forceinline__ size_t idx3(const size_t x,
                                                 const size_t y,
                                                 const size_t z,
                                                 const size_t nx,
                                                 const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Each thread computes one x/y location and walks the assigned z planes. The
// z loop also removes the 65,535-plane limit of a CUDA 3D grid dimension.
__global__ void stencil_kernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx,
                               const size_t ny,
                               const size_t global_nz,
                               const size_t global_z_start,
                               const size_t local_z_begin,
                               const size_t local_z_end,
                               const size_t local_plane_stride) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) {
        return;
    }

    for (size_t local_z = local_z_begin + blockIdx.z;
         local_z <= local_z_end;
         local_z += gridDim.z) {
        const size_t global_z = global_z_start + local_z - 1;
        if (global_z == 0 || global_z >= global_nz - 1) {
            continue;
        }

        const size_t index = local_z * local_plane_stride + y * nx + x;
        const Real center = input[index];
        const Real left = input[index - 1];
        const Real right = input[index + 1];
        const Real front = input[index - nx];
        const Real back = input[index + nx];
        const Real bottom = input[index - local_plane_stride];
        const Real top = input[index + local_plane_stride];

        // Keep the original operation order and arithmetic semantics.
        output[index] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

struct Slab {
    size_t global_z_start = 0;
    size_t local_nz = 0;
};

Slab slab_for_rank(const int rank, const int active_ranks, const size_t global_nz) {
    const size_t ranks = static_cast<size_t>(active_ranks);
    const size_t rank_index = static_cast<size_t>(rank);
    const size_t base = global_nz / ranks;
    const size_t remainder = global_nz % ranks;

    if (rank_index < remainder) {
        return {rank_index * (base + 1), base + 1};
    }

    return {remainder * (base + 1) + (rank_index - remainder) * base, base};
}

void initialize_local_grid(std::vector<Real>& grid,
                           const size_t nx,
                           const size_t ny,
                           const size_t global_nz,
                           const Slab slab) {
    const size_t plane = checked_product(nx, ny, "grid plane");
    const size_t local_planes = slab.local_nz + 2;

    // OpenMP supplies the CPU-side parallel portion of the hybrid execution:
    // all ranks initialize their local slab and its two halo planes in parallel.
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t local_z = 0; local_z < local_planes; ++local_z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const long long global_z = static_cast<long long>(slab.global_z_start) +
                                           static_cast<long long>(local_z) - 1;
                const size_t local_index = idx3(x, y, local_z, nx, ny);
                if (global_z >= 0 && static_cast<size_t>(global_z) < global_nz) {
                    const size_t global_index = static_cast<size_t>(global_z) * plane + y * nx + x;
                    grid[local_index] = static_cast<Real>(global_index % 19);
                } else {
                    // These out-of-domain halo cells are never read by a
                    // boundary point, but initializing them keeps both device
                    // buffers fully defined for every MPI layout.
                    grid[local_index] = 0.0;
                }
            }
        }
    }
}

struct HaloBuffers {
    Real* send_lower = nullptr;
    Real* send_upper = nullptr;
    Real* receive_lower = nullptr;
    Real* receive_upper = nullptr;
};

void allocate_halo_buffers(HaloBuffers& buffers, const size_t bytes) {
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.send_lower), bytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.send_upper), bytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.receive_lower), bytes,
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&buffers.receive_upper), bytes,
                             cudaHostAllocPortable));
}

void free_halo_buffers(HaloBuffers& buffers) {
    if (buffers.send_lower != nullptr) {
        CUDA_CHECK(cudaFreeHost(buffers.send_lower));
    }
    if (buffers.send_upper != nullptr) {
        CUDA_CHECK(cudaFreeHost(buffers.send_upper));
    }
    if (buffers.receive_lower != nullptr) {
        CUDA_CHECK(cudaFreeHost(buffers.receive_lower));
    }
    if (buffers.receive_upper != nullptr) {
        CUDA_CHECK(cudaFreeHost(buffers.receive_upper));
    }
    buffers = {};
}

void launch_stencil(const Real* input,
                    Real* output,
                    const size_t nx,
                    const size_t ny,
                    const size_t global_nz,
                    const size_t global_z_start,
                    const size_t local_z_begin,
                    const size_t local_z_end,
                    const size_t local_plane_stride,
                    cudaStream_t stream) {
    if (local_z_begin > local_z_end) {
        return;
    }

    constexpr unsigned int block_x = 32;
    constexpr unsigned int block_y = 8;
    const size_t z_planes = local_z_end - local_z_begin + 1;
    constexpr size_t max_cuda_grid_z = 65535;
    const unsigned int grid_z = static_cast<unsigned int>(
        std::min<size_t>(z_planes, max_cuda_grid_z));
    const dim3 block(block_x, block_y, 1);
    const dim3 grid(static_cast<unsigned int>((nx - 2 + block_x - 1) / block_x),
                    static_cast<unsigned int>((ny - 2 + block_y - 1) / block_y),
                    grid_z);

    stencil_kernel<<<grid, block, 0, stream>>>(input, output, nx, ny, global_nz,
                                                global_z_start, local_z_begin,
                                                local_z_end, local_plane_stride);
    CUDA_CHECK(cudaGetLastError());
}

void exchange_halos_and_compute(Real* input,
                                Real* output,
                                const size_t nx,
                                const size_t ny,
                                const size_t global_nz,
                                const Slab slab,
                                const int active_rank,
                                const int active_ranks,
                                const MPI_Comm active_comm,
                                const int mpi_plane_count,
                                const size_t plane_bytes,
                                HaloBuffers& halos,
                                cudaStream_t compute_stream,
                                cudaStream_t communication_stream,
                                cudaEvent_t halos_ready) {
    // Interior planes do not depend on neighboring ranks. Start that work on
    // the compute stream while the boundary planes are copied and exchanged.
    if (slab.local_nz > 2) {
        launch_stencil(input, output, nx, ny, global_nz, slab.global_z_start,
                       2, slab.local_nz - 1, nx * ny, compute_stream);
    }

    const bool has_lower_neighbor = active_rank > 0;
    const bool has_upper_neighbor = active_rank + 1 < active_ranks;
    const bool needs_exchange = has_lower_neighbor || has_upper_neighbor;

    if (needs_exchange) {
        if (has_lower_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(halos.send_lower,
                                       input + nx * ny,
                                       plane_bytes,
                                       cudaMemcpyDeviceToHost,
                                       communication_stream));
        }
        if (has_upper_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(halos.send_upper,
                                       input + slab.local_nz * nx * ny,
                                       plane_bytes,
                                       cudaMemcpyDeviceToHost,
                                       communication_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(communication_stream));

        MPI_Request requests[4]{};
        int request_count = 0;
        if (has_lower_neighbor) {
            MPI_CHECK_COMM(MPI_Irecv(halos.receive_lower, mpi_plane_count, MPI_DOUBLE,
                                     active_rank - 1, kHaloTagUpper, active_comm,
                                     &requests[request_count++]), active_comm);
        }
        if (has_upper_neighbor) {
            MPI_CHECK_COMM(MPI_Irecv(halos.receive_upper, mpi_plane_count, MPI_DOUBLE,
                                     active_rank + 1, kHaloTagLower, active_comm,
                                     &requests[request_count++]), active_comm);
        }
        if (has_lower_neighbor) {
            MPI_CHECK_COMM(MPI_Isend(halos.send_lower, mpi_plane_count, MPI_DOUBLE,
                                     active_rank - 1, kHaloTagLower, active_comm,
                                     &requests[request_count++]), active_comm);
        }
        if (has_upper_neighbor) {
            MPI_CHECK_COMM(MPI_Isend(halos.send_upper, mpi_plane_count, MPI_DOUBLE,
                                     active_rank + 1, kHaloTagUpper, active_comm,
                                     &requests[request_count++]), active_comm);
        }
        MPI_CHECK_COMM(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE), active_comm);

        if (has_lower_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(input,
                                       halos.receive_lower,
                                       plane_bytes,
                                       cudaMemcpyHostToDevice,
                                       communication_stream));
        }
        if (has_upper_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(input + (slab.local_nz + 1) * nx * ny,
                                       halos.receive_upper,
                                       plane_bytes,
                                       cudaMemcpyHostToDevice,
                                       communication_stream));
        }
        CUDA_CHECK(cudaEventRecord(halos_ready, communication_stream));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, halos_ready, 0));
    }

    // For a one-rank run this is the complete computation. For a distributed
    // run this updates the two planes adjacent to the exchanged halos.
    launch_stencil(input, output, nx, ny, global_nz, slab.global_z_start,
                   1, slab.local_nz, nx * ny, compute_stream);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}

bool validate_result(const std::vector<Real>& grid) {
    if (grid.empty()) {
        std::printf("Validation failed: empty result\n");
        return false;
    }

    int invalid = 0;
#pragma omp parallel for reduction(| : invalid) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(grid.size()); ++i) {
        if (!std::isfinite(grid[static_cast<size_t>(i)])) {
            invalid = 1;
        }
    }
    if (invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    Real min_value = grid[0];
    Real max_value = grid[0];
#pragma omp parallel for reduction(min : min_value) reduction(max : max_value) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    std::printf("Value range: [%.6f, %.6f]\n", min_value, max_value);
    if (max_value > 1e6 || min_value < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool print_results = false;
    bool show_help = false;
    bool valid = true;
};

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.print_results = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.show_help = true;
        } else {
            options.valid = false;
        }
    }

    if (options.ny == 0) {
        options.ny = options.nx;
    }
    if (options.nz == 0) {
        options.nz = options.nx;
    }
    if (options.nx < 3 || options.ny < 3 || options.nz < 3 || options.iterations < 0) {
        options.valid = false;
    }
    return options;
}

std::vector<int> gather_counts(const int world_size,
                               const int active_ranks,
                               const size_t plane,
                               const size_t global_nz,
                               std::vector<int>& displacements) {
    std::vector<int> counts(static_cast<size_t>(world_size), 0);
    displacements.assign(static_cast<size_t>(world_size), 0);
    for (int rank = 0; rank < active_ranks; ++rank) {
        const Slab slab = slab_for_rank(rank, active_ranks, global_nz);
        const size_t count = checked_product(slab.local_nz, plane, "MPI gather count");
        const size_t displacement = checked_product(slab.global_z_start, plane,
                                                    "MPI gather displacement");
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "Grid is too large for MPI_Gatherv's 32-bit counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            std::abort();
        }
        counts[static_cast<size_t>(rank)] = static_cast<int>(count);
        displacements[static_cast<size_t>(rank)] = static_cast<int>(displacement);
    }
    return counts;
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        abort_all(MPI_COMM_WORLD, "MPI implementation does not provide MPI_THREAD_FUNNELED", 1);
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    const Options options = parse_options(argc, argv);
    if (options.show_help) {
        if (world_rank == 0) {
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (!options.valid) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Invalid command line or dimensions.\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int active_ranks = std::min(world_size, static_cast<int>(options.nz));
    const bool active = world_rank < active_ranks;
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED,
                             world_rank, &active_comm));

    int active_rank = -1;
    if (active) {
        MPI_CHECK_COMM(MPI_Comm_rank(active_comm, &active_rank), active_comm);
    }

    // Bind each process to a GPU using its node-local MPI rank. This remains
    // correct when a cluster launches multiple MPI ranks on every node.
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK_COMM(MPI_Comm_rank(local_comm, &local_rank), local_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        abort_all(MPI_COMM_WORLD, "no CUDA devices available", 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    if (world_rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Hybrid parallelism: MPI=%d ranks, OpenMP=%d threads/rank, CUDA=%d devices/node\n",
                    world_size, omp_get_max_threads(), device_count);
    }

    const size_t plane = checked_product(options.nx, options.ny, "grid plane");
    const size_t global_grid_size = checked_product(plane, options.nz, "global grid");

    std::vector<Real> local_final;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t communication_stream = nullptr;
    cudaEvent_t halos_ready = nullptr;
    HaloBuffers halos;
    Real* device_grid_a = nullptr;
    Real* device_grid_b = nullptr;

    if (active) {
        const Slab slab = slab_for_rank(active_rank, active_ranks, options.nz);
        const size_t local_planes_with_halos = checked_sum(slab.local_nz, 2,
                                                           "local halo planes");
        const size_t local_size = checked_product(local_planes_with_halos, plane,
                                                  "local grid");
        const size_t local_bytes = checked_product(local_size, sizeof(Real),
                                                   "local grid bytes");
        const size_t plane_bytes = checked_product(plane, sizeof(Real), "halo bytes");
        if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
            abort_all(MPI_COMM_WORLD, "grid plane exceeds MPI count limit", 1);
        }

        std::vector<Real> initial_grid(local_size);
        initialize_local_grid(initial_grid, options.nx, options.ny, options.nz, slab);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_grid_a), local_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_grid_b), local_bytes));
        CUDA_CHECK(cudaMemcpy(device_grid_a, initial_grid.data(), local_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_grid_b, initial_grid.data(), local_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
        allocate_halo_buffers(halos, plane_bytes);

        MPI_CHECK_COMM(MPI_Barrier(active_comm), active_comm);
        const double start = MPI_Wtime();

        Real* input = device_grid_a;
        Real* output = device_grid_b;
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            exchange_halos_and_compute(input, output, options.nx, options.ny, options.nz,
                                       slab, active_rank, active_ranks, active_comm,
                                       static_cast<int>(plane), plane_bytes, halos,
                                       compute_stream, communication_stream, halos_ready);
            std::swap(input, output);
        }

        MPI_CHECK_COMM(MPI_Barrier(active_comm), active_comm);
        const double elapsed = MPI_Wtime() - start;
        const long long milliseconds = static_cast<long long>(std::llround(elapsed * 1000.0));

        if (world_rank == 0) {
            std::printf("Computation time: %lld ms\n", milliseconds);
            const double cell_updates = static_cast<double>(options.nx - 2) *
                                         static_cast<double>(options.ny - 2) *
                                         static_cast<double>(options.nz - 2) *
                                         static_cast<double>(options.iterations);
            const double mcups = cell_updates / std::max(elapsed, 1.0e-12) / 1.0e6;
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        const Real* final_device_grid = (options.iterations % 2 == 0)
                                             ? device_grid_a
                                             : device_grid_b;
        local_final.resize(checked_product(slab.local_nz, plane, "final local grid"));
        CUDA_CHECK(cudaMemcpy(local_final.data(), final_device_grid + plane,
                              local_final.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    std::vector<Real> global_grid;
    std::vector<int> counts;
    std::vector<int> displacements;
    if (world_rank == 0) {
        global_grid.resize(global_grid_size);
        counts = gather_counts(world_size, active_ranks, plane, options.nz, displacements);
    }

    const int send_count = active ? static_cast<int>(local_final.size()) : 0;
    MPI_CHECK(MPI_Gatherv(active ? local_final.data() : nullptr, send_count, MPI_DOUBLE,
                          world_rank == 0 ? global_grid.data() : nullptr,
                          world_rank == 0 ? counts.data() : nullptr,
                          world_rank == 0 ? displacements.data() : nullptr,
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));

    bool valid = true;
    if (world_rank == 0) {
        if (options.print_results) {
            print_results(global_grid, "Grid");
        }
        if (options.validate) {
            std::printf("Validating result...\n");
            valid = validate_result(global_grid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    if (options.validate) {
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD));
    }

    if (active) {
        free_halo_buffers(halos);
        CUDA_CHECK(cudaEventDestroy(halos_ready));
        CUDA_CHECK(cudaStreamDestroy(communication_stream));
        CUDA_CHECK(cudaStreamDestroy(compute_stream));
        CUDA_CHECK(cudaFree(device_grid_b));
        CUDA_CHECK(cudaFree(device_grid_a));
        MPI_CHECK_COMM(MPI_Comm_free(&active_comm), MPI_COMM_WORLD);
    }
    MPI_CHECK_COMM(MPI_Comm_free(&local_comm), MPI_COMM_WORLD);
    MPI_CHECK(MPI_Finalize());
    return options.validate && !valid ? 1 : 0;
}
