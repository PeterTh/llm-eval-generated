#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int block_x = 32;
constexpr int block_y = 8;

struct Parameters {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool print_results = false;
};

struct HaloBuffers {
    double* send_low = nullptr;
    double* send_high = nullptr;
    double* recv_low = nullptr;
    double* recv_high = nullptr;
};

struct DeviceFields {
    double* concentration = nullptr;
    double* next_concentration = nullptr;
    double* chemical_potential = nullptr;
};

struct HaloExchange {
    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
};

void mpi_check(const int status, const char* expression, const int rank) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING] = {};
    int error_length = 0;
    MPI_Error_string(status, error, &error_length);
    std::fprintf(stderr, "MPI error on rank %d in %s: %.*s\n", rank, expression, error_length, error);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

void cuda_check(const cudaError_t status, const char* expression, const int rank) {
    if (status == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "CUDA error on rank %d in %s: %s\n", rank, expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define MPI_CHECK(call) mpi_check((call), #call, rank)
#define CUDA_CHECK(call) cuda_check((call), #call, rank)

bool multiply_overflows(const size_t a, const size_t b) {
    return a != 0 && b > std::numeric_limits<size_t>::max() / a;
}

bool parse_positive_size(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parse_nonnegative_int(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void print_usage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Each CUDA block owns one z plane.  The x/y stencil neighbourhood is cached
// in shared memory; z neighbours are supplied by the local slab or its halos.
__global__ void chemical_potential_kernel(const double* __restrict__ concentration,
                                          double* __restrict__ chemical_potential,
                                          const int nx, const int ny, const int z_begin,
                                          const int z_count, const double inv_dx2,
                                          const double inv_dy2, const double inv_dz2,
                                          const double gamma, const double e_aa,
                                          const double e_bb, const double e_ab) {
    __shared__ double tile[block_y + 2][block_x + 2];

    const int z = z_begin + static_cast<int>(blockIdx.z);
    const int block_origin_x = static_cast<int>(blockIdx.x) * block_x;
    const int block_origin_y = static_cast<int>(blockIdx.y) * block_y;
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t plane_offset = static_cast<size_t>(z) * plane;

    for (int sy = static_cast<int>(threadIdx.y); sy < block_y + 2; sy += block_y) {
        for (int sx = static_cast<int>(threadIdx.x); sx < block_x + 2; sx += block_x) {
            const int gx = max(0, min(block_origin_x + sx - 1, nx - 1));
            const int gy = max(0, min(block_origin_y + sy - 1, ny - 1));
            tile[sy][sx] = concentration[plane_offset + static_cast<size_t>(gy) * nx + gx];
        }
    }
    __syncthreads();

    const int x = block_origin_x + static_cast<int>(threadIdx.x);
    const int y = block_origin_y + static_cast<int>(threadIdx.y);
    if (x >= nx || y >= ny || static_cast<int>(blockIdx.z) >= z_count) {
        return;
    }

    const size_t index = plane_offset + static_cast<size_t>(y) * nx + x;
    const double value = tile[threadIdx.y + 1][threadIdx.x + 1];
    const double laplacian =
        (tile[threadIdx.y + 1][threadIdx.x + 2] + tile[threadIdx.y + 1][threadIdx.x] - 2.0 * value) * inv_dx2 +
        (tile[threadIdx.y + 2][threadIdx.x + 1] + tile[threadIdx.y][threadIdx.x + 1] - 2.0 * value) * inv_dy2 +
        (concentration[index + plane] + concentration[index - plane] - 2.0 * value) * inv_dz2;

    chemical_potential[index] =
        4.5 * ((value + 1.0) * e_aa + (value - 1.0) * e_bb - 2.0 * value * e_ab) +
        3.0 * value + value * value * value - gamma * laplacian;
}

__global__ void concentration_update_kernel(double* __restrict__ next_concentration,
                                            const double* __restrict__ concentration,
                                            const double* __restrict__ chemical_potential,
                                            const int nx, const int ny, const int z_begin,
                                            const int z_count, const double inv_dx2,
                                            const double inv_dy2, const double inv_dz2,
                                            const double diffusion_dt) {
    __shared__ double tile[block_y + 2][block_x + 2];

    const int z = z_begin + static_cast<int>(blockIdx.z);
    const int block_origin_x = static_cast<int>(blockIdx.x) * block_x;
    const int block_origin_y = static_cast<int>(blockIdx.y) * block_y;
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t plane_offset = static_cast<size_t>(z) * plane;

    for (int sy = static_cast<int>(threadIdx.y); sy < block_y + 2; sy += block_y) {
        for (int sx = static_cast<int>(threadIdx.x); sx < block_x + 2; sx += block_x) {
            const int gx = max(0, min(block_origin_x + sx - 1, nx - 1));
            const int gy = max(0, min(block_origin_y + sy - 1, ny - 1));
            tile[sy][sx] = chemical_potential[plane_offset + static_cast<size_t>(gy) * nx + gx];
        }
    }
    __syncthreads();

    const int x = block_origin_x + static_cast<int>(threadIdx.x);
    const int y = block_origin_y + static_cast<int>(threadIdx.y);
    if (x >= nx || y >= ny || static_cast<int>(blockIdx.z) >= z_count) {
        return;
    }

    const size_t index = plane_offset + static_cast<size_t>(y) * nx + x;
    const double value = tile[threadIdx.y + 1][threadIdx.x + 1];
    const double laplacian =
        (tile[threadIdx.y + 1][threadIdx.x + 2] + tile[threadIdx.y + 1][threadIdx.x] - 2.0 * value) * inv_dx2 +
        (tile[threadIdx.y + 2][threadIdx.x + 1] + tile[threadIdx.y][threadIdx.x + 1] - 2.0 * value) * inv_dy2 +
        (chemical_potential[index + plane] + chemical_potential[index - plane] - 2.0 * value) * inv_dz2;

    next_concentration[index] = concentration[index] + diffusion_dt * laplacian;
}

void launch_chemical_potential(const double* concentration, double* chemical_potential,
                               const int nx, const int ny, const int z_begin,
                               const int z_count, const double inv_dx2,
                               const double inv_dy2, const double inv_dz2,
                               const double gamma, const double e_aa,
                               const double e_bb, const double e_ab,
                               cudaStream_t compute_stream, const int rank) {
    if (z_count <= 0) {
        return;
    }
    const dim3 threads(block_x, block_y);
    const dim3 blocks((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, z_count);
    chemical_potential_kernel<<<blocks, threads, 0, compute_stream>>>(
        concentration, chemical_potential, nx, ny, z_begin, z_count,
        inv_dx2, inv_dy2, inv_dz2, gamma, e_aa, e_bb, e_ab);
    CUDA_CHECK(cudaGetLastError());
}

void launch_concentration_update(double* next_concentration, const double* concentration,
                                 const double* chemical_potential, const int nx,
                                 const int ny, const int z_begin, const int z_count,
                                 const double inv_dx2, const double inv_dy2,
                                 const double inv_dz2, const double diffusion_dt,
                                 cudaStream_t compute_stream, const int rank) {
    if (z_count <= 0) {
        return;
    }
    const dim3 threads(block_x, block_y);
    const dim3 blocks((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y, z_count);
    concentration_update_kernel<<<blocks, threads, 0, compute_stream>>>(
        next_concentration, concentration, chemical_potential, nx, ny, z_begin,
        z_count, inv_dx2, inv_dy2, inv_dz2, diffusion_dt);
    CUDA_CHECK(cudaGetLastError());
}

void initialize_local_concentration(std::vector<double>& concentration, const size_t nx,
                                    const size_t ny, const size_t global_z_offset,
                                    const size_t global_volume) {
    const size_t face_cells = nx * ny;
    const long long local_cells = static_cast<long long>(concentration.size());

#pragma omp parallel for schedule(static)
    for (long long local_index = 0; local_index < local_cells; ++local_index) {
        const size_t global_linear_id = global_z_offset * face_cells + static_cast<size_t>(local_index);
        const size_t pseudo = ((global_linear_id + 1) * static_cast<size_t>(1299709)) % global_volume;
        concentration[static_cast<size_t>(local_index)] = -1.0 + 2.0 *
            (static_cast<double>(pseudo) / static_cast<double>(global_volume));
    }
}

void copy_face(double* destination, const double* source, const size_t face_cells) {
    const long long count = static_cast<long long>(face_cells);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < count; ++i) {
        destination[i] = source[i];
    }
}

// Queue D2H transfers after the source-producing event.  The caller can run
// the interior CUDA stencil while this transfer and the MPI exchange progress.
void begin_halo_exchange(const double* field, const size_t face_cells, const size_t local_nz,
                         const HaloBuffers& halos, const cudaEvent_t source_ready,
                         const cudaStream_t transfer_stream, const int rank) {
    const size_t face_bytes = face_cells * sizeof(double);
    CUDA_CHECK(cudaStreamWaitEvent(transfer_stream, source_ready, 0));
    CUDA_CHECK(cudaMemcpyAsync(halos.send_low, field + face_cells, face_bytes,
                               cudaMemcpyDeviceToHost, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(halos.send_high, field + local_nz * face_cells, face_bytes,
                               cudaMemcpyDeviceToHost, transfer_stream));
}

void post_halo_exchange(HaloExchange& exchange, const HaloBuffers& halos, const size_t face_cells,
                        const int previous_rank, const int next_rank,
                        const cudaStream_t transfer_stream, const int rank) {
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

    if (previous_rank == MPI_PROC_NULL) {
        copy_face(halos.recv_low, halos.send_low, face_cells);
    }
    if (next_rank == MPI_PROC_NULL) {
        copy_face(halos.recv_high, halos.send_high, face_cells);
    }

    const int count = static_cast<int>(face_cells);
    MPI_CHECK(MPI_Irecv(halos.recv_low, count, MPI_DOUBLE, previous_rank, 1,
                        MPI_COMM_WORLD, &exchange.requests[0]));
    MPI_CHECK(MPI_Irecv(halos.recv_high, count, MPI_DOUBLE, next_rank, 0,
                        MPI_COMM_WORLD, &exchange.requests[1]));
    MPI_CHECK(MPI_Isend(halos.send_low, count, MPI_DOUBLE, previous_rank, 0,
                        MPI_COMM_WORLD, &exchange.requests[2]));
    MPI_CHECK(MPI_Isend(halos.send_high, count, MPI_DOUBLE, next_rank, 1,
                        MPI_COMM_WORLD, &exchange.requests[3]));
}

// Complete the exchange and make the two ghost planes visible to a CUDA stream.
void finish_halo_exchange(HaloExchange& exchange, double* field, const size_t face_cells,
                          const size_t local_nz, const HaloBuffers& halos,
                          const cudaStream_t transfer_stream, const cudaEvent_t halos_ready,
                          const int rank) {
    MPI_CHECK(MPI_Waitall(4, exchange.requests, MPI_STATUSES_IGNORE));
    const size_t face_bytes = face_cells * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(field, halos.recv_low, face_bytes, cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(field + (local_nz + 1) * face_cells, halos.recv_high, face_bytes,
                               cudaMemcpyHostToDevice, transfer_stream));
    CUDA_CHECK(cudaEventRecord(halos_ready, transfer_stream));
}

bool validate_global_field(const std::vector<double>& local_field, const int rank) {
    double local_minimum = std::numeric_limits<double>::infinity();
    double local_maximum = -std::numeric_limits<double>::infinity();
    int local_invalid = 0;
    const long long local_count = static_cast<long long>(local_field.size());

#pragma omp parallel for schedule(static) reduction(min : local_minimum) reduction(max : local_maximum) reduction(| : local_invalid)
    for (long long i = 0; i < local_count; ++i) {
        const double value = local_field[static_cast<size_t>(i)];
        local_minimum = std::min(local_minimum, value);
        local_maximum = std::max(local_maximum, value);
        local_invalid |= !std::isfinite(value);
    }

    double global_minimum = 0.0;
    double global_maximum = 0.0;
    int global_invalid = 0;
    MPI_CHECK(MPI_Reduce(&local_minimum, &global_minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD));

    if (rank != 0) {
        return true;
    }
    if (global_invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", global_minimum, global_maximum);
    if (global_maximum > 10.0 || global_minimum < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void release_resources(DeviceFields& fields, HaloBuffers& halos, cudaStream_t compute_stream,
                       cudaStream_t transfer_stream, cudaEvent_t concentration_ready,
                       cudaEvent_t concentration_halos_ready, cudaEvent_t potential_ready,
                       cudaEvent_t potential_halos_ready) {
    if (concentration_ready != nullptr) cudaEventDestroy(concentration_ready);
    if (concentration_halos_ready != nullptr) cudaEventDestroy(concentration_halos_ready);
    if (potential_ready != nullptr) cudaEventDestroy(potential_ready);
    if (potential_halos_ready != nullptr) cudaEventDestroy(potential_halos_ready);
    if (compute_stream != nullptr) cudaStreamDestroy(compute_stream);
    if (transfer_stream != nullptr) cudaStreamDestroy(transfer_stream);
    if (fields.concentration != nullptr) cudaFree(fields.concentration);
    if (fields.next_concentration != nullptr) cudaFree(fields.next_concentration);
    if (fields.chemical_potential != nullptr) cudaFree(fields.chemical_potential);
    if (halos.send_low != nullptr) cudaFreeHost(halos.send_low);
    if (halos.send_high != nullptr) cudaFreeHost(halos.send_high);
    if (halos.recv_low != nullptr) cudaFreeHost(halos.recv_low);
    if (halos.recv_high != nullptr) cudaFreeHost(halos.recv_high);
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    int rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    Parameters parameters;
    bool show_usage = false;
    bool parse_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parse_ok = parse_positive_size(argv[++i], parameters.nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parse_ok = parse_positive_size(argv[++i], parameters.ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parse_ok = parse_positive_size(argv[++i], parameters.nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parse_ok = parse_nonnegative_int(argv[++i], parameters.iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            parameters.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            parameters.print_results = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_usage = true;
        } else {
            parse_ok = false;
        }
        if (!parse_ok) {
            break;
        }
    }
    if (show_usage || !parse_ok) {
        if (rank == 0) {
            if (!parse_ok) std::fprintf(stderr, "Invalid command-line option or value\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return parse_ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (parameters.ny == 0) parameters.ny = parameters.nx;
    if (parameters.nz == 0) parameters.nz = parameters.nx;
    if (multiply_overflows(parameters.nx, parameters.ny) ||
        multiply_overflows(parameters.nx * parameters.ny, parameters.nz)) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions overflow the address space\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const size_t face_cells = parameters.nx * parameters.ny;
    const size_t global_cells = face_cells * parameters.nz;
    if (parameters.nx > INT_MAX || parameters.ny > INT_MAX || parameters.nz > INT_MAX ||
        face_cells > static_cast<size_t>(INT_MAX) ||
        world_size > static_cast<int>(parameters.nz)) {
        if (rank == 0) {
            std::fprintf(stderr, "A grid dimension or MPI halo face is too large, or there are more MPI ranks than z planes\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // One MPI rank is assigned to each visible GPU in its node.  Ranks beyond
    // the number of local GPUs share them round-robin, preserving a runnable
    // mapping for launcher configurations with multiple ranks per accelerator.
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA-capable device is visible\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    const size_t base_local_nz = parameters.nz / static_cast<size_t>(world_size);
    const size_t remainder = parameters.nz % static_cast<size_t>(world_size);
    const size_t local_nz = base_local_nz + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t global_z_offset = static_cast<size_t>(rank) * base_local_nz +
                                   std::min(static_cast<size_t>(rank), remainder);
    const size_t local_cells = local_nz * face_cells;
    const size_t halo_cells = (local_nz + 2) * face_cells;
    if (multiply_overflows(halo_cells, sizeof(double)) ||
        local_cells > static_cast<size_t>(LLONG_MAX) || face_cells > static_cast<size_t>(LLONG_MAX)) {
        if (rank == 0) std::fprintf(stderr, "Local grid is too large\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    const int previous_rank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next_rank = rank == world_size - 1 ? MPI_PROC_NULL : rank + 1;
    const double inv_dx2 = 1.0;
    const double inv_dy2 = 1.0;
    const double inv_dz2 = 1.0;
    const double diffusion_dt = 0.01;
    const double e_aa = -(2.0 / 9.0);
    const double e_bb = -(2.0 / 9.0);
    const double e_ab = 2.0 / 9.0;
    const double gamma = 0.5;

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", parameters.nx, parameters.ny, parameters.nz);
        std::printf("Time steps: %d\n", parameters.iterations);
        std::printf("Validation: %s\n", parameters.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, CUDA execution: enabled, OpenMP threads/rank: %d\n",
                    world_size, omp_get_max_threads());
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> host_local(local_cells);
    initialize_local_concentration(host_local, parameters.nx, parameters.ny, global_z_offset, global_cells);

    DeviceFields fields;
    HaloBuffers halos;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaEvent_t concentration_ready = nullptr;
    cudaEvent_t concentration_halos_ready = nullptr;
    cudaEvent_t potential_ready = nullptr;
    cudaEvent_t potential_halos_ready = nullptr;

    const size_t halo_bytes = halo_cells * sizeof(double);
    const size_t face_bytes = face_cells * sizeof(double);
    CUDA_CHECK(cudaMalloc(&fields.concentration, halo_bytes));
    CUDA_CHECK(cudaMalloc(&fields.next_concentration, halo_bytes));
    CUDA_CHECK(cudaMalloc(&fields.chemical_potential, halo_bytes));
    CUDA_CHECK(cudaHostAlloc(&halos.send_low, face_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&halos.send_high, face_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&halos.recv_low, face_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&halos.recv_high, face_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&concentration_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&concentration_halos_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&potential_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&potential_halos_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaFuncSetCacheConfig(chemical_potential_kernel, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(concentration_update_kernel, cudaFuncCachePreferL1));

    CUDA_CHECK(cudaMemcpyAsync(fields.concentration + face_cells, host_local.data(), local_cells * sizeof(double),
                               cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaEventRecord(concentration_ready, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(concentration_ready));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    const double start_time = MPI_Wtime();

    for (int timestep = 0; timestep < parameters.iterations; ++timestep) {
        HaloExchange concentration_exchange;
        begin_halo_exchange(fields.concentration, face_cells, local_nz, halos,
                            concentration_ready, transfer_stream, rank);

        // Planes that do not touch a z halo proceed while MPI exchanges c.
        launch_chemical_potential(fields.concentration, fields.chemical_potential,
                                  static_cast<int>(parameters.nx), static_cast<int>(parameters.ny),
                                  2, static_cast<int>(local_nz) - 2,
                                  inv_dx2, inv_dy2, inv_dz2, gamma, e_aa, e_bb, e_ab,
                                  compute_stream, rank);
        post_halo_exchange(concentration_exchange, halos, face_cells, previous_rank, next_rank,
                           transfer_stream, rank);
        finish_halo_exchange(concentration_exchange, fields.concentration, face_cells, local_nz,
                             halos, transfer_stream, concentration_halos_ready, rank);

        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, concentration_halos_ready, 0));
        launch_chemical_potential(fields.concentration, fields.chemical_potential,
                                  static_cast<int>(parameters.nx), static_cast<int>(parameters.ny),
                                  1, 1, inv_dx2, inv_dy2, inv_dz2, gamma, e_aa, e_bb, e_ab,
                                  compute_stream, rank);
        if (local_nz > 1) {
            launch_chemical_potential(fields.concentration, fields.chemical_potential,
                                      static_cast<int>(parameters.nx), static_cast<int>(parameters.ny),
                                      static_cast<int>(local_nz), 1,
                                      inv_dx2, inv_dy2, inv_dz2, gamma, e_aa, e_bb, e_ab,
                                      compute_stream, rank);
        }
        CUDA_CHECK(cudaEventRecord(potential_ready, compute_stream));

        HaloExchange potential_exchange;
        begin_halo_exchange(fields.chemical_potential, face_cells, local_nz, halos,
                            potential_ready, transfer_stream, rank);
        // The update's interior similarly overlaps the exchange of mu halos.
        launch_concentration_update(fields.next_concentration, fields.concentration,
                                    fields.chemical_potential, static_cast<int>(parameters.nx),
                                    static_cast<int>(parameters.ny), 2,
                                    static_cast<int>(local_nz) - 2,
                                    inv_dx2, inv_dy2, inv_dz2, diffusion_dt,
                                    compute_stream, rank);
        post_halo_exchange(potential_exchange, halos, face_cells, previous_rank, next_rank,
                           transfer_stream, rank);
        finish_halo_exchange(potential_exchange, fields.chemical_potential, face_cells, local_nz,
                             halos, transfer_stream, potential_halos_ready, rank);

        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, potential_halos_ready, 0));
        launch_concentration_update(fields.next_concentration, fields.concentration,
                                    fields.chemical_potential, static_cast<int>(parameters.nx),
                                    static_cast<int>(parameters.ny), 1, 1,
                                    inv_dx2, inv_dy2, inv_dz2, diffusion_dt,
                                    compute_stream, rank);
        if (local_nz > 1) {
            launch_concentration_update(fields.next_concentration, fields.concentration,
                                        fields.chemical_potential, static_cast<int>(parameters.nx),
                                        static_cast<int>(parameters.ny), static_cast<int>(local_nz), 1,
                                        inv_dx2, inv_dy2, inv_dz2, diffusion_dt,
                                        compute_stream, rank);
        }
        CUDA_CHECK(cudaEventRecord(concentration_ready, compute_stream));
        std::swap(fields.concentration, fields.next_concentration);
    }

    CUDA_CHECK(cudaEventSynchronize(concentration_ready));
    const double local_elapsed_seconds = MPI_Wtime() - start_time;
    double elapsed_seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&local_elapsed_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        const long elapsed_milliseconds = std::max(1L, static_cast<long>(std::llround(elapsed_seconds * 1000.0)));
        const double cell_updates = static_cast<double>(global_cells) * parameters.iterations;
        const double mcups = elapsed_seconds > 0.0 ? cell_updates / elapsed_seconds / 1.0e6 : 0.0;
        std::printf("Computation time: %ld ms\n", elapsed_milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    bool valid = true;
    if (parameters.validate || parameters.print_results) {
        CUDA_CHECK(cudaMemcpyAsync(host_local.data(), fields.concentration + face_cells,
                                   local_cells * sizeof(double), cudaMemcpyDeviceToHost, transfer_stream));
        CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
    }

    if (parameters.print_results) {
        if (global_cells > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                std::fprintf(stderr, "-r requires a global result that fits MPI_Gatherv's int displacement range\n");
            }
            release_resources(fields, halos, compute_stream, transfer_stream, concentration_ready,
                              concentration_halos_ready, potential_ready, potential_halos_ready);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        std::vector<double> global_field;
        std::vector<int> receive_counts;
        std::vector<int> receive_displacements;
        if (rank == 0) {
            global_field.resize(global_cells);
            receive_counts.resize(static_cast<size_t>(world_size));
            receive_displacements.resize(static_cast<size_t>(world_size));
            for (int process = 0; process < world_size; ++process) {
                const size_t process_nz = base_local_nz + (static_cast<size_t>(process) < remainder ? 1 : 0);
                const size_t process_offset = static_cast<size_t>(process) * base_local_nz +
                                              std::min(static_cast<size_t>(process), remainder);
                receive_counts[static_cast<size_t>(process)] = static_cast<int>(process_nz * face_cells);
                receive_displacements[static_cast<size_t>(process)] = static_cast<int>(process_offset * face_cells);
            }
        }
        MPI_CHECK(MPI_Gatherv(host_local.data(), static_cast<int>(local_cells), MPI_DOUBLE,
                              rank == 0 ? global_field.data() : nullptr,
                              rank == 0 ? receive_counts.data() : nullptr,
                              rank == 0 ? receive_displacements.data() : nullptr,
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            print_results(global_field, "Concentration");
        }
    }

    if (parameters.validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validate_global_field(host_local, rank);
        int valid_as_int = valid ? 1 : 0;
        MPI_CHECK(MPI_Bcast(&valid_as_int, 1, MPI_INT, 0, MPI_COMM_WORLD));
        valid = valid_as_int != 0;
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    release_resources(fields, halos, compute_stream, transfer_stream, concentration_ready,
                      concentration_halos_ready, potential_ready, potential_halos_ready);
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
