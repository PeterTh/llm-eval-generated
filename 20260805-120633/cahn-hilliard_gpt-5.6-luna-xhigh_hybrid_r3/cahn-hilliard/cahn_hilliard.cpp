#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

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

namespace {

// The domain is distributed in contiguous slabs in z.  Every local array has
// one extra plane on either side for the MPI halo exchange.
constexpr unsigned int BLOCK_X = 8;
constexpr unsigned int BLOCK_Y = 8;
constexpr unsigned int BLOCK_Z = 4;
constexpr unsigned int TILE_X = BLOCK_X + 2;
constexpr unsigned int TILE_Y = BLOCK_Y + 2;
constexpr unsigned int TILE_Z = BLOCK_Z + 2;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void mpiAbort(MPI_Comm comm, const char* message) {
    std::fprintf(stderr, "%s\n", message);
    MPI_Abort(comm, EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* expression,
               const char* file, const int line, MPI_Comm comm) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error at %s:%d (%s): %s",
                      file, line, expression, cudaGetErrorString(error));
        mpiAbort(comm, message);
    }
}

#define CUDA_CHECK(comm, expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__, (comm))

// Maps a z coordinate in the shared-memory tile to a local array slot.  The
// two halo planes are already clamped on the physical domain boundaries.
__device__ __forceinline__ size_t tileZToSlot(const int tile_z,
                                               const size_t block_z,
                                               const size_t local_nz,
                                               const size_t global_z_start,
                                               const size_t global_nz) {
    const int local_z = static_cast<int>(block_z) + tile_z - 1;
    if (local_z < 0) {
        return (global_z_start == 0) ? 1 : 0;
    }
    if (local_z >= static_cast<int>(local_nz)) {
        return (global_z_start + local_nz == global_nz) ? local_nz
                                                         : local_nz + 1;
    }
    return static_cast<size_t>(local_z) + 1;
}

template <bool ChemicalPotential>
__global__ void stencilKernel(const double* __restrict__ input,
                              double* __restrict__ output,
                              const double* __restrict__ old_concentration,
                              const size_t nx, const size_t ny,
                              const size_t local_nz,
                              const size_t global_z_start,
                              const size_t global_nz,
                              const double inv_dx2, const double inv_dy2,
                              const double inv_dz2, const double dt_D,
                              const double gamma, const double e_AA,
                              const double e_BB, const double e_AB) {
    __shared__ double tile[TILE_X * TILE_Y * TILE_Z];

    const unsigned int thread_id = threadIdx.z * blockDim.y * blockDim.x
                                 + threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
    const unsigned int tile_size = TILE_X * TILE_Y * TILE_Z;

    for (unsigned int linear = thread_id; linear < tile_size;
         linear += thread_count) {
        const unsigned int tile_x = linear % TILE_X;
        const unsigned int tile_y = (linear / TILE_X) % TILE_Y;
        const unsigned int tile_z = linear / (TILE_X * TILE_Y);

        const int requested_x = static_cast<int>(blockIdx.x * blockDim.x)
                              + static_cast<int>(tile_x) - 1;
        const int requested_y = static_cast<int>(blockIdx.y * blockDim.y)
                              + static_cast<int>(tile_y) - 1;
        const size_t global_x = requested_x < 0
            ? 0 : min(static_cast<size_t>(requested_x), nx - 1);
        const size_t global_y = requested_y < 0
            ? 0 : min(static_cast<size_t>(requested_y), ny - 1);
        const size_t slot_z = tileZToSlot(static_cast<int>(tile_z),
                                          static_cast<size_t>(blockIdx.z * blockDim.z),
                                          local_nz, global_z_start, global_nz);
        tile[(tile_z * TILE_Y + tile_y) * TILE_X + tile_x] =
            input[(slot_z * ny + global_y) * nx + global_x];
    }
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t y = static_cast<size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const size_t local_z = static_cast<size_t>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || local_z >= local_nz) {
        return;
    }

    const unsigned int sx = threadIdx.x + 1;
    const unsigned int sy = threadIdx.y + 1;
    const unsigned int sz = threadIdx.z + 1;
    const unsigned int center = (sz * TILE_Y + sy) * TILE_X + sx;

    const double cxx = (tile[center + 1] + tile[center - 1]
                        - 2.0 * tile[center]) * inv_dx2;
    const double cyy = (tile[center + TILE_X] + tile[center - TILE_X]
                        - 2.0 * tile[center]) * inv_dy2;
    const double czz = (tile[center + TILE_X * TILE_Y]
                        + tile[center - TILE_X * TILE_Y]
                        - 2.0 * tile[center]) * inv_dz2;
    const double laplacian = cxx + cyy + czz;
    const size_t index = ((local_z + 1) * ny + y) * nx + x;

    if constexpr (ChemicalPotential) {
        const double cv = tile[center];
        output[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                               - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - gamma * laplacian;
    } else {
        output[index] = old_concentration[index] + dt_D * laplacian;
    }
}

void launchChemicalPotential(const double* concentration, double* chemical_potential,
                             const size_t nx, const size_t ny, const size_t local_nz,
                             const size_t global_z_start, const size_t global_nz,
                             const double dx, const double dy, const double dz,
                             const double gamma, const double e_AA,
                             const double e_BB, const double e_AB,
                             cudaStream_t stream, MPI_Comm comm) {
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((local_nz + BLOCK_Z - 1) / BLOCK_Z));
    stencilKernel<true><<<grid, block, 0, stream>>>(
        concentration, chemical_potential, nullptr, nx, ny, local_nz,
        global_z_start, global_nz, 1.0 / (dx * dx), 1.0 / (dy * dy),
        1.0 / (dz * dz), 0.0, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(comm, cudaGetLastError());
}

void launchUpdate(const double* chemical_potential, double* concentration_new,
                  const double* concentration_old, const size_t nx,
                  const size_t ny, const size_t local_nz,
                  const size_t global_z_start, const size_t global_nz,
                  const double D, const double dt, const double dx,
                  const double dy, const double dz, cudaStream_t stream,
                  MPI_Comm comm) {
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((local_nz + BLOCK_Z - 1) / BLOCK_Z));
    stencilKernel<false><<<grid, block, 0, stream>>>(
        chemical_potential, concentration_new, concentration_old, nx, ny,
        local_nz, global_z_start, global_nz, 1.0 / (dx * dx),
        1.0 / (dy * dy), 1.0 / (dz * dz), dt * D, 0.0, 0.0, 0.0, 0.0);
    CUDA_CHECK(comm, cudaGetLastError());
}

struct HaloBuffers {
    double* send_lower = nullptr;
    double* send_upper = nullptr;
    double* receive_lower = nullptr;
    double* receive_upper = nullptr;

    void allocate(const size_t plane, MPI_Comm comm) {
        const size_t bytes = plane * sizeof(double);
        CUDA_CHECK(comm, cudaHostAlloc(&send_lower, bytes, cudaHostAllocPortable));
        CUDA_CHECK(comm, cudaHostAlloc(&send_upper, bytes, cudaHostAllocPortable));
        CUDA_CHECK(comm, cudaHostAlloc(&receive_lower, bytes, cudaHostAllocPortable));
        CUDA_CHECK(comm, cudaHostAlloc(&receive_upper, bytes, cudaHostAllocPortable));
    }

    void release(MPI_Comm comm) {
        if (send_lower != nullptr) CUDA_CHECK(comm, cudaFreeHost(send_lower));
        if (send_upper != nullptr) CUDA_CHECK(comm, cudaFreeHost(send_upper));
        if (receive_lower != nullptr) CUDA_CHECK(comm, cudaFreeHost(receive_lower));
        if (receive_upper != nullptr) CUDA_CHECK(comm, cudaFreeHost(receive_upper));
        send_lower = send_upper = receive_lower = receive_upper = nullptr;
    }
};

void exchangeHalos(double* concentration, const size_t nx, const size_t ny,
                   const size_t local_nz, const size_t plane,
                   const size_t global_z_start, const size_t global_nz,
                   const int active_rank, const int active_size,
                   HaloBuffers& halos, cudaStream_t stream, MPI_Comm comm) {
    const size_t bytes = plane * sizeof(double);
    const int lower = (active_rank > 0) ? active_rank - 1 : MPI_PROC_NULL;
    const int upper = (active_rank + 1 < active_size) ? active_rank + 1
                                                       : MPI_PROC_NULL;

    if (lower != MPI_PROC_NULL) {
        CUDA_CHECK(comm, cudaMemcpyAsync(halos.send_lower,
                                         concentration + plane, bytes,
                                         cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(comm, cudaMemcpyAsync(concentration,
                                         concentration + plane, bytes,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    if (upper != MPI_PROC_NULL) {
        CUDA_CHECK(comm, cudaMemcpyAsync(halos.send_upper,
                                         concentration + local_nz * plane, bytes,
                                         cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(comm, cudaMemcpyAsync(
            concentration + (local_nz + 1) * plane,
            concentration + local_nz * plane, bytes,
            cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(comm, cudaStreamSynchronize(stream));

    const int count = static_cast<int>(plane);
    if (lower != MPI_PROC_NULL) {
        MPI_Sendrecv(halos.send_lower, count, MPI_DOUBLE, lower, 200,
                     halos.receive_lower, count, MPI_DOUBLE, lower, 201, comm,
                     MPI_STATUS_IGNORE);
    }
    if (upper != MPI_PROC_NULL) {
        MPI_Sendrecv(halos.send_upper, count, MPI_DOUBLE, upper, 201,
                     halos.receive_upper, count, MPI_DOUBLE, upper, 200, comm,
                     MPI_STATUS_IGNORE);
    }

    if (lower != MPI_PROC_NULL) {
        CUDA_CHECK(comm, cudaMemcpyAsync(concentration, halos.receive_lower,
                                         bytes, cudaMemcpyHostToDevice, stream));
    }
    if (upper != MPI_PROC_NULL) {
        CUDA_CHECK(comm, cudaMemcpyAsync(
            concentration + (local_nz + 1) * plane, halos.receive_upper,
            bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(comm, cudaStreamSynchronize(stream));

    (void)nx;
    (void)ny;
    (void)global_z_start;
    (void)global_nz;
}

void initializeConcentration(std::vector<double>& c, const size_t nx,
                             const size_t ny, const size_t local_nz,
                             const size_t global_z_start,
                             const size_t global_nz) {
    const size_t plane = nx * ny;
    const size_t volume = plane * global_nz;

    // This is intentionally OpenMP-parallel: each MPI rank initializes only
    // its own slab, with the same global sequence as the original program.
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_linear_id = (global_z_start + z) * plane
                                              + y * nx + x;
                const size_t pseudo_integer =
                    ((global_linear_id + 1) * static_cast<size_t>(1299709)) % volume;
                c[idx3(x, y, z, nx, ny)] =
                    -1.0 + 2.0 * static_cast<double>(pseudo_integer)
                    / static_cast<double>(volume);
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    int has_nonfinite = 0;
#pragma omp parallel for reduction(|:has_nonfinite) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(c.size()); ++i) {
        has_nonfinite |= (std::isnan(c[static_cast<size_t>(i)])
                          || std::isinf(c[static_cast<size_t>(i)])) ? 1 : 0;
    }
    if (has_nonfinite != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
#pragma omp parallel for reduction(min:min_val) reduction(max:max_val) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(c.size()); ++i) {
        const double value = c[static_cast<size_t>(i)];
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", min_val, max_val);
    if (max_val > 10.0 || min_val < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parsePositiveSize(const char* text, size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0
        || parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseNonnegativeInt(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0
        || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int mpi_provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (mpi_provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return 1;
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool parse_ok = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parse_ok = parsePositiveSize(argv[++i], nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parse_ok = parsePositiveSize(argv[++i], ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parse_ok = parsePositiveSize(argv[++i], nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parse_ok = parseNonnegativeInt(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
        if (!parse_ok) {
            if (world_rank == 0) {
                std::printf("Invalid value for option %s\n", argv[i - 1]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    if (plane / nx != ny || gridSize / plane != nz
        || plane > static_cast<size_t>(std::numeric_limits<int>::max())
        || (gridSize / plane) != nz) {
        if (world_rank == 0) std::fprintf(stderr, "Grid dimensions are too large\n");
        MPI_Finalize();
        return 1;
    }

    // Determine the rank local to the node so each process selects a distinct
    // accelerator even when MPI uses global ranks across multiple nodes.
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);

    int device_count = 0;
    CUDA_CHECK(MPI_COMM_WORLD, cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        mpiAbort(MPI_COMM_WORLD, "No CUDA device is available for an MPI rank");
    }
    CUDA_CHECK(MPI_COMM_WORLD, cudaSetDevice(local_rank % device_count));

    const int active_size = std::min(world_size, static_cast<int>(nz));
    const bool active = world_rank < active_size;
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, world_rank,
                   &active_comm);

    int active_rank = -1;
    if (active) MPI_Comm_rank(active_comm, &active_rank);

    const size_t base_z = nz / static_cast<size_t>(active_size);
    const size_t remainder_z = nz % static_cast<size_t>(active_size);
    const size_t local_nz = active
        ? base_z + (static_cast<size_t>(active_rank) < remainder_z ? 1 : 0)
        : 0;
    const size_t global_z_start = active
        ? static_cast<size_t>(active_rank) * base_z
          + std::min(static_cast<size_t>(active_rank), remainder_z)
        : 0;

    if (world_rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
                    world_size, omp_get_max_threads(), device_count);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> host_owned(local_nz * plane);
    if (active) {
        initializeConcentration(host_owned, nx, ny, local_nz,
                                global_z_start, nz);
    }

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    cudaStream_t stream = nullptr;
    HaloBuffers halos;

    if (active) {
        const size_t local_with_halos = (local_nz + 2) * plane;
        const size_t bytes = local_with_halos * sizeof(double);
        CUDA_CHECK(MPI_COMM_WORLD, cudaMalloc(&d_cold, bytes));
        CUDA_CHECK(MPI_COMM_WORLD, cudaMalloc(&d_cnew, bytes));
        CUDA_CHECK(MPI_COMM_WORLD, cudaMalloc(&d_mu, bytes));
        CUDA_CHECK(MPI_COMM_WORLD, cudaStreamCreateWithFlags(&stream,
                                                              cudaStreamNonBlocking));
        halos.allocate(plane, MPI_COMM_WORLD);
        CUDA_CHECK(MPI_COMM_WORLD, cudaMemcpyAsync(
            d_cold + plane, host_owned.data(), host_owned.size() * sizeof(double),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(MPI_COMM_WORLD, cudaStreamSynchronize(stream));
    }

    // Physical parameters, unchanged from the original benchmark.
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    if (world_rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_seconds = 0.0;
    if (active) {
        MPI_Barrier(active_comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            exchangeHalos(d_cold, nx, ny, local_nz, plane, global_z_start, nz,
                          active_rank, active_size, halos, stream, MPI_COMM_WORLD);

            launchChemicalPotential(d_cold, d_mu, nx, ny, local_nz,
                                     global_z_start, nz, dx, dy, dz, gamma,
                                     e_AA, e_BB, e_AB, stream, MPI_COMM_WORLD);

            // The second stencil needs the chemical-potential halo, so it is
            // exchanged after the first kernel and before the update kernel.
            exchangeHalos(d_mu, nx, ny, local_nz, plane, global_z_start, nz,
                          active_rank, active_size, halos, stream, MPI_COMM_WORLD);
            launchUpdate(d_mu, d_cnew, d_cold, nx, ny, local_nz,
                         global_z_start, nz, D, dt, dx, dy, dz, stream,
                         MPI_COMM_WORLD);
            std::swap(d_cold, d_cnew);
        }
        CUDA_CHECK(MPI_COMM_WORLD, cudaStreamSynchronize(stream));
        elapsed_seconds = MPI_Wtime() - start;
    }

    double maximum_elapsed_seconds = 0.0;
    if (active) {
        MPI_Reduce(&elapsed_seconds, &maximum_elapsed_seconds, 1, MPI_DOUBLE,
                   MPI_MAX, 0, active_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (active) {
        CUDA_CHECK(MPI_COMM_WORLD, cudaMemcpy(
            host_owned.data(), d_cold + plane, host_owned.size() * sizeof(double),
            cudaMemcpyDeviceToHost));
    }

    std::vector<double> global_result;
    std::vector<int> gather_counts;
    std::vector<int> gather_displacements;
    if (world_rank == 0) {
        global_result.resize(gridSize);
        gather_counts.resize(static_cast<size_t>(world_size), 0);
        gather_displacements.resize(static_cast<size_t>(world_size), 0);
        for (int rank = 0; rank < active_size; ++rank) {
            const size_t rank_nz = base_z
                + (static_cast<size_t>(rank) < remainder_z ? 1 : 0);
            const size_t rank_start = static_cast<size_t>(rank) * base_z
                + std::min(static_cast<size_t>(rank), remainder_z);
            const size_t count = rank_nz * plane;
            if (count > static_cast<size_t>(std::numeric_limits<int>::max())
                || rank_start * plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
                mpiAbort(MPI_COMM_WORLD, "MPI gather dimensions exceed MPI int count range");
            }
            gather_counts[static_cast<size_t>(rank)] = static_cast<int>(count);
            gather_displacements[static_cast<size_t>(rank)] =
                static_cast<int>(rank_start * plane);
        }
    }

    const int local_count = static_cast<int>(host_owned.size());
    MPI_Gatherv(host_owned.data(), local_count, MPI_DOUBLE,
                world_rank == 0 ? global_result.data() : nullptr,
                world_rank == 0 ? gather_counts.data() : nullptr,
                world_rank == 0 ? gather_displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        const double safe_elapsed = std::max(maximum_elapsed_seconds, 1.0e-12);
        const double cell_updates = static_cast<double>(gridSize)
                                  * static_cast<double>(iterations);
        const double mcups = cell_updates / safe_elapsed / 1.0e6;
        std::printf("Computation time: %.3f ms\n", maximum_elapsed_seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(global_result, "Concentration");
        }

        int validation_result = 1;
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(global_result);
            if (valid) {
                std::printf("Validation: PASSED\n");
                validation_result = 0;
            } else {
                std::printf("Validation: FAILED\n");
            }
        } else {
            validation_result = 0;
        }
        MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (active) {
            halos.release(MPI_COMM_WORLD);
            CUDA_CHECK(MPI_COMM_WORLD, cudaFree(d_mu));
            CUDA_CHECK(MPI_COMM_WORLD, cudaFree(d_cnew));
            CUDA_CHECK(MPI_COMM_WORLD, cudaFree(d_cold));
            CUDA_CHECK(MPI_COMM_WORLD, cudaStreamDestroy(stream));
        }
        if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm);
        MPI_Comm_free(&local_comm);
        MPI_Finalize();
        return validation_result;
    }

    // Non-root ranks still participate in the validation result broadcast and
    // release their accelerator resources before finalizing MPI.
    int validation_result = 1;
    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (active) {
        halos.release(MPI_COMM_WORLD);
        CUDA_CHECK(MPI_COMM_WORLD, cudaFree(d_mu));
        CUDA_CHECK(MPI_COMM_WORLD, cudaFree(d_cnew));
        CUDA_CHECK(MPI_COMM_WORLD, cudaFree(d_cold));
        CUDA_CHECK(MPI_COMM_WORLD, cudaStreamDestroy(stream));
    }
    if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm);
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return validation_result;
}
