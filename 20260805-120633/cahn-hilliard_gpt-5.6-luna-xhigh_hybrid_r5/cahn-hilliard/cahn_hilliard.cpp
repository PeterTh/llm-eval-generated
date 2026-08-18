#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// The global array is split into contiguous slabs in Z.  Each slab has one
// ghost plane on either side, so both stencil applications are local after
// the two neighboring planes have been exchanged.
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y,
                                                 const size_t z, const size_t nx,
                                                 const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Slab {
    size_t z_start;
    size_t nz;
};

Slab slabForRank(const size_t global_nz, const int rank, const int ranks) noexcept {
    const size_t base = global_nz / static_cast<size_t>(ranks);
    const size_t remainder = global_nz % static_cast<size_t>(ranks);
    const size_t rank_as_size = static_cast<size_t>(rank);
    return {
        rank_as_size * base + std::min(rank_as_size, remainder),
        base + (rank_as_size < remainder ? 1 : 0)
    };
}

[[noreturn]] void mpiFailure(const int error, const char* expression,
                             const char* file, const int line) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int message_length = 0;
    MPI_Error_string(error, message, &message_length);
    std::fprintf(stderr, "MPI error at %s:%d (%s): %.*s\n",
                 file, line, expression, message_length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(expression) \
    do { \
        const int mpi_error__ = (expression); \
        if (mpi_error__ != MPI_SUCCESS) { \
            mpiFailure(mpi_error__, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cuda_error__ = (expression); \
        if (cuda_error__ != cudaSuccess) { \
            cudaFailure(cuda_error__, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

__device__ __forceinline__ double deviceLaplacian(
    const double* __restrict__ field,
    const size_t x, const size_t y, const size_t z,
    const size_t nx, const size_t ny,
    const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t center = idx3(x, y, z, nx, ny);
    const double center_value = field[center];
    const double cxx = (field[idx3(xp, y, z, nx, ny)] +
                        field[idx3(xn, y, z, nx, ny)] -
                        2.0 * center_value) * inv_dx2;
    const double cyy = (field[idx3(x, yp, z, nx, ny)] +
                        field[idx3(x, yn, z, nx, ny)] -
                        2.0 * center_value) * inv_dy2;
    const double czz = (field[idx3(x, y, z + 1, nx, ny)] +
                        field[idx3(x, y, z - 1, nx, ny)] -
                        2.0 * center_value) * inv_dz2;
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ concentration,
    double* __restrict__ chemical_potential,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double inv_dx2, const double inv_dy2, const double inv_dz2,
    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t local_z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || local_z >= local_nz) {
        return;
    }

    // z=0 and z=local_nz+1 are the MPI ghost planes.
    const size_t z = local_z + 1;
    const size_t index = idx3(x, y, z, nx, ny);
    const double cv = concentration[index];
    chemical_potential[index] =
        4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
        3.0 * cv + cv * cv * cv -
        gamma * deviceLaplacian(concentration, x, y, z, nx, ny,
                                inv_dx2, inv_dy2, inv_dz2);
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ concentration_new,
    const double* __restrict__ concentration_old,
    const double* __restrict__ chemical_potential,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double inv_dx2, const double inv_dy2, const double inv_dz2,
    const double D, const double dt) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t local_z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || local_z >= local_nz) {
        return;
    }

    const size_t z = local_z + 1;
    const size_t index = idx3(x, y, z, nx, ny);
    concentration_new[index] = concentration_old[index] + dt * D *
        deviceLaplacian(chemical_potential, x, y, z, nx, ny,
                        inv_dx2, inv_dy2, inv_dz2);
}

class DeviceAllocation {
public:
    explicit DeviceAllocation(const size_t elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(double)));
    }

    ~DeviceAllocation() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    double* data() const noexcept { return data_; }

private:
    double* data_ = nullptr;
};

class PinnedHaloBuffers {
public:
    explicit PinnedHaloBuffers(const size_t plane_elements) {
        const size_t bytes = plane_elements * sizeof(double);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send_lower_), bytes,
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send_upper_), bytes,
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&receive_lower_), bytes,
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&receive_upper_), bytes,
                                 cudaHostAllocPortable));
    }

    ~PinnedHaloBuffers() {
        if (send_lower_ != nullptr) cudaFreeHost(send_lower_);
        if (send_upper_ != nullptr) cudaFreeHost(send_upper_);
        if (receive_lower_ != nullptr) cudaFreeHost(receive_lower_);
        if (receive_upper_ != nullptr) cudaFreeHost(receive_upper_);
    }

    PinnedHaloBuffers(const PinnedHaloBuffers&) = delete;
    PinnedHaloBuffers& operator=(const PinnedHaloBuffers&) = delete;

    double* sendLower() const noexcept { return send_lower_; }
    double* sendUpper() const noexcept { return send_upper_; }
    double* receiveLower() const noexcept { return receive_lower_; }
    double* receiveUpper() const noexcept { return receive_upper_; }

private:
    double* send_lower_ = nullptr;
    double* send_upper_ = nullptr;
    double* receive_lower_ = nullptr;
    double* receive_upper_ = nullptr;
};

void exchangeHalos(double* const field, const size_t plane_elements,
                   const size_t local_nz, const int rank, const int ranks,
                   const int tag, const MPI_Comm communicator,
                   PinnedHaloBuffers& halos, const cudaStream_t stream) {
    const size_t bytes = plane_elements * sizeof(double);
    const bool has_lower_neighbor = rank > 0;
    const bool has_upper_neighbor = rank + 1 < ranks;

    if (has_lower_neighbor) {
        CUDA_CHECK(cudaMemcpyAsync(halos.sendLower(), field + plane_elements,
                                   bytes, cudaMemcpyDeviceToHost, stream));
    }
    if (has_upper_neighbor) {
        CUDA_CHECK(cudaMemcpyAsync(halos.sendUpper(),
                                   field + local_nz * plane_elements,
                                   bytes, cudaMemcpyDeviceToHost, stream));
    }
    if (has_lower_neighbor || has_upper_neighbor) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Request requests[4];
    int request_count = 0;
    if (has_lower_neighbor) {
        MPI_CHECK(MPI_Irecv(halos.receiveLower(), static_cast<int>(plane_elements),
                            MPI_DOUBLE, rank - 1, tag, communicator,
                            &requests[request_count++]));
        MPI_CHECK(MPI_Isend(halos.sendLower(), static_cast<int>(plane_elements),
                            MPI_DOUBLE, rank - 1, tag, communicator,
                            &requests[request_count++]));
    }
    if (has_upper_neighbor) {
        MPI_CHECK(MPI_Irecv(halos.receiveUpper(), static_cast<int>(plane_elements),
                            MPI_DOUBLE, rank + 1, tag, communicator,
                            &requests[request_count++]));
        MPI_CHECK(MPI_Isend(halos.sendUpper(), static_cast<int>(plane_elements),
                            MPI_DOUBLE, rank + 1, tag, communicator,
                            &requests[request_count++]));
    }
    if (request_count != 0) {
        MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
    }

    if (has_lower_neighbor) {
        CUDA_CHECK(cudaMemcpyAsync(field, halos.receiveLower(), bytes,
                                   cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(field, field + plane_elements, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (has_upper_neighbor) {
        CUDA_CHECK(cudaMemcpyAsync(field + (local_nz + 1) * plane_elements,
                                   halos.receiveUpper(), bytes,
                                   cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(field + (local_nz + 1) * plane_elements,
                                   field + local_nz * plane_elements, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Initialize only the owned portion of a slab.  The global linear index is
// retained so MPI decomposition produces exactly the same initial field as
// the original single-process implementation.
void initializeConcentration(std::vector<double>& concentration,
                             const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t z_start,
                             const size_t global_nz) {
    const size_t plane = nx * ny;
    const size_t volume = plane * global_nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (std::ptrdiff_t z = 0; z < static_cast<std::ptrdiff_t>(local_nz); ++z) {
        for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(ny); ++y) {
            for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(nx); ++x) {
                const size_t local_index = idx3(static_cast<size_t>(x),
                                                static_cast<size_t>(y),
                                                static_cast<size_t>(z), nx, ny);
                const size_t global_index =
                    (z_start + static_cast<size_t>(z)) * plane +
                    static_cast<size_t>(y) * nx + static_cast<size_t>(x);
                const double pseudo =
                    ((((global_index + 1) * 1299709) % volume) /
                     static_cast<double>(volume));
                concentration[local_index] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& concentration) {
    if (concentration.empty()) {
        std::printf("Validation failed: empty concentration field\n");
        return false;
    }

    double min_value = concentration[0];
    double max_value = concentration[0];
    int invalid = 0;

    #pragma omp parallel for reduction(min:min_value) reduction(max:max_value) reduction(|:invalid) schedule(static)
    for (std::ptrdiff_t i = 0;
         i < static_cast<std::ptrdiff_t>(concentration.size()); ++i) {
        const double value = concentration[static_cast<size_t>(i)];
        if (std::isnan(value) || std::isinf(value)) {
            invalid = 1;
        }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    if (invalid != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
    if (max_value > 10.0 || min_value < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

int mpiCount(const size_t elements) {
    if (elements > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "MPI slab exceeds the supported 32-bit MPI count\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(elements);
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

int main(int argc, char** argv) {
    int mpi_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_level));
    if (mpi_thread_level < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const size_t size_max = std::numeric_limits<size_t>::max();
    const bool dimensions_valid =
        nx != 0 && ny != 0 && nz != 0 && iterations >= 0 &&
        nx <= size_max / ny && nx * ny <= size_max / nz &&
        nx * ny <= static_cast<size_t>(INT_MAX) &&
        nx <= static_cast<size_t>(UINT_MAX) &&
        ny <= static_cast<size_t>(UINT_MAX) &&
        nz <= static_cast<size_t>(UINT_MAX);
    if (!dimensions_valid) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    // Ranks beyond the number of Z planes remain idle, while active ranks
    // retain a one-to-one, contiguous slab neighborhood.
    const int active_ranks = static_cast<int>(std::min(nz, static_cast<size_t>(world_size)));
    MPI_Comm active_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                             world_rank, &active_communicator));

    MPI_Comm shared_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                  MPI_INFO_NULL, &shared_communicator));

    if (world_rank >= active_ranks) {
        MPI_CHECK(MPI_Comm_free(&shared_communicator));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        MPI_CHECK(MPI_Finalize());
        return 0;
    }

    int rank = 0;
    int ranks = 1;
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(active_communicator, &rank));
    MPI_CHECK(MPI_Comm_size(active_communicator, &ranks));
    MPI_CHECK(MPI_Comm_rank(shared_communicator, &local_rank));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "MPI rank %d found no CUDA device\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, local_rank % device_count));
    omp_set_dynamic(0);

    const Slab slab = slabForRank(nz, rank, ranks);
    const size_t plane = nx * ny;
    const size_t global_size = plane * nz;
    const size_t local_size = plane * slab.nz;
    const size_t device_size = plane * (slab.nz + 2);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", world_size);
        std::printf("OpenMP threads/rank: %d\n", omp_get_max_threads());
        std::printf("CUDA device: %s\n", device_properties.name);
    }

    std::vector<double> local_concentration(local_size);
    std::printf("%s", rank == 0 ? "Initializing concentration field...\n" : "");
    initializeConcentration(local_concentration, nx, ny, slab.nz, slab.z_start, nz);

    DeviceAllocation concentration_old(device_size);
    DeviceAllocation concentration_new(device_size);
    DeviceAllocation chemical_potential(device_size);
    PinnedHaloBuffers halo_buffers(plane);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(concentration_old.data() + plane,
                               local_concentration.data(), local_size * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const dim3 block(8, 8, 4);
    const dim3 grid(
        static_cast<unsigned int>((nx + block.x - 1) / block.x),
        static_cast<unsigned int>((ny + block.y - 1) / block.y),
        static_cast<unsigned int>((slab.nz + block.z - 1) / block.z));
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(active_communicator));
    const double start = MPI_Wtime();

    double* current = concentration_old.data();
    double* next = concentration_new.data();
    if (iterations > 0) {
        exchangeHalos(current, plane, slab.nz, rank, ranks, 100,
                      active_communicator, halo_buffers, stream);
    }

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
            current, chemical_potential.data(), nx, ny, slab.nz,
            inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        exchangeHalos(chemical_potential.data(), plane, slab.nz, rank, ranks, 200,
                      active_communicator, halo_buffers, stream);

        cahnHilliardUpdateKernel<<<grid, block, 0, stream>>>(
            next, current, chemical_potential.data(), nx, ny, slab.nz,
            inv_dx2, inv_dy2, inv_dz2, D, dt);
        CUDA_CHECK(cudaGetLastError());

        std::swap(current, next);
        if (t + 1 < iterations) {
            exchangeHalos(current, plane, slab.nz, rank, ranks, 100,
                          active_communicator, halo_buffers, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double maximum_elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                         0, active_communicator));

    if (rank == 0) {
        const long long duration_ms = static_cast<long long>(std::llround(maximum_elapsed * 1000.0));
        std::printf("Computation time: %lld ms\n", duration_ms);
        const double cell_updates = static_cast<double>(global_size) * iterations;
        const double seconds = std::max(maximum_elapsed, std::numeric_limits<double>::min());
        std::printf("Performance: %.3f MCellUpdates/s\n",
                     cell_updates / seconds / 1e6);
    }

    int valid = 1;
    if (printResults || validate) {
        std::vector<double> local_result(local_size);
        CUDA_CHECK(cudaMemcpy(local_result.data(), current + plane,
                              local_size * sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> global_result;
        std::vector<int> receive_counts;
        std::vector<int> displacements;
        if (rank == 0) {
            global_result.resize(global_size);
            receive_counts.resize(static_cast<size_t>(ranks));
            displacements.resize(static_cast<size_t>(ranks));
            size_t displacement = 0;
            for (int r = 0; r < ranks; ++r) {
                const Slab other_slab = slabForRank(nz, r, ranks);
                receive_counts[static_cast<size_t>(r)] = mpiCount(plane * other_slab.nz);
                displacements[static_cast<size_t>(r)] = mpiCount(displacement);
                displacement += plane * other_slab.nz;
            }
        }
        MPI_CHECK(MPI_Gatherv(local_result.data(), mpiCount(local_size), MPI_DOUBLE,
                              rank == 0 ? global_result.data() : nullptr,
                              rank == 0 ? receive_counts.data() : nullptr,
                              rank == 0 ? displacements.data() : nullptr,
                              MPI_DOUBLE, 0, active_communicator));

        if (rank == 0) {
            if (printResults) {
                print_results(global_result, "Concentration");
            }
            if (validate) {
                std::printf("Validating result...\n");
                valid = validateResult(global_result) ? 1 : 0;
                std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, active_communicator));

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_CHECK(MPI_Comm_free(&shared_communicator));
    MPI_CHECK(MPI_Comm_free(&active_communicator));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return valid == 0 ? 1 : 0;
}
