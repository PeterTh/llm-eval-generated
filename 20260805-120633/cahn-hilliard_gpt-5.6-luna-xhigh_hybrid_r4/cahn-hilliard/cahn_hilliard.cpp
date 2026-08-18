#include <algorithm>
#include <cmath>
#include <cstddef>
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

namespace {

// All MPI ranks participate in the program and every active rank owns a
// contiguous slab in z.  The two extra planes in each device field are the
// lower and upper z halos.
struct Domain {
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    size_t nx = 0;
    size_t ny = 0;
    size_t nz = 0;
    size_t plane = 0;
    size_t z_begin = 0;
    size_t local_nz = 0;

    int lower() const noexcept { return rank > 0 ? rank - 1 : MPI_PROC_NULL; }
    int upper() const noexcept { return rank + 1 < size ? rank + 1 : MPI_PROC_NULL; }
};

[[noreturn]] void abortWithMessage(const char* message, const int error_code = 1) {
    std::fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, error_code);
    std::abort();
}

void checkCuda(const cudaError_t result, const char* expression, const char* file, const int line) {
    if (result != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error at %s:%d (%s): %s", file, line,
                      expression, cudaGetErrorString(result));
        abortWithMessage(message);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(const size_t elements) { allocate(elements); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    void allocate(const size_t elements) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), elements * sizeof(double)));
    }

    double* data() noexcept { return data_; }

private:
    double* data_ = nullptr;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(const size_t elements) { allocate(elements); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void allocate(const size_t elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(double)));
    }

    double* data() noexcept { return data_; }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
    }

private:
    double* data_ = nullptr;
};

// The x and y boundaries are clamped exactly as in the original solver.  In
// z, the adjacent entries are always valid because the MPI/physical-boundary
// halo exchange has completed before each stencil is launched.
__device__ __forceinline__ double laplacian(const double* __restrict__ field, const size_t i,
                                            const size_t x, const size_t y, const size_t nx,
                                            const size_t ny, const size_t plane,
                                            const double inv_dx2, const double inv_dy2,
                                            const double inv_dz2) {
    const size_t x_plus = (x + 1 < nx) ? 1 : 0;
    const size_t x_minus = (x > 0) ? 1 : 0;
    const size_t y_plus = (y + 1 < ny) ? nx : 0;
    const size_t y_minus = (y > 0) ? nx : 0;
    const double center = field[i];

    const double two_center = 2.0 * center;
    const double cxx = (field[i + x_plus] + field[i - x_minus] - two_center) * inv_dx2;
    const double cyy = (field[i + y_plus] + field[i - y_minus] - two_center) * inv_dy2;
    const double czz = (field[i + plane] + field[i - plane] - two_center) * inv_dz2;
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ concentration,
                                                double* __restrict__ chemical_potential,
                                                const size_t nx, const size_t ny,
                                                const size_t plane, const size_t z_begin,
                                                const size_t z_end, const double inv_dx2,
                                                const double inv_dy2, const double inv_dz2,
                                                const double gamma, const double e_AA,
                                                const double e_BB, const double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    // A grid-stride loop in z also supports slabs larger than CUDA's 65,535
    // block limit while retaining coalesced x-major accesses.
    for (size_t z = z_begin + blockIdx.z; z < z_end; z += gridDim.z) {
        const size_t i = z * plane + y * nx + x;
        const double cv = concentration[i];
        const double lap = laplacian(concentration, i, x, y, nx, ny, plane, inv_dx2, inv_dy2,
                                     inv_dz2);

        chemical_potential[i] =
            4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
            3.0 * cv + cv * cv * cv - gamma * lap;
    }
}

__global__ void updateConcentrationKernel(double* __restrict__ concentration_new,
                                           const double* __restrict__ concentration_old,
                                           const double* __restrict__ chemical_potential,
                                           const size_t nx, const size_t ny, const size_t plane,
                                           const size_t z_begin, const size_t z_end,
                                           const double inv_dx2, const double inv_dy2,
                                           const double inv_dz2, const double D, const double dt) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    for (size_t z = z_begin + blockIdx.z; z < z_end; z += gridDim.z) {
        const size_t i = z * plane + y * nx + x;
        const double lap = laplacian(chemical_potential, i, x, y, nx, ny, plane, inv_dx2,
                                     inv_dy2, inv_dz2);
        concentration_new[i] = concentration_old[i] + dt * D * lap;
    }
}

dim3 stencilGrid(const size_t nx, const size_t ny, const size_t local_nz) {
    constexpr unsigned int block_x = 32;
    constexpr unsigned int block_y = 8;
    constexpr size_t max_grid_z = 65535;
    return dim3(static_cast<unsigned int>((nx + block_x - 1) / block_x),
                static_cast<unsigned int>((ny + block_y - 1) / block_y),
                static_cast<unsigned int>(std::min(local_nz, max_grid_z)));
}

void initializeConcentration(double* concentration, const size_t nx, const size_t plane,
                             const size_t local_nz, const size_t z_begin,
                             const size_t global_volume) {
    const size_t local_volume = plane * local_nz;

    // Initialization is independent for every cell and uses OpenMP so that
    // host-side setup scales with the number of MPI ranks/GPUs.
#pragma omp parallel for schedule(static)
    for (std::int64_t local_id = 0; local_id < static_cast<std::int64_t>(local_volume);
         ++local_id) {
        const size_t id = static_cast<size_t>(local_id);
        const size_t global_id = z_begin * plane + id;
        const size_t pseudo = (((global_id + 1) * static_cast<size_t>(1299709)) % global_volume);
        concentration[id] = -1.0 + 2.0 * (pseudo / static_cast<double>(global_volume));
    }
}

// Exchange a field's first and last owned planes.  OpenMPI installations are
// not required to be CUDA-aware, so communication uses pinned host staging;
// all numerical work remains on the GPU.  The copies are asynchronous with
// respect to the host and remain ordered with subsequent kernels on stream.
void exchangeHalos(double* const device_field, const Domain& domain,
                   PinnedBuffer& send_lower, PinnedBuffer& send_upper,
                   PinnedBuffer& receive_lower, PinnedBuffer& receive_upper,
                   cudaStream_t stream, const int tag) {
    const size_t bytes = domain.plane * sizeof(double);
    const int lower = domain.lower();
    const int upper = domain.upper();

    CUDA_CHECK(cudaMemcpyAsync(send_lower.data(), device_field + domain.plane, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(send_upper.data(),
                               device_field + domain.local_nz * domain.plane, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                               MPI_REQUEST_NULL};
    if (lower != MPI_PROC_NULL) {
        MPI_Irecv(receive_lower.data(), static_cast<int>(domain.plane), MPI_DOUBLE, lower, tag,
                  domain.communicator, &requests[0]);
        MPI_Isend(send_lower.data(), static_cast<int>(domain.plane), MPI_DOUBLE, lower, tag,
                  domain.communicator, &requests[2]);
    }
    if (upper != MPI_PROC_NULL) {
        MPI_Irecv(receive_upper.data(), static_cast<int>(domain.plane), MPI_DOUBLE, upper, tag,
                  domain.communicator, &requests[1]);
        MPI_Isend(send_upper.data(), static_cast<int>(domain.plane), MPI_DOUBLE, upper, tag,
                  domain.communicator, &requests[3]);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    if (lower != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(device_field, receive_lower.data(), bytes,
                                   cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(device_field, device_field + domain.plane, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (upper != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(device_field + (domain.local_nz + 1) * domain.plane,
                                   receive_upper.data(), bytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(device_field + (domain.local_nz + 1) * domain.plane,
                                   device_field + domain.local_nz * domain.plane, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<double>& concentration) {
    if (concentration.empty()) {
        std::printf("Validation failed: empty concentration field\n");
        return false;
    }

    int non_finite = 0;
#pragma omp parallel for reduction(| : non_finite) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(concentration.size()); ++i) {
        const double value = concentration[static_cast<size_t>(i)];
        if (std::isnan(value) || std::isinf(value)) {
            non_finite = 1;
        }
    }
    if (non_finite != 0) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double min_value = concentration.front();
    double max_value = concentration.front();
#pragma omp parallel for reduction(min : min_value) reduction(max : max_value) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(concentration.size()); ++i) {
        const double value = concentration[static_cast<size_t>(i)];
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
    if (max_value > 10.0 || min_value < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* prog_name) {
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

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool print_results_requested = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (nx == 0 || ny == 0 || nz == 0) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Grid dimensions must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t grid_size = plane * nz;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (world_rank == 0) {
            std::fprintf(stderr, "A halo plane is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Ranks beyond the number of z planes have no useful work.  They still
    // initialized MPI and cleanly leave after creating the active domain.
    const int active_size = static_cast<int>(std::min<size_t>(nz, static_cast<size_t>(world_size)));
    MPI_Comm domain_comm = MPI_COMM_NULL;
    const int color = world_rank < active_size ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &domain_comm);
    if (color == MPI_UNDEFINED) {
        MPI_Finalize();
        return 0;
    }

    Domain domain;
    domain.communicator = domain_comm;
    domain.nx = nx;
    domain.ny = ny;
    domain.nz = nz;
    domain.plane = plane;
    MPI_Comm_rank(domain_comm, &domain.rank);
    MPI_Comm_size(domain_comm, &domain.size);

    const size_t base_z = nz / static_cast<size_t>(domain.size);
    const size_t remainder_z = nz % static_cast<size_t>(domain.size);
    domain.local_nz = base_z + (static_cast<size_t>(domain.rank) < remainder_z ? 1 : 0);
    domain.z_begin = static_cast<size_t>(domain.rank) * base_z +
                     std::min(static_cast<size_t>(domain.rank), remainder_z);

    // One MPI rank maps to one visible GPU.  MPI_Comm_split_type makes this
    // use the local rank on multi-node launches instead of the global rank.
    MPI_Comm shared_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(domain_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);
    int local_rank = 0;
    MPI_Comm_rank(shared_comm, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        abortWithMessage("The hybrid benchmark requires at least one CUDA device per node");
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));
    MPI_Comm_free(&shared_comm);

    if (domain.rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", domain.size);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA device: %s\n", device_properties.name);
    }

    // Physical parameters.  Unit spacing is retained, but inverse squares
    // avoid repeated divisions in every CUDA thread.
    constexpr double inv_dx2 = 1.0;
    constexpr double inv_dy2 = 1.0;
    constexpr double inv_dz2 = 1.0;
    constexpr double dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double D = 1.0;

    const size_t local_volume = plane * domain.local_nz;
    const size_t padded_volume = plane * (domain.local_nz + 2);

    PinnedBuffer host_owned(local_volume);
    PinnedBuffer send_lower(plane);
    PinnedBuffer send_upper(plane);
    PinnedBuffer receive_lower(plane);
    PinnedBuffer receive_upper(plane);
    DeviceBuffer device_cold(padded_volume);
    DeviceBuffer device_new(padded_volume);
    DeviceBuffer device_mu(padded_volume);

    if (domain.rank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(host_owned.data(), nx, plane, domain.local_nz, domain.z_begin,
                            grid_size);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(device_cold.data() + plane, host_owned.data(),
                               local_volume * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (domain.rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(domain.communicator);
    const double start = MPI_Wtime();
    const dim3 block(32, 8, 1);
    const dim3 grid = stencilGrid(nx, ny, domain.local_nz);

    for (int t = 0; t < iterations; ++t) {
        // The old concentration must have current z halos before computing μ.
        exchangeHalos(device_cold.data(), domain, send_lower, send_upper, receive_lower,
                      receive_upper, stream, 100);
        computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
            device_cold.data(), device_mu.data(), nx, ny, plane, 1,
            domain.local_nz + 1, inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // The update stencil needs μ halos generated by neighboring ranks.
        exchangeHalos(device_mu.data(), domain, send_lower, send_upper, receive_lower,
                      receive_upper, stream, 200);
        updateConcentrationKernel<<<grid, block, 0, stream>>>(
            device_new.data(), device_cold.data(), device_mu.data(), nx, ny, plane, 1,
            domain.local_nz + 1, inv_dx2, inv_dy2, inv_dz2, D, dt);
        CUDA_CHECK(cudaGetLastError());
        device_cold.swap(device_new);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.communicator);

    CUDA_CHECK(cudaMemcpyAsync(host_owned.data(), device_cold.data() + plane,
                               local_volume * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> global_concentration;
    if (domain.rank == 0) {
        counts.resize(static_cast<size_t>(domain.size));
        displacements.resize(static_cast<size_t>(domain.size));
        size_t displacement = 0;
        for (int rank = 0; rank < domain.size; ++rank) {
            const size_t rank_local_nz = base_z + (static_cast<size_t>(rank) < remainder_z ? 1 : 0);
            const size_t rank_count = rank_local_nz * plane;
            if (rank_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
                abortWithMessage("The result is too large for MPI_Gatherv counts");
            }
            counts[static_cast<size_t>(rank)] = static_cast<int>(rank_count);
            displacements[static_cast<size_t>(rank)] = static_cast<int>(displacement);
            displacement += rank_count;
        }
        global_concentration.resize(grid_size);
    }

    MPI_Gatherv(host_owned.data(), static_cast<int>(local_volume), MPI_DOUBLE,
                domain.rank == 0 ? global_concentration.data() : nullptr,
                domain.rank == 0 ? counts.data() : nullptr,
                domain.rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                domain.communicator);

    if (domain.rank == 0) {
        const double cell_updates = static_cast<double>(grid_size) * iterations;
        const double mcups = elapsed > 0.0 ? cell_updates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (print_results_requested) {
            print_results(global_concentration, "Concentration");
        }

        int valid = 1;
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateResult(global_concentration) ? 1 : 0;
            std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, domain.communicator);
        CUDA_CHECK(cudaStreamDestroy(stream));
        MPI_Comm_free(&domain_comm);
        MPI_Finalize();
        return validate && valid == 0 ? 1 : 0;
    }

    int valid = 1;
    MPI_Bcast(&valid, 1, MPI_INT, 0, domain.communicator);
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Comm_free(&domain_comm);
    MPI_Finalize();
    return validate && valid == 0 ? 1 : 0;
}
