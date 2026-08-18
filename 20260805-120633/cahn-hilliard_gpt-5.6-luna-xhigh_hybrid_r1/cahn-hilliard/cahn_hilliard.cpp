#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 4;
constexpr int TILE_X = BLOCK_X + 2;
constexpr int TILE_Y = BLOCK_Y + 2;
constexpr int TILE_Z = BLOCK_Z + 2;

[[noreturn]] void mpiFailure(const MPI_Comm communicator, const char* message) {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::fprintf(stderr, "MPI rank %d: %s\n", rank, message);
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                    \
        if (error__ != cudaSuccess) {                                          \
            char message__[512];                                               \
            std::snprintf(message__, sizeof(message__),                        \
                          "CUDA error at %s:%d: %s", __FILE__, __LINE__,       \
                          cudaGetErrorString(error__));                        \
            mpiFailure(MPI_COMM_WORLD, message__);                              \
        }                                                                       \
    } while (false)

struct Options {
    std::size_t nx = 64;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool print_results = false;
    bool help = false;
    bool valid = true;
};

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

bool parseSize(const char* text, std::size_t& value) {
    if (text == nullptr || *text == '\0' || text[0] == '-') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return static_cast<unsigned long long>(value) == parsed;
}

bool parseInt(const char* text, int& value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

Options parseOptions(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.valid = parseSize(argv[++i], options.nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.valid = parseSize(argv[++i], options.ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.valid = parseSize(argv[++i], options.nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.valid = parseInt(argv[++i], options.iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.print_results = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            options.valid = false;
        }
        if (!options.valid) {
            break;
        }
    }
    if (options.ny == 0) {
        options.ny = options.nx;
    }
    if (options.nz == 0) {
        options.nz = options.nx;
    }
    return options;
}

// The host initialization is parallelized with OpenMP.  global_z makes the
// result independent of the MPI slab that owns a cell.
void initializeConcentration(std::vector<double>& field, const std::size_t nx,
                             const std::size_t ny, const std::size_t local_nz,
                             const std::size_t z_offset,
                             const std::size_t global_volume) {
    const std::size_t plane = nx * ny;

#pragma omp parallel for collapse(2) schedule(static)
    for (std::int64_t local_z = 0;
         local_z < static_cast<std::int64_t>(local_nz); ++local_z) {
        for (std::int64_t y = 0; y < static_cast<std::int64_t>(ny); ++y) {
#pragma omp simd
            for (std::int64_t x = 0; x < static_cast<std::int64_t>(nx); ++x) {
                const std::size_t global_z = z_offset + static_cast<std::size_t>(local_z);
                const std::size_t linear_id = global_z * plane +
                                              static_cast<std::size_t>(y) * nx +
                                              static_cast<std::size_t>(x);
                const std::size_t idx = static_cast<std::size_t>(local_z) * plane +
                                        static_cast<std::size_t>(y) * nx +
                                        static_cast<std::size_t>(x);
                const double pseudo = ((((linear_id + 1) * 1299709) % global_volume) /
                                       static_cast<double>(global_volume));
                field[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Load a shared-memory tile.  The z halo planes are already populated by MPI;
// the x and y boundaries are clamped directly while loading the tile.
__device__ inline void loadTile(const double* __restrict__ field, double* tile,
                                const std::size_t nx, const std::size_t ny,
                                const std::size_t local_nz,
                                const std::size_t plane) {
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const std::int64_t block_x = static_cast<std::int64_t>(blockIdx.x) * BLOCK_X;
    const std::int64_t block_y = static_cast<std::int64_t>(blockIdx.y) * BLOCK_Y;
    const std::int64_t block_z = static_cast<std::int64_t>(blockIdx.z) * BLOCK_Z;

    for (int tile_z = tz; tile_z < TILE_Z; tile_z += BLOCK_Z) {
        const std::int64_t raw_z = block_z + tile_z - 1;
        std::size_t source_z;
        if (raw_z < 0) {
            source_z = 0; // lower global boundary or lower MPI halo
        } else if (raw_z >= static_cast<std::int64_t>(local_nz)) {
            // raw_z == local_nz is the upper MPI halo.  Further-out tile
            // entries occur only in a partial final block and clamp inward.
            source_z = (raw_z == static_cast<std::int64_t>(local_nz))
                           ? local_nz + 1
                           : local_nz;
        } else {
            source_z = static_cast<std::size_t>(raw_z) + 1;
        }

        for (int tile_y = ty; tile_y < TILE_Y; tile_y += BLOCK_Y) {
            const std::int64_t raw_y = block_y + tile_y - 1;
            const std::size_t source_y = raw_y < 0
                                             ? 0
                                             : (raw_y >= static_cast<std::int64_t>(ny)
                                                    ? ny - 1
                                                    : static_cast<std::size_t>(raw_y));
            for (int tile_x = tx; tile_x < TILE_X; tile_x += BLOCK_X) {
                const std::int64_t raw_x = block_x + tile_x - 1;
                const std::size_t source_x = raw_x < 0
                                                 ? 0
                                                 : (raw_x >= static_cast<std::int64_t>(nx)
                                                        ? nx - 1
                                                        : static_cast<std::size_t>(raw_x));
                tile[(tile_z * TILE_Y + tile_y) * TILE_X + tile_x] =
                    field[source_z * plane + source_y * nx + source_x];
            }
        }
    }
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ concentration, double* __restrict__ chemical_potential,
    const std::size_t nx, const std::size_t ny, const std::size_t local_nz,
    const std::size_t plane, const double inv_dx2, const double inv_dy2,
    const double inv_dz2, const double gamma, const double e_AA,
    const double e_BB, const double e_AB) {
    extern __shared__ double tile[];
    loadTile(concentration, tile, nx, ny, local_nz, plane);
    __syncthreads();

    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const std::size_t local_z = static_cast<std::size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || local_z >= local_nz) {
        return;
    }

    const int center = ((static_cast<int>(threadIdx.z) + 1) * TILE_Y +
                        (static_cast<int>(threadIdx.y) + 1)) * TILE_X +
                       static_cast<int>(threadIdx.x) + 1;
    const double cv = tile[center];
    const double cxx = (tile[center + 1] + tile[center - 1] - 2.0 * cv) * inv_dx2;
    const double cyy = (tile[center + TILE_X] + tile[center - TILE_X] - 2.0 * cv) * inv_dy2;
    const double czz = (tile[center + TILE_X * TILE_Y] +
                        tile[center - TILE_X * TILE_Y] - 2.0 * cv) * inv_dz2;
    const double laplacian = cxx + cyy + czz;

    const std::size_t output = (local_z + 1) * plane + y * nx + x;
    chemical_potential[output] =
        4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
        3.0 * cv + cv * cv * cv - gamma * laplacian;
}

__global__ void updateConcentrationKernel(
    double* __restrict__ next, const double* __restrict__ current,
    const double* __restrict__ chemical_potential, const std::size_t nx,
    const std::size_t ny, const std::size_t local_nz, const std::size_t plane,
    const double inv_dx2, const double inv_dy2, const double inv_dz2,
    const double dt_times_D) {
    extern __shared__ double tile[];
    loadTile(chemical_potential, tile, nx, ny, local_nz, plane);
    __syncthreads();

    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const std::size_t local_z = static_cast<std::size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || local_z >= local_nz) {
        return;
    }

    const int center = ((static_cast<int>(threadIdx.z) + 1) * TILE_Y +
                        (static_cast<int>(threadIdx.y) + 1)) * TILE_X +
                       static_cast<int>(threadIdx.x) + 1;
    const double mu = tile[center];
    const double laplacian =
        (tile[center + 1] + tile[center - 1] - 2.0 * mu) * inv_dx2 +
        (tile[center + TILE_X] + tile[center - TILE_X] - 2.0 * mu) * inv_dy2 +
        (tile[center + TILE_X * TILE_Y] + tile[center - TILE_X * TILE_Y] - 2.0 * mu) * inv_dz2;

    const std::size_t output = (local_z + 1) * plane + y * nx + x;
    next[output] = current[output] + dt_times_D * laplacian;
}

void launchChemicalPotential(const double* concentration, double* chemical_potential,
                             const std::size_t nx, const std::size_t ny,
                             const std::size_t local_nz, const std::size_t plane,
                             const double inv_dx2, const double inv_dy2,
                             const double inv_dz2, const double gamma,
                             const double e_AA, const double e_BB, const double e_AB,
                             const cudaStream_t stream) {
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((local_nz + BLOCK_Z - 1) / BLOCK_Z));
    computeChemicalPotentialKernel<<<grid, block,
                                     TILE_X * TILE_Y * TILE_Z * sizeof(double), stream>>>(
        concentration, chemical_potential, nx, ny, local_nz, plane, inv_dx2, inv_dy2,
        inv_dz2, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdate(double* next, const double* current, const double* chemical_potential,
                  const std::size_t nx, const std::size_t ny, const std::size_t local_nz,
                  const std::size_t plane, const double inv_dx2, const double inv_dy2,
                  const double inv_dz2, const double dt_times_D,
                  const cudaStream_t stream) {
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((local_nz + BLOCK_Z - 1) / BLOCK_Z));
    updateConcentrationKernel<<<grid, block,
                                TILE_X * TILE_Y * TILE_Z * sizeof(double), stream>>>(
        next, current, chemical_potential, nx, ny, local_nz, plane, inv_dx2, inv_dy2,
        inv_dz2, dt_times_D);
    CUDA_CHECK(cudaGetLastError());
}

// MPI itself only receives ordinary host pointers on many clusters.  Pinned
// staging buffers make the device-to-host and host-to-device portions cheap,
// while keeping the code independent of CUDA-aware MPI extensions.
class HaloExchange {
  public:
    HaloExchange(const std::size_t plane, const MPI_Comm communicator,
                 const int rank, const int size)
        : plane_(plane), bytes_(plane * sizeof(double)), communicator_(communicator),
          rank_(rank), size_(size) {
        if (plane_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            mpiFailure(communicator_, "A halo plane is too large for MPI's int count API");
        }
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&send_lower_), bytes_));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&send_upper_), bytes_));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&receive_lower_), bytes_));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&receive_upper_), bytes_));
    }

    ~HaloExchange() {
        cudaFreeHost(send_lower_);
        cudaFreeHost(send_upper_);
        cudaFreeHost(receive_lower_);
        cudaFreeHost(receive_upper_);
    }

    void exchange(double* field, const std::size_t local_nz, const cudaStream_t stream) {
        if (rank_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(send_lower_, field + plane_, bytes_,
                                       cudaMemcpyDeviceToHost, stream));
        }
        if (rank_ + 1 < size_) {
            CUDA_CHECK(cudaMemcpyAsync(send_upper_, field + local_nz * plane_, bytes_,
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int count = static_cast<int>(plane_);
        MPI_Sendrecv(send_lower_, count, MPI_DOUBLE, rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL,
                     701, receive_upper_, count, MPI_DOUBLE,
                     rank_ + 1 < size_ ? rank_ + 1 : MPI_PROC_NULL, 701, communicator_,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_upper_, count, MPI_DOUBLE,
                     rank_ + 1 < size_ ? rank_ + 1 : MPI_PROC_NULL, 702, receive_lower_,
                     count, MPI_DOUBLE, rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL, 702,
                     communicator_,
                     MPI_STATUS_IGNORE);

        if (rank_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(field, receive_lower_, bytes_,
                                       cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(field, field + plane_, bytes_,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        if (rank_ + 1 < size_) {
            CUDA_CHECK(cudaMemcpyAsync(field + (local_nz + 1) * plane_, receive_upper_,
                                       bytes_, cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(field + (local_nz + 1) * plane_,
                                       field + local_nz * plane_, bytes_,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

  private:
    std::size_t plane_;
    std::size_t bytes_;
    MPI_Comm communicator_;
    int rank_;
    int size_;
    double* send_lower_ = nullptr;
    double* send_upper_ = nullptr;
    double* receive_lower_ = nullptr;
    double* receive_upper_ = nullptr;
};

struct LocalValidation {
    int invalid = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
};

LocalValidation validateLocal(const std::vector<double>& field) {
    LocalValidation result;
    int invalid = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

#pragma omp parallel for reduction(| : invalid) reduction(min : minimum) reduction(max : maximum) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(field.size()); ++i) {
        const double value = field[static_cast<std::size_t>(i)];
        invalid |= !std::isfinite(value);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    result.invalid = invalid;
    result.minimum = minimum;
    result.maximum = maximum;
    return result;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    (void)provided;

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const Options options = parseOptions(argc, argv);
    if (options.help || !options.valid) {
        if (world_rank == 0) {
            if (!options.valid) {
                std::printf("Unknown or invalid option.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return options.valid ? 0 : 1;
    }

    if (options.nx == 0 || options.ny == 0 || options.nz == 0) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Grid dimensions must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (options.nx > std::numeric_limits<unsigned int>::max() ||
        options.ny > std::numeric_limits<unsigned int>::max() ||
        options.nz > std::numeric_limits<unsigned int>::max()) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Grid dimensions are too large for CUDA launch geometry.\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (options.nx > std::numeric_limits<std::size_t>::max() / options.ny ||
        options.nx * options.ny > std::numeric_limits<std::size_t>::max() / options.nz) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Grid volume overflows size_t.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int active_size = std::min(world_size, static_cast<int>(options.nz));
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (active_comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(active_comm, &rank);
    MPI_Comm_size(active_comm, &size);

    MPI_Comm shared_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(active_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);
    int local_rank = 0;
    MPI_Comm_rank(shared_comm, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        mpiFailure(MPI_COMM_WORLD, "The hybrid benchmark requires at least one CUDA device");
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    CUDA_CHECK(cudaFree(nullptr));

    const std::size_t plane = options.nx * options.ny;
    const std::size_t global_volume = plane * options.nz;
    const std::size_t base_nz = options.nz / static_cast<std::size_t>(size);
    const std::size_t remainder = options.nz % static_cast<std::size_t>(size);
    const std::size_t local_nz = base_nz + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
    const std::size_t z_offset = static_cast<std::size_t>(rank) * base_nz +
                                 std::min(static_cast<std::size_t>(rank), remainder);
    const std::size_t local_cells = plane * local_nz;
    const std::size_t device_cells = plane * (local_nz + 2);
    if (local_cells > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        mpiFailure(active_comm, "A local slab is too large for MPI gather counts");
    }

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Time steps: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Hybrid parallelism: %d MPI ranks, %d OpenMP threads/rank, CUDA\n",
                     size, omp_get_max_threads());
    }

    std::vector<double> local_initial(local_cells);
    if (rank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(local_initial, options.nx, options.ny, local_nz, z_offset,
                            global_volume);
    if (rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }

    const std::size_t bytes = device_cells * sizeof(double);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), bytes));
    CUDA_CHECK(cudaMemcpy(d_cold + plane, local_initial.data(), local_cells * sizeof(double),
                          cudaMemcpyHostToDevice));
    local_initial.clear();
    local_initial.shrink_to_fit();

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    HaloExchange halos(plane, active_comm, rank, size);

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
    const double dt_times_D = dt * D;

    MPI_Barrier(active_comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < options.iterations; ++t) {
        halos.exchange(d_cold, local_nz, stream);
        launchChemicalPotential(d_cold, d_mu, options.nx, options.ny, local_nz, plane,
                                 inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB,
                                 stream);
        halos.exchange(d_mu, local_nz, stream);
        launchUpdate(d_cnew, d_cold, d_mu, options.nx, options.ny, local_nz, plane,
                     inv_dx2, inv_dy2, inv_dz2, dt_times_D, stream);
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(active_comm);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        const double cell_updates = static_cast<double>(global_volume) * options.iterations;
        const double mcups = milliseconds > 0.0 ? cell_updates / (milliseconds / 1000.0) / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> local_result;
    if (options.print_results || options.validate) {
        local_result.resize(local_cells);
        CUDA_CHECK(cudaMemcpy(local_result.data(), d_cold + plane,
                              local_cells * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::vector<double> global_result;
    std::vector<int> gather_counts;
    std::vector<int> gather_displacements;
    if (options.print_results) {
        if (rank == 0) {
            global_result.resize(global_volume);
            gather_counts.resize(static_cast<std::size_t>(size));
            gather_displacements.resize(static_cast<std::size_t>(size));
            for (int source = 0; source < size; ++source) {
                const std::size_t source_nz = base_nz +
                    (static_cast<std::size_t>(source) < remainder ? 1 : 0);
                const std::size_t source_offset = static_cast<std::size_t>(source) * base_nz +
                    std::min(static_cast<std::size_t>(source), remainder);
                gather_counts[static_cast<std::size_t>(source)] =
                    static_cast<int>(source_nz * plane);
                gather_displacements[static_cast<std::size_t>(source)] =
                    static_cast<int>(source_offset * plane);
            }
        }
        MPI_Gatherv(local_result.data(), static_cast<int>(local_cells), MPI_DOUBLE,
                    rank == 0 ? global_result.data() : nullptr,
                    rank == 0 ? gather_counts.data() : nullptr,
                    rank == 0 ? gather_displacements.data() : nullptr, MPI_DOUBLE, 0,
                    active_comm);
        if (rank == 0) {
            print_results(global_result, "Concentration");
        }
    }

    if (options.validate) {
        const LocalValidation local_validation = validateLocal(local_result);
        int invalid = 0;
        double minimum = 0.0;
        double maximum = 0.0;
        MPI_Reduce(&local_validation.invalid, &invalid, 1, MPI_INT, MPI_SUM, 0, active_comm);
        MPI_Reduce(&local_validation.minimum, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, active_comm);
        MPI_Reduce(&local_validation.maximum, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm);

        if (rank == 0) {
            std::printf("Validating result...\n");
            bool valid = true;
            if (invalid != 0) {
                std::printf("Validation failed: found NaN or Inf value\n");
                valid = false;
            } else {
                std::printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
                if (maximum > 10.0 || minimum < -10.0) {
                    std::printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        int root_valid = 1;
        if (rank == 0) {
            root_valid = (invalid == 0 && maximum <= 10.0 && minimum >= -10.0) ? 1 : 0;
        }
        MPI_Bcast(&root_valid, 1, MPI_INT, 0, active_comm);
        if (root_valid == 0) {
            CUDA_CHECK(cudaFree(d_cold));
            CUDA_CHECK(cudaFree(d_cnew));
            CUDA_CHECK(cudaFree(d_mu));
            CUDA_CHECK(cudaStreamDestroy(stream));
            MPI_Comm_free(&shared_comm);
            MPI_Comm_free(&active_comm);
            MPI_Finalize();
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Comm_free(&shared_comm);
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return 0;
}
