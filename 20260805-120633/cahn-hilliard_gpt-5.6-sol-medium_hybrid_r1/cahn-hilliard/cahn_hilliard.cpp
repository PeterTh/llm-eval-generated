#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error_));                               \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

struct Slab {
    size_t first_z;
    size_t nz;
};

static Slab decompose(const size_t global_nz, const int rank, const int ranks) {
    const size_t base = global_nz / static_cast<size_t>(ranks);
    const size_t extra = global_nz % static_cast<size_t>(ranks);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, extra), base + (r < extra ? 1u : 0u)};
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                        double* __restrict__ mu,
                                        const size_t nx, const size_t ny,
                                        const size_t z_begin, const size_t z_end,
                                        const double inv_dx2, const double inv_dy2,
                                        const double inv_dz2, const double gamma,
                                        const double e_AA, const double e_BB,
                                        const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z_begin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z > z_end) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const double cv = c[i];
    const double lap = (c[xp] + c[xm] - 2.0 * cv) * inv_dx2
                     + (c[yp] + c[ym] - 2.0 * cv) * inv_dy2
                     + (c[i + plane] + c[i - plane] - 2.0 * cv) * inv_dz2;
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                 - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void updateKernel(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             const size_t nx, const size_t ny,
                             const size_t z_begin, const size_t z_end,
                             const double scale, const double inv_dx2,
                             const double inv_dy2, const double inv_dz2) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z_begin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z > z_end) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const double center = mu[i];
    const double lap = (mu[xp] + mu[xm] - 2.0 * center) * inv_dx2
                     + (mu[yp] + mu[ym] - 2.0 * center) * inv_dy2
                     + (mu[i + plane] + mu[i - plane] - 2.0 * center) * inv_dz2;
    cnew[i] = cold[i] + scale * lap;
}

static void launchChemical(const double* c, double* mu, const size_t nx,
                           const size_t ny, const size_t first, const size_t last,
                           const double gamma, const double e_AA,
                           const double e_BB, const double e_AB,
                           cudaStream_t stream) {
    if (first > last) return;
    const dim3 block(32, 4, 1);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y, last - first + 1);
    chemicalPotentialKernel<<<grid, block, 0, stream>>>(
        c, mu, nx, ny, first, last, 1.0, 1.0, 1.0,
        gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

static void launchUpdate(double* cnew, const double* cold, const double* mu,
                         const size_t nx, const size_t ny, const size_t first,
                         const size_t last, const double scale,
                         cudaStream_t stream) {
    if (first > last) return;
    const dim3 block(32, 4, 1);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y, last - first + 1);
    updateKernel<<<grid, block, 0, stream>>>(cnew, cold, mu, nx, ny, first,
                                             last, scale, 1.0, 1.0, 1.0);
    CUDA_CHECK(cudaGetLastError());
}

// Pinned staging makes this work with every MPI implementation.  MPI traffic
// overlaps the GPU computation for all slab planes that do not need a halo.
class HaloExchange {
public:
    HaloExchange(const size_t plane, const int rank, const int ranks)
        : plane_(plane), rank_(rank), ranks_(ranks) {
        CUDA_CHECK(cudaHostAlloc(&storage_, 4 * plane_ * sizeof(double),
                                 cudaHostAllocPortable));
        send_low_ = storage_;
        send_high_ = storage_ + plane_;
        recv_low_ = storage_ + 2 * plane_;
        recv_high_ = storage_ + 3 * plane_;
    }
    ~HaloExchange() { cudaFreeHost(storage_); }

    void begin(double* field, const size_t local_nz, cudaStream_t stream) {
        const size_t bytes = plane_ * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(send_low_, field + plane_, bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(send_high_, field + local_nz * plane_, bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        count_ = 0;
        if (rank_ > 0) {
            MPI_Irecv(recv_low_, static_cast<int>(plane_), MPI_DOUBLE, rank_ - 1,
                      100, MPI_COMM_WORLD, &requests_[count_++]);
            MPI_Isend(send_low_, static_cast<int>(plane_), MPI_DOUBLE, rank_ - 1,
                      101, MPI_COMM_WORLD, &requests_[count_++]);
        } else {
            std::memcpy(recv_low_, send_low_, bytes);
        }
        if (rank_ + 1 < ranks_) {
            MPI_Irecv(recv_high_, static_cast<int>(plane_), MPI_DOUBLE, rank_ + 1,
                      101, MPI_COMM_WORLD, &requests_[count_++]);
            MPI_Isend(send_high_, static_cast<int>(plane_), MPI_DOUBLE, rank_ + 1,
                      100, MPI_COMM_WORLD, &requests_[count_++]);
        } else {
            std::memcpy(recv_high_, send_high_, bytes);
        }
    }

    void finish(double* field, const size_t local_nz, cudaStream_t stream) {
        if (count_) MPI_Waitall(count_, requests_, MPI_STATUSES_IGNORE);
        const size_t bytes = plane_ * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(field, recv_low_, bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(field + (local_nz + 1) * plane_, recv_high_,
                                   bytes, cudaMemcpyHostToDevice, stream));
    }

private:
    size_t plane_;
    int rank_, ranks_, count_ = 0;
    double *storage_ = nullptr, *send_low_ = nullptr, *send_high_ = nullptr;
    double *recv_low_ = nullptr, *recv_high_ = nullptr;
    MPI_Request requests_[4]{};
};

static void initializeConcentration(std::vector<double>& c, const size_t nx,
                                    const size_t ny, const size_t global_nz,
                                    const Slab slab) {
    const size_t plane = nx * ny;
    const size_t volume = plane * global_nz;
    #pragma omp parallel for schedule(static)
    for (long long local_z = 0; local_z < static_cast<long long>(slab.nz); ++local_z) {
        const size_t global_z = slab.first_z + static_cast<size_t>(local_z);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_i = static_cast<size_t>(local_z) * plane + y * nx + x;
                const size_t global_i = global_z * plane + y * nx + x;
                const double pseudo = (((global_i + 1) * 1299709) % volume)
                                    / static_cast<double>(volume);
                c[local_i] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResult(const std::vector<double>& c, const int rank) {
    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    #pragma omp parallel for reduction(|:local_bad) reduction(min:local_min) reduction(max:local_max)
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
        const double value = c[static_cast<size_t>(i)];
        local_bad |= !std::isfinite(value);
        local_min = std::min(local_min, value);
        local_max = std::max(local_max, value);
    }
    int bad = 0;
    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_bad, &bad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        if (bad) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (!bad && (global_max > 10.0 || global_min < -10.0))
            std::printf("Validation failed: values out of expected range\n");
    }
    return !bad && global_max <= 10.0 && global_min >= -10.0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  Grid size in X (default: 64)\n");
    std::printf("  -y <num>  Grid size in Y (default: same as X)\n");
    std::printf("  -z <num>  Grid size in Z (default: same as X)\n");
    std::printf("  -i <num>  Number of time steps (default: 20)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, args_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else args_ok = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || !args_ok) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return args_ok ? 0 : 1;
    }
    if (!nx || !ny || !nz || iterations < 0 || static_cast<size_t>(ranks) > nz
        || nx > static_cast<size_t>(std::numeric_limits<int>::max()) / ny) {
        if (rank == 0) std::fprintf(stderr, "Invalid grid/iteration count, or more MPI ranks than Z planes\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    const Slab slab = decompose(nz, rank, ranks);
    const size_t plane = nx * ny;
    const size_t local_cells = plane * slab.nz;
    const size_t allocated_cells = plane * (slab.nz + 2);
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "An XY plane exceeds the MPI count limit\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Parallel configuration: %d MPI rank(s), OpenMP host threads, CUDA GPUs\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> host(local_cells);
    initializeConcentration(host, nx, ny, nz, slab);
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, allocated_cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, allocated_cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, allocated_cells * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), local_cells * sizeof(double),
                          cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    HaloExchange halo(plane, rank, ranks);

    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0);
    constexpr double e_AB = 2.0 / 9.0, gamma = 0.5, dt_D = 0.01;
    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        halo.begin(cold, slab.nz, stream);
        launchChemical(cold, mu, nx, ny, 2, slab.nz > 1 ? slab.nz - 1 : 0,
                       gamma, e_AA, e_BB, e_AB, stream);
        halo.finish(cold, slab.nz, stream);
        launchChemical(cold, mu, nx, ny, 1, 1, gamma, e_AA, e_BB, e_AB, stream);
        if (slab.nz > 1)
            launchChemical(cold, mu, nx, ny, slab.nz, slab.nz,
                           gamma, e_AA, e_BB, e_AB, stream);

        halo.begin(mu, slab.nz, stream);
        launchUpdate(cnew, cold, mu, nx, ny, 2,
                     slab.nz > 1 ? slab.nz - 1 : 0, dt_D, stream);
        halo.finish(mu, slab.nz, stream);
        launchUpdate(cnew, cold, mu, nx, ny, 1, 1, dt_D, stream);
        if (slab.nz > 1)
            launchUpdate(cnew, cold, mu, nx, ny, slab.nz, slab.nz, dt_D, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double updates = static_cast<double>(nx) * ny * nz * iterations;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    elapsed > 0.0 ? updates / elapsed / 1.0e6 : 0.0);
    }

    CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_cells * sizeof(double),
                          cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> global;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            global.resize(nx * ny * nz);
            for (int r = 0; r < ranks; ++r) {
                const Slab s = decompose(nz, r, ranks);
                const size_t count = plane * s.nz;
                if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    std::fprintf(stderr, "Local result exceeds MPI_Gatherv count limit\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                counts[r] = static_cast<int>(count);
                displacements[r] = static_cast<int>(plane * s.first_z);
            }
        }
        MPI_Gatherv(host.data(), static_cast<int>(local_cells), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Concentration");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(host, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    MPI_Finalize();
    return valid ? 0 : 1;
}
