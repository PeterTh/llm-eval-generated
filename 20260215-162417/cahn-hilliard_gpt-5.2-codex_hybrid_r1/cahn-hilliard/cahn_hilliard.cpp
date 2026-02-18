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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

struct HaloBuffers {
    double* send_lower = nullptr;
    double* send_upper = nullptr;
    double* recv_lower = nullptr;
    double* recv_upper = nullptr;
};

static int getLocalRank() {
    const char* envs[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID"
    };
    for (const char* env : envs) {
        const char* val = std::getenv(env);
        if (val) {
            return std::atoi(val);
        }
    }
    return -1;
}

static void allocateHaloBuffers(HaloBuffers& buffers, const size_t plane) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaHostAlloc(&buffers.send_lower, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.send_upper, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.recv_lower, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&buffers.recv_upper, bytes, cudaHostAllocPortable));
}

static void freeHaloBuffers(HaloBuffers& buffers) {
    if (buffers.send_lower) {
        CUDA_CHECK(cudaFreeHost(buffers.send_lower));
    }
    if (buffers.send_upper) {
        CUDA_CHECK(cudaFreeHost(buffers.send_upper));
    }
    if (buffers.recv_lower) {
        CUDA_CHECK(cudaFreeHost(buffers.recv_lower));
    }
    if (buffers.recv_upper) {
        CUDA_CHECK(cudaFreeHost(buffers.recv_upper));
    }
    buffers = {};
}

static void computeLocalRange(const int rank, const int size, const size_t nz, size_t& z_start, size_t& local_nz) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    local_nz = base + (r < rem ? 1 : 0);
    z_start = r * base + (r < rem ? r : rem);
}

static void exchangeHalos(double* d_field, const size_t plane, const size_t local_nz,
                          const int rank, const int size, MPI_Comm comm, HaloBuffers& buffers) {
    if (local_nz == 0) {
        return;
    }

    const size_t bytes = plane * sizeof(double);
    const int count = static_cast<int>(plane);

    CUDA_CHECK(cudaMemcpy(buffers.send_lower, d_field + plane, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buffers.send_upper, d_field + plane * local_nz, bytes, cudaMemcpyDeviceToHost));

    const int prev = rank - 1;
    const int next = rank + 1;

    if (prev >= 0) {
        MPI_Sendrecv(buffers.send_lower, count, MPI_DOUBLE, prev, 100,
                     buffers.recv_lower, count, MPI_DOUBLE, prev, 101,
                     comm, MPI_STATUS_IGNORE);
    } else {
        std::memcpy(buffers.recv_lower, buffers.send_lower, bytes);
    }

    if (next < size) {
        MPI_Sendrecv(buffers.send_upper, count, MPI_DOUBLE, next, 101,
                     buffers.recv_upper, count, MPI_DOUBLE, next, 100,
                     comm, MPI_STATUS_IGNORE);
    } else {
        std::memcpy(buffers.recv_upper, buffers.send_upper, bytes);
    }

    CUDA_CHECK(cudaMemcpy(d_field, buffers.recv_lower, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field + plane * (local_nz + 1), buffers.recv_upper, bytes, cudaMemcpyHostToDevice));
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const int nx, const int ny, const int local_nz,
                                               const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= local_nz) {
        return;
    }

    const int lz = z + 1;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : y;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_xp = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(xp);
    const size_t idx_xn = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(xn);
    const size_t idx_yp = static_cast<size_t>(lz) * plane + static_cast<size_t>(yp) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_yn = static_cast<size_t>(lz) * plane + static_cast<size_t>(yn) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_zp = static_cast<size_t>(lz + 1) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_zn = static_cast<size_t>(lz - 1) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);

    const double center = c[idx];
    const double lap = (c[idx_xp] + c[idx_xn] - 2.0 * center) * inv_dx2
                     + (c[idx_yp] + c[idx_yn] - 2.0 * center) * inv_dy2
                     + (c[idx_zp] + c[idx_zn] - 2.0 * center) * inv_dz2;

    mu[idx] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
            + 3.0 * center + center * center * center - gamma * lap;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                             const int nx, const int ny, const int local_nz,
                             const double inv_dx2, const double inv_dy2, const double inv_dz2, const double coeff) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= local_nz) {
        return;
    }

    const int lz = z + 1;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : y;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_xp = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(xp);
    const size_t idx_xn = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(xn);
    const size_t idx_yp = static_cast<size_t>(lz) * plane + static_cast<size_t>(yp) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_yn = static_cast<size_t>(lz) * plane + static_cast<size_t>(yn) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_zp = static_cast<size_t>(lz + 1) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const size_t idx_zn = static_cast<size_t>(lz - 1) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);

    const double center = mu[idx];
    const double lap = (mu[idx_xp] + mu[idx_xn] - 2.0 * center) * inv_dx2
                     + (mu[idx_yp] + mu[idx_yn] - 2.0 * center) * inv_dy2
                     + (mu[idx_zp] + mu[idx_zn] - 2.0 * center) * inv_dz2;

    cnew[idx] = cold[idx] + coeff * lap;
}

static void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                                    const size_t local_nz, const size_t global_z_start) {
    const size_t plane = nx * ny;
    const size_t vol = plane * nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = global_z_start + z;
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                const size_t linear_id = global_z * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                           [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    int invalid = 0;
    double minVal = std::numeric_limits<double>::max();
    double maxVal = -std::numeric_limits<double>::max();

    #pragma omp parallel for reduction(|:invalid) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            invalid |= 1;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI implementation does not support MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool parse_ok = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_ok = false;
            break;
        }
    }

    if (!parse_ok) {
        MPI_Finalize();
        return 1;
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rank = getLocalRank();
    if (local_rank < 0) {
        local_rank = rank;
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

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
    const double coeff = dt * D;

    size_t local_z_start = 0;
    size_t local_nz = 0;
    computeLocalRange(rank, size, nz, local_z_start, local_nz);

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Plane size too large for MPI count.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t local_with_ghost = (local_nz + 2) * plane;
    const size_t bytes = local_with_ghost * sizeof(double);

    std::vector<double> h_cold(local_with_ghost, 0.0);
    if (local_nz > 0) {
        initializeConcentration(h_cold, nx, ny, nz, local_nz, local_z_start);
    }

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, h_cold.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cnew, 0, bytes));
    CUDA_CHECK(cudaMemset(d_mu, 0, bytes));

    HaloBuffers halos;
    if (local_nz > 0) {
        allocateHaloBuffers(halos, plane);
        exchangeHalos(d_cold, plane, local_nz, rank, size, MPI_COMM_WORLD, halos);
    }

    dim3 block(8, 8, 4);
    dim3 grid((static_cast<int>(nx) + block.x - 1) / block.x,
              (static_cast<int>(ny) + block.y - 1) / block.y,
              (static_cast<int>(local_nz) + block.z - 1) / block.z);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        if (local_nz > 0) {
            exchangeHalos(d_cold, plane, local_nz, rank, size, MPI_COMM_WORLD, halos);
            computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu,
                                                            static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(local_nz),
                                                            inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());

            exchangeHalos(d_mu, plane, local_nz, rank, size, MPI_COMM_WORLD, halos);
            updateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu,
                                          static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(local_nz),
                                          inv_dx2, inv_dy2, inv_dz2, coeff);
            CUDA_CHECK(cudaGetLastError());

            std::swap(d_cold, d_cnew);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    const double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double ms = max_time * 1000.0;
        const double cellUpdates = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz) * iterations;
        const double mcups = cellUpdates / max_time / 1e6;
        printf("Computation time: %.3f ms\n", ms);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exit_code = 0;
    if (printResults || validate) {
        std::vector<double> local_c(local_nz * plane);
        if (local_nz > 0) {
            CUDA_CHECK(cudaMemcpy(local_c.data(), d_cold + plane, local_nz * plane * sizeof(double), cudaMemcpyDeviceToHost));
        }

        std::vector<double> global_c;
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            global_c.resize(plane * nz);
            counts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                size_t r_start = 0;
                size_t r_nz = 0;
                computeLocalRange(r, size, nz, r_start, r_nz);
                counts[r] = static_cast<int>(r_nz * plane);
                displs[r] = static_cast<int>(offset);
                offset += r_nz * plane;
            }
        }

        MPI_Gatherv(local_c.data(), static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? global_c.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(global_c, "Concentration");
        }

        if (validate) {
            int valid_int = 1;
            if (rank == 0) {
                const bool valid = validateResult(global_c, nx, ny, nz);
                valid_int = valid ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!valid_int) {
                exit_code = 1;
            }
        }
    }

    if (local_nz > 0) {
        freeHaloBuffers(halos);
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return exit_code;
}
