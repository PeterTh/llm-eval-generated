#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define MPI_CHECK(call)                                                                                      \
    do {                                                                                                      \
        const int _err = (call);                                                                              \
        if (_err != MPI_SUCCESS) {                                                                            \
            char _es[MPI_MAX_ERROR_STRING];                                                                   \
            int _len = 0;                                                                                    \
            MPI_Error_string(_err, _es, &_len);                                                               \
            fprintf(stderr, "MPI error %s:%d: %.*s\n", __FILE__, __LINE__, _len, _es);                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                     \
        }                                                                                                     \
    } while (0)

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t _err = (call);                                                                     \
        if (_err != cudaSuccess) {                                                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err));        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                    \
        }                                                                                                    \
    } while (0)

__host__ __device__ __forceinline__ size_t idx3u(const size_t x, const size_t y, const size_t z, const size_t nx,
                                                 const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__global__ void initConcentrationKernel(double* __restrict__ c, const int nx, const int ny, const int local_nz,
                                        const long long z0, const long long nz_global, const unsigned long long vol_global) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z; // 0..local_nz-1 (owned)
    if (x >= nx || y >= ny || z >= local_nz) return;

    const long long gz = z0 + z;
    const unsigned long long linear_id = (unsigned long long)gz * (unsigned long long)(nx * ny) + (unsigned long long)y * (unsigned long long)nx +
                                         (unsigned long long)x;

    // Deterministic pseudo-random value in [-1, 1], matching the original formula.
    const double pseudo = (double)(((linear_id + 1ULL) * 1299709ULL) % vol_global) / (double)vol_global;
    const double val = -1.0 + 2.0 * pseudo;

    const size_t plane = (size_t)nx * (size_t)ny;
    c[(size_t)(z + 1) * plane + (size_t)y * (size_t)nx + (size_t)x] = val;

    (void)nz_global; // kept to make the clamped boundary semantics explicit
}

__global__ void computeMuKernel(const double* __restrict__ c, double* __restrict__ mu, const int nx, const int ny,
                                const int local_nz, const long long z0, const long long nz_global,
                                const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                const int z_begin, const int z_end) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = z_begin + (blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z > z_end) return;

    const size_t plane = (size_t)nx * (size_t)ny;
    const size_t idx = (size_t)z * plane + (size_t)y * (size_t)nx + (size_t)x;

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const long long gz = z0 + (long long)(z - 1);
    const int zn = (gz > 0) ? (z - 1) : z;
    const int zp = (gz + 1 < nz_global) ? (z + 1) : z;

    const double cv = c[idx];

    const double cxx = (c[(size_t)z * plane + (size_t)y * (size_t)nx + (size_t)xp] +
                        c[(size_t)z * plane + (size_t)y * (size_t)nx + (size_t)xn] - 2.0 * cv) * inv_dx2;
    const double cyy = (c[(size_t)z * plane + (size_t)yp * (size_t)nx + (size_t)x] +
                        c[(size_t)z * plane + (size_t)yn * (size_t)nx + (size_t)x] - 2.0 * cv) * inv_dy2;
    const double czz = (c[(size_t)zp * plane + (size_t)y * (size_t)nx + (size_t)x] +
                        c[(size_t)zn * plane + (size_t)y * (size_t)nx + (size_t)x] - 2.0 * cv) * inv_dz2;

    const double lap = cxx + cyy + czz;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;

    (void)local_nz;
}

__global__ void updateCKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu, const int nx,
                              const int ny, const long long z0, const long long nz_global, const double inv_dx2, const double inv_dy2,
                              const double inv_dz2, const double dtD, const int z_begin, const int z_end) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = z_begin + (blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z > z_end) return;

    const size_t plane = (size_t)nx * (size_t)ny;
    const size_t idx = (size_t)z * plane + (size_t)y * (size_t)nx + (size_t)x;

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const long long gz = z0 + (long long)(z - 1);
    const int zn = (gz > 0) ? (z - 1) : z;
    const int zp = (gz + 1 < nz_global) ? (z + 1) : z;

    const double muv = mu[idx];

    const double mxx = (mu[(size_t)z * plane + (size_t)y * (size_t)nx + (size_t)xp] +
                        mu[(size_t)z * plane + (size_t)y * (size_t)nx + (size_t)xn] - 2.0 * muv) * inv_dx2;
    const double myy = (mu[(size_t)z * plane + (size_t)yp * (size_t)nx + (size_t)x] +
                        mu[(size_t)z * plane + (size_t)yn * (size_t)nx + (size_t)x] - 2.0 * muv) * inv_dy2;
    const double mzz = (mu[(size_t)zp * plane + (size_t)y * (size_t)nx + (size_t)x] +
                        mu[(size_t)zn * plane + (size_t)y * (size_t)nx + (size_t)x] - 2.0 * muv) * inv_dz2;

    cnew[idx] = cold[idx] + dtD * (mxx + myy + mzz);
}

static void exchangeHaloPlanes(double* d_field, const size_t plane, const int local_nz, const long long z0,
                               const long long nz_global, const int rank, const int size,
                               double* h_send_lo, double* h_send_hi, double* h_recv_lo, double* h_recv_hi,
                               const int tag_up, const int tag_down, cudaStream_t stream) {
    if (local_nz <= 0) return;

    const int prev = (z0 == 0) ? MPI_PROC_NULL : (rank - 1);
    const int next = (z0 + (long long)local_nz == nz_global) ? MPI_PROC_NULL : (rank + 1);
    (void)size;

    const size_t bytes = plane * sizeof(double);

    // Pack boundary planes to pinned host buffers.
    if (prev != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_lo, d_field + 1 * plane, bytes, cudaMemcpyDeviceToHost, stream));
    }
    if (next != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(h_send_hi, d_field + (size_t)local_nz * plane, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int count = (plane > (size_t)std::numeric_limits<int>::max()) ? 0 : (int)plane;
    if (count == 0) {
        fprintf(stderr, "MPI halo plane too large for int count\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Exchange with prev: receive prev's last plane into our z=0 ghost; send our first plane down.
    MPI_CHECK(MPI_Sendrecv((prev == MPI_PROC_NULL) ? nullptr : h_send_lo, (prev == MPI_PROC_NULL) ? 0 : count, MPI_DOUBLE, prev, tag_down,
                           (prev == MPI_PROC_NULL) ? nullptr : h_recv_lo, (prev == MPI_PROC_NULL) ? 0 : count, MPI_DOUBLE, prev, tag_up,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    // Exchange with next: receive next's first plane into our z=local_nz+1 ghost; send our last plane up.
    MPI_CHECK(MPI_Sendrecv((next == MPI_PROC_NULL) ? nullptr : h_send_hi, (next == MPI_PROC_NULL) ? 0 : count, MPI_DOUBLE, next, tag_up,
                           (next == MPI_PROC_NULL) ? nullptr : h_recv_hi, (next == MPI_PROC_NULL) ? 0 : count, MPI_DOUBLE, next, tag_down,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    // Unpack to device ghost planes (or clamp at global z-edges).
    if (prev != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_field + 0 * plane, h_recv_lo, bytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_field + 0 * plane, d_field + 1 * plane, bytes, cudaMemcpyDeviceToDevice, stream));
    }

    if (next != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_field + (size_t)(local_nz + 1) * plane, h_recv_hi, bytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_field + (size_t)(local_nz + 1) * plane, d_field + (size_t)local_nz * plane, bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

static bool validateResult(const std::vector<double>& c) {
    if (c.empty()) return true;

    double minVal = c[0];
    double maxVal = c[0];
    int bad = 0;

#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) reduction(| : bad)
    for (size_t i = 0; i < c.size(); ++i) {
        const double v = c[i];
        if (std::isnan(v) || std::isinf(v)) bad |= 1;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }

    if (bad) {
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    // Bind each process to a GPU on the node (local-rank mapping).
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devCount));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const long long nz_i = (long long)nz;

    const unsigned long long vol_global = (unsigned long long)nx * (unsigned long long)ny * (unsigned long long)nz;
    const size_t plane = nx * ny;

    // Block decomposition in Z.
    const long long base = nz_i / (long long)size;
    const long long rem = nz_i % (long long)size;
    const long long local_nz_ll = base + ((long long)rank < rem ? 1LL : 0LL);
    const long long z0 = (long long)rank * base + std::min((long long)rank, rem);

    const int local_nz = (int)local_nz_ll;

    int omp_threads_used = 1;
#pragma omp parallel
    {
#pragma omp single
        omp_threads_used = omp_get_num_threads();
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("OpenMP threads used: %d\n", omp_threads_used);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters (kept identical to the original).
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
    const double dtD = dt * D;

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    cudaStream_t compute_stream{};
    cudaStream_t comm_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));

    if (local_nz > 0) {
        const size_t local_with_ghost = (size_t)(local_nz + 2) * plane;
        CUDA_CHECK(cudaMalloc(&d_cold, local_with_ghost * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, local_with_ghost * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, local_with_ghost * sizeof(double)));

        // Initialize owned planes on-GPU.
        const dim3 block(8, 8, 4);
        const dim3 grid((unsigned int)((nx + block.x - 1) / block.x), (unsigned int)((ny + block.y - 1) / block.y),
                        (unsigned int)((local_nz + block.z - 1) / block.z));
        initConcentrationKernel<<<grid, block, 0, compute_stream>>>(d_cold, (int)nx, (int)ny, local_nz, z0, nz_i, vol_global);
        CUDA_CHECK(cudaGetLastError());

        // Initialize ghost planes (clamp at global z-edges) so the first mu compute is well-defined.
        CUDA_CHECK(cudaMemcpyAsync(d_cold + 0 * plane, d_cold + 1 * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cold + (size_t)(local_nz + 1) * plane, d_cold + (size_t)local_nz * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice,
                                   compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_mu, d_cold, (size_t)(local_nz + 2) * plane * sizeof(double), cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cnew, d_cold, (size_t)(local_nz + 2) * plane * sizeof(double), cudaMemcpyDeviceToDevice, compute_stream));
    }

    // Pinned buffers for halo exchanges.
    double* h_send_lo = nullptr;
    double* h_send_hi = nullptr;
    double* h_recv_lo = nullptr;
    double* h_recv_hi = nullptr;
    if (local_nz > 0 && size > 1) {
        CUDA_CHECK(cudaHostAlloc(&h_send_lo, plane * sizeof(double), cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_send_hi, plane * sizeof(double), cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_recv_lo, plane * sizeof(double), cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&h_recv_hi, plane * sizeof(double), cudaHostAllocPortable));
    }

    // Warm up/ensure init is complete.
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t0 = MPI_Wtime();

    if (local_nz > 0) {
        const dim3 block(8, 8, 4);
        const dim3 grid_xy((unsigned int)((nx + block.x - 1) / block.x), (unsigned int)((ny + block.y - 1) / block.y), 1U);

        for (int it = 0; it < iterations; ++it) {
            // 1) Exchange c halos.
            if (size > 1) {
                exchangeHaloPlanes(d_cold, plane, local_nz, z0, nz_i, rank, size, h_send_lo, h_send_hi, h_recv_lo, h_recv_hi, 100, 101, comm_stream);
            } else {
                // Clamp (single-rank) via ghost copies.
                CUDA_CHECK(cudaMemcpyAsync(d_cold + 0 * plane, d_cold + 1 * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice, comm_stream));
                CUDA_CHECK(cudaMemcpyAsync(d_cold + (size_t)(local_nz + 1) * plane, d_cold + (size_t)local_nz * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice,
                                           comm_stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(comm_stream));

            // 2) mu = f'(c) - gamma * laplacian(c)
            {
                const int zb = 1;
                const int ze = local_nz;
                const unsigned int gz = (unsigned int)((ze - zb + block.z) / block.z);
                const dim3 grid(grid_xy.x, grid_xy.y, gz);
                computeMuKernel<<<grid, block, 0, compute_stream>>>(d_cold, d_mu, (int)nx, (int)ny, local_nz, z0, nz_i, inv_dx2, inv_dy2, inv_dz2, gamma, e_AA,
                                                                    e_BB, e_AB, zb, ze);
                CUDA_CHECK(cudaGetLastError());
            }

            // 3) Exchange mu halos.
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            if (size > 1) {
                exchangeHaloPlanes(d_mu, plane, local_nz, z0, nz_i, rank, size, h_send_lo, h_send_hi, h_recv_lo, h_recv_hi, 200, 201, comm_stream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_mu + 0 * plane, d_mu + 1 * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice, comm_stream));
                CUDA_CHECK(cudaMemcpyAsync(d_mu + (size_t)(local_nz + 1) * plane, d_mu + (size_t)local_nz * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice,
                                           comm_stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(comm_stream));

            // 4) c_{t+1} = c_t + dt*D*laplacian(mu)
            {
                const int zb = 1;
                const int ze = local_nz;
                const unsigned int gz = (unsigned int)((ze - zb + block.z) / block.z);
                const dim3 grid(grid_xy.x, grid_xy.y, gz);
                updateCKernel<<<grid, block, 0, compute_stream>>>(d_cnew, d_cold, d_mu, (int)nx, (int)ny, z0, nz_i, inv_dx2, inv_dy2, inv_dz2, dtD, zb, ze);
                CUDA_CHECK(cudaGetLastError());
            }

            // Swap buffers.
            std::swap(d_cold, d_cnew);
        }

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t1 = MPI_Wtime();

    const double dt_local = t1 - t0;
    double dt_max = 0.0;
    MPI_CHECK(MPI_Reduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        const double ms = dt_max * 1000.0;
        printf("Computation time: %.3f ms\n", ms);
        const double gridSize = (double)(nx * ny * nz);
        const double cellUpdates = gridSize * (double)iterations;
        const double mcups = cellUpdates / dt_max / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final field to rank 0 for -r and -v.
    std::vector<double> local_out;
    if (local_nz > 0) {
        local_out.resize((size_t)local_nz * plane);
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_cold + 1 * plane, (size_t)local_nz * plane * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> global_out;

    if (rank == 0) {
        counts.resize((size_t)size);
        displs.resize((size_t)size);
        long long disp = 0;
        for (int r = 0; r < size; ++r) {
            const long long lnz = base + ((long long)r < rem ? 1LL : 0LL);
            const long long ccount = lnz * (long long)plane;
            counts[(size_t)r] = (int)ccount;
            displs[(size_t)r] = (int)disp;
            disp += ccount;
        }
        global_out.resize((size_t)(nx * ny * nz));
    }

    const int sendcount = (int)local_out.size();
    MPI_CHECK(MPI_Gatherv(local_out.empty() ? nullptr : local_out.data(), sendcount, MPI_DOUBLE,
                          (rank == 0) ? global_out.data() : nullptr, (rank == 0) ? counts.data() : nullptr, (rank == 0) ? displs.data() : nullptr,
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        if (printResults) {
            print_results(global_out, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(global_out);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            if (!ok) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    if (h_send_lo) CUDA_CHECK(cudaFreeHost(h_send_lo));
    if (h_send_hi) CUDA_CHECK(cudaFreeHost(h_send_hi));
    if (h_recv_lo) CUDA_CHECK(cudaFreeHost(h_recv_lo));
    if (h_recv_hi) CUDA_CHECK(cudaFreeHost(h_recv_hi));

    if (d_cold) CUDA_CHECK(cudaFree(d_cold));
    if (d_cnew) CUDA_CHECK(cudaFree(d_cnew));
    if (d_mu) CUDA_CHECK(cudaFree(d_mu));

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(comm_stream));

    MPI_CHECK(MPI_Finalize());
    return 0;
}
