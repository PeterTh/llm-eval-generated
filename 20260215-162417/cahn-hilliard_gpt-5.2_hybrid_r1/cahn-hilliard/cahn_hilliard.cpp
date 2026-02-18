#include <algorithm>
#include <chrono>
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

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t _e = (call);                                                                     \
        if (_e != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));  \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                            \
        }                                                                                           \
    } while (0)

__host__ __device__ __forceinline__ size_t idx3(const int x, const int y, const int z, const int nx, const int ny) noexcept {
    return static_cast<size_t>(z) * static_cast<size_t>(nx) * static_cast<size_t>(ny) +
           static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

__device__ __forceinline__ double laplacian_clamped_xyz(const double* __restrict__ a,
                                                        const int x, const int y, const int z,
                                                        const int nx, const int ny,
                                                        const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const int xp = (x + 1 < nx) ? (x + 1) : x;
    const int xn = (x > 0) ? (x - 1) : 0;
    const int yp = (y + 1 < ny) ? (y + 1) : y;
    const int yn = (y > 0) ? (y - 1) : 0;

    const size_t c0 = idx3(x, y, z, nx, ny);
    const double center = a[c0];

    const double cxx = (a[idx3(xp, y, z, nx, ny)] + a[idx3(xn, y, z, nx, ny)] - 2.0 * center) * inv_dx2;
    const double cyy = (a[idx3(x, yp, z, nx, ny)] + a[idx3(x, yn, z, nx, ny)] - 2.0 * center) * inv_dy2;
    const double czz = (a[idx3(x, y, z + 1, nx, ny)] + a[idx3(x, y, z - 1, nx, ny)] - 2.0 * center) * inv_dz2;

    return cxx + cyy + czz;
}

__global__ void chemical_potential_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                         int nx, int ny, int local_nz,
                                         double inv_dx2, double inv_dy2, double inv_dz2,
                                         double gamma, double e_AA, double e_BB, double e_AB) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int slice = nx * ny;
    const int n = slice * local_nz;
    if (tid >= n) return;

    const int x = tid % nx;
    const int y = (tid / nx) % ny;
    const int z = (tid / slice) + 1;  // interior z in [1, local_nz]

    const size_t id = idx3(x, y, z, nx, ny);
    const double cv = c[id];

    const double lap = laplacian_clamped_xyz(c, x, y, z, nx, ny, inv_dx2, inv_dy2, inv_dz2);

    mu[id] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void update_kernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             int nx, int ny, int local_nz,
                             double inv_dx2, double inv_dy2, double inv_dz2,
                             double Ddt) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int slice = nx * ny;
    const int n = slice * local_nz;
    if (tid >= n) return;

    const int x = tid % nx;
    const int y = (tid / nx) % ny;
    const int z = (tid / slice) + 1;

    const size_t id = idx3(x, y, z, nx, ny);
    const double lap_mu = laplacian_clamped_xyz(mu, x, y, z, nx, ny, inv_dx2, inv_dy2, inv_dz2);
    cnew[id] = cold[id] + Ddt * lap_mu;
}

static void exchange_halos_z(double* d_c_with_halo,
                            int nx, int ny, int local_nz,
                            int rank, int size,
                            std::vector<double>& send_lower,
                            std::vector<double>& send_upper,
                            std::vector<double>& recv_lower,
                            std::vector<double>& recv_upper) {
    const int slice = nx * ny;

    // Copy boundary slices from device to host
    CUDA_CHECK(cudaMemcpy(send_lower.data(), d_c_with_halo + static_cast<size_t>(1) * slice,
                          static_cast<size_t>(slice) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(send_upper.data(), d_c_with_halo + static_cast<size_t>(local_nz) * slice,
                          static_cast<size_t>(slice) * sizeof(double), cudaMemcpyDeviceToHost));

    const int down = rank - 1;
    const int up = rank + 1;

    // Lower halo
    if (down >= 0) {
        MPI_Sendrecv(send_lower.data(), slice, MPI_DOUBLE, down, 100,
                     recv_lower.data(), slice, MPI_DOUBLE, down, 101,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Clamped global boundary
        std::memcpy(recv_lower.data(), send_lower.data(), static_cast<size_t>(slice) * sizeof(double));
    }

    // Upper halo
    if (up < size) {
        MPI_Sendrecv(send_upper.data(), slice, MPI_DOUBLE, up, 101,
                     recv_upper.data(), slice, MPI_DOUBLE, up, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Clamped global boundary
        std::memcpy(recv_upper.data(), send_upper.data(), static_cast<size_t>(slice) * sizeof(double));
    }

    // Copy received halos back to device
    CUDA_CHECK(cudaMemcpy(d_c_with_halo + static_cast<size_t>(0) * slice, recv_lower.data(),
                          static_cast<size_t>(slice) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_with_halo + static_cast<size_t>(local_nz + 1) * slice, recv_upper.data(),
                          static_cast<size_t>(slice) * sizeof(double), cudaMemcpyHostToDevice));
}

static void initializeConcentration_local(double* d_c_with_halo,
                                         int nx, int ny, int nz,
                                         int z_start, int local_nz) {
    const size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t vol = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    // Build host buffer for interior, then upload once.
    std::vector<double> host(static_cast<size_t>(local_nz) * slice);

#pragma omp parallel for collapse(3)
    for (int z = 0; z < local_nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int gz = z_start + z;
                const size_t linear_id = static_cast<size_t>(gz) * slice + static_cast<size_t>(y) * static_cast<size_t>(nx) +
                                         static_cast<size_t>(x);
                const double pseudo = static_cast<double>((((linear_id + 1) * 1299709ULL) % vol)) / static_cast<double>(vol);
                host[static_cast<size_t>(z) * slice + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x)] =
                    -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Copy interior to device at z=1
    CUDA_CHECK(cudaMemcpy(d_c_with_halo + slice, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
}

static bool validateResult_mpi(const std::vector<double>& local, int rank, int size) {
    (void)size;

    int local_ok = 1;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

#pragma omp parallel
    {
        int ok_priv = 1;
        double min_priv = std::numeric_limits<double>::infinity();
        double max_priv = -std::numeric_limits<double>::infinity();

#pragma omp for nowait
        for (size_t i = 0; i < local.size(); ++i) {
            const double v = local[i];
            if (std::isnan(v) || std::isinf(v)) ok_priv = 0;
            min_priv = std::min(min_priv, v);
            max_priv = std::max(max_priv, v);
        }

#pragma omp critical
        {
            local_ok = local_ok && ok_priv;
            local_min = std::min(local_min, min_priv);
            local_max = std::max(local_max, max_priv);
        }
    }

    int global_ok = 0;
    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        if (!global_ok) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);

        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    return global_ok != 0;
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx_in = 64;
    size_t ny_in = 0;
    size_t nz_in = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx_in = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny_in = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz_in = static_cast<size_t>(atoi(argv[++i]));
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

    if (ny_in == 0) ny_in = nx_in;
    if (nz_in == 0) nz_in = nx_in;

    const int nx = static_cast<int>(nx_in);
    const int ny = static_cast<int>(ny_in);
    const int nz = static_cast<int>(nz_in);

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid sizes\n");
        MPI_Finalize();
        return 1;
    }

    // 1D slab decomposition along Z
    const int base = nz / size;
    const int rem = nz % size;
    const int local_nz = base + (rank < rem ? 1 : 0);
    const int z_start = rank * base + (rank < rem ? rank : rem);

    if (local_nz <= 0) {
        if (rank == 0) fprintf(stderr, "Too many MPI ranks for nz=%d\n", nz);
        MPI_Finalize();
        return 1;
    }

    // Select GPU per rank
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
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
    const double Ddt = dt * D;

    const size_t slice = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t local_with_halo_elems = static_cast<size_t>(local_nz + 2) * slice;

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    CUDA_CHECK(cudaMalloc(&d_cold, local_with_halo_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_with_halo_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_with_halo_elems * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_cold, 0, local_with_halo_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, local_with_halo_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, local_with_halo_elems * sizeof(double)));

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration_local(d_cold, nx, ny, nz, z_start, local_nz);

    std::vector<double> send_lower(slice), send_upper(slice), recv_lower(slice), recv_upper(slice);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    const int threads = 256;
    const int nInterior = static_cast<int>(slice) * local_nz;
    const int blocks = (nInterior + threads - 1) / threads;

    const double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Halo exchange for current cold field
        exchange_halos_z(d_cold, nx, ny, local_nz, rank, size, send_lower, send_upper, recv_lower, recv_upper);

        chemical_potential_kernel<<<blocks, threads>>>(d_cold, d_mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        update_kernel<<<blocks, threads>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, Ddt);
        CUDA_CHECK(cudaGetLastError());

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double gridSize = static_cast<double>(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));
        const double cellUpdates = gridSize * static_cast<double>(iterations);
        const double mcups = cellUpdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy local interior back to host
    std::vector<double> local_host(static_cast<size_t>(local_nz) * slice);
    CUDA_CHECK(cudaMemcpy(local_host.data(), d_cold + slice, local_host.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const int lnz = base + (r < rem ? 1 : 0);
            counts[r] = lnz * static_cast<int>(slice);
            displs[r] = (r == 0) ? 0 : (displs[r - 1] + counts[r - 1]);
        }

        std::vector<double> full;
        if (rank == 0) full.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));

        MPI_Gatherv(local_host.data(), static_cast<int>(local_host.size()), MPI_DOUBLE,
                    rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full, "Concentration");
        }
    }

    // Validation
    int exit_code = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool ok = validateResult_mpi(local_host, rank, size);
        if (rank == 0) {
            if (ok) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return exit_code;
}
