#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// -----------------------------------------------------------------------
// CUDA error checking macro
// -----------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        MPI_Abort(MPI_COMM_WORLD, 1);                                         \
    }                                                                         \
} while(0)

// -----------------------------------------------------------------------
// Index calculation -- works in host and device code.
// Arrays include ghost cells:  z=0 and z=nz_local+1 are ghosts,
// real cells live at z in [1, nz_local].
// -----------------------------------------------------------------------
__host__ __device__ inline constexpr size_t
idx3(const size_t x, const size_t y, const size_t z,
     const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// -----------------------------------------------------------------------
// Device laplacian + CUDA kernels
// -----------------------------------------------------------------------

// 7-point Laplacian with clamped BC in X/Y.
// Z neighbours use ghost cells (kept valid by MPI + clamping).
__device__ static inline double
laplacian_device(const double* c,
                 size_t nx, size_t ny,
                 double dx, double dy, double dz,
                 size_t x, size_t y, size_t z) {
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0)     ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0)     ? y - 1 : 0;

    double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)]
                  - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)]
                  - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)]
                  - 2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

// Chemical potential:  mu = f'(c) - gamma * laplacian(c)
__global__ static void
chem_potential_kernel(const double* c, double* mu,
                      size_t nx, size_t ny, size_t nz_local,
                      double dx, double dy, double dz,
                      double gamma,
                      double e_AA, double e_BB, double e_AB) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    for (size_t z = 1; z <= nz_local; ++z) {
        size_t idx = idx3(x, y, z, nx, ny);
        double cv = c[idx];
        mu[idx] = (4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                          - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv)
                  - gamma * laplacian_device(c, nx, ny, dx, dy, dz, x, y, z);
    }
}

// Cahn-Hilliard time-stepping:  cnew = cold + dt * D * laplacian(mu)
__global__ static void
update_kernel(double* cnew, const double* cold, const double* mu,
              size_t nx, size_t ny, size_t nz_local,
              double D, double dt,
              double dx, double dy, double dz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    for (size_t z = 1; z <= nz_local; ++z) {
        size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D
                    * laplacian_device(mu, nx, ny, dx, dy, dz, x, y, z);
    }
}

// -----------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------

// Exchange ghost layers on a device array:
//   copy boundary planes  device->host,
//   MPI_Sendrecv with neighbours,
//   copy received data   host->device ghost cells.
// Physical boundaries use clamped (Neumann=0) condition.
static void
exchange_ghosts(double* d_arr,
                size_t nx, size_t ny, size_t nz_local,
                int rank, int num_ranks) {
    size_t plane     = nx * ny;
    size_t plane_bytes = plane * sizeof(double);

    // Pinned host buffers for faster device<->host transfers
    double *h_send_up = nullptr, *h_send_down = nullptr;
    double *h_recv_up = nullptr, *h_recv_down = nullptr;
    cudaMallocHost(&h_send_up,   plane_bytes);
    cudaMallocHost(&h_send_down, plane_bytes);
    cudaMallocHost(&h_recv_up,   plane_bytes);
    cudaMallocHost(&h_recv_down, plane_bytes);

    double* d_lower_real  = d_arr + idx3(0, 0, 1,          nx, ny);
    double* d_upper_real  = d_arr + idx3(0, 0, nz_local,   nx, ny);
    double* d_lower_ghost = d_arr + idx3(0, 0, 0,          nx, ny);
    double* d_upper_ghost = d_arr + idx3(0, 0, nz_local+1, nx, ny);

    // Device -> host
    CUDA_CHECK(cudaMemcpy(h_send_up,   d_upper_real, plane_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_down, d_lower_real, plane_bytes,
                          cudaMemcpyDeviceToHost));

    // MPI halo exchange
    MPI_Status st;
    if (rank > 0)
        MPI_Sendrecv(h_send_down, (int)plane, MPI_DOUBLE, rank-1, 0,
                     h_recv_down, (int)plane, MPI_DOUBLE, rank-1, 1,
                     MPI_COMM_WORLD, &st);
    if (rank < num_ranks - 1)
        MPI_Sendrecv(h_send_up,   (int)plane, MPI_DOUBLE, rank+1, 1,
                     h_recv_up,   (int)plane, MPI_DOUBLE, rank+1, 0,
                     MPI_COMM_WORLD, &st);

    // Host -> device  (or device->device for clamped boundaries)
    if (rank > 0)
        CUDA_CHECK(cudaMemcpy(d_lower_ghost, h_recv_down, plane_bytes,
                              cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemcpy(d_lower_ghost, d_lower_real, plane_bytes,
                              cudaMemcpyDeviceToDevice));

    if (rank < num_ranks - 1)
        CUDA_CHECK(cudaMemcpy(d_upper_ghost, h_recv_up, plane_bytes,
                              cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemcpy(d_upper_ghost, d_upper_real, plane_bytes,
                              cudaMemcpyDeviceToDevice));

    cudaFreeHost(h_send_up);   cudaFreeHost(h_send_down);
    cudaFreeHost(h_recv_up);   cudaFreeHost(h_recv_down);
}

// Initialise host slab (real cells only) using the same deterministic
// pseudo-random sequence as the original serial code.
static void
init_slab(std::vector<double>& slab,
          size_t nx, size_t ny, size_t nz_local, size_t nz_global,
          size_t z_start) {
    size_t vol = nx * ny * nz_global;
#pragma omp parallel for collapse(3)
    for (size_t zl = 1; zl <= nz_local; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_z = z_start + (zl - 1);
                size_t lin = global_z * (nx * ny) + y * nx + x;
                double pseudo = ((lin + 1) * 1299709ull % vol)
                                / static_cast<double>(vol);
                slab[idx3(x, y, zl, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// -----------------------------------------------------------------------
// Validation with MPI reduction
// -----------------------------------------------------------------------
static bool
validate_result_mpi(const std::vector<double>& slab,
                    size_t nx, size_t ny, size_t nz_local,
                    int rank) {
    size_t local_size = nx * ny * nz_local;
    double local_min = slab[idx3(0, 0, 1, nx, ny)];
    double local_max = local_min;

    // Real cells start at offset nx*ny (skipping the lower ghost layer)
    const double* base = slab.data() + nx * ny;
#pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (size_t i = 0; i < local_size; ++i) {
        double val = base[i];
        if (std::isnan(val) || std::isinf(val)) {
            fprintf(stderr, "[rank %d] Validation failed: NaN/Inf\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (val < local_min) local_min = val;
        if (val > local_max) local_max = val;
    }

    double global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
        }
    }
    int ok = (global_max <= 10.0 && global_min >= -10.0) ? 1 : 0;
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
}

// -----------------------------------------------------------------------
// Usage
// -----------------------------------------------------------------------
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

// -----------------------------------------------------------------------
// Main -- hybrid MPI + OpenMP + CUDA driver
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int num_ranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ---- parse arguments (identical on every rank) ----
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

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

    // ---- CUDA device: round-robin across visible devices ----
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int dev_id = (ndev > 0) ? (rank % ndev) : 0;
    CUDA_CHECK(cudaSetDevice(dev_id));

    // ---- MPI domain decomposition along Z ----
    size_t base = nz / (size_t)num_ranks;
    size_t rem  = nz % (size_t)num_ranks;
    size_t nz_local = base + ((size_t)rank < rem ? 1 : 0);
    size_t z_start  = 0;
    for (int r = 0; r < rank; ++r)
        z_start += base + (r < (int)rem ? 1 : 0);

    size_t nz_with_ghosts = nz_local + 2;     // +2 ghost layers
    size_t slab_bytes = nx * ny * nz_with_ghosts * sizeof(double);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark"
               " (hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d | CUDA devices: %d\n", num_ranks, ndev);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters (same as original)
    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    // ---- memory ----
    // Host slab (includes ghost cells)
    std::vector<double> h_slab(nx * ny * nz_with_ghosts, 0.0);

    // Device arrays
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, slab_bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, slab_bytes));
    CUDA_CHECK(cudaMalloc(&d_mu,   slab_bytes));

    // ---- initialise on host, push to device ----
    init_slab(h_slab, nx, ny, nz_local, nz, z_start);

    // Set ghost cells for initial state
    if (num_ranks == 1) {
        // Clamped BC at both ends
        memcpy(&h_slab[0],
               &h_slab[idx3(0, 0, 1, nx, ny)],
               nx * ny * sizeof(double));
        memcpy(&h_slab[idx3(0, 0, nz_local + 1, nx, ny)],
               &h_slab[idx3(0, 0, nz_local, nx, ny)],
               nx * ny * sizeof(double));
    } else {
        // Physical boundary clamping
        if (rank == 0)
            memcpy(&h_slab[0],
                   &h_slab[idx3(0, 0, 1, nx, ny)],
                   nx * ny * sizeof(double));
        if (rank == num_ranks - 1)
            memcpy(&h_slab[idx3(0, 0, nz_local + 1, nx, ny)],
                   &h_slab[idx3(0, 0, nz_local, nx, ny)],
                   nx * ny * sizeof(double));

        // Interior ghosts via MPI exchange
        size_t plane = nx * ny;
        MPI_Status st;
        double* h_lr = &h_slab[idx3(0, 0, 1,          nx, ny)];
        double* h_ur = &h_slab[idx3(0, 0, nz_local,   nx, ny)];
        double* h_lg = &h_slab[0];
        double* h_ug = &h_slab[idx3(0, 0, nz_local+1, nx, ny)];
        if (rank > 0)
            MPI_Sendrecv(h_lr, (int)plane, MPI_DOUBLE, rank-1, 0,
                         h_lg, (int)plane, MPI_DOUBLE, rank-1, 1,
                         MPI_COMM_WORLD, &st);
        if (rank < num_ranks - 1)
            MPI_Sendrecv(h_ur, (int)plane, MPI_DOUBLE, rank+1, 1,
                         h_ug, (int)plane, MPI_DOUBLE, rank+1, 0,
                         MPI_COMM_WORLD, &st);
    }

    CUDA_CHECK(cudaMemcpy(d_cold, h_slab.data(), slab_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cnew, h_slab.data(), slab_bytes,
                          cudaMemcpyHostToDevice));

    // ---- main time-stepping ----
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    dim3 block(16, 16);
    dim3 grid((nx + 15) / 16, (ny + 15) / 16);

    for (int t = 0; t < iterations; ++t) {
        // 1. Refresh ghost cells for cold (MPI exchange or clamped BC)
        exchange_ghosts(d_cold, nx, ny, nz_local, rank, num_ranks);

        // 2. Chemical potential on GPU
        chem_potential_kernel<<<grid, block>>>(
            d_cold, d_mu, nx, ny, nz_local,
            dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // 3. Refresh ghost cells for mu (needed by update step)
        exchange_ghosts(d_mu, nx, ny, nz_local, rank, num_ranks);

        // 4. Cahn-Hilliard update on GPU
        update_kernel<<<grid, block>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz_local,
            D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        // 5. Swap hot/cold device buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);
    size_t global_grid = nx * ny * nz;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        double cellUpdates = (double)global_grid * iterations;
        double mcups = cellUpdates / (global_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- copy result back to host ----
    // d_cold holds the newest state after the final swap
    CUDA_CHECK(cudaMemcpy(h_slab.data(), d_cold, slab_bytes,
                          cudaMemcpyDeviceToHost));

    // ---- optional: print_results (gather on rank 0) ----
    if (printResults) {
        size_t my_real = nx * ny * nz_local;
        std::vector<int> recvcounts(num_ranks), displs(num_ranks);
        MPI_Gather((int*)&my_real, 1, MPI_INT,
                   recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            int offset = 0;
            for (int r = 0; r < num_ranks; ++r) {
                displs[r] = offset;
                offset += recvcounts[r];
            }
        }

        std::vector<double> full;
        double* send_buf = h_slab.data() + nx * ny; // skip lower ghost

        if (rank == 0) {
            full.resize(global_grid);
            MPI_Gatherv(send_buf, (int)my_real, MPI_DOUBLE,
                        full.data(), recvcounts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            print_results(full, "Concentration");
        } else {
            MPI_Gatherv(send_buf, (int)my_real, MPI_DOUBLE,
                        nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // ---- validation ----
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validate_result_mpi(h_slab, nx, ny, nz_local, rank);
        if (rank == 0)
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_Finalize();
    return 0;
}
