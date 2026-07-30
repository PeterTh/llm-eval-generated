#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ─── CUDA Kernels ───────────────────────────────────────────────────────────
// Each rank owns a slab in Z: local_nz planes with nx*ny elements each.
// Array layout with one ghost plane at top and bottom:
//   index(z, y, x) = (z+1) * plane_size + y * nx + x
// where z in [0, local_nz), plane_size = nx * ny
// Ghost cells: z = -1  → index 0..plane_size-1 (below)
//              z = nz  → index (nz+1)*plane_size..(nz+2)*plane_size-1 (above)

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* mu,
    int nx, int ny, int nz_local, size_t plane_size,
    int global_z_start, int total_nz,
    double dx, double dy, double dz,
    double gamma, double e_AA, double e_BB, double e_AB
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz_local) return;

    // Z-neighbor indices: when z ± 1 falls outside local range, it refers
    // to a ghost cell that was filled by MPI halo exchange.
    int z_up   = (z < nz_local - 1) ? z + 1 : z + 1; // always valid (ghost at top)
    int z_down = (z > 0)            ? z - 1 : z - 1; // always valid (ghost at bottom)
    int global_z = global_z_start + z;

    // Clamped boundary conditions at global domain boundaries
    int xp = (x < nx - 1) ? x + 1 : x;
    int xn = (x > 0)      ? x - 1 : 0;
    int yp = (y < ny - 1) ? y + 1 : y;
    int yn = (y > 0)      ? y - 1 : 0;

    // For Z: if at global boundary, use self instead of ghost
    if (global_z == 0)           z_down = z;      // clamped at bottom
    if (global_z == total_nz - 1) z_up   = z;      // clamped at top

    size_t plane = plane_size;

    // Indices
    size_t idx_self = (size_t)(z + 1) * plane + (size_t)y * nx + (size_t)x;
    // X neighbours (same y,z)
    size_t idx_xp   = (size_t)(z + 1) * plane + (size_t)y    * nx + (size_t)xp;
    size_t idx_xn   = (size_t)(z + 1) * plane + (size_t)y    * nx + (size_t)xn;
    // Y neighbours (same x,z)
    size_t idx_yp   = (size_t)(z + 1) * plane + (size_t)yp   * nx + (size_t)x;
    size_t idx_yn   = (size_t)(z + 1) * plane + (size_t)yn   * nx + (size_t)x;
    // Z neighbours (same x,y)
    size_t idx_zp   = (size_t)(z_up   + 1) * plane + (size_t)y    * nx + (size_t)x;
    size_t idx_zn   = (size_t)(z_down + 1) * plane + (size_t)y    * nx + (size_t)x;

    double cv = c[idx_self];

    // Laplacian of c at (x, y, global_z)
    double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * cv) / (dx * dx);
    double cyy = (c[idx_yp] + c[idx_yn] - 2.0 * cv) / (dy * dy);
    double czz = (c[idx_zp] + c[idx_zn] - 2.0 * cv) / (dz * dz);
    double laplacian = cxx + cyy + czz;

    // Chemical potential
    mu[idx_self] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv
                   - gamma * laplacian;
}

__global__ void cahnHilliardUpdateKernel(
    double* cnew, const double* __restrict__ cold, const double* __restrict__ mu,
    int nx, int ny, int nz_local, size_t plane_size,
    int global_z_start, int total_nz,
    double D, double dt, double dx, double dy, double dz
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz_local) return;

    int z_up   = (z < nz_local - 1) ? z + 1 : z + 1;
    int z_down = (z > 0)            ? z - 1 : z - 1;
    int global_z = global_z_start + z;

    int xp = (x < nx - 1) ? x + 1 : x;
    int xn = (x > 0)      ? x - 1 : 0;
    int yp = (y < ny - 1) ? y + 1 : y;
    int yn = (y > 0)      ? y - 1 : 0;

    if (global_z == 0)           z_down = z;
    if (global_z == total_nz - 1) z_up   = z;

    size_t plane = plane_size;

    size_t idx_self = (size_t)(z + 1) * plane + (size_t)y * nx + (size_t)x;
    size_t idx_xp   = (size_t)(z + 1) * plane + (size_t)y    * nx + (size_t)xp;
    size_t idx_xn   = (size_t)(z + 1) * plane + (size_t)y    * nx + (size_t)xn;
    size_t idx_yp   = (size_t)(z + 1) * plane + (size_t)yp   * nx + (size_t)x;
    size_t idx_yn   = (size_t)(z + 1) * plane + (size_t)yn   * nx + (size_t)x;
    size_t idx_zp   = (size_t)(z_up   + 1) * plane + (size_t)y    * nx + (size_t)x;
    size_t idx_zn   = (size_t)(z_down + 1) * plane + (size_t)y    * nx + (size_t)x;

    double muv = mu[idx_self];
    double muxx = (mu[idx_xp] + mu[idx_xn] - 2.0 * muv) / (dx * dx);
    double muyy = (mu[idx_yp] + mu[idx_yn] - 2.0 * muv) / (dy * dy);
    double muzb = (mu[idx_zp] + mu[idx_zn] - 2.0 * muv) / (dz * dz);

    cnew[idx_self] = cold[idx_self] + dt * D * (muxx + muyy + muzb);
}

// ─── Host Helper: MPI halo exchange via pinned host memory ──────────────────

static void exchangeHaloGPU(double* d_arr,
                             size_t plane_size, int nz_local,
                             int rank_below, int rank_above,
                             double* h_send_buf, double* h_recv_buf,
                             MPI_Comm comm) {
    int plane_elems = (int)plane_size;
    size_t plane_bytes = plane_size * sizeof(double);

    // Exchange with bottom neighbour
    if (rank_below >= 0) {
        // Our first local plane → send to rank_below (becomes their top ghost)
        cudaMemcpy(h_send_buf, d_arr + plane_size, plane_bytes, cudaMemcpyDeviceToHost);
        // Receive rank_below's last local plane → our bottom ghost
        MPI_Sendrecv(h_send_buf, plane_elems, MPI_DOUBLE, rank_below, 0,
                     h_recv_buf, plane_elems, MPI_DOUBLE, rank_below, 0,
                     comm, MPI_STATUS_IGNORE);
        cudaMemcpy(d_arr, h_recv_buf, plane_bytes, cudaMemcpyHostToDevice);
    }

    // Exchange with top neighbour
    if (rank_above >= 0) {
        // Our last local plane → send to rank_above (becomes their bottom ghost)
        cudaMemcpy(h_send_buf, d_arr + (size_t)nz_local * plane_size,
                   plane_bytes, cudaMemcpyDeviceToHost);
        // Receive rank_above's first local plane → our top ghost
        MPI_Sendrecv(h_send_buf, plane_elems, MPI_DOUBLE, rank_above, 0,
                     h_recv_buf, plane_elems, MPI_DOUBLE, rank_above, 0,
                     comm, MPI_STATUS_IGNORE);
        cudaMemcpy(d_arr + (size_t)(nz_local + 1) * plane_size,
                   h_recv_buf, plane_bytes, cudaMemcpyHostToDevice);
    }
}

// ─── Initialization (MPI + OpenMP) ─────────────────────────────────────────

static void initializeConcentrationHybrid(double* c, size_t nx, size_t ny,
                                           size_t local_nz, size_t global_z_start,
                                           size_t plane_size, size_t total_vol) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = (z + 1) * plane_size + y * nx + x;
                size_t linear_id = (global_z_start + z) * plane_size + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709) % total_vol)
                                 / (double)total_vol);
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// ─── Validation (MPI + OpenMP) ─────────────────────────────────────────────

static bool validateResultHybrid(const double* c, size_t nx, size_t ny,
                                  size_t local_nz, size_t plane_size,
                                  int rank) {
    bool local_ok = true;
    double local_min = c[plane_size];  // first local point
    double local_max = c[plane_size];

    #pragma omp parallel for reduction(&&: local_ok) reduction(min: local_min) reduction(max: local_max)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = (z + 1) * plane_size + y * nx + x;
                double val = c[idx];
                if (std::isnan(val) || std::isinf(val)) local_ok = false;
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
            }
        }
    }

    // Reduce across all ranks
    int all_ok = local_ok ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&all_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (!global_ok) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }
    return global_ok != 0;
}

// ─── Usage ──────────────────────────────────────────────────────────────────

void printUsage(const char* progName) {
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

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line options (all ranks)
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

    size_t total_nz = nz;            // global number of Z planes
    size_t plane_size = nx * ny;     // elements per Z plane

    // 1D domain decomposition along Z
    size_t q = total_nz / (size_t)size;
    size_t r = total_nz % (size_t)size;
    size_t local_nz = ((size_t)rank < r) ? q + 1 : q;
    size_t global_z_start = ((size_t)rank < r)
                                ? (size_t)rank * (q + 1)
                                : r * (q + 1) + ((size_t)rank - r) * q;

    int rank_below = (rank > 0) ? rank - 1 : -1;
    int rank_above = (rank < size - 1) ? rank + 1 : -1;

    // Select GPU (round-robin across ranks)
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA-capable device found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(rank % num_devices);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d  |  OpenMP threads: %d  |  GPUs available: %d\n",
               size, omp_get_max_threads(), num_devices);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, total_nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    // ── Physical parameters ──────────────────────────────────────────────
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    // ── Allocate device memory (with ghost planes) ──────────────────────
    size_t alloc_planes = local_nz + 2;         // +2 for ghost
    size_t device_elems = alloc_planes * plane_size;

    double *cold_d = nullptr, *cnew_d = nullptr, *mu_d = nullptr;
    cudaMalloc(&cold_d, device_elems * sizeof(double));
    cudaMalloc(&cnew_d, device_elems * sizeof(double));
    cudaMalloc(&mu_d,   device_elems * sizeof(double));

    // Pinned host buffers for MPI halo exchange (one plane each)
    double *h_send_buf = nullptr, *h_recv_buf = nullptr;
    cudaMallocHost(&h_send_buf, plane_size * sizeof(double));
    cudaMallocHost(&h_recv_buf, plane_size * sizeof(double));

    // ── Initialize concentration field on GPU ───────────────────────────
    if (rank == 0) printf("Initializing concentration field...\n");

    // Allocate host buffer for initialization, then copy to GPU
    std::vector<double> cold_host(device_elems, 0.0);
    size_t total_vol = nx * ny * total_nz;
    initializeConcentrationHybrid(cold_host.data(), nx, ny, local_nz,
                                   global_z_start, plane_size, total_vol);
    cudaMemcpy(cold_d, cold_host.data(), device_elems * sizeof(double),
               cudaMemcpyHostToDevice);

    // ── Main simulation loop ────────────────────────────────────────────
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    // Exchange initial cold halo
    exchangeHaloGPU(cold_d, plane_size, (int)local_nz,
                    rank_below, rank_above, h_send_buf, h_recv_buf,
                    MPI_COMM_WORLD);

    dim3 block(16, 16);
    dim3 grid((int)((nx + 15) / 16), (int)((ny + 15) / 16), (int)local_nz);

    for (int t = 0; t < iterations; ++t) {
        // Step 1: compute chemical potential  c → mu
        computeChemicalPotentialKernel<<<grid, block>>>(
            cold_d, mu_d,
            (int)nx, (int)ny, (int)local_nz, plane_size,
            (int)global_z_start, (int)total_nz,
            dx, dy, dz, gamma, e_AA, e_BB, e_AB
        );

        // Step 2: exchange mu halo
        exchangeHaloGPU(mu_d, plane_size, (int)local_nz,
                        rank_below, rank_above, h_send_buf, h_recv_buf,
                        MPI_COMM_WORLD);

        // Step 3: Cahn-Hilliard update  (cold, mu) → cnew
        cahnHilliardUpdateKernel<<<grid, block>>>(
            cnew_d, cold_d, mu_d,
            (int)nx, (int)ny, (int)local_nz, plane_size,
            (int)global_z_start, (int)total_nz,
            D, dt, dx, dy, dz
        );

        // Step 4: exchange cnew halo (next iteration's cold halo)
        exchangeHaloGPU(cnew_d, plane_size, (int)local_nz,
                        rank_below, rank_above, h_send_buf, h_recv_buf,
                        MPI_COMM_WORLD);

        // Step 5: swap buffers
        std::swap(cold_d, cnew_d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // After the final swap, the result is in cold_d.  Copy it back to host.
    std::vector<double> cold_local(device_elems);
    cudaMemcpy(cold_local.data(), cold_d, device_elems * sizeof(double),
               cudaMemcpyDeviceToHost);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)(nx * ny * total_nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ── Print results (gather to rank 0) ─────────────────────────────────
    if (printResults) {
        // Gather local data sizes
        std::vector<int> recv_counts((size_t)size, 0);
        std::vector<int> recv_displs((size_t)size, 0);
        int local_elems = (int)(local_nz * plane_size);

        MPI_Gather(&local_elems, 1, MPI_INT,
                   recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            int offset = 0;
            for (int i = 0; i < size; ++i) {
                recv_displs[i] = offset;
                offset += recv_counts[i];
            }
        }

        // Gather into full array on rank 0
        std::vector<double> full_cold;
        if (rank == 0) full_cold.resize(total_nz * plane_size);

        // Extract just the local domain (no ghost) from cold_local
        std::vector<double> cold_no_ghost((size_t)local_elems);
        #pragma omp parallel for
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t i = 0; i < plane_size; ++i) {
                cold_no_ghost[z * plane_size + i] = cold_local[(z + 1) * plane_size + i];
            }
        }

        MPI_Gatherv(cold_no_ghost.data(), local_elems, MPI_DOUBLE,
                    rank == 0 ? full_cold.data() : nullptr,
                    recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full_cold, "Concentration");
        }
    }

    // ── Validation ──────────────────────────────────────────────────────
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResultHybrid(cold_local.data(), nx, ny,
                                           local_nz, plane_size, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    cudaFreeHost(h_recv_buf);
    cudaFreeHost(h_send_buf);
    cudaFree(mu_d);
    cudaFree(cnew_d);
    cudaFree(cold_d);

    MPI_Finalize();
    return 0;
}
