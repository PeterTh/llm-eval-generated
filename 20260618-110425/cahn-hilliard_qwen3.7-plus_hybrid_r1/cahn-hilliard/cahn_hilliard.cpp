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

// MPI tags for halo exchange
#define TAG_DOWN 100
#define TAG_UP   200

// ============================================================
// CUDA Kernels
// ============================================================

// Compute chemical potential: mu = f(c) - gamma * laplacian(c)
// Array layout with halos: index (z+1)*nxny + y*nx + x for local z
// Halo at index 0 (lower) and (local_nz+1)*nxny (upper)
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z;

    if (x >= nx || y >= ny || z >= local_nz) return;

    const size_t nxny = nx * ny;
    const size_t base = y * nx + x;
    const size_t center = (z + 1) * nxny + base;
    const double cv = c[center];

    // Clamped X neighbors
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    // Clamped Y neighbors
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t zrow = (z + 1) * nxny;

    // Laplacian
    const double lap_x = (c[zrow + y * nx + xp] + c[zrow + y * nx + xn] - 2.0 * cv) / (dx * dx);
    const double lap_y = (c[zrow + yp * nx + x] + c[zrow + yn * nx + x] - 2.0 * cv) / (dy * dy);
    const double lap_z = (c[(z + 2) * nxny + base] + c[z * nxny + base] - 2.0 * cv) / (dz * dz);

    mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * (lap_x + lap_y + lap_z);
}

// Cahn-Hilliard update: cnew = cold + dt*D*laplacian(mu)
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double D, const double dt, const double dx, const double dy, const double dz)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z;

    if (x >= nx || y >= ny || z >= local_nz) return;

    const size_t nxny = nx * ny;
    const size_t base = y * nx + x;
    const size_t center = (z + 1) * nxny + base;

    // Clamped X neighbors
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    // Clamped Y neighbors
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double mu_c = mu[center];
    const size_t zrow = (z + 1) * nxny;

    const double lap_x = (mu[zrow + y * nx + xp] + mu[zrow + y * nx + xn] - 2.0 * mu_c) / (dx * dx);
    const double lap_y = (mu[zrow + yp * nx + x] + mu[zrow + yn * nx + x] - 2.0 * mu_c) / (dy * dy);
    const double lap_z = (mu[(z + 2) * nxny + base] + mu[z * nxny + base] - 2.0 * mu_c) / (dz * dz);

    cnew[center] = cold[center] + dt * D * (lap_x + lap_y + lap_z);
}

// Clamp halos at global boundaries (copy boundary plane to halo)
__global__ void clampHalosKernel(
    double* __restrict__ arr,
    const size_t nx, const size_t ny, const size_t local_nz,
    const bool clamp_lower, const bool clamp_upper)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    const size_t nxny = nx * ny;
    const size_t base = y * nx + x;

    if (clamp_lower) {
        arr[base] = arr[nxny + base]; // halo[0] = local[0]
    }
    if (clamp_upper) {
        arr[(local_nz + 1) * nxny + base] = arr[local_nz * nxny + base]; // halo[upper] = local[last]
    }
}

// ============================================================
// CPU functions with OpenMP (fallback and initialization)
// ============================================================

void computeChemicalPotentialCPU(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const size_t nxny = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t base = y * nx + x;
                const size_t center = (z + 1) * nxny + base;
                const double cv = c[center];

                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t yn = (y > 0) ? y - 1 : 0;

                const size_t zrow = (z + 1) * nxny;

                const double lap_x = (c[zrow + y * nx + xp] + c[zrow + y * nx + xn] - 2.0 * cv) / (dx * dx);
                const double lap_y = (c[zrow + yp * nx + x] + c[zrow + yn * nx + x] - 2.0 * cv) / (dy * dy);
                const double lap_z = (c[(z + 2) * nxny + base] + c[z * nxny + base] - 2.0 * cv) / (dz * dz);

                mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                            + 3.0 * cv + cv * cv * cv
                            - gamma * (lap_x + lap_y + lap_z);
            }
        }
    }
}

void cahnHilliardUpdateCPU(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double D, const double dt, const double dx, const double dy, const double dz)
{
    const size_t nxny = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t base = y * nx + x;
                const size_t center = (z + 1) * nxny + base;

                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t yn = (y > 0) ? y - 1 : 0;

                const double mu_c = mu[center];
                const size_t zrow = (z + 1) * nxny;

                const double lap_x = (mu[zrow + y * nx + xp] + mu[zrow + y * nx + xn] - 2.0 * mu_c) / (dx * dx);
                const double lap_y = (mu[zrow + yp * nx + x] + mu[zrow + yn * nx + x] - 2.0 * mu_c) / (dy * dy);
                const double lap_z = (mu[(z + 2) * nxny + base] + mu[z * nxny + base] - 2.0 * mu_c) / (dz * dz);

                cnew[center] = cold[center] + dt * D * (lap_x + lap_y + lap_z);
            }
        }
    }
}

// ============================================================
// Halo exchange helpers
// ============================================================

// Clamp halos on host (for CPU path or global boundaries)
void clampHalosHost(double* arr, size_t nx, size_t ny, size_t local_nz,
                    bool clamp_lower, bool clamp_upper)
{
    const size_t nxny = nx * ny;
    if (clamp_lower) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nxny; ++i) {
            arr[i] = arr[nxny + i];
        }
    }
    if (clamp_upper) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nxny; ++i) {
            arr[(local_nz + 1) * nxny + i] = arr[local_nz * nxny + i];
        }
    }
}

// MPI halo exchange on host arrays
void exchangeHalosHost(double* arr, size_t nx, size_t ny, size_t local_nz,
                       int rank, int size,
                       double* h_send_buf, double* h_recv_buf)
{
    const size_t nxny = nx * ny;

    if (size == 1) return;

    // h_send_buf: [first_plane | last_plane] = 2*nxny
    // h_recv_buf: [lower_halo | upper_halo] = 2*nxny
    double* h_send_first = h_send_buf;
    double* h_send_last  = h_send_buf + nxny;
    double* h_recv_lower = h_recv_buf;
    double* h_recv_upper = h_recv_buf + nxny;

    // Pack send buffers
    std::memcpy(h_send_first, arr + nxny, nxny * sizeof(double));          // first local plane
    std::memcpy(h_send_last, arr + local_nz * nxny, nxny * sizeof(double)); // last local plane

    MPI_Request reqs[4];
    int nreq = 0;

    if (rank > 0) {
        MPI_Irecv(h_recv_lower, (int)nxny, MPI_DOUBLE, rank - 1, TAG_UP, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(h_send_first, (int)nxny, MPI_DOUBLE, rank - 1, TAG_DOWN, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(h_recv_upper, (int)nxny, MPI_DOUBLE, rank + 1, TAG_DOWN, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(h_send_last, (int)nxny, MPI_DOUBLE, rank + 1, TAG_UP, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Unpack received halos
    if (rank > 0) {
        std::memcpy(arr, h_recv_lower, nxny * sizeof(double)); // lower halo
    }
    if (rank < size - 1) {
        std::memcpy(arr + (local_nz + 1) * nxny, h_recv_upper, nxny * sizeof(double)); // upper halo
    }
}

// GPU halo exchange: copy planes to host, MPI exchange, copy back
void exchangeHalosGPU(double* d_arr, size_t nx, size_t ny, size_t local_nz,
                      int rank, int size,
                      double* h_send_buf, double* h_recv_buf)
{
    const size_t nxny = nx * ny;
    const size_t plane_bytes = nxny * sizeof(double);

    if (size == 1) return;

    double* h_send_first = h_send_buf;
    double* h_send_last  = h_send_buf + nxny;
    double* h_recv_lower = h_recv_buf;
    double* h_recv_upper = h_recv_buf + nxny;

    // Copy boundary planes from device to host
    cudaMemcpy(h_send_first, d_arr + nxny, plane_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_send_last, d_arr + local_nz * nxny, plane_bytes, cudaMemcpyDeviceToHost);

    MPI_Request reqs[4];
    int nreq = 0;

    if (rank > 0) {
        MPI_Irecv(h_recv_lower, (int)nxny, MPI_DOUBLE, rank - 1, TAG_UP, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(h_send_first, (int)nxny, MPI_DOUBLE, rank - 1, TAG_DOWN, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(h_recv_upper, (int)nxny, MPI_DOUBLE, rank + 1, TAG_DOWN, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(h_send_last, (int)nxny, MPI_DOUBLE, rank + 1, TAG_UP, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Copy received halos to device
    if (rank > 0) {
        cudaMemcpy(d_arr, h_recv_lower, plane_bytes, cudaMemcpyHostToDevice);
    }
    if (rank < size - 1) {
        cudaMemcpy(d_arr + (local_nz + 1) * nxny, h_recv_upper, plane_bytes, cudaMemcpyHostToDevice);
    }
}

// ============================================================
// Initialization and validation
// ============================================================

void initializeConcentration(double* arr, size_t nx, size_t ny, size_t local_nz,
                             size_t z_start, size_t global_nz)
{
    const size_t nxny = nx * ny;
    const size_t vol = nx * ny * global_nz;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z + z_start;
                const size_t linear_id = global_z * nxny + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                // Store at halo-offset position
                arr[(z + 1) * nxny + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t nx, size_t ny, size_t nz) {
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

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

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printRes = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printRes = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
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

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
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
    const double gamma_val = 0.5;
    const double D = 1.0;

    // Domain decomposition along Z
    size_t z_per_rank = nz / size;
    size_t z_start = rank * z_per_rank;
    size_t local_nz = (rank == size - 1) ? (nz - z_start) : z_per_rank;

    const size_t nxny = nx * ny;
    const size_t alloc_nz = local_nz + 2; // including halos
    const size_t alloc_size = nxny * alloc_nz;

    bool is_lower_global = (rank == 0);
    bool is_upper_global = (rank == size - 1);

    // Check GPU availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    bool use_gpu = (device_count > 0 && local_nz > 0);

    if (rank == 0) {
        if (use_gpu) {
            printf("Using GPU acceleration (%d device(s) detected)\n", device_count);
        } else {
            printf("Using CPU with OpenMP (%d threads)\n", omp_get_max_threads());
        }
    }

    // Host buffers for halo exchange
    std::vector<double> h_send_buf(2 * nxny);
    std::vector<double> h_recv_buf(2 * nxny);

    // Global grid size for performance calculation
    size_t global_gridSize = nx * ny * nz;

    if (use_gpu) {
        // ========================================
        // GPU PATH
        // ========================================
        int gpu_id = rank % device_count;
        cudaSetDevice(gpu_id);

        // Allocate device arrays (with halos)
        double *d_cold, *d_cnew, *d_mu;
        cudaMalloc(&d_cold, alloc_size * sizeof(double));
        cudaMalloc(&d_cnew, alloc_size * sizeof(double));
        cudaMalloc(&d_mu, alloc_size * sizeof(double));

        // Allocate host array for initialization and gather
        std::vector<double> cold_host(alloc_size, 0.0);

        // Initialize concentration field
        if (rank == 0) printf("Initializing concentration field...\n");
        if (local_nz > 0) {
            initializeConcentration(cold_host.data(), nx, ny, local_nz, z_start, nz);
        }

        // Set halos for initial data
        clampHalosHost(cold_host.data(), nx, ny, local_nz, is_lower_global, is_upper_global);
        if (size > 1 && local_nz > 0) {
            exchangeHalosHost(cold_host.data(), nx, ny, local_nz, rank, size,
                            h_send_buf.data(), h_recv_buf.data());
        }

        // Copy to device
        cudaMemcpy(d_cold, cold_host.data(), alloc_size * sizeof(double), cudaMemcpyHostToDevice);

        // CUDA kernel configuration
        dim3 block(16, 16, 1);
        dim3 grid((unsigned int)((nx + block.x - 1) / block.x),
                  (unsigned int)((ny + block.y - 1) / block.y),
                  (unsigned int)local_nz);
        dim3 halo_block(16, 16, 1);
        dim3 halo_grid((unsigned int)((nx + halo_block.x - 1) / halo_block.x),
                       (unsigned int)((ny + halo_block.y - 1) / halo_block.y));

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            // 1. Fill halos of cold
            if (size == 1) {
                clampHalosKernel<<<halo_grid, halo_block>>>(d_cold, nx, ny, local_nz, true, true);
            } else {
                // Clamp global boundaries
                clampHalosKernel<<<halo_grid, halo_block>>>(d_cold, nx, ny, local_nz,
                                                            is_lower_global, is_upper_global);
                // MPI exchange
                exchangeHalosGPU(d_cold, nx, ny, local_nz, rank, size,
                               h_send_buf.data(), h_recv_buf.data());
            }

            // 2. Compute chemical potential
            computeChemicalPotentialKernel<<<grid, block>>>(
                d_cold, d_mu, nx, ny, local_nz, dx, dy, dz,
                gamma_val, e_AA, e_BB, e_AB);

            // 3. Fill halos of mu
            if (size == 1) {
                clampHalosKernel<<<halo_grid, halo_block>>>(d_mu, nx, ny, local_nz, true, true);
            } else {
                clampHalosKernel<<<halo_grid, halo_block>>>(d_mu, nx, ny, local_nz,
                                                            is_lower_global, is_upper_global);
                exchangeHalosGPU(d_mu, nx, ny, local_nz, rank, size,
                               h_send_buf.data(), h_recv_buf.data());
            }

            // 4. Compute update
            cahnHilliardUpdateKernel<<<grid, block>>>(
                d_cnew, d_cold, d_mu, nx, ny, local_nz, D, dt, dx, dy, dz);

            // 5. Swap pointers
            double* temp = d_cold;
            d_cold = d_cnew;
            d_cnew = temp;
        }

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        // Copy result back to host (only local data, skip lower halo)
        std::vector<double> local_result(local_nz * nxny);
        if (local_nz > 0) {
            cudaMemcpy(local_result.data(), d_cold + nxny,
                       local_nz * nxny * sizeof(double), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_cold);
        cudaFree(d_cnew);
        cudaFree(d_mu);

        // Gather results to rank 0
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            size_t zs = i * z_per_rank;
            size_t lnz = (i == size - 1) ? (nz - zs) : z_per_rank;
            recvcounts[i] = (int)(lnz * nxny);
            displs[i] = (int)(zs * nxny);
        }

        std::vector<double> global_cold;
        if (rank == 0) global_cold.resize(global_gridSize);

        MPI_Gatherv(local_result.data(), (int)(local_nz * nxny), MPI_DOUBLE,
                    global_cold.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            long long duration_ms = duration.count();
            printf("Computation time: %lld ms\n", duration_ms);

            double cellUpdates = (double)global_gridSize * iterations;
            double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);

            if (printRes) {
                print_results(global_cold, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_cold, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }

    } else {
        // ========================================
        // CPU PATH with OpenMP
        // ========================================
        std::vector<double> cold(alloc_size, 0.0);
        std::vector<double> cnew(alloc_size, 0.0);
        std::vector<double> mu(alloc_size, 0.0);

        if (rank == 0) printf("Initializing concentration field...\n");
        if (local_nz > 0) {
            initializeConcentration(cold.data(), nx, ny, local_nz, z_start, nz);
        }

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            // 1. Fill halos of cold
            clampHalosHost(cold.data(), nx, ny, local_nz, is_lower_global, is_upper_global);
            if (size > 1 && local_nz > 0) {
                exchangeHalosHost(cold.data(), nx, ny, local_nz, rank, size,
                                h_send_buf.data(), h_recv_buf.data());
            }

            // 2. Compute chemical potential
            if (local_nz > 0) {
                computeChemicalPotentialCPU(cold.data(), mu.data(), nx, ny, local_nz,
                                           dx, dy, dz, gamma_val, e_AA, e_BB, e_AB);
            }

            // 3. Fill halos of mu
            clampHalosHost(mu.data(), nx, ny, local_nz, is_lower_global, is_upper_global);
            if (size > 1 && local_nz > 0) {
                exchangeHalosHost(mu.data(), nx, ny, local_nz, rank, size,
                                h_send_buf.data(), h_recv_buf.data());
            }

            // 4. Compute update
            if (local_nz > 0) {
                cahnHilliardUpdateCPU(cnew.data(), cold.data(), mu.data(), nx, ny, local_nz,
                                     D, dt, dx, dy, dz);
            }

            // 5. Swap
            std::swap(cold, cnew);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        // Extract local data (skip lower halo)
        std::vector<double> local_result(local_nz * nxny);
        if (local_nz > 0) {
            std::memcpy(local_result.data(), cold.data() + nxny, local_nz * nxny * sizeof(double));
        }

        // Gather results to rank 0
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            size_t zs = i * z_per_rank;
            size_t lnz = (i == size - 1) ? (nz - zs) : z_per_rank;
            recvcounts[i] = (int)(lnz * nxny);
            displs[i] = (int)(zs * nxny);
        }

        std::vector<double> global_cold;
        if (rank == 0) global_cold.resize(global_gridSize);

        MPI_Gatherv(local_result.data(), (int)(local_nz * nxny), MPI_DOUBLE,
                    global_cold.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            long long duration_ms = duration.count();
            printf("Computation time: %lld ms\n", duration_ms);

            double cellUpdates = (double)global_gridSize * iterations;
            double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);

            if (printRes) {
                print_results(global_cold, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_cold, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
