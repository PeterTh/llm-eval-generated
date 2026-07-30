#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Compute chemical potential kernel
// Local storage layout: [ghost_bottom(z=0) | data(z=1..local_nz) | ghost_top(z=local_nz+1)]
// Each thread handles one data point
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const size_t plane = nx * ny;
    const size_t total = nx * ny * local_nz;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t z_local = tid / (nx * ny);
    size_t rem     = tid % (nx * ny);
    size_t y       = rem / nx;
    size_t x       = rem % nx;

    size_t sz = z_local + 1; // storage z (data rows start at 1)

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0)      ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0)      ? y - 1 : 0;
    size_t zp = sz + 1;
    size_t zn = sz - 1;

    size_t ic  = sz  * plane + y * nx + x;
    double cv  = c[ic];

    double cxx = (c[sz * plane + y * nx + xp] + c[sz * plane + y * nx + xn] - 2.0 * cv) / (dx * dx);
    double cyy = (c[sz * plane + yp * nx + x] + c[sz * plane + yn * nx + x] - 2.0 * cv) / (dy * dy);
    double czz = (c[zp * plane + y * nx + x] + c[zn * plane + y * nx + x] - 2.0 * cv) / (dz * dz);

    mu[ic] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * (cxx + cyy + czz);
}

// Cahn-Hilliard update kernel
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double D, const double dt, const double dx, const double dy, const double dz)
{
    const size_t plane = nx * ny;
    const size_t total = nx * ny * local_nz;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t z_local = tid / (nx * ny);
    size_t rem     = tid % (nx * ny);
    size_t y       = rem / nx;
    size_t x       = rem % nx;

    size_t sz = z_local + 1;

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0)      ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0)      ? y - 1 : 0;
    size_t zp = sz + 1;
    size_t zn = sz - 1;

    size_t ic     = sz * plane + y * nx + x;
    double mu_c   = mu[ic];

    double muxx = (mu[sz * plane + y * nx + xp] + mu[sz * plane + y * nx + xn] - 2.0 * mu_c) / (dx * dx);
    double muyy = (mu[sz * plane + yp * nx + x] + mu[sz * plane + yn * nx + x] - 2.0 * mu_c) / (dy * dy);
    double muzz = (mu[zp * plane + y * nx + x] + mu[zn * plane + y * nx + x] - 2.0 * mu_c) / (dz * dz);

    cnew[ic] = cold[ic] + dt * D * (muxx + muyy + muzz);
}

// Initialize concentration field kernel
__global__ void initializeConcentrationKernel(
    double* __restrict__ c,
    const size_t nx, const size_t ny, const size_t local_nz,
    const size_t z_offset, const size_t vol)
{
    const size_t plane = nx * ny;
    const size_t total = nx * ny * local_nz;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (tid >= total) return;

    size_t z_local = tid / (nx * ny);
    size_t rem     = tid % (nx * ny);
    size_t y       = rem / nx;
    size_t x       = rem % nx;

    size_t sz = z_local + 1; // storage z

    size_t global_z = z_offset + z_local;
    size_t global_linear_id = global_z * (nx * ny) + y * nx + x;
    double pseudo = ((((global_linear_id + 1) * 1299709) % vol) / (double)vol);
    c[sz * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
}

// Set clamped ghost layers for boundary ranks
__global__ void setClampedGhostKernel(
    double* __restrict__ data,
    const size_t nx, const size_t ny, const size_t local_nz,
    const int is_bottom, const int is_top)
{
    const size_t plane = nx * ny;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (tid >= plane) return;

    if (is_bottom) {
        // Ghost at z=0 clamped to data at z=1
        data[0 * plane + tid] = data[1 * plane + tid];
    }
    if (is_top) {
        // Ghost at z=local_nz+1 clamped to data at z=local_nz
        data[(local_nz + 1) * plane + tid] = data[local_nz * plane + tid];
    }
}

// Exchange ghost layers between MPI ranks
void exchangeGhostLayers(double* d_data, size_t nx, size_t ny, size_t local_nz,
                         int rank, int nprocs, int is_bottom, int is_top,
                         double* h_send_up, double* h_recv_up,
                         double* h_send_down, double* h_recv_down)
{
    const size_t plane = nx * ny;

    // Copy boundary data to host send buffers
    CUDA_CHECK(cudaMemcpy(h_send_up,   d_data + local_nz       * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_down, d_data + 1              * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));

    if (nprocs == 1) {
        // Single rank: copy boundary data to ghost layers on device
        CUDA_CHECK(cudaMemcpy(d_data + (local_nz + 1) * plane, h_send_up,   plane * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data + 0              * plane, h_send_down, plane * sizeof(double), cudaMemcpyHostToDevice));
        return;
    }

    int up_rank   = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
    int down_rank = (rank > 0)          ? rank - 1 : MPI_PROC_NULL;
    MPI_Status status;

    // Send top data up, receive bottom data from below
    MPI_Sendrecv(h_send_up,   (int)plane, MPI_DOUBLE, up_rank,   0,
                 h_recv_down, (int)plane, MPI_DOUBLE, down_rank, 0,
                 MPI_COMM_WORLD, &status);

    // Send bottom data down, receive top data from above
    MPI_Sendrecv(h_send_down, (int)plane, MPI_DOUBLE, down_rank, 1,
                 h_recv_up,   (int)plane, MPI_DOUBLE, up_rank,   1,
                 MPI_COMM_WORLD, &status);

    // Copy received ghost data to device
    if (down_rank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_data + 0              * plane, h_recv_down, plane * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (up_rank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_data + (local_nz + 1) * plane, h_recv_up,   plane * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Set clamped BC for boundary ranks
    int bs = 256;
    int nb = ((int)plane + bs - 1) / bs;
    setClampedGhostKernel<<<nb, bs>>>(d_data, nx, ny, local_nz, is_bottom, is_top);
    CUDA_CHECK(cudaDeviceSynchronize());
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Select GPU (round-robin)
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    // Rank 0 parses args
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)       nx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)  ny = atoi(argv[++i]);
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)  nz = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)  iterations = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)                   validate = true;
            else if (strcmp(argv[i], "-r") == 0)                   printResults = true;
            else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); MPI_Finalize(); return 0; }
            else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); MPI_Finalize(); return 1; }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    // Broadcast parameters
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int vi = validate ? 1 : 0, ri = printResults ? 1 : 0;
    MPI_Bcast(&vi, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ri, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = vi; printResults = ri;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs available: %d\n", nprocs, num_gpus);
    }

    // Z-domain decomposition
    size_t base_nz = nz / nprocs;
    size_t rem = nz % nprocs;
    size_t local_nz = base_nz + (size_t)(rank < (int)rem ? 1 : 0);
    size_t z_offset = 0;
    for (int r = 0; r < rank; ++r)
        z_offset += base_nz + (size_t)(r < (int)rem ? 1 : 0);

    int is_bottom = (rank == 0) ? 1 : 0;
    int is_top    = (rank == nprocs - 1) ? 1 : 0;

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma_val = 0.5;
    const double D = 1.0;

    const size_t plane = nx * ny;
    const size_t storage_size = plane * (local_nz + 2); // ghost + data + ghost

    // Device arrays
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, storage_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, storage_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,   storage_size * sizeof(double)));

    // Host MPI buffers
    std::vector<double> h_send_up(plane), h_recv_up(plane);
    std::vector<double> h_send_down(plane), h_recv_down(plane);

    // Initialize
    if (rank == 0) printf("Initializing concentration field...\n");
    size_t vol = nx * ny * nz;
    size_t total_pts = nx * ny * local_nz;
    int bs = 256;
    int nb = (total_pts + bs - 1) / bs;

    initializeConcentrationKernel<<<nb, bs>>>(d_cold, nx, ny, local_nz, z_offset, vol);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initial ghost exchange
    exchangeGhostLayers(d_cold, nx, ny, local_nz, rank, nprocs, is_bottom, is_top,
                        h_send_up.data(), h_recv_up.data(),
                        h_send_down.data(), h_recv_down.data());

    // Simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // 1. Compute chemical potential from cold
        computeChemicalPotentialKernel<<<nb, bs>>>(
            d_cold, d_mu, nx, ny, local_nz, dx, dy, dz,
            gamma_val, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Exchange ghost layers for mu (needed for Laplacian of mu in update)
        exchangeGhostLayers(d_mu, nx, ny, local_nz, rank, nprocs, is_bottom, is_top,
                            h_send_up.data(), h_recv_up.data(),
                            h_send_down.data(), h_recv_down.data());

        // 3. Update concentration
        cahnHilliardUpdateKernel<<<nb, bs>>>(
            d_cnew, d_cold, d_mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Swap pointers
        double* tmp = d_cold; d_cold = d_cnew; d_cnew = tmp;

        // 5. Exchange ghost layers for new cold
        exchangeGhostLayers(d_cold, nx, ny, local_nz, rank, nprocs, is_bottom, is_top,
                            h_send_up.data(), h_recv_up.data(),
                            h_send_down.data(), h_recv_down.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Report timing
    long long local_ms = duration.count(), max_ms;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        double cellUpdates = (double)nx * ny * nz * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0
    if (printResults || validate) {
        // Extract local data from device (skip bottom ghost)
        std::vector<double> h_storage(storage_size);
        CUDA_CHECK(cudaMemcpy(h_storage.data(), d_cold, storage_size * sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> h_local(local_nz * plane);
        #pragma omp parallel for schedule(static)
        for (long long z = 0; z < (long long)local_nz; ++z) {
            memcpy(h_local.data() + z * plane,
                   h_storage.data() + (z + 1) * plane,
                   plane * sizeof(double));
        }

        // Gather to rank 0
        std::vector<int> recv_counts(nprocs);
        int local_count = (int)(local_nz * plane);
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(nprocs, 0);
        if (rank == 0) {
            for (int i = 1; i < nprocs; ++i)
                displs[i] = displs[i-1] + recv_counts[i-1];
        }

        size_t total_size = nx * ny * nz;
        std::vector<double> h_global;
        if (rank == 0) h_global.resize(total_size);

        MPI_Gatherv(h_local.data(), local_count, MPI_DOUBLE,
                     h_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(h_global, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = true;
                double minVal = h_global[0], maxVal = h_global[0];
                #pragma omp parallel for reduction(min:minVal,max:maxVal) schedule(static)
                for (long long i = 0; i < (long long)total_size; ++i) {
                    if (std::isnan(h_global[i]) || std::isinf(h_global[i])) {
                        valid = false;
                    }
                    minVal = std::min(minVal, h_global[i]);
                    maxVal = std::max(maxVal, h_global[i]);
                }
                if (!valid) {
                    printf("Validation failed: found NaN or Inf value\n");
                } else {
                    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                    if (maxVal > 10.0 || minVal < -10.0) {
                        printf("Validation failed: values out of expected range\n");
                        valid = false;
                    }
                }
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            }
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return 0;
}
