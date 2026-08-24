#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int mpi_err = (call); \
    if (mpi_err != MPI_SUCCESS) { \
        char err_str[MPI_MAX_ERROR_STRING]; \
        int resultlen; \
        MPI_Error_string(mpi_err, err_str, &resultlen); \
        fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, err_str); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Block dimensions for stencil kernels
#define BX 32
#define BY 8
#define SMEM_W (BX + 2)
#define SMEM_H (BY + 2)
#define SMEM_SIZE (SMEM_W * SMEM_H)

// Initialize concentration field on GPU
__global__ void initializeConcentrationKernel(double* __restrict__ c,
    size_t nx, size_t ny, size_t local_nz, size_t z_offset, size_t vol)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= local_nz) return;

    size_t global_z = z + z_offset;
    size_t linear_id = global_z * (nx * ny) + y * nx + x;
    double pseudo = ((((linear_id + 1) * 1299709) % vol) / (double)vol);

    size_t lz = z + 1; // ghost zone offset
    c[lz * nx * ny + y * nx + x] = -1.0 + 2.0 * pseudo;
}

// Compute chemical potential with shared memory for X-Y stencil
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c,
    double* __restrict__ mu,
    size_t nx, size_t ny, size_t local_nz,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    __shared__ double smem[SMEM_SIZE];

    const size_t plane = nx * ny;

    size_t x = blockIdx.x * BX + threadIdx.x;
    size_t y = blockIdx.y * BY + threadIdx.y;
    size_t z = blockIdx.z;
    size_t lz = z + 1;

    // Load shared memory tile with halos for X-Y
    size_t tid = threadIdx.y * BX + threadIdx.x;
    size_t num_threads = BX * BY;

    int base_gx = (int)(blockIdx.x * BX) - 1;
    int base_gy = (int)(blockIdx.y * BY) - 1;
    int max_gx = (int)nx - 1;
    int max_gy = (int)ny - 1;

    for (size_t i = tid; i < SMEM_SIZE; i += num_threads) {
        int sy = (int)(i / SMEM_W);
        int sx = (int)(i % SMEM_W);
        int gx = base_gx + sx;
        int gy = base_gy + sy;
        gx = max(0, min(gx, max_gx));
        gy = max(0, min(gy, max_gy));
        smem[i] = c[lz * plane + (size_t)gy * nx + (size_t)gx];
    }
    __syncthreads();

    if (x >= nx || y >= ny) return;

    size_t lx = threadIdx.x + 1;
    size_t ly = threadIdx.y + 1;
    size_t center = lz * plane + y * nx + x;

    double cv = smem[ly * SMEM_W + lx];

    // X-Y stencil from shared memory
    double cxx = (smem[ly * SMEM_W + lx + 1] + smem[ly * SMEM_W + lx - 1] - 2.0 * cv) * inv_dx2;
    double cyy = (smem[(ly + 1) * SMEM_W + lx] + smem[(ly - 1) * SMEM_W + lx] - 2.0 * cv) * inv_dy2;

    // Z stencil from global memory
    double czz = (c[(lz + 1) * plane + y * nx + x] + c[(lz - 1) * plane + y * nx + x] - 2.0 * cv) * inv_dz2;

    mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * (cxx + cyy + czz);
}

// Cahn-Hilliard update with shared memory for X-Y stencil
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew,
    const double* __restrict__ cold,
    const double* __restrict__ mu,
    size_t nx, size_t ny, size_t local_nz,
    double dt_D, double inv_dx2, double inv_dy2, double inv_dz2)
{
    __shared__ double smem[SMEM_SIZE];

    const size_t plane = nx * ny;

    size_t x = blockIdx.x * BX + threadIdx.x;
    size_t y = blockIdx.y * BY + threadIdx.y;
    size_t z = blockIdx.z;
    size_t lz = z + 1;

    // Load shared memory tile for mu
    size_t tid = threadIdx.y * BX + threadIdx.x;
    size_t num_threads = BX * BY;

    int base_gx = (int)(blockIdx.x * BX) - 1;
    int base_gy = (int)(blockIdx.y * BY) - 1;
    int max_gx = (int)nx - 1;
    int max_gy = (int)ny - 1;

    for (size_t i = tid; i < SMEM_SIZE; i += num_threads) {
        int sy = (int)(i / SMEM_W);
        int sx = (int)(i % SMEM_W);
        int gx = base_gx + sx;
        int gy = base_gy + sy;
        gx = max(0, min(gx, max_gx));
        gy = max(0, min(gy, max_gy));
        smem[i] = mu[lz * plane + (size_t)gy * nx + (size_t)gx];
    }
    __syncthreads();

    if (x >= nx || y >= ny) return;

    size_t lx = threadIdx.x + 1;
    size_t ly = threadIdx.y + 1;
    size_t center = lz * plane + y * nx + x;

    double muv = smem[ly * SMEM_W + lx];

    // X-Y stencil from shared memory
    double muxx = (smem[ly * SMEM_W + lx + 1] + smem[ly * SMEM_W + lx - 1] - 2.0 * muv) * inv_dx2;
    double muyy = (smem[(ly + 1) * SMEM_W + lx] + smem[(ly - 1) * SMEM_W + lx] - 2.0 * muv) * inv_dy2;

    // Z stencil from global memory
    double muzz = (mu[(lz + 1) * plane + y * nx + x] + mu[(lz - 1) * plane + y * nx + x] - 2.0 * muv) * inv_dz2;

    cnew[center] = cold[center] + dt_D * (muxx + muyy + muzz);
}

// Fill ghost zones for a device array (1-layer halos along Z)
void fillGhostZones(double* d_data, double* h_send, double* h_recv,
                    size_t nx, size_t ny, size_t local_nz,
                    int rank, int num_ranks)
{
    size_t slice_elems = nx * ny;
    size_t slice_bytes = slice_elems * sizeof(double);

    // Bottom ghost zone
    if (rank > 0) {
        // Copy first data slice (z=1) to host send buffer
        CUDA_CHECK(cudaMemcpy(h_send, d_data + slice_elems, slice_bytes, cudaMemcpyDeviceToHost));
        // Send first data down, receive bottom ghost from below
        MPI_Sendrecv(h_send, (int)slice_elems, MPI_DOUBLE, rank - 1, 0,
                     h_recv, (int)slice_elems, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Copy received data to bottom ghost zone (z=0)
        CUDA_CHECK(cudaMemcpy(d_data, h_recv, slice_bytes, cudaMemcpyHostToDevice));
    } else {
        // Clamped BC: copy first data slice to bottom ghost
        CUDA_CHECK(cudaMemcpy(d_data, d_data + slice_elems, slice_bytes, cudaMemcpyDeviceToDevice));
    }

    // Top ghost zone
    if (rank < num_ranks - 1) {
        // Copy last data slice (z=local_nz) to host send buffer
        CUDA_CHECK(cudaMemcpy(h_send, d_data + local_nz * slice_elems, slice_bytes, cudaMemcpyDeviceToHost));
        // Send last data up, receive top ghost from above
        MPI_Sendrecv(h_send, (int)slice_elems, MPI_DOUBLE, rank + 1, 1,
                     h_recv, (int)slice_elems, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Copy received data to top ghost zone (z=local_nz+1)
        CUDA_CHECK(cudaMemcpy(d_data + (local_nz + 1) * slice_elems, h_recv, slice_bytes, cudaMemcpyHostToDevice));
    } else {
        // Clamped BC: copy last data slice to top ghost
        CUDA_CHECK(cudaMemcpy(d_data + (local_nz + 1) * slice_elems,
                              d_data + local_nz * slice_elems,
                              slice_bytes, cudaMemcpyDeviceToDevice));
    }
}

// Validate result with OpenMP parallelism
bool validateResult(const std::vector<double>& c, size_t /*nx*/, size_t /*ny*/, size_t /*nz*/) {
    const size_t N = c.size();

    bool has_nan_inf = false;
    #pragma omp parallel for reduction(||:has_nan_inf) schedule(static)
    for (size_t i = 0; i < N; i++) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            has_nan_inf = true;
        }
    }

    if (has_nan_inf) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < N; i++) {
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

int main(int argc, char** argv) {
    // MPI initialization
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Parse command line arguments (all ranks)
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Use MPI communicator (potentially reduced if nz < world_size)
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = world_rank;
    int num_ranks = world_size;

    // If more ranks than Z slices, reduce to nz active ranks
    if ((size_t)world_size > nz) {
        int color = ((size_t)world_rank < nz) ? 0 : 1;
        MPI_Comm sub_comm;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &sub_comm));
        if (color == 1) {
            MPI_Comm_free(&sub_comm);
            MPI_Finalize();
            return 0;
        }
        MPI_CHECK(MPI_Comm_rank(sub_comm, &rank));
        MPI_CHECK(MPI_Comm_size(sub_comm, &num_ranks));
        comm = sub_comm;
    }

    // Print header (rank 0 only)
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition along Z
    size_t base_nz = nz / num_ranks;
    size_t rem = nz % num_ranks;
    size_t z_start, local_nz;
    if ((size_t)rank < rem) {
        local_nz = base_nz + 1;
        z_start = (size_t)rank * (base_nz + 1);
    } else {
        local_nz = base_nz;
        z_start = rem * (base_nz + 1) + ((size_t)rank - rem) * base_nz;
    }

    // GPU setup - assign one GPU per local rank
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(comm, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % num_gpus));

    if (rank == 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, local_rank % num_gpus));
        printf("GPU: %s (%d SMs, %.0f MHz)\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma_val = 0.5;
    const double D = 1.0;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double dt_D = dt * D;

    size_t total_vol = nx * ny * nz;
    size_t slice_elems = nx * ny;
    size_t local_vol = nx * ny * (local_nz + 2); // including ghost zones

    // Allocate device memory (with ghost zones)
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_vol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_vol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_vol * sizeof(double)));

    // Zero-initialize ghost zones
    CUDA_CHECK(cudaMemset(d_cold, 0, local_vol * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, local_vol * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, local_vol * sizeof(double)));

    // Allocate pinned host buffers for halo exchange
    double *h_send, *h_recv;
    CUDA_CHECK(cudaHostAlloc(&h_send, slice_elems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv, slice_elems * sizeof(double), cudaHostAllocDefault));

    // Initialize concentration field on GPU
    if (rank == 0) printf("Initializing concentration field...\n");

    {
        dim3 init_block(16, 16, 4);
        dim3 init_grid((unsigned int)((nx + 15) / 16),
                       (unsigned int)((ny + 15) / 16),
                       (unsigned int)((local_nz + 3) / 4));
        initializeConcentrationKernel<<<init_grid, init_block>>>(
            d_cold, nx, ny, local_nz, z_start, total_vol);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    // Setup kernel launch parameters
    dim3 block(BX, BY);
    dim3 grid_dim((unsigned int)((nx + BX - 1) / BX),
                  (unsigned int)((ny + BY - 1) / BY),
                  (unsigned int)local_nz);

    // Synchronize all ranks before timing
    MPI_CHECK(MPI_Barrier(comm));
    double start_time = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Fill c ghost zones (MPI exchange + clamped BCs at global boundaries)
        fillGhostZones(d_cold, h_send, h_recv, nx, ny, local_nz, rank, num_ranks);

        // Compute chemical potential: mu = f(c) - gamma * laplacian(c)
        computeChemicalPotentialKernel<<<grid_dim, block>>>(
            d_cold, d_mu, nx, ny, local_nz,
            inv_dx2, inv_dy2, inv_dz2,
            gamma_val, e_AA, e_BB, e_AB);

        // Fill mu ghost zones
        fillGhostZones(d_mu, h_send, h_recv, nx, ny, local_nz, rank, num_ranks);

        // Update concentration: cnew = cold + dt*D * laplacian(mu)
        cahnHilliardUpdateKernel<<<grid_dim, block>>>(
            d_cnew, d_cold, d_mu, nx, ny, local_nz,
            dt_D, inv_dx2, inv_dy2, inv_dz2);

        // Swap cold and cnew pointers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(comm));
    double end_time = MPI_Wtime();
    double elapsed_ms = (end_time - start_time) * 1000.0;
    double global_elapsed_ms;
    MPI_CHECK(MPI_Reduce(&elapsed_ms, &global_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm));

    // Copy result from device to host (data region only, excluding ghost zones)
    size_t local_data_elems = nx * ny * local_nz;
    std::vector<double> local_result(local_data_elems);
    CUDA_CHECK(cudaMemcpy(local_result.data(), d_cold + slice_elems,
                          local_data_elems * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather all local_nz values for Gatherv
    std::vector<int> all_local_nz(num_ranks);
    int my_local_nz = (int)local_nz;
    MPI_CHECK(MPI_Allgather(&my_local_nz, 1, MPI_INT,
                            all_local_nz.data(), 1, MPI_INT, comm));

    // Compute receive counts and displacements
    std::vector<int> recv_counts(num_ranks);
    std::vector<int> displacements(num_ranks);
    int disp = 0;
    for (int i = 0; i < num_ranks; i++) {
        recv_counts[i] = all_local_nz[i] * (int)slice_elems;
        displacements[i] = disp;
        disp += recv_counts[i];
    }

    // Gather full result to rank 0
    std::vector<double> full_result;
    if (rank == 0) {
        full_result.resize(total_vol);
    }
    MPI_CHECK(MPI_Gatherv(local_result.data(), (int)local_data_elems, MPI_DOUBLE,
                          full_result.data(), recv_counts.data(), displacements.data(), MPI_DOUBLE,
                          0, comm));

    // Post-processing on rank 0
    int return_code = 0;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", global_elapsed_ms);

        double cellUpdates = (double)total_vol * iterations;
        double mcups = cellUpdates / (global_elapsed_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(full_result, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_result, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                return_code = 1;
            }
        }
    }

    // Broadcast return code to all ranks
    MPI_CHECK(MPI_Bcast(&return_code, 1, MPI_INT, 0, comm));

    // Cleanup
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));

    if (comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&comm);
    }
    MPI_Finalize();
    return return_code;
}
