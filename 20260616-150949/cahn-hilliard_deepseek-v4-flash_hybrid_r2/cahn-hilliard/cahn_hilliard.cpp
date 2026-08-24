#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_err));             \
        MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                      \
} while(0)

// ─── CUDA Kernels ─────────────────────────────────────────────────────────

// Compute chemical potential mu from concentration c.
// The array layout includes a bottom ghost (z-index 0) and a top ghost
// (z-index nz_local+1) which are pre-filled via MPI exchange or clamped BC.
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c,
    double* __restrict__ mu,
    const int nx, const int ny, const int nz_local,
    const double dx, const double dy, const double dz,
    const double gamma,
    const double e_AA, const double e_BB, const double e_AB,
    const int nx_ny)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz_local) return;

    const int az = z + 1; // offset past bottom ghost
    const int idx = az * nx_ny + y * nx + x;
    const double cv = c[idx];

    // Clamped boundary conditions in X and Y
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const double lap =
        (c[az * nx_ny + y * nx + xp] +
         c[az * nx_ny + y * nx + xn] - 2.0 * cv) / (dx * dx)
      + (c[az * nx_ny + yp * nx + x] +
         c[az * nx_ny + yn * nx + x] - 2.0 * cv) / (dy * dy)
      + (c[(az + 1) * nx_ny + y * nx + x] +
         c[(az - 1) * nx_ny + y * nx + x] - 2.0 * cv) / (dz * dz);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * lap;
}

// Cahn-Hilliard update: cnew = cold + dt * D * laplacian(mu)
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew,
    const double* __restrict__ cold,
    const double* __restrict__ mu,
    const int nx, const int ny, const int nz_local,
    const double D, const double dt,
    const double dx, const double dy, const double dz,
    const int nx_ny)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz_local) return;

    const int az = z + 1;
    const int idx = az * nx_ny + y * nx + x;

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const double mv = mu[idx];
    const double lap_mu =
        (mu[az * nx_ny + y * nx + xp] +
         mu[az * nx_ny + y * nx + xn] - 2.0 * mv) / (dx * dx)
      + (mu[az * nx_ny + yp * nx + x] +
         mu[az * nx_ny + yn * nx + x] - 2.0 * mv) / (dy * dy)
      + (mu[(az + 1) * nx_ny + y * nx + x] +
         mu[(az - 1) * nx_ny + y * nx + x] - 2.0 * mv) / (dz * dz);

    cnew[idx] = cold[idx] + dt * D * lap_mu;
}

// ─── Host Structures and Helpers ──────────────────────────────────────────

// Pinned host buffers for MPI ghost-exchange staging
struct PinnedBuffers {
    double *send_down, *send_up;
    double *recv_down, *recv_up;
};

static void allocatePinnedBuffers(PinnedBuffers& buf, int nx_ny) {
    const size_t sz = (size_t)nx_ny * sizeof(double);
    CUDA_CHECK(cudaHostAlloc(&buf.send_down, sz, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&buf.send_up,   sz, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&buf.recv_down, sz, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&buf.recv_up,   sz, cudaHostAllocDefault));
}

static void freePinnedBuffers(PinnedBuffers& buf) {
    cudaFreeHost(buf.send_down);
    cudaFreeHost(buf.send_up);
    cudaFreeHost(buf.recv_down);
    cudaFreeHost(buf.recv_up);
}

// Exchange one ghost layer along the Z axis between neighboring MPI ranks.
// Global-boundary ranks apply clamped BC. Uses non-blocking MPI.
static void exchangeGhosts(
    double* d_arr, int nx, int ny, int nz_local,
    int rank, int nprocs, MPI_Comm comm,
    PinnedBuffers& buf, cudaStream_t stream)
{
    const int nx_ny = nx * ny;
    const size_t plane_bytes = (size_t)nx_ny * sizeof(double);

    // Copy boundary planes from device to host (async on stream)
    CUDA_CHECK(cudaMemcpyAsync(buf.send_up,
        d_arr + (size_t)nz_local * nx_ny, // last interior plane
        plane_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.send_down,
        d_arr + (size_t)1 * nx_ny, // first interior plane
        plane_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Request reqs[4];
    int nreqs = 0;

    // Post non-blocking receives
    if (rank > 0)
        MPI_Irecv(buf.recv_down, nx_ny, MPI_DOUBLE, rank - 1, 1, comm, &reqs[nreqs++]);
    if (rank < nprocs - 1)
        MPI_Irecv(buf.recv_up,   nx_ny, MPI_DOUBLE, rank + 1, 0, comm, &reqs[nreqs++]);

    // Post non-blocking sends
    if (rank > 0)
        MPI_Isend(buf.send_down, nx_ny, MPI_DOUBLE, rank - 1, 0, comm, &reqs[nreqs++]);
    if (rank < nprocs - 1)
        MPI_Isend(buf.send_up,   nx_ny, MPI_DOUBLE, rank + 1, 1, comm, &reqs[nreqs++]);

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Copy received data (or clamped) to device ghost layers
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_arr + 0, buf.recv_down,
                                   plane_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // Clamped: bottom ghost = first interior plane
        CUDA_CHECK(cudaMemcpyAsync(d_arr + 0, d_arr + (size_t)1 * nx_ny,
                                   plane_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    if (rank < nprocs - 1) {
        CUDA_CHECK(cudaMemcpyAsync(d_arr + (size_t)(nz_local + 1) * nx_ny, buf.recv_up,
                                   plane_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // Clamped: top ghost = last interior plane
        CUDA_CHECK(cudaMemcpyAsync(d_arr + (size_t)(nz_local + 1) * nx_ny,
                                   d_arr + (size_t)nz_local * nx_ny,
                                   plane_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Initialize local portion of concentration field using the same
// deterministic pseudo-random sequence as the original serial code.
// Uses OpenMP for host-side parallel initialization.
static void initializeConcentrationHost(
    double* h_c, int nx, int ny, int nz_local, int offset_z, int nz_global)
{
    const size_t nx_ny = (size_t)nx * ny;
    const size_t vol   = nx_ny * (size_t)nz_global;

    #pragma omp parallel for collapse(3)
    for (int z = 0; z < nz_local; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const size_t idx = (size_t)(z + 1) * nx_ny + (size_t)y * nx + (size_t)x;
                const size_t global_z = (size_t)(offset_z + z);
                const size_t linear_id = global_z * nx_ny + (size_t)y * nx + (size_t)x;
                const double pseudo =
                    (double)(((linear_id + 1) * 1299709) % vol) / (double)vol;
                h_c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validate the local concentration data using OpenMP for min/max
// and MPI_Allreduce to combine results across ranks.
static bool validateResultLocal(
    const double* h_c, int nx, int ny, int nz_local,
    double& global_min, double& global_max, int& has_nan_inf)
{
    const int nx_ny = nx * ny;
    double lmin = +1e100, lmax = -1e100;
    int lnan = 0;

    #pragma omp parallel for reduction(min:lmin) reduction(max:lmax) reduction(|:lnan)
    for (int z = 0; z < nz_local; ++z) {
        const int base = (z + 1) * nx_ny;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const double val = h_c[base + y * nx + x];
                if (val < lmin) lmin = val;
                if (val > lmax) lmax = val;
                if (isnan(val) || isinf(val)) lnan = 1;
            }
        }
    }

    MPI_Allreduce(&lmin, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lmax, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lnan, &has_nan_inf, 1, MPI_INT,   MPI_LOR,  MPI_COMM_WORLD);

    if (has_nan_inf) return false;
    if (global_max > 10.0 || global_min < -10.0) return false;
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

// ─── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // CUDA device selection: round-robin across available GPUs
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA-capable device found on rank %d\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int dev_id = mpi_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(dev_id));

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev_id);

    if (mpi_rank == 0) {
        printf("Hybrid MPI+OpenMP+CUDA Cahn-Hilliard Benchmark\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("GPU: %s (device %d, CC %d.%d)\n",
               dev_prop.name, dev_id, dev_prop.major, dev_prop.minor);
    }

    // ── Parse command-line arguments ────────────────────────────────
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // ── MPI domain decomposition along Z ────────────────────────────
    const int base_nz = (int)nz / mpi_size;
    const int rem     = (int)nz % mpi_size;
    const int nz_local = base_nz + (mpi_rank < rem ? 1 : 0);

    int offset_z = 0;
    for (int i = 0; i < mpi_rank; ++i)
        offset_z += base_nz + (i < rem ? 1 : 0);

    if (nz_local == 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "Error: domain too small for %d MPI ranks\n", mpi_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int nx_int    = (int)nx;
    const int ny_int    = (int)ny;
    const int nx_ny     = nx_int * ny_int;
    const int alloc_z   = nz_local + 2; // interior + 2 ghost layers
    const size_t alloc_size = (size_t)nx_ny * alloc_z;

    if (mpi_rank == 0) {
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Local Z planes per rank: %d\n", nz_local);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters (identical to original)
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB =  (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    // ── Allocate memory ─────────────────────────────────────────────
    // Host array (includes ghost layers to match device layout)
    std::vector<double> h_c(alloc_size);

    // Pinned host buffers for MPI ghost-exchange staging
    PinnedBuffers pinned{};
    allocatePinnedBuffers(pinned, nx_ny);

    // Device memory
    double *d_c = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_c,    alloc_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_cnew, alloc_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_mu,   alloc_size * sizeof(double)));

    // CUDA stream for serializing async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Initialize concentration on host (OpenMP), copy to device ───
    if (mpi_rank == 0) printf("Initializing concentration field...\n");
    initializeConcentrationHost(h_c.data(), nx_int, ny_int, nz_local, offset_z, (int)nz);

    // Copy interior planes to device (skip ghost area)
    CUDA_CHECK(cudaMemcpyAsync(d_c + (size_t)nx_ny, h_c.data() + (size_t)nx_ny,
                               (size_t)nx_ny * nz_local * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Initial ghost exchange to fill boundary + MPI ghost layers
    exchangeGhosts(d_c, nx_int, ny_int, nz_local, mpi_rank, mpi_size,
                   MPI_COMM_WORLD, pinned, stream);

    // ── Simulation loop ─────────────────────────────────────────────
    if (mpi_rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    // Kernel launch configuration: 1024 threads per block (max for CC 8.6)
    const dim3 block(16, 16, 4);
    const dim3 grid((nx_int + 15) / 16, (ny_int + 15) / 16, (nz_local + 3) / 4);

    // Working pointers (swapped each iteration)
    double *d_c_cur = d_c;    // current concentration
    double *d_c_nxt = d_cnew; // next concentration

    for (int t = 0; t < iterations; ++t) {
        // 1. Exchange ghost layers for concentration field
        exchangeGhosts(d_c_cur, nx_int, ny_int, nz_local, mpi_rank, mpi_size,
                       MPI_COMM_WORLD, pinned, stream);

        // 2. Compute chemical potential on GPU
        computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
            d_c_cur, d_mu, nx_int, ny_int, nz_local,
            dx, dy, dz, gamma, e_AA, e_BB, e_AB, nx_ny);
        CUDA_CHECK(cudaGetLastError());

        // 3. Exchange ghost layers for chemical potential
        exchangeGhosts(d_mu, nx_int, ny_int, nz_local, mpi_rank, mpi_size,
                       MPI_COMM_WORLD, pinned, stream);

        // 4. Update concentration on GPU
        cahnHilliardUpdateKernel<<<grid, block, 0, stream>>>(
            d_c_nxt, d_c_cur, d_mu, nx_int, ny_int, nz_local,
            D, dt, dx, dy, dz, nx_ny);
        CUDA_CHECK(cudaGetLastError());

        // 5. Swap pointers for next iteration
        std::swap(d_c_cur, d_c_nxt);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        const double total_cells = (double)nx * ny * nz * iterations;
        const double mcups = total_cells / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ── Copy final concentration back to host ───────────────────────
    // d_c_cur holds the concentration at the end of the last iteration
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c_cur, alloc_size * sizeof(double),
                           cudaMemcpyDeviceToHost));

    // ── Validation (OpenMP + MPI_Allreduce) ─────────────────────────
    if (validate) {
        double gmin = 0.0, gmax = 0.0;
        int has_nan = 0;
        const bool valid = validateResultLocal(h_c.data(), nx_int, ny_int, nz_local,
                                                gmin, gmax, has_nan);
        if (mpi_rank == 0) {
            printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
            printf("Validating result...\n");
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    // ── Print results for external validation (gather to rank 0) ────
    if (printResults) {
        std::vector<int> recvcounts(mpi_size);
        std::vector<int> displs(mpi_size);
        int offset = 0;
        for (int i = 0; i < mpi_size; ++i) {
            const int nz_i = base_nz + (i < rem ? 1 : 0);
            recvcounts[i] = nx_ny * nz_i;
            displs[i] = offset;
            offset += recvcounts[i];
        }

        const int local_size = nx_ny * nz_local;
        std::vector<double> local_flat(local_size);
        #pragma omp parallel for
        for (int z = 0; z < nz_local; ++z) {
            memcpy(&local_flat[(size_t)z * nx_ny],
                   &h_c[(size_t)(z + 1) * nx_ny],
                   (size_t)nx_ny * sizeof(double));
        }

        std::vector<double> full_c;
        if (mpi_rank == 0) full_c.resize((size_t)nx * ny * nz);

        MPI_Gatherv(local_flat.data(), local_size, MPI_DOUBLE,
                    mpi_rank == 0 ? full_c.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            print_results(full_c, "Concentration");
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    freePinnedBuffers(pinned);

    MPI_Finalize();
    return 0;
}
