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

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)

// 3D index calculation (global)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local indexing with ghost layers: z in [0, local_nz-1], where 0 and local_nz-1 are ghosts
inline size_t idx3_local(const size_t x, const size_t y, const size_t z_local, const size_t nx, const size_t ny, const size_t local_nz) noexcept {
    return z_local * (nx * ny) + y * nx + x;
}

// Device-side idx
__device__ inline size_t d_idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__device__ double d_computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz_local,
                                     const double dx, const double dy, const double dz,
                                     const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz_local - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[d_idx3(xp, y, z, nx, ny)] + c[d_idx3(xn, y, z, nx, ny)] - 2.0 * c[d_idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[d_idx3(x, yp, z, nx, ny)] + c[d_idx3(x, yn, z, nx, ny)] - 2.0 * c[d_idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[d_idx3(x, y, zp, nx, ny)] + c[d_idx3(x, y, zn, nx, ny)] - 2.0 * c[d_idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Kernel to compute chemical potential on device (operates on local array including ghosts but writes mu only for inner region)
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                                               const size_t nx, const size_t ny, const size_t nz_local,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                               const size_t z_offset) {
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t inner_nz = nz_local - 2; // exclude ghosts
    const size_t inner_size = nx * ny * inner_nz;
    if (gid >= inner_size) return;

    const size_t iz = gid / (nx * ny);
    const size_t rem = gid % (nx * ny);
    const size_t iy = rem / nx;
    const size_t ix = rem % nx;

    const size_t z_local = iz + 1; // shift by one due to ghost
    const size_t idx = d_idx3(ix, iy, z_local, nx, ny);
    const double cv = c[idx];

    double lap = d_computeLaplacian(c, nx, ny, nz_local, dx, dy, dz, ix, iy, z_local);
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv - gamma * lap;
}

// Kernel to update concentration on device (writes inner region)
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                         const size_t nx, const size_t ny, const size_t nz_local,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz) {
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t inner_nz = nz_local - 2;
    const size_t inner_size = nx * ny * inner_nz;
    if (gid >= inner_size) return;

    const size_t iz = gid / (nx * ny);
    const size_t rem = gid % (nx * ny);
    const size_t iy = rem / nx;
    const size_t ix = rem % nx;

    const size_t z_local = iz + 1;
    const size_t idx = d_idx3(ix, iy, z_local, nx, ny);

    double lap_mu = d_computeLaplacian(mu, nx, ny, nz_local, dx, dy, dz, ix, iy, z_local);
    cnew[idx] = cold[idx] + dt * D * lap_mu;
}

// Host-side exchange of ghost layers for a local array
void exchangeGhostLayers(std::vector<double>& data, const size_t nx, const size_t ny, const size_t local_nz,
                         int rank, int size, MPI_Comm comm, MPI_Datatype layer_type, std::vector<double>& temp_recv) {
    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // send first real layer (z=1) to prev, receive into ghost at z=0 from prev
    // send last real layer (z=local_nz-2) to next, receive into ghost at z=local_nz-1 from next
    MPI_Status st[4];
    MPI_Request reqs[4];

    // Send to prev, recv from next
    MPI_Isend(&data[idx3_local(0,0,1,nx,ny,local_nz)], 1, layer_type, prev, 0, comm, &reqs[0]);
    MPI_Irecv(&data[idx3_local(0,0,local_nz-1,nx,ny,local_nz)], 1, layer_type, next, 0, comm, &reqs[1]);

    // Send to next, recv from prev
    MPI_Isend(&data[idx3_local(0,0,local_nz-2,nx,ny,local_nz)], 1, layer_type, next, 1, comm, &reqs[2]);
    MPI_Irecv(&data[idx3_local(0,0,0,nx,ny,local_nz)], 1, layer_type, prev, 1, comm, &reqs[3]);

    MPI_Waitall(4, reqs, st);

    // Handle clamped BC for edges (MPI_PROC_NULL results in no receive; ensure ghost copies clamped)
    if (prev == MPI_PROC_NULL) {
        // copy first real layer into ghost 0
        std::memcpy(&data[idx3_local(0,0,0,nx,ny,local_nz)], &data[idx3_local(0,0,1,nx,ny,local_nz)], sizeof(double) * nx * ny);
    }
    if (next == MPI_PROC_NULL) {
        // copy last real layer into ghost last
        std::memcpy(&data[idx3_local(0,0,local_nz-1,nx,ny,local_nz)], &data[idx3_local(0,0,local_nz-2,nx,ny,local_nz)], sizeof(double) * nx * ny);
    }
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0 for printing help)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
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
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t global_vol = nx * ny * nz;

    // Decompose along Z dimension
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t local_nz_main = base + (rank < (int)rem ? 1 : 0);
    size_t z_start = (base * rank) + std::min((size_t)rank, rem);

    // local array includes 2 ghost layers
    size_t local_nz = local_nz_main + 2;
    size_t local_size = nx * ny * local_nz;

    // Allocate local arrays with ghost layers
    std::vector<double> cold_local(local_size);
    std::vector<double> cnew_local(local_size);
    std::vector<double> mu_local(local_size);

    // Initialize local concentration based on global linear id to match original
    for (size_t zz = 0; zz < local_nz_main; ++zz) {
        size_t gz = z_start + zz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx_local = idx3_local(x, y, zz + 1, nx, ny, local_nz);
                size_t linear_id = gz * (nx * ny) + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
                cold_local[idx_local] = -1.0 + 2.0 * pseudo;
            }
        }
    }
    // initialize ghosts with clamped BC copies
    if (local_nz_main > 0) {
        std::memcpy(&cold_local[idx3_local(0,0,0,nx,ny,local_nz)], &cold_local[idx3_local(0,0,1,nx,ny,local_nz)], sizeof(double)*nx*ny);
        std::memcpy(&cold_local[idx3_local(0,0,local_nz-1,nx,ny,local_nz)], &cold_local[idx3_local(0,0,local_nz-2,nx,ny,local_nz)], sizeof(double)*nx*ny);
    }

    // Create MPI datatype for a layer (nx*ny contiguous doubles)
    MPI_Datatype layer_type;
    MPI_Type_contiguous(nx * ny, MPI_DOUBLE, &layer_type);
    MPI_Type_commit(&layer_type);

    // Prepare device buffers
    double *d_c = nullptr, *d_mu = nullptr, *d_cnew = nullptr;
    CUDA_CHECK(cudaSetDevice(0)); // assume one GPU per node/rank for simplicity
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(double) * local_size));
    CUDA_CHECK(cudaMalloc(&d_mu, sizeof(double) * local_size));
    CUDA_CHECK(cudaMalloc(&d_cnew, sizeof(double) * local_size));

    // Start timing (global)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = 0.0;
    if (rank == 0) t_start = MPI_Wtime();

    const size_t inner_nz = local_nz - 2;
    const size_t inner_size = nx * ny * inner_nz;
    const size_t threads = 256;
    const size_t blocks = (inner_size + threads - 1) / threads;

    std::vector<double> temp_recv(nx * ny);

    for (int it = 0; it < iterations; ++it) {
        // Exchange ghost layers for cold_local
        exchangeGhostLayers(cold_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD, layer_type, temp_recv);

        // Copy cold_local to device
        CUDA_CHECK(cudaMemcpy(d_c, cold_local.data(), sizeof(double) * local_size, cudaMemcpyHostToDevice));

        // Compute chemical potential on device (writes mu for inner region)
        computeChemicalPotentialKernel<<<blocks, threads>>>(d_c, d_mu, nx, ny, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB, z_start);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy mu back to host and exchange ghost layers for mu
        CUDA_CHECK(cudaMemcpy(mu_local.data(), d_mu, sizeof(double) * local_size, cudaMemcpyDeviceToHost));
        exchangeGhostLayers(mu_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD, layer_type, temp_recv);
        CUDA_CHECK(cudaMemcpy(d_mu, mu_local.data(), sizeof(double) * local_size, cudaMemcpyHostToDevice));

        // Update concentration on device
        cahnHilliardUpdateKernel<<<blocks, threads>>>(d_cnew, d_c, d_mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy cnew back
        CUDA_CHECK(cudaMemcpy(cnew_local.data(), d_cnew, sizeof(double) * local_size, cudaMemcpyDeviceToHost));

        // Swap cold_local and cnew_local
        cold_local.swap(cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = 0.0;
    if (rank == 0) t_end = MPI_Wtime();

    if (rank == 0) {
        double duration = (t_end - t_start) * 1000.0; // ms
        printf("Computation time: %ld ms\n", (long)duration);

        double cellUpdates = (double) (nx * ny * nz) * iterations;
        double mcups = cellUpdates / ((t_end - t_start)) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for printing/validation
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::vector<double> gathered;
    for (int r = 0; r < size; ++r) {
        size_t local_nz_r = base + (r < (int)rem ? 1 : 0);
        recvcounts[r] = (int)(local_nz_r * nx * ny);
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];

    if (rank == 0) gathered.resize(nx * ny * nz);

    // Prepare send buffer without ghosts
    std::vector<double> sendbuf(local_nz_main * nx * ny);
    for (size_t zz = 0; zz < local_nz_main; ++zz) {
        size_t base_idx = zz * (nx * ny);
        size_t local_idx = idx3_local(0,0,zz+1,nx,ny,local_nz);
        std::memcpy(&sendbuf[base_idx], &cold_local[local_idx], sizeof(double)*nx*ny);
    }

    MPI_Gatherv(sendbuf.data(), (int)sendbuf.size(), MPI_DOUBLE,
                gathered.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        print_results(gathered, "Concentration");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = true;
        // Check for NaN/Inf and range
        for (const auto &val : gathered) {
            if (std::isnan(val) || std::isinf(val)) { valid = false; break; }
        }
        double minVal = *std::min_element(gathered.begin(), gathered.end());
        double maxVal = *std::max_element(gathered.begin(), gathered.end());
        printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
        if (maxVal > 10.0 || minVal < -10.0) valid = false;
        if (valid) { printf("Validation: PASSED\n"); } else { printf("Validation: FAILED\n"); }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    MPI_Type_free(&layer_type);

    MPI_Finalize();
    return 0;
}
