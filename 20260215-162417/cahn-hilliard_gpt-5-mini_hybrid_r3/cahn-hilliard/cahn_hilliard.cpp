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

// Map 3D index to linear index (no halos)
inline __host__ __device__ size_t idx3_local(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device Laplacian using clamped boundaries within local buffer that includes 1-layer halos at z=0 and z=local_nz+1
__device__ double deviceLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t local_nz_with_halo,
                                  const double dx, const double dy, const double dz,
                                  const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z + 1 < local_nz_with_halo) ? z + 1 : z; // safe due to halos
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const size_t base = z * (nx * ny);
    const size_t idx = base + y * nx + x;
    const size_t idx_xp = base + y * nx + xp;
    const size_t idx_xn = base + y * nx + xn;
    const size_t idx_yp = base + yp * nx + x;
    const size_t idx_yn = base + yn * nx + x;
    const size_t idx_zp = (zp) * (nx * ny) + y * nx + x;
    const size_t idx_zn = (zn) * (nx * ny) + y * nx + x;

    const double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * c[idx]) / (dx * dx);
    const double cyy = (c[idx_yp] + c[idx_yn] - 2.0 * c[idx]) / (dy * dy);
    const double czz = (c[idx_zp] + c[idx_zn] - 2.0 * c[idx]) / (dz * dz);

    return cxx + cyy + czz;
}

// CUDA kernel to compute chemical potential for interior z layers (1..local_nz)
__global__ void computeChemicalPotentialKernel(const double* __restrict__ cold, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t local_nz_with_halo,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nxny = nx * ny;
    const size_t local_interior_nz = local_nz_with_halo > 2 ? local_nz_with_halo - 2 : 0;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = local_interior_nz * nxny;
    if (tid >= total) return;

    const size_t iz = tid / nxny; // 0-based in interior
    const size_t rem = tid % nxny;
    const size_t iy = rem / nx;
    const size_t ix = rem % nx;

    const size_t z = iz + 1; // shift for halo
    const size_t idx = idx3_local(ix, iy, z, nx, ny);
    const double cv = cold[idx];

    const double lap = deviceLaplacian(cold, nx, ny, local_nz_with_halo, dx, dy, dz, ix, iy, z);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv - gamma * lap;
}

// CUDA kernel to perform update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t local_nz_with_halo,
                                         const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nxny = nx * ny;
    const size_t local_interior_nz = local_nz_with_halo > 2 ? local_nz_with_halo - 2 : 0;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = local_interior_nz * nxny;
    if (tid >= total) return;

    const size_t iz = tid / nxny; // 0-based interior
    const size_t rem = tid % nxny;
    const size_t iy = rem / nx;
    const size_t ix = rem % nx;

    const size_t z = iz + 1;
    const size_t idx = idx3_local(ix, iy, z, nx, ny);

    const double lapMu = deviceLaplacian(mu, nx, ny, local_nz_with_halo, dx, dy, dz, ix, iy, z);
    cnew[idx] = cold[idx] + dt * D * lapMu;
}

// Initialize concentration field on host for local slab (excluding halos)
void initializeConcentrationHost(double* cold_managed, const size_t nx, const size_t ny, const size_t local_nz,
                                 const size_t global_z_offset) {
    const size_t nxny = nx * ny;
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_z = z + 1; // shift for halo
                const size_t idx = local_z * nxny + y * nx + x;
                const size_t global_z = global_z_offset + z;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const size_t vol_global = nx * ny * (global_z_offset + local_nz); // not critical for deterministic seeds
                const double pseudo = ((((linear_id + 1) * 1299709) % (nx * ny * local_nz)) / static_cast<double>(nx * ny * local_nz));
                cold_managed[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResultManaged(const double* cold_managed, const size_t nx, const size_t ny, const size_t local_nz) {
    const size_t nxny = nx * ny;
    double minVal = cold_managed[1 * nxny + 0];
    double maxVal = minVal;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t i = 0; i < nxny; ++i) {
            double val = cold_managed[z * nxny + i];
            if (std::isnan(val) || std::isinf(val)) {
                printf("Validation failed: found NaN or Inf value\n");
                return false;
            }
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 prints usage)
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
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

    const size_t nxny = nx * ny;

    // Decompose Z dimension among ranks (simple block distribution)
    size_t base = nz / world_size;
    size_t rem = nz % world_size;
    size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    size_t z_offset = (base * static_cast<size_t>(rank)) + std::min(static_cast<size_t>(rank), rem);

    if (local_nz == 0) {
        if (rank == 0) fprintf(stderr, "More ranks than Z-slices; reduce MPI ranks or increase nz.\n");
        MPI_Finalize();
        return 1;
    }

    // Allocate managed arrays with 1-layer halos at each end: size = (local_nz + 2) * nx * ny
    const size_t local_nz_with_halo = local_nz + 2;
    const size_t localSize = local_nz_with_halo * nxny;

    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;

    cudaMallocManaged(&cold, localSize * sizeof(double));
    cudaMallocManaged(&cnew, localSize * sizeof(double));
    cudaMallocManaged(&mu, localSize * sizeof(double));

    // Initialize halos to clamp values
    for (size_t i = 0; i < localSize; ++i) {
        cold[i] = 0.0;
        cnew[i] = 0.0;
        mu[i] = 0.0;
    }

    // Initialize local interior
    initializeConcentrationHost(cold, nx, ny, local_nz, z_offset);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Running Cahn-Hilliard simulation (MPI+OpenMP+CUDA)...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Buffers for halo exchange (one z-layer size)
    const size_t layerSize = nxny;

    for (int t = 0; t < iterations; ++t) {
        // Exchange cold halos: send first interior to prev rank's last halo and last interior to next rank's first halo
        MPI_Request reqs[4];
        int reqCount = 0;

        int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int next = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;

        // send first interior layer (z=1) to prev's halo (z=local_nz+1)
        MPI_Isend(&cold[1 * layerSize], layerSize, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[reqCount++]);
        // recv prev's last interior into our halo z=0
        MPI_Irecv(&cold[0 * layerSize], layerSize, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[reqCount++]);

        // send last interior layer (z=local_nz) to next's halo z=0
        MPI_Isend(&cold[local_nz * layerSize], layerSize, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[reqCount++]);
        // recv next's first interior into our halo z=local_nz+1
        MPI_Irecv(&cold[(local_nz + 1) * layerSize], layerSize, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[reqCount++]);

        MPI_Waitall(reqCount, reqs, MPI_STATUSES_IGNORE);

        // Compute chemical potential on GPU for local interior
        size_t interiorCells = local_nz * nxny;
        size_t threads = 256;
        size_t blocks = (interiorCells + threads - 1) / threads;
        computeChemicalPotentialKernel<<<blocks, threads>>>(cold, mu, nx, ny, local_nz_with_halo, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        cudaDeviceSynchronize();

        // Exchange mu halos similarly
        reqCount = 0;
        MPI_Isend(&mu[1 * layerSize], layerSize, MPI_DOUBLE, prev, 2, MPI_COMM_WORLD, &reqs[reqCount++]);
        MPI_Irecv(&mu[0 * layerSize], layerSize, MPI_DOUBLE, prev, 3, MPI_COMM_WORLD, &reqs[reqCount++]);
        MPI_Isend(&mu[local_nz * layerSize], layerSize, MPI_DOUBLE, next, 3, MPI_COMM_WORLD, &reqs[reqCount++]);
        MPI_Irecv(&mu[(local_nz + 1) * layerSize], layerSize, MPI_DOUBLE, next, 2, MPI_COMM_WORLD, &reqs[reqCount++]);
        MPI_Waitall(reqCount, reqs, MPI_STATUSES_IGNORE);

        // Update on GPU
        cahnHilliardUpdateKernel<<<blocks, threads>>>(cnew, cold, mu, nx, ny, local_nz_with_halo, D, dt, dx, dy, dz);
        cudaDeviceSynchronize();

        // Swap pointers for next iteration
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
    }

    // Aggregate for performance metrics: global cell updates
    double localCellUpdates = static_cast<double>(local_nz) * iterations * static_cast<double>(nxny);
    double globalCellUpdates = 0.0;
    MPI_Reduce(&localCellUpdates, &globalCellUpdates, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double mcups = globalCellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation: gather to rank 0
    if (printResults) {
        // Gather naive: each rank sends its interior to rank 0
        if (rank == 0) {
            std::vector<double> full(nxny * nz);
            // copy own
            for (size_t z = 0; z < local_nz; ++z) {
                for (size_t i = 0; i < nxny; ++i) {
                    full[(z + z_offset) * nxny + i] = cold[(z + 1) * nxny + i];
                }
            }
            for (int r = 1; r < world_size; ++r) {
                // receive header ranks sizes
                int rbase = nz / world_size;
                int rrem = nz % world_size;
                size_t rlocal_nz = rbase + (static_cast<size_t>(r) < rrem ? 1 : 0);
                size_t recvCount = rlocal_nz * nxny;
                MPI_Recv(full.data() + ((static_cast<size_t>(r) * rbase + std::min(static_cast<size_t>(r), static_cast<size_t>(rrem))) * nxny), recvCount, MPI_DOUBLE, r, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            print_results(full, "Concentration");
        } else {
            MPI_Send(&cold[1 * layerSize], local_nz * layerSize, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
        }
    }

    // Validation
    int valid_ok = 1;
    if (validate) {
        bool local_valid = validateResultManaged(cold, nx, ny, local_nz);
        int local_valid_int = local_valid ? 1 : 0;
        MPI_Reduce(&local_valid_int, &valid_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (valid_ok) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    cudaFree(cold);
    cudaFree(cnew);
    cudaFree(mu);

    MPI_Finalize();
    return (validate && rank == 0 && !valid_ok) ? 1 : 0;
}
