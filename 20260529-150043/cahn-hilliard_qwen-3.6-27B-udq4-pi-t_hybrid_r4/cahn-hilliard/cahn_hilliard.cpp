#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ size_t idx3d(const size_t x, const size_t y, const size_t z,
                                        const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ double computeLaplacianDev(const double* __restrict__ c,
                                      const size_t nx, const size_t ny, const size_t nz,
                                      const double dx, const double dy, const double dz,
                                      const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3d(xp, y, z, nx, ny)] + c[idx3d(xn, y, z, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3d(x, yp, z, nx, ny)] + c[idx3d(x, yn, z, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3d(x, y, zp, nx, ny)] + c[idx3d(x, y, zn, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3d(x, y, z, nx, ny);
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDev(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3d(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D *
                    computeLaplacianDev(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                              const size_t nx, const size_t ny, const size_t nz,
                                              const size_t global_offset, const size_t global_vol) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3d(x, y, z, nx, ny);
        const size_t linear_id = global_offset + z * (nx * ny) + y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % global_vol) /
                               static_cast<double>(global_vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void getLaunchConfig(const size_t nx, const size_t ny, const size_t nz,
                     dim3& blockDim, dim3& gridDim) {
    const size_t total = nx * ny * nz;
    if (total <= 1024) {
        blockDim = dim3(8, 8, 8);
    } else if (total <= 65536) {
        blockDim = dim3(16, 8, 4);
    } else {
        blockDim = dim3(16, 8, 2);
    }
    gridDim.x = (nx + blockDim.x - 1) / blockDim.x;
    gridDim.y = (ny + blockDim.y - 1) / blockDim.y;
    gridDim.z = (nz + blockDim.z - 1) / blockDim.z;

    cudaFuncAttributes attr;
    cudaError_t e = cudaFuncGetAttributes(&attr, computeChemicalPotentialKernel);
    if (e == cudaSuccess) {
        const size_t tpb = blockDim.x * blockDim.y * blockDim.z;
        if (tpb > static_cast<size_t>(attr.maxThreadsPerBlock)) {
            blockDim.z = 1;
            blockDim.y = static_cast<unsigned int>(
                std::min(static_cast<size_t>(blockDim.y),
                         static_cast<size_t>(attr.maxThreadsPerBlock) / blockDim.x));
            gridDim.y = (ny + blockDim.y - 1) / blockDim.y;
            gridDim.z = (nz + blockDim.z - 1) / blockDim.z;
        }
    }
}

void computeZDecomposition(const size_t nz_global, const int rank, const int nRanks,
                           size_t& nz_local, size_t& z_start) {
    nz_local = nz_global / nRanks;
    size_t remainder = nz_global % nRanks;
    z_start = rank * nz_local + std::min(static_cast<size_t>(rank), remainder);
    if (rank < (int)remainder) nz_local += 1;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minVal = c[0], maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <n> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along Z
    size_t nz_local, z_start;
    computeZDecomposition(nz, rank, nRanks, nz_local, z_start);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("MPI ranks: %d\n", nRanks);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    const size_t localGridSize = nx * ny * nz_local;
    const size_t globalVol = nx * ny * nz;

    // Select GPU device
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    cudaSetDevice(rank % std::max(1, numGPUs));

    // Device pointers
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    cudaMalloc(&d_cold, localGridSize * sizeof(double));
    cudaMalloc(&d_cnew, localGridSize * sizeof(double));
    cudaMalloc(&d_mu, localGridSize * sizeof(double));

    // Launch configuration
    dim3 blockDim, gridDim;
    getLaunchConfig(nx, ny, nz_local, blockDim, gridDim);

    // Initialize concentration field on GPU
    initializeConcentrationKernel<<<gridDim, blockDim>>>(
        d_cold, nx, ny, nz_local, z_start * nx * ny, globalVol);
    cudaDeviceSynchronize();

    // Host buffers for Z-direction halo exchange (XY planes)
    const size_t haloSize = nx * ny;
    std::vector<double> sendTop(haloSize), recvTop(haloSize);
    std::vector<double> sendBottom(haloSize), recvBottom(haloSize);

    MPI_Request requests[4];

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // ---- Synchronize GPU before halo exchange ----
        cudaDeviceSynchronize();

        // ---- Extract Z-halos from device to host ----
        if (nz_local > 1) {
            cudaMemcpy(sendTop.data(),
                       d_cold + idx3(0, 0, nz_local - 1, nx, ny),
                       haloSize * sizeof(double), cudaMemcpyDeviceToHost);
        }
        if (nz_local > 0) {
            cudaMemcpy(sendBottom.data(),
                       d_cold + idx3(0, 0, 0, nx, ny),
                       haloSize * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // ---- Post non-blocking halo exchange ----
        // Tag 100: rank i sends TOP to rank i+1; rank i+1 receives (fills BOTTOM)
        // Tag 200: rank i+1 sends BOTTOM to rank i; rank i receives (fills TOP)
        int reqIdx = 0;

        if (rank < nRanks - 1) {
            MPI_Isend(sendTop.data(), static_cast<int>(haloSize), MPI_DOUBLE,
                      rank + 1, 100, MPI_COMM_WORLD, &requests[reqIdx++]);
            MPI_Irecv(recvTop.data(), static_cast<int>(haloSize), MPI_DOUBLE,
                      rank + 1, 200, MPI_COMM_WORLD, &requests[reqIdx++]);
        }

        if (rank > 0) {
            MPI_Isend(sendBottom.data(), static_cast<int>(haloSize), MPI_DOUBLE,
                      rank - 1, 200, MPI_COMM_WORLD, &requests[reqIdx++]);
            MPI_Irecv(recvBottom.data(), static_cast<int>(haloSize), MPI_DOUBLE,
                      rank - 1, 100, MPI_COMM_WORLD, &requests[reqIdx++]);
        }

        if (reqIdx > 0) {
            MPI_Waitall(reqIdx, requests, MPI_STATUSES_IGNORE);
        }

        // ---- Copy received halos back to device ----
        if (nz_local > 1 && rank < nRanks - 1) {
            cudaMemcpy(d_cold + idx3(0, 0, nz_local - 1, nx, ny),
                       recvTop.data(),
                       haloSize * sizeof(double), cudaMemcpyHostToDevice);
        }
        if (nz_local > 0 && rank > 0) {
            cudaMemcpy(d_cold + idx3(0, 0, 0, nx, ny),
                       recvBottom.data(),
                       haloSize * sizeof(double), cudaMemcpyHostToDevice);
        }

        // ---- CUDA computation ----
        computeChemicalPotentialKernel<<<gridDim, blockDim>>>(
            d_cold, d_mu, nx, ny, nz_local, dx, dy, dz,
            gamma, e_AA, e_BB, e_AB);

        cahnHilliardUpdateKernel<<<gridDim, blockDim>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz_local, D, dt, dx, dy, dz);

        // Swap device pointers for next iteration
        std::swap(d_cold, d_cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cudaDeviceSynchronize();

    // Copy final result back to host
    std::vector<double> localCold(localGridSize);
    cudaMemcpy(localCold.data(), d_cold, localGridSize * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Gather all results to rank 0
    std::vector<double> globalCold;
    std::vector<int> counts(nRanks), displs(nRanks);
    if (rank == 0) globalCold.resize(nx * ny * nz);

    for (int r = 0; r < nRanks; ++r) {
        size_t nz_r, zs;
        computeZDecomposition(nz, r, nRanks, nz_r, zs);
        counts[r] = static_cast<int>(nx * ny * nz_r);
        displs[r] = static_cast<int>(zs * nx * ny);
    }

    MPI_Gatherv(localCold.data(), static_cast<int>(localGridSize), MPI_DOUBLE,
                globalCold.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Timing
    double cellUpdates = static_cast<double>(localGridSize) * iterations;
    double mcups = duration.count() > 0 ? cellUpdates / (duration.count() / 1000.0) / 1e6 : 0;
    printf("Rank %d: Computation time: %ld ms, Performance: %.3f MCellUpdates/s (local)\n",
           rank, static_cast<long>(duration.count()), mcups);

    if (rank == 0) {
        double totalCellUpdates = static_cast<double>(nx * ny * nz) * iterations;
        double totalMcups = duration.count() > 0 ?
            totalCellUpdates / (duration.count() / 1000.0) / 1e6 : 0;
        printf("Total Performance: %.3f MCellUpdates/s\n", totalMcups);

        if (printResults) print_results(globalCold, "Concentration");

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(globalCold, nx, ny, nz))
                printf("Validation: PASSED\n");
            else
                printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    cudaFree(d_cold); cudaFree(d_cnew); cudaFree(d_mu);
    MPI_Finalize();
    return 0;
}
