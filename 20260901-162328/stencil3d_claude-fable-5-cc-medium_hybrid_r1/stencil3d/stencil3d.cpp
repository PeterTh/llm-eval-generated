// Hybrid MPI + OpenMP + CUDA 3D stencil benchmark.
//
// Parallelization strategy:
//  - MPI: 1D domain decomposition along Z (slabs), one GPU per rank.
//  - CUDA: 7-point stencil kernel on each rank's slab; halo exchange is
//    overlapped with interior computation using two CUDA streams.
//  - OpenMP: host-side grid initialization and validation scans.

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 7-point stencil kernel operating on local planes [zlStart, zlStart+znum).
// Local plane zl holds global plane zOffset + zl - 1 (planes 0 and nzl+1 are halos).
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t zlStart, const size_t znum, const size_t zOffset) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zi = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zi >= znum) return;

    const size_t zl = zlStart + zi;
    const size_t gz = zOffset + zl - 1;
    const size_t idx = idx3(x, y, zl, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz - 1) {
        // Boundary values are copied
        output[idx] = input[idx];
    } else {
        const Real center = input[idx];
        const Real left   = input[idx3(x - 1, y, zl, nx, ny)];
        const Real right  = input[idx3(x + 1, y, zl, nx, ny)];
        const Real front  = input[idx3(x, y - 1, zl, nx, ny)];
        const Real back   = input[idx3(x, y + 1, zl, nx, ny)];
        const Real bottom = input[idx3(x, y, zl - 1, nx, ny)];
        const Real top    = input[idx3(x, y, zl + 1, nx, ny)];

        // Simple averaging stencil
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Initialize this rank's owned planes on the host (OpenMP).
void initializeLocalGrid(Real* grid, const size_t nx, const size_t ny,
                         const size_t nzl, const size_t zOffset) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, zOffset + z, nx, ny);
                grid[idx3(x, y, z, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool bad = false;
    const size_t n = grid.size();
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
    }

    // Bind each rank to a GPU (round-robin over local devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const size_t plane = nx * ny;
    const size_t gridSize = nx * ny * nz;

    // Z decomposition: rank r owns nzl contiguous planes starting at zOffset
    const size_t base = nz / (size_t)nprocs;
    const size_t rem = nz % (size_t)nprocs;
    const size_t nzl = base + ((size_t)rank < rem ? 1 : 0);
    const size_t zOffset = (size_t)rank * base + std::min((size_t)rank, rem);

    // Neighbors along Z; no exchange needed across the global boundary
    const int downNbr = (nzl > 0 && zOffset > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upNbr = (nzl > 0 && zOffset + nzl < nz) ? rank + 1 : MPI_PROC_NULL;

    // Device buffers: owned planes plus one halo plane on each side
    const size_t localSize = (nzl + 2) * plane;
    Real *dIn = nullptr, *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dOut, localSize * sizeof(Real)));

    // Pinned host staging: owned planes (init/gather) and halo planes
    Real* hLocal = nullptr;
    Real* hHalo = nullptr;  // [sendDown | sendUp | recvDown | recvUp]
    CUDA_CHECK(cudaMallocHost(&hLocal, std::max(nzl, (size_t)1) * plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hHalo, 4 * plane * sizeof(Real)));
    Real* sendDown = hHalo;
    Real* sendUp = hHalo + plane;
    Real* recvDown = hHalo + 2 * plane;
    Real* recvUp = hHalo + 3 * plane;

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    if (nzl > 0) {
        initializeLocalGrid(hLocal, nx, ny, nzl, zOffset);
        CUDA_CHECK(cudaMemcpy(dIn + plane, hLocal, nzl * plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(dIn, 0, plane * sizeof(Real)));
    CUDA_CHECK(cudaMemset(dIn + (nzl + 1) * plane, 0, plane * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dOut, dIn, localSize * sizeof(Real), cudaMemcpyDeviceToDevice));

    cudaStream_t sInterior, sHalo;
    CUDA_CHECK(cudaStreamCreate(&sInterior));
    CUDA_CHECK(cudaStreamCreate(&sHalo));

    const dim3 block(32, 8, 2);
    auto launch = [&](const Real* in, Real* out, size_t zlStart, size_t znum, cudaStream_t s) {
        if (znum == 0) return;
        const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                        (unsigned)((ny + block.y - 1) / block.y),
                        (unsigned)((znum + block.z - 1) / block.z));
        stencilKernel<<<grid, block, 0, s>>>(in, out, nx, ny, nz, zlStart, znum, zOffset);
    };

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool needHalo = (downNbr != MPI_PROC_NULL) || (upNbr != MPI_PROC_NULL);
    for (int iter = 0; iter < iterations; ++iter) {
        if (nzl > 0) {
            // Stage outgoing halo planes while the interior computes
            if (downNbr != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendDown, dIn + plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sHalo));
            if (upNbr != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendUp, dIn + nzl * plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sHalo));

            // Interior planes (do not depend on incoming halos)
            if (nzl > 2) launch(dIn, dOut, 2, nzl - 2, sInterior);

            if (needHalo) {
                CUDA_CHECK(cudaStreamSynchronize(sHalo));
                MPI_Sendrecv(sendDown, (int)plane, MPI_DOUBLE, downNbr, 0,
                             recvUp, (int)plane, MPI_DOUBLE, upNbr, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(sendUp, (int)plane, MPI_DOUBLE, upNbr, 1,
                             recvDown, (int)plane, MPI_DOUBLE, downNbr, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (downNbr != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(dIn, recvDown, plane * sizeof(Real),
                                               cudaMemcpyHostToDevice, sHalo));
                if (upNbr != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(dIn + (nzl + 1) * plane, recvUp,
                                               plane * sizeof(Real), cudaMemcpyHostToDevice, sHalo));
            }

            // Slab-edge planes once their halos are in place
            launch(dIn, dOut, 1, 1, sHalo);
            if (nzl > 1) launch(dIn, dOut, nzl, 1, sHalo);

            CUDA_CHECK(cudaStreamSynchronize(sInterior));
            CUDA_CHECK(cudaStreamSynchronize(sHalo));
        }
        std::swap(dIn, dOut);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Calculate performance metrics
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid on rank 0 (result is in dIn after the last swap)
    if (nzl > 0) {
        CUDA_CHECK(cudaMemcpy(hLocal, dIn + plane, nzl * plane * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<Real> finalGrid;
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const size_t rn = base + ((size_t)r < rem ? 1 : 0);
        const size_t ro = (size_t)r * base + std::min((size_t)r, rem);
        counts[r] = (int)(rn * plane);
        displs[r] = (int)(ro * plane);
    }
    if (rank == 0) finalGrid.resize(gridSize);
    MPI_Gatherv(hLocal, (int)(nzl * plane), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(sInterior));
    CUDA_CHECK(cudaStreamDestroy(sHalo));
    CUDA_CHECK(cudaFreeHost(hLocal));
    CUDA_CHECK(cudaFreeHost(hHalo));
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));

    MPI_Finalize();
    return exitCode;
}
