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

using Real = double;

// ─── Error-checking macros ─────────────────────────────────────────────────────

#define MPI_CHECK(call) do {                                            \
    int _mpi_err = (call);                                              \
    if (_mpi_err != MPI_SUCCESS) {                                      \
        char _buf[256];                                                 \
        MPI_Error_string(_mpi_err, _buf, nullptr);                      \
        fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, _buf); \
        MPI_Abort(MPI_COMM_WORLD, 1);                                   \
    }                                                                   \
} while(0)

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _cuda_err = (call);                                     \
    if (_cuda_err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_cuda_err), #call); \
        MPI_Abort(MPI_COMM_WORLD, 1);                                   \
    }                                                                   \
} while(0)

// ─── MPI globals ───────────────────────────────────────────────────────────────

static int mpi_rank = 0, mpi_size = 1;

// ─── 3D index ──────────────────────────────────────────────────────────────────

inline __device__ __host__ constexpr size_t idx3(size_t x, size_t y, size_t z,
                                                size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ─── CUDA kernel: 7-point stencil on interior ─────────────────────────────────

__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               size_t nx, size_t ny, size_t nz_local) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= 1 && z < nz_local - 1) {
        size_t idx = z * (nx * ny) + y * nx + x;
        Real val = (input[idx]
                   + input[idx - 1]           // left  (x-1)
                   + input[idx + 1]           // right (x+1)
                   + input[idx - nx]          // front (y-1)
                   + input[idx + nx]          // back  (y+1)
                   + input[idx - nx * ny]     // bottom (z-1)
                   + input[idx + nx * ny])    // top    (z+1)
                   / 7.0;
        output[idx] = val;
    }
}

// ─── Copy boundary values from input to output (host, OpenMP) ─────────────────

static void copyBoundaries(const std::vector<Real>& input,
                           std::vector<Real>& output,
                           size_t nx, size_t ny, size_t nz_local) {
    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                    z == 0 || z == nz_local - 1) {
                    output[idx3(x, y, z, nx, ny)] = input[idx3(x, y, z, nx, ny)];
                }
            }
        }
    }
}

// ─── Initialize grid (host, OpenMP) ───────────────────────────────────────────

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t nz_local,
                           size_t globalZOffset) {
    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t globalIdx = (globalZOffset + z) * (nx * ny) + y * nx + x;
                grid[idx3(x, y, z, nx, ny)] = static_cast<Real>(globalIdx % 19) * 1.0;
            }
        }
    }
}

// ─── Halo exchange (MPI) ──────────────────────────────────────────────────────

static void exchangeHalos(Real* localGrid,
                          size_t nx, size_t ny, size_t nz_local,
                          MPI_Comm comm) {
    if (mpi_size == 1) return;

    size_t slice = nx * ny;
    std::vector<Real> sendBuf(slice);
    std::vector<Real> recvBuf(slice);

    MPI_Request reqs[4];
    int nreq = 0;

    // Send top interior (z=nz_local-2) to rank+1, recv top halo from rank+1
    if (mpi_rank < mpi_size - 1) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                sendBuf[y * nx + x] = localGrid[idx3(x, y, nz_local - 2, nx, ny)];
            }
        }
        MPI_CHECK(MPI_Isend(sendBuf.data(), slice, MPI_DOUBLE, mpi_rank + 1, 100, comm, &reqs[nreq++]));
        MPI_CHECK(MPI_Irecv(recvBuf.data(), slice, MPI_DOUBLE, mpi_rank + 1, 101, comm, &reqs[nreq++]));
    }

    // Send bottom interior (z=1) to rank-1, recv bottom halo from rank-1
    if (mpi_rank > 0) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                sendBuf[y * nx + x] = localGrid[idx3(x, y, 1, nx, ny)];
            }
        }
        MPI_CHECK(MPI_Isend(sendBuf.data(), slice, MPI_DOUBLE, mpi_rank - 1, 101, comm, &reqs[nreq++]));
        MPI_CHECK(MPI_Irecv(recvBuf.data(), slice, MPI_DOUBLE, mpi_rank - 1, 100, comm, &reqs[nreq++]));
    }

    MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));

    // Unpack received data
    if (mpi_rank < mpi_size - 1) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                localGrid[idx3(x, y, nz_local - 1, nx, ny)] = recvBuf[y * nx + x];
            }
        }
    }
    if (mpi_rank > 0) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                localGrid[idx3(x, y, 0, nx, ny)] = recvBuf[y * nx + x];
            }
        }
    }
}

// ─── GPU context (persistent allocations + streams) ───────────────────────────

struct GPUContext {
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    cudaStream_t stream;
    size_t totalBytes;

    GPUContext(size_t totalElements, int gpuId) : totalBytes(totalElements * sizeof(Real)) {
        CUDA_CHECK(cudaSetDevice(gpuId));
        CUDA_CHECK(cudaMalloc(&d_grid1, totalBytes));
        CUDA_CHECK(cudaMalloc(&d_grid2, totalBytes));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~GPUContext() {
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
    }

    // Non-copyable
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;
};

// ─── Stencil iteration (hybrid MPI+OpenMP+CUDA) ───────────────────────────────

static void stencilIteration(const std::vector<Real>& input,
                              std::vector<Real>& output,
                              size_t nx, size_t ny, size_t nz_local,
                              GPUContext& gpuCtx) {
    // Upload input to GPU (async)
    CUDA_CHECK(cudaMemcpyAsync(gpuCtx.d_grid1, input.data(), gpuCtx.totalBytes,
                               cudaMemcpyHostToDevice, gpuCtx.stream));

    // Launch stencil kernel
    dim3 blockSize(8, 8, 4);
    dim3 gridSize(
        (nx + blockSize.x - 1) / blockSize.x,
        (ny + blockSize.y - 1) / blockSize.y,
        (nz_local + blockSize.z - 1) / blockSize.z
    );
    stencilKernel<<<gridSize, blockSize, 0, gpuCtx.stream>>>(
        gpuCtx.d_grid1, gpuCtx.d_grid2, nx, ny, nz_local);
    CUDA_CHECK(cudaGetLastError());

    // Download output from GPU (async)
    CUDA_CHECK(cudaMemcpyAsync(output.data(), gpuCtx.d_grid2, gpuCtx.totalBytes,
                               cudaMemcpyDeviceToHost, gpuCtx.stream));

    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(gpuCtx.stream));

    // Copy boundary values (host, OpenMP parallelized)
    copyBoundaries(input, output, nx, ny, nz_local);
}

// ─── Usage ─────────────────────────────────────────────────────────────────────

static void printUsage(const char* progName) {
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

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    // Determine GPU assignment: one GPU per rank (round-robin)
    int numGpus;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    int gpuId = mpi_rank % numGpus;

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
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

    // ─── Domain decomposition along Z ────────────────────────────────────────
    size_t nz_per_rank = nz / mpi_size;
    size_t remainder   = nz % mpi_size;
    size_t nz_interior_local = nz_per_rank + (mpi_rank < static_cast<int>(remainder) ? 1 : 0);
    size_t nz_local = nz_interior_local + 2;  // +2 for halos
    size_t localGridSize = nx * ny * nz_local;

    // Compute global Z offset for this rank's interior
    size_t z_offset = 0;
    for (int r = 0; r < mpi_rank; ++r) {
        z_offset += nz_per_rank + (r < static_cast<int>(remainder) ? 1 : 0);
    }

    // Set GPU device
    CUDA_CHECK(cudaSetDevice(gpuId));

    // Print config from rank 0
    if (mpi_rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d, GPUs: %d\n", mpi_size, numGpus);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate grids (double buffering)
    std::vector<Real> grid1(localGridSize);
    std::vector<Real> grid2(localGridSize);

    // Initialize
    if (mpi_rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz_local, z_offset);

    // Create GPU context with persistent allocations
    GPUContext gpuCtx(localGridSize, gpuId);

    // Run stencil iterations
    if (mpi_rank == 0) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halos before computation
        exchangeHalos(grid1.data(), nx, ny, nz_local, MPI_COMM_WORLD);
        exchangeHalos(grid2.data(), nx, ny, nz_local, MPI_COMM_WORLD);

        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz_local, gpuCtx);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz_local, gpuCtx);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather timing info
    long long localMs = duration.count();
    long long globalMs = 0;
    MPI_CHECK(MPI_Reduce(&localMs, &globalMs, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD));

    // Calculate performance
    double localCellUpdates = static_cast<double>((nx - 2) * (ny - 2) * nz_interior_local) * iterations;
    double totalCellUpdates = 0;
    MPI_CHECK(MPI_Reduce(&localCellUpdates, &totalCellUpdates, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    if (mpi_rank == 0) {
        printf("Computation time: %lld ms\n", globalMs);
        double mcups = totalCellUpdates / (globalMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ─── Gather full grid for validation/results on rank 0 ──────────────────
    std::vector<Real> fullGrid;
    if (mpi_rank == 0) fullGrid.resize(nx * ny * nz);

    // Compute z-offset for each rank
    std::vector<size_t> rankZOffset(mpi_size);
    {
        size_t runningOffset = 0;
        for (int r = 0; r < mpi_size; ++r) {
            rankZOffset[r] = runningOffset;
            runningOffset += nz_per_rank + (r < static_cast<int>(remainder) ? 1 : 0);
        }
    }

    // Determine which grid holds the final result
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    if (mpi_rank == 0) {
        // Rank 0: place own data + receive from all other ranks
        {
            int r = 0;
            size_t r_nz_interior = nz_per_rank + (r < static_cast<int>(remainder) ? 1 : 0);
            size_t r_localSize   = nx * ny * r_nz_interior;
            std::vector<Real> interior(nx * ny * r_nz_interior);
            for (size_t z = 0; z < r_nz_interior; ++z) {
                size_t srcPlane = (z + 1) * nx * ny;
                size_t dstPlane = z * nx * ny;
                std::memcpy(interior.data() + dstPlane,
                            finalGrid.data() + srcPlane,
                            nx * ny * sizeof(Real));
            }
            std::memcpy(fullGrid.data() + rankZOffset[0] * nx * ny,
                        interior.data(), r_localSize * sizeof(Real));
        }
        for (int r = 1; r < mpi_size; ++r) {
            size_t r_nz_interior = nz_per_rank + (r < static_cast<int>(remainder) ? 1 : 0);
            size_t r_localSize   = nx * ny * r_nz_interior;
            MPI_CHECK(MPI_Recv(fullGrid.data() + rankZOffset[r] * nx * ny,
                               r_localSize, MPI_DOUBLE, r, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
    } else {
        size_t r_nz_interior = nz_per_rank + (mpi_rank < static_cast<int>(remainder) ? 1 : 0);
        size_t r_localSize   = nx * ny * r_nz_interior;
        std::vector<Real> interior(nx * ny * r_nz_interior);
        for (size_t z = 0; z < r_nz_interior; ++z) {
            size_t srcPlane = (z + 1) * nx * ny;
            size_t dstPlane = z * nx * ny;
            std::memcpy(interior.data() + dstPlane,
                        finalGrid.data() + srcPlane,
                        nx * ny * sizeof(Real));
        }
        MPI_CHECK(MPI_Send(interior.data(), r_localSize, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD));
    }

    // Print results for external validation (rank 0)
    if (printResults && mpi_rank == 0) {
        print_results(fullGrid, "Grid");
    }

    // Validation (rank 0)
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        bool valid = true;

        for (const auto& val : fullGrid) {
            if (std::isnan(val) || std::isinf(val)) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = false;
                break;
            }
        }

        if (valid) {
            Real minVal = fullGrid[0], maxVal = fullGrid[0];
            for (const auto& val : fullGrid) {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
            printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

            if (maxVal > 1e6 || minVal < -1e6) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
