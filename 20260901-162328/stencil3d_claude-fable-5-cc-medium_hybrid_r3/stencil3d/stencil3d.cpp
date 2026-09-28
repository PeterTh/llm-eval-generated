// Hybrid MPI + OpenMP + CUDA 3D stencil benchmark.
//
// Parallelization strategy:
//  - MPI: 1D domain decomposition along Z; one GPU per rank, halo exchange of
//    the two boundary XY planes each iteration (staged through pinned host
//    buffers). Interior-plane computation overlaps with the halo exchange.
//  - CUDA: the 7-point stencil update runs entirely on the GPU.
//  - OpenMP: host-side grid initialization, validation, and result assembly.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize the local slab (planes with global z in [z0, z0 + nzl)), stored
// at local plane offset 1 (plane 0 and plane nzl+1 are ghost planes).
void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                         const size_t z0, const size_t nzl) {
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < nzl; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, z0 + lz, nx, ny);
                grid[(lz + 1) * plane + y * nx + x] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on local planes lz in [lzBegin, lzEnd). Local plane lz maps
// to global plane gz = z0 + lz - 1. Global boundary points copy the input
// value; interior points apply the averaging stencil (same FP order as the
// original scalar code).
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t z0, const int lzBegin, const int lzEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lz = lzBegin + static_cast<int>(blockIdx.z);
    if (x >= nx || y >= ny || lz >= lzEnd) return;

    const size_t gz = z0 + lz - 1;
    const size_t plane = nx * ny;
    const size_t idx = static_cast<size_t>(lz) * plane + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - plane];
    const Real top    = input[idx + plane];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

static void launchStencil(const Real* d_in, Real* d_out,
                          const size_t nx, const size_t ny, const size_t nz,
                          const size_t z0, const int lzBegin, const int lzEnd,
                          cudaStream_t stream) {
    if (lzEnd <= lzBegin) return;
    const dim3 block(32, 8, 1);
    const dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ny) + block.y - 1) / block.y,
                    static_cast<unsigned>(lzEnd - lzBegin));
    stencilKernel<<<grid, block, 0, stream>>>(d_in, d_out, nx, ny, nz, z0, lzBegin, lzEnd);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();
    bool bad = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];

#pragma omp parallel for schedule(static) reduction(||:bad) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        const Real val = grid[i];
        // 1. No NaN or Inf values
        bad = bad || std::isnan(val) || std::isinf(val);
        // 2. Values should be reasonable (bounded)
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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

    if (static_cast<size_t>(nranks) > nz) {
        if (rank == 0)
            printf("Error: number of MPI ranks (%d) exceeds grid size in Z (%zu)\n", nranks, nz);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", nranks, omp_get_max_threads());
    }

    // Select a GPU: round-robin over the devices visible to this node,
    // using the node-local rank.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) printf("Error: no CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // 1D decomposition along Z: rank r owns global planes [z0, z0 + nzl)
    const size_t base = nz / nranks;
    const size_t rem = nz % nranks;
    const size_t nzl = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const size_t plane = nx * ny;
    const size_t planeBytes = plane * sizeof(Real);
    const size_t localSize = (nzl + 2) * plane;  // +2 ghost planes
    const size_t gridSize = nx * ny * nz;

    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank < nranks - 1) ? rank + 1 : MPI_PROC_NULL;

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> hostSlab(localSize, 0.0);
    initializeLocalGrid(hostSlab, nx, ny, z0, nzl);

    // Device buffers (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_grid1, hostSlab.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grid2, 0, localSize * sizeof(Real)));

    // Pinned host staging buffers for the halo exchange
    Real *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_sendLo, planeBytes));
    CUDA_CHECK(cudaMallocHost(&h_sendHi, planeBytes));
    CUDA_CHECK(cudaMallocHost(&h_recvLo, planeBytes));
    CUDA_CHECK(cudaMallocHost(&h_recvHi, planeBytes));

    cudaStream_t computeStream, commStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&commStream));

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    Real* d_in = d_grid1;
    Real* d_out = d_grid2;
    const int inzl = static_cast<int>(nzl);

    for (int iter = 0; iter < iterations; ++iter) {
        // Stage the boundary planes of the input to the host (comm stream)...
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendLo, d_in + plane, planeBytes,
                                       cudaMemcpyDeviceToHost, commStream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendHi, d_in + nzl * plane, planeBytes,
                                       cudaMemcpyDeviceToHost, commStream));

        // ...while the interior planes (which need no fresh halo) compute.
        launchStencil(d_in, d_out, nx, ny, nz, z0, 2, inzl, computeStream);

        CUDA_CHECK(cudaStreamSynchronize(commStream));
        MPI_Sendrecv(h_sendLo, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                     h_recvHi, static_cast<int>(plane), MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_sendHi, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                     h_recvLo, static_cast<int>(plane), MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(d_in, h_recvLo, planeBytes,
                                       cudaMemcpyHostToDevice, commStream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(d_in + (nzl + 1) * plane, h_recvHi, planeBytes,
                                       cudaMemcpyHostToDevice, commStream));

        // Boundary planes of the slab, once the fresh halos have arrived.
        launchStencil(d_in, d_out, nx, ny, nz, z0, 1, 2, commStream);
        if (nzl > 1)
            launchStencil(d_in, d_out, nx, ny, nz, z0, inzl, inzl + 1, commStream);

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaStreamSynchronize(commStream));

        std::swap(d_in, d_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = static_cast<long>((tEnd - tStart) * 1000.0);
    long globalDurationMs = 0;
    MPI_Reduce(&durationMs, &globalDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy the final local slab back and gather the full grid on rank 0.
    CUDA_CHECK(cudaMemcpy(hostSlab.data() + plane, d_in + plane, nzl * planeBytes,
                          cudaMemcpyDeviceToHost));

    std::vector<Real> finalGrid;
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t rNzl = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        const size_t rZ0 = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
        counts[r] = static_cast<int>(rNzl * plane);
        displs[r] = static_cast<int>(rZ0 * plane);
    }
    if (rank == 0) finalGrid.resize(gridSize);
    MPI_Gatherv(hostSlab.data() + plane, static_cast<int>(nzl * plane), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDurationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (globalDurationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

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

    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(commStream));
    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Finalize();
    return exitCode;
}
