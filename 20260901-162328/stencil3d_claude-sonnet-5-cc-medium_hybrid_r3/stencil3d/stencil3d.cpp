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

#define CUDA_CHECK(call)                                                              \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                    __FILE__, __LINE__);                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                            \
        }                                                                            \
    } while (0)

// 3D index calculation (z is a *local* index into whichever buffer is being addressed)
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize the interior (non-ghost) part of a rank's local slab directly on the GPU.
// The value stored matches the value the original single-process code would compute
// for the corresponding global index, so results are bit-identical to the reference.
__global__ void initGridKernel(Real* grid, size_t nx, size_t ny, size_t local_nz, size_t z_start) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zz = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zz >= local_nz) return;

    const size_t z_local = zz + 1; // skip the bottom ghost layer
    const size_t gz = z_start + zz;
    const size_t global_idx = gz * (nx * ny) + y * nx + x;
    const size_t local_idx = idx3(x, y, z_local, nx, ny);
    grid[local_idx] = static_cast<Real>(global_idx % 19) * 1.0;
}

// 7-point stencil kernel, operating on a contiguous range of local z-layers
// [zBegin, zBegin + zCount). Layers on the global domain boundary are copied
// through unchanged, exactly like the reference implementation.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               size_t nx, size_t ny, size_t local_nz, size_t z_start,
                               size_t nz_global, size_t zBegin, size_t zCount) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zz = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zz >= zCount) return;

    const size_t z_local = zBegin + zz; // in [1, local_nz]
    const size_t gz = z_start + (z_local - 1);
    const size_t idx = idx3(x, y, z_local, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz_global - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - nx * ny];
    const Real top = input[idx + nx * ny];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    (void)local_nz;
}

namespace {

dim3 makeGrid(size_t nx, size_t ny, size_t nz, dim3 block) {
    return dim3(static_cast<unsigned>((nx + block.x - 1) / block.x),
                static_cast<unsigned>((ny + block.y - 1) / block.y),
                static_cast<unsigned>((nz + block.z - 1) / block.z));
}

} // namespace

// Sanity-check the fully gathered result grid. Parallelized with OpenMP since this
// is a plain CPU-side reduction over the (potentially large) global grid.
bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    bool foundInvalid = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];

    #pragma omp parallel for reduction(||:foundInvalid) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            foundInvalid = true;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (foundInvalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank: same argv via mpirun)
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

    if (static_cast<size_t>(size) > nz) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds Z grid size (%zu)\n", size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    // Bind this rank to a GPU: node-local rank modulo the number of visible devices.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Error: no CUDA devices visible to rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // 1D domain decomposition along Z: contiguous slabs, remainder spread over the
    // first ranks so slab sizes differ by at most one layer.
    const size_t baseSlab = nz / static_cast<size_t>(size);
    const size_t remainder = nz % static_cast<size_t>(size);
    std::vector<size_t> localNzOf(size), zStartOf(size);
    {
        size_t z = 0;
        for (int r = 0; r < size; ++r) {
            const size_t sz = baseSlab + (static_cast<size_t>(r) < remainder ? 1 : 0);
            zStartOf[r] = z;
            localNzOf[r] = sz;
            z += sz;
        }
    }
    const size_t local_nz = localNzOf[rank];
    const size_t z_start = zStartOf[rank];
    const int downRank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int upRank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs/node: %d\n", size, deviceCount);
    }

    const size_t localDepth = local_nz + 2; // + top/bottom ghost layers
    const size_t localSlabElems = nx * ny * local_nz;
    const size_t localBufElems = nx * ny * localDepth;

    Real *grid1_d = nullptr, *grid2_d = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1_d, localBufElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&grid2_d, localBufElems * sizeof(Real)));

    cudaStream_t computeStream, haloStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&haloStream));

    // Pinned host staging buffers for halo exchange.
    Real *sendDown = nullptr, *sendUp = nullptr, *recvDown = nullptr, *recvUp = nullptr;
    CUDA_CHECK(cudaHostAlloc(&sendDown, nx * ny * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&sendUp, nx * ny * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recvDown, nx * ny * sizeof(Real), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recvUp, nx * ny * sizeof(Real), cudaHostAllocDefault));

    const dim3 block(32, 4, 1);
    const dim3 initGrid = makeGrid(nx, ny, local_nz, block);

    if (rank == 0) printf("Initializing grid...\n");
    initGridKernel<<<initGrid, block, 0, computeStream>>>(grid1_d, nx, ny, local_nz, z_start);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    const int TAG_UP = 100, TAG_DOWN = 101;

    auto exchangeHalo = [&](Real* input_d) {
        // Stage this rank's edge layers to host.
        CUDA_CHECK(cudaMemcpyAsync(sendDown, input_d + idx3(0, 0, 1, nx, ny),
                                    nx * ny * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaMemcpyAsync(sendUp, input_d + idx3(0, 0, local_nz, nx, ny),
                                    nx * ny * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));

        MPI_Request reqs[4];
        MPI_Isend(sendUp, static_cast<int>(nx * ny), MPI_DOUBLE, upRank, TAG_UP, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recvDown, static_cast<int>(nx * ny), MPI_DOUBLE, downRank, TAG_UP, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(sendDown, static_cast<int>(nx * ny), MPI_DOUBLE, downRank, TAG_DOWN, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recvUp, static_cast<int>(nx * ny), MPI_DOUBLE, upRank, TAG_DOWN, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (downRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(input_d + idx3(0, 0, 0, nx, ny), recvDown,
                                        nx * ny * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        }
        if (upRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(input_d + idx3(0, 0, local_nz + 1, nx, ny), recvUp,
                                        nx * ny * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
    };

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    Real* cur = grid1_d;
    Real* nxt = grid2_d;
    for (int iter = 0; iter < iterations; ++iter) {
        // Halo exchange for this iteration's input, overlapped with computing the
        // interior layers that do not depend on neighbor ghost data.
        const bool haveInterior = local_nz >= 3;
        const size_t interiorCount = haveInterior ? (local_nz - 2) : 0;

        if (haveInterior) {
            const dim3 interiorGrid = makeGrid(nx, ny, interiorCount, block);
            stencilKernel<<<interiorGrid, block, 0, computeStream>>>(
                cur, nxt, nx, ny, local_nz, z_start, nz, 2, interiorCount);
            CUDA_CHECK(cudaGetLastError());
        }

        exchangeHalo(cur);

        // Compute the boundary-adjacent layer(s) that needed the freshly received halo.
        if (local_nz == 1) {
            const dim3 g = makeGrid(nx, ny, 1, block);
            stencilKernel<<<g, block, 0, computeStream>>>(cur, nxt, nx, ny, local_nz, z_start, nz, 1, 1);
            CUDA_CHECK(cudaGetLastError());
        } else {
            const dim3 g1 = makeGrid(nx, ny, 1, block);
            stencilKernel<<<g1, block, 0, computeStream>>>(cur, nxt, nx, ny, local_nz, z_start, nz, 1, 1);
            CUDA_CHECK(cudaGetLastError());
            stencilKernel<<<g1, block, 0, computeStream>>>(cur, nxt, nx, ny, local_nz, z_start, nz, local_nz, 1);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        std::swap(cur, nxt);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int returnCode = 0;
    if (printResults || validate) {
        // Copy this rank's slab (without ghosts) back to the host, then gather the
        // full grid onto rank 0 for output / validation, exactly as the reference does.
        std::vector<Real> localHost(localSlabElems);
        CUDA_CHECK(cudaMemcpy(localHost.data(), cur + idx3(0, 0, 1, nx, ny),
                               localSlabElems * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> recvCounts, displs;
        std::vector<Real> fullGrid;
        if (rank == 0) {
            recvCounts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                recvCounts[r] = static_cast<int>(localNzOf[r] * nx * ny);
                displs[r] = static_cast<int>(zStartOf[r] * nx * ny);
            }
            fullGrid.resize(nx * ny * nz);
        }

        MPI_Gatherv(localHost.data(), static_cast<int>(localSlabElems), MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                    returnCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    returnCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFreeHost(sendDown));
    CUDA_CHECK(cudaFreeHost(sendUp));
    CUDA_CHECK(cudaFreeHost(recvDown));
    CUDA_CHECK(cudaFreeHost(recvUp));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaFree(grid1_d));
    CUDA_CHECK(cudaFree(grid2_d));

    MPI_Finalize();
    return returnCode;
}
