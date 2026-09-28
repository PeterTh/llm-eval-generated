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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// 3D index calculation (global layout: z-major, matches original serial code)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Fill the local extended sub-volume (including halo planes) with the same
// deterministic function of the *global* index that the original serial code
// used, so every rank can compute its own halo without communication.
void initializeGridLocal(std::vector<Real>& localGrid, const size_t nx, const size_t ny,
                          const size_t nzLocalExt, const size_t zOffset) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < nzLocalExt; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            // global z corresponds to zOffset - 1 + lz
            const long long gz = static_cast<long long>(zOffset) - 1 + static_cast<long long>(lz);
            for (size_t x = 0; x < nx; ++x) {
                const size_t localIdx = lz * (nx * ny) + y * nx + x;
                if (gz < 0) {
                    // Halo below the global domain: never read, value irrelevant.
                    localGrid[localIdx] = 0.0;
                    continue;
                }
                const size_t globalIdx = idx3(x, y, static_cast<size_t>(gz), nx, ny);
                localGrid[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation for one local extended sub-volume.
// input/output layout: nx * ny * (nzLocal + 2); lz in [1, nzLocal] are the
// "real" planes owned by this rank, lz == 0 and lz == nzLocal+1 are halo
// planes filled in via MPI exchange (or unused at global boundaries).
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nzLocal,
                               const size_t zOffset, const size_t nzGlobal) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || lz > nzLocal) return;

    const size_t stride_xy = nx * ny;
    const size_t gz = zOffset - 1 + lz;
    const size_t idx = lz * stride_xy + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nzGlobal - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - stride_xy];
    const Real top = input[idx + stride_xy];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// Exchange the halo planes of `d_buf` (the buffer that was just written by the
// stencil kernel) with the up/down MPI neighbors so it is ready to be used as
// input for the next iteration.
void exchangeHalo(Real* d_buf, Real* h_sendDown, Real* h_sendUp, Real* h_recvDown, Real* h_recvUp,
                   const size_t nx, const size_t ny, const size_t nzLocal,
                   const int downRank, const int upRank, MPI_Comm comm) {
    const size_t planeElems = nx * ny;
    const size_t planeBytes = planeElems * sizeof(Real);
    const size_t stride_xy = nx * ny;

    // Pull the boundary-most real planes down to the host.
    CUDA_CHECK(cudaMemcpy(h_sendDown, d_buf + 1 * stride_xy, planeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sendUp, d_buf + nzLocal * stride_xy, planeBytes, cudaMemcpyDeviceToHost));

    // Send bottom plane down / receive our top halo from the up neighbor.
    MPI_Sendrecv(h_sendDown, static_cast<int>(planeElems), MPI_DOUBLE, downRank, 0,
                 h_recvUp, static_cast<int>(planeElems), MPI_DOUBLE, upRank, 0,
                 comm, MPI_STATUS_IGNORE);
    // Send top plane up / receive our bottom halo from the down neighbor.
    MPI_Sendrecv(h_sendUp, static_cast<int>(planeElems), MPI_DOUBLE, upRank, 1,
                 h_recvDown, static_cast<int>(planeElems), MPI_DOUBLE, downRank, 1,
                 comm, MPI_STATUS_IGNORE);

    if (downRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_buf + 0 * stride_xy, h_recvDown, planeBytes, cudaMemcpyHostToDevice));
    }
    if (upRank != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_buf + (nzLocal + 1) * stride_xy, h_recvUp, planeBytes, cudaMemcpyHostToDevice));
    }
}

// Compute this rank's [zStart, zEnd) slab of the global z range using a
// balanced block decomposition (first `nz % size` ranks get one extra plane).
void computeLocalRange(const size_t nz, const int size, const int rank, size_t& zStart, size_t& zEnd) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    if (static_cast<size_t>(rank) < rem) {
        zStart = static_cast<size_t>(rank) * (base + 1);
        zEnd = zStart + (base + 1);
    } else {
        zStart = rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
        zEnd = zStart + base;
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Bind this rank to a GPU: ranks sharing a node round-robin over the
    // GPUs visible on that node.
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_free(&shmComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank, so every
    // rank parses independently with no need to broadcast).
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

    if (nz < static_cast<size_t>(size)) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds grid depth (%zu)\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", size, omp_get_max_threads(), deviceCount);
    }

    size_t zStart, zEnd;
    computeLocalRange(nz, size, rank, zStart, zEnd);
    const size_t nzLocal = zEnd - zStart;
    const size_t nzLocalExt = nzLocal + 2;
    const size_t planeElems = nx * ny;
    const size_t localExtElems = planeElems * nzLocalExt;

    const int downRank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int upRank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Host staging buffers for the initial fill and for halo exchange.
    std::vector<Real> hostInit(localExtElems);
    Real* h_sendDown; Real* h_sendUp; Real* h_recvDown; Real* h_recvUp;
    CUDA_CHECK(cudaMallocHost(&h_sendDown, planeElems * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_sendUp, planeElems * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recvDown, planeElems * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recvUp, planeElems * sizeof(Real)));

    // Initialize (OpenMP-parallel host fill, matching the original global
    // index formula, then upload to the GPU).
    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(hostInit, nx, ny, nzLocalExt, zStart);

    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, localExtElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, localExtElems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_grid1, hostInit.data(), localExtElems * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, hostInit.data(), localExtElems * sizeof(Real), cudaMemcpyHostToDevice));

    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                     static_cast<unsigned>((ny + block.y - 1) / block.y),
                     static_cast<unsigned>((nzLocal + block.z - 1) / block.z));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, nzLocal, zStart, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Refresh d_out's halo planes for the next iteration (skip on the
        // last iteration since they will not be read again).
        if (iter != iterations - 1) {
            exchangeHalo(d_out, h_sendDown, h_sendUp, h_recvDown, h_recvUp, nx, ny, nzLocal,
                         downRank, upRank, MPI_COMM_WORLD);
        }
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

    // Copy this rank's real (non-halo) planes back to the host; they are
    // contiguous in device memory, so a single copy suffices.
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> hostLocal(planeElems * nzLocal);
    CUDA_CHECK(cudaMemcpy(hostLocal.data(), d_final + planeElems, planeElems * nzLocal * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    // Gather the full grid onto rank 0, in the same z-major order the
    // original serial code used, so results/hash are bit-identical.
    std::vector<Real> globalGrid;
    std::vector<int> recvCounts, displs;
    if (rank == 0) {
        globalGrid.resize(nx * ny * nz);
        recvCounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            size_t rs, re;
            computeLocalRange(nz, size, r, rs, re);
            recvCounts[r] = static_cast<int>(planeElems * (re - rs));
            displs[r] = static_cast<int>(planeElems * rs);
        }
    }
    MPI_Gatherv(hostLocal.data(), static_cast<int>(hostLocal.size()), MPI_DOUBLE,
                rank == 0 ? globalGrid.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(globalGrid, "Grid");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(globalGrid, nx, ny, nz);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFreeHost(h_sendDown));
    CUDA_CHECK(cudaFreeHost(h_sendUp));
    CUDA_CHECK(cudaFreeHost(h_recvDown));
    CUDA_CHECK(cudaFreeHost(h_recvUp));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Finalize();

    if (validate) {
        return valid ? 0 : 1;
    }
    return 0;
}
