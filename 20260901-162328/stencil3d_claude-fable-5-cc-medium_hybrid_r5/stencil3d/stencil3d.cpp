#include <algorithm>
#include <chrono>
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

// Initialize this rank's slab (local planes z0..z0+lnz-1 of the global grid,
// stored with one halo plane on each side at local z index 0 and lnz+1).
void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                         const size_t z0, const size_t lnz) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < lnz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, z0 + z, nx, ny);
                grid[idx3(x, y, z + 1, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil kernel over local planes [lzStart, lzStart+lzCount) of one
// rank's slab. Local plane lz in [1, lnz] corresponds to global plane
// gz = z0 + lz - 1. Global boundary points are copied through unchanged;
// interior points get the averaging stencil.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t z0, const size_t lzStart, const size_t lzCount) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + lzStart;
    if (x >= nx || y >= ny || lz >= lzStart + lzCount) return;

    const size_t gz = z0 + lz - 1;
    const size_t idx = idx3(x, y, lz, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - nx * ny];
    const Real top    = input[idx + nx * ny];

    // Simple averaging stencil
    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values, 2. value range (fused, OpenMP reductions)
    bool bad = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    const size_t n = grid.size();
#pragma omp parallel for reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) bad = true;
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", nprocs, omp_get_max_threads());
    }

    // Select a GPU per rank based on the rank's position on its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Block decomposition of the nz planes along Z
    const size_t lnz = nz / nprocs + (static_cast<size_t>(rank) < nz % nprocs ? 1 : 0);
    const size_t z0 = static_cast<size_t>(rank) * (nz / nprocs) + std::min<size_t>(rank, nz % nprocs);
    const size_t plane = nx * ny;
    const size_t localSize = plane * (lnz + 2);  // + halo plane on each side

    const int down = (z0 > 0 && lnz > 0) ? rank - 1 : MPI_PROC_NULL;             // holds gz = z0-1
    const int up   = (lnz > 0 && z0 + lnz < nz) ? rank + 1 : MPI_PROC_NULL;      // holds gz = z0+lnz

    // Host slab (with halos) and device double buffers
    std::vector<Real> hostGrid(localSize, 0.0);

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeLocalGrid(hostGrid, nx, ny, z0, lnz);

    Real* dIn = nullptr;
    Real* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dOut, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dIn, hostGrid.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Pinned staging buffers for halo exchange
    Real* hSend = nullptr;  // [0]: bottom plane, [1]: top plane
    Real* hRecv = nullptr;
    CUDA_CHECK(cudaMallocHost(&hSend, 2 * plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hRecv, 2 * plane * sizeof(Real)));

    cudaStream_t sInterior, sBoundary;
    CUDA_CHECK(cudaStreamCreate(&sInterior));
    CUDA_CHECK(cudaStreamCreate(&sBoundary));

    const dim3 block(64, 4, 2);
    const size_t gx = (nx + block.x - 1) / block.x;
    const size_t gy = (ny + block.y - 1) / block.y;
    auto launchPlanes = [&](Real* in, Real* out, size_t lzStart, size_t lzCount, cudaStream_t s) {
        if (lzCount == 0) return;
        const dim3 grid(gx, gy, (lzCount + block.z - 1) / block.z);
        stencilKernel<<<grid, block, 0, s>>>(in, out, nx, ny, nz, z0, lzStart, lzCount);
        CUDA_CHECK(cudaGetLastError());
    };

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool hasNeighbor = (down != MPI_PROC_NULL) || (up != MPI_PROC_NULL);

    for (int iter = 0; iter < iterations; ++iter) {
        if (hasNeighbor) {
            // Stage outgoing halo planes, then exchange with Z neighbors while
            // the interior planes (which do not touch the halos) compute.
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSend, dIn + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, sBoundary));
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSend + plane, dIn + plane * lnz, plane * sizeof(Real), cudaMemcpyDeviceToHost, sBoundary));
            CUDA_CHECK(cudaStreamSynchronize(sBoundary));

            if (lnz > 2) launchPlanes(dIn, dOut, 2, lnz - 2, sInterior);

            MPI_Request reqs[4];
            int nreq = 0;
            MPI_Irecv(hRecv, plane, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(hRecv + plane, plane, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(hSend + plane, plane, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(hSend, plane, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(dIn, hRecv, plane * sizeof(Real), cudaMemcpyHostToDevice, sBoundary));
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(dIn + plane * (lnz + 1), hRecv + plane, plane * sizeof(Real), cudaMemcpyHostToDevice, sBoundary));

            // Boundary planes depend on the freshly received halos
            launchPlanes(dIn, dOut, 1, 1, sBoundary);
            if (lnz > 1) launchPlanes(dIn, dOut, lnz, 1, sBoundary);
            CUDA_CHECK(cudaStreamSynchronize(sInterior));
            CUDA_CHECK(cudaStreamSynchronize(sBoundary));
        } else if (lnz > 0) {
            launchPlanes(dIn, dOut, 1, lnz, sInterior);
            CUDA_CHECK(cudaStreamSynchronize(sInterior));
        }

        std::swap(dIn, dOut);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDuration);

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (globalDuration / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the final slab back to the host (result is in dIn after the last swap)
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), dIn, localSize * sizeof(Real), cudaMemcpyDeviceToHost));

    // Gather the full grid on rank 0 for result printing / validation
    std::vector<Real> finalGrid;
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const size_t rlnz = nz / nprocs + (static_cast<size_t>(r) < nz % nprocs ? 1 : 0);
        const size_t rz0 = static_cast<size_t>(r) * (nz / nprocs) + std::min<size_t>(r, nz % nprocs);
        counts[r] = static_cast<int>(rlnz * plane);
        displs[r] = static_cast<int>(rz0 * plane);
    }
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    MPI_Gatherv(hostGrid.data() + plane, static_cast<int>(lnz * plane), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(sInterior));
    CUDA_CHECK(cudaStreamDestroy(sBoundary));
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFreeHost(hSend));
    CUDA_CHECK(cudaFreeHost(hRecv));

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(finalGrid, nx, ny, nz) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
