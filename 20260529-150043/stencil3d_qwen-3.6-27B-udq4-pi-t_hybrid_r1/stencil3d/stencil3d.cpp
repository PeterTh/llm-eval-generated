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

using Real = double;

// 3D index calculation (Z-major: z * (nx*ny) + y * nx + x)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Unified kernel: stencil on interior, copy boundaries/halos
__global__ void stencilUnifiedKernel(const Real* __restrict__ input,
                                     Real* __restrict__ output,
                                     const size_t nx, const size_t ny,
                                     const size_t nz,
                                     const size_t localNZ,
                                     const size_t localInteriorZStart,
                                     const size_t localInteriorZEnd)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t localZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || localZ >= localNZ) return;

    const size_t idx = localZ * (nx * ny) + y * nx + x;

    // Interior stencil point: not on any boundary, not in halo regions
    bool isInterior = (x >= 1 && x < nx - 1 &&
                       y >= 1 && y < ny - 1 &&
                       localZ >= localInteriorZStart &&
                       localZ < localInteriorZEnd);

    if (isInterior) {
        // 7-point stencil: average of center + 6 neighbors
        const Real center = input[idx];
        const Real left   = input[idx - 1];
        const Real right  = input[idx + 1];
        const Real front  = input[idx - nx];
        const Real back   = input[idx + nx];
        const Real bottom = input[idx - nx * ny];
        const Real top    = input[idx + nx * ny];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    } else {
        // Boundary or halo: copy from input unchanged
        output[idx] = input[idx];
    }
}

// Initialization kernel
__global__ void initKernel(Real* __restrict__ grid,
                           const size_t nx, const size_t ny,
                           const size_t localNZ,
                           const size_t globalZStart)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t localZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && localZ < localNZ) {
        const size_t globalZ = localZ + globalZStart;
        const size_t globalIdx = globalZ * (nx * ny) + y * nx + x;
        const size_t localIdx = localZ * (nx * ny) + y * nx + x;
        grid[localIdx] = static_cast<Real>(globalIdx % 19) * 1.0;
    }
}

// ============================================================================
// MPI Domain Decomposition
// ============================================================================

struct DomainInfo {
    int rank;
    int numRanks;
    size_t nx, ny, nz;

    // Local domain (includes halos)
    size_t localZStart;   // First local Z (global index)
    size_t localZEnd;     // Last local Z (global index, exclusive)
    size_t localNZ;       // Number of local Z slices
    size_t localSize;     // Total local elements

    // Owned slices (no halos)
    size_t globalZStart;  // First owned global Z
    size_t globalZEnd;    // Last owned global Z (exclusive)

    // Interior for stencil (no halos, no global boundaries)
    size_t localInteriorZStart;  // Local index of first interior Z
    size_t localInteriorZEnd;    // Local index of last interior Z (exclusive)

    // MPI neighbors
    int sendUpRank;
    int recvUpRank;
    int sendDownRank;
    int recvDownRank;
};

DomainInfo computeDomain(int rank, int numRanks, size_t nx, size_t ny, size_t nz) {
    DomainInfo info;
    info.rank = rank;
    info.numRanks = numRanks;
    info.nx = nx;
    info.ny = ny;
    info.nz = nz;

    // Distribute Z dimension across ranks
    size_t base = nz / numRanks;
    size_t rem = nz % numRanks;

    // Compute global owned range
    size_t gzs = 0;
    for (int r = 0; r < rank; ++r) {
        gzs += base + (r < static_cast<int>(rem) ? 1 : 0);
    }
    info.globalZStart = gzs;
    info.globalZEnd = gzs + base + (rank < static_cast<int>(rem) ? 1 : 0);

    // Local domain with halos (1 cell on each internal face)
    info.localZStart = (rank > 0) ? gzs - 1 : gzs;
    info.localZEnd = (rank < numRanks - 1) ? info.globalZEnd + 1 : info.globalZEnd;
    info.localNZ = info.localZEnd - info.localZStart;
    info.localSize = nx * ny * info.localNZ;

    // Interior Z range (local indices): exclude halos and global boundaries
    {
        size_t iGZStart = info.globalZStart;  // Skip bottom halo
        size_t iGZEnd = info.globalZEnd;      // Skip top halo
        // Exclude global boundaries z==0 and z==nz-1
        if (iGZStart < 1) iGZStart = 1;
        if (iGZEnd > nz - 1) iGZEnd = nz - 1;
        info.localInteriorZStart = iGZStart - info.localZStart;
        info.localInteriorZEnd = iGZEnd - info.localZStart;
    }

    // MPI neighbors
    info.sendUpRank = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;
    info.recvUpRank = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;
    info.sendDownRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    info.recvDownRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

    return info;
}

// ============================================================================
// Halo Exchange using MPI (non-blocking)
// ============================================================================

void exchangeHalos(Real* localGrid, const DomainInfo& dom) {
    const size_t sliceSize = dom.nx * dom.ny;

    // Send: interior slice adjacent to each interface
    const size_t bottomOwnedLocal = dom.globalZStart - dom.localZStart;
    const size_t topOwnedLocal = dom.globalZEnd - dom.localZStart - 1;

    const Real* sendBottom = localGrid + bottomOwnedLocal * sliceSize;
    const Real* sendTop = localGrid + topOwnedLocal * sliceSize;

    // Receive: into halo regions
    Real* recvBottom = localGrid;
    Real* recvTop = localGrid + (dom.localNZ - 1) * sliceSize;

    MPI_Request reqs[4];
    int n = 0;

    // Send top interior slice to rank above (goes into their bottom halo)
    if (dom.sendUpRank != MPI_PROC_NULL) {
        MPI_Isend(const_cast<Real*>(sendTop), sliceSize, MPI_DOUBLE,
                  dom.sendUpRank, 100 + dom.rank, MPI_COMM_WORLD, &reqs[n++]);
    }
    // Send bottom interior slice to rank below (goes into their top halo)
    if (dom.sendDownRank != MPI_PROC_NULL) {
        MPI_Isend(const_cast<Real*>(sendBottom), sliceSize, MPI_DOUBLE,
                  dom.sendDownRank, 100 + dom.rank, MPI_COMM_WORLD, &reqs[n++]);
    }
    // Receive top halo from rank above (their bottom interior slice)
    if (dom.recvUpRank != MPI_PROC_NULL) {
        MPI_Irecv(recvTop, sliceSize, MPI_DOUBLE,
                  dom.recvUpRank, 100 + dom.recvUpRank, MPI_COMM_WORLD, &reqs[n++]);
    }
    // Receive bottom halo from rank below (their top interior slice)
    if (dom.recvDownRank != MPI_PROC_NULL) {
        MPI_Irecv(recvBottom, sliceSize, MPI_DOUBLE,
                  dom.recvDownRank, 100 + dom.recvDownRank, MPI_COMM_WORLD, &reqs[n++]);
    }

    if (n > 0) MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0], maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    DomainInfo dom = computeDomain(rank, numRanks, nx, ny, nz);

    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const size_t sliceSize = nx * ny;

    // Host buffers (double buffering)
    std::vector<Real> h_grid1(dom.localSize), h_grid2(dom.localSize);

    // Device buffers (double buffering)
    Real* d_grid1 = nullptr, * d_grid2 = nullptr;
    cudaMalloc(&d_grid1, dom.localSize * sizeof(Real));
    cudaMalloc(&d_grid2, dom.localSize * sizeof(Real));

    // Initialize on GPU
    if (rank == 0) printf("Initializing grid...\n");
    {
        dim3 bs(16, 16, 4);
        dim3 gs((nx + bs.x - 1) / bs.x,
                (ny + bs.y - 1) / bs.y,
                (dom.localNZ + bs.z - 1) / bs.z);
        initKernel<<<gs, bs>>>(d_grid1, nx, ny, dom.localNZ, dom.localZStart);
        cudaDeviceSynchronize();
    }

    // Copy to host and exchange initial halos
    cudaMemcpy(h_grid1.data(), d_grid1, dom.localSize * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid2.data(), d_grid2, dom.localSize * sizeof(Real), cudaMemcpyDeviceToHost);

    if (numRanks > 1) {
        exchangeHalos(h_grid1.data(), dom);
        exchangeHalos(h_grid2.data(), dom);
    }

    // Copy back to GPU
    cudaMemcpy(d_grid1, h_grid1.data(), dom.localSize * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid2, h_grid2.data(), dom.localSize * sizeof(Real), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 bs(16, 16, 4);
    dim3 gs((nx + bs.x - 1) / bs.x,
            (ny + bs.y - 1) / bs.y,
            (dom.localNZ + bs.z - 1) / bs.z);

    // Run iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // GPU -> CPU for halo exchange
        cudaMemcpy(h_grid1.data(), d_grid1, dom.localSize * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_grid2.data(), d_grid2, dom.localSize * sizeof(Real), cudaMemcpyDeviceToHost);

        // Exchange halos between MPI ranks
        if (numRanks > 1) {
            exchangeHalos(h_grid1.data(), dom);
            exchangeHalos(h_grid2.data(), dom);
        }

        // CPU -> GPU
        cudaMemcpy(d_grid1, h_grid1.data(), dom.localSize * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grid2, h_grid2.data(), dom.localSize * sizeof(Real), cudaMemcpyHostToDevice);

        // Run unified stencil + boundary kernel
        if (iter % 2 == 0) {
            stencilUnifiedKernel<<<gs, bs>>>(d_grid1, d_grid2, nx, ny, nz,
                                             dom.localNZ,
                                             dom.localInteriorZStart, dom.localInteriorZEnd);
        } else {
            stencilUnifiedKernel<<<gs, bs>>>(d_grid2, d_grid1, nx, ny, nz,
                                             dom.localNZ,
                                             dom.localInteriorZStart, dom.localInteriorZEnd);
        }
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
    }

    // Gather final result
    const std::vector<Real>* localFinal;
    if (iterations % 2 == 0) {
        cudaMemcpy(h_grid1.data(), d_grid1, dom.localSize * sizeof(Real), cudaMemcpyDeviceToHost);
        localFinal = &h_grid1;
    } else {
        cudaMemcpy(h_grid2.data(), d_grid2, dom.localSize * sizeof(Real), cudaMemcpyDeviceToHost);
        localFinal = &h_grid2;
    }

    // Gather owned slices (no halos) to rank 0
    std::vector<Real> fullGrid;
    std::vector<int> recvcounts(numRanks), displs(numRanks);
    {
        size_t totalSize = nx * ny * nz;
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            DomainInfo rd = computeDomain(r, numRanks, nx, ny, nz);
            size_t owned = rd.globalZEnd - rd.globalZStart;
            recvcounts[r] = static_cast<int>(owned * sliceSize);
            displs[r] = static_cast<int>(offset);
            offset += owned * sliceSize;
        }
        fullGrid.resize(totalSize);
    }

    {
        size_t ownedStart = dom.globalZStart - dom.localZStart;
        size_t ownedSlices = dom.globalZEnd - dom.globalZStart;
        std::vector<Real> localOwned(ownedSlices * sliceSize);
        memcpy(localOwned.data(), localFinal->data() + ownedStart * sliceSize,
               ownedSlices * sliceSize * sizeof(Real));

        MPI_Gatherv(localOwned.data(), static_cast<int>(ownedSlices * sliceSize),
                    MPI_DOUBLE,
                    fullGrid.data(), recvcounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Performance and validation on rank 0
    if (rank == 0) {
        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) print_results(fullGrid, "Grid");

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(fullGrid, nx, ny, nz))
                printf("Validation: PASSED\n");
            else
                printf("Validation: FAILED\n");
        }
    }

    cudaFree(d_grid1);
    cudaFree(d_grid2);
    MPI_Finalize();
    return 0;
}
