#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (global coordinates)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local 3D index calculation (local array, includes halo)
inline constexpr size_t lidx3(const int lx, const int ly, const int lz, const int lx0, const int ly0) noexcept {
    return lz * (lx0 * ly0) + ly * lx0 + lx;
}

// Compute 3D decomposition: distribute ranks across Z, then Y, then X dimensions
void computeDecomposition(int numRanks, size_t nx, size_t ny, size_t nz,
                          int& px, int& py, int& pz) {
    // Try to find a balanced 3D decomposition
    // Start by dividing evenly, then adjust
    double cubeRoot = cbrt(static_cast<double>(numRanks));
    pz = static_cast<int>(std::round(cubeRoot));
    py = static_cast<int>(std::round(cubeRoot));
    px = static_cast<int>(std::round(cubeRoot));

    // Adjust to match numRanks exactly
    while (px * py * pz > numRanks) {
        // Reduce the largest dimension that can be reduced
        if (pz > 1 && (pz == py || pz == px)) --pz;
        else if (py > 1 && (py == px || py <= pz)) --py;
        else --px;
    }
    while (px * py * pz < numRanks) {
        // Increase the dimension that gives best balance
        if (pz <= py && pz <= px) ++pz;
        else if (py <= px) ++py;
        else ++px;
    }

    // Cap by grid dimensions (can't decompose a dimension smaller than 3)
    if (px > static_cast<int>(nx)) { px = static_cast<int>(nx); }
    if (py > static_cast<int>(ny)) { py = static_cast<int>(ny); }
    if (pz > static_cast<int>(nz)) { pz = static_cast<int>(nz); }

    // If we still have unused ranks, redistribute
    int total = px * py * pz;
    while (total < numRanks) {
        // Try adding to the smallest dimension first (up to grid limit)
        if (px < static_cast<int>(nx) && px <= py && px <= pz) { ++px; }
        else if (py < static_cast<int>(ny) && py <= pz) { ++py; }
        else if (pz < static_cast<int>(nz)) { ++pz; }
        else break;
        total = px * py * pz;
    }
    // If total still exceeds numRanks, reduce
    while (total > numRanks) {
        if (pz > 1) { --pz; }
        else if (py > 1) { --py; }
        else { --px; }
        total = px * py * pz;
    }
    // If we can't use all ranks, that's fine - some will be idle
}

// Map (procX, procY, procZ) to rank ID
inline int rankFromCoords(int px, int py, int pz, int nxp) {
    return pz * (nxp) + py * nxp + px;
}

void initializeGrid(std::vector<Real>& grid, const int lx0, const int ly0, const int lz0,
                    const size_t globalNx, const size_t globalNy, const size_t globalNz,
                    const size_t gxStart, const size_t gyStart, const size_t gzStart) {
    for (int lz = 0; lz < lz0; ++lz) {
        for (int ly = 0; ly < ly0; ++ly) {
            for (int lx = 0; lx < lx0; ++lx) {
                // Local index 0 is the halo, so owned cells start at local index 1
                // Global coord = gxStart + (local - 1), clamped to [0, globalDim)
                ssize_t gx = static_cast<ssize_t>(gxStart) + lx - 1;
                ssize_t gy = static_cast<ssize_t>(gyStart) + ly - 1;
                ssize_t gz = static_cast<ssize_t>(gzStart) + lz - 1;
                gx = std::max<ssize_t>(0, std::min<ssize_t>(gx, globalNx - 1));
                gy = std::max<ssize_t>(0, std::min<ssize_t>(gy, globalNy - 1));
                gz = std::max<ssize_t>(0, std::min<ssize_t>(gz, globalNz - 1));
                const size_t globalIdx = idx3(static_cast<size_t>(gx),
                                              static_cast<size_t>(gy),
                                              static_cast<size_t>(gz),
                                              globalNx, globalNy);
                grid[lidx3(lx, ly, lz, lx0, ly0)] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// Exchange halo layers with neighbors using non-blocking MPI.
// For each axis, two exchanges happen:
//   - My "plus" face (top/right/back) is sent to my "+" neighbor, who receives it as their "minus" face.
//   - My "minus" face (bottom/left/front) is sent to my "-" neighbor, who receives it as their "plus" face.
// Tags are consistent: TAG_Z_PLUS is used for all zPlusSend/zMinusRecv communications, etc.
void exchangeHalos(std::vector<Real>& grid, const int lx0, const int ly0, const int lz0,
                   int myRank, int px, int py, int pz) {
    const int xySize = lx0 * ly0;
    const int xzSize = lx0 * lz0;
    const int yzSize = ly0 * lz0;

    std::vector<Real> zMinusSend(xySize), zMinusRecv(xySize);
    std::vector<Real> zPlusSend(xySize), zPlusRecv(xySize);
    std::vector<Real> yMinusSend(xzSize), yMinusRecv(xzSize);
    std::vector<Real> yPlusSend(xzSize), yPlusRecv(xzSize);
    std::vector<Real> xMinusSend(yzSize), xMinusRecv(yzSize);
    std::vector<Real> xPlusSend(yzSize), xPlusRecv(yzSize);

    // Pack send buffers: send the owned face adjacent to each halo.
    // The minus neighbor needs my first owned face (local index 1).
    // The plus neighbor needs my last owned face (local index dim-2).
    for (int ly = 0; ly < ly0; ++ly)
        for (int lx = 0; lx < lx0; ++lx) {
            zMinusSend[ly * lx0 + lx] = grid[lidx3(lx, ly, 1, lx0, ly0)];
            zPlusSend[ly * lx0 + lx]  = grid[lidx3(lx, ly, lz0 - 2, lx0, ly0)];
        }
    for (int lz = 0; lz < lz0; ++lz)
        for (int lx = 0; lx < lx0; ++lx) {
            yMinusSend[lz * lx0 + lx] = grid[lidx3(lx, 1, lz, lx0, ly0)];
            yPlusSend[lz * lx0 + lx]  = grid[lidx3(lx, ly0 - 2, lz, lx0, ly0)];
        }
    for (int lz = 0; lz < lz0; ++lz)
        for (int ly = 0; ly < ly0; ++ly) {
            xMinusSend[lz * ly0 + ly] = grid[lidx3(1, ly, lz, lx0, ly0)];
            xPlusSend[lz * ly0 + ly]  = grid[lidx3(lx0 - 2, ly, lz, lx0, ly0)];
        }

    int myPx = myRank % px;
    int myPy = (myRank / px) % py;
    int myPz = myRank / (px * py);

    int zMinusRank = (myPz > 0) ? myRank - px * py : MPI_PROC_NULL;
    int zPlusRank  = (myPz < pz - 1) ? myRank + px * py : MPI_PROC_NULL;
    int yMinusRank = (myPy > 0) ? myRank - px : MPI_PROC_NULL;
    int yPlusRank  = (myPy < py - 1) ? myRank + px : MPI_PROC_NULL;
    int xMinusRank = (myPx > 0) ? myRank - 1 : MPI_PROC_NULL;
    int xPlusRank  = (myPx < px - 1) ? myRank + 1 : MPI_PROC_NULL;

    // Tags: consistent across all ranks for matching
    constexpr int TAG_Z_PLUS = 10, TAG_Z_MINUS = 11;
    constexpr int TAG_Y_PLUS = 12, TAG_Y_MINUS = 13;
    constexpr int TAG_X_PLUS = 14, TAG_X_MINUS = 15;

    MPI_Request requests[12];
    int count = 0;

    auto post_pair = [&](Real* sbuf, int ssize, int srank, int sTag,
                         Real* rbuf, int rsize, int rrank, int rTag) {
        MPI_Isend(sbuf, ssize, MPI_DOUBLE, srank, sTag, MPI_COMM_WORLD, &requests[count++]);
        MPI_Irecv(rbuf, rsize, MPI_DOUBLE, rrank, rTag, MPI_COMM_WORLD, &requests[count++]);
    };

    // Z+: send my top face to zPlusRank (they receive as their bottom = zMinusRecv)
    //     recv from zPlusRank their bottom face (they send zMinusSend) → I receive as zPlusRecv
    post_pair(zPlusSend.data(), xySize, zPlusRank, TAG_Z_PLUS,
              zMinusRecv.data(), xySize, zMinusRank, TAG_Z_PLUS);
    // Z-: send my bottom face to zMinusRank (they receive as their top = zPlusRecv)
    //     recv from zMinusRank their top face (they send zPlusSend) → I receive as zMinusRecv
    post_pair(zMinusSend.data(), xySize, zMinusRank, TAG_Z_MINUS,
              zPlusRecv.data(), xySize, zPlusRank, TAG_Z_MINUS);

    // Y+: send my back face to yPlusRank → they receive as yMinusRecv
    post_pair(yPlusSend.data(), xzSize, yPlusRank, TAG_Y_PLUS,
              yMinusRecv.data(), xzSize, yMinusRank, TAG_Y_PLUS);
    post_pair(yMinusSend.data(), xzSize, yMinusRank, TAG_Y_MINUS,
              yPlusRecv.data(), xzSize, yPlusRank, TAG_Y_MINUS);

    // X+: send my right face to xPlusRank → they receive as xMinusRecv
    post_pair(xPlusSend.data(), yzSize, xPlusRank, TAG_X_PLUS,
              xMinusRecv.data(), yzSize, xMinusRank, TAG_X_PLUS);
    post_pair(xMinusSend.data(), yzSize, xMinusRank, TAG_X_MINUS,
              xPlusRecv.data(), yzSize, xPlusRank, TAG_X_MINUS);

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

    // Unpack recv buffers into halo layers
    for (int ly = 0; ly < ly0; ++ly)
        for (int lx = 0; lx < lx0; ++lx) {
            grid[lidx3(lx, ly, 0, lx0, ly0)]        = zMinusRecv[ly * lx0 + lx];
            grid[lidx3(lx, ly, lz0 - 1, lx0, ly0)]  = zPlusRecv[ly * lx0 + lx];
        }
    for (int lz = 0; lz < lz0; ++lz)
        for (int lx = 0; lx < lx0; ++lx) {
            grid[lidx3(lx, 0, lz, lx0, ly0)]        = yMinusRecv[lz * lx0 + lx];
            grid[lidx3(lx, ly0 - 1, lz, lx0, ly0)]  = yPlusRecv[lz * lx0 + lx];
        }
    for (int lz = 0; lz < lz0; ++lz)
        for (int ly = 0; ly < ly0; ++ly) {
            grid[lidx3(0, ly, lz, lx0, ly0)]        = xMinusRecv[lz * ly0 + ly];
            grid[lidx3(lx0 - 1, ly, lz, lx0, ly0)]  = xPlusRecv[lz * ly0 + ly];
        }
}

// 7-point stencil computation on local subdomain (interior only, excluding halo)
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const int lx0, const int ly0, const int lz0,
                      const size_t globalNx, const size_t globalNy, const size_t globalNz,
                      const size_t gxStart, const size_t gyStart, const size_t gzStart) {
    // Process interior points (excluding halo layer on all sides)
    for (int lz = 1; lz < lz0 - 1; ++lz) {
        for (int ly = 1; ly < ly0 - 1; ++ly) {
            for (int lx = 1; lx < lx0 - 1; ++lx) {
                const size_t idx = lidx3(lx, ly, lz, lx0, ly0);

                const Real center = input[idx];
                const Real left   = input[lidx3(lx - 1, ly, lz, lx0, ly0)];
                const Real right  = input[lidx3(lx + 1, ly, lz, lx0, ly0)];
                const Real front  = input[lidx3(lx, ly - 1, lz, lx0, ly0)];
                const Real back   = input[lidx3(lx, ly + 1, lz, lx0, ly0)];
                const Real bottom = input[lidx3(lx, ly, lz - 1, lx0, ly0)];
                const Real top    = input[lidx3(lx, ly, lz + 1, lx0, ly0)];

                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }

    // Copy boundary/halo values from input to output
    for (int lz = 0; lz < lz0; ++lz) {
        for (int ly = 0; ly < ly0; ++ly) {
            for (int lx = 0; lx < lx0; ++lx) {
                // Global coordinates (local index 0 is halo, so offset by -1)
                ssize_t gx = static_cast<ssize_t>(gxStart) + lx - 1;
                ssize_t gy = static_cast<ssize_t>(gyStart) + ly - 1;
                ssize_t gz = static_cast<ssize_t>(gzStart) + lz - 1;

                // Check if this is a global boundary — these must not change
                if (gx == 0 || gx == static_cast<ssize_t>(globalNx) - 1 ||
                    gy == 0 || gy == static_cast<ssize_t>(globalNy) - 1 ||
                    gz == 0 || gz == static_cast<ssize_t>(globalNz) - 1) {
                    const size_t idx = lidx3(lx, ly, lz, lx0, ly0);
                    output[idx] = input[idx];
                }
                // Also copy halo cells (they'll be overwritten by exchangeHalos)
                else if (lz == 0 || lz == lz0 - 1 ||
                         ly == 0 || ly == ly0 - 1 ||
                         lx == 0 || lx == lx0 - 1) {
                    const size_t idx = lidx3(lx, ly, lz, lx0, ly0);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

int main(int argc, char** argv) {
    int mpiRank, mpiSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same args)
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // 3D domain decomposition
    int px, py, pz;
    computeDecomposition(mpiSize, nx, ny, nz, px, py, pz);

    // Determine my position in the 3D processor grid
    int myPx = mpiRank % px;
    int myPy = (mpiRank / px) % py;
    int myPz = mpiRank / (px * py);

    // Compute global start coordinates for this rank's subdomain
    // Use floor-based division for even distribution
    // Split dimension D into p chunks: chunk i starts at floor(i * D / p)
    auto chunkStart = [](size_t dim, int rank, int procs) -> size_t {
        return static_cast<size_t>(rank) * dim / procs;
    };
    auto chunkEnd = [](size_t dim, int rank, int procs) -> size_t {
        return static_cast<size_t>(rank + 1) * dim / procs;
    };

    const size_t gxStart = chunkStart(nx, myPx, px);
    const size_t gxEnd   = chunkEnd(nx, myPx, px);
    const size_t gyStart = chunkStart(ny, myPy, py);
    const size_t gyEnd   = chunkEnd(ny, myPy, py);
    const size_t gzStart = chunkStart(nz, myPz, pz);
    const size_t gzEnd   = chunkEnd(nz, myPz, pz);

    // Local dimensions (including 1-cell halo on each side)
    const int lx0 = static_cast<int>(gxEnd - gxStart) + 2;
    const int ly0 = static_cast<int>(gyEnd - gyStart) + 2;
    const int lz0 = static_cast<int>(gzEnd - gzStart) + 2;

    const size_t localSize = static_cast<size_t>(lx0) * ly0 * lz0;

    // Only active ranks participate (ranks beyond px*py*pz are idle)
    bool isActive = (mpiRank < px * py * pz);

    // Allocate grids (double buffering) with halo
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    if (mpiRank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d (decomposition: %d x %d x %d)\n", mpiSize, px, py, pz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (isActive) {
        // Initialize local grid
        if (mpiRank == 0) printf("Initializing grid...\n");
        initializeGrid(grid1, lx0, ly0, lz0, nx, ny, nz, gxStart, gyStart, gzStart);

        // Exchange initial halos so boundary conditions are correct
        exchangeHalos(grid1, lx0, ly0, lz0, mpiRank, px, py, pz);

        // Run stencil iterations
        if (mpiRank == 0) printf("Running stencil computation...\n");
        auto start = std::chrono::high_resolution_clock::now();
        MPI_Barrier(MPI_COMM_WORLD);
        start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, lx0, ly0, lz0,
                                 nx, ny, nz, gxStart, gyStart, gzStart);
                exchangeHalos(grid2, lx0, ly0, lz0, mpiRank, px, py, pz);
            } else {
                stencilIteration(grid2, grid1, lx0, ly0, lz0,
                                 nx, ny, nz, gxStart, gyStart, gzStart);
                exchangeHalos(grid1, lx0, ly0, lz0, mpiRank, px, py, pz);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Calculate performance metrics
        // Each rank computes its interior cells
        double localCellUpdates = 0.0;
        if (lx0 > 2 && ly0 > 2 && lz0 > 2) {
            localCellUpdates = static_cast<double>((lx0 - 2) * (ly0 - 2) * (lz0 - 2)) * iterations;
        }
        double totalCellUpdates;
        MPI_Reduce(&localCellUpdates, &totalCellUpdates, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Broadcast timing from rank 0
        double durationMs = static_cast<double>(duration.count());
        double durationAll;
        MPI_Bcast(&durationMs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        durationAll = durationMs;

        double mcups = totalCellUpdates / (durationAll / 1000.0) / 1e6;

        if (mpiRank == 0) {
            printf("Computation time: %.0f ms\n", durationAll);
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // Gather full grid on rank 0 for validation/results
        const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

        if (printResults || validate) {
            // Gather only the owned (non-halo) cells from each rank to rank 0.
            // Each rank owns (lx0-2) x (ly0-2) x (lz0-2) interior cells.
            // These map to global coords [gxStart, gxEnd) x [gyStart, gyEnd) x [gzStart, gzEnd).
            size_t gridSize = nx * ny * nz;
            std::vector<Real> fullGrid;
            if (mpiRank == 0) fullGrid.resize(gridSize);

            // Extract owned cells (skip 1-cell halo on each side) into a contiguous buffer
            const int ownX = lx0 - 2;
            const int ownY = ly0 - 2;
            const int ownZ = lz0 - 2;
            const int ownSize = ownX * ownY * ownZ;
            std::vector<Real> ownedCells(ownSize);
            for (int lz = 0; lz < ownZ; ++lz)
                for (int ly = 0; ly < ownY; ++ly)
                    for (int lx = 0; lx < ownX; ++lx)
                        ownedCells[lz * ownY * ownX + ly * ownX + lx]
                            = finalGrid[lidx3(lx + 1, ly + 1, lz + 1, lx0, ly0)];

            // Gather all owned cells contiguously to rank 0
            std::vector<int> recvcounts(px * py * pz);
            std::vector<int> displs(px * py * pz);
            int totalOwned = 0;
            for (int pz_ = 0; pz_ < pz; ++pz_) {
                for (int py_ = 0; py_ < py; ++py_) {
                    for (int px_ = 0; px_ < px; ++px_) {
                        const int r = rankFromCoords(px_, py_, pz_, px);
                        const size_t gex = chunkEnd(nx, px_, px);
                        const size_t gsx = chunkStart(nx, px_, px);
                        const size_t ges = chunkEnd(ny, py_, py);
                        const size_t gsy = chunkStart(ny, py_, py);
                        const size_t gez = chunkEnd(nz, pz_, pz);
                        const size_t gsz = chunkStart(nz, pz_, pz);
                        const int ox = static_cast<int>(gex - gsx);
                        const int oy = static_cast<int>(ges - gsy);
                        const int oz = static_cast<int>(gez - gsz);
                        recvcounts[r] = ox * oy * oz;
                        displs[r] = totalOwned;
                        totalOwned += ox * oy * oz;
                    }
                }
            }

            std::vector<Real> gathered;
            if (mpiRank == 0) gathered.resize(totalOwned);

            MPI_Gatherv(ownedCells.data(), ownSize, MPI_DOUBLE,
                        mpiRank == 0 ? gathered.data() : nullptr,
                        recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);

            if (mpiRank == 0) {
                // Scatter gathered data into fullGrid at correct global positions
                int offset = 0;
                for (int pz_ = 0; pz_ < pz; ++pz_) {
                    for (int py_ = 0; py_ < py; ++py_) {
                        for (int px_ = 0; px_ < px; ++px_) {
                            const size_t gsx = chunkStart(nx, px_, px);
                            const size_t gex = chunkEnd(nx, px_, px);
                            const size_t gsy = chunkStart(ny, py_, py);
                            const size_t ges = chunkEnd(ny, py_, py);
                            const size_t gsz = chunkStart(nz, pz_, pz);
                            const size_t gez = chunkEnd(nz, pz_, pz);
                            const int ox = static_cast<int>(gex - gsx);
                            const int oy = static_cast<int>(ges - gsy);
                            const int oz = static_cast<int>(gez - gsz);

                            for (int gz = 0; gz < oz; ++gz)
                                for (int gy = 0; gy < oy; ++gy)
                                    for (int gx = 0; gx < ox; ++gx)
                                        fullGrid[idx3(gsx + gx, gsy + gy, gsz + gz, nx, ny)]
                                            = gathered[offset++];
                        }
                    }
                }

                if (printResults) {
                    print_results(fullGrid, "Grid");
                }

                if (validate) {
                    printf("Validating result...\n");
                    bool valid = validateResult(fullGrid, nx, ny, nz);
                    if (valid) {
                        printf("Validation: PASSED\n");
                        MPI_Finalize();
                        return 0;
                    } else {
                        printf("Validation: FAILED\n");
                        MPI_Finalize();
                        return 1;
                    }
                }
            }
        } else {
            (void)finalGrid;
        }
    }

    MPI_Finalize();
    return 0;
}
