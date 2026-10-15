#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize local slab (owned planes z0..z0+lz-1 plus one halo plane on each side).
// Local plane p (0..lz+1) corresponds to global plane z0 + p - 1.
void initializeSlab(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t z0, const size_t lz, const size_t nz) {
    const size_t plane = nx * ny;
    for (size_t p = 0; p < lz + 2; ++p) {
        const size_t gz = z0 + p;  // global plane + 1
        if (gz < 1 || gz > nz) {
            continue;  // outside global domain (never read)
        }
        const size_t base = (gz - 1) * plane;
        Real* row = &grid[p * plane];
        for (size_t i = 0; i < plane; ++i) {
            row[i] = ((base + i) % 19) * 1.0;
        }
    }
}

// 7-point stencil on local planes [pBegin, pEnd] (inclusive, local indices)
static inline void stencilPlanes(const Real* __restrict input, Real* __restrict output,
                                 const size_t nx, const size_t ny,
                                 const size_t pBegin, const size_t pEnd) {
    if (nx < 3 || ny < 3) return;
    const size_t plane = nx * ny;
    for (size_t p = pBegin; p <= pEnd; ++p) {
        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = p * plane + y * nx;
            const Real* __restrict c = input + row;
            const Real* __restrict f = input + row - nx;
            const Real* __restrict bk = input + row + nx;
            const Real* __restrict bo = input + row - plane;
            const Real* __restrict t = input + row + plane;
            Real* __restrict o = output + row;
            for (size_t x = 1; x < nx - 1; ++x) {
                o[x] = (c[x] + c[x-1] + c[x+1] + f[x] + bk[x] + bo[x] + t[x]) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t nOwned, MPI_Comm comm, int rank) {
    // 1. No NaN or Inf values
    int bad = 0;
    for (size_t i = 0; i < nOwned; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) { bad = 1; break; }
    }
    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real mm[2] = {grid[0], -grid[0]};  // min, -max
    for (size_t i = 0; i < nOwned; ++i) {
        mm[0] = std::min(mm[0], grid[i]);
        mm[1] = std::min(mm[1], -grid[i]);
    }
    Real gmm[2];
    MPI_Allreduce(mm, gmm, 2, MPI_DOUBLE, MPI_MIN, comm);
    const Real minVal = gmm[0], maxVal = -gmm[1];

    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    // 1D decomposition along z; only as many ranks as there are planes participate
    const int active = (int)std::min<size_t>((size_t)worldSize, std::max<size_t>(nz, 1));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    auto zStart = [&](int r) { return (nz / nranks) * r + std::min<size_t>(r, nz % nranks); };
    const size_t z0 = zStart(rank);
    const size_t z1 = zStart(rank + 1);
    const size_t lz = z1 - z0;
    const size_t plane = nx * ny;
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank < nranks - 1 ? rank + 1 : MPI_PROC_NULL;

    std::vector<Real> grid1((lz + 2) * plane);
    std::vector<Real> grid2;

    if (rank == 0) printf("Initializing grid...\n");
    initializeSlab(grid1, nx, ny, z0, lz, nz);
    // Boundary cells are never written, so both buffers start identical
    grid2 = grid1;

    // Local planes updated by the stencil (exclude global boundary planes)
    const size_t updBegin = (z0 == 0) ? 2 : 1;
    const size_t updEnd = (z1 == nz) ? lz - 1 : lz;  // may underflow only if lz==0, impossible
    // Planes not requiring halo data
    const size_t innerBegin = std::max<size_t>(updBegin, 2);
    const size_t innerEnd = (lz >= 1) ? std::min<size_t>(updEnd, lz - 1) : 0;

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    Real* cur = grid1.data();
    Real* nxt = grid2.data();
    const bool doStencil = nz >= 3;
    for (int iter = 0; iter < iterations && doStencil; ++iter) {
        // Halo exchange (plane-sized messages; chunk-safe for counts < 2^31)
        MPI_Request r[4];
        MPI_Irecv(cur, (int)plane, MPI_DOUBLE, prev, 0, comm, &r[0]);
        MPI_Irecv(cur + (lz + 1) * plane, (int)plane, MPI_DOUBLE, next, 1, comm, &r[1]);
        MPI_Isend(cur + plane, (int)plane, MPI_DOUBLE, prev, 1, comm, &r[2]);
        MPI_Isend(cur + lz * plane, (int)plane, MPI_DOUBLE, next, 0, comm, &r[3]);

        // Overlap: interior planes
        if (innerBegin <= innerEnd) stencilPlanes(cur, nxt, nx, ny, innerBegin, innerEnd);

        MPI_Waitall(4, r, MPI_STATUSES_IGNORE);

        // Edge planes needing halos
        if (updBegin <= 1 && 1 <= updEnd) stencilPlanes(cur, nxt, nx, ny, 1, 1);
        if (lz > 1 && updBegin <= lz && lz <= updEnd) stencilPlanes(cur, nxt, nx, ny, lz, lz);

        std::swap(cur, nxt);
    }

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localMs = duration.count();
    long globalMs = 0;
    MPI_Reduce(&localMs, &globalMs, 1, MPI_LONG, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalMs);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (globalMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // doStencil false => no swaps happened; parity otherwise follows iteration count
    const Real* finalLocal = (doStencil && iterations > 0) ? cur : grid1.data();
    // Without stencil, cur == grid1 anyway.

    if (printResults) {
        std::vector<Real> full;
        if (rank == 0) {
            full.resize(nx * ny * nz);
            std::memcpy(full.data(), finalLocal + plane, lz * plane * sizeof(Real));
            for (int r = 1; r < nranks; ++r) {
                const size_t rz0 = zStart(r), rlz = zStart(r + 1) - rz0;
                Real* dst = full.data() + rz0 * plane;
                size_t total = rlz * plane, off = 0;
                const size_t chunk = (size_t)1 << 26;
                while (off < total) {
                    const int n = (int)std::min(chunk, total - off);
                    MPI_Recv(dst + off, n, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    off += n;
                }
            }
            print_results(full, "Grid");
        } else {
            size_t total = lz * plane, off = 0;
            const size_t chunk = (size_t)1 << 26;
            while (off < total) {
                const int n = (int)std::min(chunk, total - off);
                MPI_Send(finalLocal + plane + off, n, MPI_DOUBLE, 0, 2, comm);
                off += n;
            }
        }
    }

    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        // Owned planes start at local plane 1; build pointer-based view
        std::vector<Real> owned(finalLocal + plane, finalLocal + (lz + 1) * plane);
        bool valid = validateResult(owned, owned.size(), comm, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return rc;
}
