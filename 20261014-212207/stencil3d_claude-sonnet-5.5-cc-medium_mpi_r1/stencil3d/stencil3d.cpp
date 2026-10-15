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

// Slab decomposition along Z. Each rank owns planes [z0, z0+lz) of the global grid and keeps
// one halo plane on each side: local plane k corresponds to global plane z0 - 1 + k.
// Boundary cells are never modified, so both buffers are initialized identically and only
// interior cells are computed (equivalent to the original boundary copy).
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t z0, const size_t lz) {
    const size_t plane = nx * ny;
    // Includes halo planes (local 0 and lz+1) when they exist in the global grid
    for (size_t k = 0; k < lz + 2; ++k) {
        if (z0 + k == 0) continue;  // no halo below global plane 0 (never read)
        const size_t gz = z0 + k - 1;
        const size_t base = gz * plane;
        Real* p = &grid[k * plane];
        for (size_t i = 0; i < plane; ++i) {
            p[i] = ((base + i) % 19) * 1.0;
        }
    }
}

// 7-point stencil on local plane k (interior x,y only)
static inline void stencilPlane(const Real* __restrict in, Real* __restrict out,
                                const size_t nx, const size_t ny, const size_t k) {
    const size_t plane = nx * ny;
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = k * plane + y * nx;
        const Real* __restrict c = in + row;
        const Real* __restrict f = in + row - nx;
        const Real* __restrict bk = in + row + nx;
        const Real* __restrict bo = in + row - plane;
        const Real* __restrict t = in + row + plane;
        Real* __restrict o = out + row;
        for (size_t x = 1; x < nx - 1; ++x) {
            o[x] = (c[x] + c[x - 1] + c[x + 1] + f[x] + bk[x] + bo[x] + t[x]) / 7.0;
        }
    }
}

// Distributed sanity checks over the owned planes
bool validateResult(const Real* data, const size_t n, MPI_Comm comm, const int rank) {
    int bad = 0;
    Real minVal = data[0];
    Real maxVal = data[0];
    for (size_t i = 0; i < n; ++i) {
        const Real val = data[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
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
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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
        printf("Initializing grid...\n");
    }

    // Distribute interior Z planes evenly; use at most one rank per interior plane.
    const size_t nInterior = nz >= 3 ? nz - 2 : 0;
    int active = (int)std::min<size_t>((size_t)worldSize, std::max<size_t>(nInterior, 1));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Owned plane range per rank; first/last rank also own the global boundary planes
    auto zStart = [&](int r) -> size_t {
        if (r == 0) return 0;
        if (r >= size) return nz;
        return 1 + (size_t)((unsigned __int128)r * nInterior / size);
    };
    const size_t z0 = zStart(rank);
    const size_t lz = zStart(rank + 1) - z0;
    const size_t plane = nx * ny;

    std::vector<Real> grid1((lz + 2) * plane);
    std::vector<Real> grid2((lz + 2) * plane);
    initializeGrid(grid1, nx, ny, z0, lz);
    grid2 = grid1;

    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    MPI_Datatype planeType;
    MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    const bool hasStencil = nx >= 3 && ny >= 3 && nz >= 3;
    // Local planes that are global interior planes
    auto computeRange = [&](const Real* in, Real* out, size_t kb, size_t ke) {
        if (!hasStencil) return;
        for (size_t k = kb; k < ke; ++k) {
            const size_t gz = z0 + k - 1;
            if (gz == 0 || gz == nz - 1) continue;
            stencilPlane(in, out, nx, ny, k);
        }
    };

    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    Real* bufs[2] = {grid1.data(), grid2.data()};
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = bufs[iter & 1];
        Real* out = bufs[(iter & 1) ^ 1];
        MPI_Request req[4];
        int nreq = 0;
        if (size > 1) {
            MPI_Irecv(in, 1, planeType, prev, 0, comm, &req[nreq++]);
            MPI_Irecv(in + (lz + 1) * plane, 1, planeType, next, 1, comm, &req[nreq++]);
            MPI_Isend(in + lz * plane, 1, planeType, next, 0, comm, &req[nreq++]);
            MPI_Isend(in + plane, 1, planeType, prev, 1, comm, &req[nreq++]);
        }
        // Overlap: inner planes need no halo data
        if (lz > 2) computeRange(in, out, 2, lz);
        if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        computeRange(in, out, 1, std::min<size_t>(2, lz + 1));
        if (lz > 1) computeRange(in, out, lz, lz + 1);
    }

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const Real* finalLocal = bufs[iterations & 1];
    const Real* owned = finalLocal + plane;

    if (printResults) {
        std::vector<Real> full;
        std::vector<int> counts(size), displs(size);
        if (rank == 0) {
            full.resize(nx * ny * nz);
            for (int r = 0; r < size; ++r) {
                displs[r] = (int)zStart(r);
                counts[r] = (int)(zStart(r + 1) - zStart(r));
            }
        }
        MPI_Gatherv(owned, (int)lz, planeType, full.data(), counts.data(), displs.data(),
                    planeType, 0, comm);
        if (rank == 0) print_results(full, "Grid");
    }

    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(owned, lz * plane, comm, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        ret = valid ? 0 : 1;
    }

    MPI_Type_free(&planeType);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
