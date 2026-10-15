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

// Local slab: nl owned z-planes stored at local planes 1..nl, ghost planes at 0 and nl+1.
void initializeSlab(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nl, const size_t z0) {
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= nl; ++lz) {
        const size_t base = (z0 + lz - 1) * plane;  // global linear index of plane start
        Real* p = &grid[lz * plane];
        for (size_t i = 0; i < plane; ++i) {
            p[i] = ((base + i) % 19) * 1.0;
        }
    }
}

// Update one owned plane (local plane index lz) if it is a global interior plane.
inline void stencilPlane(const Real* __restrict in, Real* __restrict out,
                         const size_t nx, const size_t ny, const size_t lz) {
    const size_t plane = nx * ny;
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t row = lz * plane + y * nx;
        const Real* __restrict c = in + row;
        const Real* __restrict yn = in + row - nx;
        const Real* __restrict yp = in + row + nx;
        const Real* __restrict zn = in + row - plane;
        const Real* __restrict zp = in + row + plane;
        Real* __restrict o = out + row;
        for (size_t x = 1; x < nx - 1; ++x) {
            o[x] = (c[x] + c[x-1] + c[x+1] + yn[x] + yp[x] + zn[x] + zp[x]) / 7.0;
        }
    }
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
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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
            if (wrank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (wrank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (wrank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Use at most one rank per z-plane; surplus ranks idle.
    const int nactive = (int)std::min<size_t>((size_t)wsize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < nactive ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = wrank;
    const int np = nactive;

    // Block distribution of z-planes
    const size_t qz = nz / np, rz = nz % np;
    const size_t nl = qz + ((size_t)rank < rz ? 1 : 0);
    const size_t z0 = (size_t)rank * qz + std::min<size_t>((size_t)rank, rz);
    const size_t plane = nx * ny;

    std::vector<Real> grid1((nl + 2) * plane, 0.0);
    std::vector<Real> grid2((nl + 2) * plane, 0.0);

    if (rank == 0) printf("Initializing grid...\n");
    initializeSlab(grid1, nx, ny, nl, z0);
    // Boundary cells are never modified; keep both buffers identical on them.
    grid2 = grid1;

    if (rank == 0) printf("Running stencil computation...\n");

    const int down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int up = rank < np - 1 ? rank + 1 : MPI_PROC_NULL;
    const bool ownsLow = (z0 == 0);
    const bool ownsHigh = (z0 + nl == nz);
    // Owned local planes lz in [lo, hi] are global interior planes
    const size_t lo = ownsLow ? 2 : 1;
    const size_t hi = ownsHigh ? nl - 1 : nl;  // nl>=1; if ownsHigh and nl==1, hi=0 -> empty

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    std::vector<Real>* cur = &grid1;
    std::vector<Real>* nxt = &grid2;
    for (int iter = 0; iter < iterations; ++iter) {
        Real* c = cur->data();
        Real* n = nxt->data();
        MPI_Request req[4];
        int nr = 0;
        MPI_Irecv(c, (int)plane, MPI_DOUBLE, down, 0, comm, &req[nr++]);
        MPI_Irecv(c + (nl + 1) * plane, (int)plane, MPI_DOUBLE, up, 1, comm, &req[nr++]);
        MPI_Isend(c + plane, (int)plane, MPI_DOUBLE, down, 1, comm, &req[nr++]);
        MPI_Isend(c + nl * plane, (int)plane, MPI_DOUBLE, up, 0, comm, &req[nr++]);

        // Planes not depending on ghosts: overlap with communication
        const size_t innerHi = std::min(hi, nl - 1);
        for (size_t lz = std::max<size_t>(lo, 2); lz <= innerHi; ++lz) {
            stencilPlane(c, n, nx, ny, lz);
        }
        MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        // Edge planes (depend on ghosts)
        const bool edgeLow = lo <= hi && lo == 1;
        const bool edgeHigh = lo <= hi && hi == nl && !(edgeLow && nl == 1);
        if (edgeLow) stencilPlane(c, n, nx, ny, 1);
        if (edgeHigh) stencilPlane(c, n, nx, ny, nl);
        std::swap(cur, nxt);
    }

    MPI_Barrier(comm);
    double elapsed = MPI_Wtime() - t0;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    long ms = (long)(maxElapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = *cur;  // == (iterations%2==0) ? grid1 : grid2

    if (printResults) {
        std::vector<Real> full;
        std::vector<int> counts(np), displs(np);
        if (rank == 0) {
            full.resize(nz * plane);
            for (int r = 0; r < np; ++r) {
                counts[r] = (int)(qz + ((size_t)r < rz ? 1 : 0));
                displs[r] = (int)((size_t)r * qz + std::min<size_t>((size_t)r, rz));
            }
        }
        MPI_Datatype planeType;
        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        MPI_Gatherv(finalLocal.data() + plane, (int)nl, planeType, full.data(), counts.data(), displs.data(),
                    planeType, 0, comm);
        MPI_Type_free(&planeType);
        if (rank == 0) print_results(full, "Grid");
    }

    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        // local checks
        int bad = 0;
        Real mn = INFINITY, mx = -INFINITY;
        const Real* d = finalLocal.data() + plane;
        for (size_t i = 0; i < nl * plane; ++i) {
            if (std::isnan(d[i]) || std::isinf(d[i])) bad = 1;
            mn = std::min(mn, d[i]);
            mx = std::max(mx, d[i]);
        }
        int gbad;
        Real gmn, gmx;
        MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&mn, &gmn, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&mx, &gmx, 1, MPI_DOUBLE, MPI_MAX, comm);
        bool valid = true;
        if (gbad) {
            if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        } else {
            if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmn, gmx);
            if (gmx > 1e6 || gmn < -1e6) {
                if (rank == 0) printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return rc;
}
