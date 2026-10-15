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

// Each rank owns a contiguous slab of z-planes (1D decomposition along the slowest dimension).
// Local arrays cover global planes [g0, g1): owned interior planes plus one halo plane per side.
struct Slab {
    size_t g0 = 0, g1 = 0;   // global planes stored locally
    size_t cnt = 0;          // number of owned interior planes (computed each iteration)
    size_t lo = 0, hi = 0;   // global planes contributed to gather / validation: [lo, hi)
};

void initializeGrid(std::vector<Real>& grid, const Slab& s, const size_t nx, const size_t ny) {
    const size_t plane = nx * ny;
    for (size_t gz = s.g0; gz < s.g1; ++gz) {
        Real* p = grid.data() + (gz - s.g0) * plane;
        const size_t base = gz * plane;
        for (size_t i = 0; i < plane; ++i) {
            p[i] = ((base + i) % 19) * 1.0;
        }
    }
}

// Compute one local plane (lp = local plane index) of the 7-point stencil; x/y boundaries are copied.
static inline void stencilPlane(const Real* __restrict in, Real* __restrict out,
                                const size_t lp, const size_t nx, const size_t ny) {
    const size_t plane = nx * ny;
    const Real* c = in + lp * plane;
    const Real* bo = c - plane;
    const Real* to = c + plane;
    Real* o = out + lp * plane;
    // y boundary rows
    std::memcpy(o, c, nx * sizeof(Real));
    std::memcpy(o + (ny - 1) * nx, c + (ny - 1) * nx, nx * sizeof(Real));
    for (size_t y = 1; y < ny - 1; ++y) {
        const size_t r = y * nx;
        o[r] = c[r];
        o[r + nx - 1] = c[r + nx - 1];
        for (size_t x = 1; x < nx - 1; ++x) {
            const size_t i = r + x;
            o[i] = (c[i] + c[i - 1] + c[i + 1] + c[i - nx] + c[i + nx] + bo[i] + to[i]) / 7.0;
        }
    }
}

// Send/recv large buffers in chunks that fit an int count
static void sendBig(const Real* p, size_t n, int dst, MPI_Comm comm) {
    const size_t chunk = (size_t)1 << 27;
    for (size_t off = 0; off < n; off += chunk)
        MPI_Send(p + off, (int)std::min(chunk, n - off), MPI_DOUBLE, dst, 100, comm);
}
static void recvBig(Real* p, size_t n, int src, MPI_Comm comm) {
    const size_t chunk = (size_t)1 << 27;
    for (size_t off = 0; off < n; off += chunk)
        MPI_Recv(p + off, (int)std::min(chunk, n - off), MPI_DOUBLE, src, 100, comm, MPI_STATUS_IGNORE);
}

bool validateResult(const Real* data, size_t n, MPI_Comm comm, int rank) {
    int bad = 0;
    Real minVal = INFINITY, maxVal = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        const Real val = data[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    int gbad;
    MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm);
    if (gbad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    Real gmin, gmax;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 1e6 || gmin < -1e6) {
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
    int rank, nprocs;
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
        printf("MPI ranks: %d\n", nprocs);
    }

    const size_t plane = nx * ny;

    // Domain decomposition of interior z-planes [1, nz-1)
    const size_t zi = nz > 2 ? nz - 2 : 0;
    const size_t P = (size_t)nprocs;
    Slab s;
    size_t active;  // ranks 0..active-1 own planes
    if (zi == 0) {
        active = 1;
        if (rank == 0) { s.g0 = 0; s.g1 = nz; s.cnt = 0; s.lo = 0; s.hi = nz; }
    } else {
        active = std::min(P, zi);
        if ((size_t)rank < active) {
            const size_t base = zi / active, rem = zi % active, r = (size_t)rank;
            const size_t z0 = 1 + r * base + std::min(r, rem);
            s.cnt = base + (r < rem ? 1 : 0);
            const size_t z1 = z0 + s.cnt;
            s.g0 = z0 - 1;
            s.g1 = z1 + 1;
            s.lo = (r == 0) ? 0 : z0;
            s.hi = (r == active - 1) ? nz : z1;
        }
    }
    const bool isActive = (size_t)rank < active;
    const bool hasDown = isActive && rank > 0 && zi > 0;
    const bool hasUp = isActive && (size_t)rank + 1 < active && zi > 0;

    const size_t localPlanes = s.g1 - s.g0;
    std::vector<Real> grid1(localPlanes * plane);
    std::vector<Real> grid2(localPlanes * plane);

    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, s, nx, ny);
    // Boundary cells and halo planes of the second buffer must match (they are copied / constant)
    initializeGrid(grid2, s, nx, ny);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (zi > 0 && isActive) {
        const size_t cnt = s.cnt;
        for (int iter = 0; iter < iterations; ++iter) {
            Real* in = (iter % 2 == 0) ? grid1.data() : grid2.data();
            Real* out = (iter % 2 == 0) ? grid2.data() : grid1.data();
            MPI_Request req[4];
            int nreq = 0;
            if (hasDown) {
                MPI_Irecv(in, (int)plane, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req[nreq++]);
                MPI_Isend(in + plane, (int)plane, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &req[nreq++]);
            }
            if (hasUp) {
                MPI_Irecv(in + (cnt + 1) * plane, (int)plane, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &req[nreq++]);
                MPI_Isend(in + cnt * plane, (int)plane, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req[nreq++]);
            }
            // Interior planes need no halo data: overlap with communication
            for (size_t lp = 2; lp + 1 <= cnt; ++lp) stencilPlane(in, out, lp, nx, ny);
            if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
            stencilPlane(in, out, 1, nx, ny);
            if (cnt > 1) stencilPlane(in, out, cnt, nx, ny);
        }
    } else if (zi == 0 && rank == 0) {
        // No interior: every iteration just copies the grid; result is unchanged
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* mine = finalLocal.data() + (s.lo - s.g0) * plane;
    const size_t myCount = (s.hi - s.lo) * plane;

    if (printResults) {
        if (rank == 0) {
            std::vector<Real> full(nx * ny * nz);
            std::memcpy(full.data(), mine, myCount * sizeof(Real));
            for (size_t r = 1; r < active && zi > 0; ++r) {
                const size_t base = zi / active, rem = zi % active;
                const size_t z0 = 1 + r * base + std::min(r, rem);
                const size_t c = base + (r < rem ? 1 : 0);
                const size_t hi = (r == active - 1) ? nz : z0 + c;
                recvBig(full.data() + z0 * plane, (hi - z0) * plane, (int)r, MPI_COMM_WORLD);
            }
            print_results(full, "Grid");
        } else if (isActive && zi > 0) {
            sendBig(mine, myCount, 0, MPI_COMM_WORLD);
        }
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(mine, myCount, MPI_COMM_WORLD, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
