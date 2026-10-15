#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Row-wise 7-point Laplacian with clamped boundaries. Calls f(x, laplacian) for every x in the row.
// c: center row, ym/yp: rows at y-1/y+1 (clamped), zm/zp: rows at z-1/z+1 (clamped)
template <typename F>
static inline void laplacianRow(const double* __restrict c, const double* __restrict ym, const double* __restrict yp,
                                const double* __restrict zm, const double* __restrict zp, const size_t nx,
                                const double dx2, const double dy2, const double dz2, F&& f) {
    auto edge = [&](const size_t x) {
        const size_t xp = (x < nx - 1) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : 0;
        const double cxx = (c[xp] + c[xn] - 2.0 * c[x]) / dx2;
        const double cyy = (yp[x] + ym[x] - 2.0 * c[x]) / dy2;
        const double czz = (zp[x] + zm[x] - 2.0 * c[x]) / dz2;
        f(x, cxx + cyy + czz);
    };
    edge(0);
    if (nx > 1) {
        for (size_t x = 1; x + 1 < nx; ++x) {
            const double cxx = (c[x + 1] + c[x - 1] - 2.0 * c[x]) / dx2;
            const double cyy = (yp[x] + ym[x] - 2.0 * c[x]) / dy2;
            const double czz = (zp[x] + zm[x] - 2.0 * c[x]) / dz2;
            f(x, cxx + cyy + czz);
        }
        edge(nx - 1);
    }
}

// Chemical potential for local plane k (c has plane offset 2, mu offset 1). gz = global z index.
static void computeMuPlane(const double* __restrict c, double* __restrict mu, const long k, const size_t gz,
                           const size_t nx, const size_t ny, const size_t nz,
                           const double dx2, const double dy2, const double dz2,
                           const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const double* cc = c + (k + 2) * (long)plane;
    const double* cm = (gz > 0) ? cc - plane : cc;
    const double* cp = (gz < nz - 1) ? cc + plane : cc;
    double* m = mu + (k + 1) * (long)plane;
    for (size_t y = 0; y < ny; ++y) {
        const double* row = cc + y * nx;
        const double* ym = (y > 0) ? row - nx : row;
        const double* yp = (y < ny - 1) ? row + nx : row;
        double* mrow = m + y * nx;
        laplacianRow(row, ym, yp, cm + y * nx, cp + y * nx, nx, dx2, dy2, dz2, [&](size_t x, double lap) {
            const double cv = row[x];
            mrow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - gamma * lap;
        });
    }
}

// Update for local plane k
static void updatePlane(double* __restrict cnew, const double* __restrict cold, const double* __restrict mu,
                        const long k, const size_t gz, const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx2, const double dy2, const double dz2) {
    const size_t plane = nx * ny;
    const double* mc = mu + (k + 1) * (long)plane;
    const double* mm = (gz > 0) ? mc - plane : mc;
    const double* mp = (gz < nz - 1) ? mc + plane : mc;
    const double* co = cold + (k + 2) * (long)plane;
    double* cn = cnew + (k + 2) * (long)plane;
    for (size_t y = 0; y < ny; ++y) {
        const double* row = mc + y * nx;
        const double* ym = (y > 0) ? row - nx : row;
        const double* yp = (y < ny - 1) ? row + nx : row;
        const double* orow = co + y * nx;
        double* nrow = cn + y * nx;
        laplacianRow(row, ym, yp, mm + y * nx, mp + y * nx, nx, dx2, dy2, dz2, [&](size_t x, double lap) {
            nrow[x] = orow[x] + dt * D * lap;
        });
    }
}

// Initialize local slab [z0, z0+lz) of the concentration field (plane offset 2)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t lz) {
    const size_t vol = nx * ny * nz;
    for (size_t k = 0; k < lz; ++k) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = (z0 + k) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[(k + 2) * nx * ny + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Distributed validation over the local interior [off, off+n) of each rank
bool validateResult(const double* c, const size_t n, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    double gmin, gmax;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 10.0 || gmin < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // 1D slab decomposition along z; each active rank owns at least 2 planes
    const int active = (int)std::max<size_t>(1, std::min<size_t>((size_t)size, nz / 2));
    const bool isActive = rank < active;
    const size_t base = nz / active, rem = nz % active;
    auto slabLen = [&](int r) { return base + ((size_t)r < rem ? 1 : 0); };
    auto slabStart = [&](int r) { return (size_t)r * base + std::min((size_t)r, rem); };
    const size_t lz = isActive ? slabLen(rank) : 0;
    const size_t z0 = isActive ? slabStart(rank) : 0;
    const bool hasLower = isActive && rank > 0;
    const bool hasUpper = isActive && rank < active - 1;
    const int lower = hasLower ? rank - 1 : MPI_PROC_NULL;
    const int upper = hasUpper ? rank + 1 : MPI_PROC_NULL;

    // c arrays: lz + 4 planes (2 halo each side); mu: lz + 2 planes (1 halo each side)
    std::vector<double> cold, cnew, mu;
    if (isActive) {
        cold.assign((lz + 4) * plane, 0.0);
        cnew.assign((lz + 4) * plane, 0.0);
        mu.assign((lz + 2) * plane, 0.0);
        initializeConcentration(cold, nx, ny, nz, z0, lz);
    }

    if (rank == 0) {
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const long L = (long)lz;
    const long kLo = hasLower ? -1 : 0;
    const long kHi = hasUpper ? L : L - 1;
    const int haloCount = (int)(2 * plane);

    if (isActive) {
        for (int t = 0; t < iterations; ++t) {
            double* c = cold.data();
            MPI_Request req[4];
            // send my lowest/highest 2 planes, receive neighbors' into halos
            MPI_Irecv(c, haloCount, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(c + (lz + 2) * plane, haloCount, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Isend(c + 2 * plane, haloCount, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[2]);
            MPI_Isend(c + lz * plane, haloCount, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[3]);

            // Interior planes do not depend on halos
            for (long k = 1; k <= L - 2; ++k)
                computeMuPlane(c, mu.data(), k, z0 + k, nx, ny, nz, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);

            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

            for (long k = kLo; k <= 0; ++k)
                computeMuPlane(c, mu.data(), k, z0 + k, nx, ny, nz, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
            for (long k = std::max(1L, L - 1); k <= kHi; ++k)
                computeMuPlane(c, mu.data(), k, z0 + k, nx, ny, nz, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);

            for (long k = 0; k < L; ++k)
                updatePlane(cnew.data(), c, mu.data(), k, z0 + k, nx, ny, nz, D, dt, dx2, dy2, dz2);

            std::swap(cold, cnew);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double global_elapsed = 0.0;
    MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long long ms = (long long)(global_elapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", ms);
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather full field on rank 0)
    if (printResults) {
        std::vector<double> full;
        std::vector<int> counts(size, 0), displs(size, 0);
        for (int r = 0; r < active; ++r) {
            counts[r] = (int)(slabLen(r) * plane);
            displs[r] = (int)(slabStart(r) * plane);
        }
        if (rank == 0) full.resize(gridSize);
        MPI_Gatherv(isActive ? cold.data() + 2 * plane : nullptr, (int)(lz * plane), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "Concentration");
    }

    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(isActive ? cold.data() + 2 * plane : nullptr, lz * plane, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        ret = valid ? 0 : 1;
    }

    MPI_Finalize();
    return ret;
}
