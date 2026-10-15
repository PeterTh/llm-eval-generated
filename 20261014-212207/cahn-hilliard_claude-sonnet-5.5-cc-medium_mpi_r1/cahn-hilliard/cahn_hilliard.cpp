#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Domain is decomposed into slabs along Z. Each rank stores its planes plus one halo plane on
// each side (local plane 0 and nzl+1). At the global boundaries the halo plane is a copy of the
// adjacent edge plane, which reproduces the clamped boundary condition with a uniform stencil.

// Laplacian for one row; c = row (z,y), cyp/cyn = rows y+1/y-1 (clamped), czp/czn = rows z+1/z-1
static inline void laplacianRow(double* __restrict out, const double* __restrict c,
                                const double* __restrict cyp, const double* __restrict cyn,
                                const double* __restrict czp, const double* __restrict czn,
                                const size_t nx,
                                const double dx, const double dy, const double dz) {
    auto edge = [&](size_t x) {
        const size_t xp = (x < nx - 1) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : 0;
        const double cxx = (c[xp] + c[xn] - 2.0 * c[x]) / (dx * dx);
        const double cyy = (cyp[x] + cyn[x] - 2.0 * c[x]) / (dy * dy);
        const double czz = (czp[x] + czn[x] - 2.0 * c[x]) / (dz * dz);
        out[x] = cxx + cyy + czz;
    };
    edge(0);
    if (nx > 1) edge(nx - 1);
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    for (size_t x = 1; x + 1 < nx; ++x) {
        const double cxx = (c[x + 1] + c[x - 1] - 2.0 * c[x]) / dx2;
        const double cyy = (cyp[x] + cyn[x] - 2.0 * c[x]) / dy2;
        const double czz = (czp[x] + czn[x] - 2.0 * c[x]) / dz2;
        out[x] = cxx + cyy + czz;
    }
}

// Chemical potential for local plane lz (1..nzl)
static void chemPotentialPlane(const double* c, double* mu, const size_t lz,
                               const size_t nx, const size_t ny,
                               const double dx, const double dy, const double dz,
                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t P = nx * ny;
    for (size_t y = 0; y < ny; ++y) {
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;
        const double* row = c + lz * P + y * nx;
        double* o = mu + lz * P + y * nx;
        laplacianRow(o, row, c + lz * P + yp * nx, c + lz * P + yn * nx,
                     c + (lz + 1) * P + y * nx, c + (lz - 1) * P + y * nx, nx, dx, dy, dz);
        // o currently holds the Laplacian; combine in place
        for (size_t x = 0; x < nx; ++x) {
            const double cv = row[x];
            o[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv
                   - gamma * o[x];
        }
    }
}

// Cahn-Hilliard update for local plane lz
static void updatePlane(double* cnew, const double* cold, const double* mu, const size_t lz,
                        const size_t nx, const size_t ny,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t P = nx * ny;
    for (size_t y = 0; y < ny; ++y) {
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;
        double* o = cnew + lz * P + y * nx;
        laplacianRow(o, mu + lz * P + y * nx, mu + lz * P + yp * nx, mu + lz * P + yn * nx,
                     mu + (lz + 1) * P + y * nx, mu + (lz - 1) * P + y * nx, nx, dx, dy, dz);
        const double* co = cold + lz * P + y * nx;
        for (size_t x = 0; x < nx; ++x) o[x] = co[x] + dt * D * o[x];
    }
}

// Initialize local planes 0..nzl+1 (including halos, clamped globally); z0 = global z of plane 1
static void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                                    const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;
    for (size_t lz = 0; lz < nzl + 2; ++lz) {
        long gz = (long)z0 + (long)lz - 1;
        if (gz < 0) gz = 0;
        if (gz > (long)nz - 1) gz = (long)nz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = (size_t)gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[lz * nx * ny + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Distributed validation over the interior planes of each rank; output only on rank 0
bool validateResult(const double* c, const size_t n, MPI_Comm comm, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
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
    int wrank = 0, wsize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    int rc = 0;
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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
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
    
    size_t gridSize = nx * ny * nz;
    const size_t P = nx * ny;
    
    // Slab decomposition along Z; ranks beyond nz stay idle
    const int active = (int)std::min<size_t>((size_t)wsize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < active ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    const size_t base = nz / size, rem = nz % size;
    const size_t nzl = base + ((size_t)rank < rem ? 1 : 0);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, rem);
    const int down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int up = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    
    std::vector<double> cold((nzl + 2) * P);
    std::vector<double> cnew((nzl + 2) * P);
    std::vector<double> mu((nzl + 2) * P);
    
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z0, nzl);
    // Keep halo planes of the other buffers initialised
    cnew = cold;
    
    // Halo exchange start: send edge planes, receive into halo planes; clamped at global edges
    auto startExchange = [&](double* f, MPI_Request* req) {
        MPI_Irecv(f, (int)P, MPI_DOUBLE, down, 0, comm, &req[0]);
        MPI_Irecv(f + (nzl + 1) * P, (int)P, MPI_DOUBLE, up, 1, comm, &req[1]);
        MPI_Isend(f + P, (int)P, MPI_DOUBLE, down, 1, comm, &req[2]);
        MPI_Isend(f + nzl * P, (int)P, MPI_DOUBLE, up, 0, comm, &req[3]);
    };
    auto finishExchange = [&](double* f, MPI_Request* req) {
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (rank == 0) std::memcpy(f, f + P, P * sizeof(double));
        if (rank == size - 1) std::memcpy(f + (nzl + 1) * P, f + nzl * P, P * sizeof(double));
    };
    
    MPI_Barrier(comm);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    MPI_Request req[4];
    for (int t = 0; t < iterations; ++t) {
        // Chemical potential: edge planes first, overlap mu halo exchange with interior
        chemPotentialPlane(cold.data(), mu.data(), 1, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        if (nzl > 1)
            chemPotentialPlane(cold.data(), mu.data(), nzl, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        startExchange(mu.data(), req);
        for (size_t lz = 2; lz + 1 <= nzl; ++lz)
            chemPotentialPlane(cold.data(), mu.data(), lz, nx, ny, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        finishExchange(mu.data(), req);
        
        // Update: edge planes first, overlap concentration halo exchange with interior
        updatePlane(cnew.data(), cold.data(), mu.data(), 1, nx, ny, D, dt, dx, dy, dz);
        if (nzl > 1)
            updatePlane(cnew.data(), cold.data(), mu.data(), nzl, nx, ny, D, dt, dx, dy, dz);
        startExchange(cnew.data(), req);
        for (size_t lz = 2; lz + 1 <= nzl; ++lz)
            updatePlane(cnew.data(), cold.data(), mu.data(), lz, nx, ny, D, dt, dx, dy, dz);
        finishExchange(cnew.data(), req);
        
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    if (printResults) {
        std::vector<double> full;
        std::vector<int> counts(size), displs(size);
        if (rank == 0) {
            full.resize(gridSize);
            for (int r = 0; r < size; ++r) {
                const size_t n = base + ((size_t)r < rem ? 1 : 0);
                const size_t z = (size_t)r * base + std::min((size_t)r, rem);
                counts[r] = (int)(n * P);
                displs[r] = (int)(z * P);
            }
        }
        MPI_Gatherv(cold.data() + P, (int)(nzl * P), MPI_DOUBLE, full.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(full, "Concentration");
    }
    
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold.data() + P, nzl * P, comm, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }
    
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return rc;
}
