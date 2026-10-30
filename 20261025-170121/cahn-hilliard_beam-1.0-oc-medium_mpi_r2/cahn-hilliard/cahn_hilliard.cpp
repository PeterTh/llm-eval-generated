#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static inline size_t planeSize(const size_t nx, const size_t ny) { return nx * ny; }
static inline size_t localIndex(const size_t x, const size_t y, const size_t lz, const size_t nx, const size_t ny) {
    return (lz * ny + y) * nx + x;
}

// Halo exchange for z decomposition (ghost planes at lz=0 and lz=nz_local+1)
static void exchangeHalos(std::vector<double>& a, size_t nx, size_t ny, size_t nz_local, int rank, int size) {
    const size_t ps = planeSize(nx, ny);
    if (size == 1) {
        // clamped boundaries
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                a[localIndex(x, y, 0, nx, ny)] = a[localIndex(x, y, 1, nx, ny)];
                a[localIndex(x, y, nz_local + 1, nx, ny)] = a[localIndex(x, y, nz_local, nx, ny)];
            }
        return;
    }
    // send/recv with neighbors
    std::vector<double> sendDown(ps), sendUp(ps), recvDown(ps), recvUp(ps);
    for (size_t i = 0; i < ps; ++i) {
        sendDown[i] = a[localIndex(0,0,1,nx,ny) + i]; // not exactly; do plane copy below
    }
    // proper plane copies
    for (size_t y = 0; y < ny; ++y)
        for (size_t x = 0; x < nx; ++x) {
            sendDown[y * nx + x] = a[localIndex(x, y, 1, nx, ny)];
            sendUp[y * nx + x] = a[localIndex(x, y, nz_local, nx, ny)];
        }

    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Sendrecv(sendDown.data(), (int)ps, MPI_DOUBLE, prev, 0,
                 recvDown.data(), (int)ps, MPI_DOUBLE, prev, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendUp.data(), (int)ps, MPI_DOUBLE, next, 1,
                 recvUp.data(), (int)ps, MPI_DOUBLE, next, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (prev == MPI_PROC_NULL) {
        // clamped at bottom
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                a[localIndex(x, y, 0, nx, ny)] = a[localIndex(x, y, 1, nx, ny)];
    } else {
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                a[localIndex(x, y, 0, nx, ny)] = recvDown[y * nx + x];
    }
    if (next == MPI_PROC_NULL) {
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                a[localIndex(x, y, nz_local + 1, nx, ny)] = a[localIndex(x, y, nz_local, nx, ny)];
    } else {
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                a[localIndex(x, y, nz_local + 1, nx, ny)] = recvUp[y * nx + x];
    }
}

static double laplacian(const std::vector<double>& c, size_t nx, size_t ny, size_t nz_global,
                        double dx, double dy, double dz, size_t x, size_t y, size_t z_global, size_t lz) {
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;

    // map global z to local lz for z neighbors
    // zp local index: if z_global < nz-1 then lz+1 else lz (clamped)
    size_t lzp = (z_global < nz_global - 1) ? lz + 1 : lz;
    size_t lzn = (z_global > 0) ? lz - 1 : lz;

    const double cxx = (c[localIndex(xp, y, lz, nx, ny)] + c[localIndex(xn, y, lz, nx, ny)] - 2.0 * c[localIndex(x, y, lz, nx, ny)]) / (dx * dx);
    const double cyy = (c[localIndex(x, yp, lz, nx, ny)] + c[localIndex(x, yn, lz, nx, ny)] - 2.0 * c[localIndex(x, y, lz, nx, ny)]) / (dy * dy);
    const double czz = (c[localIndex(x, y, lzp, nx, ny)] + c[localIndex(x, y, lzn, nx, ny)] - 2.0 * c[localIndex(x, y, lz, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

static void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                                    size_t nx, size_t ny, size_t nz_global, size_t nz_local, size_t z_start,
                                    double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB) {
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        size_t z_global = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = localIndex(x, y, lz, nx, ny);
                double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * laplacian(c, nx, ny, nz_global, dx, dy, dz, x, y, z_global, lz);
            }
        }
    }
}

static void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               size_t nx, size_t ny, size_t nz_global, size_t nz_local, size_t z_start,
                               double D, double dt, double dx, double dy, double dz) {
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        size_t z_global = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = localIndex(x, y, lz, nx, ny);
                cnew[idx] = cold[idx] + dt * D * laplacian(mu, nx, ny, nz_global, dx, dy, dz, x, y, z_global, lz);
            }
        }
    }
}

static void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny, size_t nz_global, size_t nz_local, size_t z_start) {
    size_t vol = nx * ny * nz_global;
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        size_t z = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = localIndex(x, y, lz, nx, ny);
                size_t linear_id = z * (nx * ny) + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
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
    }

    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    // decompose nz
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t nz_local = base + (rank < (int)rem ? 1 : 0);
    size_t z_start = rank * base + (rank < (int)rem ? rank : rem);

    size_t localTotal = nx * ny * (nz_local + 2);
    std::vector<double> cold(localTotal), cnew(localTotal), mu(localTotal);

    initializeConcentration(cold, nx, ny, nz, nz_local, z_start);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, nz_local, rank, size);
        computeChemicalPotential(cold, mu, nx, ny, nz, nz_local, z_start, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        exchangeHalos(mu, nx, ny, nz_local, rank, size);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, nz_local, z_start, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long ms = duration.count();
    long long global_ms;
    MPI_Reduce(&ms, &global_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_ms);
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (global_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // gather results to rank 0
    std::vector<double> globalCold;
    if (rank == 0) globalCold.resize(nx * ny * nz);
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t nl = base + (r < (int)rem ? 1 : 0);
        counts[r] = (int)(nl * nx * ny);
        displs[r] = (int)((r * base + (r < (int)rem ? r : rem)) * nx * ny);
    }
    // extract interior planes
    std::vector<double> localInterior(nz_local * nx * ny);
    for (size_t lz = 1; lz <= nz_local; ++lz)
        for (size_t i = 0; i < nx * ny; ++i)
            localInterior[(lz - 1) * nx * ny + i] = cold[lz * nx * ny + i];

    MPI_Gatherv(localInterior.data(), (int)localInterior.size(), MPI_DOUBLE,
                globalCold.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(globalCold, "Concentration");
    }

    int rc = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(globalCold, nx, ny, nz);
        if (valid) { printf("Validation: PASSED\n"); rc = 0; }
        else { printf("Validation: FAILED\n"); rc = 1; }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}
