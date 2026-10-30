#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation (local arrays include halo in z)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nzLocal,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= nzLocal; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nzLocal,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z <= nzLocal; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
            }
        }
    }
}

void exchangeHalos(std::vector<double>& arr, size_t nx, size_t ny, size_t nzLocal,
                   int rank, int size, MPI_Comm comm) {
    const size_t plane = nx * ny;
    // send last owned (z=nzLocal) to next; recv into halo z=nzLocal+1
    if (rank + 1 < size) {
        MPI_Sendrecv(arr.data() + idx3(0,0,nzLocal,nx,ny), plane, MPI_DOUBLE, rank+1, 0,
                     arr.data() + idx3(0,0,nzLocal+1,nx,ny), plane, MPI_DOUBLE, rank+1, 1, comm, MPI_STATUS_IGNORE);
    } else {
        // clamp: copy last owned to halo
        std::copy(arr.begin() + idx3(0,0,nzLocal,nx,ny),
                  arr.begin() + idx3(0,0,nzLocal,nx,ny) + plane,
                  arr.begin() + idx3(0,0,nzLocal+1,nx,ny));
    }
    if (rank > 0) {
        MPI_Sendrecv(arr.data() + idx3(0,0,1,nx,ny), plane, MPI_DOUBLE, rank-1, 1,
                     arr.data() + idx3(0,0,0,nx,ny), plane, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);
    } else {
        std::copy(arr.begin() + idx3(0,0,1,nx,ny),
                  arr.begin() + idx3(0,0,1,nx,ny) + plane,
                  arr.begin() + idx3(0,0,0,nx,ny));
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             size_t zStart, size_t nzLocal, size_t vol) {
    for (size_t lz = 1; lz <= nzLocal; ++lz) {
        size_t gz = zStart + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, size_t nx, size_t ny, size_t nz) {
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
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0){ printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const double dx=1.0,dy=1.0,dz=1.0,dt=0.01;
    const double e_AA=-(2.0/9.0), e_BB=-(2.0/9.0), e_AB=(2.0/9.0), gamma=0.5, D=1.0;

    // partition z
    size_t base = nz / size, rem = nz % size;
    size_t zStart, nzLocal;
    if ((size_t)rank < rem) { nzLocal = base + 1; zStart = rank*(base+1); }
    else { nzLocal = base; zStart = rem*(base+1) + (rank-rem)*base; }

    const size_t vol = nx*ny*nz;
    const size_t localTotalZ = nzLocal + 2;
    std::vector<double> cold(nx*ny*localTotalZ), cnew(nx*ny*localTotalZ), mu(nx*ny*localTotalZ);

    initializeConcentration(cold, nx, ny, nz, zStart, nzLocal, vol);

    if (rank == 0) printf("Initializing concentration field...\n");
    // exchange initial halos
    exchangeHalos(cold, nx, ny, nzLocal, rank, size, MPI_COMM_WORLD);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, nx, ny, nzLocal, rank, size, MPI_COMM_WORLD);
        computeChemicalPotential(cold, mu, nx, ny, nzLocal, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        exchangeHalos(mu, nx, ny, nzLocal, rank, size, MPI_COMM_WORLD);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nzLocal, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long ms = duration.count();
    long totalMs;
    MPI_Reduce(&ms, &totalMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", totalMs);
        double cellUpdates = (double)(nx*ny*nz) * iterations;
        double mcups = cellUpdates / (totalMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // gather
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t bn = nz/size, rm = nz%size;
        size_t nl = (r < (int)rm) ? bn+1 : bn;
        counts[r] = (int)(nl * nx * ny);
    }
    displs[0]=0;
    for (int r=1;r<size;++r) displs[r]=displs[r-1]+counts[r-1];

    std::vector<double> global;
    if (rank == 0) global.resize(nx*ny*nz);
    // extract owned planes (z=1..nzLocal) contiguous? Need gather into global with correct offsets.
    std::vector<double> sendbuf(nzLocal*nx*ny);
    for (size_t lz=1; lz<=nzLocal; ++lz)
        std::copy(cold.begin()+idx3(0,0,lz,nx,ny), cold.begin()+idx3(0,0,lz,nx,ny)+nx*ny, sendbuf.begin()+(lz-1)*nx*ny);

    MPI_Gatherv(sendbuf.data(), (int)sendbuf.size(), MPI_DOUBLE,
                rank==0?global.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) print_results(global, "Concentration");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global, nx, ny, nz);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
