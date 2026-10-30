#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ inline double laplacian(const double* c, size_t nx, size_t ny, size_t nz,
                                   double dx, double dy, double dz, size_t x, size_t y, size_t z) {
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : 0;
    double cxx = (c[xp + y*nx + z*nx*ny] + c[xn + y*nx + z*nx*ny] - 2.0*c[x + y*nx + z*nx*ny])/(dx*dx);
    double cyy = (c[x + yp*nx + z*nx*ny] + c[x + yn*nx + z*nx*ny] - 2.0*c[x + y*nx + z*nx*ny])/(dy*dy);
    double czz = (c[x + y*nx + zp*nx*ny] + c[x + y*nx + zn*nx*ny] - 2.0*c[x + y*nx + z*nx*ny])/(dz*dz);
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* c, double* mu, size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx*ny*nz;
    if (idx >= total) return;
    size_t x = idx % nx;
    size_t y = (idx / nx) % ny;
    size_t z = idx / (nx*ny);
    double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * laplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu, size_t nx, size_t ny, size_t nz,
    double D, double dt, double dx, double dy, double dz) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx*ny*nz;
    if (idx >= total) return;
    size_t x = idx % nx;
    size_t y = (idx / nx) % ny;
    size_t z = idx / (nx*ny);
    cnew[idx] = cold[idx] + dt * D * laplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    bool ok = true;
    #pragma omp parallel for reduction(min:ok) // placeholder not valid for bool min? use manual
    // fallback serial
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) { ok = false; }
    }
    if (!ok) { printf("Validation failed: found NaN or Inf value\n"); return false; }
    double minVal = c[0], maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) { printf("Validation failed: values out of expected range\n"); return false; }
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
        else { if (rank==0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
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
    const double e_AA=-(2.0/9.0), e_BB=-(2.0/9.0), e_AB=(2.0/9.0);
    const double gamma=0.5, D=1.0;
    size_t gridSize = nx*ny*nz;

    std::vector<double> cold(gridSize), cnew(gridSize), mu(gridSize);
    if (rank == 0) {
        printf("Initializing concentration field...\n");
        initializeConcentration(cold, nx, ny, nz);
    }
    MPI_Bcast(cold.data(), gridSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    double *d_cold,*d_cnew,*d_mu;
    cudaMalloc(&d_cold, gridSize*sizeof(double));
    cudaMalloc(&d_cnew, gridSize*sizeof(double));
    cudaMalloc(&d_mu, gridSize*sizeof(double));

    for (int t = 0; t < iterations; ++t) {
        cudaMemcpy(d_cold, cold.data(), gridSize*sizeof(double), cudaMemcpyHostToDevice);
        int blocks = (gridSize + 255)/256;
        computeChemicalPotentialKernel<<<blocks,256>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        cudaMemcpy(mu.data(), d_mu, gridSize*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(d_cold, cold.data(), gridSize*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mu, mu.data(), gridSize*sizeof(double), cudaMemcpyHostToDevice);
        cahnHilliardUpdateKernel<<<blocks,256>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        cudaMemcpy(cnew.data(), d_cnew, gridSize*sizeof(double), cudaMemcpyDeviceToHost);
        std::swap(cold, cnew);
    }
    cudaDeviceSynchronize();

    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if (printResults) print_results(cold, "Concentration");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }
    cudaFree(d_cold); cudaFree(d_cnew); cudaFree(d_mu);
    MPI_Finalize();
    return 0;
}
