#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

inline __host__ __device__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__global__ void initKernel(double* c, size_t nx, size_t ny, size_t nz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    if (i >= total) return;
    size_t z = i / (nx * ny);
    size_t rem = i % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;
    size_t linear_id = z * (nx * ny) + y * nx + x;
    double pseudo = ((((linear_id + 1) * 1299709ULL) % total) / (double)total);
    c[i] = -1.0 + 2.0 * pseudo;
}

__device__ double laplacian(const double* c, size_t nx, size_t ny, size_t nz, double dx, double dy, double dz, size_t x, size_t y, size_t z) {
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : 0;
    double cxx = (c[idx3(xp,y,z,nx,ny)] + c[idx3(xn,y,z,nx,ny)] - 2.0*c[idx3(x,y,z,nx,ny)])/(dx*dx);
    double cyy = (c[idx3(x,yp,z,nx,ny)] + c[idx3(x,yn,z,nx,ny)] - 2.0*c[idx3(x,y,z,nx,ny)])/(dy*dy);
    double czz = (c[idx3(x,y,zp,nx,ny)] + c[idx3(x,y,zn,nx,ny)] - 2.0*c[idx3(x,y,z,nx,ny)])/(dz*dz);
    return cxx + cyy + czz;
}

__global__ void chemPotKernel(const double* c, double* mu, size_t nx, size_t ny, size_t nz, double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    if (i >= total) return;
    size_t z = i / (nx * ny);
    size_t rem = i % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;
    double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * laplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

__global__ void updateKernel(double* cnew, const double* cold, const double* mu, size_t nx, size_t ny, size_t nz, double D, double dt, double dx, double dy, double dz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    if (i >= total) return;
    size_t z = i / (nx * ny);
    size_t rem = i % (nx * ny);
    size_t y = rem / nx;
    size_t x = rem % nx;
    cnew[i] = cold[i] + dt * D * laplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
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
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0){printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]);} MPI_Finalize(); return 1; }
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
    const double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = (2.0/9.0);
    const double gamma = 0.5, D = 1.0;

    size_t gridSize = nx * ny * nz;

    std::vector<double> cold(gridSize), cnew(gridSize), mu(gridSize);

    double *dcold, *dcnew, *dmu;
    cudaMalloc(&dcold, gridSize*sizeof(double));
    cudaMalloc(&dcnew, gridSize*sizeof(double));
    cudaMalloc(&dmu, gridSize*sizeof(double));

    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    int bs = 256;
    size_t blocks = (gridSize + bs - 1)/bs;
    initKernel<<<blocks, bs>>>(dcold, nx, ny, nz);
    cudaDeviceSynchronize();

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        chemPotKernel<<<blocks, bs>>>(dcold, dmu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        cudaDeviceSynchronize();
        updateKernel<<<blocks, bs>>>(dcnew, dcold, dmu, nx, ny, nz, D, dt, dx, dy, dz);
        cudaDeviceSynchronize();
        std::swap(dcold, dcnew);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cudaMemcpy(cold.data(), dcold, gridSize*sizeof(double), cudaMemcpyDeviceToHost);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if (printResults) print_results(cold, "Concentration");
        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            #pragma omp parallel for reduction(&&:valid)
            for (size_t i = 0; i < gridSize; ++i) {
                if (std::isnan(cold[i]) || std::isinf(cold[i])) valid = false;
            }
            double minVal = cold[0], maxVal = cold[0];
            #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
            for (size_t i = 0; i < gridSize; ++i) {
                minVal = std::min(minVal, cold[i]);
                maxVal = std::max(maxVal, cold[i]);
            }
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0) valid = false;
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    cudaFree(dcold); cudaFree(dcnew); cudaFree(dmu);
    MPI_Finalize();
    return 0;
}
