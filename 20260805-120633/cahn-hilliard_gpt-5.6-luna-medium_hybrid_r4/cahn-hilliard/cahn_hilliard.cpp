#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline constexpr size_t index3(const size_t x, const size_t y, const size_t z,
                               const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

__device__ inline double laplacian(const double* a, size_t x, size_t y, size_t z,
                                   size_t nx, size_t ny, size_t nz, double invx2,
                                   double invy2, double invz2) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t zp = (z + 1 < nz) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const double center = a[index3(x, y, z, nx, ny)];
    return (a[index3(xp, y, z, nx, ny)] + a[index3(xn, y, z, nx, ny)] - 2.0 * center) * invx2
         + (a[index3(x, yp, z, nx, ny)] + a[index3(x, yn, z, nx, ny)] - 2.0 * center) * invy2
         + (a[index3(x, y, zp, nx, ny)] + a[index3(x, y, zn, nx, ny)] - 2.0 * center) * invz2;
}

__global__ void chemicalPotential(const double* c, double* mu, size_t nx, size_t ny,
                                  size_t nz, double gamma, double eAA, double eBB,
                                  double eAB, double invx2, double invy2, double invz2) {
    const size_t n = nx * ny * nz;
    for (size_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < n;
         linear += blockDim.x * gridDim.x) {
        const size_t z = linear / (nx * ny);
        const size_t rem = linear - z * nx * ny;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const double cv = c[index3(x, y, z, nx, ny)];
        mu[index3(x, y, z, nx, ny)] =
            4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
            + 3.0 * cv + cv * cv * cv
            - gamma * laplacian(c, x, y, z, nx, ny, nz, invx2, invy2, invz2);
    }
}

__global__ void updateConcentration(double* cnew, const double* cold, const double* mu,
                                    size_t nx, size_t ny, size_t nz, double D, double dt,
                                    double invx2, double invy2, double invz2) {
    const size_t n = nx * ny * nz;
    for (size_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < n;
         linear += blockDim.x * gridDim.x) {
        const size_t z = linear / (nx * ny);
        const size_t rem = linear - z * nx * ny;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const size_t i = index3(x, y, z, nx, ny);
        cnew[i] = cold[i] + dt * D * laplacian(mu, x, y, z, nx, ny, nz,
                                                invx2, invy2, invz2);
    }
}

static void checkCuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void exchangeHalos(double* device, std::vector<double>& sendLower,
                          std::vector<double>& sendUpper, std::vector<double>& recvLower,
                          std::vector<double>& recvUpper, size_t plane, size_t localNz,
                          size_t rank, size_t ranks) {
    checkCuda(cudaMemcpy(sendLower.data(), device + plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "D2H lower halo");
    checkCuda(cudaMemcpy(sendUpper.data(), device + localNz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "D2H upper halo");
    if (ranks > 1) {
        MPI_Sendrecv(sendLower.data(), static_cast<int>(plane), MPI_DOUBLE,
                     rank > 0 ? static_cast<int>(rank - 1) : MPI_PROC_NULL, 40,
                     recvLower.data(), static_cast<int>(plane), MPI_DOUBLE,
                     rank > 0 ? static_cast<int>(rank - 1) : MPI_PROC_NULL, 41,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sendUpper.data(), static_cast<int>(plane), MPI_DOUBLE,
                     rank + 1 < ranks ? static_cast<int>(rank + 1) : MPI_PROC_NULL, 41,
                     recvUpper.data(), static_cast<int>(plane), MPI_DOUBLE,
                     rank + 1 < ranks ? static_cast<int>(rank + 1) : MPI_PROC_NULL, 40,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > 0) checkCuda(cudaMemcpy(device, recvLower.data(), plane * sizeof(double), cudaMemcpyHostToDevice), "H2D lower halo");
        if (rank + 1 < ranks) checkCuda(cudaMemcpy(device + (localNz + 1) * plane, recvUpper.data(), plane * sizeof(double), cudaMemcpyHostToDevice), "H2D upper halo");
    }
    // A physical clamped boundary is represented by repeating its edge plane.
    // This also covers the one-rank case, where there is no MPI message.
    if (rank == 0) {
        checkCuda(cudaMemcpy(device, device + plane, plane * sizeof(double), cudaMemcpyDeviceToDevice), "clamp lower halo");
    }
    if (rank + 1 == ranks) {
        checkCuda(cudaMemcpy(device + (localNz + 1) * plane, device + localNz * plane,
                             plane * sizeof(double), cudaMemcpyDeviceToDevice), "clamp upper halo");
    }
    checkCuda(cudaDeviceSynchronize(), "halo exchange");
}

static void initialize(std::vector<double>& c, size_t nx, size_t ny, size_t localNz,
                       size_t globalZ, size_t globalNz) {
    const size_t plane = nx * ny;
    const size_t volume = plane * globalNz;
    #pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(localNz * plane); ++local) {
        const size_t z = static_cast<size_t>(local) / plane;
        const size_t id = (globalZ + z) * plane + static_cast<size_t>(local) % plane;
        const double pseudo = (((id + 1) * 1299709) % volume) / static_cast<double>(volume);
        c[(z + 1) * plane + static_cast<size_t>(local) % plane] = -1.0 + 2.0 * pseudo;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printf("Usage: %s [-x n] [-y n] [-z n] [-i steps] [-v] [-r]\n", argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) fprintf(stderr, "Unknown option: %s\n", argv[i]); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (ny == 0) ny = nx; if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || static_cast<size_t>(ranks) > nz) MPI_Abort(MPI_COMM_WORLD, 1);
    const size_t plane = nx * ny;
    const size_t base = nz / static_cast<size_t>(ranks), extra = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t globalZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t localCells = localNz * plane;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    checkCuda(cudaSetDevice(rank % std::max(1, [] { int n = 0; cudaGetDeviceCount(&n); return n; }())), "select GPU");
    std::vector<double> cold((localNz + 2) * plane), cnew((localNz + 2) * plane), mu((localNz + 2) * plane);
    std::vector<double> sendLower(plane), sendUpper(plane), recvLower(plane), recvUpper(plane);
    initialize(cold, nx, ny, localNz, globalZ, nz);
    initialize(cnew, nx, ny, localNz, globalZ, nz);
    double* dcold = nullptr; double* dcnew = nullptr; double* dmu = nullptr;
    checkCuda(cudaMalloc(&dcold, cold.size() * sizeof(double)), "allocate cold");
    checkCuda(cudaMalloc(&dcnew, cnew.size() * sizeof(double)), "allocate cnew");
    checkCuda(cudaMalloc(&dmu, mu.size() * sizeof(double)), "allocate mu");
    checkCuda(cudaMemcpy(dcold, cold.data(), cold.size() * sizeof(double), cudaMemcpyHostToDevice), "upload concentration");
    const double invx2 = 1.0, invy2 = 1.0, invz2 = 1.0;
    const int blocks = std::min(4096, static_cast<int>((localCells + 255) / 256));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(dcold, sendLower, sendUpper, recvLower, recvUpper, plane, localNz, rank, ranks);
        chemicalPotential<<<blocks, 256>>>(dcold + plane, dmu + plane, nx, ny, localNz, 0.5, -2.0 / 9.0, -2.0 / 9.0, 2.0 / 9.0, invx2, invy2, invz2);
        checkCuda(cudaGetLastError(), "chemical potential kernel");
        exchangeHalos(dmu, sendLower, sendUpper, recvLower, recvUpper, plane, localNz, rank, ranks);
        updateConcentration<<<blocks, 256>>>(dcnew + plane, dcold + plane, dmu + plane, nx, ny, localNz, 1.0, 0.01, invx2, invy2, invz2);
        checkCuda(cudaGetLastError(), "concentration kernel");
        std::swap(dcold, dcnew);
    }
    checkCuda(cudaDeviceSynchronize(), "finish simulation");
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(cold.data(), dcold, cold.size() * sizeof(double), cudaMemcpyDeviceToHost), "download result");
    std::vector<double> result;
    if (rank == 0) result.resize(nx * ny * nz);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { size_t rz = base + (static_cast<size_t>(r) < extra ? 1 : 0); counts[r] = static_cast<int>(rz * plane); displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * plane); }
    MPI_Gatherv(cold.data() + plane, static_cast<int>(localCells), MPI_DOUBLE, result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", static_cast<long>(maxElapsed * 1000.0), (nx * ny * nz * iterations) / maxElapsed / 1e6);
        if (printResults) print_results(result, "Concentration");
        if (validate) {
            int valid = 1; double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:valid)
            for (long long i = 0; i < static_cast<long long>(result.size()); ++i) { lo = std::min(lo, result[i]); hi = std::max(hi, result[i]); valid &= std::isfinite(result[i]) ? 1 : 0; }
            printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n", lo, hi, valid && hi <= 10.0 && lo >= -10.0 ? "PASSED" : "FAILED");
            if (!valid || hi > 10.0 || lo < -10.0) { cudaFree(dcold); cudaFree(dcnew); cudaFree(dmu); MPI_Finalize(); return 1; }
        }
    }
    cudaFree(dcold); cudaFree(dcnew); cudaFree(dmu);
    MPI_Finalize();
    return 0;
}
