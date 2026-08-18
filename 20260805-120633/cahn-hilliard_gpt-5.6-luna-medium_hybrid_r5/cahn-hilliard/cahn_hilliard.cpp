#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error__)); \
    } \
} while (false)

__device__ __forceinline__ size_t cell(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) {
    return z * nx * ny + y * nx + x;
}

__device__ __forceinline__ double laplacian(const double* __restrict__ a, const size_t x,
                                            const size_t y, const size_t z, const size_t nx,
                                            const size_t ny, const size_t nz) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;
    // z=0 and z=nz-1 are halo planes.  Their values are set to the
    // clamped endpoint by exchange_halo(), including at MPI domain edges.
    const size_t zp = z + 1;
    const size_t zn = z - 1;
    const double center = a[cell(x, y, z, nx, ny)];
    return (a[cell(xp, y, z, nx, ny)] + a[cell(xn, y, z, nx, ny)] - 2.0 * center)
               + (a[cell(x, yp, z, nx, ny)] + a[cell(x, yn, z, nx, ny)] - 2.0 * center)
               + (a[cell(x, y, zp, nx, ny)] + a[cell(x, y, zn, nx, ny)] - 2.0 * center);
}

__global__ void chemical_potential(const double* __restrict__ c, double* __restrict__ mu,
                                   const size_t nx, const size_t ny, const size_t local_nz,
                                   const double gamma, const double e_AA, const double e_BB,
                                   const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t i = cell(x, y, z, nx, ny);
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
          + 3.0 * cv + cv * cv * cv - gamma * laplacian(c, x, y, z, nx, ny, local_nz + 2);
}

__global__ void update_concentration(double* __restrict__ cnew, const double* __restrict__ cold,
                                     const double* __restrict__ mu, const size_t nx,
                                     const size_t ny, const size_t local_nz, const double D,
                                     const double dt) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t i = cell(x, y, z, nx, ny);
    cnew[i] = cold[i] + dt * D * laplacian(mu, x, y, z, nx, ny, local_nz + 2);
}

static void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                                    const size_t local_nz, const size_t global_z0,
                                    const size_t global_nz) {
    const size_t plane = nx * ny;
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = (global_z0 + z) * plane + y * nx + x;
                const double pseudo = (((global_id + 1) * 1299709) % (nx * ny * global_nz))
                                    / static_cast<double>(nx * ny * global_nz);
                c[z * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static void exchange_halo(double* device_data, const size_t nx, const size_t ny,
                           const size_t local_nz, const int rank, const int ranks) {
    const size_t plane = nx * ny;
    std::vector<double> lower(plane), upper(plane), recv_lower(plane), recv_upper(plane);
    CUDA_CHECK(cudaMemcpy(lower.data(), device_data + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(upper.data(), device_data + local_nz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost));

    if (rank > 0)
        MPI_Sendrecv(lower.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                     recv_lower.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else recv_lower = lower;
    if (rank + 1 < ranks)
        MPI_Sendrecv(upper.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1,
                     recv_upper.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else recv_upper = upper;

    CUDA_CHECK(cudaMemcpy(device_data, recv_lower.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_data + (local_nz + 1) * plane, recv_upper.data(),
                          plane * sizeof(double), cudaMemcpyHostToDevice));
}

static bool validateResult(const std::vector<double>& c) {
    int invalid = 0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    #pragma omp parallel for reduction(||:invalid) reduction(min:min_val) reduction(max:max_val) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        invalid = invalid || !std::isfinite(c[i]);
        min_val = std::min(min_val, c[i]);
        max_val = std::max(max_val, c[i]);
    }
    if (invalid) { printf("Validation failed: found NaN or Inf value\n"); return false; }
    printf("Concentration range: [%.6f, %.6f]\n", min_val, max_val);
    return max_val <= 10.0 && min_val >= -10.0;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num> Grid size in X (default 64)\n  -y <num> Grid size in Y\n");
    printf("  -z <num> Grid size in Z\n  -i <num> Time steps (default 20)\n");
    printf("  -v Validate\n  -r Print results\n  -h Help\n");
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
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx; if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nz < static_cast<size_t>(ranks)) {
        if (rank == 0) fprintf(stderr, "Invalid grid or too many MPI ranks for Z dimension\n");
        MPI_Finalize(); return 1;
    }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % devices));
    const size_t base = nz / ranks, remainder = nz % ranks;
    const size_t local_nz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t global_z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny, local_size = (local_nz + 2) * plane;
    std::vector<double> host(local_nz * plane), result;
    initializeConcentration(host, nx, ny, local_nz, global_z0, nz);
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, local_size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Hybrid MPI/OpenMP/CUDA: %d MPI ranks, %d OpenMP threads, %d CUDA devices\n", ranks, omp_get_max_threads(), devices);
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    (local_nz + block.z - 1) / block.z);
    for (int t = 0; t < iterations; ++t) {
        exchange_halo(cold, nx, ny, local_nz, rank, ranks);
        chemical_potential<<<grid, block>>>(cold, mu, nx, ny, local_nz, 0.5, -2.0 / 9.0, -2.0 / 9.0, 2.0 / 9.0);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        exchange_halo(mu, nx, ny, local_nz, rank, ranks);
        update_concentration<<<grid, block>>>(cnew, cold, mu, nx, ny, local_nz, 1.0, 0.01);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(cold, cnew);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double local_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsed_ms = 0.0;
    MPI_Reduce(&local_ms, &elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, host.size() * sizeof(double), cudaMemcpyDeviceToHost));

    int count = static_cast<int>(host.size());
    std::vector<int> counts, displacements;
    if (rank == 0) { counts.resize(ranks); displacements.resize(ranks); }
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) { int offset = 0; for (int r = 0; r < ranks; ++r) { displacements[r] = offset; offset += counts[r]; } result.resize(nx * ny * nz); }
    MPI_Gatherv(host.data(), count, MPI_DOUBLE, result.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", elapsed_ms,
               static_cast<double>(nx) * ny * nz * iterations / (elapsed_ms * 1000.0));
        if (printResults) print_results(result, "Concentration");
        if (validate) { printf("Validating result...\n"); const bool valid = validateResult(result); printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu)); MPI_Finalize(); return valid ? 0 : 1; }
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_Finalize(); return 0;
}
