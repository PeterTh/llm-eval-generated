#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

__device__ __forceinline__ size_t cell(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) {
    return (z * ny + y) * nx + x;
}

__device__ __forceinline__ double laplacian(const double* a, const size_t x, const size_t y,
                                            const size_t z, const size_t nx, const size_t ny,
                                            const size_t nz, const double idx2, const double idy2,
                                            const double idz2) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;
    const size_t i = cell(x, y, z, nx, ny);
    return (a[cell(xp, y, z, nx, ny)] + a[cell(xn, y, z, nx, ny)] - 2.0 * a[i]) * idx2
         + (a[cell(x, yp, z, nx, ny)] + a[cell(x, yn, z, nx, ny)] - 2.0 * a[i]) * idy2
         + (a[cell(x, y, z + 1, nx, ny)] + a[cell(x, y, z - 1, nx, ny)] - 2.0 * a[i]) * idz2;
}

__global__ void chemical_kernel(const double* c, double* mu, size_t nx, size_t ny, size_t nz,
                                double gamma, double eAA, double eBB, double eAB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > nz) return;
    const size_t i = cell(x, y, z, nx, ny);
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
          + 3.0 * cv + cv * cv * cv
          - gamma * laplacian(c, x, y, z, nx, ny, nz, 1.0, 1.0, 1.0);
}

__global__ void update_kernel(double* cnew, const double* cold, const double* mu,
                              size_t nx, size_t ny, size_t nz, double dtD) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > nz) return;
    const size_t i = cell(x, y, z, nx, ny);
    cnew[i] = cold[i] + dtD * laplacian(mu, x, y, z, nx, ny, nz, 1.0, 1.0, 1.0);
}

static void cuda_check(cudaError_t error, const char* what, MPI_Comm comm) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(comm, 1);
    }
}

static void launch_check(MPI_Comm comm) {
    cuda_check(cudaGetLastError(), "kernel launch", comm);
    cuda_check(cudaDeviceSynchronize(), "kernel execution", comm);
}

// Exchange one-cell z halos. MPI is deliberately kept on host buffers for portability;
// the expensive stencil work and the two device copies remain fully GPU resident.
static void exchange_halo(double* device, size_t nx, size_t ny, size_t localNz,
                          int rank, int ranks, MPI_Comm comm,
                          std::vector<double>& sendLower, std::vector<double>& sendUpper,
                          std::vector<double>& recvLower, std::vector<double>& recvUpper) {
    const size_t plane = nx * ny;
    cuda_check(cudaMemcpy(sendLower.data(), device + plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "lower halo read", comm);
    cuda_check(cudaMemcpy(sendUpper.data(), device + localNz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "upper halo read", comm);

    if (rank > 0)
        MPI_Sendrecv(sendLower.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 11,
                     recvLower.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 22, comm, MPI_STATUS_IGNORE);
    else
        recvLower = sendLower; // clamped physical boundary
    if (rank + 1 < ranks)
        MPI_Sendrecv(sendUpper.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 22,
                     recvUpper.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 11, comm, MPI_STATUS_IGNORE);
    else
        recvUpper = sendUpper;

    cuda_check(cudaMemcpy(device, recvLower.data(), plane * sizeof(double), cudaMemcpyHostToDevice), "lower halo write", comm);
    cuda_check(cudaMemcpy(device + (localNz + 1) * plane, recvUpper.data(), plane * sizeof(double), cudaMemcpyHostToDevice), "upper halo write", comm);
}

static void initialize_local(std::vector<double>& c, size_t nx, size_t ny, size_t localNz,
                             size_t globalZ, size_t zStart) {
    const size_t plane = nx * ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < localNz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalId = (zStart + z) * plane + y * nx + x;
                const double pseudo = (((globalId + 1) * 1299709) % globalZ) / static_cast<double>(globalZ);
                c[(z + 1) * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validate_result(const std::vector<double>& c) {
    double minVal = c.empty() ? 0.0 : c[0], maxVal = minVal;
    bool finite = true;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:finite)
    for (size_t i = 0; i < c.size(); ++i) {
        finite = finite && std::isfinite(c[i]);
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    if (!finite) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
    std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void print_usage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -x <num> Grid X (default 64)\n  -y <num> Grid Y (default X)\n  -z <num> Grid Z (default X)\n  -i <num> Time steps (default 20)\n  -v Validation\n  -r Print results\n  -h Help\n", name);
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
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) print_usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) print_usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (rank == 0) std::fprintf(stderr, "Invalid grid or MPI decomposition\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int localRank = 0;
    MPI_Comm shared = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    MPI_Comm_rank(shared, &localRank);
    int deviceCount = 0;
    cuda_check(cudaGetDeviceCount(&deviceCount), "device discovery", MPI_COMM_WORLD);
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cuda_check(cudaSetDevice(localRank % deviceCount), "device selection", MPI_COMM_WORLD);
    MPI_Comm_free(&shared);

    const size_t plane = nx * ny, gridSize = nx * ny * nz;
    const size_t base = nz / static_cast<size_t>(ranks), remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t localCells = (localNz + 2) * plane;
    std::vector<double> initial(localCells, 0.0);
    initialize_local(initial, nx, ny, localNz, gridSize, zStart);

    double* dCold = nullptr; double* dCnew = nullptr; double* dMu = nullptr;
    cuda_check(cudaMalloc(&dCold, localCells * sizeof(double)), "cold allocation", MPI_COMM_WORLD);
    cuda_check(cudaMalloc(&dCnew, localCells * sizeof(double)), "new allocation", MPI_COMM_WORLD);
    cuda_check(cudaMalloc(&dMu, localCells * sizeof(double)), "mu allocation", MPI_COMM_WORLD);
    cuda_check(cudaMemcpy(dCold, initial.data(), localCells * sizeof(double), cudaMemcpyHostToDevice), "initial copy", MPI_COMM_WORLD);
    std::vector<double> sendLower(plane), sendUpper(plane), recvLower(plane), recvUpper(plane);
    exchange_halo(dCold, nx, ny, localNz, rank, ranks, MPI_COMM_WORLD, sendLower, sendUpper, recvLower, recvUpper);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: MPI + OpenMP + CUDA (%d ranks)\n", ranks);
        std::printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (localNz + block.z - 1) / block.z);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        chemical_kernel<<<grid, block>>>(dCold, dMu, nx, ny, localNz, 0.5, -2.0 / 9.0, -2.0 / 9.0, 2.0 / 9.0);
        launch_check(MPI_COMM_WORLD);
        exchange_halo(dMu, nx, ny, localNz, rank, ranks, MPI_COMM_WORLD, sendLower, sendUpper, recvLower, recvUpper);
        update_kernel<<<grid, block>>>(dCnew, dCold, dMu, nx, ny, localNz, 0.01);
        launch_check(MPI_COMM_WORLD);
        std::swap(dCold, dCnew);
        exchange_halo(dCold, nx, ny, localNz, rank, ranks, MPI_COMM_WORLD, sendLower, sendUpper, recvLower, recvUpper);
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", gridSize * static_cast<double>(iterations) / elapsed / 1.0e6);
    }

    std::vector<double> localResult(localNz * plane);
    cuda_check(cudaMemcpy(localResult.data(), dCold + plane, localResult.size() * sizeof(double), cudaMemcpyDeviceToHost), "result copy", MPI_COMM_WORLD);
    std::vector<double> result;
    std::vector<int> counts, displacements;
    if (rank == 0) { result.resize(gridSize); counts.resize(ranks); displacements.resize(ranks); }
    int localCount = static_cast<int>(localResult.size());
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            const size_t rz = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(rz * plane);
            displacements[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * plane);
        }
    }
    MPI_Gatherv(localResult.data(), localCount, MPI_DOUBLE, rank == 0 ? result.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bool valid = true;
    if (rank == 0) {
        if (printResults) print_results(result, "Concentration");
        if (validate) { std::printf("Validating result...\n"); valid = validate_result(result); std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
    }
    cudaFree(dCold); cudaFree(dCnew); cudaFree(dMu);
    MPI_Finalize();
    return (rank == 0 && validate && !valid) ? 1 : 0;
}
