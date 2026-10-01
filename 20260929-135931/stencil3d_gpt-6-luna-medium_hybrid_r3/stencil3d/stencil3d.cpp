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

using Real = double;

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void stencilKernel(const Real* in, Real* out, size_t nx, size_t ny,
                              size_t localNz, size_t globalStart, size_t globalNz) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = nx * ny;
    const size_t total = localNz * plane;
    if (p >= total) return;
    const size_t z = p / plane + 1;
    const size_t rem = p % plane;
    const size_t y = rem / nx;
    const size_t x = rem % nx;
    const size_t i = z * plane + rem;
    const size_t globalZ = globalStart + z - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        globalZ == 0 || globalZ + 1 == globalNz) {
        out[i] = in[i];
    } else {
        out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] +
                  in[i-plane] + in[i+plane]) / 7.0;
    }
}

static void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

bool validateResult(const std::vector<Real>& grid) {
    Real minVal = grid[0], maxVal = grid[0];
    bool valid = true;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:valid)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real v = grid[static_cast<size_t>(i)];
        if (!std::isfinite(v)) valid = false;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) valid = false;
    if (!valid) printf("Validation failed: invalid or out-of-range value\n");
    return valid;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -x <num> Grid size in X (default: 128)\n  -y <num> Grid size in Y (default: X)\n");
    printf("  -z <num> Grid size in Z (default: X)\n  -i <num> Iterations (default: 10)\n");
    printf("  -v Validate\n  -r Print results\n  -h Show this help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") || !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i+1 < argc) {
                const char opt = argv[i++][1];
                const long long n = atoll(argv[i]);
                if (n <= 0) bad = true;
                else if (opt == 'x') nx = static_cast<size_t>(n);
                else if (opt == 'y') ny = static_cast<size_t>(n);
                else if (opt == 'z') nz = static_cast<size_t>(n);
                else if (n > 2147483647LL) bad = true;
                else iterations = static_cast<int>(n);
            } else if (!strcmp(argv[i], "-v")) validate = true;
            else if (!strcmp(argv[i], "-r")) printResults = true;
            else if (!strcmp(argv[i], "-h")) help = true;
            else bad = true;
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
        if (nx < 3 || ny < 3 || nz < 3 || nranks > static_cast<int>(nz)) bad = true;
        if (help || bad) printUsage(argv[0]);
    }
    MPI_Bcast(&bad, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&help, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (bad || help) { MPI_Finalize(); return bad ? 1 : 0; }
    unsigned long long dims[3] = {nx, ny, nz};
    int flags[2] = {validate, printResults};
    MPI_Bcast(dims, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    nx = dims[0]; ny = dims[1]; nz = dims[2]; validate = flags[0]; printResults = flags[1];

    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d has no usable CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    const size_t base = nz / static_cast<size_t>(nranks), extra = nz % static_cast<size_t>(nranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t startZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny, localCount = (localNz + 2) * plane;
    std::vector<Real> hostA(localCount), hostB(localCount);
    #pragma omp parallel for
    for (long long q = 0; q < static_cast<long long>(localNz * plane); ++q) {
        const size_t z = static_cast<size_t>(q) / plane;
        const size_t rem = static_cast<size_t>(q) % plane;
        const size_t globalIndex = (startZ + z) * plane + rem;
        hostA[(z+1)*plane + rem] = static_cast<Real>(globalIndex % 19);
    }

    Real *dA = nullptr, *dB = nullptr;
    cudaCheck(cudaMalloc(&dA, localCount * sizeof(Real)), "cudaMalloc A");
    cudaCheck(cudaMalloc(&dB, localCount * sizeof(Real)), "cudaMalloc B");
    cudaCheck(cudaMemcpy(dA, hostA.data(), localCount*sizeof(Real), cudaMemcpyHostToDevice), "copy input");
    std::vector<Real> sendLo(plane), sendHi(plane), recvLo(plane), recvHi(plane);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* src = (iter % 2 == 0) ? dA : dB;
        Real* dst = (iter % 2 == 0) ? dB : dA;
        cudaCheck(cudaMemcpy(sendLo.data(), src + plane, plane*sizeof(Real), cudaMemcpyDeviceToHost), "copy low halo");
        cudaCheck(cudaMemcpy(sendHi.data(), src + localNz*plane, plane*sizeof(Real), cudaMemcpyDeviceToHost), "copy high halo");
        MPI_Sendrecv(sendLo.data(), static_cast<int>(plane), MPI_DOUBLE, rank > 0 ? rank-1 : MPI_PROC_NULL, 0,
                     recvHi.data(), static_cast<int>(plane), MPI_DOUBLE, rank+1 < nranks ? rank+1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sendHi.data(), static_cast<int>(plane), MPI_DOUBLE, rank+1 < nranks ? rank+1 : MPI_PROC_NULL, 1,
                     recvLo.data(), static_cast<int>(plane), MPI_DOUBLE, rank > 0 ? rank-1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > 0) cudaCheck(cudaMemcpy(src, recvLo.data(), plane*sizeof(Real), cudaMemcpyHostToDevice), "update low halo");
        if (rank+1 < nranks) cudaCheck(cudaMemcpy(src + (localNz+1)*plane, recvHi.data(), plane*sizeof(Real), cudaMemcpyHostToDevice), "update high halo");
        const size_t work = localNz * plane;
        stencilKernel<<<static_cast<unsigned>((work + 255) / 256), 256>>>(src, dst, nx, ny, localNz, startZ, nz);
        cudaCheck(cudaGetLastError(), "stencil launch");
    }
    cudaCheck(cudaDeviceSynchronize(), "stencil completion");
    const auto end = std::chrono::steady_clock::now();
    const double localElapsed = std::chrono::duration<double>(end-begin).count();
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const Real* finalDevice = (iterations % 2 == 0) ? dA : dB;
    std::vector<Real> localFinal(localNz*plane);
    cudaCheck(cudaMemcpy(localFinal.data(), finalDevice + plane, localFinal.size()*sizeof(Real), cudaMemcpyDeviceToHost), "copy result");
    const size_t globalCount = nx*ny*nz;
    std::vector<Real> result(rank == 0 ? globalCount : 0);
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t rz = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        const size_t rzStart = static_cast<size_t>(r)*base + std::min(static_cast<size_t>(r), extra);
        counts[r] = static_cast<int>(rz*plane); displs[r] = static_cast<int>(rzStart*plane);
    }
    MPI_Gatherv(localFinal.data(), static_cast<int>(localFinal.size()), MPI_DOUBLE,
                result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double mcups = static_cast<double>((nx-2)*(ny-2)*(nz-2)) * iterations / elapsed / 1e6;
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", elapsed*1000.0, mcups);
        if (printResults) print_results(result, "Grid");
        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(result);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            cudaFree(dA); cudaFree(dB); MPI_Finalize(); return ok ? 0 : 1;
        }
    }
    cudaFree(dA); cudaFree(dB);
    MPI_Finalize();
    return 0;
}
