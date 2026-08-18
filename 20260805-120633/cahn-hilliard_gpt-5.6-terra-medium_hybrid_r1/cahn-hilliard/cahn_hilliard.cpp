#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// The decomposition is along z.  Each rank owns local_nz planes and keeps one
// ghost plane at either end.  Device data stay resident for stencil execution;
// pinned staging planes make the MPI exchange work with every MPI implementation.
inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static void cudaCheck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA failure at %s: %s\n", where, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void initializeKernel(double* c, size_t nx, size_t ny, size_t local_nz,
                                 size_t global_z0, size_t global_nz) {
    const size_t n = nx * ny * local_nz;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        const size_t z = i / (nx * ny);
        const size_t rem = i - z * nx * ny;
        const size_t linear = (global_z0 + z) * nx * ny + rem;
        const size_t volume = nx * ny * global_nz;
        c[i + nx * ny] = -1.0 + 2.0 * ((((linear + 1) * (size_t)1299709) % volume) /
                                        (double)volume);
    }
}

__global__ void copyBoundaryKernel(double* a, size_t plane, size_t local_nz, bool lower) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < plane;
         i += (size_t)blockDim.x * gridDim.x)
        a[lower ? i : (local_nz + 1) * plane + i] = a[lower ? plane + i : local_nz * plane + i];
}

__device__ inline double laplacian(const double* __restrict__ a, size_t x, size_t y, size_t z,
                                   size_t nx, size_t ny, size_t plane) {
    const size_t p = z * plane + y * nx + x;
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    return a[z * plane + y * nx + xp] + a[z * plane + y * nx + xn] +
           a[z * plane + yp * nx + x] + a[z * plane + yn * nx + x] +
           a[p + plane] + a[p - plane] - 6.0 * a[p];
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t local_nz) {
    const size_t plane = nx * ny, n = plane * local_nz;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        const size_t z = i / plane + 1, rem = i % plane, y = rem / nx, x = rem % nx;
        const size_t p = z * plane + rem;
        const double v = c[p];
        // Original constants folded exactly into their arithmetic form.
        mu[p] = 4.5 * ((v + 1.0) * (-2.0 / 9.0) + (v - 1.0) * (-2.0 / 9.0) -
                       2.0 * v * (2.0 / 9.0)) + 3.0 * v + v * v * v -
                0.5 * laplacian(c, x, y, z, nx, ny, plane);
    }
}

__global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                             double* __restrict__ cnew, size_t nx, size_t ny, size_t local_nz) {
    const size_t plane = nx * ny, n = plane * local_nz;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        const size_t z = i / plane + 1, rem = i % plane, y = rem / nx, x = rem % nx;
        const size_t p = z * plane + rem;
        cnew[p] = cold[p] + 0.01 * laplacian(mu, x, y, z, nx, ny, plane);
    }
}

static void exchangeHalos(double* field, size_t plane, size_t local_nz, int rank, int ranks,
                          double* send_lo, double* send_hi, double* recv_lo, double* recv_hi) {
    const int lo = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int hi = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    // Synchronous copies also establish the CUDA-to-MPI ordering.  Pinned buffers
    // retain high PCIe/NVLink bandwidth on MPI stacks without CUDA-aware support.
    cudaCheck(cudaMemcpy(send_lo, field + plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "copy lower halo");
    cudaCheck(cudaMemcpy(send_hi, field + local_nz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "copy upper halo");
    MPI_Sendrecv(send_lo, (int)plane, MPI_DOUBLE, lo, 11,
                 recv_hi, (int)plane, MPI_DOUBLE, hi, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_hi, (int)plane, MPI_DOUBLE, hi, 12,
                 recv_lo, (int)plane, MPI_DOUBLE, lo, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (lo != MPI_PROC_NULL) cudaCheck(cudaMemcpy(field, recv_lo, plane * sizeof(double), cudaMemcpyHostToDevice), "restore lower halo");
    if (hi != MPI_PROC_NULL) cudaCheck(cudaMemcpy(field + (local_nz + 1) * plane, recv_hi, plane * sizeof(double), cudaMemcpyHostToDevice), "restore upper halo");
    const int blocks = (int)std::min<size_t>((plane + 255) / 256, 65535);
    if (lo == MPI_PROC_NULL) copyBoundaryKernel<<<blocks, 256>>>(field, plane, local_nz, true);
    if (hi == MPI_PROC_NULL) copyBoundaryKernel<<<blocks, 256>>>(field, plane, local_nz, false);
    cudaCheck(cudaGetLastError(), "halo boundary kernel");
}

static bool validateResult(const std::vector<double>& c) {
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    int bad = 0;
#pragma omp parallel for reduction(min:minv) reduction(max:maxv) reduction(|:bad) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        const double v = c[i];
        if (!std::isfinite(v)) bad = 1;
        minv = std::min(minv, v); maxv = std::max(maxv, v);
    }
    if (bad) { printf("Validation failed: found NaN or Inf value\n"); return false; }
    printf("Concentration range: [%.6f, %.6f]\n", minv, maxv);
    if (maxv > 10.0 || minv < -10.0) { printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

static void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) { if (!rank) fprintf(stderr, "No CUDA device is available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
    MPI_Comm_free(&local_comm);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (nz < (size_t)ranks || nx * ny > (size_t)INT_MAX) { if (!rank) fprintf(stderr, "Grid is too small for MPI ranks or a plane exceeds MPI count limits\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    const size_t base = nz / ranks, rem = nz % ranks;
    const size_t local_nz = base + ((size_t)rank < rem);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, rem);
    const size_t plane = nx * ny, local_count = local_nz * plane, alloc_count = (local_nz + 2) * plane;
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); }
    double *cold, *cnew, *mu, *send_lo, *send_hi, *recv_lo, *recv_hi;
    cudaCheck(cudaMalloc(&cold, alloc_count * sizeof(double)), "allocate cold");
    cudaCheck(cudaMalloc(&cnew, alloc_count * sizeof(double)), "allocate cnew");
    cudaCheck(cudaMalloc(&mu, alloc_count * sizeof(double)), "allocate mu");
    cudaCheck(cudaMallocHost(&send_lo, plane * sizeof(double)), "allocate lower send staging");
    cudaCheck(cudaMallocHost(&send_hi, plane * sizeof(double)), "allocate upper send staging");
    cudaCheck(cudaMallocHost(&recv_lo, plane * sizeof(double)), "allocate lower receive staging");
    cudaCheck(cudaMallocHost(&recv_hi, plane * sizeof(double)), "allocate upper receive staging");
    const int blocks = (int)std::min<size_t>((local_count + 255) / 256, 65535);
    initializeKernel<<<blocks, 256>>>(cold, nx, ny, local_nz, z0, nz); cudaCheck(cudaGetLastError(), "initialization");
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    cudaCheck(cudaDeviceSynchronize(), "initialization sync"); MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, local_nz, rank, ranks, send_lo, send_hi, recv_lo, recv_hi);
        chemicalPotentialKernel<<<blocks, 256>>>(cold, mu, nx, ny, local_nz); cudaCheck(cudaGetLastError(), "chemical potential");
        exchangeHalos(mu, plane, local_nz, rank, ranks, send_lo, send_hi, recv_lo, recv_hi);
        updateKernel<<<blocks, 256>>>(cold, mu, cnew, nx, ny, local_nz); cudaCheck(cudaGetLastError(), "update");
        std::swap(cold, cnew);
    }
    cudaCheck(cudaDeviceSynchronize(), "simulation sync"); MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (!rank) { const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms, (double)(nx*ny*nz)*iterations/(ms/1000.0)/1.e6); }
    std::vector<double> local(local_count), result;
    cudaCheck(cudaMemcpy(local.data(), cold + plane, local_count * sizeof(double), cudaMemcpyDeviceToHost), "copy result");
    std::vector<int> counts, displs;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); for (int r=0;r<ranks;++r) { size_t zc=base+((size_t)r<rem); counts[r]=(int)(zc*plane); displs[r]=(int)(((size_t)r*base+std::min((size_t)r,rem))*plane); } result.resize(nx*ny*nz); }
    MPI_Gatherv(local.data(), (int)local_count, MPI_DOUBLE, rank ? nullptr : result.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int ok = 1;
    if (!rank) { if (printResults) print_results(result, "Concentration"); if (validate) ok = validateResult(result); if (validate) printf("Validation: %s\n", ok ? "PASSED" : "FAILED"); }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFreeHost(send_lo); cudaFreeHost(send_hi); cudaFreeHost(recv_lo); cudaFreeHost(recv_hi);
    cudaFree(cold); cudaFree(cnew); cudaFree(mu); MPI_Finalize(); return ok ? 0 : 1;
}
