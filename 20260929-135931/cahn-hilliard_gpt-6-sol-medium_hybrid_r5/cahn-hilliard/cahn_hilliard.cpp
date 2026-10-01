#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int error = (call); if (error != MPI_SUCCESS) { \
    char message[MPI_MAX_ERROR_STRING]; int length = 0; MPI_Error_string(error, message, &length); \
    fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length, message); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            size_t p, size_t x, size_t y, size_t z,
                                            size_t nx, size_t ny, size_t localNz,
                                            bool firstRank, bool lastRank) {
    size_t plane = nx * ny;
    double center = field[p];
    double cxx = (field[x + 1 < nx ? p + 1 : p] + field[x ? p - 1 : p] - 2.0 * center);
    double cyy = (field[y + 1 < ny ? p + nx : p] + field[y ? p - nx : p] - 2.0 * center);
    double czz = (field[z == localNz && lastRank ? p : p + plane] +
                  field[z == 1 && firstRank ? p : p - plane] - 2.0 * center);
    return cxx + cyy + czz;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t localNz, size_t zBegin,
                                  bool firstRank, bool lastRank) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) + zBegin;
    if (x >= nx || y >= ny || z > localNz) return;
    size_t p = z * nx * ny + y * nx + x;
    double cv = c[p];
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    mu[p] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
          + 3.0 * cv + cv * cv * cv
          - 0.5 * laplacian(c, p, x, y, z, nx, ny, localNz, firstRank, lastRank);
}

__global__ void update(const double* __restrict__ oldC, const double* __restrict__ mu,
                       double* __restrict__ newC, size_t nx, size_t ny, size_t localNz,
                       size_t zBegin, bool firstRank, bool lastRank) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) + zBegin;
    if (x >= nx || y >= ny || z > localNz) return;
    size_t p = z * nx * ny + y * nx + x;
    newC[p] = oldC[p] + 0.01 * laplacian(mu, p, x, y, z, nx, ny, localNz, firstRank, lastRank);
}

static void launchChemical(const double* c, double* mu, size_t nx, size_t ny, size_t localNz,
                           size_t zBegin, size_t zEnd, int rank, int ranks, cudaStream_t stream) {
    if (zBegin > zEnd) return;
    dim3 block(32, 4), grid((nx + 31) / 32, (ny + 3) / 4, zEnd - zBegin + 1);
    chemicalPotential<<<grid, block, 0, stream>>>(c, mu, nx, ny, localNz, zBegin, rank == 0, rank == ranks - 1);
    CUDA_CHECK(cudaGetLastError());
}

static void launchUpdate(const double* c, const double* mu, double* next,
                         size_t nx, size_t ny, size_t localNz, size_t zBegin, size_t zEnd,
                         int rank, int ranks, cudaStream_t stream) {
    if (zBegin > zEnd) return;
    dim3 block(32, 4), grid((nx + 31) / 32, (ny + 3) / 4, zEnd - zBegin + 1);
    update<<<grid, block, 0, stream>>>(c, mu, next, nx, ny, localNz, zBegin, rank == 0, rank == ranks - 1);
    CUDA_CHECK(cudaGetLastError());
}

// Exchange the two contiguous Z faces through pinned host memory. The compute stream
// can work on interior planes while the transfer stream and MPI move boundary data.
static void exchangeHalos(double* field, size_t plane, size_t localNz, int rank, int ranks,
                          double* buffers, cudaStream_t transfer, MPI_Comm comm) {
    if (ranks == 1) return;
    double *sendFirst = buffers, *sendLast = buffers + plane;
    double *recvFirst = buffers + 2 * plane, *recvLast = buffers + 3 * plane;
    size_t bytes = plane * sizeof(double);
    if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(sendFirst, field + plane, bytes, cudaMemcpyDeviceToHost, transfer));
    if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(sendLast, field + localNz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
    CUDA_CHECK(cudaStreamSynchronize(transfer));
    int previous = rank ? rank - 1 : MPI_PROC_NULL;
    int following = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Request requests[4];
    MPI_CHECK(MPI_Irecv(recvFirst, static_cast<int>(plane), MPI_DOUBLE, previous, 0, comm, &requests[0]));
    MPI_CHECK(MPI_Irecv(recvLast, static_cast<int>(plane), MPI_DOUBLE, following, 1, comm, &requests[1]));
    MPI_CHECK(MPI_Isend(sendFirst, static_cast<int>(plane), MPI_DOUBLE, previous, 1, comm, &requests[2]));
    MPI_CHECK(MPI_Isend(sendLast, static_cast<int>(plane), MPI_DOUBLE, following, 0, comm, &requests[3]));
    MPI_CHECK(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE));
    if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(field, recvFirst, bytes, cudaMemcpyHostToDevice, transfer));
    if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, recvLast, bytes, cudaMemcpyHostToDevice, transfer));
    CUDA_CHECK(cudaStreamSynchronize(transfer));
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("  -x <num>  Grid size in X dimension (default: 64)\n");
    printf("  -y <num>  Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>  Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>  Number of time steps (default: 20)\n");
    printf("  -v        Enable validation\n");
    printf("  -r        Print results for external validation\n");
    printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int worldRank, worldSize;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny > INT_MAX) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or time step count\n");
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    int ranks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (worldRank >= ranks) {
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    int rank = worldRank;
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm shared;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
    int localRank;
    MPI_CHECK(MPI_Comm_rank(shared, &localRank));
    MPI_CHECK(MPI_Comm_free(&shared));
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    size_t plane = nx * ny;
    size_t startZ = nz * static_cast<size_t>(rank) / ranks;
    size_t endZ = nz * static_cast<size_t>(rank + 1) / ranks;
    size_t localNz = endZ - startZ;
    size_t localCells = localNz * plane;
    double *oldC, *newC, *mu, *buffers;
    CUDA_CHECK(cudaMalloc(&oldC, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&newC, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, (localNz + 2) * plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&buffers, 4 * plane * sizeof(double)));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    // Initialize the distributed host slabs in parallel, preserving the original sequence.
    std::vector<double> initial(localCells);
    const size_t volume = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localCells; ++i) {
        size_t linear = startZ * plane + i;
        size_t pseudoInt = ((linear + 1) * static_cast<size_t>(1299709)) % volume;
        initial[i] = -1.0 + 2.0 * (pseudoInt / static_cast<double>(volume));
    }
    CUDA_CHECK(cudaMemcpy(oldC + plane, initial.data(), localCells * sizeof(double), cudaMemcpyHostToDevice));
    initial.clear();
    initial.shrink_to_fit();
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        launchChemical(oldC, mu, nx, ny, localNz, 2, localNz > 1 ? localNz - 1 : 0, rank, ranks, compute);
        exchangeHalos(oldC, plane, localNz, rank, ranks, buffers, transfer, comm);
        launchChemical(oldC, mu, nx, ny, localNz, 1, 1, rank, ranks, compute);
        if (localNz > 1) launchChemical(oldC, mu, nx, ny, localNz, localNz, localNz, rank, ranks, compute);
        CUDA_CHECK(cudaStreamSynchronize(compute));
        launchUpdate(oldC, mu, newC, nx, ny, localNz, 2, localNz > 1 ? localNz - 1 : 0, rank, ranks, compute);
        exchangeHalos(mu, plane, localNz, rank, ranks, buffers, transfer, comm);
        launchUpdate(oldC, mu, newC, nx, ny, localNz, 1, 1, rank, ranks, compute);
        if (localNz > 1) launchUpdate(oldC, mu, newC, nx, ny, localNz, localNz, localNz, rank, ranks, compute);
        CUDA_CHECK(cudaStreamSynchronize(compute));
        std::swap(oldC, newC);
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? nx * ny * nz * static_cast<double>(iterations) / duration / 1e6 : 0.0);
    }
    int resultCode = 0;
    if (printResults || validate) {
        std::vector<double> local(localCells);
        CUDA_CHECK(cudaMemcpy(local.data(), oldC + plane, localCells * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            std::vector<int> counts, displacements;
            if (rank == 0) {
                counts.resize(ranks); displacements.resize(ranks);
                for (int r = 0; r < ranks; ++r) {
                    size_t a = nz * static_cast<size_t>(r) / ranks;
                    size_t b = nz * static_cast<size_t>(r + 1) / ranks;
                    if ((b - a) * plane > INT_MAX || a * plane > INT_MAX) {
                        fprintf(stderr, "Grid too large for MPI result gathering\n"); MPI_Abort(comm, 1);
                    }
                    counts[r] = static_cast<int>((b - a) * plane);
                    displacements[r] = static_cast<int>(a * plane);
                }
            }
            if (localCells > INT_MAX) { fprintf(stderr, "Local grid too large for MPI result gathering\n"); MPI_Abort(comm, 1); }
            std::vector<double> global;
            if (rank == 0) global.resize(nx * ny * nz);
            MPI_CHECK(MPI_Gatherv(local.data(), static_cast<int>(localCells), MPI_DOUBLE,
                                  rank == 0 ? global.data() : nullptr,
                                  rank == 0 ? counts.data() : nullptr,
                                  rank == 0 ? displacements.data() : nullptr,
                                  MPI_DOUBLE, 0, comm));
            if (rank == 0) print_results(global, "Concentration");
        }
        if (validate) {
            double localMin = local[0], localMax = local[0];
            int invalid = 0;
            #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(|:invalid)
            for (size_t i = 0; i < localCells; ++i) {
                if (!std::isfinite(local[i])) invalid = 1;
                else { localMin = std::min(localMin, local[i]); localMax = std::max(localMax, local[i]); }
            }
            double minValue, maxValue;
            int anyInvalid;
            MPI_CHECK(MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
            MPI_CHECK(MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            MPI_CHECK(MPI_Reduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, 0, comm));
            if (rank == 0) {
                printf("Validating result...\n");
                if (anyInvalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
                    if (maxValue > 10.0 || minValue < -10.0)
                        printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", anyInvalid || maxValue > 10.0 || minValue < -10.0 ? "FAILED" : "PASSED");
            }
            int valid = 0;
            if (rank == 0) valid = !(anyInvalid || maxValue > 10.0 || minValue < -10.0);
            MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, comm));
            if (!valid) resultCode = 1;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFreeHost(buffers));
    CUDA_CHECK(cudaFree(oldC)); CUDA_CHECK(cudaFree(newC)); CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return resultCode;
}
