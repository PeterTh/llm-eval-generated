#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

static int worldRank = 0;

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", worldRank, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int err_ = (call); \
    if (err_ != MPI_SUCCESS) { \
        char msg_[MPI_MAX_ERROR_STRING]; int len_; MPI_Error_string(err_, msg_, &len_); \
        fprintf(stderr, "Rank %d: MPI error at %s:%d: %.*s\n", worldRank, __FILE__, __LINE__, len_, msg_); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// zBegin and zEnd are inclusive/exclusive indices in the local, halo-padded slab.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t zBegin, size_t zEnd) {
    size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t y = (size_t)blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t z = zBegin + blockIdx.z;
    if (x >= nx - 1 || z >= zEnd) return;
    size_t plane = nx * ny;
    for (; y < ny - 1; y += (size_t)gridDim.y * blockDim.y) {
        size_t i = z * plane + y * nx + x;
        out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - nx] +
                  in[i + nx] + in[i - plane] + in[i + plane]) / 7.0;
    }
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny,
                          size_t begin, size_t end, cudaStream_t stream) {
    if (nx <= 2 || ny <= 2 || begin >= end) return;
    dim3 block(32, 8);
    // CUDA's z grid dimension is limited to 65535.
    for (size_t z = begin; z < end;) {
        size_t n = std::min(end - z, (size_t)65535);
        dim3 grid((unsigned)((nx - 2 + block.x - 1) / block.x),
                  (unsigned)std::min((ny - 2 + block.y - 1) / block.y, (size_t)65535), (unsigned)n);
        stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, z, z + n);
        CUDA_CHECK(cudaGetLastError());
        z += n;
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Split large transfers so MPI's int count never truncates a plane or slab.
static void postPlane(Real* data, size_t count, int peer, int tag, bool receive,
                      MPI_Comm comm, std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        int n = (int)std::min(count - offset, (size_t)INT_MAX);
        MPI_Request req;
        if (receive) MPI_CHECK(MPI_Irecv(data + offset, n, MPI_DOUBLE, peer, tag, comm, &req));
        else MPI_CHECK(MPI_Isend(data + offset, n, MPI_DOUBLE, peer, tag, comm, &req));
        requests.push_back(req);
        offset += (size_t)n;
    }
}

static void transferSlab(Real* data, size_t count, int peer, bool receive, MPI_Comm comm) {
    for (size_t offset = 0; offset < count;) {
        int n = (int)std::min(count - offset, (size_t)INT_MAX);
        if (receive) MPI_CHECK(MPI_Recv(data + offset, n, MPI_DOUBLE, peer, 9, comm, MPI_STATUS_IGNORE));
        else MPI_CHECK(MPI_Send(data + offset, n, MPI_DOUBLE, peer, 9, comm));
        offset += (size_t)n;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!worldRank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!worldRank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(Real) || iterations < 0) {
        if (!worldRank) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize(); return 1;
    }
    size_t plane = nx * ny;
    if (plane > SIZE_MAX / sizeof(Real) || plane > SIZE_MAX / (nz + 2)) {
        if (!worldRank) fprintf(stderr, "Grid is too large\n");
        MPI_Finalize(); return 1;
    }
    if (!worldRank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Ranks without a plane are excluded from the computation and final collection.
    int active = (int)std::min(nz, (size_t)worldSize);
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (worldRank >= active) { MPI_Finalize(); return 0; }
    int rank = worldRank;
    size_t localNz = nz / active + ((size_t)rank < nz % active);
    size_t startZ = (size_t)rank * (nz / active) + std::min((size_t)rank, nz % active);
    size_t localSize = (localNz + 2) * plane;
    if (localSize > SIZE_MAX / sizeof(Real)) {
        fprintf(stderr, "Rank %d: local grid is too large\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&localComm));

    std::vector<Real> host(localSize, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localNz * plane; ++i)
        host[plane + i] = (Real)((startZ * plane + i) % 19);
    Real *in, *out;
    CUDA_CHECK(cudaMalloc(&in, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&out, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(in, host.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t compute, halo;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&halo, cudaStreamNonBlocking));
    Real *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;
    if (rank > 0) {
        CUDA_CHECK(cudaMallocHost(&sendLow, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recvLow, plane * sizeof(Real)));
    }
    if (rank + 1 < active) {
        CUDA_CHECK(cudaMallocHost(&sendHigh, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&recvHigh, plane * sizeof(Real)));
    }
    if (!rank) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (active == 1) {
            size_t begin = startZ == 0 ? 2 : 1;
            size_t end = localNz + 1 - (startZ + localNz == nz ? 1 : 0);
            launchStencil(in, out, nx, ny, begin, end, compute);
            // The next iteration depends on this kernel; stream ordering suffices.
        } else {
            std::vector<MPI_Request> requests;
            if (rank > 0) postPlane(recvLow, plane, rank - 1, 1, true, comm, requests);
            if (rank + 1 < active) postPlane(recvHigh, plane, rank + 1, 0, true, comm, requests);
            if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(sendLow, in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, halo));
            if (rank + 1 < active) CUDA_CHECK(cudaMemcpyAsync(sendHigh, in + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, halo));
            launchStencil(in, out, nx, ny, 2, localNz, compute);
            CUDA_CHECK(cudaStreamSynchronize(halo));
            if (rank > 0) postPlane(sendLow, plane, rank - 1, 0, false, comm, requests);
            if (rank + 1 < active) postPlane(sendHigh, plane, rank + 1, 1, false, comm, requests);
            if (!requests.empty()) MPI_CHECK(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE));
            if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(in, recvLow, plane * sizeof(Real), cudaMemcpyHostToDevice, halo));
            if (rank + 1 < active) CUDA_CHECK(cudaMemcpyAsync(in + (localNz + 1) * plane, recvHigh, plane * sizeof(Real), cudaMemcpyHostToDevice, halo));
            if (startZ > 0 && startZ + 1 < nz) launchStencil(in, out, nx, ny, 1, 2, halo);
            if (localNz > 1 && startZ + localNz - 1 > 0 && startZ + localNz < nz)
                launchStencil(in, out, nx, ny, localNz, localNz + 1, halo);
            CUDA_CHECK(cudaStreamSynchronize(compute));
            CUDA_CHECK(cudaStreamSynchronize(halo));
        }
        std::swap(in, out);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        long milliseconds = (long)(maxElapsed * 1000.0);
        double updates = (double)(nx > 2 ? nx - 2 : 0) * (double)(ny > 2 ? ny - 2 : 0) *
                         (double)(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\n", milliseconds);
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }
    if (validate || printResults)
        CUDA_CHECK(cudaMemcpy(host.data() + plane, in + plane, localNz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (!rank) {
            std::vector<Real> grid(nx * ny * nz);
            std::copy(host.begin() + plane, host.begin() + (localNz + 1) * plane, grid.begin());
            for (int p = 1; p < active; ++p) {
                size_t pn = nz / active + ((size_t)p < nz % active);
                size_t pz = (size_t)p * (nz / active) + std::min((size_t)p, nz % active);
                transferSlab(grid.data() + pz * plane, pn * plane, p, true, comm);
            }
            print_results(grid, "Grid");
        } else transferSlab(host.data() + plane, localNz * plane, 0, false, comm);
    }
    int valid = 1;
    if (validate) {
        Real minVal = std::numeric_limits<Real>::infinity(), maxVal = -minVal;
        int bad = 0;
        #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:bad) schedule(static)
        for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
            Real value = host[i];
            if (!std::isfinite(value)) bad = 1;
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        }
        Real globalMin, globalMax; int globalBad;
        MPI_CHECK(MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
        MPI_CHECK(MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        MPI_CHECK(MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, 0, comm));
        if (!rank) {
            printf("Validating result...\n");
            if (globalBad) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 1e6 || globalMin < -1e6) { printf("Validation failed: values out of expected range\n"); valid = 0; }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, comm));
    }
    if (sendLow) CUDA_CHECK(cudaFreeHost(sendLow));
    if (sendHigh) CUDA_CHECK(cudaFreeHost(sendHigh));
    if (recvLow) CUDA_CHECK(cudaFreeHost(recvLow));
    if (recvHigh) CUDA_CHECK(cudaFreeHost(recvHigh));
    CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(halo));
    CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(out));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_Finalize();
    return valid ? 0 : 1;
}
