#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __LINE__)

// Each slab has one ghost plane on each side. Physical faces clamp to the
// central cell; MPI faces read the adjacent rank's ghost plane. The benchmark's
// fixed dx = dy = dz = D = 1, dt = 0.01 and gamma = 0.5 are folded into kernels.
__device__ __forceinline__ double laplacian(const double* c, size_t i, size_t x,
                                          size_t y, size_t z, size_t nx, size_t ny,
                                          size_t localZ, bool first, bool last) {
    const size_t plane = nx * ny;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t xn = x ? i - 1 : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t yn = y ? i - nx : i;
    const size_t zp = last && z == localZ ? i : i + plane;
    const size_t zn = first && z == 1 ? i : i - plane;
    const double cxx = c[xp] + c[xn] - 2.0 * c[i];
    const double cyy = c[yp] + c[yn] - 2.0 * c[i];
    const double czz = c[zp] + c[zn] - 2.0 * c[i];
    return cxx + cyy + czz;
}

template<bool Chemical>
__global__ void stencil(const double* __restrict__ input, double* __restrict__ output,
                        const double* __restrict__ cold, size_t nx, size_t ny,
                        size_t localZ, bool first, bool last, size_t beginZ,
                        size_t planes, bool boundary) {
    const size_t plane = nx * ny;
    const size_t count = planes * plane;
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < count;
         t += size_t(blockDim.x) * gridDim.x) {
        const size_t z = boundary ? (t / plane == 0 ? 1 : localZ) : beginZ + t / plane;
        const size_t p = t % plane;
        const size_t i = z * plane + p;
        const double lap = laplacian(input, i, p % nx, p / nx, z, nx, ny, localZ, first, last);
        if (Chemical) {
            const double cv = input[i];
            constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            output[i] = cold[i] + 0.01 * lap;
        }
    }
}

struct Slab {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, received;
    double* staging;

    template<bool Chemical>
    void launch(const double* input, double* output, const double* cold,
                size_t begin, size_t planes, bool boundary = false) {
        if (!planes) return;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>((planes * plane + 255) / 256, 65535));
        stencil<Chemical><<<blocks, 256, 0, compute>>>(input, output, cold, nx, ny, nz,
                                                     rank == 0, rank == ranks - 1,
                                                     begin, planes, boundary);
        CUDA_CHECK(cudaGetLastError());
    }

    template<bool Chemical>
    void step(double* input, double* output, const double* cold) {
        if (ranks == 1) {
            launch<Chemical>(input, output, cold, 1, nz);
            return;
        }
        MPI_Request requests[4];
        int n = 0;
        const int count = static_cast<int>(plane);
        const size_t bytes = plane * sizeof(double);
        // Receive buffers and send buffers are disjoint and remain pinned until
        // both MPI and the asynchronous CUDA transfers have completed.
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        if (rank > 0) MPI_Irecv(staging + 2 * plane, count, MPI_DOUBLE, rank - 1, 1, comm, &requests[n++]);
        if (rank + 1 < ranks) MPI_Irecv(staging + 3 * plane, count, MPI_DOUBLE, rank + 1, 0, comm, &requests[n++]);
        CUDA_CHECK(cudaEventRecord(ready, compute));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(staging, input + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (rank + 1 < ranks)
            CUDA_CHECK(cudaMemcpyAsync(staging + plane, input + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        launch<Chemical>(input, output, cold, 2, nz > 2 ? nz - 2 : 0);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        if (rank > 0) MPI_Isend(staging, count, MPI_DOUBLE, rank - 1, 0, comm, &requests[n++]);
        if (rank + 1 < ranks) MPI_Isend(staging + plane, count, MPI_DOUBLE, rank + 1, 1, comm, &requests[n++]);
        MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(input, staging + 2 * plane, bytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks)
            CUDA_CHECK(cudaMemcpyAsync(input + (nz + 1) * plane, staging + 3 * plane, bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(received, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
        launch<Chemical>(input, output, cold, 1, nz > 1 ? 2 : 1, true);
    }
};

bool validateResult(const std::vector<double>& c, MPI_Comm comm, int rank) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
    int invalid = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        invalid |= !std::isfinite(c[i]);
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        if (invalid) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !invalid && maxVal <= 10.0 && minVal >= -10.0;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx > static_cast<size_t>(INT_MAX) / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / sizeof(double) / (nz + 2)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or unsupported grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    // Ranks in excess of the number of planes have no cells to evolve.
    const int activeRanks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "A CUDA device is required on every active rank\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", activeRanks, omp_get_max_threads());
    }
    const size_t plane = nx * ny;
    const size_t localZ = nz / activeRanks + (static_cast<size_t>(rank) < nz % activeRanks);
    const size_t startZ = rank * (nz / activeRanks) + std::min<size_t>(rank, nz % activeRanks);
    const size_t gridSize = plane * nz;
    const size_t localSize = plane * localZ;
    const size_t bytes = (localZ + 2) * plane * sizeof(double);
    std::vector<double> host(localSize);
    if (rank == 0) printf("Initializing concentration field...\n");
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localSize; ++i) {
        const size_t linear_id = startZ * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), localSize * sizeof(double), cudaMemcpyHostToDevice));
    Slab slab{nx, ny, localZ, plane, rank, activeRanks, comm, {}, {}, {}, {}, nullptr};
    CUDA_CHECK(cudaStreamCreateWithFlags(&slab.compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&slab.transfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&slab.ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&slab.received, cudaEventDisableTiming));
    if (activeRanks > 1) CUDA_CHECK(cudaMallocHost(&slab.staging, 4 * plane * sizeof(double)));
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        slab.step<true>(cold, mu, cold);
        slab.step<false>(mu, cnew, cold);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(slab.compute));
    double seconds = MPI_Wtime() - start;
    MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", double(gridSize) * iterations / seconds / 1e6);
    }
    if (printResults || validate)
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, localSize * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Gather only for requested output, in the original global array order.
        // Chunked transfers avoid MPI's int displacement/count limitation.
        std::vector<double> global;
        if (rank == 0) {
            global.resize(gridSize);
            std::copy(host.begin(), host.end(), global.begin());
        }
        for (int r = 1; r < activeRanks; ++r) {
            const size_t rz = nz / activeRanks + (static_cast<size_t>(r) < nz % activeRanks);
            const size_t offset = (r * (nz / activeRanks) + std::min<size_t>(r, nz % activeRanks)) * plane;
            for (size_t pos = 0; pos < rz * plane;) {
                const int count = static_cast<int>(std::min<size_t>(rz * plane - pos, INT_MAX));
                if (rank == r) MPI_Send(host.data() + pos, count, MPI_DOUBLE, 0, 2, comm);
                if (rank == 0) MPI_Recv(global.data() + offset + pos, count, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                pos += count;
            }
        }
        if (rank == 0) print_results(global, "Concentration");
    }
    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(host, comm, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        result = valid ? 0 : 1;
    }
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    if (slab.staging) CUDA_CHECK(cudaFreeHost(slab.staging));
    CUDA_CHECK(cudaEventDestroy(slab.ready));
    CUDA_CHECK(cudaEventDestroy(slab.received));
    CUDA_CHECK(cudaStreamDestroy(slab.compute));
    CUDA_CHECK(cudaStreamDestroy(slab.transfer));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return result;
}
