#include <algorithm>
#include <chrono>
#include <climits>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

static void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0, rank = 0;
        MPI_Error_string(status, error, &length);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %.*s\n", rank, operation, length, error);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}
#define MPI_CHECK(call) mpiCheck((call), #call)

// Arrays contain one ghost plane on each side of the local z slab.
__device__ inline double laplacian(const double* a, size_t x, size_t y, size_t index,
                                   size_t nx, size_t ny, size_t zm, size_t zp) {
    const size_t xp = x < nx - 1 ? index + 1 : index;
    const size_t xn = x > 0 ? index - 1 : index;
    const size_t yp = y < ny - 1 ? index + nx : index;
    const size_t yn = y > 0 ? index - nx : index;
    const double cxx = (a[xp] + a[xn] - 2.0 * a[index]) / (1.0 * 1.0);
    const double cyy = (a[yp] + a[yn] - 2.0 * a[index]) / (1.0 * 1.0);
    const double czz = (a[zp] + a[zm] - 2.0 * a[index]) / (1.0 * 1.0);
    return cxx + cyy + czz;
}

template <bool Chemical, bool ClampZ>
__global__ void stepKernel(const double* c, const double* muIn, double* out,
                           size_t nx, size_t ny, size_t plane, size_t firstZ,
                           size_t depth, size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += stride) {
        const size_t index = firstZ * plane + p;
        const size_t x = p % nx;
        const size_t y = (p / nx) % ny;
        size_t zm = index - plane, zp = index + plane;
        if constexpr (ClampZ) {
            const size_t z = firstZ + p / plane;
            if (z == 1) zm = index;
            if (z == depth) zp = index;
        }
        if constexpr (Chemical) {
            const double cv = c[index];
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            out[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - 0.5 * laplacian(c, x, y, index, nx, ny, zm, zp);
        } else {
            out[index] = c[index] + 0.01 * 1.0 * laplacian(muIn, x, y, index, nx, ny, zm, zp);
        }
    }
}

static bool validateResult(const std::vector<double>& c) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minVal = c[0], maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Host-staged communication works with ordinary MPI implementations and does not
// require CUDA-aware MPI. Copies and communication overlap the slab interior.
struct HaloExchange {
    size_t nx, ny, plane, depth;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, halosReady;
    double *send, *recv;

    HaloExchange(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), plane(x * y), depth(z), rank(r), ranks(n), comm(c) {
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
        CUDA(cudaMallocHost(&send, 2 * plane * sizeof(double)));
        CUDA(cudaMallocHost(&recv, 2 * plane * sizeof(double)));
        CUDA(cudaEventRecord(ready, compute));
    }
    ~HaloExchange() {
        CUDA(cudaFreeHost(send));
        CUDA(cudaFreeHost(recv));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaEventDestroy(halosReady));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(transfer));
    }
    template<bool Chemical>
    void run(const double* c, const double* mu, double* out, double* field) {
        if (ranks == 1) {
            launch<Chemical, true>(c, mu, out, 1, depth * plane);
            return;
        }
        const size_t bytes = plane * sizeof(double);
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        if (rank > 0)
            CUDA(cudaMemcpyAsync(send, field + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        else
            CUDA(cudaMemcpyAsync(field, field + plane, bytes, cudaMemcpyDeviceToDevice, transfer));
        if (rank + 1 < ranks)
            CUDA(cudaMemcpyAsync(send + plane, field + depth * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        else
            CUDA(cudaMemcpyAsync(field + (depth + 1) * plane, field + depth * plane, bytes, cudaMemcpyDeviceToDevice, transfer));

        // The first and last local planes need the exchanged ghost planes.
        if (depth > 2) launch<Chemical>(c, mu, out, 2, (depth - 2) * plane);

        if (ranks > 1) {
            CUDA(cudaStreamSynchronize(transfer));
            std::vector<MPI_Request> requests;
            for (size_t offset = 0; offset < plane;) {
                const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
                MPI_Request request;
                if (rank > 0) {
                    MPI_CHECK(MPI_Irecv(recv + offset, count, MPI_DOUBLE, rank - 1, 1, comm, &request));
                    requests.push_back(request);
                    MPI_CHECK(MPI_Isend(send + offset, count, MPI_DOUBLE, rank - 1, 0, comm, &request));
                    requests.push_back(request);
                }
                if (rank + 1 < ranks) {
                    MPI_CHECK(MPI_Irecv(recv + plane + offset, count, MPI_DOUBLE, rank + 1, 0, comm, &request));
                    requests.push_back(request);
                    MPI_CHECK(MPI_Isend(send + plane + offset, count, MPI_DOUBLE, rank + 1, 1, comm, &request));
                    requests.push_back(request);
                }
                offset += static_cast<size_t>(count);
            }
            MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            if (rank > 0)
                CUDA(cudaMemcpyAsync(field, recv, bytes, cudaMemcpyHostToDevice, transfer));
            if (rank + 1 < ranks)
                CUDA(cudaMemcpyAsync(field + (depth + 1) * plane, recv + plane, bytes, cudaMemcpyHostToDevice, transfer));
        }
        CUDA(cudaEventRecord(halosReady, transfer));
        CUDA(cudaStreamWaitEvent(compute, halosReady, 0));
        launch<Chemical>(c, mu, out, 1, plane);
        if (depth > 1) launch<Chemical>(c, mu, out, depth, plane);
        CUDA(cudaEventRecord(ready, compute));
    }
    template<bool Chemical, bool ClampZ = false>
    void launch(const double* c, const double* mu, double* out, size_t first, size_t count) {
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>(std::min((count - 1) / threads + 1, size_t(65535)));
        stepKernel<Chemical, ClampZ><<<blocks, threads, 0, compute>>>(c, mu, out, nx, ny, plane, first, depth, count);
        CUDA(cudaGetLastError());
    }
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int worldRank, worldSize;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (worldRank == 0) printUsage(argv[0]); parseStatus = 2; break; }
        else { if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseStatus = 1; break; }
    }
    if (parseStatus) { MPI_Finalize(); return parseStatus == 2 ? 0 : 1; }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxSize / ny || nx * ny > maxSize / nz ||
        nx * ny > maxSize / (2 * sizeof(double)) || nx * ny * nz > maxSize / sizeof(double)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or excessively large grid size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int activeSize = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &active));
    int exitStatus = 0;
    if (worldRank < activeSize) {
        int localRank = 0;
        MPI_Comm node;
        MPI_CHECK(MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node));
        MPI_CHECK(MPI_Comm_rank(node, &localRank));
        int deviceCount = 0;
        CUDA(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) { fprintf(stderr, "Rank %d: no CUDA devices available\n", worldRank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(localRank % deviceCount));
        MPI_CHECK(MPI_Comm_free(&node));
        const size_t depth = nz / activeSize + (static_cast<size_t>(worldRank) < nz % activeSize);
        const size_t zStart = static_cast<size_t>(worldRank) * (nz / activeSize) + std::min(static_cast<size_t>(worldRank), nz % activeSize);
        const size_t localSize = depth * plane;
        if (depth > maxSize / plane - 2 || (depth + 2) * plane > maxSize / sizeof(double)) {
            fprintf(stderr, "Rank %d: local grid size overflow\n", worldRank); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (worldRank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        std::vector<double> initial(localSize);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localSize; ++i) {
            const size_t linear_id = zStart * plane + i;
            const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            initial[i] = -1.0 + 2.0 * pseudo;
        }
        double *cold, *cnew, *mu;
        const size_t allocBytes = (depth + 2) * plane * sizeof(double);
        CUDA(cudaMalloc(&cold, allocBytes)); CUDA(cudaMalloc(&cnew, allocBytes)); CUDA(cudaMalloc(&mu, allocBytes));
        CUDA(cudaMemcpy(cold + plane, initial.data(), localSize * sizeof(double), cudaMemcpyHostToDevice));
        if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
        {
            HaloExchange halo(nx, ny, depth, worldRank, activeSize, active);
            MPI_CHECK(MPI_Barrier(active));
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                halo.run<true>(cold, nullptr, mu, cold);
                halo.run<false>(cold, mu, cnew, mu);
                std::swap(cold, cnew);
            }
            CUDA(cudaStreamSynchronize(halo.compute));
            const double elapsed = MPI_Wtime() - start;
            double maxElapsed = 0.0;
            MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active));
            if (worldRank == 0) {
                const long millis = static_cast<long>(maxElapsed * 1000.0);
                printf("Computation time: %ld ms\n", millis);
                printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / maxElapsed / 1e6);
            }
        }
        if (validate || printResults) {
            CUDA(cudaMemcpy(initial.data(), cold + plane, localSize * sizeof(double), cudaMemcpyDeviceToHost));
            if (printResults) {
                std::vector<double> global;
                if (worldRank == 0) {
                    global.resize(gridSize);
                    std::copy(initial.begin(), initial.end(), global.begin());
                    for (int r = 1; r < activeSize; ++r) {
                        const size_t rd = nz / activeSize + (static_cast<size_t>(r) < nz % activeSize);
                        const size_t rz = static_cast<size_t>(r) * (nz / activeSize) + std::min(static_cast<size_t>(r), nz % activeSize);
                        for (size_t off = 0; off < rd * plane;) {
                            const int count = static_cast<int>(std::min(rd * plane - off, static_cast<size_t>(INT_MAX)));
                            MPI_CHECK(MPI_Recv(global.data() + rz * plane + off, count, MPI_DOUBLE, r, 2, active, MPI_STATUS_IGNORE));
                            off += static_cast<size_t>(count);
                        }
                    }
                    print_results(global, "Concentration");
                    if (validate) {
                        printf("Validating result...\n");
                        const bool valid = validateResult(global);
                        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                        exitStatus = valid ? 0 : 1;
                    }
                } else {
                    for (size_t off = 0; off < localSize;) {
                        const int count = static_cast<int>(std::min(localSize - off, static_cast<size_t>(INT_MAX)));
                        MPI_CHECK(MPI_Send(initial.data() + off, count, MPI_DOUBLE, 0, 2, active));
                        off += static_cast<size_t>(count);
                    }
                }
            } else {
                double localMin = std::numeric_limits<double>::infinity();
                double localMax = -std::numeric_limits<double>::infinity();
                int localInvalid = 0;
                #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(|:localInvalid) schedule(static)
                for (size_t i = 0; i < localSize; ++i) {
                    const double value = initial[i];
                    if (!std::isfinite(value)) localInvalid = 1;
                    else { localMin = std::min(localMin, value); localMax = std::max(localMax, value); }
                }
                double globalMin = 0.0, globalMax = 0.0;
                int globalInvalid = 0;
                MPI_CHECK(MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, active));
                MPI_CHECK(MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, active));
                MPI_CHECK(MPI_Reduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, 0, active));
                if (worldRank == 0) {
                    printf("Validating result...\n");
                    if (globalInvalid) {
                        printf("Validation failed: found NaN or Inf value\n");
                        exitStatus = 1;
                    } else {
                        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                        if (globalMax > 10.0 || globalMin < -10.0) {
                            printf("Validation failed: values out of expected range\n");
                            exitStatus = 1;
                        }
                    }
                    printf("Validation: %s\n", exitStatus ? "FAILED" : "PASSED");
                }
            }
        }
        CUDA(cudaFree(cold)); CUDA(cudaFree(cnew)); CUDA(cudaFree(mu));
        MPI_CHECK(MPI_Comm_free(&active));
    }
    MPI_CHECK(MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitStatus;
}
