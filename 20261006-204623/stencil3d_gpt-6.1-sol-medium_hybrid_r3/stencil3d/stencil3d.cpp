#include <algorithm>
#include <cerrno>
#include <climits>
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

using Real = double;

static void mpiCheck(int code) {
    if (code != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(code, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}
static void cudaCheck(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(code));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each warp accesses consecutive X coordinates. Rolling Z registers reuse two
// of the three plane loads; the remaining neighbors reuse the GPU's L1 cache.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output, size_t nx, size_t ny,
                              size_t begin, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z0 = begin + size_t(blockIdx.z) * 8;
    if (x >= nx - 1 || y >= ny - 1 || z0 >= end) return;
    const size_t plane = nx * ny;
    size_t i = z0 * plane + y * nx + x;
    Real bottom = input[i - plane];
    Real center = input[i];
    const size_t stop = end < z0 + 8 ? end : z0 + 8;
    for (size_t z = z0; z < stop; ++z, i += plane) {
        const Real top = input[i + plane];
        // Preserve the reference's addition order and correctly rounded division.
        output[i] = __ddiv_rn(center + input[i-1] + input[i+1] +
                             input[i-nx] + input[i+nx] + bottom + top, 7.0);
        bottom = center;
        center = top;
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t begin, size_t end, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || begin >= end) return;
    const dim3 block(32, 4);
    // Chunk Z to respect CUDA's grid dimension limit even for deep slabs.
    const size_t chunk = size_t(65535) * 8;
    for (size_t z = begin; z < end; z += std::min(chunk, end-z)) {
        const size_t stop = z + std::min(chunk, end-z);
        const dim3 grid((nx-2+31)/32, (ny-2+3)/4, (stop-z+7)/8);
        stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, z, stop);
        cudaCheck(cudaGetLastError());
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}
static bool parseNumber(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = size_t(parsed);
    return true;
}

// MPI counts are int: chunk transfers rather than limiting the global grid.
static void postPlane(Real* data, size_t count, int peer, int tag,
                      bool send, MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        const int n = int(std::min(count-offset, size_t(INT_MAX)));
        MPI_Request request;
        mpiCheck(send ? MPI_Isend(data+offset, n, MPI_DOUBLE, peer, tag, comm, &request)
                      : MPI_Irecv(data+offset, n, MPI_DOUBLE, peer, tag, comm, &request));
        requests.push_back(request);
        offset += size_t(n);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int worldRank, worldSize;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    size_t nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false, help = false, good = true;
    for (int i = 1; i < argc; ++i) {
        size_t* target = nullptr;
        if (!strcmp(argv[i], "-x")) target = &nx;
        else if (!strcmp(argv[i], "-y")) target = &ny;
        else if (!strcmp(argv[i], "-z")) target = &nz;
        else if (!strcmp(argv[i], "-i")) target = &iterations;
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { good = false; break; }
        if (target && (i+1 == argc || !parseNumber(argv[++i], *target))) {
            good = false; break;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help || !good || !nx || !ny || !nz) {
        if (!worldRank) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx > limit / ny || nx*ny > limit / nz ||
        (nx > 2 && (nx-2+31)/32 > UINT_MAX) || (ny > 2 && (ny-2+3)/4 > 65535)) {
        if (!worldRank) fprintf(stderr, "Grid dimensions exceed address or CUDA grid limits\n");
        MPI_Finalize();
        return 1;
    }
    // Empty ranks do not participate in slab communication, but join the final
    // world-wide status reduction. At least one plane belongs to each active rank.
    const int activeSize = int(std::min(nz, size_t(worldSize)));
    MPI_Comm comm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                           worldRank, &comm));
    int status = 0;
    if (comm != MPI_COMM_NULL) {
        const int rank = worldRank;
        MPI_Comm node;
        mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
        int nodeRank, devices;
        mpiCheck(MPI_Comm_rank(node, &nodeRank));
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        cudaCheck(cudaSetDevice(nodeRank % devices));
        mpiCheck(MPI_Comm_free(&node));
        const size_t plane = nx * ny;
        const size_t localZ = nz / activeSize + (size_t(rank) < nz % activeSize);
        const size_t startZ = size_t(rank) * (nz / activeSize) + std::min(size_t(rank), nz % activeSize);
        if (localZ > limit / plane || limit / plane - localZ < 2) MPI_Abort(MPI_COMM_WORLD, 1);
        const size_t count = (localZ + 2) * plane;
        std::vector<Real> host(count);
        #pragma omp parallel for schedule(static)
        for (size_t z = 0; z < localZ + 2; ++z) {
            if ((z == 0 && startZ == 0) || startZ + z > nz) continue;
            const size_t globalZ = startZ + z - 1;
            for (size_t p = 0; p < plane; ++p)
                host[z*plane+p] = Real((globalZ*plane+p) % 19);
        }
        Real *a = nullptr, *b = nullptr, *halos = nullptr;
        cudaCheck(cudaMalloc(&a, count*sizeof(Real)));
        cudaCheck(cudaMalloc(&b, count*sizeof(Real)));
        cudaCheck(cudaMemcpy(a, host.data(), count*sizeof(Real), cudaMemcpyHostToDevice));
        // Both buffers start with the fixed boundaries, so kernels only write interiors.
        cudaCheck(cudaMemcpy(b, a, count*sizeof(Real), cudaMemcpyDeviceToDevice));
        const int lower = rank == 0 ? MPI_PROC_NULL : rank-1;
        const int upper = rank+1 == activeSize ? MPI_PROC_NULL : rank+1;
        const bool exchange = activeSize > 1 && nx > 2 && ny > 2 && nz > 2;
        if (exchange) {
            if (plane > limit / 4) MPI_Abort(MPI_COMM_WORLD, 1);
            cudaCheck(cudaMallocHost(&halos, 4*plane*sizeof(Real)));
        }
        cudaStream_t compute, transfer;
        cudaEvent_t ready, done;
        cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        cudaCheck(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        cudaCheck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        cudaCheck(cudaEventRecord(done, compute));
        const size_t begin = startZ == 0 ? 2 : 1;
        const size_t end = startZ + localZ == nz ? localZ : localZ+1;
        std::vector<MPI_Request> requests;
        requests.reserve(4 * (plane / INT_MAX + 1));
        if (!rank) {
            printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %zu\n", nx, ny, nz, iterations);
            printf("Validation: %s\nInitializing grid...\nRunning stencil computation...\n", validate ? "enabled" : "disabled");
        }
        mpiCheck(MPI_Barrier(comm));
        const double start = MPI_Wtime();
        for (size_t iter = 0; iter < iterations; ++iter) {
            if (!exchange || iter == 0) {
                // Initialization already supplied the exact first-iteration halos.
                launch(a, b, nx, ny, begin, end, compute);
                if (exchange) cudaCheck(cudaEventRecord(done, compute));
            } else {
                // Receives are posted early; pinned staging works with ordinary
                // MPI as well as CUDA-aware MPI without requiring vendor extensions.
                // MPI must not overwrite a receive buffer while the previous
                // asynchronous host-to-device transfer is still reading it.
                if (iter > 1) cudaCheck(cudaEventSynchronize(ready));
                requests.clear();
                postPlane(halos+2*plane, plane, lower, 1, false, comm, requests);
                postPlane(halos+3*plane, plane, upper, 0, false, comm, requests);
                cudaCheck(cudaStreamWaitEvent(transfer, done, 0));
                if (lower != MPI_PROC_NULL)
                    cudaCheck(cudaMemcpyAsync(halos, a+plane, plane*sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                if (upper != MPI_PROC_NULL)
                    cudaCheck(cudaMemcpyAsync(halos+plane, a+localZ*plane, plane*sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                launch(a, b, nx, ny, std::max(begin, size_t(2)), std::min(end, localZ), compute);
                cudaCheck(cudaStreamSynchronize(transfer));
                postPlane(halos, plane, lower, 0, true, comm, requests);
                postPlane(halos+plane, plane, upper, 1, true, comm, requests);
                mpiCheck(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
                if (lower != MPI_PROC_NULL)
                    cudaCheck(cudaMemcpyAsync(a, halos+2*plane, plane*sizeof(Real), cudaMemcpyHostToDevice, transfer));
                if (upper != MPI_PROC_NULL)
                    cudaCheck(cudaMemcpyAsync(a+(localZ+1)*plane, halos+3*plane, plane*sizeof(Real), cudaMemcpyHostToDevice, transfer));
                cudaCheck(cudaEventRecord(ready, transfer));
                cudaCheck(cudaStreamWaitEvent(compute, ready, 0));
                if (begin == 1 && end > 1) launch(a, b, nx, ny, 1, 2, compute);
                if (localZ > 1 && end > localZ) launch(a, b, nx, ny, localZ, localZ+1, compute);
                cudaCheck(cudaEventRecord(done, compute));
            }
            std::swap(a, b);
        }
        cudaCheck(cudaStreamSynchronize(compute));
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!rank) {
            const double updates = double(nx > 2 ? nx-2 : 0) * double(ny > 2 ? ny-2 : 0) * double(nz > 2 ? nz-2 : 0) * double(iterations);
            printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", seconds*1000, seconds > 0 ? updates/seconds/1e6 : 0);
        }
        if (validate || printResults)
            cudaCheck(cudaMemcpy(host.data()+plane, a+plane, localZ*plane*sizeof(Real), cudaMemcpyDeviceToHost));
        if (printResults) {
            std::vector<Real> global;
            if (!rank) global.resize(nx*ny*nz);
            // Ordered, chunked collection retains the exact global output/hash
            // and avoids replicating the global grid in normal benchmark runs.
            for (int source = 0; source < activeSize; ++source) {
                const size_t depth = nz/activeSize + (size_t(source) < nz%activeSize);
                const size_t offsetZ = size_t(source)*(nz/activeSize) + std::min(size_t(source), nz%activeSize);
                if (source == 0 && rank == 0)
                    std::copy_n(host.data()+plane, depth*plane, global.data());
                else for (size_t p = 0; p < depth*plane;) {
                    const int n = int(std::min(depth*plane-p, size_t(INT_MAX)));
                    if (rank == source) mpiCheck(MPI_Send(host.data()+plane+p, n, MPI_DOUBLE, 0, 2, comm));
                    if (rank == 0) mpiCheck(MPI_Recv(global.data()+offsetZ*plane+p, n, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE));
                    p += size_t(n);
                }
            }
            if (!rank) print_results(global, "Grid");
        }
        if (validate) {
            Real low = std::numeric_limits<Real>::infinity(), high = -low;
            int bad = 0;
            #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(|:bad) schedule(static)
            for (size_t i = 0; i < localZ*plane; ++i) {
                const Real value = host[plane+i];
                low = std::min(low, value);
                high = std::max(high, value);
                bad |= !std::isfinite(value);
                const size_t z = startZ + i/plane, y = (i%plane)/nx, x = i%nx;
                if (!x || x+1 == nx || !y || y+1 == ny || !z || z+1 == nz)
                    bad |= value != Real((startZ*plane+i)%19);
            }
            Real globalLow, globalHigh;
            mpiCheck(MPI_Allreduce(&bad, &status, 1, MPI_INT, MPI_MAX, comm));
            mpiCheck(MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
            mpiCheck(MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            if (!rank) {
                status |= globalLow < -1e6 || globalHigh > 1e6;
                printf("Validating result...\nValue range: [%.6f, %.6f]\nValidation: %s\n", globalLow, globalHigh, status ? "FAILED" : "PASSED");
            }
        }
        cudaCheck(cudaEventDestroy(ready));
        cudaCheck(cudaEventDestroy(done));
        cudaCheck(cudaStreamDestroy(compute));
        cudaCheck(cudaStreamDestroy(transfer));
        if (halos) cudaCheck(cudaFreeHost(halos));
        cudaCheck(cudaFree(a));
        cudaCheck(cudaFree(b));
        mpiCheck(MPI_Comm_free(&comm));
    }
    int finalStatus;
    mpiCheck(MPI_Allreduce(&status, &finalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return finalStatus;
}
