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

static void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA(call) do { const cudaError_t error = (call); \
    if (error != cudaSuccess) fail(cudaGetErrorString(error)); } while (0)
#define MPI_CHECK(call) do { if ((call) != MPI_SUCCESS) fail("MPI operation failed"); } while (0)

// Each warp accesses contiguous X values. Rolling Z registers reuse two of
// the seven operands; short Z tiles retain enough blocks for GPU occupancy.
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t first, size_t last) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t begin = first + size_t(blockIdx.z) * 8;
    if (x >= nx - 1 || y >= ny - 1 || begin >= last) return;
    const size_t plane = nx * ny;
    size_t i = begin * plane + y * nx + x;
    Real below = input[i - plane], center = input[i];
    const size_t end = begin + 8 < last ? begin + 8 : last;
    for (size_t z = begin; z < end; ++z, i += plane) {
        const Real above = input[i + plane];
        output[i] = (center + input[i-1] + input[i+1] + input[i-nx]
                     + input[i+nx] + below + above) / 7.0;
        below = center;
        center = above;
    }
}

static void launch(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t first, size_t last, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || first >= last) return;
    // Split unusually long slabs to respect CUDA's grid-Z limit.
    for (size_t z = first; z < last; ) {
        const size_t end = std::min(last, z + size_t(65535) * 8);
        stencil<<<dim3((nx-2+31)/32, (ny-2+3)/4, (end-z+7)/8),
                  dim3(32,4), 0, stream>>>(input, output, nx, ny, z, end);
        CUDA(cudaGetLastError());
        z = end;
    }
}

static size_t number(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end || errno ||
        value > std::numeric_limits<size_t>::max()) fail("Invalid numeric argument");
    return size_t(value);
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i+1 < argc) nx = number(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i+1 < argc) ny = number(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i+1 < argc) nz = number(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i+1 < argc) {
            size_t n = number(argv[++i]);
            if (n > INT_MAX) fail("Too many iterations");
            iterations = int(n);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || nx > limit / ny || nx*ny > limit / nz)
        fail("Grid is empty or too large");
    const size_t plane = nx * ny;
    // Halo messages use MPI's portable int-count interface.
    if (plane > INT_MAX || (ny > 2 && (ny-2+3)/4 > 65535))
        fail("Grid cross-section exceeds MPI/CUDA limits");

    // Exclude empty slabs when there are more ranks than Z planes.
    MPI_Comm comm;
    const int active = int(std::min(nz, size_t(ranks)));
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm));
    if (rank >= active) {
        MPI_Finalize();
        return 0;
    }
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int localRank, devices;
    MPI_CHECK(MPI_Comm_rank(local, &localRank));
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA GPU is required on every participating node");
    CUDA(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&local));

    const size_t base = nz / active, extra = nz % active;
    const size_t depth = base + (size_t(rank) < extra);
    const size_t startZ = size_t(rank) * base + std::min(size_t(rank), extra);
    if (depth > limit/plane - 2) fail("Local slab is too large");
    const size_t count = (depth + 2) * plane, bytes = count * sizeof(Real);
    Real *host, *a, *b, *halo = nullptr;
    CUDA(cudaMallocHost(&host, bytes));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const size_t z = i / plane;
        // Exterior ghost planes are never read by a physical interior cell.
        host[i] = ((startZ + z == 0 || startZ + z > nz) ? 0.0 :
                   Real(((startZ + z - 1) * plane + i % plane) % 19));
    }
    CUDA(cudaMalloc(&a, bytes));
    CUDA(cudaMalloc(&b, bytes));
    CUDA(cudaMemcpy(a, host, bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b, host, bytes, cudaMemcpyHostToDevice));
    // Both buffers have identical fixed boundaries; kernels only write interiors.
    if (!validate && !printResults) {
        CUDA(cudaFreeHost(host));
        host = nullptr;
    }
    if (active > 1) CUDA(cudaMallocHost(&halo, 4 * plane * sizeof(Real)));
    cudaStream_t compute, transfer;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    const int lower = rank ? rank-1 : MPI_PROC_NULL;
    const int upper = rank+1 < active ? rank+1 : MPI_PROC_NULL;
    const size_t first = startZ == 0 ? 2 : 1;
    const size_t last = startZ + depth == nz ? depth : depth+1;
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_CHECK(MPI_Barrier(comm));
    double elapsed = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (active == 1 || nx < 3 || ny < 3 || nz < 3) {
            launch(a, b, nx, ny, first, last, compute);
        } else {
            MPI_Request requests[4];
            int nr = 0;
            if (lower != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(halo+2*plane, int(plane), MPI_DOUBLE, lower, 1, comm, &requests[nr++]));
                CUDA(cudaMemcpyAsync(halo, a+plane, plane*sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            if (upper != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(halo+3*plane, int(plane), MPI_DOUBLE, upper, 0, comm, &requests[nr++]));
                CUDA(cudaMemcpyAsync(halo+plane, a+depth*plane, plane*sizeof(Real), cudaMemcpyDeviceToHost, transfer));
            }
            launch(a, b, nx, ny, std::max(first, size_t(2)), std::min(last, depth), compute);
            CUDA(cudaStreamSynchronize(transfer));
            if (lower != MPI_PROC_NULL)
                MPI_CHECK(MPI_Isend(halo, int(plane), MPI_DOUBLE, lower, 0, comm, &requests[nr++]));
            if (upper != MPI_PROC_NULL)
                MPI_CHECK(MPI_Isend(halo+plane, int(plane), MPI_DOUBLE, upper, 1, comm, &requests[nr++]));
            // MPI progresses communication while the GPU computes the bulk.
            MPI_CHECK(MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE));
            if (lower != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(a, halo+2*plane, plane*sizeof(Real), cudaMemcpyHostToDevice, transfer));
            if (upper != MPI_PROC_NULL)
                CUDA(cudaMemcpyAsync(a+(depth+1)*plane, halo+3*plane, plane*sizeof(Real), cudaMemcpyHostToDevice, transfer));
            if (first <= 1 && 1 < last) launch(a, b, nx, ny, 1, 2, transfer);
            if (depth > 1 && first <= depth && depth < last) launch(a, b, nx, ny, depth, depth+1, transfer);
            CUDA(cudaStreamSynchronize(transfer));
            CUDA(cudaStreamSynchronize(compute));
        }
        std::swap(a, b);
    }
    CUDA(cudaStreamSynchronize(compute));
    elapsed = MPI_Wtime() - elapsed;
    double seconds;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        const double updates = double(nx > 2 ? nx-2 : 0) * double(ny > 2 ? ny-2 : 0)
                             * double(nz > 2 ? nz-2 : 0) * iterations;
        printf("Computation time: %lld ms\n", (long long)(seconds*1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates/seconds/1e6 : 0);
    }
    if (host) CUDA(cudaMemcpy(host, a+plane, depth*plane*sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Chunk transfers to avoid MPI_Gatherv's int displacement/size limits.
        std::vector<Real> result;
        if (!rank) result.resize(nx*ny*nz);
        for (int owner = 0; owner < active; ++owner) {
            const size_t offset = (size_t(owner)*base + std::min(size_t(owner), extra))*plane;
            const size_t length = (base + (size_t(owner) < extra))*plane;
            for (size_t i = 0; i < length; ) {
                const int n = int(std::min(length-i, size_t(INT_MAX)));
                if (!owner && !rank) std::copy(host+i, host+i+n, result.data()+offset+i);
                else if (rank == owner) MPI_CHECK(MPI_Send(host+i, n, MPI_DOUBLE, 0, 2, comm));
                else if (!rank) MPI_CHECK(MPI_Recv(result.data()+offset+i, n, MPI_DOUBLE, owner, 2, comm, MPI_STATUS_IGNORE));
                i += n;
            }
        }
        if (!rank) print_results(result, "Grid");
    }
    int valid = 1;
    if (validate) {
        Real minimum = std::numeric_limits<Real>::infinity(), maximum = -minimum;
        int bad = 0;
        #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(|:bad) schedule(static)
        for (size_t i = 0; i < depth*plane; ++i) {
            const Real v = host[i];
            bad |= !std::isfinite(v);
            minimum = std::min(minimum, v);
            maximum = std::max(maximum, v);
        }
        Real globalMin, globalMax;
        int globalBad;
        MPI_CHECK(MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, comm));
        MPI_CHECK(MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm));
        MPI_CHECK(MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm));
        valid = !globalBad && globalMin >= -1e6 && globalMax <= 1e6;
        if (!rank) printf("Validating result...\nValue range: [%.6f, %.6f]\nValidation: %s\n",
                          globalMin, globalMax, valid ? "PASSED" : "FAILED");
    }
    CUDA(cudaStreamDestroy(compute));
    CUDA(cudaStreamDestroy(transfer));
    CUDA(cudaFree(a));
    CUDA(cudaFree(b));
    if (halo) CUDA(cudaFreeHost(halo));
    if (host) CUDA(cudaFreeHost(host));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_Finalize();
    return valid ? 0 : 1;
}
