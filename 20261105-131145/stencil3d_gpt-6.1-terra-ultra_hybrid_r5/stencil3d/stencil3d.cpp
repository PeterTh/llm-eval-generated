#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int error_ = (call); \
    if (error_ != MPI_SUCCESS) { \
        char message_[MPI_MAX_ERROR_STRING]; int length_; \
        MPI_Error_string(error_, message_, &length_); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length_, message_); \
        MPI_Abort(MPI_COMM_WORLD, error_); \
    } \
} while (0)

// Each block covers a tile in one plane. The z loop also handles grids deeper
// than CUDA's maximum grid.z, without introducing divisions in the inner loop.
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t first, size_t count,
                        size_t second, bool edges) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    const size_t plane = nx * ny;
    for (size_t k = blockIdx.z; k < count; k += gridDim.z) {
        const size_t z = edges && k == 1 ? second : first + k;
        const size_t i = z * plane + y * nx + x;
        output[i] = (input[i] + input[i-1] + input[i+1] + input[i-nx] +
                     input[i+nx] + input[i-plane] + input[i+plane]) / 7.0;
    }
}

static void launch_stencil(const Real* in, Real* out, size_t nx, size_t ny,
                           size_t first, size_t count, size_t second,
                           bool edges, cudaStream_t stream) {
    if (!count) return;
    dim3 block(32, 4);
    dim3 grid((nx - 2 + block.x - 1) / block.x,
              (ny - 2 + block.y - 1) / block.y,
              std::min(count, size_t(65535)));
    stencil<<<grid, block, 0, stream>>>(in, out, nx, ny, first, count, second, edges);
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parse_size(const char* text, size_t& value) {
    if (!*text || *text == '-') return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long n = strtoull(text, &end, 10);
    if (errno || *end || n > std::numeric_limits<size_t>::max()) return false;
    value = size_t(n);
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false, help = false, valid_args = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
            !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) {
            const char option = argv[i][1];
            size_t value = 0;
            if (++i >= argc || !parse_size(argv[i], value)) { valid_args = false; break; }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterations = value;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { valid_args = false; break; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(Real) || nx * ny > INT_MAX ||
        nx * ny > SIZE_MAX / sizeof(Real) / 4 ||
        (nx > 2 && (nx - 2 + 31) / 32 > INT_MAX) ||
        (ny > 2 && (ny - 2 + 3) / 4 > 65535)) valid_args = false;
    if (help || !valid_args) {
        if (!world_rank) {
            if (!valid_args) fprintf(stderr, "Invalid arguments or grid size.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return valid_args ? 0 : 1;
    }

    // Bind ranks using their node-local rank, respecting CUDA_VISIBLE_DEVICES.
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                  MPI_INFO_NULL, &local_comm));
    int local_rank, devices;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available on rank %d\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    const int active_size = int(std::min(nz, size_t(world_size)));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                             world_rank, &comm));
    if (world_rank >= active_size) { MPI_Finalize(); return 0; }
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &size));
    const size_t local_nz = nz / size + (size_t(rank) < nz % size);
    const size_t z0 = size_t(rank) * (nz / size) + std::min(size_t(rank), nz % size);
    const size_t plane = nx * ny, local_cells = plane * local_nz;
    if (local_nz > SIZE_MAX / plane / sizeof(Real) - 2) {
        fprintf(stderr, "Local grid including halos is too large on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int below = rank ? rank - 1 : MPI_PROC_NULL;
    const int above = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    const bool compute = nx > 2 && ny > 2 && nz > 2 && iterations > 0;
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %zu\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    std::vector<Real> host(local_cells);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_cells; ++i) host[i] = Real((z0 * plane + i) % 19);
    Real *a, *b;
    CUDA_CHECK(cudaMalloc(&a, (local_nz + 2) * plane * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&b, (local_nz + 2) * plane * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(a + plane, host.data(), local_cells * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b + plane, host.data(), local_cells * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t interior_stream, halo_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&halo_stream, cudaStreamNonBlocking));
    Real* staging = nullptr;
    if (compute && size > 1) CUDA_CHECK(cudaMallocHost(&staging, 4 * plane * sizeof(Real)));
    Real *send_below = staging, *send_above = staging ? staging + plane : nullptr;
    Real *recv_below = staging ? staging + 2 * plane : nullptr;
    Real *recv_above = staging ? staging + 3 * plane : nullptr;

    if (!rank) printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    for (size_t iter = 0; iter < iterations && compute; ++iter) {
        if (size == 1) {
            launch_stencil(a, b, nx, ny, 2, nz - 2, 0, false, interior_stream);
            std::swap(a, b);
            continue;
        }
        MPI_Request requests[4];
        int nrequests = 0;
        if (below != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_below, int(plane), MPI_DOUBLE, below, 1, comm, &requests[nrequests++]));
            CUDA_CHECK(cudaMemcpyAsync(send_below, a + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, halo_stream));
        }
        if (above != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_above, int(plane), MPI_DOUBLE, above, 0, comm, &requests[nrequests++]));
            CUDA_CHECK(cudaMemcpyAsync(send_above, a + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, halo_stream));
        }
        // Owned planes strictly inside the slab never depend on incoming halos.
        const size_t lo = std::max(size_t(2), size_t(1) + (z0 == 0));
        const size_t hi = std::min(local_nz > 0 ? local_nz - 1 : 0,
                                   local_nz - (z0 + local_nz == nz));
        if (lo <= hi) launch_stencil(a, b, nx, ny, lo, hi - lo + 1, 0, false, interior_stream);
        if (size > 1) {
            CUDA_CHECK(cudaStreamSynchronize(halo_stream));
            if (below != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(send_below, int(plane), MPI_DOUBLE, below, 0, comm, &requests[nrequests++]));
            if (above != MPI_PROC_NULL) MPI_CHECK(MPI_Isend(send_above, int(plane), MPI_DOUBLE, above, 1, comm, &requests[nrequests++]));
            MPI_CHECK(MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE));
            if (below != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(a, recv_below, plane * sizeof(Real), cudaMemcpyHostToDevice, halo_stream));
            if (above != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(a + (local_nz + 1) * plane, recv_above, plane * sizeof(Real), cudaMemcpyHostToDevice, halo_stream));
        }
        const bool first_valid = z0 > 0 && z0 < nz - 1;
        const bool last_valid = local_nz > 1 && z0 + local_nz - 1 > 0 && z0 + local_nz - 1 < nz - 1;
        if (first_valid || last_valid) {
            const size_t first = first_valid ? 1 : local_nz;
            launch_stencil(a, b, nx, ny, first, size_t(first_valid) + size_t(last_valid), local_nz, true, halo_stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(interior_stream));
        CUDA_CHECK(cudaStreamSynchronize(halo_stream));
        std::swap(a, b);
    }
    if (compute && size == 1) CUDA_CHECK(cudaStreamSynchronize(interior_stream));
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (!rank) {
        printf("Computation time: %ld ms\n", long(max_elapsed * 1000.0));
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * double(iterations);
        printf("Performance: %.3f MCellUpdates/s\n", updates / max_elapsed / 1e6);
    }
    if (validate || printResults) CUDA_CHECK(cudaMemcpy(host.data(), a + plane, local_cells * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Use chunks to avoid MPI's int count limit while preserving global order.
        std::vector<Real> result;
        if (!rank) {
            result.resize(nx * ny * nz);
            std::copy(host.begin(), host.end(), result.begin());
            for (int src = 1; src < size; ++src) {
                const size_t count = (nz / size + (size_t(src) < nz % size)) * plane;
                const size_t offset = (size_t(src) * (nz / size) + std::min(size_t(src), nz % size)) * plane;
                for (size_t pos = 0; pos < count;) {
                    int chunk = int(std::min(count - pos, size_t(INT_MAX)));
                    MPI_CHECK(MPI_Recv(result.data() + offset + pos, chunk, MPI_DOUBLE, src, 2, comm, MPI_STATUS_IGNORE));
                    pos += chunk;
                }
            }
            print_results(result, "Grid");
        } else {
            for (size_t pos = 0; pos < local_cells;) {
                int chunk = int(std::min(local_cells - pos, size_t(INT_MAX)));
                MPI_CHECK(MPI_Send(host.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm));
                pos += chunk;
            }
        }
    }
    int all_valid = 1;
    if (validate) {
        int finite = 1;
        Real min_value = std::numeric_limits<Real>::infinity();
        Real max_value = -std::numeric_limits<Real>::infinity();
        #pragma omp parallel for reduction(&:finite) reduction(min:min_value) reduction(max:max_value) schedule(static)
        for (size_t i = 0; i < local_cells; ++i) {
            finite &= std::isfinite(host[i]);
            min_value = std::min(min_value, host[i]);
            max_value = std::max(max_value, host[i]);
        }
        Real global_min, global_max;
        MPI_CHECK(MPI_Reduce(&finite, &all_valid, 1, MPI_INT, MPI_BAND, 0, comm));
        MPI_CHECK(MPI_Reduce(&min_value, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
        MPI_CHECK(MPI_Reduce(&max_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!rank) {
            printf("Validating result...\n");
            if (!all_valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 1e6 || global_min < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    all_valid = 0;
                }
            }
            printf("Validation: %s\n", all_valid ? "PASSED" : "FAILED");
        }
        MPI_CHECK(MPI_Bcast(&all_valid, 1, MPI_INT, 0, comm));
    }
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaStreamDestroy(interior_stream));
    CUDA_CHECK(cudaStreamDestroy(halo_stream));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return all_valid ? 0 : 1;
}
