#include <mpi.h>
#if defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h>
#  endif
#endif
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

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int error = (call); \
    if (error != MPI_SUCCESS) { \
        char message[MPI_MAX_ERROR_STRING]; int length = 0; \
        MPI_Error_string(error, message, &length); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length, message); \
        MPI_Abort(MPI_COMM_WORLD, error); \
    } \
} while (0)

// Arrays contain one halo plane on either side of the rank's owned planes.
// The boundary kernel visits at most two planes; the interior kernel visits
// all remaining planes, so communication can overlap with useful GPU work.
template <bool Chemical, int Mode>
__global__ void stencil(const double* input, const double* cold, double* output,
                        size_t nx, size_t ny, size_t local_nz,
                        bool global_low, bool global_high) {
    const size_t plane = nx * ny;
    const size_t planes = Mode == 2 ? local_nz : (Mode == 1 ? (local_nz == 1 ? 1 : 2) : local_nz - 2);
    for (size_t p = blockIdx.z; p < planes; p += gridDim.z) {
      const size_t z = Mode == 2 ? p + 1 : (Mode == 1 ? (p == 0 ? 1 : local_nz) : p + 2);
      for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
           y < ny; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
       for (size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            x < nx; x += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t xy = y * nx + x;
        const size_t index = z * plane + xy;
        const double center = input[index];
        const size_t xm = x ? index - 1 : index;
        const size_t xp = x + 1 < nx ? index + 1 : index;
        const size_t ym = y ? index - nx : index;
        const size_t yp = y + 1 < ny ? index + nx : index;
        const size_t zm = (z == 1 && global_low) ? index : index - plane;
        const size_t zp = (z == local_nz && global_high) ? index : index + plane;
        const double cxx = input[xp] + input[xm] - 2.0 * center;
        const double cyy = input[yp] + input[ym] - 2.0 * center;
        const double czz = input[zp] + input[zm] - 2.0 * center;
        const double laplacian = cxx + cyy + czz;
        if (Chemical) {
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            output[index] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
                            + 3.0 * center + center * center * center - 0.5 * laplacian;
        } else {
            output[index] = cold[index] + 0.01 * laplacian;
        }
       }
      }
    }
}

static dim3 grid(size_t nx, size_t ny, size_t planes) {
    return dim3(static_cast<unsigned>(std::min<size_t>((nx + 31) / 32, 65535)),
                static_cast<unsigned>(std::min<size_t>((ny + 3) / 4, 65535)),
                static_cast<unsigned>(std::min<size_t>(planes, 65535)));
}

struct Slab {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    bool cuda_aware;
    MPI_Comm comm;
    cudaStream_t interior, boundary;
    double *send_low, *send_high, *recv_low, *recv_high;

    Slab(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x * y), rank(r), ranks(n), comm(c),
          send_low(nullptr), send_high(nullptr), recv_low(nullptr), recv_high(nullptr) {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        cuda_aware = MPIX_Query_cuda_support() != 0;
#else
        cuda_aware = false;
#endif
        CUDA_CHECK(cudaStreamCreateWithFlags(&interior, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&boundary, cudaStreamNonBlocking));
        if (ranks > 1 && !cuda_aware) {
            CUDA_CHECK(cudaMallocHost(&send_low, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&send_high, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_low, plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&recv_high, plane * sizeof(double)));
        }
    }
    ~Slab() {
        if (send_low) CUDA_CHECK(cudaFreeHost(send_low));
        if (send_high) CUDA_CHECK(cudaFreeHost(send_high));
        if (recv_low) CUDA_CHECK(cudaFreeHost(recv_low));
        if (recv_high) CUDA_CHECK(cudaFreeHost(recv_high));
        CUDA_CHECK(cudaStreamDestroy(interior));
        CUDA_CHECK(cudaStreamDestroy(boundary));
    }

    void post_plane(double* data, int peer, int tag, bool send, std::vector<MPI_Request>& requests) {
        for (size_t offset = 0; offset < plane;) {
            const int count = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
            requests.emplace_back();
            if (send) MPI_CHECK(MPI_Isend(data + offset, count, MPI_DOUBLE, peer, tag, comm, &requests.back()));
            else MPI_CHECK(MPI_Irecv(data + offset, count, MPI_DOUBLE, peer, tag, comm, &requests.back()));
            offset += count;
        }
    }

    template <bool Chemical>
    void apply(double* input, const double* cold, double* output) {
        // A single rank needs no halos, so both stages run in order on one stream.
        if (ranks == 1) {
            stencil<Chemical, 2><<<grid(nx, ny, nz), dim3(32, 4), 0, interior>>>(
                input, cold, output, nx, ny, nz, true, true);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        // Both streams from the preceding stage must finish before input is read.
        CUDA_CHECK(cudaStreamSynchronize(interior));
        CUDA_CHECK(cudaStreamSynchronize(boundary));
        std::vector<MPI_Request> requests;
        requests.reserve(4 * (1 + (plane - 1) / INT_MAX));
        if (ranks > 1) {
            if (rank > 0) {
                post_plane(cuda_aware ? input : recv_low, rank - 1, 1, false, requests);
                if (!cuda_aware) CUDA_CHECK(cudaMemcpyAsync(send_low, input + plane, plane * sizeof(double), cudaMemcpyDeviceToHost, boundary));
            }
            if (rank + 1 < ranks) {
                post_plane(cuda_aware ? input + (nz + 1) * plane : recv_high, rank + 1, 0, false, requests);
                if (!cuda_aware) CUDA_CHECK(cudaMemcpyAsync(send_high, input + nz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost, boundary));
            }
        }
        if (nz > 2) {
            stencil<Chemical, 0><<<grid(nx, ny, nz - 2), dim3(32, 4), 0, interior>>>(
                input, cold, output, nx, ny, nz, rank == 0, rank == ranks - 1);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks > 1) {
            CUDA_CHECK(cudaStreamSynchronize(boundary));
            if (rank > 0) post_plane(cuda_aware ? input + plane : send_low, rank - 1, 0, true, requests);
            if (rank + 1 < ranks) post_plane(cuda_aware ? input + nz * plane : send_high, rank + 1, 1, true, requests);
            MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            if (!cuda_aware && rank > 0) CUDA_CHECK(cudaMemcpyAsync(input, recv_low, plane * sizeof(double), cudaMemcpyHostToDevice, boundary));
            if (!cuda_aware && rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(input + (nz + 1) * plane, recv_high, plane * sizeof(double), cudaMemcpyHostToDevice, boundary));
        }
        stencil<Chemical, 1><<<grid(nx, ny, nz == 1 ? 1 : 2), dim3(32, 4), 0, boundary>>>(
            input, cold, output, nx, ny, nz, rank == 0, rank == ranks - 1);
        CUDA_CHECK(cudaGetLastError());
    }
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(interior));
        CUDA_CHECK(cudaStreamSynchronize(boundary));
    }
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

// Transfer arbitrarily large contiguous slabs without relying on MPI-4 large counts.
static void transfer(double* data, size_t n, int peer, int tag, MPI_Comm comm, bool send) {
    while (n) {
        const int chunk = static_cast<int>(std::min<size_t>(n, INT_MAX));
        if (send) MPI_CHECK(MPI_Send(data, chunk, MPI_DOUBLE, peer, tag, comm));
        else MPI_CHECK(MPI_Recv(data, chunk, MPI_DOUBLE, peer, tag, comm, MPI_STATUS_IGNORE));
        data += chunk;
        n -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / sizeof(double) ||
        nx * ny > SIZE_MAX / (nz + 2) || nx * ny * (nz + 2) > SIZE_MAX / sizeof(double)) {
        if (!world_rank) fprintf(stderr, "Invalid or unsupported grid size\n");
        MPI_Finalize(); return 1;
    }
    // The first min(nz, world_size) ranks own nonempty slabs.
    int active = static_cast<int>(std::min<size_t>(nz, world_size));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm));
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &ranks));
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    const size_t plane = nx * ny, gridSize = plane * nz;
    const size_t local_nz = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t start_z = static_cast<size_t>(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    const size_t local_size = plane * local_nz;
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(local_size);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_size; ++i) {
        const size_t linear_id = start_z * plane + i;
        const double pseudo = (((linear_id + 1) * static_cast<size_t>(1299709)) % gridSize) / static_cast<double>(gridSize);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    const size_t bytes = (local_nz + 2) * plane * sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), local_size * sizeof(double), cudaMemcpyHostToDevice));
    {
        Slab slab(nx, ny, local_nz, rank, ranks, comm);
        if (!rank) printf("Running Cahn-Hilliard simulation...\n");
        // Load the kernels and establish communication before timing; these writes
        // only touch scratch buffers and leave the initial concentration intact.
        if (iterations > 0) {
            slab.apply<true>(cold, cold, mu);
            slab.apply<false>(mu, cold, cnew);
            slab.synchronize();
        }
        MPI_CHECK(MPI_Barrier(comm));
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            slab.apply<true>(cold, cold, mu);
            slab.apply<false>(mu, cold, cnew);
            std::swap(cold, cnew);
        }
        slab.synchronize();
        const double local_time = MPI_Wtime() - start;
        double elapsed = 0;
        MPI_CHECK(MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!rank) {
            printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / elapsed / 1e6);
        }
    }
    int valid = 1;
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, local_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            if (!rank) {
                std::vector<double> result(gridSize);
                std::copy(host.begin(), host.end(), result.begin());
                for (int r = 1; r < ranks; ++r) {
                    const size_t start = static_cast<size_t>(r) * (nz / ranks) + std::min<size_t>(r, nz % ranks);
                    const size_t length = (nz / ranks + (static_cast<size_t>(r) < nz % ranks)) * plane;
                    transfer(result.data() + start * plane, length, r, 2, comm, false);
                }
                print_results(result, "Concentration");
            } else transfer(host.data(), local_size, 0, 2, comm, true);
        }
        if (validate) {
            double minVal = std::numeric_limits<double>::infinity();
            double maxVal = -std::numeric_limits<double>::infinity();
            int nonfinite = 0;
            #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:nonfinite) schedule(static)
            for (size_t i = 0; i < local_size; ++i) {
                const double value = host[i];
                if (!std::isfinite(value)) nonfinite = 1;
                else { minVal = std::min(minVal, value); maxVal = std::max(maxVal, value); }
            }
            double globalMin, globalMax;
            int globalNonfinite;
            MPI_CHECK(MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
            MPI_CHECK(MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            MPI_CHECK(MPI_Reduce(&nonfinite, &globalNonfinite, 1, MPI_INT, MPI_LOR, 0, comm));
            if (!rank) {
                printf("Validating result...\n");
                if (globalNonfinite) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                    if (globalMax > 10.0 || globalMin < -10.0) { printf("Validation failed: values out of expected range\n"); valid = 0; }
                }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, comm));
        }
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_CHECK(MPI_Comm_free(&comm));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
