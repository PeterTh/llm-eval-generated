#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <exception>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Owned planes are [1, localNz]; planes 0 and localNz+1 are MPI ghosts.
__device__ __forceinline__ double laplacian(const double* c, size_t i,
        size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t localNz,
        bool first, bool last) {
    const size_t plane = nx * ny;
    const double center = c[i];
    const double cxx = (c[x + 1 < nx ? i + 1 : i] + c[x ? i - 1 : i]
                        - 2.0 * center);
    const double cyy = (c[y + 1 < ny ? i + nx : i] + c[y ? i - nx : i]
                        - 2.0 * center);
    const double czz = (c[last && z == localNz ? i : i + plane]
                        + c[first && z == 1 ? i : i - plane] - 2.0 * center);
    return cxx + cyy + czz;
}

template<bool chemical, bool boundary>
__global__ void stencil(double* __restrict__ out, const double* __restrict__ in,
        const double* __restrict__ cold, size_t nx, size_t ny, size_t localNz,
        size_t begin, size_t planes, bool first, bool last) {
    const size_t plane = nx * ny;
    // Three-dimensional tiles avoid integer division/modulo in the stencil's
    // hot path. X is contiguous in memory and in the CUDA thread layout.
    for (size_t p = size_t(blockIdx.z) * blockDim.z + threadIdx.z; p < planes;
         p += size_t(blockDim.z) * gridDim.z) {
        const size_t z = boundary ? (p == 0 ? 1 : localNz) : begin + p;
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y; y < ny;
             y += size_t(blockDim.y) * gridDim.y) {
            for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < nx;
                 x += size_t(blockDim.x) * gridDim.x) {
                const size_t i = z * plane + y * nx + x;
                const double lap = laplacian(in, i, x, y, z, nx, ny, localNz, first, last);
                if (chemical) {
                    const double cv = in[i];
                    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
                    out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                             + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                } else {
                    out[i] = cold[i] + 0.01 * 1.0 * lap;
                }
            }
        }
    }
}

template<bool chemical, bool boundary = false>
static void launch(double* out, const double* in, const double* cold,
        size_t nx, size_t ny, size_t localNz, size_t begin, size_t planes,
        bool first, bool last, cudaStream_t stream) {
    if (planes) {
        const unsigned bx = static_cast<unsigned>(std::min(nx, size_t(32)));
        const unsigned by = static_cast<unsigned>(std::min(ny, size_t(8)));
        const unsigned bz = std::min(64u, 256 / (bx * by));
        const dim3 block(bx, by, bz);
        const dim3 grid(static_cast<unsigned>(std::min(size_t(65535), (nx + bx - 1) / bx)),
                        static_cast<unsigned>(std::min(size_t(65535), (ny + by - 1) / by)),
                        static_cast<unsigned>(std::min(size_t(65535), (planes + bz - 1) / bz)));
        stencil<chemical, boundary><<<grid, block, 0, stream>>>(out, in, cold,
            nx, ny, localNz, begin, planes, first, last);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Use device buffers directly when MPI advertises CUDA support, otherwise use
// portable pinned staging. MPI calls stay on the main thread.
struct HaloExchange {
    size_t plane, localNz;
    int lower, upper;
    MPI_Comm comm;
    double* host = nullptr;
    bool deviceMPI = false;
    cudaStream_t transfer;
    cudaEvent_t ready, arrived;

    HaloExchange(size_t p, size_t n, int rank, int ranks, MPI_Comm c)
        : plane(p), localNz(n), lower(rank ? rank - 1 : MPI_PROC_NULL),
          upper(rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL), comm(c) {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
        if (!deviceMPI) CUDA_CHECK(cudaMallocHost(&host, 4 * plane * sizeof(double)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&arrived, cudaEventDisableTiming));
    }

    void exchange(double* field, cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(transfer, ready, 0));
        if (!deviceMPI && lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(host, field + plane, plane * sizeof(double),
                                        cudaMemcpyDeviceToHost, transfer));
        if (!deviceMPI && upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(host + plane, field + localNz * plane,
                                        plane * sizeof(double), cudaMemcpyDeviceToHost, transfer));
        // Interior kernels are already running on the independent compute stream.
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        double* sendLower = deviceMPI ? field + plane : host;
        double* sendUpper = deviceMPI ? field + localNz * plane : host + plane;
        double* recvLower = deviceMPI ? field : host + 2 * plane;
        double* recvUpper = deviceMPI ? field + (localNz + 1) * plane : host + 3 * plane;
        // Chunk messages so very large XY planes do not overflow MPI's int count.
        for (size_t offset = 0; offset < plane; ) {
            const int count = static_cast<int>(std::min(plane - offset, size_t(INT_MAX)));
            MPI_Request requests[4];
            MPI_Irecv(recvLower + offset, count, MPI_DOUBLE, lower, 1, comm, &requests[0]);
            MPI_Irecv(recvUpper + offset, count, MPI_DOUBLE, upper, 0, comm, &requests[1]);
            MPI_Isend(sendLower + offset, count, MPI_DOUBLE, lower, 0, comm, &requests[2]);
            MPI_Isend(sendUpper + offset, count, MPI_DOUBLE, upper, 1, comm, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            offset += count;
        }
        if (!deviceMPI && lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field, host + 2 * plane, plane * sizeof(double),
                                        cudaMemcpyHostToDevice, transfer));
        if (!deviceMPI && upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field + (localNz + 1) * plane, host + 3 * plane,
                                        plane * sizeof(double), cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaEventRecord(arrived, transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, arrived, 0));
    }

    ~HaloExchange() {
        CUDA_CHECK(cudaStreamDestroy(transfer));
        CUDA_CHECK(cudaEventDestroy(ready));
        CUDA_CHECK(cudaEventDestroy(arrived));
        if (host) CUDA_CHECK(cudaFreeHost(host));
    }
};

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
    int invalid = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        invalid |= !std::isfinite(c[i]);
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    double globalMin, globalMax;
    int globalInvalid, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalInvalid) {
        if (!rank) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    minVal = globalMin;
    maxVal = globalMax;
    
    if (!rank) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        if (!rank) printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
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

// A rank owns a contiguous Z slab, so gathering in rank order preserves the
// original global array order, Kahan sum, samples, and hash exactly.
static void gatherResults(const std::vector<double>& local, size_t plane,
        size_t nz, int rank, int ranks) {
    const int active = static_cast<int>(std::min(nz, size_t(ranks)));
    std::vector<double> global;
    if (!rank) global.resize(plane * nz);
    for (int source = 0; source < active; ++source) {
        const size_t planes = nz / active + (size_t(source) < nz % active);
        const size_t start = size_t(source) * (nz / active) + std::min(size_t(source), nz % active);
        const size_t count = planes * plane;
        if (source == 0) {
            if (!rank) std::copy(local.begin(), local.end(), global.begin());
            continue;
        }
        for (size_t offset = 0; offset < count; ) {
            const int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
            if (rank == source)
                MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            if (!rank)
                MPI_Recv(global.data() + start * plane + offset, chunk, MPI_DOUBLE,
                         source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += chunk;
        }
    }
    if (!rank) print_results(global, "Concentration");
}

static std::vector<double> simulate(size_t nx, size_t ny, size_t nz, int iterations,
        bool copyResult, MPI_Comm comm, double& elapsed) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const size_t localNz = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t startZ = size_t(rank) * (nz / ranks) + std::min(size_t(rank), nz % ranks);
    const size_t plane = nx * ny, cells = plane * localNz;
    const size_t bytes = (localNz + 2) * plane * sizeof(double);

    // Node-local rank selects the GPU; this also works with one visible GPU per
    // process as commonly configured by accelerator-cluster schedulers.
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    std::vector<double> host(cells);
    const size_t volume = plane * nz;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cells; ++i) {
        const size_t linear_id = startZ * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % volume) / static_cast<double>(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    double *cold, *cnew, *mu;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&cnew, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    CUDA_CHECK(cudaMemcpy(cold + plane, host.data(), cells * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    {
        HaloExchange halo(plane, localNz, rank, ranks, comm);
        CUDA_CHECK(cudaEventRecord(halo.ready, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            if (ranks == 1) {
                launch<true>(mu, cold, cold, nx, ny, localNz, 1, localNz, true, true, compute);
                launch<false>(cnew, mu, cold, nx, ny, localNz, 1, localNz, true, true, compute);
            } else {
                const bool first = rank == 0, last = rank + 1 == ranks;
                const size_t interior = localNz > 2 ? localNz - 2 : 0;
                const size_t boundaries = std::min(localNz, size_t(2));
                launch<true>(mu, cold, cold, nx, ny, localNz, 2, interior, first, last, compute);
                halo.exchange(cold, compute);
                launch<true, true>(mu, cold, cold, nx, ny, localNz, 1, boundaries, first, last, compute);
                CUDA_CHECK(cudaEventRecord(halo.ready, compute));
                launch<false>(cnew, mu, cold, nx, ny, localNz, 2, interior, first, last, compute);
                halo.exchange(mu, compute);
                launch<false, true>(cnew, mu, cold, nx, ny, localNz, 1, boundaries, first, last, compute);
                CUDA_CHECK(cudaEventRecord(halo.ready, compute));
            }
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        elapsed = MPI_Wtime() - start;
        if (copyResult)
            CUDA_CHECK(cudaMemcpy(host.data(), cold + plane, cells * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    if (!copyResult) host.clear();
    return host;
}

static int runBenchmark(int argc, char** argv, int rank, int ranks) {
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
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || nx > limit / ny || nx * ny > limit / nz ||
        nx * ny > limit / 4 || nz > limit / (nx * ny) - 2 || iterations < 0) {
        if (!rank) fprintf(stderr, "Invalid grid size or iteration count\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }
    // Ranks beyond the number of Z planes have no cells but participate in
    // world-wide validation and timing, including very thin grids.
    const int active = static_cast<int>(std::min(nz, size_t(ranks)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    double localElapsed = 0.0, elapsed;
    std::vector<double> cold;
    if (comm != MPI_COMM_NULL) {
        cold = simulate(nx, ny, nz, iterations, validate || printResults, comm, localElapsed);
        MPI_Comm_free(&comm);
    }
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double mcups = elapsed > 0.0 ? double(gridSize) * iterations / elapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        gatherResults(cold, nx * ny, nz, rank, ranks);
    }
    
    // Validation
    if (validate) {
        if (!rank) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
        if (valid) {
            if (!rank) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (!rank) printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
