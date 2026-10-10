#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

namespace {
constexpr int tile = 256;
int rankId = 0;

[[noreturn]] void fail(const char* message, int code = 1) {
    fprintf(stderr, "Rank %d: %s (code %d)\n", rankId, message, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}
void check(cudaError_t code) {
    if (code != cudaSuccess) fail(cudaGetErrorString(code), static_cast<int>(code));
}
void check(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS) fail("cuBLAS operation failed", static_cast<int>(code));
}
void check(cusolverStatus_t code) {
    if (code != CUSOLVER_STATUS_SUCCESS) fail("cuSOLVER operation failed", static_cast<int>(code));
}
void mpi(int code) {
    if (code != MPI_SUCCESS) fail("MPI operation failed", code);
}

struct DeviceBuffer {
    double* p = nullptr;
    explicit DeviceBuffer(size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~DeviceBuffer() { cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct HostBuffer {
    double* p = nullptr;
    explicit HostBuffer(size_t count) {
        check(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~HostBuffer() { cudaFreeHost(p); }
};
struct Accelerator {
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    Accelerator() {
        MPI_Comm local;
        mpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &local));
        int localRank = 0, devices = 0;
        mpi(MPI_Comm_rank(local, &localRank));
        mpi(MPI_Comm_free(&local));
        check(cudaGetDeviceCount(&devices));
        if (!devices) fail("The hybrid benchmark requires a CUDA GPU on every rank");
        check(cudaSetDevice(localRank % devices));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cublasCreate(&blas));
        check(cusolverDnCreate(&solver));
        check(cublasSetStream(blas, stream));
        check(cusolverDnSetStream(solver, stream));
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    }
    void sync() { check(cudaStreamSynchronize(stream)); }
    ~Accelerator() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
};

// Tile rows are cyclically assigned to ranks. Each local matrix is column-major
// with all n columns; global row indices remain sorted within each rank.
struct Distribution {
    int n, ranks, rows, ld;
    std::vector<int> globalRows;
    Distribution(int size, int processes) : n(size), ranks(processes) {
        for (int start = rankId * tile; start < n; start += ranks * tile)
            for (int i = start; i < std::min(n, start + tile); ++i) globalRows.push_back(i);
        rows = static_cast<int>(globalRows.size());
        ld = std::max(1, rows);
    }
    int first(int global) const {
        return static_cast<int>(std::lower_bound(globalRows.begin(), globalRows.end(), global) - globalRows.begin());
    }
    int owner(int global) const { return (global / tile) % ranks; }
    int rowCount(int r, int begin) const {
        int count = 0;
        for (int start = r * tile; start < n; start += ranks * tile)
            count += std::max(0, std::min(n, start + tile) - std::max(begin, start));
        return count;
    }
};

__global__ void addDiagonal(double* a, int ld, int n, int rows, const int* globals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) a[i + size_t(globals[i]) * ld] += n;
}
__global__ void zeroUpper(double* a, int ld, int n, int rows, const int* globals) {
    size_t total = size_t(rows) * n;
    for (size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < total;
         x += size_t(blockDim.x) * gridDim.x) {
        int i = x % rows, j = x / rows;
        if (j > globals[i]) a[i + size_t(j) * ld] = 0.0;
    }
}

// glibc rand_r advances its 32-bit LCG three times per output. Jump to each
// row's seed so OpenMP generates exactly the original seed-42 random sequence.
// Other libc implementations use the portable serial seed checkpoint path.
std::vector<unsigned int> rowSeeds(int n) {
    std::vector<unsigned int> seeds(n);
#if defined(__GLIBC__)
    uint32_t multiplier = 1, increment = 0, a = 1103515245u, c = 12345u;
    uint64_t steps = uint64_t(n) * 3;
    while (steps) {
        if (steps & 1) { increment = a * increment + c; multiplier *= a; }
        c *= a + 1u;
        a *= a;
        steps >>= 1;
    }
    uint32_t seed = 42;
    for (int i = 0; i < n; ++i) { seeds[i] = seed; seed = multiplier * seed + increment; }
#else
    unsigned int seed = 42;
    for (int i = 0; i < n; ++i) {
        seeds[i] = seed;
        for (int j = 0; j < n; ++j) (void)rand_r(&seed);
    }
#endif
    return seeds;
}

void generatePositiveDefiniteMatrix(DeviceBuffer& matrix, const Distribution& d,
                                   Accelerator& gpu, const int* globalDevice) {
    const auto seeds = rowSeeds(d.n);
    DeviceBuffer localB(size_t(d.ld) * d.n);
    HostBuffer hostLocal(size_t(d.ld) * d.n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d.rows; ++i) {
        unsigned int seed = seeds[d.globalRows[i]];
        for (int k = 0; k < d.n; ++k)
            hostLocal.p[i + size_t(k) * d.ld] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    check(cudaMemcpyAsync(localB.p, hostLocal.p, size_t(d.ld) * d.n * sizeof(double),
                          cudaMemcpyHostToDevice, gpu.stream));
    // Stream B in bounded chunks instead of replicating an n-by-n matrix.
    const int chunk = std::min(d.n, 4 * tile);
    HostBuffer hostBlock(size_t(d.n) * chunk);
    DeviceBuffer block(size_t(d.n) * chunk);
    const double one = 1, zero = 0;
    for (int start = 0; start < d.n; start += chunk) {
        int width = std::min(chunk, d.n - start);
        gpu.sync(); // The pinned buffer must not be reused during a DMA transfer.
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < width; ++j) {
            unsigned int seed = seeds[start + j];
            for (int k = 0; k < d.n; ++k)
                hostBlock.p[k + size_t(j) * d.n] = rand_r(&seed) / double(RAND_MAX) - 0.5;
        }
        check(cudaMemcpyAsync(block.p, hostBlock.p, size_t(d.n) * width * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.stream));
        if (d.rows) check(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_N,
            d.rows, width, d.n, &one, localB.p, d.ld, block.p, d.n, &zero,
            matrix.p + size_t(start) * d.ld, d.ld));
    }
    if (d.rows) addDiagonal<<<(d.rows + 255) / 256, 256, 0, gpu.stream>>>(
        matrix.p, d.ld, d.n, d.rows, globalDevice);
    check(cudaGetLastError());
    gpu.sync();
}

// Allocate communication buffers and solver scratch before benchmark timing.
struct CholeskyWorkspace {
    DeviceBuffer diagonal, panel;
    HostBuffer hostDiagonal, localPanel, gathered, hostPanel;
    int workspaceSize;
    DeviceBuffer workspace, info;
    std::vector<int> counts, offsets;
    static int query(Accelerator& gpu, double* diagonal, int size) {
        int count = 0;
        check(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER,
                                        size, diagonal, size, &count));
        return count;
    }
    CholeskyWorkspace(const Distribution& d, Accelerator& gpu)
        : diagonal(size_t(std::min(tile, d.n)) * std::min(tile, d.n)),
          panel(size_t(d.n) * std::min(tile, d.n)),
          hostDiagonal(size_t(std::min(tile, d.n)) * std::min(tile, d.n)),
          localPanel(size_t(d.ld) * std::min(tile, d.n)),
          gathered(size_t(d.n) * std::min(tile, d.n)),
          hostPanel(size_t(d.n) * std::min(tile, d.n)),
          workspaceSize(query(gpu, diagonal.p, std::min(tile, d.n))),
          workspace(workspaceSize), info(1), counts(d.ranks), offsets(d.ranks) {}
};

bool choleskyDecomposition(DeviceBuffer& matrix, const Distribution& d,
                           Accelerator& gpu, const int* globalDevice, CholeskyWorkspace& scratch) {
    auto& diagonal = scratch.diagonal;
    auto& panel = scratch.panel;
    auto& hostDiagonal = scratch.hostDiagonal;
    auto& localPanel = scratch.localPanel;
    auto& gathered = scratch.gathered;
    auto& hostPanel = scratch.hostPanel;
    auto& workspace = scratch.workspace;
    auto& counts = scratch.counts;
    auto& offsets = scratch.offsets;
    int workspaceSize = scratch.workspaceSize;
    int* deviceInfo = reinterpret_cast<int*>(scratch.info.p);
    const double one = 1, minusOne = -1;
    bool success = true;
    for (int k = 0; k < d.n; k += tile) {
        int width = std::min(tile, d.n - k), end = k + width;
        int owner = d.owner(k), info = 0;
        if (rankId == owner) {
            double* diag = matrix.p + d.first(k) + size_t(k) * d.ld;
            check(cudaMemcpy2DAsync(diagonal.p, width * sizeof(double), diag,
                d.ld * sizeof(double), width * sizeof(double), width,
                cudaMemcpyDeviceToDevice, gpu.stream));
            check(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, width,
                diagonal.p, width, workspace.p, workspaceSize, deviceInfo));
            check(cudaMemcpy2DAsync(diag, d.ld * sizeof(double), diagonal.p,
                width * sizeof(double), width * sizeof(double), width,
                cudaMemcpyDeviceToDevice, gpu.stream));
            check(cudaMemcpyAsync(&info, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost, gpu.stream));
            check(cudaMemcpy2DAsync(hostDiagonal.p, width * sizeof(double), diag,
                d.ld * sizeof(double), width * sizeof(double), width,
                cudaMemcpyDeviceToHost, gpu.stream));
            gpu.sync();
        }
        mpi(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info < 0) fail("Invalid cuSOLVER factorization argument", info);
        if (info) {
            if (!rankId) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
            success = false;
            break;
        }
        if (end == d.n) break;
        mpi(MPI_Bcast(hostDiagonal.p, width * width, MPI_DOUBLE, owner, MPI_COMM_WORLD));
        check(cudaMemcpyAsync(diagonal.p, hostDiagonal.p, size_t(width) * width * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.stream));
        int first = d.first(end), remaining = d.rows - first;
        if (remaining) {
            double* local = matrix.p + first + size_t(k) * d.ld;
            check(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, remaining, width, &one,
                diagonal.p, width, local, d.ld));
            check(cudaMemcpy2DAsync(localPanel.p, remaining * sizeof(double), local,
                d.ld * sizeof(double), remaining * sizeof(double), width,
                cudaMemcpyDeviceToHost, gpu.stream));
        }
        gpu.sync();
        int total = 0;
        for (int r = 0; r < d.ranks; ++r) {
            offsets[r] = total;
            counts[r] = d.rowCount(r, end) * width;
            total += counts[r];
        }
        // Host staging works with ordinary MPI; CUDA-aware MPI is not required.
        mpi(MPI_Allgatherv(localPanel.p, remaining * width, MPI_DOUBLE,
            gathered.p, counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD));
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < d.ranks; ++r) {
            for (int j = 0; j < width; ++j) {
                int localRow = 0, rankRows = counts[r] / width;
                for (int start = r * tile; start < d.n; start += d.ranks * tile)
                    for (int i = std::max(start, end); i < std::min(d.n, start + tile); ++i, ++localRow)
                        hostPanel.p[i - end + size_t(j) * (d.n - end)] =
                            gathered.p[offsets[r] + localRow + size_t(j) * rankRows];
            }
        }
        check(cudaMemcpyAsync(panel.p, hostPanel.p, size_t(d.n - end) * width * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.stream));
        // A_local -= L_local,panel * L_global,panel^T. Updating the full local
        // rectangle allows one large BLAS-3 operation per rank and panel.
        if (remaining) check(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T,
            remaining, d.n - end, width, &minusOne,
            matrix.p + first + size_t(k) * d.ld, d.ld, panel.p, d.n - end,
            &one, matrix.p + first + size_t(end) * d.ld, d.ld));
        // Keep updates queued: on other ranks they can overlap the next
        // owner's diagonal factorization and broadcast. Stream ordering and
        // the next panel's DMA synchronization protect all buffer reuse.
    }
    if (success && d.rows) zeroUpper<<<static_cast<int>(std::min(size_t(65535), (size_t(d.rows) * d.n + 255) / 256)),
        256, 0, gpu.stream>>>(matrix.p, d.ld, d.n, d.rows, globalDevice);
    check(cudaGetLastError());
    gpu.sync();
    return success;
}

bool validateCholesky(const DeviceBuffer& matrix, const std::vector<double>& original,
                      const Distribution& d, Accelerator& gpu) {
    HostBuffer hostBlock(size_t(d.n) * std::min(tile, d.n));
    HostBuffer reconstructed(size_t(d.n) * std::min(tile, d.n));
    DeviceBuffer block(size_t(d.n) * std::min(tile, d.n));
    DeviceBuffer product(size_t(d.ld) * std::min(tile, d.n));
    const double one = 1, zero = 0;
    double maxError = 0, relError = 0;
    for (int k = 0; k < d.n; k += tile) {
        int width = std::min(tile, d.n - k), owner = d.owner(k);
        if (rankId == owner) {
            // Pack rows of L as columns of L^T for the streamed product.
            check(cudaMemcpy2DAsync(reconstructed.p, width * sizeof(double), matrix.p + d.first(k),
                d.ld * sizeof(double), width * sizeof(double), d.n,
                cudaMemcpyDeviceToHost, gpu.stream));
            gpu.sync();
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < width; ++j)
                for (int i = 0; i < d.n; ++i)
                    hostBlock.p[i + size_t(j) * d.n] = reconstructed.p[j + size_t(i) * width];
        }
        mpi(MPI_Bcast(hostBlock.p, d.n * width, MPI_DOUBLE, owner, MPI_COMM_WORLD));
        check(cudaMemcpyAsync(block.p, hostBlock.p, size_t(d.n) * width * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.stream));
        if (d.rows) {
            check(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_N, d.rows, width, d.n,
                &one, matrix.p, d.ld, block.p, d.n, &zero, product.p, d.ld));
            check(cudaMemcpyAsync(reconstructed.p, product.p, size_t(d.ld) * width * sizeof(double),
                                  cudaMemcpyDeviceToHost, gpu.stream));
        }
        gpu.sync();
        #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
        for (size_t x = 0; x < size_t(d.rows) * width; ++x) {
            int i = x % d.rows, j = x / d.rows;
            double orig = original[i + size_t(k + j) * d.ld];
            double error = std::abs(reconstructed.p[i + size_t(j) * d.ld] - orig);
            if (!std::isfinite(error)) { maxError = relError = std::numeric_limits<double>::infinity(); }
            else {
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (std::abs(orig) + 1e-10));
            }
        }
    }
    double errors[2] = {maxError, relError};
    mpi(MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (!rankId) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void printDistributedResults(const DeviceBuffer& matrix, const Distribution& d, Accelerator& gpu) {
    // Gather only on request, streaming tile rows to avoid MPI's INT_MAX count
    // limit and a second full matrix on the root.
    std::vector<double> result;
    if (!rankId) result.resize(size_t(d.n) * d.n);
    HostBuffer packed(size_t(d.n) * std::min(tile, d.n));
    for (int k = 0; k < d.n; k += tile) {
        int width = std::min(tile, d.n - k), owner = d.owner(k);
        int count = d.n * width;
        if (rankId == owner) {
            check(cudaMemcpy2DAsync(packed.p, width * sizeof(double), matrix.p + d.first(k),
                d.ld * sizeof(double), width * sizeof(double), d.n,
                cudaMemcpyDeviceToHost, gpu.stream));
            gpu.sync();
            if (owner) mpi(MPI_Send(packed.p, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
        }
        if (!rankId) {
            if (owner) mpi(MPI_Recv(packed.p, count, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < width; ++i)
                for (int j = 0; j < d.n; ++j)
                    result[size_t(k + i) * d.n + j] = packed.p[i + size_t(j) * width];
        }
    }
    if (!rankId) print_results(result, "CholeskyL");
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int ranks) {
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            // All collectives operate on at most n * tile doubles.
            if (errno || *end || value <= 0 || value > INT_MAX / tile) {
                if (!rankId) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rankId) printUsage(argv[0]);
            return 0;
        } else {
            if (!rankId) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    // Respect OMP_NUM_THREADS. Otherwise avoid oversubscribing a node with one
    // full OpenMP team per MPI rank.
    if (!getenv("OMP_NUM_THREADS")) {
        MPI_Comm local;
        mpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &local));
        int localSize;
        mpi(MPI_Comm_size(local, &localSize));
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
        mpi(MPI_Comm_free(&local));
    }
    Accelerator gpu;
    Distribution d(n, ranks);
    DeviceBuffer matrix(size_t(d.ld) * n);
    int* globalDevice = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&globalDevice), size_t(d.ld) * sizeof(int)));
    if (d.rows) check(cudaMemcpy(globalDevice, d.globalRows.data(), size_t(d.rows) * sizeof(int), cudaMemcpyHostToDevice));
    if (!rankId) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", ranks, omp_get_max_threads());
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(matrix, d, gpu, globalDevice);
    std::vector<double> original;
    if (validate) {
        original.resize(size_t(d.ld) * n);
        check(cudaMemcpy(original.data(), matrix.p, original.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CholeskyWorkspace scratch(d, gpu);
    if (!rankId) printf("Computing Cholesky decomposition...\n");
    mpi(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(matrix, d, gpu, globalDevice, scratch);
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    mpi(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    check(cudaFree(globalDevice));
    if (!success) {
        if (!rankId) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (!rankId) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000);
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0 * maxElapsed * 1e9));
    }
    if (results) printDistributedResults(matrix, d, gpu);
    if (validate) {
        if (!rankId) printf("Validating result...\n");
        bool valid = validateCholesky(matrix, original, d, gpu);
        if (!rankId) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    mpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int ranks;
    mpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result = 1;
    try { result = run(argc, argv, ranks); }
    catch (const std::exception& error) { fail(error.what()); }
    mpi(MPI_Finalize());
    return result;
}
