#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// All MPI calls are made by the main thread. Each node-local rank selects a
// GPU; launch one rank per GPU and give each rank its own OpenMP CPU cores.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
struct DeviceBuffer {
    double* p = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~DeviceBuffer() { cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct PinnedBuffer {
    double* p = nullptr;
    explicit PinnedBuffer(size_t count) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~PinnedBuffer() { cudaFreeHost(p); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};
struct Accelerator {
    cudaStream_t stream;
    cublasHandle_t blas;
    Accelerator() {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blasCheck(cublasCreate(&blas));
        blasCheck(cublasSetStream(blas, stream));
        // Keep all arithmetic in FP64, including on GPUs with tensor cores.
        blasCheck(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    }
    ~Accelerator() { cublasDestroy(blas); cudaStreamDestroy(stream); }
    void sync() { cudaCheck(cudaStreamSynchronize(stream)); }
};

// Block-cyclic rows keep the trailing work balanced as the factorization
// progresses. Local storage is column-major for direct use by cuBLAS.
struct Rows {
    int n, block, rank, ranks;
    std::vector<int> global;
    std::vector<std::vector<int>> byRank;
    Rows(int size, int tile, int r, int p) : n(size), block(tile), rank(r), ranks(p), byRank(p) {
        for (int i = 0; i < n; ++i) byRank[(i / block) % ranks].push_back(i);
        global = byRank[rank];
    }
    int start(int row) const {
        return static_cast<int>(std::lower_bound(global.begin(), global.end(), row) - global.begin());
    }
    int count() const { return static_cast<int>(global.size()); }
};

static void generatePositiveDefiniteMatrix(DeviceBuffer& A, const Rows& rows,
                                           Accelerator& gpu, std::vector<double>& original,
                                           bool validate) {
    const int n = rows.n, m = rows.count(), ld = std::max(1, m);
    std::vector<double> B(size_t(n) * n);
    // Preserve the original rand_r stream exactly, independent of MPI/OMP size.
    unsigned int seed = 42;
    for (double& x : B) x = rand_r(&seed) / double(RAND_MAX) - 0.5;
    std::vector<double> local(size_t(m) * n);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            local[size_t(j) * m + i] = B[size_t(rows.global[i]) * n + j];
    {
        const int slab = std::min(n, 1024);
        DeviceBuffer fullB(size_t(n) * slab), localB(local.size());
        if (m) {
            cudaCheck(cudaMemcpyAsync(localB.p, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
            const double one = 1.0, zero = 0.0;
            // Row-major B is the column-major representation of B^T.
            // Stream its output-column slabs so no GPU stores a full n*n B.
            for (int col = 0; col < n; col += slab) {
                const int width = std::min(slab, n - col);
                cudaCheck(cudaMemcpyAsync(fullB.p, B.data() + size_t(col) * n,
                    size_t(width) * n * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
                blasCheck(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, width, n, &one, localB.p, ld, fullB.p, n,
                    &zero, A.p + size_t(col) * ld, ld));
            }
            cudaCheck(cudaMemcpyAsync(local.data(), A.p, local.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
        }
        gpu.sync();
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) local[size_t(rows.global[i]) * m + i] += n;
    if (validate) original = local;
    if (m) cudaCheck(cudaMemcpyAsync(A.p, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
    gpu.sync();
}

// A small diagonal tile is factored on the CPU while cubic trailing updates
// stay on the GPU. One OpenMP team handles all columns of the tile.
static bool factorDiagonal(double* tile, int b) {
    bool good = true;
    #pragma omp parallel shared(good)
    {
        for (int j = 0; j < b; ++j) {
            #pragma omp single
            {
                double v = tile[size_t(j) * b + j];
                for (int k = 0; k < j; ++k) v -= tile[size_t(k) * b + j] * tile[size_t(k) * b + j];
                if (!(v > 0.0) || !std::isfinite(v)) good = false;
                else tile[size_t(j) * b + j] = std::sqrt(v);
            }
            #pragma omp for schedule(static)
            for (int i = j + 1; i < b; ++i) {
                if (good) {
                    double v = tile[size_t(j) * b + i];
                    for (int k = 0; k < j; ++k) v -= tile[size_t(k) * b + i] * tile[size_t(k) * b + j];
                    tile[size_t(j) * b + i] = v / tile[size_t(j) * b + j];
                }
                tile[size_t(i) * b + j] = 0.0;
            }
        }
    }
    return good;
}

struct FactorWorkspace {
    DeviceBuffer diagonal, panel;
    PinnedBuffer hostDiagonal, send, received, hostPanel;
    std::vector<int> counts, offsets, begins;
    explicit FactorWorkspace(const Rows& rows)
        : diagonal(size_t(rows.block) * rows.block),
          panel(rows.ranks > 1 ? size_t(rows.n) * rows.block : 1),
          hostDiagonal(size_t(rows.block) * rows.block),
          send(size_t(rows.count()) * rows.block),
          received(rows.ranks > 1 ? size_t(rows.n) * rows.block : 1),
          hostPanel(rows.ranks > 1 ? size_t(rows.n) * rows.block : 1),
          counts(rows.ranks), offsets(rows.ranks), begins(rows.ranks) {}
};

static bool choleskyDecomposition(DeviceBuffer& A, const Rows& rows, Accelerator& gpu,
                                  FactorWorkspace& work) {
    const int n = rows.n, ld = std::max(1, rows.count()), tile = rows.block;
    auto& diagonal = work.diagonal;
    auto& panel = work.panel;
    auto& hostDiagonal = work.hostDiagonal;
    auto& send = work.send;
    auto& received = work.received;
    auto& hostPanel = work.hostPanel;
    auto& counts = work.counts;
    auto& offsets = work.offsets;
    auto& begins = work.begins;
    const double one = 1.0, minusOne = -1.0;
    for (int k = 0; k < n; k += tile) {
        const int b = std::min(tile, n - k), owner = (k / tile) % rows.ranks;
        int good = 1;
        if (rows.rank == owner) {
            int pos = rows.start(k);
            cudaCheck(cudaMemcpy2DAsync(hostDiagonal.p, size_t(b) * sizeof(double),
                A.p + size_t(k) * ld + pos, size_t(ld) * sizeof(double),
                size_t(b) * sizeof(double), b, cudaMemcpyDeviceToHost, gpu.stream));
            gpu.sync();
            good = factorDiagonal(hostDiagonal.p, b);
        }
        MPI_Bcast(&good, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!good) {
            if (rows.rank == 0) printf("Error: Matrix is not positive definite in diagonal block %d\n", k);
            return false;
        }
        MPI_Bcast(hostDiagonal.p, b * b, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpyAsync(diagonal.p, hostDiagonal.p, size_t(b) * b * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
        if (rows.rank == owner) {
            cudaCheck(cudaMemcpy2DAsync(A.p + size_t(k) * ld + rows.start(k), size_t(ld) * sizeof(double),
                hostDiagonal.p, size_t(b) * sizeof(double), size_t(b) * sizeof(double), b,
                cudaMemcpyHostToDevice, gpu.stream));
        }
        const int first = rows.start(k + b), m = rows.count() - first, trailing = n - k - b;
        if (m) {
            blasCheck(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, b, &one, diagonal.p, b,
                A.p + size_t(k) * ld + first, ld));
        }
        if (rows.ranks == 1) {
            if (m) blasCheck(cublasDsyrk(gpu.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                m, b, &minusOne, A.p + size_t(k) * ld + first, ld,
                &one, A.p + size_t(k + b) * ld + first, ld));
            gpu.sync();
            continue;
        }
        if (m) {
            cudaCheck(cudaMemcpy2DAsync(send.p, size_t(m) * sizeof(double),
                A.p + size_t(k) * ld + first, size_t(ld) * sizeof(double),
                size_t(m) * sizeof(double), b, cudaMemcpyDeviceToHost, gpu.stream));
        }
        gpu.sync();
        if (!trailing) break;
        int total = 0;
        for (int r = 0; r < rows.ranks; ++r) {
            const auto& indices = rows.byRank[r];
            begins[r] = static_cast<int>(std::lower_bound(indices.begin(), indices.end(), k + b) - indices.begin());
            counts[r] = (static_cast<int>(indices.size()) - begins[r]) * b;
            offsets[r] = total;
            total += counts[r];
        }
        // Pinned host staging works with ordinary MPI as well as GPU-aware MPI.
        // Only the solved panel, never the trailing matrix, crosses the network.
        MPI_Allgatherv(send.p, m * b, MPI_DOUBLE, received.p, counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        #pragma omp parallel for collapse(2) schedule(static) if(m)
        for (int r = 0; r < rows.ranks; ++r) {
            for (int j = 0; j < b; ++j) {
                const int nr = counts[r] / b;
                for (int i = 0; i < nr; ++i) {
                    int row = rows.byRank[r][begins[r] + i] - k - b;
                    hostPanel.p[size_t(j) * trailing + row] = received.p[offsets[r] + size_t(j) * nr + i];
                }
            }
        }
        if (m) {
            cudaCheck(cudaMemcpyAsync(panel.p, hostPanel.p, size_t(trailing) * b * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
            // Rectangles to the left of each owned tile and a symmetric
            // diagonal update compute only the useful lower triangle.
            for (int pos = first; pos < rows.count(); pos += tile) {
                const int row = rows.global[pos];
                const int height = std::min(tile, rows.count() - pos);
                const int width = row - k - b;
                if (width) blasCheck(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    height, width, b, &minusOne, A.p + size_t(k) * ld + pos, ld,
                    panel.p, trailing, &one, A.p + size_t(k + b) * ld + pos, ld));
                blasCheck(cublasDsyrk(gpu.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    height, b, &minusOne, A.p + size_t(k) * ld + pos, ld,
                    &one, A.p + size_t(row) * ld + pos, ld));
            }
        }
        // The next iteration reuses host/device panel buffers. Finish their
        // consumers before MPI or the CPU overwrites those buffers.
        gpu.sync();
    }
    gpu.sync();
    return true;
}

static std::vector<double> collectFactor(DeviceBuffer& A, const Rows& rows,
                                         Accelerator& gpu, bool allRanks) {
    int m = rows.count(), n = rows.n;
    std::vector<double> local(size_t(m) * n);
    if (m) cudaCheck(cudaMemcpyAsync(local.data(), A.p, local.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
    gpu.sync();
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (j > rows.global[i]) local[size_t(j) * m + i] = 0.0;
    // Reuse the now-triangular device factor for GPU validation.
    if (allRanks && m) cudaCheck(cudaMemcpyAsync(A.p, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
    std::vector<int> counts(rows.ranks), offsets(rows.ranks);
    int total = 0;
    for (int r = 0; r < rows.ranks; ++r) {
        counts[r] = static_cast<int>(rows.byRank[r].size()) * n;
        offsets[r] = total;
        total += counts[r];
    }
    const bool receive = allRanks || rows.rank == 0;
    std::vector<double> packed(receive ? size_t(n) * n : 0);
    if (allRanks) MPI_Allgatherv(local.data(), m * n, MPI_DOUBLE, packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    else MPI_Gatherv(local.data(), m * n, MPI_DOUBLE, packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Full factor is row-major, matching the original results interface.
    std::vector<double> full(receive ? size_t(n) * n : 0);
    if (receive) {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows.ranks; ++r) {
            int nr = static_cast<int>(rows.byRank[r].size());
            for (int i = 0; i < nr; ++i)
                for (int j = 0; j < n; ++j)
                    full[size_t(rows.byRank[r][i]) * n + j] = packed[offsets[r] + size_t(j) * nr + i];
        }
    }
    gpu.sync();
    return full;
}

static bool validateCholesky(DeviceBuffer& A, const std::vector<double>& full,
                             const std::vector<double>& original, const Rows& rows, Accelerator& gpu) {
    const int n = rows.n, m = rows.count();
    double maxError = 0.0, relError = 0.0;
    if (m) {
        DeviceBuffer L(size_t(n) * n), product(size_t(m) * n);
        cudaCheck(cudaMemcpyAsync(L.p, full.data(), full.size() * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
        const double one = 1.0, zero = 0.0;
        blasCheck(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
            &one, A.p, m, L.p, n, &zero, product.p, m));
        std::vector<double> reconstructed(size_t(m) * n);
        cudaCheck(cudaMemcpyAsync(reconstructed.data(), product.p, reconstructed.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
        gpu.sync();
        #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
        for (size_t i = 0; i < reconstructed.size(); ++i) {
            const double error = std::abs(reconstructed[i] - original[i]);
            const double rel = error / (std::abs(original[i]) + 1e-10);
            maxError = std::max(maxError, std::isfinite(error) ? error : INFINITY);
            relError = std::max(relError, std::isfinite(rel) ? rel : INFINITY);
        }
    }
    double errors[2] = {maxError, relError}, global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rows.rank == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            // MPI vector collectives use int counts; guard all products.
            if (errno || *end || end == argv[i] || value <= 0 || value > 46340) {
                if (rank == 0) fprintf(stderr, "Matrix size must be between 1 and 46340\n");
                return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "This hybrid benchmark requires a CUDA GPU on every node\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, omp_get_max_threads()));
    Accelerator gpu;
    // Enough row blocks for load balance, but large BLAS-3 operations at scale.
    const int block = static_cast<int>(std::min(256LL, std::max(32LL, (n / (4LL * ranks) / 32) * 32LL)));
    Rows rows(n, block, rank, ranks);
    DeviceBuffer A(size_t(rows.count()) * n);
    std::vector<double> original;
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; block size: %d\n", ranks, omp_get_max_threads(), block);
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, rows, gpu, original, validate);
    FactorWorkspace workspace(rows);
    // Load the BLAS kernels before timing; GPU library lazy initialization is
    // setup cost, rather than work in the decomposition.
    const double one = 1.0, zero = 0.0;
    workspace.hostDiagonal.p[0] = 1.0;
    cudaCheck(cudaMemcpyAsync(workspace.diagonal.p, workspace.hostDiagonal.p,
        sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
    cudaCheck(cudaMemsetAsync(workspace.panel.p, 0, sizeof(double), gpu.stream));
    blasCheck(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1, 1, &one,
        workspace.diagonal.p, 1, workspace.panel.p, 1));
    blasCheck(cublasDsyrk(gpu.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        1, 1, &zero, workspace.diagonal.p, 1, &zero, workspace.panel.p, 1));
    gpu.sync();
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    bool success = choleskyDecomposition(A, rows, gpu, workspace);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0 * duration * 1e9));
    }
    if (printResults || validate) {
        auto full = collectFactor(A, rows, gpu, validate);
        if (printResults && rank == 0) print_results(full, "CholeskyL");
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateCholesky(A, full, original, rows, gpu);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try { result = benchmark(argc, argv, rank, ranks); }
    catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
