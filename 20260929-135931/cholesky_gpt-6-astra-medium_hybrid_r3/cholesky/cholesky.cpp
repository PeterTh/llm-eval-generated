#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// All MPI calls are made by the main thread. Host staging also works with MPI
// implementations without CUDA-aware support; matrices stay resident on GPUs.
static void check(int status, const char* call, int line) {
    if (status != 0) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s failed (%d), line %d\n", rank, call, status, line);
        MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
}
#define CHECK(call) check(static_cast<int>(call), #call, __LINE__)

struct DeviceBuffer {
    double* p = nullptr;
    explicit DeviceBuffer(size_t count) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~DeviceBuffer() { cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct HostBuffer {
    double* p = nullptr;
    explicit HostBuffer(size_t count) {
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~HostBuffer() { cudaFreeHost(p); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

// A two-dimensional block-cyclic layout, packed as one column-major local
// matrix. This allows large BLAS-3 updates without a kernel launch per tile.
struct Grid {
    int rank, size, dims[2] = {0, 0}, row, col, n, block;
    MPI_Comm rows, cols;
    std::vector<int> ri, ci;
    Grid(int order) : n(order) {
        CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        CHECK(MPI_Dims_create(size, 2, dims));
        row = rank / dims[1];
        col = rank % dims[1];
        block = std::min(256, std::max(32, int((static_cast<long long>(n) + 4LL * dims[0] - 1) /
                                             (4LL * dims[0]) + 31) / 32 * 32));
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows));
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols));
        for (int i = 0; i < n; ++i) {
            if ((i / block) % dims[0] == row) ri.push_back(i);
            if ((i / block) % dims[1] == col) ci.push_back(i);
        }
    }
    ~Grid() { MPI_Comm_free(&rows); MPI_Comm_free(&cols); }
    int m() const { return static_cast<int>(ri.size()); }
    int q() const { return static_cast<int>(ci.size()); }
    int rowStart(int k) const { return std::lower_bound(ri.begin(), ri.end(), k) - ri.begin(); }
    int colStart(int k) const { return std::lower_bound(ci.begin(), ci.end(), k) - ci.begin(); }
};

struct Cholesky {
    Grid& g;
    int ld;
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    double* work = nullptr;
    int workSize = 0;
    int* infoDevice = nullptr;
    DeviceBuffer a, diagonal, rp, cp;
    HostBuffer hr, hc, packed, received, hd;
    std::vector<double> original;
    std::vector<int> counts, displs, slot;

    explicit Cholesky(Grid& grid) : g(grid), ld(std::max(1, g.m())),
        a(size_t(ld) * g.q()), diagonal(size_t(g.block) * g.block),
        rp(size_t(ld) * g.block), cp(size_t(std::max(1, g.q())) * g.block),
        hr(size_t(ld) * g.block), hc(size_t(std::max(1, g.q())) * g.block),
        packed(size_t(std::max(1, g.q())) * g.block),
        received(size_t(std::max(1, g.q())) * g.block), hd(size_t(g.block) * g.block),
        counts(g.dims[0]), displs(g.dims[0]), slot(g.q()) {
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cusolverDnCreate(&solver));
        CHECK(cublasSetStream(blas, stream));
        CHECK(cusolverDnSetStream(solver, stream));
        // Keep all arithmetic in double precision, including matrix products.
        CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, g.block,
                                          diagonal.p, g.block, &workSize));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&work), size_t(std::max(1, workSize)) * sizeof(double)));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&infoDevice), sizeof(int)));
        // Initialize the libraries before timing: their first calls can load
        // device code lazily. This small identity factorization also prewarms
        // the reusable solver workspace without touching the input matrix.
        std::fill(hd.p, hd.p + size_t(g.block) * g.block, 0.0);
        for (int i = 0; i < g.block; ++i) hd.p[i + size_t(i) * g.block] = 1.0;
        CHECK(cudaMemcpyAsync(diagonal.p, hd.p, size_t(g.block) * g.block * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, g.block, diagonal.p,
                               g.block, work, workSize, infoDevice));
        const double one = 1.0;
        CHECK(cudaMemcpyAsync(work, &one, sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1, 1, &one, diagonal.p, g.block, work, 1));
        sync();
    }
    ~Cholesky() {
        cudaFree(infoDevice);
        cudaFree(work);
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
    void sync() { CHECK(cudaStreamSynchronize(stream)); }

    void generate(bool saveOriginal) {
        // Checkpoint the original rand_r stream. Each row can then be generated
        // independently by OpenMP without changing the benchmark's input.
        std::vector<unsigned int> seeds(g.n);
        if (g.rank == 0) {
            unsigned int seed = 42;
            for (int i = 0; i < g.n; ++i) {
                seeds[i] = seed;
                for (int j = 0; j < g.n; ++j) (void)rand_r(&seed);
            }
        }
        CHECK(MPI_Bcast(seeds.data(), g.n, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
        // Stream B in inner-dimension panels, so generation does not require
        // replicated n-by-n random matrices on the host or accelerator.
        const int depth = std::min(g.n, 1024);
        std::vector<double> br(size_t(g.m()) * depth), bc(size_t(g.q()) * depth);
        std::vector<unsigned int> rowSeeds(g.m()), colSeeds(g.q());
        for (int i = 0; i < g.m(); ++i) rowSeeds[i] = seeds[g.ri[i]];
        for (int j = 0; j < g.q(); ++j) colSeeds[j] = seeds[g.ci[j]];
        DeviceBuffer dr(br.size()), dc(bc.size());
        const double one = 1.0, zero = 0.0;
        for (int first = 0; first < g.n; first += depth) {
            const int width = std::min(depth, g.n - first);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < g.m() + g.q(); ++i) {
                const bool isRow = i < g.m();
                const int local = isRow ? i : i - g.m();
                unsigned int seed = isRow ? rowSeeds[local] : colSeeds[local];
                double* dst = (isRow ? br.data() : bc.data()) + size_t(local) * width;
                for (int k = 0; k < width; ++k)
                    dst[k] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                (isRow ? rowSeeds[local] : colSeeds[local]) = seed;
            }
            CHECK(cudaMemcpyAsync(dr.p, br.data(), size_t(g.m()) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(dc.p, bc.data(), size_t(g.q()) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
            if (g.m() && g.q())
                CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, g.m(), g.q(), width,
                    &one, dr.p, width, dc.p, width, first == 0 ? &zero : &one, a.p, ld));
            sync();
        }
        std::vector<double> host(size_t(g.m()) * g.q());
        CHECK(cudaMemcpyAsync(host.data(), a.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        sync();
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < g.q(); ++j) {
            const int i = g.rowStart(g.ci[j]);
            if (i < g.m() && g.ri[i] == g.ci[j]) host[i + size_t(j) * ld] += g.n;
        }
        CHECK(cudaMemcpyAsync(a.p, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        sync();
        if (saveOriginal) original.swap(host);
    }

    // Broadcast L(:,k) across process rows, then redistribute it within process
    // columns to obtain the other operand of L(:,k)*L(:,k)^T. Allgatherv handles
    // rectangular grids, partial tiles and ranks with no local rows/columns.
    void exchangePanel(int k, int width, int first, bool triangular) {
        const int rs = g.rowStart(first), cs = g.colStart(first);
        const int m = g.m() - rs, q = g.q() - cs;
        const int ownerCol = (k / g.block) % g.dims[1];
        if (g.col == ownerCol && m) {
            CHECK(cudaMemcpy2DAsync(hr.p, size_t(m) * sizeof(double),
                a.p + rs + size_t(g.colStart(k)) * ld, size_t(ld) * sizeof(double),
                size_t(m) * sizeof(double), width, cudaMemcpyDeviceToHost, stream));
        }
        sync();
        if (triangular && g.col == ownerCol) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < width; ++t)
                for (int i = 0; i < m && g.ri[rs + i] < k + t; ++i)
                    hr.p[i + size_t(t) * m] = 0.0;
        }
        CHECK(MPI_Bcast(hr.p, m * width, MPI_DOUBLE, ownerCol, g.rows));
        // Overlap the row-panel upload with host packing and column exchange.
        CHECK(cudaMemcpyAsync(rp.p, hr.p, size_t(m) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
        std::fill(counts.begin(), counts.end(), 0);
        for (int j = cs; j < g.q(); ++j) {
            const int ownerRow = (g.ci[j] / g.block) % g.dims[0];
            slot[j] = counts[ownerRow];
            counts[ownerRow] += width;
        }
        int total = 0;
        for (int r = 0; r < g.dims[0]; ++r) {
            displs[r] = total;
            total += counts[r];
        }
        #pragma omp parallel for schedule(static)
        for (int j = cs; j < g.q(); ++j) {
            if ((g.ci[j] / g.block) % g.dims[0] == g.row) {
                const int i = (g.ci[j] / (g.block * g.dims[0])) * g.block + g.ci[j] % g.block - rs;
                for (int t = 0; t < width; ++t)
                    packed.p[slot[j] + t] = hr.p[i + size_t(t) * m];
            }
        }
        CHECK(MPI_Allgatherv(packed.p, counts[g.row], MPI_DOUBLE, received.p,
                             counts.data(), displs.data(), MPI_DOUBLE, g.cols));
        #pragma omp parallel for schedule(static)
        for (int j = cs; j < g.q(); ++j) {
            const int source = displs[(g.ci[j] / g.block) % g.dims[0]] + slot[j];
            for (int t = 0; t < width; ++t)
                hc.p[j - cs + size_t(t) * q] = received.p[source + t];
        }
        CHECK(cudaMemcpyAsync(cp.p, hc.p, size_t(q) * width * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    // Update only the lower block triangle. A whole vertical group of local
    // tiles is updated by each GEMM; no O(n^3) work is replicated across ranks.
    void update(double* dst, int first, int width, double alpha) {
        const int rs = g.rowStart(first), cs = g.colStart(first);
        const int m = g.m() - rs, q = g.q() - cs;
        const double one = 1.0;
        for (int j = cs; j < g.q();) {
            const int global = g.ci[j];
            const int w = std::min(g.block, g.n - global);
            const int r = g.rowStart(global);
            if (r < g.m())
                CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, g.m() - r, w, width,
                    &alpha, rp.p + r - rs, std::max(1, m), cp.p + j - cs, std::max(1, q),
                    &one, dst + r + size_t(j) * ld, ld));
            j += w;
        }
    }

    bool factor() {
        const double one = 1.0;
        for (int k = 0; k < g.n; k += g.block) {
            const int b = std::min(g.block, g.n - k);
            const int ownerRow = (k / g.block) % g.dims[0];
            const int ownerCol = (k / g.block) % g.dims[1];
            const int owner = ownerRow * g.dims[1] + ownerCol;
            int info = 0;
            if (g.rank == owner) {
                double* tile = a.p + g.rowStart(k) + size_t(g.colStart(k)) * ld;
                CHECK(cudaMemcpy2DAsync(diagonal.p, size_t(b) * sizeof(double), tile,
                    size_t(ld) * sizeof(double), size_t(b) * sizeof(double), b, cudaMemcpyDeviceToDevice, stream));
                CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, diagonal.p,
                                       b, work, workSize, infoDevice));
                CHECK(cudaMemcpyAsync(&info, infoDevice, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CHECK(cudaMemcpy2DAsync(tile, size_t(ld) * sizeof(double), diagonal.p,
                    size_t(b) * sizeof(double), size_t(b) * sizeof(double), b, cudaMemcpyDeviceToDevice, stream));
                CHECK(cudaMemcpyAsync(hd.p, diagonal.p, size_t(b) * b * sizeof(double), cudaMemcpyDeviceToHost, stream));
                sync();
            }
            CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (info != 0) {
                if (g.rank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
                return false;
            }
            if (g.col == ownerCol) {
                CHECK(MPI_Bcast(hd.p, b * b, MPI_DOUBLE, ownerRow, g.cols));
                CHECK(cudaMemcpyAsync(diagonal.p, hd.p, size_t(b) * b * sizeof(double), cudaMemcpyHostToDevice, stream));
                const int r = g.rowStart(k + b);
                if (r < g.m())
                    CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, g.m() - r, b, &one,
                        diagonal.p, b, a.p + r + size_t(g.colStart(k)) * ld, ld));
            }
            if (k + b < g.n) {
                exchangePanel(k, b, k + b, false);
                update(a.p, k + b, b, -1.0);
            }
            // Host staging buffers cannot be reused until their async copies
            // complete. This also completes local updates before the next panel.
            sync();
        }
        return true;
    }

    bool validate() {
        DeviceBuffer reconstructed(size_t(ld) * g.q());
        CHECK(cudaMemsetAsync(reconstructed.p, 0, size_t(ld) * g.q() * sizeof(double), stream));
        for (int k = 0; k < g.n; k += g.block) {
            const int b = std::min(g.block, g.n - k);
            exchangePanel(k, b, k, true);
            update(reconstructed.p, k, b, 1.0);
            sync();
        }
        std::vector<double> host(original.size());
        CHECK(cudaMemcpyAsync(host.data(), reconstructed.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        sync();
        double absolute = 0.0, relative = 0.0;
        #pragma omp parallel for reduction(max:absolute,relative) schedule(static)
        for (int j = 0; j < g.q(); ++j) {
            for (int i = g.rowStart(g.ci[j]); i < g.m(); ++i) {
                const size_t idx = i + size_t(j) * ld;
                const double error = std::fabs(host[idx] - original[idx]);
                const double rel = error / (std::fabs(original[idx]) + 1e-10);
                absolute = std::max(absolute, std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
                relative = std::max(relative, std::isfinite(rel) ? rel : std::numeric_limits<double>::infinity());
            }
        }
        double local[2] = {absolute, relative}, global[2];
        CHECK(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (g.rank == 0) {
            std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            if (global[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
        }
        return global[1] <= 1e-6;
    }

    void printResults() {
        std::vector<double> local(size_t(g.m()) * g.q());
        CHECK(cudaMemcpyAsync(local.data(), a.p, local.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        sync();
        std::vector<double> result;
        if (g.rank == 0) result.resize(size_t(g.n) * g.n, 0.0);
        // Gathering is needed only for the original row-major output/hash. Chunk
        // messages so that even very large matrices do not overflow MPI counts.
        for (int rank = 0; rank < g.size; ++rank) {
            std::vector<int> ri, ci;
            for (int i = 0; i < g.n; ++i) {
                if ((i / g.block) % g.dims[0] == rank / g.dims[1]) ri.push_back(i);
                if ((i / g.block) % g.dims[1] == rank % g.dims[1]) ci.push_back(i);
            }
            const size_t count = ri.size() * ci.size();
            std::vector<double> recv;
            if (g.rank == 0 && rank != 0) recv.resize(count);
            double* data = rank == 0 ? local.data() : recv.data();
            for (size_t offset = 0; offset < count; offset += INT_MAX) {
                const int chunk = static_cast<int>(std::min(size_t(INT_MAX), count - offset));
                if (g.rank == rank && rank != 0)
                    CHECK(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                if (g.rank == 0 && rank != 0)
                    CHECK(MPI_Recv(data + offset, chunk, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            if (g.rank == 0) {
                #pragma omp parallel for schedule(static)
                for (size_t j = 0; j < ci.size(); ++j)
                    for (size_t i = 0; i < ri.size(); ++i)
                        if (ri[i] >= ci[j]) result[size_t(ri[i]) * g.n + ci[j]] = data[i + j * ri.size()];
            }
        }
        if (g.rank == 0) print_results(result, "CholeskyL");
    }
};

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv, int rank) {
    int n = 512;
    bool validation = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const long long value = std::strtoll(argv[++i], &end, 10);
            if (errno || !*argv[i] || *end || value < 0 || value > INT_MAX / 256) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validation = true;
        else if (std::strcmp(argv[i], "-r") == 0) results = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    MPI_Comm node;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank, localSize, devices;
    CHECK(MPI_Comm_rank(node, &localRank));
    CHECK(MPI_Comm_size(node, &localSize));
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("MPI/OpenMP/CUDA benchmark requires a CUDA device on every rank");
    CHECK(cudaSetDevice(localRank % devices));
    // Host work is bandwidth bound and panels are small. Avoid launching a
    // machine-wide OpenMP team per rank unless the user explicitly requests it.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    CHECK(MPI_Comm_free(&node));

    Grid grid(n);
    Cholesky matrix(grid);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n",
                    n, n, validation ? "enabled" : "disabled");
        std::printf("MPI grid: %d x %d; block size: %d; OpenMP threads/rank: %d\n",
                    grid.dims[0], grid.dims[1], grid.block, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }
    matrix.generate(validation);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success = matrix.factor();
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", seconds > 0 ? double(n) * n * n / (3e9 * seconds) : 0.0);
    }
    if (results) matrix.printResults();
    if (validation) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = matrix.validate();
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided;
    CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank;
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int result = 1;
    try { result = run(argc, argv, rank); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(MPI_Finalize());
    return result;
}
