#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
void check(int status, const char* expression) {
    if (status) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s failed (status %d)\n", rank, expression, status);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CHECK(call) check((call), #call)

template<class T>
struct Device {
    T* p = nullptr;
    explicit Device(size_t count) {
        if (count) CHECK(cudaMalloc(&p, count * sizeof(T)));
    }
    ~Device() { if (p) CHECK(cudaFree(p)); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

struct Grid {
    int n, rank, ranks, width, blocks, cols = 0;
    Grid(int size, int id, int count) : n(size), rank(id), ranks(count) {
        // Aim for at least four panels per rank, with 32 to 256 columns per panel.
        width = 32 * std::clamp<int>((size + 128LL * count - 1) / (128LL * count), 1, 8);
        blocks = (n - 1) / width + 1;
        for (int k = rank; k < blocks; k += ranks) cols += span(k);
    }
    int globalColumn(int local) const {
        return (local / width * ranks + rank) * width + local % width;
    }
    int span(int k) const { return std::min(width, n - k * width); }
    size_t offset(int k) const { return size_t(k / ranks) * width * n; }
};

constexpr double one = 1., zero = 0., minusOne = -1.;

// MPI implementations need not support device pointers. Pack only the active
// rows into pinned memory, retaining the original stride on the owning GPU.
struct Panels {
    const Grid& g;
    Device<double> received;
    double* host = nullptr;
    explicit Panels(const Grid& grid) : g(grid),
        received(grid.ranks > 1 ? size_t(grid.n) * grid.span(0) : 0) {
        if (g.ranks > 1)
            CHECK(cudaMallocHost(&host, size_t(g.n) * g.span(0) * sizeof(double)));
    }
    ~Panels() { if (host) CHECK(cudaFreeHost(host)); }

    const double* broadcast(const Device<double>& matrix, int k, int firstRow, int& ld,
                            cudaStream_t producer = nullptr) {
        const int rows = g.n - firstRow, cols = g.span(k), owner = k % g.ranks;
        const double* panel = nullptr;
        ld = rows;
        if (g.rank == owner) {
            ld = g.n;
            panel = matrix.p + g.offset(k) + firstRow;
            if (g.ranks > 1) {
                CHECK(cudaMemcpy2DAsync(host, rows * sizeof(double), panel, ld * sizeof(double),
                                        rows * sizeof(double), cols, cudaMemcpyDeviceToHost, producer));
                CHECK(cudaStreamSynchronize(producer));
            }
        }
        if (g.ranks > 1) {
            const size_t count = size_t(rows) * cols;
            for (size_t i = 0; i < count; i += INT_MAX)
                CHECK(MPI_Bcast(host + i, std::min<size_t>(count - i, INT_MAX), MPI_DOUBLE,
                                owner, MPI_COMM_WORLD));
            if (g.rank != owner) {
                CHECK(cudaMemcpy(received.p, host, count * sizeof(double), cudaMemcpyHostToDevice));
                panel = received.p;
            }
        }
        return panel;
    }
};

// glibc rand_r advances this 32-bit LCG three times per sample. Jump
// directly to the start of each independently generated row of the serial B.
#ifdef __GLIBC__
unsigned rowSeed(size_t samples) {
    unsigned a = 1103515245U, b = 12345U, seed = 42;
    for (size_t steps = 3 * samples; steps; steps >>= 1) {
        if (steps & 1) seed = a * seed + b;
        b *= a + 1;
        a *= a;
    }
    return seed;
}
#endif

void generate(const Grid& g, Panels& panels, cublasHandle_t blas, Device<double>& a) {
    Device<double> b(size_t(g.cols) * g.n);
    std::vector<double> rows(size_t(g.cols) * g.n);
#ifndef __GLIBC__
    std::vector<unsigned> seeds(g.n);
    unsigned state = 42;
    for (int i = 0; i < g.n; ++i) {
        seeds[i] = state;
        for (int j = 0; j < g.n; ++j) rand_r(&state);
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int local = 0; local < g.cols; ++local) {
        int row = g.globalColumn(local);
#ifdef __GLIBC__
        unsigned seed = rowSeed(size_t(row) * g.n);
#else
        unsigned seed = seeds[row];
#endif
        for (int i = 0; i < g.n; ++i)
            rows[size_t(local) * g.n + i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    if (g.cols)
        CHECK(cudaMemcpy(b.p, rows.data(), rows.size() * sizeof(double), cudaMemcpyHostToDevice));

    if (g.ranks == 1) {
        CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, g.n, g.n,
                          &one, b.p, g.n, &zero, a.p, g.n));
    } else {
        for (int k = 0; k < g.blocks; ++k) {
            int ld;
            const double* panel = panels.broadcast(b, k, 0, ld);
            // Local block columns through k comprise a contiguous prefix.
            const int owned = k < g.rank ? 0 : (k - g.rank) / g.ranks + 1;
            const int cols = std::min<long long>(g.cols, 1LL * owned * g.width);
            if (cols)
                CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, g.span(k), cols, g.n,
                                  &one, panel, ld, b.p, g.n, &zero, a.p + k * g.width, g.n));
        }
    }
    Device<double> increment(g.span(0));
    std::vector<double> values(g.span(0), g.n);
    CHECK(cudaMemcpy(increment.p, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    for (int k = g.rank; k < g.blocks; k += g.ranks)
        CHECK(cublasDaxpy(blas, g.span(k), &one, increment.p, 1, a.p + g.offset(k) + k * g.width,
                          g.n + 1));
}

// Apply a panel outer product to each locally owned block column. Only
// the lower triangle of the diagonal block is read by the next POTRF.
void update(const Grid& g, cublasHandle_t blas, Device<double>& target, int k,
            const double* panel, int ld, int firstRow, bool reconstruct,
            cudaEvent_t nextReady = nullptr) {
    int start = reconstruct ? k : k + 1;
    for (int j = start + (g.rank - start % g.ranks + g.ranks) % g.ranks; j < g.blocks; j += g.ranks) {
        int row = j * g.width;
        const double* slice = panel + (row - firstRow);
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, g.n - row, g.span(j), g.span(k),
                          reconstruct ? &one : &minusOne, slice, ld, slice, ld, &one,
                          target.p + g.offset(j) + row, g.n));
        if (nextReady && j == k + 1)
            CHECK(cudaEventRecord(nextReady, nullptr));
    }
}

int factor(const Grid& g, Panels& panels, cublasHandle_t blas, cusolverDnHandle_t solver,
           Device<double>& a, double& elapsed) {
    int order = g.ranks == 1 ? g.n : g.span(0);
    int length = 0;
    if (g.cols)
        CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, order, a.p,
                                          g.n, &length));
    Device<double> work(length);
    const int panelsToFactor = g.ranks == 1 ? 1 : g.blocks;
    const int localPanels = g.rank < panelsToFactor ? 1 + (panelsToFactor - 1 - g.rank) / g.ranks : 0;
    Device<int> info(localPanels);
    // Look ahead one panel. Its owner can start POTRF and TRSM as soon as
    // that block column is updated, while other DGEMMs remain in flight.
    cudaStream_t panelStream;
    cudaEvent_t nextReady;
    cublasHandle_t panelBlas;
    CHECK(cudaStreamCreateWithFlags(&panelStream, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&nextReady, cudaEventDisableTiming));
    CHECK(cublasCreate(&panelBlas));
    CHECK(cublasSetStream(panelBlas, panelStream));
    CHECK(cusolverDnSetStream(solver, panelStream));
    CHECK(cudaDeviceSynchronize());
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    for (int k = 0; k < g.blocks; ++k) {
        int row = k * g.width, w = g.ranks == 1 ? g.n : g.span(k);
        if (k % g.ranks == g.rank) {
            if (k) CHECK(cudaStreamWaitEvent(panelStream, nextReady, 0));
            double* diag = a.p + g.offset(k) + row;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, w, diag, g.n,
                                   work.p, length, info.p + k / g.ranks));
            if (row + w < g.n)
                CHECK(cublasDtrsm(panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                  CUBLAS_DIAG_NON_UNIT, g.n - row - w, w, &one,
                                  diag, g.n, diag + w, g.n));
        }
        if (row + w == g.n) break;
        int ld;
        const double* panel = panels.broadcast(a, k, row + w, ld, panelStream);
        update(g, blas, a, k, panel, ld, row + w, false, nextReady);
    }
    CHECK(cudaDeviceSynchronize());
    double local = MPI_Wtime() - start;
    CHECK(MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    std::vector<int> status(localPanels);
    if (localPanels)
        CHECK(cudaMemcpy(status.data(), info.p, status.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int bad = INT_MAX;
    for (int k = g.rank; k < panelsToFactor; k += g.ranks) {
        int code = status[k / g.ranks];
        if (code) bad = std::min(bad, k * g.width + std::max(0, code - 1));
    }
    CHECK(MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    CHECK(cusolverDnSetStream(solver, nullptr));
    CHECK(cublasDestroy(panelBlas));
    CHECK(cudaEventDestroy(nextReady));
    CHECK(cudaStreamDestroy(panelStream));
    return bad;
}

std::vector<double> lowerFactor(const Grid& g, const Device<double>& a) {
    std::vector<double> lower(size_t(g.n) * g.cols);
    if (g.cols)
        CHECK(cudaMemcpy(lower.data(), a.p, lower.size() * sizeof(double), cudaMemcpyDeviceToHost));
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < g.cols; ++j) {
        int col = g.globalColumn(j);
        std::fill_n(lower.data() + size_t(j) * g.n, col, 0.);
    }
    return lower;
}

void results(const Grid& g, const std::vector<double>& lower) {
    std::vector<double> output, buffer;
    if (!g.rank) {
        output.resize(size_t(g.n) * g.n);
        buffer.resize(size_t(g.span(0)) * g.n);
    }
    for (int k = 0; k < g.blocks; ++k) {
        int owner = k % g.ranks, w = g.span(k);
        size_t count = size_t(w) * g.n;
        const double* block = nullptr;
        if (owner == g.rank) block = lower.data() + g.offset(k);
        if (owner && (g.rank == 0 || g.rank == owner)) {
            for (size_t i = 0; i < count; i += INT_MAX) {
                int part = std::min<size_t>(count - i, INT_MAX);
                if (g.rank)
                    CHECK(MPI_Send(block + i, part, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                else
                    CHECK(MPI_Recv(buffer.data() + i, part, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            if (!g.rank) block = buffer.data();
        }
        if (!g.rank) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < g.n; ++i)
                for (int j = 0; j < w; ++j)
                    output[size_t(i) * g.n + k * g.width + j] = block[size_t(j) * g.n + i];
        }
    }
    if (!g.rank) print_results(output, "CholeskyL");
}

bool validate(const Grid& g, Panels& panels, cublasHandle_t blas, Device<double>& a,
              const std::vector<double>& lower, const Device<double>& original) {
    Device<double> product(lower.size());
    if (g.cols) {
        CHECK(cudaMemcpy(a.p, lower.data(), lower.size() * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(product.p, 0, lower.size() * sizeof(double)));
    }
    if (g.ranks == 1) {
        CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, g.n, g.n,
                          &one, a.p, g.n, &zero, product.p, g.n));
    } else {
        for (int k = 0; k < g.blocks; ++k) {
            int ld, row = k * g.width;
            const double* panel = panels.broadcast(a, k, row, ld);
            update(g, blas, product, k, panel, ld, row, true);
        }
    }
    std::vector<double> expected(lower.size()), actual(lower.size());
    if (g.cols) {
        CHECK(cudaMemcpy(expected.data(), original.p, lower.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(actual.data(), product.p, lower.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    double absolute = 0., relative = 0.;
    #pragma omp parallel for reduction(max:absolute,relative)
    for (int j = 0; j < g.cols; ++j) {
        int col = g.globalColumn(j);
        for (int i = col; i < g.n; ++i) {
            size_t index = size_t(j) * g.n + i;
            double error = std::abs(actual[index] - expected[index]);
            if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
            absolute = std::max(absolute, error);
            relative = std::max(relative, error / (std::abs(expected[index]) + 1e-10));
        }
    }
    double errors[] = {absolute, relative};
    CHECK(MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (!g.rank)
        std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
    return errors[1] <= 1e-6 && std::isfinite(errors[0]);
}

void usage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n"
                "  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int run(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool verify = false, report = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end;
            long size = std::strtol(argv[++i], &end, 10);
            if (*end || size < 0 || size >= INT_MAX) {
                if (!rank) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = size;
        } else if (!std::strcmp(argv[i], "-v")) verify = true;
        else if (!std::strcmp(argv[i], "-r")) report = true;
        else if (!std::strcmp(argv[i], "-h")) {
            if (!rank) usage(argv[0]);
            return 0;
        } else {
            if (!rank) { std::printf("Unknown option: %s\n", argv[i]); usage(argv[0]); }
            return 1;
        }
    }

    MPI_Comm node;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank, devices;
    CHECK(MPI_Comm_rank(node, &localRank));
    CHECK(MPI_Comm_free(&node));
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { std::fprintf(stderr, "Rank %d has no visible CUDA device\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CHECK(cudaSetDevice(localRank % devices));
    if (!rank)
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n"
                    "Generating positive definite matrix...\n", n, n, verify ? "enabled" : "disabled");
    if (!n) {
        if (!rank) {
            std::puts("Computing Cholesky decomposition...\nComputation time: 0 ms\nPerformance: 0.000 GFLOPS");
            if (report) print_results(std::vector<double>{}, "CholeskyL");
            if (verify) std::puts("Validating result...\nMax absolute error: 0.0000000000e+00\n"
                                  "Max relative error: 0.0000000000e+00\nValidation: PASSED");
        }
        return 0;
    }

    Grid grid(n, rank, ranks);
    Panels panels(grid);
    size_t elements = size_t(n) * grid.cols;
    Device<double> a(elements), original(verify ? elements : 0);
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));
    generate(grid, panels, blas, a);
    if (verify && elements)
        CHECK(cudaMemcpy(original.p, a.p, elements * sizeof(double), cudaMemcpyDeviceToDevice));
    if (!rank) std::puts("Computing Cholesky decomposition...");
    double elapsed = 0.;
    int bad = factor(grid, panels, blas, solver, a, elapsed);
    bool success = bad == INT_MAX;
    if (!success && !rank)
        std::printf("Error: Matrix is not positive definite at diagonal element %d\nCholesky decomposition failed\n", bad);
    if (success && !rank)
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", elapsed * 1e3,
                    double(n) * n * n / (3.e9 * elapsed));
    if (success && (report || verify)) {
        auto lower = lowerFactor(grid, a);
        if (report) results(grid, lower);
        if (verify) {
            if (!rank) std::puts("Validating result...");
            success = validate(grid, panels, blas, a, lower, original);
            if (!rank) std::puts(success ? "Validation: PASSED" :
                                 "Validation failed: relative error too large\nValidation: FAILED");
        }
    }
    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    return success ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided, rank, ranks;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    CHECK(provided < MPI_THREAD_FUNNELED);
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
