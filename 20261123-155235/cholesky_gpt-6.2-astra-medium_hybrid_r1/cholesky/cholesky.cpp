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
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
int rank, ranks;
constexpr double one = 1, zero = 0, minusOne = -1;

[[noreturn]] void fail(const char* operation, int status) {
    std::fprintf(stderr, "Rank %d: %s failed (status %d)\n", rank, operation, status);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

// CUDA, cuBLAS and cuSOLVER all use zero for success.
#define CHECK(call) do { int status = (call); if (status) fail(#call, status); } while (0)

template<class T>
struct Device {
    T* data = nullptr;
    explicit Device(size_t count) {
        if (count) CHECK(cudaMalloc(&data, count * sizeof(T)));
    }
    Device(const Device&) = delete;
    ~Device() { if (data) cudaFree(data); }
};

struct Layout {
    int n, width, blocks, columns;
    explicit Layout(int size) : n(size) {
        // Keep small problems distributed while allowing BLAS-3 sized panels.
        int maxWidth = n >= 4096 ? 256 : 128;
        int balancedWidth = ((size - 1) / ranks / 32 + 1) * 32;
        width = ranks == 1 ? std::max(1, n) : std::clamp(balancedWidth, 32, maxWidth);
        blocks = n ? 1 + (n - 1) / width : 0;
        columns = 0;
        for (int b = rank; b < blocks; b += ranks) columns += count(b);
    }
    int first(int b) const { return b * width; }
    int count(int b) const { return std::min(width, n - first(b)); }
    size_t offset(int b) const { return size_t(b / ranks) * width * n; }
    int globalColumn(int col) const {
        return (col / width * ranks + rank) * width + col % width;
    }
    size_t entries() const { return size_t(n) * columns; }
};

// Prefer device buffers when the MPI runtime advertises CUDA support.
struct Exchange {
    bool deviceTransport = false;
    double* host = nullptr;
    Device<double> remote;
    explicit Exchange(size_t capacity) : remote(ranks > 1 ? capacity : 0) {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        deviceTransport = MPIX_Query_cuda_support() != 0;
#endif
        if (!deviceTransport && ranks > 1 && capacity) CHECK(cudaMallocHost(&host, capacity * sizeof(double)));
    }
    ~Exchange() { if (host) cudaFreeHost(host); }

    const double* broadcast(const double* local, int rows, int cols, int stride, int owner) {
        if (ranks == 1) return local;
        size_t count = size_t(rows) * cols;
        double* buffer = deviceTransport ? remote.data : host;
        if (rank == owner) {
            CHECK(cudaMemcpy2D(buffer, rows * sizeof(double), local, stride * sizeof(double),
                               rows * sizeof(double), cols,
                               deviceTransport ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));
        }
        // MPI must see completed device writes and may reuse a receive buffer
        // only after the previous BLAS operation has finished reading it.
        if (deviceTransport) CHECK(cudaStreamSynchronize(nullptr));
        for (size_t pos = 0; pos < count;) {
            int chunk = int(std::min<size_t>(INT_MAX, count - pos));
            MPI_Bcast(buffer + pos, chunk, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            pos += chunk;
        }
        if (rank == owner) return local;
        if (!deviceTransport)
            CHECK(cudaMemcpy(remote.data, host, count * sizeof(double), cudaMemcpyHostToDevice));
        return remote.data;
    }
};

#ifdef __GLIBC__
unsigned int rowSeed(size_t offset) {
    unsigned int seed = 42;
    // glibc rand_r advances its 32-bit LCG three times per sample.
    // Exponentiate the affine map to start each row at its serial seed.
    unsigned int mul = 1103515245u, add = 12345u;
    for (size_t steps = 3 * offset; steps; steps >>= 1) {
        if (steps & 1) seed = mul * seed + add;
        add *= mul + 1;
        mul *= mul;
    }
    return seed;
}
#endif

void generate(const Layout& d, cublasHandle_t blas, Exchange& exchange, double* a) {
    std::vector<double> host(d.entries());
#ifndef __GLIBC__
    // On other C libraries rand_r may use a different state transition.
    std::vector<unsigned int> seeds(d.columns);
    unsigned int serialSeed = 42;
    int localCol = 0;
    for (int row = 0; localCol < d.columns; ++row) {
        if (row == d.globalColumn(localCol)) seeds[localCol++] = serialSeed;
        for (int k = 0; k < d.n; ++k) rand_r(&serialSeed);
    }
#endif
    #pragma omp parallel for schedule(static) if(d.entries() >= 16384)
    for (int col = 0; col < d.columns; ++col) {
#ifdef __GLIBC__
        unsigned int seed = rowSeed(size_t(d.globalColumn(col)) * d.n);
#else
        unsigned int seed = seeds[col];
#endif
        for (int k = 0; k < d.n; ++k)
            host[size_t(col) * d.n + k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    Device<double> random(d.entries());
    if (d.columns)
        CHECK(cudaMemcpy(random.data, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    // Each column of random is one row of the original row-major B.
    for (int b = 0; b < d.blocks; ++b) {
        int owner = b % ranks, w = d.count(b);
        const double* panel = exchange.broadcast(rank == owner ? random.data + d.offset(b) : nullptr,
                                                  d.n, w, d.n, owner);
        // Only local block columns at or to the left of this row are needed.
        int used = b / ranks * d.width;
        if (rank < owner) used += d.width;
        if (rank == owner) used += w;
        if (ranks == 1)
            CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, d.n, d.n,
                              &one, panel, d.n, &zero, a, d.n));
        else if (used)
            CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, w, used, d.n,
                              &one, panel, d.n, random.data, d.n, &zero, a + d.first(b), d.n));
    }
    if (d.columns) {
        std::fill_n(host.data(), std::min(d.width, d.columns), double(d.n));
        CHECK(cudaMemcpy(random.data, host.data(), std::min(d.width, d.columns) * sizeof(double),
                         cudaMemcpyHostToDevice));
        for (int b = rank; b < d.blocks; b += ranks)
            CHECK(cublasDaxpy(blas, d.count(b), &one, random.data, 1,
                              a + d.offset(b) + d.first(b), d.n + 1));
    }
}

// Update the part on and below each owned block diagonal. Panel points at
// the first active row. Remote panels have a packed leading dimension.
void update(const Layout& d, cublasHandle_t blas, int firstBlock, int panelRow,
            int w, const double* panel, int stride, double scale, double* target) {
    int b = firstBlock + (rank - firstBlock % ranks + ranks) % ranks;
    for (; b < d.blocks; b += ranks) {
        int row = d.first(b);
        const double* slice = panel + (row - panelRow);
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, d.n - row, d.count(b), w,
                          &scale, slice, stride, slice, stride, &one, target + d.offset(b) + row, d.n));
    }
}

int factor(const Layout& d, cublasHandle_t blas, cusolverDnHandle_t solver,
           Exchange& exchange, double* a, double* work, int workSize, int* info) {
    for (int b = 0; b < d.blocks; ++b) {
        int row = d.first(b), w = d.count(b), owner = b % ranks;
        double* diagonal = nullptr;
        int pivot = 0;
        if (rank == owner) {
            diagonal = a + d.offset(b) + row;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, w, diagonal, d.n,
                                    work, workSize, info));
            CHECK(cudaMemcpy(&pivot, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!pivot && row + w < d.n)
                CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                  CUBLAS_DIAG_NON_UNIT, d.n - row - w, w, &one, diagonal,
                                  d.n, diagonal + w, d.n));
        }
        MPI_Bcast(&pivot, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (pivot) return row + pivot;
        if (b + 1 == d.blocks) break;
        // Send only the active rectangle below the diagonal tile.
        const double* panel = exchange.broadcast(diagonal ? diagonal + w : nullptr,
                                                  d.n - row - w, w, d.n, owner);
        update(d, blas, b + 1, row + w, w, panel, rank == owner ? d.n : d.n - row - w, minusOne, a);
    }
    CHECK(cudaDeviceSynchronize());
    return 0;
}

void printFactor(const Layout& d, const std::vector<double>& local) {
    std::vector<double> result(rank == 0 ? size_t(d.n) * d.n : 0);
    std::vector<double> buffer(rank == 0 && ranks > 1 ? size_t(std::min(d.width, d.n)) * d.n : 0);
    for (int b = 0; b < d.blocks; ++b) {
        int owner = b % ranks, w = d.count(b);
        const double* src = rank == owner ? local.data() + d.offset(b) : nullptr;
        if (owner && (rank == owner || rank == 0)) {
            size_t total = size_t(w) * d.n;
            for (size_t pos = 0; pos < total;) {
                int chunk = int(std::min<size_t>(INT_MAX, total - pos));
                if (rank == owner) MPI_Send(src + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                else MPI_Recv(buffer.data() + pos, chunk, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pos += chunk;
            }
            if (!rank) src = buffer.data();
        }
        if (!rank) {
            #pragma omp parallel for schedule(static) if(size_t(w) * d.n >= 16384)
            for (int i = 0; i < d.n; ++i)
                for (int j = 0; j < w; ++j)
                    result[size_t(i) * d.n + d.first(b) + j] = src[size_t(j) * d.n + i];
        }
    }
    if (!rank) print_results(result, "CholeskyL");
}

bool validate(const Layout& d, cublasHandle_t blas, Exchange& exchange, double* a,
              std::vector<double>& local, const std::vector<double>& original) {
    Device<double> product(d.entries());
    if (d.columns) {
        CHECK(cudaMemset(product.data, 0, d.entries() * sizeof(double)));
        CHECK(cudaMemcpy(a, local.data(), d.entries() * sizeof(double), cudaMemcpyHostToDevice));
    }
    for (int b = 0; b < d.blocks; ++b) {
        int w = d.count(b), owner = b % ranks, row = d.first(b);
        const double* panel = exchange.broadcast(rank == owner ? a + d.offset(b) + row : nullptr,
                                                  d.n - row, w, d.n, owner);
        // Columns preceding this panel have zero entries in its rows of L^T.
        update(d, blas, b, row, w, panel, rank == owner ? d.n : d.n - row, one, product.data);
    }
    if (d.columns)
        CHECK(cudaMemcpy(local.data(), product.data, d.entries() * sizeof(double), cudaMemcpyDeviceToHost));
    double absErr = 0, relErr = 0;
    #pragma omp parallel for reduction(max:absErr,relErr) if(d.entries() >= 16384)
    for (int col = 0; col < d.columns; ++col) {
        int global = d.globalColumn(col);
        for (int i = global; i < d.n; ++i) {
            size_t idx = size_t(col) * d.n + i;
            double error = std::abs(local[idx] - original[idx]);
            if (!std::isfinite(error)) error = HUGE_VAL;
            absErr = std::max(absErr, error);
            relErr = std::max(relErr, error / (std::abs(original[idx]) + 1e-10));
        }
    }
    double errors[] = {absErr, relErr};
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
        printf("Validation: %s\n", errors[1] <= 1e-6 ? "PASSED" : "FAILED");
    }
    return errors[1] <= 1e-6;
}

int run(int argc, char** argv) {
    int n = 512;
    bool verify = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || value < 0 || value >= INT_MAX ||
                size_t(value) > SIZE_MAX / sizeof(double) / std::max(1L, value)) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = int(value);
        } else if (!strcmp(argv[i], "-v")) verify = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else {
            bool help = !strcmp(argv[i], "-h");
            if (!rank) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printf("Usage: %s [options]\nOptions:\n"
                       "  -n <num>     Matrix size (default: 512)\n"
                       "  -v           Enable validation\n"
                       "  -r           Print results for external validation\n"
                       "  -h           Show this help message\n", argv[0]);
            }
            return help ? 0 : 1;
        }
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) fail("no visible CUDA device", 1);
    CHECK(cudaSetDevice(localRank % devices));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));

    Layout d(n);
    Device<double> a(d.entries());
    Exchange exchange(size_t(std::min(n, d.width)) * n);
    if (!rank) printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n"
                      "Generating positive definite matrix...\n", n, n, verify ? "enabled" : "disabled");
    generate(d, blas, exchange, a.data);
    std::vector<double> original(verify ? d.entries() : 0);
    if (!original.empty())
        CHECK(cudaMemcpy(original.data(), a.data, original.size() * sizeof(double), cudaMemcpyDeviceToHost));

    int workSize = 0;
    if (d.columns) CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                          std::min(n, d.width), a.data, n, &workSize));
    Device<double> work(workSize);
    Device<int> info(1);
    CHECK(cudaDeviceSynchronize());
    if (!rank) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int pivot = factor(d, blas, solver, exchange, a.data, work.data, workSize, info.data);
    double elapsed = MPI_Wtime() - start, duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        if (pivot) printf("Error: Matrix is not positive definite at diagonal element %d\n"
                          "Cholesky decomposition failed\n", pivot - 1);
        else printf("Computation time: %.0f ms\nPerformance: %.3f GFLOPS\n", duration * 1000,
                    duration ? double(n) * n * n / (3e9 * duration) : 0);
    }
    bool success = !pivot;
    if (success && (verify || results)) {
        std::vector<double> local(d.entries());
        if (d.columns)
            CHECK(cudaMemcpy(local.data(), a.data, d.entries() * sizeof(double), cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static) if(d.entries() >= 16384)
        for (int col = 0; col < d.columns; ++col) {
            int global = d.globalColumn(col);
            std::fill_n(local.data() + size_t(col) * n, global, 0.0);
        }
        if (results) printFactor(d, local);
        if (verify) {
            if (!rank) printf("Validating result...\n");
            success = validate(d, blas, exchange, a.data, local, original);
        }
    }
    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    return success ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED unavailable", provided);
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& e) {
        fail(e.what(), 1);
    }
    MPI_Finalize();
    return status;
}
