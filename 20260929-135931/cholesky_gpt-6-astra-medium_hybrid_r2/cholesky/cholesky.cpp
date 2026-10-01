#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

namespace {
int rankId;
constexpr int blockSize = 256;
void check(int status, const char* expression, int line) {
    if (status) {
        fprintf(stderr, "Rank %d: %s failed (%d), line %d\n", rankId, expression, status, line);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}
#define CHECK(x) check(static_cast<int>(x), #x, __LINE__)
struct Device {
    double* p = nullptr;
    explicit Device(size_t count) { CHECK(cudaMalloc(&p, std::max(size_t(1), count) * sizeof(double))); }
    ~Device() { cudaFree(p); }
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) { CHECK(cudaMallocHost(&p, std::max(size_t(1), count) * sizeof(double))); }
    ~Pinned() { cudaFreeHost(p); }
};
void broadcast(double* p, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        CHECK(MPI_Bcast(p, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD));
        p += chunk; count -= chunk;
    }
}
// rand_r on glibc advances this 32-bit LCG three times per sample.
// Jump to each row independently without changing the original seed-42 sequence.
unsigned rowSeed(uint64_t samples) {
    uint32_t a = 1103515245u, c = 12345u, seed = 42;
    for (uint64_t steps = 3 * samples; steps; steps >>= 1) {
        if (steps & 1) seed = a * seed + c;
        c *= a + 1; a *= a;
    }
    return seed;
}
void randomRows(double* out, const std::vector<int>& rows, int n) {
    // Other C libraries need not use glibc's rand_r recurrence.
    #if !defined(__GLIBC__)
    std::vector<unsigned> seeds(rows.size());
    unsigned seed = 42;
    uint64_t position = 0;
    for (size_t j = 0; j < rows.size(); ++j) {
        uint64_t target = uint64_t(rows[j]) * n;
        while (position < target) { rand_r(&seed); ++position; }
        seeds[j] = seed;
    }
    #endif
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < rows.size(); ++j) {
        #if defined(__GLIBC__)
        unsigned seed = rowSeed(uint64_t(rows[j]) * n);
        #else
        unsigned seed = seeds[j];
        #endif
        for (int i = 0; i < n; ++i)
            out[j * n + i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
}
void generate(Device& a, int n, const std::vector<int>& columns, cublasHandle_t blas) {
    int nc = static_cast<int>(columns.size());
    if (!nc) return;
    Device selected(size_t(n) * nc), chunk(size_t(n) * blockSize);
    Pinned host(std::max(size_t(n) * nc, size_t(n) * blockSize));
    randomRows(host.p, columns, n);
    CHECK(cudaMemcpy(selected.p, host.p, size_t(n) * nc * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1, zero = 0;
    for (int first = 0; first < n; first += blockSize) {
        int b = std::min(blockSize, n - first);
        std::vector<int> rows(b);
        for (int i = 0; i < b; ++i) rows[i] = first + i;
        randomRows(host.p, rows, n);
        CHECK(cudaMemcpy(chunk.p, host.p, size_t(n) * b * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, b, nc, n,
                         &one, chunk.p, n, selected.p, n, &zero, a.p + first, n));
    }
    CHECK(cudaMemcpy(host.p, a.p, size_t(n) * nc * sizeof(double), cudaMemcpyDeviceToHost));
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nc; ++j) host.p[size_t(j) * n + columns[j]] += n;
    CHECK(cudaMemcpy(a.p, host.p, size_t(n) * nc * sizeof(double), cudaMemcpyHostToDevice));
}

// Gather only for requested validation/output; the timed factorization stays distributed.
std::vector<double> gather(const Device& a, int n, int ranks, bool lower) {
    std::vector<double> result(rankId == 0 ? size_t(n) * n : 0);
    Pinned tile(size_t(n) * blockSize);
    for (int k = 0, t = 0; k < n; k += blockSize, ++t) {
        int owner = t % ranks, b = std::min(blockSize, n - k);
        if (rankId == owner)
            CHECK(cudaMemcpy(tile.p, a.p + size_t(t / ranks) * blockSize * n,
                             size_t(n) * b * sizeof(double), cudaMemcpyDeviceToHost));
        // Point-to-point chunks avoid MPI's int count limit.
        if (owner != 0) {
            for (size_t pos = 0; pos < size_t(n) * b;) {
                int count = static_cast<int>(std::min(size_t(INT_MAX), size_t(n) * b - pos));
                if (rankId == owner) CHECK(MPI_Send(tile.p + pos, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                if (rankId == 0) CHECK(MPI_Recv(tile.p + pos, count, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                pos += count;
            }
        }
        if (rankId == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < b; ++j)
                    result[size_t(i) * n + k + j] = lower && i < k + j ? 0 : tile.p[size_t(j) * n + i];
        }
    }
    return result;
}
bool factor(Device& a, int n, int ranks, const std::vector<int>& cols,
            cublasHandle_t blas, cusolverDnHandle_t solver) {
    Device panel(size_t(n) * blockSize), right(size_t(cols.size()) * blockSize);
    Pinned host(size_t(n) * blockSize), packed(size_t(cols.size()) * blockSize);
    int workSize = 0;
    CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
          std::min(n, blockSize), panel.p, n, &workSize));
    Device work(workSize);
    int* info = nullptr;
    CHECK(cudaMalloc(&info, sizeof(int)));
    const double one = 1, minusOne = -1;
    for (int k = 0, t = 0; k < n; k += blockSize, ++t) {
        int b = std::min(blockSize, n - k), m = n - k, owner = t % ranks;
        int status = 0;
        if (rankId == owner) {
            double* diagonal = a.p + size_t(t / ranks) * blockSize * n + k;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, diagonal, n, work.p, workSize, info));
            CHECK(cudaMemcpy(&status, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!status && m > b)
                CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                       CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m - b, b, &one,
                       diagonal, n, diagonal + b, n));
            CHECK(cudaMemcpy2D(host.p, size_t(m) * sizeof(double), diagonal, size_t(n) * sizeof(double),
                               size_t(m) * sizeof(double), b, cudaMemcpyDeviceToHost));
        }
        CHECK(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (status) {
            if (!rankId) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + status - 1);
            cudaFree(info); return false;
        }
        if (m == b) break;
        broadcast(host.p, size_t(m) * b, owner);
        int first = static_cast<int>(std::lower_bound(cols.begin(), cols.end(), k + b) - cols.begin());
        int remaining = static_cast<int>(cols.size()) - first;
        if (!remaining) continue;
        // All owned trailing columns are contiguous locally, so one large GEMM
        // updates them. Entries above the diagonal are scratch, never factors.
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < remaining; ++j)
            for (int s = 0; s < b; ++s)
                packed.p[size_t(j) * b + s] = host.p[size_t(s) * m + cols[first + j] - k];
        CHECK(cudaMemcpy(panel.p, host.p, size_t(m) * b * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(right.p, packed.p, size_t(remaining) * b * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m - b, remaining, b,
                         &minusOne, panel.p + b, m, right.p, b, &one,
                         a.p + size_t(first) * n + k + b, n));
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(info));
    return true;
}
bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original,
                      int n, cublasHandle_t blas) {
    Device dl(size_t(n) * n), reconstructed(size_t(n) * n);
    CHECK(cudaMemcpy(dl.p, l.data(), l.size() * sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1, zero = 0;
    // The row-major L is interpreted as column-major L^T.
    CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one,
                     dl.p, n, dl.p, n, &zero, reconstructed.p, n));
    std::vector<double> host(l.size());
    CHECK(cudaMemcpy(host.data(), reconstructed.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double maxError = 0, relError = 0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (size_t i = 0; i < host.size(); ++i) {
        double error = std::abs(host[i] - original[i]);
        if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::abs(original[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    return relError <= 1e-6;
}
void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n", name);
}
int run(int argc, char** argv) {
    int n = 512, ranks;
    bool validate = false, printResults = false;
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr; errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || !*argv[i] || *end || value <= 0 || value > INT_MAX - blockSize ||
                size_t(value) > SIZE_MAX / sizeof(double) / size_t(value)) {
                if (!rankId) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rankId) printUsage(argv[0]); return 0; }
        else { if (!rankId) printUsage(argv[0]); return 1; }
    }
    MPI_Comm local;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &local));
    int localRank, devices;
    CHECK(MPI_Comm_rank(local, &localRank));
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "A CUDA GPU is required on every rank.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CHECK(cudaSetDevice(localRank % devices));
    CHECK(MPI_Comm_free(&local));
    // Respect explicit thread settings; avoid excessive threading on small panels.
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_max_threads()));
    cublasHandle_t blas; cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas)); CHECK(cusolverDnCreate(&solver));
    CHECK(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    std::vector<int> columns;
    for (int k = 0, t = 0; k < n; k += blockSize, ++t)
        if (t % ranks == rankId)
            for (int j = k; j < std::min(n, k + blockSize); ++j) columns.push_back(j);
    Device a(size_t(n) * columns.size());
    if (!rankId) printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n"
                        "Generating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled");
    generate(a, n, columns, blas);
    std::vector<double> original;
    if (validate) original = gather(a, n, ranks, false);
    CHECK(cudaDeviceSynchronize());
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!rankId) printf("Computing Cholesky decomposition...\n");
    double start = MPI_Wtime();
    bool success = factor(a, n, ranks, columns, blas, solver);
    double elapsed = MPI_Wtime() - start, maximum;
    CHECK(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    int result = success ? 0 : 1;
    if (success) {
        if (!rankId) printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",
                            long(maximum * 1000), double(n) * n * n / (3e9 * maximum));
        std::vector<double> l;
        if (printResults || validate) l = gather(a, n, ranks, true);
        if (!rankId) {
            if (printResults) print_results(l, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(l, original, n, blas);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                result = valid ? 0 : 1;
            }
        }
    } else if (!rankId) printf("Cholesky decomposition failed\n");
    CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK(cusolverDnDestroy(solver)); CHECK(cublasDestroy(blas));
    return result;
}
} // namespace
int main(int argc, char** argv) {
    int provided;
    CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int result = 1;
    try { result = run(argc, argv); }
    catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rankId, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(MPI_Finalize());
    return result;
}
