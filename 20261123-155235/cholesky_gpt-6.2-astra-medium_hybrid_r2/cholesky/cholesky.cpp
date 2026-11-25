#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
constexpr int block = 128;
constexpr double one = 1.0, zero = 0.0, minusOne = -1.0;
int rank, ranks;

void check(int status, const char* call) {
    if (status != 0) {
        std::fprintf(stderr, "Rank %d: %s returned %d\n", rank, call, status);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CHECK(call) check((call), #call)

template<class T>
struct Device {
    T* p = nullptr;
    explicit Device(size_t count) {
        if (count) CHECK(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
};

struct Staging {
    double* p;
    explicit Staging(size_t count) { CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), count * sizeof(double))); }
    ~Staging() { cudaFreeHost(p); }
};

struct Gpu {
    cudaStream_t panel, update;
    cublasHandle_t solve, gemm;
    cusolverDnHandle_t solver;
    Gpu() {
        CHECK(cudaStreamCreateWithFlags(&panel, cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&update, cudaStreamNonBlocking));
        CHECK(cublasCreate(&solve));
        CHECK(cublasCreate(&gemm));
        CHECK(cusolverDnCreate(&solver));
        CHECK(cublasSetStream(solve, panel));
        CHECK(cublasSetStream(gemm, update));
        CHECK(cusolverDnSetStream(solver, panel));
    }
    ~Gpu() {
        cusolverDnDestroy(solver);
        cublasDestroy(gemm);
        cublasDestroy(solve);
        cudaStreamDestroy(update);
        cudaStreamDestroy(panel);
    }
};

// Each block column is contiguous and column-major locally. Block t belongs
// to rank t % ranks. Only the lower triangle of the Gram matrix is needed.
struct Layout {
    int n, tiles, cols;
    explicit Layout(int size) : n(size), tiles(n ? 1 + (n - 1) / block : 0),
                                cols(localColumns(tiles)) {}
    int width(int t) const { return std::min(block, n - t * block); }
    int localColumns(int end) const {
        if (end <= rank) return 0;
        int last = rank + ((end - 1 - rank) / ranks) * ranks;
        return (last / ranks) * block + width(last);
    }
    int globalColumn(int c) const {
        return ((c / block) * ranks + rank) * block + c % block;
    }
    int firstOwned(int t) const {
        return t <= rank ? rank : rank + (1 + (t - rank - 1) / ranks) * ranks;
    }
    int hostThreads() const {
        return std::min(omp_get_max_threads(), std::max(1, (cols + 15) / 16));
    }
    size_t offset(int t) const { return size_t(t / ranks) * block * n; }
    size_t elements() const { return size_t(cols) * n; }
};

void broadcast(double* data, size_t count, int root) {
    while (count) {
        int part = static_cast<int>(std::min<size_t>(count, INT_MAX));
        CHECK(MPI_Bcast(data, part, MPI_DOUBLE, root, MPI_COMM_WORLD));
        data += part;
        count -= part;
    }
}

#ifdef __GLIBC__
// glibc rand_r advances its unsigned 32-bit LCG three times per sample.
// Exponentiate the affine map so independent rows start at the same seeds
// as in the original single random stream.
unsigned rowSeed(uint64_t sample) {
    uint32_t seed = 42, a = 1103515245u, b = 12345u;
    for (uint64_t steps = sample * 3; steps; steps >>= 1) {
        if (steps & 1) seed = a * seed + b;
        b *= a + 1;
        a *= a;
    }
    return seed;
}
#endif

void generate(const Layout& l, Gpu& gpu, double* a, Staging& host) {
    std::vector<double> random(l.elements());
#ifndef __GLIBC__
    std::vector<unsigned> seeds(l.n);
    unsigned stream = 42;
    for (int i = 0; i < l.n; ++i) {
        seeds[i] = stream;
        for (int j = 0; j < l.n; ++j) rand_r(&stream);
    }
#endif
    #pragma omp parallel for num_threads(l.hostThreads())
    for (int c = 0; c < l.cols; ++c) {
        int row = l.globalColumn(c);
#ifdef __GLIBC__
        unsigned seed = rowSeed(uint64_t(row) * l.n);
#else
        unsigned seed = seeds[row];
#endif
        for (int j = 0; j < l.n; ++j)
            random[size_t(c) * l.n + j] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    Device<double> local(l.elements()), incoming(size_t(l.n) * std::min(block, l.n));
    if (l.cols)
        CHECK(cudaMemcpyAsync(local.p, random.data(), l.elements() * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.update));
    if (ranks == 1)
        CHECK(cublasDsyrk(gpu.gemm, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, l.n, l.n,
                          &one, local.p, l.n, &zero, a, l.n));
    for (int t = 0; ranks > 1 && t < l.tiles; ++t) {
        int owner = t % ranks, w = l.width(t), cols = l.localColumns(t + 1);
        size_t count = size_t(w) * l.n;
        const double* rows = incoming.p;
        if (rank == owner) {
            rows = local.p + l.offset(t);
            std::copy_n(random.data() + l.offset(t), count, host.p);
        }
        broadcast(host.p, count, owner);
        if (!cols) continue;
        if (rank != owner)
            CHECK(cudaMemcpyAsync(incoming.p, host.p, count * sizeof(double), cudaMemcpyHostToDevice, gpu.update));
        CHECK(cublasDgemm(gpu.gemm, CUBLAS_OP_T, CUBLAS_OP_N, w, cols, l.n,
                          &one, rows, l.n, local.p, l.n, &zero, a + t * block, l.n));
        if (rank != owner) CHECK(cudaStreamSynchronize(gpu.update));
    }
    // Add n to each diagonal. A strided BLAS update touches only the diagonal.
    std::vector<double> shift(std::min(block, l.n), double(l.n));
    CHECK(cudaMemcpyAsync(incoming.p, shift.data(), shift.size() * sizeof(double),
                          cudaMemcpyHostToDevice, gpu.update));
    for (int t = rank; t < l.tiles; t += ranks)
        CHECK(cublasDaxpy(gpu.gemm, l.width(t), &one, incoming.p, 1,
                          a + l.offset(t) + t * block, l.n + 1));
    CHECK(cudaStreamSynchronize(gpu.update));
}

void updateBlock(const Layout& l, Gpu& gpu, int t, int dest, const double* panel,
                 int ld, double* matrix, const double* scale) {
    int row = dest * block, offset = row - t * block;
    CHECK(cublasDgemm(gpu.gemm, CUBLAS_OP_N, CUBLAS_OP_T, l.n - row, l.width(dest),
                      l.width(t), scale, panel + offset, ld, panel + offset, ld,
                      &one, matrix + l.offset(dest) + row, l.n));
}

int factor(const Layout& l, Gpu& gpu, double* a, Staging& first, Staging& second) {
    int workSize;
    int diagSize = ranks == 1 ? l.n : std::min(block, l.n);
    CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, diagSize,
                                      a, l.n, &workSize));
    Device<double> work(workSize);
    Device<int> info(1);
    if (ranks == 1) {
        CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, l.n, a, l.n,
                               work.p, workSize, info.p));
        int result;
        CHECK(cudaMemcpyAsync(&result, info.p, sizeof(int), cudaMemcpyDeviceToHost, gpu.panel));
        CHECK(cudaStreamSynchronize(gpu.panel));
        return result;
    }

    Device<double> incoming(size_t(l.n) * diagSize);
    cudaEvent_t nextReady, copied[2];
    CHECK(cudaEventCreateWithFlags(&nextReady, cudaEventDisableTiming));
    for (auto& e : copied) CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    int result = 0;
    for (int t = 0; t < l.tiles; ++t) {
        int owner = t % ranks, k = t * block, h = l.n - k, w = l.width(t);
        int slot = t % 2;
        double* host = slot ? second.p : first.p;
        CHECK(cudaEventSynchronize(copied[slot]));
        const double* panel = incoming.p;
        int ld = h;
        if (rank == owner) {
            double* diag = a + l.offset(t) + k;
            CHECK(cudaStreamWaitEvent(gpu.panel, nextReady, 0));
            CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, w, diag, l.n,
                                   work.p, workSize, info.p));
            if (h > w)
                CHECK(cublasDtrsm(gpu.solve, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, h - w, w, &one,
                                  diag, l.n, diag + w, l.n));
            CHECK(cudaMemcpyAsync(host, info.p, sizeof(int), cudaMemcpyDeviceToHost, gpu.panel));
            CHECK(cudaMemcpy2DAsync(host + 1, size_t(h) * sizeof(double), diag,
                                    size_t(l.n) * sizeof(double), size_t(h) * sizeof(double), w,
                                    cudaMemcpyDeviceToHost, gpu.panel));
            CHECK(cudaStreamSynchronize(gpu.panel));
            std::memcpy(&result, host, sizeof(result));
            host[0] = result;
            panel = diag;
            ld = l.n;
        }
        size_t count = size_t(h) * w;
        broadcast(host, count + 1, owner);
        result = static_cast<int>(host[0]);
        if (result) {
            if (result > 0) result += k;
            break;
        }
        int dest = l.firstOwned(t + 1);
        if (dest >= l.tiles) continue;
        if (rank != owner) {
            CHECK(cudaMemcpyAsync(incoming.p, host + 1, count * sizeof(double),
                                  cudaMemcpyHostToDevice, gpu.update));
            CHECK(cudaEventRecord(copied[slot], gpu.update));
        }
        for (; dest < l.tiles; dest += ranks) {
            updateBlock(l, gpu, t, dest, panel, ld, a, &minusOne);
            if (dest == t + 1) CHECK(cudaEventRecord(nextReady, gpu.update));
        }
    }
    CHECK(cudaStreamSynchronize(gpu.update));
    for (auto e : copied) CHECK(cudaEventDestroy(e));
    CHECK(cudaEventDestroy(nextReady));
    return result;
}

void output(const Layout& l, const std::vector<double>& local, Staging& host) {
    std::vector<double> full;
    if (rank == 0) full.resize(size_t(l.n) * l.n);
    for (int t = 0; t < l.tiles; ++t) {
        int owner = t % ranks, w = l.width(t);
        if (rank != owner && rank != 0) continue;
        const double* src = host.p;
        if (rank == owner) src = local.data() + l.offset(t);
        if (owner != 0) {
            size_t count = size_t(l.n) * w;
            for (size_t pos = 0; pos < count;) {
                int part = static_cast<int>(std::min<size_t>(count - pos, INT_MAX));
                if (rank == owner)
                    CHECK(MPI_Send(src + pos, part, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                else
                    CHECK(MPI_Recv(host.p + pos, part, MPI_DOUBLE, owner, 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                pos += part;
            }
        }
        if (rank == 0) {
            #pragma omp parallel for num_threads(std::min(omp_get_max_threads(), 1 + l.n / 256))
            for (int i = 0; i < l.n; ++i)
                for (int j = 0; j < w; ++j)
                    full[size_t(i) * l.n + t * block + j] = src[size_t(j) * l.n + i];
        }
    }
    if (!rank) print_results(full, "CholeskyL");
}

bool validate(const Layout& l, Gpu& gpu, const std::vector<double>& local,
              const std::vector<double>& original, double* reconstructed, Staging& host) {
    Device<double> panel(size_t(l.n) * std::min(block, l.n));
    if (l.cols)
        CHECK(cudaMemsetAsync(reconstructed, 0, l.elements() * sizeof(double), gpu.update));
    for (int t = 0; t < l.tiles; ++t) {
        int owner = t % ranks, k = t * block, h = l.n - k, w = l.width(t);
        if (rank == owner) {
            const double* src = local.data() + l.offset(t) + k;
            for (int j = 0; j < w; ++j)
                std::copy_n(src + size_t(j) * l.n, h, host.p + size_t(j) * h);
        }
        size_t count = size_t(h) * w;
        if (ranks > 1) broadcast(host.p, count, owner);
        CHECK(cudaMemcpyAsync(panel.p, host.p, count * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.update));
        int dest = l.firstOwned(t);
        for (; dest < l.tiles; dest += ranks)
            updateBlock(l, gpu, t, dest, panel.p, h, reconstructed, &one);
        CHECK(cudaStreamSynchronize(gpu.update));
    }
    std::vector<double> product(l.elements());
    if (l.cols)
        CHECK(cudaMemcpy(product.data(), reconstructed, l.elements() * sizeof(double),
                         cudaMemcpyDeviceToHost));
    double absolute = 0, relative = 0;
    #pragma omp parallel for reduction(max:absolute,relative) num_threads(l.hostThreads())
    for (int c = 0; c < l.cols; ++c) {
        int col = l.globalColumn(c);
        for (int i = col; i < l.n; ++i) {
            size_t idx = size_t(c) * l.n + i;
            double error = std::abs(product[idx] - original[idx]);
            if (!std::isfinite(error)) error = HUGE_VAL;
            absolute = std::max(absolute, error);
            relative = std::max(relative, error / (std::abs(original[idx]) + 1e-10));
        }
    }
    double errors[] = {absolute, relative};
    CHECK(MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (!rank) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
        if (errors[1] > 1e-6) puts("Validation failed: relative error too large");
    }
    return errors[1] <= 1e-6;
}

void usage(const char* prog) {
    printf("Usage: %s [options]\nOptions:\n", prog);
    puts("  -n <num>     Matrix size (default: 512)\n"
         "  -v           Enable validation\n"
         "  -r           Print results for external validation\n"
         "  -h           Show this help message");
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
                size_t(value) * value > SIZE_MAX / sizeof(double)) {
                if (!rank) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) verify = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) usage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); }
            return 1;
        }
    }
    MPI_Comm node;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int nodeRank, devices;
    CHECK(MPI_Comm_rank(node, &nodeRank));
    CHECK(MPI_Comm_free(&node));
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        std::fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(cudaSetDevice(nodeRank % devices));
    if (!rank) {
        puts("Cholesky Decomposition Benchmark");
        printf("Matrix size: %d x %d\nValidation: %s\n", n, n, verify ? "enabled" : "disabled");
        puts("Generating positive definite matrix...");
    }
    Layout l(n);
    Gpu gpu;
    Device<double> a(std::max<size_t>(1, l.elements()));
    size_t panelSize = size_t(n) * std::min(n, block) + 1;
    Staging host(panelSize), second(ranks > 1 ? panelSize : 1);
    if (n) generate(l, gpu, a.p, host);
    std::vector<double> original;
    if (verify && l.cols) {
        original.resize(l.elements());
        CHECK(cudaMemcpy(original.data(), a.p, l.elements() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (!rank) puts("Computing Cholesky decomposition...");
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    int info = n ? factor(l, gpu, a.p, host, second) : 0;
    double elapsed = MPI_Wtime() - start, seconds;
    CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (info) {
        if (!rank) {
            if (info > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
            puts("Cholesky decomposition failed");
        }
        return 1;
    }
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? double(n) * n * n / (3e9 * seconds) : 0);
    }
    if (results || verify) {
        std::vector<double> local(l.elements());
        if (l.cols)
            CHECK(cudaMemcpy(local.data(), a.p, l.elements() * sizeof(double), cudaMemcpyDeviceToHost));
        #pragma omp parallel for num_threads(l.hostThreads())
        for (int c = 0; c < l.cols; ++c) {
            int col = l.globalColumn(c);
            std::fill_n(local.data() + size_t(c) * n, col, 0.0);
        }
        if (results) output(l, local, host);
        if (verify) {
            if (!rank) puts("Validating result...");
            bool ok = validate(l, gpu, local, original, a.p, host);
            if (!rank) printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            return ok ? 0 : 1;
        }
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
