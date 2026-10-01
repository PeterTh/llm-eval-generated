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

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 256;
int rankId = 0, ranks = 1;

void check(int status, const char* operation) {
    if (status != 0) {
        fprintf(stderr, "Rank %d: %s failed (status %d)\n", rankId, operation, status);
        MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
}
#define CHECK(call) check(static_cast<int>(call), #call)

// CUDA libraries are part of the required CUDA toolkit; no CPU fallback is used.
struct Accelerator {
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    Accelerator() {
        MPI_Comm local;
        CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId,
                                  MPI_INFO_NULL, &local));
        int localRank, devices;
        CHECK(MPI_Comm_rank(local, &localRank));
        CHECK(MPI_Comm_free(&local));
        CHECK(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("A CUDA GPU is required on every rank");
        CHECK(cudaSetDevice(localRank % devices));
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cusolverDnCreate(&solver));
        CHECK(cublasSetStream(blas, stream));
        CHECK(cusolverDnSetStream(solver, stream));
        CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    }
    ~Accelerator() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
    void sync() { CHECK(cudaStreamSynchronize(stream)); }
};

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(T)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) {
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    }
    ~Pinned() { cudaFreeHost(p); }
};

// Split large messages to avoid the MPI int-count limit.
void broadcast(double* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        CHECK(MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD));
        data += chunk;
        count -= chunk;
    }
}

struct Columns {
    int n, tiles;
    size_t count = 0;
    std::vector<size_t> offset;
    explicit Columns(int size) : n(size), tiles((size - 1) / blockSize + 1), offset(tiles) {
        for (int t = rankId; t < tiles; t += ranks) {
            offset[t] = count;
            count += size_t(n) * width(t);
        }
    }
    int width(int t) const { return std::min(blockSize, n - t * blockSize); }
};

// Each rank generates the identical random sequence used by the original program.
// Only its own columns of B B^T are computed and retained on the GPU.
void generate(Accelerator& gpu, const Columns& cols, Device<double>& a,
              std::vector<double>& original, bool validate) {
    const int n = cols.n;
    std::vector<double> b(size_t(n) * n);
    unsigned seed = 42;
    for (double& x : b) x = rand_r(&seed) / double(RAND_MAX) - 0.5;
    Device<double> db(b.size());
    CHECK(cudaMemcpyAsync(db.p, b.data(), b.size() * sizeof(double),
                          cudaMemcpyHostToDevice, gpu.stream));
    const double one = 1.0, zero = 0.0;
    for (int t = rankId; t < cols.tiles; t += ranks) {
        // Row-major B is interpreted as column-major B^T.
        CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, cols.width(t), n,
                          &one, db.p, n, db.p + size_t(t * blockSize) * n, n,
                          &zero, a.p + cols.offset[t], n));
    }
    std::vector<double> host(cols.count);
    CHECK(cudaMemcpyAsync(host.data(), a.p, cols.count * sizeof(double),
                          cudaMemcpyDeviceToHost, gpu.stream));
    gpu.sync();
    #pragma omp parallel for schedule(static)
    for (int t = rankId; t < cols.tiles; t += ranks) {
        for (int j = 0; j < cols.width(t); ++j)
            host[cols.offset[t] + size_t(j) * n + t * blockSize + j] += n;
    }
    CHECK(cudaMemcpyAsync(a.p, host.data(), cols.count * sizeof(double),
                          cudaMemcpyHostToDevice, gpu.stream));
    gpu.sync();
    if (validate) original.swap(host);
}

// Right-looking blocked Cholesky with block-cyclic column ownership. The matrix
// stays distributed on devices; only the current factor panel is communicated.
// Each rank has one stream, keeping its updates ordered without global barriers.
struct FactorWorkspace {
    Device<double> panel;
    Pinned hostPanel;
    Device<int> info;
    int workSize = 0;
    double* work = nullptr;
    cudaEvent_t uploaded;
    FactorWorkspace(Accelerator& gpu, const Columns& cols, Device<double>& a)
        : panel(size_t(cols.n) * std::min(cols.n, blockSize)),
          hostPanel(size_t(cols.n) * std::min(cols.n, blockSize)), info(1) {
        CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER,
                                         std::min(cols.n, blockSize), a.p, cols.n, &workSize));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&work), std::max(1, workSize) * sizeof(double)));
        CHECK(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
        CHECK(cudaEventRecord(uploaded, gpu.stream));
    }
    ~FactorWorkspace() { cudaEventDestroy(uploaded); cudaFree(work); }
};

bool factor(Accelerator& gpu, const Columns& cols, Device<double>& a, FactorWorkspace& scratch) {
    const int n = cols.n;
    auto& panel = scratch.panel;
    auto& hostPanel = scratch.hostPanel;
    auto& info = scratch.info;
    auto uploaded = scratch.uploaded;
    const double one = 1.0, minusOne = -1.0;
    for (int t = 0; t < cols.tiles; ++t) {
        int k = t * blockSize, b = cols.width(t), m = n - k;
        int owner = t % ranks, status = 0;
        // Protect the pinned buffer, but let previous trailing updates continue
        // while this rank waits for the next panel's MPI broadcast.
        CHECK(cudaEventSynchronize(uploaded));
        if (rankId == owner) {
            double* diagonal = a.p + cols.offset[t] + k;
            CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, b,
                                   diagonal, n, scratch.work, scratch.workSize, info.p));
            CHECK(cudaMemcpyAsync(&status, info.p, sizeof(int), cudaMemcpyDeviceToHost, gpu.stream));
            gpu.sync();
            if (!status && m > b) {
                CHECK(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                   CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m - b, b,
                                   &one, diagonal, n, diagonal + b, n));
                CHECK(cudaMemcpy2DAsync(hostPanel.p, size_t(m) * sizeof(double),
                                        diagonal, size_t(n) * sizeof(double),
                                        size_t(m) * sizeof(double), b,
                                        cudaMemcpyDeviceToHost, gpu.stream));
                gpu.sync();
            }
        }
        CHECK(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (status) {
            if (rankId == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + status - 1);
            return false;
        }
        if (m == b) break;
        broadcast(hostPanel.p, size_t(m) * b, owner);
        CHECK(cudaMemcpyAsync(panel.p, hostPanel.p, size_t(m) * b * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.stream));
        CHECK(cudaEventRecord(uploaded, gpu.stream));
        // A(i,j) -= L(i,k) L(j,k)^T. Each GEMM updates one owned block
        // column, including its diagonal block; unused upper entries are ignored.
        int first = t + 1 + (rankId - (t + 1) % ranks + ranks) % ranks;
        for (int j = first; j < cols.tiles; j += ranks) {
            int row = j * blockSize;
            CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                              n - row, cols.width(j), b, &minusOne,
                              panel.p + row - k, m, panel.p + row - k, m,
                              &one, a.p + cols.offset[j] + row, n));
        }
    }
    gpu.sync();
    return true;
}

// Gathering is outside the timed region and only requested for output/validation.
// MPI communication remains on the main thread (MPI_THREAD_FUNNELED).
std::vector<double> gather(const Columns& cols, const std::vector<double>& local, bool lowerOnly) {
    std::vector<double> result;
    if (rankId == 0) result.resize(size_t(cols.n) * cols.n);
    std::vector<double> tile(size_t(cols.n) * std::min(cols.n, blockSize));
    for (int t = 0; t < cols.tiles; ++t) {
        int owner = t % ranks, width = cols.width(t);
        size_t elements = size_t(cols.n) * width;
        const double* source = rankId == owner ? local.data() + cols.offset[t] : tile.data();
        if (owner != 0) {
            for (size_t pos = 0; pos < elements;) {
                int chunk = static_cast<int>(std::min(elements - pos, size_t(INT_MAX)));
                if (rankId == owner) CHECK(MPI_Send(source + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                if (rankId == 0) CHECK(MPI_Recv(tile.data() + pos, chunk, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                pos += chunk;
            }
        }
        if (rankId == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < cols.n; ++i) {
                for (int j = 0; j < width; ++j) {
                    int column = t * blockSize + j;
                    result[size_t(i) * cols.n + column] = lowerOnly && i < column ? 0.0 : source[size_t(j) * cols.n + i];
                }
            }
        }
    }
    return result;
}

bool validateCholesky(Accelerator& gpu, const std::vector<double>& l,
                       const std::vector<double>& original, int n) {
    Device<double> dl(l.size()), product(l.size());
    std::vector<double> reconstructed(l.size());
    CHECK(cudaMemcpyAsync(dl.p, l.data(), l.size() * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
    const double one = 1.0, zero = 0.0;
    CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                      &one, dl.p, n, dl.p, n, &zero, product.p, n));
    CHECK(cudaMemcpyAsync(reconstructed.data(), product.p, l.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
    gpu.sync();
    double maxError = 0.0, relError = 0.0;
    int finite = 1;
    #pragma omp parallel for reduction(max:maxError,relError) reduction(&:finite) schedule(static)
    for (size_t i = 0; i < l.size(); ++i) {
        double error = std::abs(reconstructed[i] - original[i]);
        finite &= std::isfinite(error);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::abs(original[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    return finite && relError <= 1e-6;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long long value = strtoll(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || value <= 0 || value > INT_MAX ||
                static_cast<unsigned long long>(value) > std::numeric_limits<size_t>::max() / sizeof(double) / value) {
                if (rankId == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rankId == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rankId == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    Accelerator gpu;
    Columns cols(n);
    Device<double> a(cols.count);
    std::vector<double> original;
    if (rankId == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA block size: %d\n", ranks, omp_get_max_threads(), blockSize);
        printf("Generating positive definite matrix...\n");
    }
    generate(gpu, cols, a, original, validate);
    if (rankId == 0) printf("Computing Cholesky decomposition...\n");
    FactorWorkspace scratch(gpu, cols, a);
    gpu.sync();
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = factor(gpu, cols, a, scratch);
    double elapsed = MPI_Wtime() - start, duration = 0.0;
    CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) return 1;
    if (rankId == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0e9 * duration));
    }
    int valid = 1;
    if (printResults || validate) {
        std::vector<double> local(cols.count);
        CHECK(cudaMemcpyAsync(local.data(), a.p, cols.count * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
        gpu.sync();
        auto l = gather(cols, local, true);
        std::vector<double> fullOriginal;
        if (validate) fullOriginal = gather(cols, original, false);
        if (rankId == 0) {
            if (printResults) print_results(l, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                valid = validateCholesky(gpu, l, fullOriginal, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rankId == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try { result = run(argc, argv); }
    catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rankId, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(MPI_Finalize());
    return result;
}
