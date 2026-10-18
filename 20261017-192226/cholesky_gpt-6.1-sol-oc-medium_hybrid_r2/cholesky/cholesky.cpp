#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../common/results_output.hpp"

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED). Host staging
// works with ordinary MPI installations; CUDA-aware MPI is not required.
static void check(int status, const char* operation, const char* file, int line) {
    if (status == 0) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s failed (%d) at %s:%d\n", rank, operation, status, file, line);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}
#define CHECK(call) check(static_cast<int>(call), #call, __FILE__, __LINE__)

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) { CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double))); }
    ~Pinned() { cudaFreeHost(p); }
};

__global__ void initializeTile(double* a, int b, int tileRow, int tileCol, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= b * b) return;
    int i = tileRow * b + x % b, j = tileCol * b + x / b;
    // Padding is an independent identity block; it cannot affect the real matrix.
    a[x] = i == j ? (i < n ? double(n) : 1.0) : 0.0;
}

struct Tile { int i, j; double* p; };

struct Matrix {
    int n, b, nt, rank, pr, pc, row, col, nr, nc;
    size_t area;
    MPI_Comm rows, cols;
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    std::vector<Tile> tiles;
    std::vector<double*> lookup;
    Device<double> storage;

    static size_t tileCount(int nt, int row, int col, int pr, int pc) {
        size_t count = 0;
        for (int i = row; i < nt; i += pr)
            for (int j = col; j <= i; j += pc) ++count;
        return count;
    }
    Matrix(int size, int block, int r, int p, int q)
        : n(size), b(block), nt((n + b - 1) / b), rank(r), pr(p), pc(q),
          row(r / q), col(r % q), nr((nt + pr - 1) / pr), nc((nt + pc - 1) / pc),
          area(size_t(b) * b), lookup(size_t(nr) * nc, nullptr),
          storage(tileCount(nt, row, col, pr, pc) * area) {
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows));
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols));
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas)); CHECK(cublasSetStream(blas, stream));
        CHECK(cusolverDnCreate(&solver)); CHECK(cusolverDnSetStream(solver, stream));
        for (int i = row; i < nt; i += pr) {
            for (int j = col; j <= i; j += pc) {
                double* ptr = storage.p + tiles.size() * area;
                tiles.push_back({i, j, ptr});
                lookup[size_t(i / pr) * nc + j / pc] = ptr;
                initializeTile<<<(area + 255) / 256, 256, 0, stream>>>(ptr, b, i, j, n);
            }
        }
        CHECK(cudaGetLastError());
    }
    ~Matrix() {
        cusolverDnDestroy(solver); cublasDestroy(blas); cudaStreamDestroy(stream);
        MPI_Comm_free(&rows); MPI_Comm_free(&cols);
    }
    double* at(int i, int j) { return lookup[size_t(i / pr) * nc + j / pc]; }
    int owner(int i, int j) const { return (i % pr) * pc + j % pc; }
    void sync() { CHECK(cudaStreamSynchronize(stream)); }

    void generate() {
        // Preserve the original rand_r stream exactly. Only rank zero holds B;
        // all ranks consume narrow k-panels, never a replicated device matrix.
        std::vector<double> random;
        if (rank == 0) {
            random.resize(size_t(n) * n);
            unsigned seed = 42;
            for (double& x : random) x = rand_r(&seed) / double(RAND_MAX) - 0.5;
        }
        const int padded = nt * b;
        Pinned panel(size_t(padded) * b);
        Device<double> devicePanel(size_t(padded) * b);
        const double one = 1.0;
        for (int k = 0; k < n; k += b) {
            int width = std::min(b, n - k);
            if (rank == 0) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < padded; ++i)
                    for (int x = 0; x < b; ++x)
                        panel.p[size_t(i) * b + x] = i < n && x < width ? random[size_t(i) * n + k + x] : 0.0;
            }
            CHECK(MPI_Bcast(panel.p, padded * b, MPI_DOUBLE, 0, MPI_COMM_WORLD));
            CHECK(cudaMemcpyAsync(devicePanel.p, panel.p, size_t(padded) * b * sizeof(double), cudaMemcpyHostToDevice, stream));
            for (const auto& t : tiles) {
                CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, b, b, width, &one,
                    devicePanel.p + size_t(t.i) * area, b,
                    devicePanel.p + size_t(t.j) * area, b, &one, t.p, b));
            }
            sync(); // the pinned panel can now be reused by MPI
        }
    }

    bool factor(double& elapsed) {
        Pinned diagonal(area), rowPanel(size_t(nr) * area), columnPanel(size_t(nc) * area);
        Pinned packed(size_t(nc) * area);
        Device<double> diag(area), left(size_t(nr) * area), right(size_t(nc) * area);
        Device<int> info(1);
        int workSize = 0;
        CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, b, diag.p, b, &workSize));
        Device<double> work(workSize);
        // Batched updates amortize launch overhead for the many off-diagonal tiles.
        Device<double*> aa(tiles.size()), bb(tiles.size()), cc(tiles.size());
        std::vector<double*> a(tiles.size()), bp(tiles.size()), c(tiles.size());
        const double one = 1.0, minus = -1.0;
        // Exclude workspace allocation and one-time solver kernel loading from
        // the factorization measurement, including on ranks that own later pivots.
        initializeTile<<<(area + 255) / 256, 256, 0, stream>>>(diag.p, b, 0, 0, b);
        CHECK(cudaGetLastError());
        CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, diag.p, b, work.p, workSize, info.p));
        sync();
        CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double start = MPI_Wtime();
        for (int k = 0; k < nt; ++k) {
            int root = owner(k, k), status = 0;
            if (rank == root) {
                CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, b, at(k, k), b, work.p, workSize, info.p));
                CHECK(cudaMemcpyAsync(&status, info.p, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CHECK(cudaMemcpyAsync(diagonal.p, at(k, k), area * sizeof(double), cudaMemcpyDeviceToHost, stream));
                sync();
            }
            CHECK(MPI_Bcast(&status, 1, MPI_INT, root, MPI_COMM_WORLD));
            if (status) {
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k * b + status - 1);
                elapsed = MPI_Wtime() - start;
                return false;
            }
            CHECK(MPI_Bcast(diagonal.p, int(area), MPI_DOUBLE, root, MPI_COMM_WORLD));
            CHECK(cudaMemcpyAsync(diag.p, diagonal.p, area * sizeof(double), cudaMemcpyHostToDevice, stream));
            if (col == k % pc) {
                for (const auto& t : tiles) if (t.j == k && t.i > k) {
                    CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, b, b, &one, diag.p, b, t.p, b));
                    CHECK(cudaMemcpyAsync(rowPanel.p + size_t(t.i / pr) * area, t.p,
                        area * sizeof(double), cudaMemcpyDeviceToHost, stream));
                }
            }
            sync();
            // First distribute L_ik along process rows. Then redistribute L_jk
            // down process columns. Communication volume is O(n*b/sqrt(P)).
            int firstRow = std::max(0, (k + 1 - row + pr - 1) / pr);
            int rowEnd = row < nt ? (nt - 1 - row) / pr + 1 : 0;
            int rowCount = std::max(0, rowEnd - firstRow);
            CHECK(MPI_Bcast(rowPanel.p + size_t(firstRow) * area, rowCount * int(area), MPI_DOUBLE, k % pc, rows));
            for (int source = 0; source < pr; ++source) {
                std::vector<int> indices;
                for (int j = col; j < nt; j += pc)
                    if (j > k && j % pr == source) indices.push_back(j);
                if (row == source) {
                    #pragma omp parallel for schedule(static)
                    for (size_t x = 0; x < indices.size(); ++x)
                        memcpy(packed.p + x * area, rowPanel.p + size_t(indices[x] / pr) * area, area * sizeof(double));
                }
                CHECK(MPI_Bcast(packed.p, int(indices.size() * area), MPI_DOUBLE, source, cols));
                #pragma omp parallel for schedule(static)
                for (size_t x = 0; x < indices.size(); ++x)
                    memcpy(columnPanel.p + size_t(indices[x] / pc) * area, packed.p + x * area, area * sizeof(double));
            }
            CHECK(cudaMemcpyAsync(left.p, rowPanel.p, size_t(nr) * area * sizeof(double), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(right.p, columnPanel.p, size_t(nc) * area * sizeof(double), cudaMemcpyHostToDevice, stream));
            int count = 0;
            for (const auto& t : tiles) if (t.j > k) {
                a[count] = left.p + size_t(t.i / pr) * area;
                bp[count] = right.p + size_t(t.j / pc) * area;
                c[count++] = t.p;
            }
            if (count) {
                CHECK(cudaMemcpyAsync(aa.p, a.data(), count * sizeof(double*), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(bb.p, bp.data(), count * sizeof(double*), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(cc.p, c.data(), count * sizeof(double*), cudaMemcpyHostToDevice, stream));
                CHECK(cublasDgemmBatched(blas, CUBLAS_OP_N, CUBLAS_OP_T, b, b, b,
                    &minus, aa.p, b, bb.p, b, &one, cc.p, b, count));
            }
            sync();
        }
        elapsed = MPI_Wtime() - start;
        return true;
    }

    std::vector<double> gather(bool symmetric) {
        // Only requested output/validation materializes a full matrix on rank 0.
        std::vector<double> result(rank == 0 ? size_t(n) * n : 0, 0.0);
        Pinned tile(area);
        sync();
        for (int i = 0; i < nt; ++i) for (int j = 0; j <= i; ++j) {
            int source = owner(i, j);
            if (rank == source) {
                CHECK(cudaMemcpy(tile.p, at(i, j), area * sizeof(double), cudaMemcpyDeviceToHost));
                if (rank != 0) CHECK(MPI_Send(tile.p, int(area), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            }
            if (rank == 0) {
                if (source != 0) CHECK(MPI_Recv(tile.p, int(area), MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                int height = std::min(b, n - i * b);
                #pragma omp parallel for schedule(static)
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < b && j * b + x < n; ++x) {
                        int r = i * b + y, c = j * b + x;
                        if (r >= c) {
                            result[size_t(r) * n + c] = tile.p[size_t(x) * b + y];
                            if (symmetric) result[size_t(c) * n + r] = tile.p[size_t(x) * b + y];
                        }
                    }
                }
            }
        }
        return result;
    }
};

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n) {
    Device<double> dl(size_t(n) * n), reconstructed(size_t(n) * n);
    CHECK(cudaMemcpy(dl.p, l.data(), l.size() * sizeof(double), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    CHECK(cublasCreate(&handle));
    const double one = 1.0, zero = 0.0;
    // Row-major L is column-major L^T.
    CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
        &one, dl.p, n, dl.p, n, &zero, reconstructed.p, n));
    std::vector<double> result(l.size());
    CHECK(cudaMemcpy(result.data(), reconstructed.p, result.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cublasDestroy(handle));
    double absolute = 0, relative = 0;
    #pragma omp parallel for reduction(max:absolute,relative) schedule(static)
    for (size_t i = 0; i < l.size(); ++i) {
        double error = fabs(result[i] - original[i]);
        if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
        absolute = std::max(absolute, error);
        relative = std::max(relative, error / (fabs(original[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", absolute, relative);
    if (relative > 1e-6) printf("Validation failed: relative error too large\n");
    return relative <= 1e-6;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank, size;
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, result = 0;
    bool validate = false, printResults = false, stop = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || *end || value <= 0 || value > INT_MAX / 512 - 512) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                stop = true; result = 1; break;
            }
            n = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { stop = true; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            stop = true; result = 1; break;
        }
    }
    if (stop) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize(); return result;
    }
    MPI_Comm local;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int localRank, localSize, devices;
    CHECK(MPI_Comm_rank(local, &localRank));
    CHECK(MPI_Comm_size(local, &localSize));
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        if (rank == 0) fprintf(stderr, "A CUDA device is required on every rank\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(cudaSetDevice(localRank % devices));
    CHECK(MPI_Comm_free(&local));
    // Respect explicit OMP_NUM_THREADS; avoid oversubscribing CPUs by default.
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    int dims[2] = {0, 0};
    CHECK(MPI_Dims_create(size, 2, dims));
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("MPI grid: %d x %d; OpenMP threads/rank: %d; CUDA enabled\n", dims[0], dims[1], omp_get_max_threads());
        printf("Generating positive definite matrix...\n");
    }
    try {
        const int block = 256;
        Matrix matrix(n, block, rank, dims[0], dims[1]);
        matrix.generate();
        std::vector<double> original;
        if (validate) original = matrix.gather(true);
        if (rank == 0) printf("Computing Cholesky decomposition...\n");
        double elapsed = 0, seconds;
        bool success = matrix.factor(elapsed);
        CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (!success) {
            if (rank == 0) printf("Cholesky decomposition failed\n");
            result = 1;
        } else {
            if (rank == 0) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * seconds));
            }
            if (printResults || validate) {
                auto l = matrix.gather(false);
                if (rank == 0) {
                    if (printResults) print_results(l, "CholeskyL");
                    if (validate) {
                        printf("Validating result...\n");
                        bool valid = validateCholesky(l, original, n);
                        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                        result = valid ? 0 : 1;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK(MPI_Finalize());
    return result;
}
