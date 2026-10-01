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

namespace {
int rank = 0;
[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
void check(cudaError_t s) { if (s != cudaSuccess) fail(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) fail("cuBLAS operation failed"); }
void check(cusolverStatus_t s) { if (s != CUSOLVER_STATUS_SUCCESS) fail("cuSOLVER operation failed"); }
void mpi(int s) { if (s != MPI_SUCCESS) fail("MPI operation failed"); }

// Chunked collectives also work when a panel exceeds MPI's int count limit.
void broadcast(double* p, size_t count, int root, MPI_Comm comm) {
    while (count) {
        int part = static_cast<int>(std::min(count, size_t(INT_MAX)));
        mpi(MPI_Bcast(p, part, MPI_DOUBLE, root, comm));
        p += part;
        count -= part;
    }
}
struct Device {
    double* p = nullptr;
    explicit Device(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(double)) fail("Allocation overflow");
        check(cudaMalloc(reinterpret_cast<void**>(&p), std::max(count, size_t(1)) * sizeof(double)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) {
        check(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(count, size_t(1)) * sizeof(double)));
    }
    ~Pinned() { cudaFreeHost(p); }
};
struct Tile { int i, j; size_t offset; };

// Only the lower triangle is stored, in column-major tiles. A tile (i,j)
// belongs to (i % processRows, j % processCols), balancing the shrinking work.
struct Matrix {
    int n, b, nt, pr, pc, row, col, nr, nc;
    size_t area;
    MPI_Comm rows, cols;
    std::vector<Tile> tiles;
    std::vector<size_t> offsets;
    Device data;
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;

    static size_t tileCount(int nt, int pr, int pc, int row, int col) {
        size_t result = 0;
        for (int j = col; j < nt; j += pc)
            for (int i = row; i < nt; i += pr) result += i >= j;
        return result;
    }
    Matrix(int n_, int b_, int pr_, int pc_) :
        n(n_), b(b_), nt((n - 1) / b + 1), pr(pr_), pc(pc_),
        row(rank / pc), col(rank % pc),
        nr((nt + pr - 1 - row) / pr), nc((nt + pc - 1 - col) / pc),
        area(size_t(b) * b), offsets(size_t(nr) * nc, size_t(-1)),
        data(tileCount(nt, pr, pc, row, col) * area) {
        mpi(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows));
        mpi(MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols));
        for (int j = col; j < nt; j += pc)
            for (int i = row; i < nt; i += pr) if (i >= j) {
                size_t off = tiles.size() * area;
                offsets[size_t(i / pr) * nc + j / pc] = off;
                tiles.push_back({i, j, off});
            }
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cublasCreate(&blas));
        check(cublasSetStream(blas, stream));
        check(cusolverDnCreate(&solver));
        check(cusolverDnSetStream(solver, stream));
        check(cudaMemsetAsync(data.p, 0, elements() * sizeof(double), stream));
    }
    ~Matrix() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        MPI_Comm_free(&rows);
        MPI_Comm_free(&cols);
    }
    int width(int k) const { return std::min(b, n - k * b); }
    size_t elements() const { return tiles.size() * area; }
    double* at(int i, int j) { return data.p + offsets[size_t(i / pr) * nc + j / pc]; }
    void sync() { check(cudaStreamSynchronize(stream)); }
    void copy(double* dst, const double* src, size_t count, cudaMemcpyKind kind) {
        check(cudaMemcpyAsync(dst, src, count * sizeof(double), kind, stream));
    }
};

void generatePositiveDefiniteMatrix(Matrix& a) {
    // Preserve the original rand_r stream exactly, including on non-glibc hosts.
    // Only row seeds are broadcast; no rank needs a replicated dense matrix.
    std::vector<unsigned int> seeds(a.n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (int i = 0; i < a.n; ++i) {
            seeds[i] = seed;
            for (int k = 0; k < a.n; ++k) (void)rand_r(&seed);
        }
    }
    mpi(MPI_Bcast(seeds.data(), a.n, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    // Store only B rows needed by this process row/column. B is row-major,
    // equivalently a column-major n-by-localRows array for cuBLAS.
    size_t rowCount = size_t(a.nr) * a.b, colCount = size_t(a.nc) * a.b;
    std::vector<double> br(rowCount * a.n, 0.0), bc(colCount * a.n, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rowCount + colCount; ++r) {
        bool inRows = r < rowCount;
        size_t local = inRows ? r : r - rowCount;
        size_t global = ((local / a.b) * (inRows ? a.pr : a.pc)
                         + (inRows ? a.row : a.col)) * a.b + local % a.b;
        if (global >= size_t(a.n)) continue;
        unsigned int seed = seeds[global];
        double* dest = (inRows ? br.data() : bc.data()) + local * a.n;
        for (int k = 0; k < a.n; ++k) dest[k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    Device dr(br.size()), dc(bc.size()), diagonal(a.b);
    std::vector<double> shift(a.b, double(a.n));
    a.copy(dr.p, br.data(), br.size(), cudaMemcpyHostToDevice);
    a.copy(dc.p, bc.data(), bc.size(), cudaMemcpyHostToDevice);
    a.copy(diagonal.p, shift.data(), shift.size(), cudaMemcpyHostToDevice);
    const double one = 1.0, zero = 0.0;
    for (const Tile& t : a.tiles) {
        check(cublasDgemm(a.blas, CUBLAS_OP_T, CUBLAS_OP_N,
              a.width(t.i), a.width(t.j), a.n, &one,
              dr.p + size_t(t.i / a.pr) * a.b * a.n, a.n,
              dc.p + size_t(t.j / a.pc) * a.b * a.n, a.n,
              &zero, a.data.p + t.offset, a.b));
        if (t.i == t.j)
            check(cublasDaxpy(a.blas, a.width(t.i), &one, diagonal.p, 1,
                              a.data.p + t.offset, a.b + 1));
    }
    a.sync();
}

// Pinned staging supports ordinary MPI installations as well as CUDA-aware
// MPI. Communication is confined to process rows and columns: no dense
// matrix or complete panel is replicated on every rank.
struct Panel {
    Matrix& a;
    Pinned hrow, hcol;
    Device drow, dcol;
    std::vector<MPI_Request> requests;
    explicit Panel(Matrix& a_) : a(a_), hrow(size_t(a.nr) * a.area),
        hcol(size_t(a.nc) * a.area), drow(size_t(a.nr) * a.area),
        dcol(size_t(a.nc) * a.area), requests(a.nc) {}
    double* left(int i) { return drow.p + size_t(i / a.pr) * a.area; }
    double* right(int j) { return dcol.p + size_t(j / a.pc) * a.area; }
    void exchange(int k, bool includeDiagonal) {
        int first = k + (includeDiagonal ? 0 : 1);
        int start = std::max(0, (first - a.row + a.pr - 1) / a.pr);
        start = std::min(start, a.nr);
        size_t offset = size_t(start) * a.area;
        size_t count = size_t(a.nr - start) * a.area;
        if (a.col == k % a.pc) {
            for (int ii = start; ii < a.nr; ++ii) {
                int i = ii * a.pr + a.row;
                a.copy(hrow.p + size_t(ii) * a.area, a.at(i, k), a.area, cudaMemcpyDeviceToHost);
            }
        }
        a.sync();
        broadcast(hrow.p + offset, count, k % a.pc, a.rows);
        a.copy(drow.p + offset, hrow.p + offset, count, cudaMemcpyHostToDevice);
        // The row broadcast made L(j,k) available at (j % pr, this column).
        // Forward it vertically to ranks that own tiles in block column j.
        int pending = 0;
        for (int j = a.col; j < a.nt; j += a.pc) if (j >= first) {
            double* target = hcol.p + size_t(j / a.pc) * a.area;
            if (a.row == j % a.pr)
                std::memcpy(target, hrow.p + size_t(j / a.pr) * a.area, a.area * sizeof(double));
            mpi(MPI_Ibcast(target, int(a.area), MPI_DOUBLE, j % a.pr,
                          a.cols, &requests[pending++]));
        }
        // Launch the independent column broadcasts together instead of
        // serializing their network latencies, then upload one packed panel.
        mpi(MPI_Waitall(pending, requests.data(), MPI_STATUSES_IGNORE));
        int colStart = std::max(0, (first - a.col + a.pc - 1) / a.pc);
        colStart = std::min(colStart, a.nc);
        size_t colOffset = size_t(colStart) * a.area;
        a.copy(dcol.p + colOffset, hcol.p + colOffset,
               size_t(a.nc - colStart) * a.area, cudaMemcpyHostToDevice);
    }
};

bool choleskyDecomposition(Matrix& a, Panel& panel, Device& work, int workSize,
                           Device& infoStorage, Pinned& diagonal, Device& dDiagonal) {
    const double one = 1.0, minus = -1.0;
    int* devInfo = reinterpret_cast<int*>(infoStorage.p);
    for (int k = 0; k < a.nt; ++k) {
        int owner = (k % a.pr) * a.pc + k % a.pc;
        int info = 0, w = a.width(k);
        if (rank == owner) {
            check(cusolverDnDpotrf(a.solver, CUBLAS_FILL_MODE_LOWER, w, a.at(k, k),
                                  a.b, work.p, workSize, devInfo));
            check(cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, a.stream));
            a.copy(diagonal.p, a.at(k, k), a.area, cudaMemcpyDeviceToHost);
            a.sync();
        }
        mpi(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info != 0) {
            if (rank == 0) std::printf("Error: Cholesky factorization failed at diagonal element %d (info %d)\n",
                                        k * a.b + std::max(info - 1, 0), info);
            return false;
        }
        if (a.col == k % a.pc) {
            if (rank == owner) {
                // cuSOLVER leaves the upper triangle untouched. Clear it for
                // reconstruction and for the original lower-triangular output.
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < a.b; ++j)
                    for (int i = 0; i < a.b; ++i)
                        if (i < j || i >= w || j >= w) diagonal.p[size_t(j) * a.b + i] = 0.0;
                a.copy(a.at(k, k), diagonal.p, a.area, cudaMemcpyHostToDevice);
            }
            broadcast(diagonal.p, a.area, k % a.pr, a.cols);
            a.copy(dDiagonal.p, diagonal.p, a.area, cudaMemcpyHostToDevice);
            for (int i = a.row; i < a.nt; i += a.pr) if (i > k)
                check(cublasDtrsm(a.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, a.width(i), w, &one,
                      dDiagonal.p, a.b, a.at(i, k), a.b));
        }
        panel.exchange(k, false);
        // Column-ordered tiles prioritize the next panel before the distant
        // trailing work. All dependencies stay ordered on the CUDA stream.
        for (const Tile& t : a.tiles) if (t.j > k) {
            if (t.i == t.j)
                check(cublasDsyrk(a.blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                      a.width(t.i), w, &minus, panel.left(t.i), a.b,
                      &one, a.data.p + t.offset, a.b));
            else
                check(cublasDgemm(a.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                      a.width(t.i), a.width(t.j), w, &minus, panel.left(t.i), a.b,
                      panel.right(t.j), a.b, &one, a.data.p + t.offset, a.b));
        }
    }
    a.sync();
    return true;
}

bool validateCholesky(Matrix& a, Panel& panel, const Device& original) {
    Device reconstructed(a.elements());
    check(cudaMemsetAsync(reconstructed.p, 0, a.elements() * sizeof(double), a.stream));
    const double one = 1.0;
    for (int k = 0; k < a.nt; ++k) {
        panel.exchange(k, true);
        for (const Tile& t : a.tiles) if (t.j >= k)
            check(cublasDgemm(a.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                  a.width(t.i), a.width(t.j), a.width(k), &one,
                  panel.left(t.i), a.b, panel.right(t.j), a.b,
                  &one, reconstructed.p + t.offset, a.b));
    }
    std::vector<double> ref(a.elements()), actual(a.elements());
    a.copy(ref.data(), original.p, ref.size(), cudaMemcpyDeviceToHost);
    a.copy(actual.data(), reconstructed.p, actual.size(), cudaMemcpyDeviceToHost);
    a.sync();
    double absolute = 0.0, relative = 0.0;
    #pragma omp parallel for reduction(max:absolute,relative) schedule(static)
    for (size_t ti = 0; ti < a.tiles.size(); ++ti) {
        const Tile& t = a.tiles[ti];
        for (int j = 0; j < a.width(t.j); ++j)
            for (int i = 0; i < a.width(t.i); ++i) {
                if (t.i == t.j && i < j) continue;
                size_t index = t.offset + size_t(j) * a.b + i;
                double error = std::fabs(actual[index] - ref[index]);
                if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
                absolute = std::max(absolute, error);
                relative = std::max(relative, error / (std::fabs(ref[index]) + 1e-10));
            }
    }
    double local[2] = {absolute, relative}, global[2];
    mpi(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (rank == 0) {
        std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
        std::printf("Validation: %s\n", global[1] <= 1e-6 ? "PASSED" : "FAILED");
    }
    return global[1] <= 1e-6;
}

void printFactor(Matrix& a) {
    // Gathering is optional and outside the timed region. Preserve row-major
    // output, including exactly zero entries above the diagonal.
    std::vector<double> result;
    if (rank == 0) result.resize(size_t(a.n) * a.n, 0.0);
    Pinned tile(a.area);
    for (int j = 0; j < a.nt; ++j) for (int i = j; i < a.nt; ++i) {
        int owner = (i % a.pr) * a.pc + j % a.pc;
        if (rank == owner) {
            a.copy(tile.p, a.at(i, j), a.area, cudaMemcpyDeviceToHost);
            a.sync();
            if (owner != 0) mpi(MPI_Send(tile.p, int(a.area), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
        }
        if (rank == 0) {
            if (owner != 0) mpi(MPI_Recv(tile.p, int(a.area), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < a.width(i); ++r)
                for (int c = 0; c < a.width(j); ++c)
                    if (i != j || r >= c)
                        result[size_t(i * a.b + r) * a.n + j * a.b + c] = tile.p[size_t(c) * a.b + r];
        }
    }
    if (rank == 0) print_results(result, "CholeskyL");
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long long value = std::strtoll(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || value <= 0 || value > INT_MAX - 1024 ||
                static_cast<unsigned long long>(value) > std::numeric_limits<size_t>::max() / sizeof(double) / value) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = int(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); return 0; }
        else { if (rank == 0) printUsage(argv[0]); return 1; }
    }
    MPI_Comm local;
    mpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int localRank = 0, localSize = 1, devices = 0;
    mpi(MPI_Comm_rank(local, &localRank));
    mpi(MPI_Comm_size(local, &localSize));
    check(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA GPU is required on every MPI rank");
    // Works both with a shared visible GPU list and one GPU exposed per rank.
    check(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    int processes = 1, dims[2] = {0, 0};
    mpi(MPI_Comm_size(MPI_COMM_WORLD, &processes));
    mpi(MPI_Dims_create(processes, 2, dims));
    // Smaller tiles expose parallelism on small matrices; large cases use
    // 256-wide BLAS-3 updates to amortize launches and network latency.
    int block = n <= 1024 ? 64 : (n <= 4096 ? 128 : 256);
    Matrix a(n, block, dims[0], dims[1]);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n",
                    n, n, validate ? "enabled" : "disabled");
        std::printf("MPI grid: %d x %d, tile size: %d, OpenMP threads per rank: %d\n",
                    dims[0], dims[1], block, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(a);
    Device original(validate ? a.elements() : 0);
    if (validate) a.copy(original.p, a.data.p, a.elements(), cudaMemcpyDeviceToDevice);
    Panel panel(a);
    Pinned diagonal(a.area);
    Device dDiagonal(a.area), info(1);
    int workSize = 0;
    check(cusolverDnDpotrf_bufferSize(a.solver, CUBLAS_FILL_MODE_LOWER,
                                    std::min(n, block), dDiagonal.p, block, &workSize));
    Device work(workSize);
    // Initialize the solver's lazy CUDA kernels using scratch storage so
    // library startup is not charged to the first distributed panel.
    std::fill_n(diagonal.p, a.area, 0.0);
    for (int i = 0; i < std::min(n, block); ++i) diagonal.p[size_t(i) * (block + 1)] = 1.0;
    a.copy(dDiagonal.p, diagonal.p, a.area, cudaMemcpyHostToDevice);
    check(cusolverDnDpotrf(a.solver, CUBLAS_FILL_MODE_LOWER, std::min(n, block),
                          dDiagonal.p, block, work.p, workSize, reinterpret_cast<int*>(info.p)));
    a.sync();
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    mpi(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(a, panel, work, workSize, info, diagonal, dDiagonal);
    double elapsed = MPI_Wtime() - start, maximum = 0.0;
    mpi(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) { if (rank == 0) std::printf("Cholesky decomposition failed\n"); return 1; }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", maximum * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0e9 * maximum));
    }
    if (printResults) printFactor(a);
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        if (!validateCholesky(a, panel, original)) return 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int result = 1;
    try { result = run(argc, argv); }
    catch (const std::exception& e) { fail(e.what()); }
    mpi(MPI_Finalize());
    return result;
}
