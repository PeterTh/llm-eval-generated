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

#include "../common/results_output.hpp"

namespace {
constexpr int block = 64;
constexpr int area = block * block;

// Lower-triangular tiles, cyclically distributed over a rectangular process grid.
// Padding keeps the local kernels and MPI messages contiguous, including edges.
struct Matrix {
    size_t n;
    int rank, rows, cols, row, col, nt, nr, nc;
    MPI_Comm rowComm, colComm;
    std::vector<int> index;
    std::vector<double> data;

    Matrix(size_t size, int id, int processes) : n(size), rank(id) {
        int dims[2] = {0, 0};
        MPI_Dims_create(processes, 2, dims);
        rows = dims[0]; cols = dims[1];
        row = rank / cols; col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        nt = static_cast<int>((n + block - 1) / block);
        nr = (nt + rows - 1 - row) / rows;
        nc = (nt + cols - 1 - col) / cols;
        index.assign(static_cast<size_t>(nr) * nc, -1);
        int count = 0;
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols)
                index[static_cast<size_t>(i / rows) * nc + j / cols] = count++;
        data.resize(static_cast<size_t>(count) * area);
    }
    ~Matrix() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int width(int i) const { return static_cast<int>(std::min<size_t>(block, n - size_t(i) * block)); }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
    double* tile(int i, int j) {
        return data.data() + static_cast<size_t>(index[static_cast<size_t>(i / rows) * nc + j / cols]) * area;
    }
};

// C += sign * X * Y^T. Y is transposed once outside this kernel; the
// innermost loop accesses contiguous elements and can be vectorized.
void multiply(double* c, const double* x, const double* yt,
              int m, int n, int k, double sign) {
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p) {
            const double a = sign * x[i * block + p];
            for (int j = 0; j < n; ++j)
                c[i * block + j] += a * yt[p * block + j];
        }
}

void transpose(const double* in, double* out) {
    for (int i = 0; i < block; ++i)
        for (int j = 0; j < block; ++j)
            out[j * block + i] = in[i * block + j];
}

void generatePositiveDefiniteMatrix(Matrix& a) {
    // Save the original rand_r stream's state at each row boundary. This
    // preserves the input matrix without replicating the dense random matrix.
    std::vector<unsigned int> seeds(a.n);
    if (a.rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < a.n; ++i) {
            seeds[i] = seed;
            for (size_t k = 0; k < a.n; ++k) rand_r(&seed);
        }
    }
    MPI_Bcast(seeds.data(), static_cast<int>(a.n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<unsigned int> columnSeeds = seeds;
    std::vector<double> x(static_cast<size_t>(a.nr) * area);
    std::vector<double> yt(static_cast<size_t>(a.nc) * area);
    for (size_t k = 0; k < a.n; k += block) {
        const int depth = static_cast<int>(std::min<size_t>(block, a.n - k));
        for (int i = a.row; i < a.nt; i += a.rows)
            for (int r = 0; r < a.width(i); ++r)
                for (int p = 0; p < depth; ++p)
                    x[static_cast<size_t>(i / a.rows) * area + r * block + p] =
                        rand_r(&seeds[size_t(i) * block + r]) / double(RAND_MAX) - 0.5;
        for (int j = a.col; j < a.nt; j += a.cols)
            for (int r = 0; r < a.width(j); ++r)
                for (int p = 0; p < depth; ++p)
                    yt[static_cast<size_t>(j / a.cols) * area + p * block + r] =
                        rand_r(&columnSeeds[size_t(j) * block + r]) / double(RAND_MAX) - 0.5;
        for (int i = a.row; i < a.nt; i += a.rows)
            for (int j = a.col; j <= i; j += a.cols)
                multiply(a.tile(i, j), x.data() + static_cast<size_t>(i / a.rows) * area,
                         yt.data() + static_cast<size_t>(j / a.cols) * area,
                         a.width(i), a.width(j), depth, 1.0);
    }
    for (int i = a.row; i < a.nt; i += a.rows)
        if (i % a.cols == a.col)
            for (int r = 0; r < a.width(i); ++r)
                a.tile(i, i)[r * block + r] += a.n;
}

// Replicate only the current panel along process rows and columns. Point-to-
// point redistribution handles rectangular grids without a full-panel allgather.
struct Panel {
    Matrix& a;
    std::vector<double> x, y, yt;
    std::vector<MPI_Request> requests;
    explicit Panel(Matrix& matrix) : a(matrix),
        x(static_cast<size_t>(a.nr) * area),
        y(static_cast<size_t>(a.nc) * area), yt(y.size()) {
        requests.reserve(static_cast<size_t>(a.nr) + a.nc);
    }
    void broadcast(int k, int first) {
        requests.clear();
        const int rootRow = k % a.rows, rootCol = k % a.cols;
        // Receive into disjoint column-panel tiles in a fixed source order.
        if (a.row == rootRow)
            for (int i = a.col; i < a.nt; i += a.cols) {
                if (i < first) continue;
                requests.emplace_back();
                MPI_Irecv(y.data() + static_cast<size_t>(i / a.cols) * area,
                          area, MPI_DOUBLE, a.owner(i, k), 0, MPI_COMM_WORLD,
                          &requests.back());
            }
        if (a.col == rootCol)
            for (int i = a.row; i < a.nt; i += a.rows) {
                if (i < first) continue;
                double* buffer = x.data() + static_cast<size_t>(i / a.rows) * area;
                std::copy_n(a.tile(i, k), area, buffer);
                requests.emplace_back();
                MPI_Isend(buffer, area, MPI_DOUBLE, rootRow * a.cols + i % a.cols,
                          0, MPI_COMM_WORLD, &requests.back());
            }
        // Both communication paths can progress together. Buffers are read-only
        // on senders until all point-to-point and collective requests complete.
        MPI_Request rowRequest, colRequest;
        const int rowFirst = std::min(a.nr, std::max(0, (first + a.rows - 1 - a.row) / a.rows));
        const int colFirst = std::min(a.nc, std::max(0, (first + a.cols - 1 - a.col) / a.cols));
        MPI_Ibcast(x.empty() ? nullptr : x.data() + static_cast<size_t>(rowFirst) * area,
                   (a.nr - rowFirst) * area, MPI_DOUBLE, rootCol, a.rowComm, &rowRequest);
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        MPI_Ibcast(y.empty() ? nullptr : y.data() + static_cast<size_t>(colFirst) * area,
                   (a.nc - colFirst) * area, MPI_DOUBLE, rootRow, a.colComm, &colRequest);
        MPI_Wait(&rowRequest, MPI_STATUS_IGNORE);
        MPI_Wait(&colRequest, MPI_STATUS_IGNORE);
        for (int j = colFirst; j < a.nc; ++j)
            transpose(y.data() + static_cast<size_t>(j) * area, yt.data() + static_cast<size_t>(j) * area);
    }
};

bool choleskyDecomposition(Matrix& a) {
    Panel panel(a);
    std::vector<double> diagonal(area);
    for (int k = 0; k < a.nt; ++k) {
        const int w = a.width(k), root = a.owner(k, k);
        if (a.rank == root) {
            double* d = a.tile(k, k);
            bool valid = true;
            for (int j = 0; j < w; ++j) {
                double v = d[j * block + j];
                for (int p = 0; p < j; ++p) v -= d[j * block + p] * d[j * block + p];
                if (!(v > 0.0) || !std::isfinite(v)) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", size_t(k) * block + j);
                    valid = false;
                    break;
                }
                d[j * block + j] = std::sqrt(v);
                for (int i = j + 1; i < w; ++i) {
                    double v2 = d[i * block + j];
                    for (int p = 0; p < j; ++p) v2 -= d[i * block + p] * d[j * block + p];
                    d[i * block + j] = v2 / d[j * block + j];
                }
                for (int p = j + 1; p < block; ++p) d[j * block + p] = 0.0;
            }
            std::copy_n(d, area, diagonal.data());
            if (!valid) diagonal[0] = -1.0;
        }
        // The diagonal also carries failure status so all ranks exit together.
        MPI_Bcast(diagonal.data(), area, MPI_DOUBLE, root, MPI_COMM_WORLD);
        if (diagonal[0] < 0.0) return false;
        if (a.col == k % a.cols)
            for (int i = a.row; i < a.nt; i += a.rows) {
                if (i <= k) continue;
                double* t = a.tile(i, k);
                for (int r = 0; r < a.width(i); ++r)
                    for (int j = 0; j < w; ++j) {
                        double v = t[r * block + j];
                        for (int p = 0; p < j; ++p) v -= t[r * block + p] * diagonal[j * block + p];
                        t[r * block + j] = v / diagonal[j * block + j];
                    }
            }
        if (k + 1 == a.nt) break;
        panel.broadcast(k, k + 1);
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i <= k) continue;
            for (int j = a.col; j <= i; j += a.cols) {
                if (j <= k) continue;
                multiply(a.tile(i, j), panel.x.data() + static_cast<size_t>(i / a.rows) * area,
                         panel.yt.data() + static_cast<size_t>(j / a.cols) * area,
                         a.width(i), a.width(j), w, -1.0);
            }
        }
    }
    return true;
}

bool validateCholesky(Matrix& a, const std::vector<double>& original) {
    Panel panel(a);
    std::vector<double> reconstructed(a.data.size());
    for (int k = 0; k < a.nt; ++k) {
        panel.broadcast(k, k);
        for (int i = a.row; i < a.nt; i += a.rows) {
            if (i < k) continue;
            for (int j = a.col; j <= i; j += a.cols) {
                if (j < k) continue;
                const size_t offset = static_cast<size_t>(a.tile(i, j) - a.data.data());
                multiply(reconstructed.data() + offset,
                         panel.x.data() + static_cast<size_t>(i / a.rows) * area,
                         panel.yt.data() + static_cast<size_t>(j / a.cols) * area,
                         a.width(i), a.width(j), a.width(k), 1.0);
            }
        }
    }
    double errors[2] = {0.0, 0.0}, global[2];
    for (int i = a.row; i < a.nt; i += a.rows)
        for (int j = a.col; j <= i; j += a.cols) {
            const size_t offset = static_cast<size_t>(a.tile(i, j) - a.data.data());
            for (int r = 0; r < a.width(i); ++r)
                for (int c = 0; c < a.width(j); ++c) {
                    const size_t p = offset + r * block + c;
                    double error = std::fabs(reconstructed[p] - original[p]);
                    if (!std::isfinite(error)) error = std::numeric_limits<double>::infinity();
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(original[p]) + 1e-10));
                }
        }
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void printResults(Matrix& a) {
    std::vector<double> full;
    if (a.rank == 0) full.resize(a.n * a.n);
    std::vector<double> buffer(area);
    for (int i = 0; i < a.nt; ++i)
        for (int j = 0; j <= i; ++j) {
            const int owner = a.owner(i, j);
            if (owner != 0 && a.rank == owner)
                MPI_Send(a.tile(i, j), area, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            if (a.rank == 0) {
                const double* t;
                if (owner == 0) t = a.tile(i, j);
                else {
                    MPI_Recv(buffer.data(), area, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    t = buffer.data();
                }
                for (int r = 0; r < a.width(i); ++r)
                    std::copy_n(t + r * block, a.width(j), full.data() + (size_t(i) * block + r) * a.n + size_t(j) * block);
            }
        }
    if (a.rank == 0) print_results(full, "CholeskyL");
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int processes) {
    size_t n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            // Bound panel counts for the MPI int-count API, and allocation sizes.
            if (errno || *value == '-' || end == value || *end ||
                parsed > static_cast<unsigned long long>(INT_MAX / block - block) ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Matrix a(n, rank, processes);
    generatePositiveDefiniteMatrix(a);
    std::vector<double> original;
    if (validate) original = a.data;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double ops = double(n) * double(n) * double(n) / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    }
    if (results) printResults(a);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateCholesky(a, original);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = run(argc, argv, rank, processes);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
