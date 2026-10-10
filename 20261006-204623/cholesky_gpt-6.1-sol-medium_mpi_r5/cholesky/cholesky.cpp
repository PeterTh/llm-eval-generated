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
constexpr int tileElements = block * block;

// A small register-blocked GEMM, C += sign * X * Y. Y is transposed
// beforehand, so both the inner loop and the output stores are contiguous.
// No BLAS dependency is required. Partial boundary tiles use zero padding.
void multiply(double* c, const double* x, size_t xs, const double* y,
              size_t ys, int rows, int cols, int depth, double sign) {
    for (int i = 0; i < rows; i += 4) {
        for (int j = 0; j < cols; j += 8) {
            double a0[8] = {}, a1[8] = {}, a2[8] = {}, a3[8] = {};
            for (int k = 0; k < depth; ++k) {
                const double* b = y + k * ys + j;
                const double v0 = x[i * xs + k];
                const double v1 = i + 1 < rows ? x[(i + 1) * xs + k] : 0.0;
                const double v2 = i + 2 < rows ? x[(i + 2) * xs + k] : 0.0;
                const double v3 = i + 3 < rows ? x[(i + 3) * xs + k] : 0.0;
                for (int q = 0; q < 8; ++q) {
                    a0[q] += v0 * b[q];
                    a1[q] += v1 * b[q];
                    a2[q] += v2 * b[q];
                    a3[q] += v3 * b[q];
                }
            }
            for (int q = 0; q < std::min(8, cols - j); ++q) {
                c[i * block + j + q] += sign * a0[q];
                if (i + 1 < rows) c[(i + 1) * block + j + q] += sign * a1[q];
                if (i + 2 < rows) c[(i + 2) * block + j + q] += sign * a2[q];
                if (i + 3 < rows) c[(i + 3) * block + j + q] += sign * a3[q];
            }
        }
    }
}

struct Matrix {
    int n, nt, rank, pr, pc, rr, cc, nr, nc;
    MPI_Comm rowComm, colComm;
    std::vector<std::vector<double>> tiles;
    std::vector<double> rowPanel, colPanel;

    Matrix(int size, int worldRank, int worldSize) : n(size), rank(worldRank) {
        nt = (n + block - 1) / block;
        int dims[2] = {0, 0};
        MPI_Dims_create(worldSize, 2, dims);
        pr = dims[0]; pc = dims[1]; rr = rank / pc; cc = rank % pc;
        MPI_Comm_split(MPI_COMM_WORLD, rr, cc, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, cc, rr, &colComm);
        nr = rr < nt ? (nt - 1 - rr) / pr + 1 : 0;
        nc = cc < nt ? (nt - 1 - cc) / pc + 1 : 0;
        tiles.resize(static_cast<size_t>(nr) * nc);
        for (int a = 0; a < nr; ++a)
            for (int b = 0; b < nc; ++b)
                if (rr + a * pr >= cc + b * pc)
                    tiles[static_cast<size_t>(a) * nc + b].resize(tileElements);
        rowPanel.resize(static_cast<size_t>(nr) * tileElements);
        colPanel.resize(static_cast<size_t>(nc) * tileElements);
    }
    ~Matrix() { MPI_Comm_free(&rowComm); MPI_Comm_free(&colComm); }
    int owner(int i, int j) const { return (i % pr) * pc + j % pc; }
    int width(int i) const { return std::min(block, n - i * block); }
    double* at(int i, int j) { return tiles[static_cast<size_t>(i / pr) * nc + j / pc].data(); }

    void generate() {
        // Reproduce exactly the original rand_r stream. Keep only the rows
        // needed by this process row/column, rather than replicating B.
        std::vector<double> br(static_cast<size_t>(nr) * block * n);
        std::vector<double> bc(static_cast<size_t>(nc) * n * block);
        unsigned int seed = 42;
        for (int i = 0; i < n; ++i) {
            int t = i / block, r = i % block;
            double* row = t % pr == rr ? br.data() + (static_cast<size_t>(t / pr) * block + r) * n : nullptr;
            double* col = t % pc == cc ? bc.data() + static_cast<size_t>(t / pc) * n * block + r : nullptr;
            for (int k = 0; k < n; ++k) {
                double value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                if (row) row[k] = value;
                if (col) col[static_cast<size_t>(k) * block] = value;
            }
        }
        for (int a = 0; a < nr; ++a) {
            int i = rr + a * pr;
            for (int b = 0; b < nc; ++b) {
                int j = cc + b * pc;
                if (j > i) continue;
                double* c = at(i, j);
                for (int k = 0; k < n; k += block)
                    multiply(c, br.data() + static_cast<size_t>(a) * block * n + k, n,
                             bc.data() + (static_cast<size_t>(b) * n + k) * block, block,
                             width(i), width(j), std::min(block, n - k), 1.0);
                if (i == j)
                    for (int r = 0; r < width(i); ++r) c[r * block + r] += n;
            }
        }
    }

    // Distribute L(:,k) in both orientations. Only process-row/column
    // broadcasts are needed; no full matrix is replicated during the solve.
    void panel(int k) {
        std::vector<MPI_Request> requests;
        // First move each tile to its transposed process-grid position.
        // Receives precede sends, and all requests complete before broadcasts.
        if (rr == k % pr) {
            for (int b = 0; b < nc; ++b) {
                int j = cc + b * pc;
                if (j < k) continue;
                if (owner(j, k) != rank) {
                    MPI_Request req;
                    MPI_Irecv(colPanel.data() + static_cast<size_t>(b) * tileElements,
                              tileElements, MPI_DOUBLE, owner(j, k), 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
            }
        }
        if (cc == k % pc) {
            for (int a = 0; a < nr; ++a) {
                int i = rr + a * pr;
                if (i < k) continue;
                double* p = rowPanel.data() + static_cast<size_t>(a) * tileElements;
                std::copy_n(at(i, k), tileElements, p);
                int dest = owner(k, i);
                if (dest == rank) {
                    std::copy_n(p, tileElements, colPanel.data() + static_cast<size_t>(i / pc) * tileElements);
                } else {
                    MPI_Request req;
                    MPI_Isend(p, tileElements, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
            }
        }
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        // Aggregate all tiles for a process row/column into one collective.
        // Chunk messages to keep MPI's int count valid for large matrices.
        auto broadcast = [](double* data, size_t count, int root, MPI_Comm comm) {
            while (count) {
                int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm);
                data += chunk; count -= chunk;
            }
        };
        int ra = std::min(nr, std::max(0, (k - rr + pr - 1) / pr));
        int cb = std::min(nc, std::max(0, (k - cc + pc - 1) / pc));
        if (ra < nr) broadcast(rowPanel.data() + static_cast<size_t>(ra) * tileElements,
                              static_cast<size_t>(nr - ra) * tileElements, k % pc, rowComm);
        if (cb < nc) broadcast(colPanel.data() + static_cast<size_t>(cb) * tileElements,
                              static_cast<size_t>(nc - cb) * tileElements, k % pr, colComm);
        // Store the column copy as L(j,k)^T for contiguous GEMM loads.
        for (int b = cb; b < nc; ++b) {
            double* p = colPanel.data() + static_cast<size_t>(b) * tileElements;
            for (int i = 0; i < block; ++i)
                for (int j = 0; j < i; ++j) std::swap(p[i * block + j], p[j * block + i]);
        }
    }

    bool factor() {
        std::vector<double> diagonal(tileElements);
        for (int k = 0; k < nt; ++k) {
            int w = width(k), failed = -1;
            int root = owner(k, k);
            if (rank == root) {
                double* d = at(k, k);
                for (int i = 0; i < w; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        double sum = 0.0;
                        for (int t = 0; t < j; ++t) sum += d[i * block + t] * d[j * block + t];
                        double value = d[i * block + j] - sum;
                        if (i == j) {
                            if (!(value > 0.0)) { failed = k * block + i; break; }
                            d[i * block + j] = std::sqrt(value);
                        } else d[i * block + j] = value / d[j * block + j];
                    }
                    if (failed >= 0) break;
                    for (int j = i + 1; j < block; ++j) d[i * block + j] = 0.0;
                }
            }
            MPI_Bcast(&failed, 1, MPI_INT, root, MPI_COMM_WORLD);
            if (failed >= 0) {
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
                return false;
            }
            // Only the process column owning the panel needs its diagonal.
            if (cc == k % pc) {
                if (rank == root) std::copy_n(at(k, k), tileElements, diagonal.data());
                MPI_Bcast(diagonal.data(), tileElements, MPI_DOUBLE, k % pr, colComm);
                for (int a = 0; a < nr; ++a) {
                    int i = rr + a * pr;
                    if (i <= k) continue;
                    double* t = at(i, k);
                    for (int r = 0; r < width(i); ++r)
                        for (int j = 0; j < w; ++j) {
                            double sum = 0.0;
                            for (int q = 0; q < j; ++q) sum += t[r * block + q] * diagonal[j * block + q];
                            t[r * block + j] = (t[r * block + j] - sum) / diagonal[j * block + j];
                        }
                }
            }
            panel(k);
            for (int a = 0; a < nr; ++a) {
                int i = rr + a * pr;
                if (i <= k) continue;
                for (int b = 0; b < nc; ++b) {
                    int j = cc + b * pc;
                    if (j <= k || j > i) continue;
                    multiply(at(i, j), rowPanel.data() + static_cast<size_t>(a) * tileElements, block,
                             colPanel.data() + static_cast<size_t>(b) * tileElements, block,
                             width(i), width(j), w, -1.0);
                }
            }
        }
        return true;
    }

    bool validate(const std::vector<std::vector<double>>& original) {
        auto reconstructed = original;
        for (auto& t : reconstructed) std::fill(t.begin(), t.end(), 0.0);
        for (int k = 0; k < nt; ++k) {
            panel(k);
            for (int a = 0; a < nr; ++a) {
                int i = rr + a * pr;
                if (i < k) continue;
                for (int b = 0; b < nc; ++b) {
                    int j = cc + b * pc;
                    if (j < k || j > i) continue;
                    multiply(reconstructed[static_cast<size_t>(a) * nc + b].data(),
                             rowPanel.data() + static_cast<size_t>(a) * tileElements, block,
                             colPanel.data() + static_cast<size_t>(b) * tileElements, block,
                             width(i), width(j), width(k), 1.0);
                }
            }
        }
        double error[2] = {0.0, 0.0}, global[2];
        for (int a = 0; a < nr; ++a) {
            int i = rr + a * pr;
            for (int b = 0; b < nc; ++b) {
                int j = cc + b * pc;
                if (j > i) continue;
                size_t t = static_cast<size_t>(a) * nc + b;
                for (int r = 0; r < width(i); ++r)
                    for (int c = 0; c < width(j); ++c) {
                        double expected = original[t][r * block + c];
                        double e = std::fabs(reconstructed[t][r * block + c] - expected);
                        if (!std::isfinite(e)) e = std::numeric_limits<double>::infinity();
                        error[0] = std::max(error[0], e);
                        error[1] = std::max(error[1], e / (std::fabs(expected) + 1e-10));
                    }
            }
        }
        MPI_Allreduce(error, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
        }
        return global[1] <= 1e-6;
    }

    void results() {
        // Gathering is optional and outside the benchmark timing.
        std::vector<double> full, buffer(tileElements);
        if (rank == 0) full.resize(static_cast<size_t>(n) * n);
        for (int i = 0; i < nt; ++i)
            for (int j = 0; j <= i; ++j) {
                int root = owner(i, j);
                const double* t = nullptr;
                if (rank == root) {
                    t = at(i, j);
                    if (rank != 0) MPI_Send(t, tileElements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                }
                if (rank == 0) {
                    if (root != 0) {
                        MPI_Recv(buffer.data(), tileElements, MPI_DOUBLE, root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        t = buffer.data();
                    }
                    for (int r = 0; r < width(i); ++r)
                        std::copy_n(t + r * block, width(j), full.data() + (static_cast<size_t>(i) * block + r) * n + j * block);
                }
            }
        if (rank == 0) print_results(full, "CholeskyL");
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int processes) {
    int n = 512;
    bool check = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = std::strtol(argv[++i], &end, 10);
            if (errno || *end || end == argv[i] || value < 0 || value > INT_MAX - block ||
                static_cast<size_t>(value) > std::numeric_limits<size_t>::max() / sizeof(double) / std::max(1L, value)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) check = true;
        else if (std::strcmp(argv[i], "-r") == 0) print = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nGenerating positive definite matrix...\n", check ? "enabled" : "disabled");
    }
    Matrix a(n, rank, processes);
    a.generate();
    std::vector<std::vector<double>> original;
    if (check) original = a.tiles;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = a.factor();
    double localTime = MPI_Wtime() - start, elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Performance: %.3f GFLOPS\n", elapsed > 0.0 ? static_cast<double>(n) * n * n / (3.0 * elapsed * 1e9) : 0.0);
    }
    if (print) a.results();
    if (check) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = a.validate(original);
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
    try { status = run(argc, argv, rank, processes); }
    catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
