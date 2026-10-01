#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
// Use the MPI C interface; legacy MPI C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

namespace {
// Fixed-size tiles allow contiguous, vectorizable updates without a BLAS dependency.
constexpr int BS = 64;
constexpr int AREA = BS * BS;
struct Tile {
    int i, j;
    std::vector<double> a;
    Tile(int row, int col) : i(row), j(col), a(AREA, 0.0) {}
};

struct Matrix {
    size_t n;
    int nt, rank, rows, cols, row, col;
    MPI_Comm rowComm, colComm;
    std::vector<Tile> tiles;
    Matrix(size_t size) : n(size), nt(static_cast<int>((n + BS - 1) / BS)) {
        int processes, dims[2] = {0, 0};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Dims_create(processes, 2, dims);
        rows = dims[0]; cols = dims[1];
        row = rank / cols; col = rank % cols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols)
                tiles.emplace_back(i, j);
    }
    ~Matrix() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int extent(int i) const { return static_cast<int>(std::min(size_t(BS), n - size_t(i) * BS)); }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }

    void generate() {
        // Reproduce the original rand_r stream exactly. Retain only the B rows
        // needed by this process row, rather than replicating the full matrix.
        const size_t localBlocks = (nt + rows - 1 - row) / rows;
        std::vector<double> b(localBlocks * BS * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            const bool mine = (i / BS) % rows == size_t(row);
            double* dest = mine ? b.data() + ((i / BS / rows) * BS + i % BS) * n : nullptr;
            for (size_t k = 0; k < n; ++k) {
                double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
                if (mine) dest[k] = value;
            }
        }
        // Broadcast a transposed block of B rows; each rank forms only its
        // own lower-triangular tiles of B B^T, vectorizing over output columns.
        std::vector<double> panel(n * BS);
        for (int j = 0; j < nt; ++j) {
            int width = extent(j);
            int root = owner(j, 0);
            if (rank == root) {
                const double* src = b.data() + size_t(j / rows) * BS * n;
                for (size_t k = 0; k < n; ++k)
                    for (int c = 0; c < width; ++c)
                        panel[k * BS + c] = src[size_t(c) * n + k];
            }
            // Chunk broadcasts to stay within MPI's int count limit.
            for (size_t offset = 0; offset < panel.size();) {
                int count = static_cast<int>(std::min(panel.size() - offset, size_t(INT_MAX)));
                MPI_Bcast(panel.data() + offset, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
                offset += count;
            }
            for (auto& t : tiles) {
                if (t.j != j) continue;
                const double* src = b.data() + size_t(t.i / rows) * BS * n;
                for (int r = 0; r < extent(t.i); ++r) {
                    double* out = t.a.data() + r * BS;
                    for (size_t k = 0; k < n; ++k) {
                        double x = src[size_t(r) * n + k];
                        const double* right = panel.data() + k * BS;
                        for (int c = 0; c < width; ++c) out[c] += x * right[c];
                    }
                }
                if (t.i == j)
                    for (int r = 0; r < width; ++r) t.a[r * BS + r] += n;
            }
        }
    }

    bool factor() {
        std::vector<double> diagonal(AREA);
        std::vector<double> rowPanel(size_t((nt + rows - 1 - row) / rows) * AREA);
        std::vector<double> colPanel(size_t((nt + cols - 1 - col) / cols) * AREA);
        for (int k = 0; k < nt; ++k) {
            int width = extent(k), bad = -1;
            if (rank == owner(k, k)) {
                for (auto& t : tiles) if (t.i == k && t.j == k) {
                    for (int j = 0; j < width; ++j) {
                        double d = t.a[j * BS + j];
                        for (int h = 0; h < j; ++h) d -= t.a[j * BS + h] * t.a[j * BS + h];
                        if (!(d > 0.0)) { bad = k * BS + j; break; }
                        t.a[j * BS + j] = std::sqrt(d);
                        for (int i = j + 1; i < width; ++i) {
                            double x = t.a[i * BS + j];
                            for (int h = 0; h < j; ++h) x -= t.a[i * BS + h] * t.a[j * BS + h];
                            t.a[i * BS + j] = x / t.a[j * BS + j];
                        }
                        for (int c = j + 1; c < width; ++c) t.a[j * BS + c] = 0.0;
                    }
                    diagonal = t.a;
                }
            }
            MPI_Bcast(&bad, 1, MPI_INT, owner(k, k), MPI_COMM_WORLD);
            if (bad >= 0) {
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", bad);
                return false;
            }
            MPI_Bcast(diagonal.data(), AREA, MPI_DOUBLE, owner(k, k), MPI_COMM_WORLD);
            // Solve L_ik L_kk^T = A_ik on the owners of the current tile column.
            for (auto& t : tiles) if (t.j == k && t.i > k) {
                for (int i = 0; i < extent(t.i); ++i)
                    for (int j = 0; j < width; ++j) {
                        double x = t.a[i * BS + j];
                        for (int h = 0; h < j; ++h) x -= t.a[i * BS + h] * diagonal[j * BS + h];
                        t.a[i * BS + j] = x / diagonal[j * BS + j];
                    }
                std::copy(t.a.begin(), t.a.end(), rowPanel.data() + size_t(t.i / rows) * AREA);
            }
            // Each panel travels first along its process row, then along the
            // process column that needs its transpose. No full panel replication.
            for (int i = row; i < nt; i += rows) if (i > k)
                MPI_Bcast(rowPanel.data() + size_t(i / rows) * AREA, AREA, MPI_DOUBLE, k % cols, rowComm);
            for (int j = col; j < nt; j += cols) if (j > k) {
                double* dest = colPanel.data() + size_t(j / cols) * AREA;
                if (row == j % rows) {
                    const double* src = rowPanel.data() + size_t(j / rows) * AREA;
                    for (int r = 0; r < BS; ++r)
                        for (int c = 0; c < BS; ++c) dest[r * BS + c] = src[c * BS + r];
                }
                MPI_Bcast(dest, AREA, MPI_DOUBLE, j % rows, colComm);
            }
            for (auto& t : tiles) if (t.j > k) {
                const double* left = rowPanel.data() + size_t(t.i / rows) * AREA;
                const double* right = colPanel.data() + size_t(t.j / cols) * AREA;
                // Register-blocked matrix product: four rows share each load
                // of the right panel, with contiguous inner loops for SIMD.
                int height = extent(t.i), columns = extent(t.j), i = 0;
                for (; i + 3 < height; i += 4) {
                    double* out = t.a.data() + i * BS;
                    for (int h = 0; h < width; ++h) {
                        double a0 = left[i * BS + h], a1 = left[(i + 1) * BS + h];
                        double a2 = left[(i + 2) * BS + h], a3 = left[(i + 3) * BS + h];
                        for (int j = 0; j < columns; ++j) {
                            double b = right[h * BS + j];
                            out[j] -= a0 * b; out[BS + j] -= a1 * b;
                            out[2 * BS + j] -= a2 * b; out[3 * BS + j] -= a3 * b;
                        }
                    }
                }
                for (; i < height; ++i)
                    for (int h = 0; h < width; ++h)
                        for (int j = 0; j < columns; ++j)
                            t.a[i * BS + j] -= left[i * BS + h] * right[h * BS + j];
            }
        }
        return true;
    }

    // Only optional output/validation gathers the matrix; the timed factorization
    // and default benchmark keep A distributed across the process grid.
    std::vector<double> gather(bool symmetric) const {
        std::vector<double> result(rank == 0 ? n * n : 0), buffer(AREA);
        size_t local = 0;
        for (int i = 0; i < nt; ++i) for (int j = 0; j <= i; ++j) {
            int source = owner(i, j);
            const double* data = nullptr;
            if (rank == source) {
                data = tiles[local++].a.data();
                if (rank != 0) MPI_Send(data, AREA, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            if (rank == 0) {
                if (source != 0) {
                    MPI_Recv(buffer.data(), AREA, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    data = buffer.data();
                }
                for (int r = 0; r < extent(i); ++r) for (int c = 0; c < extent(j); ++c) {
                    size_t gr = size_t(i) * BS + r, gc = size_t(j) * BS + c;
                    if (gc > gr) continue;
                    result[gr * n + gc] = data[r * BS + c];
                    if (symmetric) result[gc * n + gr] = data[r * BS + c];
                }
            }
        }
        return result;
    }
};
} // namespace

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= std::min(i, j); ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t n = 512;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || end == arg || *end || value > INT_MAX ||
                value > std::sqrt(double(std::numeric_limits<size_t>::max() / sizeof(double)))) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                status = 1; break;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            status = 1; break;
        }
    }
    if (status) { MPI_Finalize(); return status; }
    try {
        Matrix a(n);
        if (rank == 0) {
            printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Generating positive definite matrix...\n");
        }
        a.generate();
        std::vector<double> original;
        if (validate) original = a.gather(true);
        if (rank == 0) printf("Computing Cholesky decomposition...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        bool success = a.factor();
        double elapsed = MPI_Wtime() - start, seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!success) {
            if (rank == 0) printf("Cholesky decomposition failed\n");
            status = 1;
        } else {
            if (rank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
                double ops = double(n) * n * n / 3.0;
                printf("Performance: %.3f GFLOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
            }
            std::vector<double> result;
            if (printResults || validate) result = a.gather(false);
            if (rank == 0) {
                if (printResults) print_results(result, "CholeskyL");
                if (validate) {
                    printf("Validating result...\n");
                    bool valid = validateCholesky(result, original, n);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
