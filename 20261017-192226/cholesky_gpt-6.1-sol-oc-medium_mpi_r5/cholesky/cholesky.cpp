#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

// Lower-triangular tiles are distributed cyclically over a 2D process grid.
// A rank holds O(n^2 / P) matrix data, plus O(n*b/sqrt(P)) panels.
struct DistributedMatrix {
    size_t n;
    int b, nt, rows, cols, row, col, localRows, localCols;
    MPI_Comm rowComm, colComm;
    std::vector<std::vector<double>> tiles;

    DistributedMatrix(size_t size, int rank, int processes) : n(size) {
        int dims[2] = {0, 0};
        MPI_Dims_create(processes, 2, dims);
        rows = dims[0];
        cols = dims[1];
        row = rank / cols;
        col = rank % cols;
        // Keep small problems distributed as well, and cap tiles for cache reuse.
        const size_t target = 2 * static_cast<size_t>(std::max(rows, cols));
        b = static_cast<int>(std::min<size_t>(64, std::max<size_t>(1, (n + target - 1) / target)));
        nt = static_cast<int>((n + b - 1) / b);
        localRows = nt / rows + (row < nt % rows);
        localCols = nt / cols + (col < nt % cols);
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        tiles.resize(static_cast<size_t>(localRows) * localCols);
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols)
                tile(i, j).resize(b * b, 0.0);
    }

    int extent(int i) const { return static_cast<int>(std::min<size_t>(b, n - static_cast<size_t>(i) * b)); }
    int owner(int i, int j) const { return (i % rows) * cols + j % cols; }
    std::vector<double>& tile(int i, int j) {
        return tiles[static_cast<size_t>(i / rows) * localCols + j / cols];
    }
    void releaseCommunicators() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
};

void generatePositiveDefiniteMatrix(DistributedMatrix& A) {
    const size_t n = A.n;
    const int b = A.b;
    // Retain only the B rows needed by this process row/column. Replaying the
    // original random stream preserves the input exactly for any process count.
    std::vector<double> br(static_cast<size_t>(A.localRows) * b * n);
    std::vector<double> bc(static_cast<size_t>(A.localCols) * b * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const int t = static_cast<int>(i / b);
        double* r = t % A.rows == A.row ? br.data() + (static_cast<size_t>(t / A.rows) * b + i % b) * n : nullptr;
        double* c = t % A.cols == A.col ? bc.data() + (static_cast<size_t>(t / A.cols) * b + i % b) * n : nullptr;
        for (size_t k = 0; k < n; ++k) {
            const double value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
            if (r) r[k] = value;
            if (c) c[k] = value;
        }
    }
    for (int ti = A.row; ti < A.nt; ti += A.rows) {
        for (int tj = A.col; tj <= ti; tj += A.cols) {
            auto& tile = A.tile(ti, tj);
            for (int i = 0; i < A.extent(ti); ++i) {
                const double* r = br.data() + (static_cast<size_t>(ti / A.rows) * b + i) * n;
                for (int j = 0; j < A.extent(tj); ++j) {
                    const double* c = bc.data() + (static_cast<size_t>(tj / A.cols) * b + j) * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += r[k] * c[k];
                    tile[i * b + j] = sum + (ti == tj && i == j ? n : 0);
                }
            }
        }
    }
}

bool choleskyDecomposition(DistributedMatrix& A, int rank) {
    const int b = A.b;
    std::vector<double> diagonal(b * b);
    std::vector<double> rowPanel(static_cast<size_t>(A.localRows) * b * b);
    std::vector<double> colPanel(static_cast<size_t>(A.localCols) * b * b);
    for (int k = 0; k < A.nt; ++k) {
        const int width = A.extent(k);
        if (rank == A.owner(k, k)) {
            auto& d = A.tile(k, k);
            bool valid = true;
            for (int i = 0; i < width && valid; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (int l = 0; l < j; ++l) sum += d[i * b + l] * d[j * b + l];
                    const double value = d[i * b + j] - sum;
                    if (i == j) {
                        if (!(value > 0.0)) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", static_cast<size_t>(k) * b + i);
                            valid = false;
                            break;
                        }
                        d[i * b + j] = sqrt(value);
                    } else {
                        d[i * b + j] = value / d[j * b + j];
                    }
                }
            }
            diagonal = d;
            if (!valid) diagonal[0] = -1.0;
        }
        // Failure is carried with the diagonal so all ranks exit collectively.
        MPI_Bcast(diagonal.data(), b * b, MPI_DOUBLE, A.owner(k, k), MPI_COMM_WORLD);
        if (diagonal[0] < 0.0) return false;

        if (A.col == k % A.cols) {
            for (int i = A.row; i < A.nt; i += A.rows) {
                if (i <= k) continue;
                auto& t = A.tile(i, k);
                for (int r = 0; r < A.extent(i); ++r) {
                    for (int j = 0; j < width; ++j) {
                        double sum = 0.0;
                        for (int l = 0; l < j; ++l) sum += t[r * b + l] * diagonal[j * b + l];
                        t[r * b + j] = (t[r * b + j] - sum) / diagonal[j * b + j];
                    }
                }
            }
        }
        // First distribute L_ik across its process row, then down the process
        // column that owns column i. No complete panel is replicated globally.
        for (int i = A.row; i < A.nt; i += A.rows) {
            if (i <= k) continue;
            double* panel = rowPanel.data() + static_cast<size_t>(i / A.rows) * b * b;
            if (A.col == k % A.cols)
                std::copy(A.tile(i, k).begin(), A.tile(i, k).end(), panel);
            MPI_Bcast(panel, b * b, MPI_DOUBLE, k % A.cols, A.rowComm);
        }
        for (int j = A.col; j < A.nt; j += A.cols) {
            if (j <= k) continue;
            double* panel = colPanel.data() + static_cast<size_t>(j / A.cols) * b * b;
            if (A.row == j % A.rows) {
                const double* source = rowPanel.data() + static_cast<size_t>(j / A.rows) * b * b;
                std::copy(source, source + b * b, panel);
            }
            MPI_Bcast(panel, b * b, MPI_DOUBLE, j % A.rows, A.colComm);
        }
        for (int i = A.row; i < A.nt; i += A.rows) {
            if (i <= k) continue;
            const double* left = rowPanel.data() + static_cast<size_t>(i / A.rows) * b * b;
            for (int j = A.col; j <= i; j += A.cols) {
                if (j <= k) continue;
                const double* right = colPanel.data() + static_cast<size_t>(j / A.cols) * b * b;
                auto& t = A.tile(i, j);
                // Four independent dot products reuse each row of the left
                // operand and let the compiler vectorize the reduction loop.
                for (int r = 0; r < A.extent(i); ++r) {
                    const int end = i == j ? r + 1 : A.extent(j);
                    int c = 0;
                    for (; c + 3 < end; c += 4) {
                        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        for (int l = 0; l < width; ++l) {
                            const double v = left[r * b + l];
                            s0 += v * right[c * b + l];
                            s1 += v * right[(c + 1) * b + l];
                            s2 += v * right[(c + 2) * b + l];
                            s3 += v * right[(c + 3) * b + l];
                        }
                        t[r * b + c] -= s0;
                        t[r * b + c + 1] -= s1;
                        t[r * b + c + 2] -= s2;
                        t[r * b + c + 3] -= s3;
                    }
                    for (; c < end; ++c) {
                        double sum = 0.0;
                        for (int l = 0; l < width; ++l) sum += left[r * b + l] * right[c * b + l];
                        t[r * b + c] -= sum;
                    }
                }
            }
        }
    }
    return true;
}

// Gathering is needed only for the optional external report and validation.
std::vector<double> gatherMatrix(DistributedMatrix& A, int rank, bool symmetric) {
    std::vector<double> result;
    if (rank == 0) result.resize(A.n * A.n, 0.0);
    std::vector<double> buffer(A.b * A.b);
    for (int i = 0; i < A.nt; ++i) {
        for (int j = 0; j <= i; ++j) {
            const int owner = A.owner(i, j);
            if (rank == owner && owner != 0)
                MPI_Send(A.tile(i, j).data(), A.b * A.b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (rank != 0) continue;
            const double* data;
            if (owner == 0) data = A.tile(i, j).data();
            else {
                MPI_Recv(buffer.data(), A.b * A.b, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                data = buffer.data();
            }
            for (int r = 0; r < A.extent(i); ++r) {
                for (int c = 0; c < A.extent(j); ++c) {
                    const size_t gr = static_cast<size_t>(i) * A.b + r;
                    const size_t gc = static_cast<size_t>(j) * A.b + c;
                    if (gc > gr) continue;
                    result[gr * A.n + gc] = data[r * A.b + c];
                    if (symmetric) result[gc * A.n + gr] = data[r * A.b + c];
                }
            }
        }
    }
    return result;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
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

int runBenchmark(int argc, char** argv, int rank, int processes) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* text = argv[++i];
            const unsigned long long value = strtoull(text, &end, 10);
            if (errno || text[0] == '-' || end == text || *end ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", text);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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
    }
    
    // Allocate matrix
    DistributedMatrix A(n, rank, processes);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A);
    
    if (validate) {
        A_orig = gatherMatrix(A, rank, true);
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    bool success = choleskyDecomposition(A, rank);
    
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        A.releaseCommunicators();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
    
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = duration > 0.0 ? ops / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (printResults || validate) result = gatherMatrix(A, rank, false);
    A.releaseCommunicators();
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(result, "CholeskyL");
    }
    
    // Validation
    int status = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(result, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    // Finalize MPI on every normal return, including help and invalid options.
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalizeMPI;
    try {
        return runBenchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
