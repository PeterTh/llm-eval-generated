#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are distributed in cyclic blocks. A block is also the unit of
// communication during factorization, so its owner has the entire diagonal
// block available locally.
struct DistributedMatrix {
    size_t n, block;
    int rank, ranks;
    std::vector<size_t> rows;
    std::vector<double> values;

    DistributedMatrix(size_t size, size_t blockSize, int myRank, int numRanks)
        : n(size), block(blockSize), rank(myRank), ranks(numRanks) {
        for (size_t first = 0; first < n; first += block) {
            if (owner(first) == rank) {
                for (size_t i = first; i < std::min(n, first + block); ++i)
                    rows.push_back(i);
            }
        }
        values.resize(rows.size() * n, 0.0);
    }

    int owner(size_t row) const { return static_cast<int>((row / block) % ranks); }
    size_t localIndex(size_t row) const {
        return (row / block / ranks) * block + row % block;
    }
    double* row(size_t globalRow) { return values.data() + localIndex(globalRow) * n; }
    const double* row(size_t globalRow) const { return values.data() + localIndex(globalRow) * n; }
};

// MPI's count parameter is an int; split large buffers when necessary.
static void broadcastDoubles(double* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

static double dot(const double* a, const double* b, size_t count) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (size_t k = 0; k < count; ++k) sum += a[k] * b[k];
    return sum;
}

static void generatePositiveDefiniteMatrix(DistributedMatrix& A, bool full) {
    const size_t n = A.n;
    // Advance the same random stream on every rank, retaining only local rows.
    // Streaming blocks of B avoids replicating the full input on every rank.
    std::vector<double> B(A.rows.size() * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool local = A.owner(i) == A.rank;
        for (size_t j = 0; j < n; ++j) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (local) B[A.localIndex(i) * n + j] = value;
        }
    }
    std::vector<double> panel(A.block * n);
    for (size_t first = 0; first < n; first += A.block) {
        const size_t end = std::min(n, first + A.block);
        const int root = A.owner(first);
        if (A.rank == root)
            std::copy_n(B.data() + A.localIndex(first) * n, (end - first) * n, panel.data());
        broadcastDoubles(panel.data(), (end - first) * n, root);
        for (size_t local = 0; local < A.rows.size(); ++local) {
            const size_t i = A.rows[local];
            const size_t limit = full ? end : std::min(end, i + 1);
            const double* bi = B.data() + local * n;
            double* a = A.values.data() + local * n;
            for (size_t j = first; j < limit; ++j)
                a[j] = dot(bi, panel.data() + (j - first) * n, n);
        }
    }
    for (size_t i : A.rows) A.row(i)[i] += n;
}

static bool choleskyDecomposition(DistributedMatrix& A) {
    const size_t n = A.n;
    std::vector<double> panel(1 + A.block * n);
    for (size_t first = 0; first < n; first += A.block) {
        const size_t end = std::min(n, first + A.block);
        const size_t width = end - first;
        const int root = A.owner(first);
        if (A.rank == root) {
            panel[0] = 0.0;
            for (size_t i = first; i < end; ++i) {
                double* ai = A.row(i);
                for (size_t j = first; j <= i; ++j) {
                    const double* aj = A.row(j);
                    const double sum = dot(ai, aj, j);
                    if (i == j) {
                        const double value = ai[j] - sum;
                        if (value <= 0.0) {
                            panel[0] = static_cast<double>(j + 1);
                            break;
                        }
                        ai[j] = std::sqrt(value);
                    } else {
                        ai[j] = (ai[j] - sum) / aj[j];
                    }
                }
                if (panel[0] != 0.0) break;
            }
            if (panel[0] == 0.0) {
                for (size_t i = first; i < end; ++i)
                    std::copy_n(A.row(i), end, panel.data() + 1 + (i - first) * end);
            }
        }
        broadcastDoubles(panel.data(), 1 + width * end, root);
        if (panel[0] != 0.0) {
            if (A.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                            static_cast<size_t>(panel[0]) - 1);
            return false;
        }
        for (size_t i : A.rows) {
            if (i < end) continue;
            double* ai = A.row(i);
            for (size_t j = first; j < end; ++j) {
                const double* aj = panel.data() + 1 + (j - first) * end;
                ai[j] = (ai[j] - dot(ai, aj, j)) / aj[j];
            }
        }
    }
    return true;
}

// Stream the finished rows to all ranks. Validation compares each rank's rows
// against every streamed row, and rank zero assembles the optional result.
static bool outputAndValidate(const DistributedMatrix& A,
                              const std::vector<double>& original,
                              bool printResults, bool validate) {
    const size_t n = A.n;
    std::vector<double> result;
    if (printResults && A.rank == 0) result.resize(n * n);
    std::vector<double> panel(A.block * n);
    double maxError = 0.0, relError = 0.0;
    for (size_t first = 0; first < n; first += A.block) {
        const size_t end = std::min(n, first + A.block);
        const int root = A.owner(first);
        if (A.rank == root)
            for (size_t j = first; j < end; ++j)
                std::copy_n(A.row(j), n, panel.data() + (j - first) * n);
        broadcastDoubles(panel.data(), (end - first) * n, root);
        if (printResults && A.rank == 0)
            std::copy_n(panel.data(), (end - first) * n, result.data() + first * n);
        if (validate) {
            for (size_t local = 0; local < A.rows.size(); ++local) {
                const size_t i = A.rows[local];
                const double* ai = A.values.data() + local * n;
                for (size_t j = first; j < end; ++j) {
                    const double* aj = panel.data() + (j - first) * n;
                    const double reconstructed = dot(ai, aj, std::min(i, j) + 1);
                    const double orig = original[local * n + j];
                    const double error = std::fabs(reconstructed - orig);
                    maxError = std::max(maxError, error);
                    relError = std::max(relError, error / (std::fabs(orig) + 1e-10));
                }
            }
        }
    }
    if (printResults && A.rank == 0) print_results(result, "CholeskyL");
    if (validate) {
        if (A.rank == 0) std::printf("Validating result...\n");
        double localErrors[2] = {maxError, relError}, errors[2];
        MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (A.rank == 0) {
            std::printf("Max absolute error: %.10e\n", errors[0]);
            std::printf("Max relative error: %.10e\n", errors[1]);
            if (errors[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
            std::printf("Validation: %s\n", errors[1] > 1e-6 ? "FAILED" : "PASSED");
        }
        return errors[1] <= 1e-6;
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    // Keep several blocks per rank for load balancing on small matrices.
    const size_t block = std::max<size_t>(1, std::min<size_t>(32, n / (static_cast<size_t>(ranks) * 4)));
    DistributedMatrix A(n, block, rank, ranks);
    generatePositiveDefiniteMatrix(A, validate);
    std::vector<double> original;
    if (validate) original = A.values;
    // The result is a lower triangular matrix, including when validation
    // required generating the upper triangle of the original matrix.
    if (validate)
        for (size_t i : A.rows) std::fill(A.row(i) + i + 1, A.row(i) + n, 0.0);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(A);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = ops / (duration.count() / 1000.0) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }
    const bool valid = !printResults && !validate ? true : outputAndValidate(A, original, printResults, validate);
    MPI_Finalize();
    return valid ? 0 : 1;
}
