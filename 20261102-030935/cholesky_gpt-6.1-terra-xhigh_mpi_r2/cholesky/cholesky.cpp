#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Consecutive blocks of rows are distributed cyclically. This keeps the
// shrinking trailing matrix balanced while putting each diagonal block on one rank.
struct DistributedMatrix {
    size_t n, block;
    int rank, ranks;
    std::vector<size_t> rows;
    std::vector<double> values;

    DistributedMatrix(size_t dimension, size_t blockSize, int myRank, int numRanks)
        : n(dimension), block(blockSize), rank(myRank), ranks(numRanks) {
        for (size_t i = 0; i < n; ++i)
            if (owner(i) == rank) rows.push_back(i);
        values.resize(rows.size() * n, 0.0);
    }

    int owner(size_t row) const { return static_cast<int>((row / block) % ranks); }
    size_t localIndex(size_t row) const {
        return (row / block / ranks) * block + row % block;
    }
    double* row(size_t i) { return values.data() + localIndex(i) * n; }
    const double* row(size_t i) const { return values.data() + localIndex(i) * n; }
};

// Every rank generates the same B, preserving the original rand_r sequence.
// Only the owned lower-triangular rows of B * B^T are computed and stored.
void generatePositiveDefiniteMatrix(DistributedMatrix& A) {
    const size_t n = A.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i : A.rows) {
        double* a = A.row(i);
        const double* bi = B.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
            a[j] = sum;
        }
        a[i] += n;
    }
}

bool choleskyDecomposition(DistributedMatrix& A) {
    const size_t n = A.n, block = A.block;
    std::vector<double> diagonal(block * block);
    std::vector<double> send, panel(n * block);
    std::vector<int> counts(A.ranks), displacements(A.ranks);
    std::vector<size_t> offset(n);

    for (size_t start = 0; start < n; start += block) {
        const size_t width = std::min(block, n - start);
        const size_t end = start + width;
        const int root = A.owner(start);
        // The diagonal block has already received updates from earlier panels.
        int failed = 0;
        if (A.rank == root) {
            for (size_t i = 0; i < width && !failed; ++i) {
                double* ai = A.row(start + i);
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    const double* aj = A.row(start + j);
                    for (size_t k = 0; k < j; ++k)
                        sum += ai[start + k] * aj[start + k];
                    double value = ai[start + j] - sum;
                    if (i == j) {
                        if (value <= 0.0) {
                            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", start + j);
                            failed = 1;
                            break;
                        }
                        ai[start + j] = std::sqrt(value);
                    } else {
                        ai[start + j] = value / aj[start + j];
                    }
                }
            }
            if (!failed) {
                for (size_t i = 0; i < width; ++i)
                    for (size_t j = 0; j <= i; ++j)
                        diagonal[i * width + j] = A.row(start + i)[start + j];
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (failed) return false;
        MPI_Bcast(diagonal.data(), static_cast<int>(width * width), MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Solve the panel rows independently against the diagonal block.
        send.clear();
        for (size_t i : A.rows) {
            if (i < end) continue;
            double* ai = A.row(i);
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += ai[start + k] * diagonal[j * width + k];
                ai[start + j] = (ai[start + j] - sum) / diagonal[j * width + j];
                send.push_back(ai[start + j]);
            }
        }
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = end; i < n; ++i) counts[A.owner(i)] += static_cast<int>(width);
        int total = 0;
        for (int r = 0; r < A.ranks; ++r) {
            displacements[r] = total;
            total += counts[r];
        }
        MPI_Allgatherv(send.data(), static_cast<int>(send.size()), MPI_DOUBLE,
                       panel.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<int> next = displacements;
        for (size_t i = end; i < n; ++i) {
            const int owner = A.owner(i);
            offset[i] = static_cast<size_t>(next[owner]);
            next[owner] += static_cast<int>(width);
        }

        // Apply the rank-width update to the owned lower triangle.
        for (size_t i : A.rows) {
            if (i < end) continue;
            double* ai = A.row(i);
            const double* pi = panel.data() + offset[i];
            for (size_t j = end; j <= i; ++j) {
                const double* pj = panel.data() + offset[j];
                double sum = 0.0;
                for (size_t k = 0; k < width; ++k) sum += pi[k] * pj[k];
                ai[j] -= sum;
            }
        }
    }
    return true;
}

// MPI counts are int, so broadcast large contiguous blocks in chunks.
void broadcastDoubles(double* data, size_t count, int root) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

std::vector<double> collectFactor(const DistributedMatrix& A, bool allRanks) {
    std::vector<double> full;
    if (allRanks || A.rank == 0) full.resize(A.n * A.n);
    for (size_t first = 0; first < A.n; first += A.block) {
        const size_t rows = std::min(A.block, A.n - first);
        const size_t count = rows * A.n;
        const int owner = A.owner(first);
        if (allRanks) {
            if (A.rank == owner)
                std::copy_n(A.row(first), count, full.data() + first * A.n);
            broadcastDoubles(full.data() + first * A.n, count, owner);
        } else if (owner == 0) {
            if (A.rank == 0) std::copy_n(A.row(first), count, full.data() + first * A.n);
        } else {
            size_t done = 0;
            while (done < count) {
                int chunk = static_cast<int>(std::min(count - done, static_cast<size_t>(INT_MAX)));
                if (A.rank == owner)
                    MPI_Send(A.row(first) + done, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                else if (A.rank == 0)
                    MPI_Recv(full.data() + first * A.n + done, chunk, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                done += static_cast<size_t>(chunk);
            }
        }
    }
    return full;
}

bool validateCholesky(const DistributedMatrix& A, const std::vector<double>& original,
                      const std::vector<double>& factor) {
    double errors[2] = {0.0, 0.0};
    for (size_t local = 0; local < A.rows.size(); ++local) {
        const size_t i = A.rows[local];
        const double* li = factor.data() + i * A.n;
        for (size_t j = 0; j <= i; ++j) {
            const double* lj = factor.data() + j * A.n;
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) sum += li[k] * lj[k];
            const double value = original[local * A.n + j];
            const double error = std::fabs(sum - value);
            errors[0] = std::max(errors[0], error);
            errors[1] = std::max(errors[1], error / (std::fabs(value) + 1e-10));
        }
    }
    double global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (A.rank == 0) {
        std::printf("Max absolute error: %.10e\n", global[0]);
        std::printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
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
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = std::strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) std::printf("Invalid matrix size: %s\n", arg);
                exitCode = 1;
                break;
            }
            n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    const size_t block = std::max<size_t>(1, std::min<size_t>(64, n / (4 * static_cast<size_t>(ranks))));
    if (n > 0 && (n > std::numeric_limits<size_t>::max() / n ||
                  n > static_cast<size_t>(INT_MAX) / block)) {
        if (rank == 0) std::printf("Matrix size is too large\n");
        exitCode = 1;
    }
    if (exitCode) {
        MPI_Finalize();
        return exitCode;
    }
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    DistributedMatrix A(n, block, rank, ranks);
    generatePositiveDefiniteMatrix(A);
    std::vector<double> original;
    if (validate) original = A.values;
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops / duration / 1e9);
    }
    std::vector<double> factor;
    if (validate || printResults) factor = collectFactor(A, validate);
    if (printResults && rank == 0) print_results(factor, "CholeskyL");
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateCholesky(A, original, factor);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        if (!valid) exitCode = 1;
    }
    MPI_Finalize();
    return exitCode;
}
