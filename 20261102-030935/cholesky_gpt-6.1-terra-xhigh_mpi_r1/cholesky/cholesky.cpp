#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Rows are assigned in cyclic blocks. Each rank stores its rows in increasing
// global order, with the same row-major stride as the original matrix.
struct DistributedMatrix {
    size_t n;
    size_t block;
    int rank;
    int ranks;
    std::vector<size_t> rows;
    std::vector<double> data;

    DistributedMatrix(size_t size, size_t blockSize, int myRank, int numRanks)
        : n(size), block(blockSize), rank(myRank), ranks(numRanks) {
        for (size_t i = 0; i < n; ++i)
            if (owner(i) == rank) rows.push_back(i);
        data.resize(rows.size() * n);
    }

    int owner(size_t row) const { return static_cast<int>((row / block) % ranks); }
    size_t localIndex(size_t row) const {
        return (row / block / ranks) * block + row % block;
    }
};

void generatePositiveDefiniteMatrix(DistributedMatrix& A) {
    const size_t n = A.n;
    // Every rank reproduces the original random stream; rows of A are computed
    // locally, so no root process needs to generate or distribute the matrix.
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t local = 0; local < A.rows.size(); ++local) {
        const size_t i = A.rows[local];
        double* row = A.data.data() + local * n;
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            row[j] = sum;
        }
        row[i] += n;
    }
}

bool choleskyDecomposition(DistributedMatrix& A) {
    const size_t n = A.n;
    const size_t bs = A.block;
    std::vector<double> diagonal(bs * bs);
    std::vector<double> send, gathered, panel;
    std::vector<int> counts(A.ranks), offsets(A.ranks);

    for (size_t first = 0; first < n; first += bs) {
        const size_t width = std::min(bs, n - first);
        const size_t end = first + width;
        const int root = A.owner(first);
        int failed = 0;
        if (A.rank == root) {
            // Factor the diagonal block after all preceding panel updates.
            for (size_t i = 0; i < width && !failed; ++i) {
                double* row = A.data.data() + A.localIndex(first + i) * n;
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += row[first + k] * diagonal[j * width + k];
                    if (i == j) {
                        const double value = row[first + j] - sum;
                        if (value <= 0.0) {
                            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", first + j);
                            failed = 1;
                            break;
                        }
                        row[first + j] = std::sqrt(value);
                    } else {
                        row[first + j] = (row[first + j] - sum) / diagonal[j * width + j];
                    }
                    diagonal[i * width + j] = row[first + j];
                }
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (failed) return false;
        MPI_Bcast(diagonal.data(), static_cast<int>(width * width), MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Solve for this panel's rows, then exchange them for the trailing update.
        send.clear();
        for (size_t local = 0; local < A.rows.size(); ++local) {
            if (A.rows[local] < end) continue;
            double* row = A.data.data() + local * n;
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += row[first + k] * diagonal[j * width + k];
                row[first + j] = (row[first + j] - sum) / diagonal[j * width + j];
                send.push_back(row[first + j]);
            }
        }
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = end; i < n; ++i) counts[A.owner(i)] += static_cast<int>(width);
        int total = 0;
        for (int r = 0; r < A.ranks; ++r) { offsets[r] = total; total += counts[r]; }
        gathered.resize(total);
        MPI_Allgatherv(send.data(), static_cast<int>(send.size()), MPI_DOUBLE,
                       gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        panel.resize((n - end) * width);
        std::vector<int> cursor = offsets;
        for (size_t i = end; i < n; ++i) {
            const int owner = A.owner(i);
            std::copy_n(gathered.data() + cursor[owner], width, panel.data() + (i - end) * width);
            cursor[owner] += static_cast<int>(width);
        }

        // Update each local lower-triangular row. The inner loop traverses
        // contiguous panel values and is suitable for compiler vectorization.
        for (size_t local = 0; local < A.rows.size(); ++local) {
            const size_t i = A.rows[local];
            if (i < end) continue;
            double* row = A.data.data() + local * n;
            const double* pi = panel.data() + (i - end) * width;
            for (size_t j = end; j <= i; ++j) {
                const double* pj = panel.data() + (j - end) * width;
                double sum = 0.0;
                for (size_t k = 0; k < width; ++k) sum += pi[k] * pj[k];
                row[j] -= sum;
            }
        }
    }
    return true;
}

// Assemble rows only for the optional output and validation paths.
std::vector<double> gatherMatrix(const DistributedMatrix& A, bool allRanks) {
    std::vector<int> counts(A.ranks, 0), offsets(A.ranks);
    for (size_t i = 0; i < A.n; ++i) counts[A.owner(i)] += static_cast<int>(A.n);
    int total = 0;
    for (int r = 0; r < A.ranks; ++r) { offsets[r] = total; total += counts[r]; }
    std::vector<double> packed((allRanks || A.rank == 0) ? total : 0);
    if (allRanks)
        MPI_Allgatherv(A.data.data(), static_cast<int>(A.data.size()), MPI_DOUBLE,
                       packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    else
        MPI_Gatherv(A.data.data(), static_cast<int>(A.data.size()), MPI_DOUBLE,
                    packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> result;
    if (allRanks || A.rank == 0) {
        result.resize(A.n * A.n);
        std::vector<int> cursor = offsets;
        for (size_t i = 0; i < A.n; ++i) {
            int owner = A.owner(i);
            std::copy_n(packed.data() + cursor[owner], A.n, result.data() + i * A.n);
            cursor[owner] += static_cast<int>(A.n);
        }
    }
    return result;
}

bool validateCholesky(const DistributedMatrix& A, const std::vector<double>& original,
                      const std::vector<double>& L) {
    double localErrors[2] = {0.0, 0.0}, errors[2];
    for (size_t local = 0; local < A.rows.size(); ++local) {
        const size_t i = A.rows[local];
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) sum += L[i * A.n + k] * L[j * A.n + k];
            const double expected = original[local * A.n + j];
            const double error = std::fabs(sum - expected);
            localErrors[0] = std::max(localErrors[0], error);
            localErrors[1] = std::max(localErrors[1], error / (std::fabs(expected) + 1e-10));
        }
    }
    MPI_Allreduce(localErrors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (A.rank == 0) {
        std::printf("Max absolute error: %.10e\n", errors[0]);
        std::printf("Max relative error: %.10e\n", errors[1]);
        if (errors[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
    }
    return errors[1] <= 1e-6;
}

void printUsage(const char* progName) {
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
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = std::strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' || value > static_cast<unsigned long long>(std::sqrt(INT_MAX))) {
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
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            exitCode = 1;
            break;
        }
    }
    if (exitCode) { MPI_Finalize(); return exitCode; }

    const size_t block = std::min<size_t>(64, std::max<size_t>(1, (n + 4 * static_cast<size_t>(ranks) - 1) / (4 * static_cast<size_t>(ranks))));
    DistributedMatrix A(n, block, rank, ranks);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A);
    std::vector<double> original;
    if (validate) original = A.data;
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops / elapsed / 1e9);
    }
    std::vector<double> L;
    if (validate || printResults) L = gatherMatrix(A, validate);
    if (rank == 0 && printResults) print_results(L, "CholeskyL");
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateCholesky(A, original, L);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        if (!valid) exitCode = 1;
    }
    MPI_Finalize();
    return exitCode;
}
