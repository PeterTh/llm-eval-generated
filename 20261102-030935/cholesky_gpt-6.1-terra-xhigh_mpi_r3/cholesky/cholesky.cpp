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

// Block-cyclic row distribution balances the shrinking trailing matrix.
constexpr size_t blockSize = 32;

struct DistributedMatrix {
    size_t n;
    int rank, ranks;
    std::vector<size_t> rows;
    std::vector<double> data;

    DistributedMatrix(size_t size, int myRank, int numRanks)
        : n(size), rank(myRank), ranks(numRanks) {
        for (size_t i = 0; i < n; ++i)
            if (owner(i) == rank) rows.push_back(i);
        data.resize(rows.size() * n);
    }
    int owner(size_t row) const { return static_cast<int>((row / blockSize) % ranks); }
    size_t localIndex(size_t row) const {
        return (row / blockSize / ranks) * blockSize + row % blockSize;
    }
    double* row(size_t globalRow) { return data.data() + localIndex(globalRow) * n; }
    const double* row(size_t globalRow) const { return data.data() + localIndex(globalRow) * n; }
};

// B is generated in the original random-number order on each process, but only
// locally owned rows are retained. Broadcast blocks of B to form local rows of A.
void generatePositiveDefiniteMatrix(DistributedMatrix& A) {
    const size_t n = A.n;
    std::vector<double> B(A.rows.size() * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool local = A.owner(i) == A.rank;
        for (size_t j = 0; j < n; ++j) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (local) B[A.localIndex(i) * n + j] = value;
        }
    }
    std::vector<double> block(blockSize * n);
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t count = std::min(blockSize, n - first);
        const int root = A.owner(first);
        if (A.rank == root)
            std::copy_n(B.data() + A.localIndex(first) * n, count * n, block.data());
        MPI_Bcast(block.data(), static_cast<int>(count * n), MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (size_t r = 0; r < A.rows.size(); ++r) {
            const size_t i = A.rows[r];
            if (i < first) continue;
            const double* bi = B.data() + r * n;
            double* ai = A.data.data() + r * n;
            const size_t end = std::min(count, i - first + 1);
            for (size_t j = 0; j < end; ++j) {
                const double* bj = block.data() + j * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
                ai[first + j] = sum;
            }
            if (i < first + count) ai[i] += static_cast<double>(n);
        }
    }
}

bool choleskyDecomposition(DistributedMatrix& A) {
    const size_t n = A.n;
    std::vector<double> diagonal(blockSize * blockSize);
    std::vector<double> send(A.rows.size() * blockSize);
    std::vector<double> gathered(n * blockSize);
    // Transposed panel gives contiguous accesses during the trailing update.
    std::vector<double> panel(n * blockSize);
    std::vector<int> counts(A.ranks), displacements(A.ranks), offsets(A.ranks);

    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const size_t end = first + width;
        const int root = A.owner(first);
        std::fill(diagonal.begin(), diagonal.end(), 0.0);
        if (A.rank == root) {
            for (size_t i = 0; i < width; ++i) {
                double* ai = A.row(first + i);
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += ai[first + k] * diagonal[j * width + k];
                    if (i == j) {
                        const double value = ai[first + i] - sum;
                        if (value <= 0.0) {
                            diagonal[i * width + i] = -1.0;
                            break;
                        }
                        ai[first + i] = std::sqrt(value);
                    } else {
                        ai[first + j] = (ai[first + j] - sum) / diagonal[j * width + j];
                    }
                    diagonal[i * width + j] = ai[first + j];
                }
                if (diagonal[i * width + i] < 0.0) break;
            }
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(width * width), MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (size_t i = 0; i < width; ++i) {
            if (diagonal[i * width + i] < 0.0) {
                if (A.rank == 0)
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", first + i);
                return false;
            }
        }
        if (end == n) break;

        size_t sentRows = 0;
        for (size_t r = 0; r < A.rows.size(); ++r) {
            if (A.rows[r] < end) continue;
            double* ai = A.data.data() + r * n;
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += ai[first + k] * diagonal[j * width + k];
                ai[first + j] = (ai[first + j] - sum) / diagonal[j * width + j];
                send[sentRows * width + j] = ai[first + j];
            }
            ++sentRows;
        }
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = end; i < n; ++i) counts[A.owner(i)] += static_cast<int>(width);
        int total = 0;
        for (int p = 0; p < A.ranks; ++p) {
            displacements[p] = total;
            total += counts[p];
        }
        MPI_Allgatherv(send.data(), static_cast<int>(sentRows * width), MPI_DOUBLE,
                       gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        offsets = displacements;
        for (size_t i = end; i < n; ++i) {
            int& offset = offsets[A.owner(i)];
            for (size_t k = 0; k < width; ++k) panel[k * n + i] = gathered[offset++];
        }
        for (size_t r = 0; r < A.rows.size(); ++r) {
            const size_t i = A.rows[r];
            if (i < end) continue;
            double* __restrict__ ai = A.data.data() + r * n;
            for (size_t k = 0; k < width; ++k) {
                const double factor = ai[first + k];
                const double* __restrict__ pk = panel.data() + k * n;
                #pragma GCC ivdep
                for (size_t j = end; j <= i; ++j) ai[j] -= factor * pk[j];
            }
        }
    }
    return true;
}

bool validateCholesky(const DistributedMatrix& L, const std::vector<double>& original) {
    const size_t n = L.n;
    std::vector<double> block(blockSize * n);
    double errors[2] = {0.0, 0.0};
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t count = std::min(blockSize, n - first);
        const int root = L.owner(first);
        if (L.rank == root)
            std::copy_n(L.row(first), count * n, block.data());
        MPI_Bcast(block.data(), static_cast<int>(count * n), MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (size_t r = 0; r < L.rows.size(); ++r) {
            const size_t i = L.rows[r];
            if (i < first) continue;
            const double* li = L.data.data() + r * n;
            for (size_t j = 0; j < std::min(count, i - first + 1); ++j) {
                const double* lj = block.data() + j * n;
                double sum = 0.0;
                for (size_t k = 0; k <= first + j; ++k) sum += li[k] * lj[k];
                const double reference = original[r * n + first + j];
                const double error = std::fabs(sum - reference);
                errors[0] = std::max(errors[0], error);
                errors[1] = std::max(errors[1], error / (std::fabs(reference) + 1e-10));
            }
        }
    }
    double global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (L.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long value = std::strtoull(arg, &end, 10);
            if (arg[0] == '-' || end == arg || *end != '\0' || value > INT_MAX / blockSize ||
                value > std::numeric_limits<size_t>::max() / std::max(1ULL, value)) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    DistributedMatrix A(n, rank, ranks);
    generatePositiveDefiniteMatrix(A);
    std::vector<double> original;
    if (validate) original = A.data;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / maxElapsed / 1e9);
    }
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) result.resize(n * n);
        // Gather one block at a time, avoiding MPI's int count limit for n*n.
        for (size_t first = 0; first < n; first += blockSize) {
            const size_t count = std::min(blockSize, n - first) * n;
            const int owner = A.owner(first);
            if (rank == owner && owner != 0)
                MPI_Send(A.row(first), static_cast<int>(count), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                if (owner == 0) std::copy_n(A.row(first), count, result.data() + first * n);
                else MPI_Recv(result.data() + first * n, static_cast<int>(count), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (rank == 0) print_results(result, "CholeskyL");
    }
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateCholesky(A, original);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
