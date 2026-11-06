#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Consecutive rows in a block reside on one rank; blocks are assigned cyclically.
// This balances the diminishing amount of work in the trailing matrix while
// allowing a complete diagonal panel to be broadcast at once.
static size_t blockSize = 32;

static int ownerOf(size_t row, int processes) {
    return static_cast<int>((row / blockSize) % static_cast<size_t>(processes));
}

static size_t localRow(size_t row, int processes) {
    return (row / blockSize / static_cast<size_t>(processes)) * blockSize + row % blockSize;
}

// MPI counts are ints, whereas a matrix or panel can contain more elements.
static void broadcastDoubles(double* data, size_t count, int root) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void transferDoubles(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static bool choleskyDecomposition(std::vector<double>& A, size_t n,
                                  const std::vector<size_t>& rows,
                                  int rank, int processes) {
    std::vector<double> panel;
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t end = std::min(n, first + blockSize);
        const size_t width = end - first;
        const int owner = ownerOf(first, processes);
        panel.resize(width * end);
        int failed = 0;
        if (rank == owner) {
            for (size_t i = first; i < end && !failed; ++i) {
                double* row = A.data() + localRow(i, processes) * n;
                for (size_t j = first; j <= i; ++j) {
                    const double* pivot = A.data() + localRow(j, processes) * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
                    if (i == j) {
                        const double val = row[j] - sum;
                        if (val <= 0.0) {
                            failed = static_cast<int>(j - first) + 1;
                            break;
                        }
                        row[j] = std::sqrt(val);
                    } else {
                        row[j] = (row[j] - sum) / pivot[j];
                    }
                }
                std::copy(row, row + end, panel.data() + (i - first) * end);
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) {
            if (rank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                                       first + static_cast<size_t>(failed - 1));
            return false;
        }
        broadcastDoubles(panel.data(), panel.size(), owner);

        for (size_t local = 0; local < rows.size(); ++local) {
            if (rows[local] < end) continue;
            double* row = A.data() + local * n;
            for (size_t j = first; j < end; ++j) {
                const double* pivot = panel.data() + (j - first) * end;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += row[k] * pivot[k];
                row[j] = (row[j] - sum) / pivot[j];
            }
        }
    }
    return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n,
                                           const std::vector<size_t>& rows,
                                           int rank, int processes) {
    // Reproduce the same random stream on every rank, retaining only local rows.
    std::vector<double> B(rows.size() * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool local = ownerOf(i, processes) == rank;
        for (size_t j = 0; j < n; ++j) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (local) B[localRow(i, processes) * n + j] = value;
        }
    }
    std::vector<double> panel;
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const int owner = ownerOf(first, processes);
        panel.resize(width * n);
        if (rank == owner) {
            const double* begin = B.data() + localRow(first, processes) * n;
            std::copy(begin, begin + width * n, panel.data());
        }
        broadcastDoubles(panel.data(), panel.size(), owner);
        for (size_t local = 0; local < rows.size(); ++local) {
            const double* row = B.data() + local * n;
            for (size_t j = 0; j < width; ++j) {
                const double* other = panel.data() + j * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += row[k] * other[k];
                A[local * n + first + j] = sum;
            }
            if (rows[local] >= first && rows[local] < first + width)
                A[local * n + rows[local]] += n;
        }
    }
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                             size_t n, const std::vector<size_t>& rows, int rank, int processes) {
    double errors[2] = {0.0, 0.0};
    std::vector<double> panel;
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const int owner = ownerOf(first, processes);
        panel.resize(width * n);
        if (rank == owner) {
            const double* begin = L.data() + localRow(first, processes) * n;
            std::copy(begin, begin + width * n, panel.data());
        }
        broadcastDoubles(panel.data(), panel.size(), owner);
        for (size_t local = 0; local < rows.size(); ++local) {
            const double* row = L.data() + local * n;
            for (size_t j = first; j < first + width; ++j) {
                const double* other = panel.data() + (j - first) * n;
                double sum = 0.0;
                for (size_t k = 0; k <= std::min(rows[local], j); ++k) sum += row[k] * other[k];
                const double reference = original[local * n + j];
                const double error = std::fabs(sum - reference);
                errors[0] = std::max(errors[0], error);
                errors[1] = std::max(errors[1], error / (std::fabs(reference) + 1e-10));
            }
        }
    }
    double globalErrors[2];
    MPI_Allreduce(errors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Max absolute error: %.10e\n", globalErrors[0]);
        std::printf("Max relative error: %.10e\n", globalErrors[1]);
        if (globalErrors[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
    }
    return globalErrors[1] <= 1e-6;
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::atoi(argv[++i]));
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
            MPI_Finalize();
            return 1;
        }
    }
    if (n != 0 && n > std::numeric_limits<size_t>::max() / n / sizeof(double)) {
        if (rank == 0) std::fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }
    // Keep several blocks per rank for balance, including at small sizes.
    blockSize = std::max<size_t>(1, std::min<size_t>(32, n / (4 * static_cast<size_t>(processes))));
    try {
        std::vector<size_t> rows;
        for (size_t i = 0; i < n; ++i) if (ownerOf(i, processes) == rank) rows.push_back(i);
        std::vector<double> A(rows.size() * n), original;
        if (rank == 0) {
            std::printf("Cholesky Decomposition Benchmark\n");
            std::printf("Matrix size: %zu x %zu\n", n, n);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("Generating positive definite matrix...\n");
        }
        generatePositiveDefiniteMatrix(A, n, rows, rank, processes);
        if (validate) original = A;
        for (size_t local = 0; local < rows.size(); ++local)
            std::fill(A.begin() + local * n + rows[local] + 1, A.begin() + (local + 1) * n, 0.0);

        if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        const bool success = choleskyDecomposition(A, n, rows, rank, processes);
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!success) {
            if (rank == 0) std::printf("Cholesky decomposition failed\n");
            MPI_Finalize();
            return 1;
        }
        if (rank == 0) {
            const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
            std::printf("Computation time: %ld ms\n", milliseconds);
            const double ops = static_cast<double>(n) * n * n / 3.0;
            std::printf("Performance: %.3f GFLOPS\n", ops / maxElapsed / 1e9);
        }
        if (printResults) {
            std::vector<double> full;
            if (rank == 0) full.resize(n * n);
            for (size_t first = 0; first < n; first += blockSize) {
                const size_t width = std::min(blockSize, n - first);
                const int owner = ownerOf(first, processes);
                if (owner == 0 && rank == 0) {
                    const double* begin = A.data() + localRow(first, processes) * n;
                    std::copy(begin, begin + width * n, full.data() + first * n);
                } else if (rank == owner) {
                    transferDoubles(A.data() + localRow(first, processes) * n, width * n, 0, true);
                } else if (rank == 0) {
                    transferDoubles(full.data() + first * n, width * n, owner, false);
                }
            }
            if (rank == 0) print_results(full, "CholeskyL");
        }
        bool valid = true;
        if (validate) {
            if (rank == 0) std::printf("Validating result...\n");
            valid = validateCholesky(A, original, n, rows, rank, processes);
            if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
