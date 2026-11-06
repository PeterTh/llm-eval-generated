#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Consecutive blocks of rows are assigned cyclically to balance the shrinking
// trailing matrix. All ranks know the row mapping, including partial blocks.
struct Distribution {
    size_t n, block;
    int rank, size;
    std::vector<std::vector<size_t>> rows;

    Distribution(size_t n_, int rank_, int size_)
        : n(n_), block(std::min<size_t>(32, std::max<size_t>(1, (n_ + 4 * size_ - 1) / (4 * size_)))),
          rank(rank_), size(size_), rows(size_) {
        for (size_t i = 0; i < n; ++i)
            rows[(i / block) % size].push_back(i);
    }
    int owner(size_t row) const { return static_cast<int>((row / block) % size); }
    size_t localIndex(size_t row) const { return (row / (block * size)) * block + row % block; }
};

void generatePositiveDefiniteMatrix(std::vector<double>& A, const Distribution& d) {
    const size_t n = d.n;
    const auto& localRows = d.rows[d.rank];
    std::vector<double> localB(localRows.size() * n);
    unsigned int seed = 42;
    // Advance the same random stream on every rank, storing only owned rows.
    size_t local = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool owned = d.owner(i) == d.rank;
        for (size_t k = 0; k < n; ++k) {
            const double value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            if (owned) localB[local * n + k] = value;
        }
        if (owned) ++local;
    }

    size_t maxRows = 0;
    for (const auto& rows : d.rows) maxRows = std::max(maxRows, rows.size());
    std::vector<double> remoteB(maxRows * n);
    for (int owner = 0; owner < d.size; ++owner) {
        const size_t count = d.rows[owner].size() * n;
        double* source = owner == d.rank ? localB.data() : remoteB.data();
        MPI_Bcast(source, static_cast<int>(count), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (size_t li = 0; li < localRows.size(); ++li) {
            const size_t i = localRows[li];
            for (size_t lj = 0; lj < d.rows[owner].size(); ++lj) {
                const size_t j = d.rows[owner][lj];
                if (j > i) break;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k)
                    sum += localB[li * n + k] * source[lj * n + k];
                A[li * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const Distribution& d) {
    const size_t n = d.n;
    std::vector<double> diagonal(d.block * d.block + 1);
    std::vector<double> send(n * d.block), received(n * d.block), panel(n * d.block);
    std::vector<int> counts(d.size), displacements(d.size);

    for (size_t first = 0; first < n; first += d.block) {
        const size_t width = std::min(d.block, n - first);
        const size_t end = first + width;
        const int root = d.owner(first);
        if (d.rank == root) {
            diagonal[0] = -1.0;
            for (size_t i = 0; i < width; ++i) {
                double* row = A.data() + d.localIndex(first + i) * n;
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += row[first + k] * diagonal[1 + j * width + k];
                    const double value = row[first + j] - sum;
                    if (i == j) {
                        if (value <= 0.0) {
                            diagonal[0] = static_cast<double>(first + i);
                            break;
                        }
                        row[first + j] = std::sqrt(value);
                    } else {
                        row[first + j] = value / diagonal[1 + j * width + j];
                    }
                    diagonal[1 + i * width + j] = row[first + j];
                }
                if (diagonal[0] >= 0.0) break;
            }
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(1 + width * width), MPI_DOUBLE, root, MPI_COMM_WORLD);
        if (diagonal[0] >= 0.0) {
            if (d.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", static_cast<size_t>(diagonal[0]));
            return false;
        }

        size_t packed = 0;
        for (size_t local = 0; local < d.rows[d.rank].size(); ++local) {
            const size_t i = d.rows[d.rank][local];
            if (i < end) continue;
            double* row = A.data() + local * n;
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += row[first + k] * diagonal[1 + j * width + k];
                row[first + j] = (row[first + j] - sum) / diagonal[1 + j * width + j];
                send[packed++] = row[first + j];
            }
        }
        int displacement = 0;
        for (int r = 0; r < d.size; ++r) {
            const auto begin = std::lower_bound(d.rows[r].begin(), d.rows[r].end(), end);
            counts[r] = static_cast<int>((d.rows[r].end() - begin) * width);
            displacements[r] = displacement;
            displacement += counts[r];
        }
        MPI_Allgatherv(send.data(), static_cast<int>(packed), MPI_DOUBLE, received.data(), counts.data(),
                       displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (int r = 0; r < d.size; ++r) {
            size_t pos = displacements[r];
            for (size_t i : d.rows[r]) {
                if (i < end) continue;
                for (size_t k = 0; k < width; ++k)
                    panel[k * n + i] = received[pos++];
            }
        }
        // Rank-b updates with contiguous inner loops for vectorization and cache reuse.
        for (size_t local = 0; local < d.rows[d.rank].size(); ++local) {
            const size_t i = d.rows[d.rank][local];
            if (i < end) continue;
            double* row = A.data() + local * n;
            for (size_t base = end; base <= i; base += 256) {
                const size_t limit = std::min(i + 1, base + 256);
                for (size_t k = 0; k < width; ++k) {
                    const double factor = row[first + k];
                    const double* p = panel.data() + k * n;
                    for (size_t j = base; j < limit; ++j)
                        row[j] -= factor * p[j];
                }
            }
        }
    }
    return true;
}

// Gather rows in rank order, then restore global row order. Used only for
// validation or external result reporting, outside the timed factorization.
std::vector<double> gatherFactor(const std::vector<double>& A, const Distribution& d, bool all) {
    std::vector<int> counts(d.size), displacements(d.size);
    int offset = 0;
    for (int r = 0; r < d.size; ++r) {
        counts[r] = static_cast<int>(d.rows[r].size() * d.n);
        displacements[r] = offset;
        offset += counts[r];
    }
    std::vector<double> packed((all || d.rank == 0) ? d.n * d.n : 0);
    if (all)
        MPI_Allgatherv(A.data(), counts[d.rank], MPI_DOUBLE, packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    else
        MPI_Gatherv(A.data(), counts[d.rank], MPI_DOUBLE, packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> L((all || d.rank == 0) ? d.n * d.n : 0);
    if (all || d.rank == 0) {
        for (int r = 0; r < d.size; ++r) {
            size_t pos = displacements[r];
            for (size_t i : d.rows[r]) {
                std::copy_n(packed.data() + pos, d.n, L.data() + i * d.n);
                pos += d.n;
            }
        }
    }
    return L;
}

bool validateCholesky(const std::vector<double>& A, const std::vector<double>& original,
                      const std::vector<double>& L, const Distribution& d) {
    double errors[2] = {0.0, 0.0};
    for (size_t local = 0; local < d.rows[d.rank].size(); ++local) {
        const size_t i = d.rows[d.rank][local];
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k)
                sum += A[local * d.n + k] * L[j * d.n + k];
            const double expected = original[local * d.n + j];
            const double error = std::fabs(sum - expected);
            errors[0] = std::max(errors[0], error);
            errors[1] = std::max(errors[1], error / (std::fabs(expected) + 1e-10));
        }
    }
    double global[2];
    MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (d.rank == 0) {
        std::printf("Max absolute error: %.10e\n", global[0]);
        std::printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
    }
    return global[1] <= 1e-6;
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    // MPI's standard variable-count collectives use int counts/displacements.
    if (n > 0 && n > static_cast<size_t>(INT_MAX) / n) {
        if (rank == 0) std::fprintf(stderr, "Matrix size exceeds MPI count limits\n");
        MPI_Finalize();
        return 1;
    }
    Distribution d(n, rank, size);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    std::vector<double> A(d.rows[rank].size() * n);
    generatePositiveDefiniteMatrix(A, d);
    std::vector<double> original;
    if (validate) original = A;
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, d);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops / maxElapsed / 1e9);
    }
    std::vector<double> L;
    if (printResults || validate) L = gatherFactor(A, d, validate);
    if (printResults && rank == 0) print_results(L, "CholeskyL");
    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateCholesky(A, original, L, d);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
