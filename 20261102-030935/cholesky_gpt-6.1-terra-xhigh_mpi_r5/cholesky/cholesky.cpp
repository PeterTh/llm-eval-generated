#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Block-cyclic row ownership balances the shrinking trailing matrix.
constexpr size_t blockSize = 32;

struct Distribution {
    size_t n;
    int rank, ranks;
    std::vector<std::vector<size_t>> rows;
    std::vector<int> counts, displacements;

    Distribution(size_t size, int myRank, int numRanks)
        : n(size), rank(myRank), ranks(numRanks), rows(numRanks),
          counts(numRanks), displacements(numRanks) {
        for (size_t i = 0; i < n; ++i)
            rows[(i / blockSize) % ranks].push_back(i);
        int offset = 0;
        for (int p = 0; p < ranks; ++p) {
            counts[p] = static_cast<int>(rows[p].size() * n);
            displacements[p] = offset;
            offset += counts[p];
        }
    }
};

// Reproduce the original random stream on every rank, retaining only owned
// rows. Broadcast each rank's rows of B to form the distributed lower A.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const Distribution& dist) {
    const size_t n = dist.n;
    const auto& rows = dist.rows[dist.rank];
    std::vector<double> localB(rows.size() * n);
    unsigned int seed = 42;
    size_t localRow = 0;
    for (size_t i = 0; i < n; ++i) {
        const bool owned = static_cast<int>((i / blockSize) % dist.ranks) == dist.rank;
        for (size_t k = 0; k < n; ++k) {
            const double value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            if (owned) localB[localRow * n + k] = value;
        }
        if (owned) ++localRow;
    }
    size_t maxRows = 0;
    for (const auto& owned : dist.rows) maxRows = std::max(maxRows, owned.size());
    std::vector<double> remoteB(maxRows * n);
    for (int p = 0; p < dist.ranks; ++p) {
        if (p == dist.rank) std::copy(localB.begin(), localB.end(), remoteB.begin());
        MPI_Bcast(remoteB.data(), dist.counts[p], MPI_DOUBLE, p, MPI_COMM_WORLD);
        for (size_t r = 0; r < rows.size(); ++r) {
            const size_t i = rows[r];
            for (size_t q = 0; q < dist.rows[p].size(); ++q) {
                const size_t j = dist.rows[p][q];
                if (j > i) break;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k)
                    sum += localB[r * n + k] * remoteB[q * n + k];
                A[r * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const Distribution& dist) {
    const size_t n = dist.n;
    const auto& rows = dist.rows[dist.rank];
    std::vector<double> diagonal(blockSize * blockSize + 1);
    std::vector<double> panel(n * blockSize);
    std::vector<double> sendPanel(rows.size() * blockSize);
    std::vector<double> gathered(n * blockSize);
    std::vector<int> counts(dist.ranks), displacements(dist.ranks);

    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const size_t end = first + width;
        const int owner = static_cast<int>((first / blockSize) % dist.ranks);
        if (dist.rank == owner) {
            diagonal[0] = 0.0;
            const size_t base = (first / blockSize / dist.ranks) * blockSize;
            for (size_t i = 0; i < width; ++i) {
                double* row = A.data() + (base + i) * n;
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += row[first + k] * diagonal[1 + j * width + k];
                    if (i == j) {
                        const double value = row[first + j] - sum;
                        if (value <= 0.0) {
                            diagonal[0] = static_cast<double>(first + i + 1);
                            break;
                        }
                        row[first + j] = std::sqrt(value);
                    } else {
                        row[first + j] = (row[first + j] - sum) / diagonal[1 + j * width + j];
                    }
                    diagonal[1 + i * width + j] = row[first + j];
                }
                if (diagonal[0] != 0.0) break;
            }
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(1 + width * width), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (diagonal[0] != 0.0) {
            if (dist.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(diagonal[0]) - 1);
            return false;
        }

        size_t packed = 0;
        for (size_t r = 0; r < rows.size(); ++r) {
            if (rows[r] < end) continue;
            double* row = A.data() + r * n;
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += row[first + k] * diagonal[1 + j * width + k];
                row[first + j] = (row[first + j] - sum) / diagonal[1 + j * width + j];
                sendPanel[packed++] = row[first + j];
            }
        }
        int offset = 0;
        for (int p = 0; p < dist.ranks; ++p) {
            const auto& owned = dist.rows[p];
            const size_t remaining = static_cast<size_t>(owned.end() - std::lower_bound(owned.begin(), owned.end(), end));
            counts[p] = static_cast<int>(remaining * width);
            displacements[p] = offset;
            offset += counts[p];
        }
        MPI_Allgatherv(sendPanel.data(), static_cast<int>(packed), MPI_DOUBLE,
                       gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (int p = 0; p < dist.ranks; ++p) {
            size_t pos = static_cast<size_t>(displacements[p]);
            for (size_t i : dist.rows[p]) {
                if (i < end) continue;
                std::copy_n(gathered.data() + pos, width, panel.data() + i * blockSize);
                pos += width;
            }
        }
        for (size_t r = 0; r < rows.size(); ++r) {
            const size_t i = rows[r];
            if (i < end) continue;
            double* row = A.data() + r * n;
            const double* left = row + first;
            // Four independent dot products hide the dependency latency of
            // each accumulation and reuse the local row's panel entries.
            size_t j = end;
            for (; j + 3 <= i; j += 4) {
                const double* right = panel.data() + j * blockSize;
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (size_t k = 0; k < width; ++k) {
                    const double x = left[k];
                    s0 += x * right[k];
                    s1 += x * right[blockSize + k];
                    s2 += x * right[2 * blockSize + k];
                    s3 += x * right[3 * blockSize + k];
                }
                row[j] -= s0;
                row[j + 1] -= s1;
                row[j + 2] -= s2;
                row[j + 3] -= s3;
            }
            for (; j <= i; ++j) {
                const double* right = panel.data() + j * blockSize;
                double sum = 0.0;
                for (size_t k = 0; k < width; ++k)
                    sum += left[k] * right[k];
                row[j] -= sum;
            }
        }
    }
    return true;
}

// Collect the lower factor in row-major order for the existing results format
// and, when requested, make it available to each validator.
std::vector<double> collectFactor(const std::vector<double>& local, const Distribution& dist, bool allRanks) {
    const size_t n = dist.n;
    std::vector<double> packed(dist.rank == 0 ? n * n : 0);
    MPI_Gatherv(local.data(), dist.counts[dist.rank], MPI_DOUBLE, packed.data(),
                dist.counts.data(), dist.displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> result((dist.rank == 0 || allRanks) ? n * n : 0);
    if (dist.rank == 0) {
        for (int p = 0; p < dist.ranks; ++p) {
            size_t pos = static_cast<size_t>(dist.displacements[p]);
            for (size_t i : dist.rows[p]) {
                std::copy_n(packed.data() + pos, n, result.data() + i * n);
                pos += n;
            }
        }
    }
    if (allRanks)
        MPI_Bcast(result.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return result;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      const Distribution& dist) {
    const size_t n = dist.n;
    double errors[2] = {0.0, 0.0};
    const auto& rows = dist.rows[dist.rank];
    for (size_t r = 0; r < rows.size(); ++r) {
        const size_t i = rows[r];
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double reference = original[r * n + j];
            const double error = std::fabs(sum - reference);
            errors[0] = std::max(errors[0], error);
            errors[1] = std::max(errors[1], error / (std::fabs(reference) + 1e-10));
        }
    }
    double globalErrors[2];
    MPI_Allreduce(errors, globalErrors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (dist.rank == 0) {
        printf("Max absolute error: %.10e\n", globalErrors[0]);
        printf("Max relative error: %.10e\n", globalErrors[1]);
        if (globalErrors[1] > 1e-6)
            printf("Validation failed: relative error too large\n");
    }
    return globalErrors[1] <= 1e-6;
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
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' || parsed > static_cast<unsigned long long>(std::sqrt(INT_MAX))) {
                if (rank == 0) printf("Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    Distribution dist(n, rank, ranks);
    std::vector<double> A(dist.rows[rank].size() * n);
    generatePositiveDefiniteMatrix(A, dist);
    std::vector<double> original;
    if (validate) original = A;
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, dist);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        const long duration = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / maxElapsed / 1e9);
    }
    std::vector<double> factor;
    if (validate || printResults) factor = collectFactor(A, dist, validate);
    if (rank == 0 && printResults) print_results(factor, "CholeskyL");
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateCholesky(factor, original, dist);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
