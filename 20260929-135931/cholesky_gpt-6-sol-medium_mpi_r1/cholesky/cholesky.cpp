#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

// Cyclic blocks balance the shrinking trailing matrix and keep diagonal blocks local.
constexpr size_t blockSize = 64;
struct Distribution {
    size_t n;
    int rank, processes;
    std::vector<std::vector<size_t>> rows;
    std::vector<size_t> index;
    Distribution(size_t n_, int rank_, int processes_)
        : n(n_), rank(rank_), processes(processes_), rows(processes_), index(n_) {
        for (size_t i = 0; i < n; ++i) {
            int owner = static_cast<int>((i / blockSize) % processes);
            index[i] = rows[owner].size();
            rows[owner].push_back(i);
        }
    }
    int owner(size_t row) const { return static_cast<int>((row / blockSize) % processes); }
};

void broadcastDoubles(double* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const Distribution& dist) {
    const size_t n = dist.n;
    std::vector<double> B(n * n);
    if (dist.rank == 0) {
        unsigned int seed = 42;
        for (double& value : B)
            value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
    broadcastDoubles(B.data(), B.size(), 0);
    for (size_t local = 0; local < dist.rows[dist.rank].size(); ++local) {
        size_t i = dist.rows[dist.rank][local];
        double* row = A.data() + local * n;
        const double* bi = B.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
            row[j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const Distribution& dist) {
    const size_t n = dist.n;
    const auto& myRows = dist.rows[dist.rank];
    std::vector<double> diagonal(blockSize * blockSize);
    for (size_t k = 0; k < n; k += blockSize) {
        size_t width = std::min(blockSize, n - k), first = k + width;
        int owner = dist.owner(k), failed = 0;
        if (dist.rank == owner) {
            // Previous iterations have updated the complete diagonal block.
            for (size_t j = 0; j < width; ++j) {
                double* row = A.data() + dist.index[k + j] * n;
                double value = row[k + j];
                for (size_t p = 0; p < j; ++p) value -= row[k + p] * row[k + p];
                if (!(value > 0.0)) { failed = static_cast<int>(k + j + 1); break; }
                row[k + j] = diagonal[j * width + j] = std::sqrt(value);
                for (size_t i = j + 1; i < width; ++i) {
                    double* lower = A.data() + dist.index[k + i] * n;
                    double entry = lower[k + j];
                    for (size_t p = 0; p < j; ++p) entry -= lower[k + p] * row[k + p];
                    lower[k + j] = diagonal[i * width + j] = entry / row[k + j];
                }
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) {
            if (dist.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %d\n", failed - 1);
            return false;
        }
        broadcastDoubles(diagonal.data(), width * width, owner);
        if (first == n) continue;

        std::vector<int> counts(dist.processes), offsets(dist.processes);
        int total = 0;
        for (int rank = 0; rank < dist.processes; ++rank) {
            const auto& rankRows = dist.rows[rank];
            size_t begin = std::lower_bound(rankRows.begin(), rankRows.end(), first) - rankRows.begin();
            counts[rank] = static_cast<int>((rankRows.size() - begin) * width);
            offsets[rank] = total;
            total += counts[rank];
        }
        std::vector<double> send(counts[dist.rank]);
        size_t packed = 0;
        for (size_t local = 0; local < myRows.size(); ++local) {
            if (myRows[local] < first) continue;
            double* row = A.data() + local * n;
            for (size_t j = 0; j < width; ++j) {
                double entry = row[k + j];
                for (size_t p = 0; p < j; ++p)
                    entry -= row[k + p] * diagonal[j * width + p];
                row[k + j] = entry / diagonal[j * width + j];
            }
            std::copy_n(row + k, width, send.data() + packed);
            packed += width;
        }
        std::vector<double> received(total);
        MPI_Allgatherv(send.data(), counts[dist.rank], MPI_DOUBLE,
                       received.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<double> panel((n - first) * width);
        for (int rank = 0; rank < dist.processes; ++rank) {
            const auto& rankRows = dist.rows[rank];
            size_t begin = std::lower_bound(rankRows.begin(), rankRows.end(), first) - rankRows.begin();
            size_t source = offsets[rank];
            for (size_t r = begin; r < rankRows.size(); ++r) {
                std::copy_n(received.data() + source, width,
                            panel.data() + (rankRows[r] - first) * width);
                source += width;
            }
        }
        // Every rank updates only its owned lower-triangular rows.
        for (size_t local = 0; local < myRows.size(); ++local) {
            size_t i = myRows[local];
            if (i < first) continue;
            double* row = A.data() + local * n;
            const double* li = panel.data() + (i - first) * width;
            for (size_t j = first; j <= i; ++j) {
                const double* lj = panel.data() + (j - first) * width;
                double product = 0.0;
                for (size_t p = 0; p < width; ++p) product += li[p] * lj[p];
                row[j] -= product;
            }
        }
    }
    return true;
}

std::vector<double> collectMatrix(const std::vector<double>& A,
                                  const Distribution& dist, bool allRanks) {
    std::vector<int> counts(dist.processes), offsets(dist.processes);
    int total = 0;
    for (int rank = 0; rank < dist.processes; ++rank) {
        counts[rank] = static_cast<int>(dist.rows[rank].size() * dist.n);
        offsets[rank] = total;
        total += counts[rank];
    }
    std::vector<double> packed(allRanks || dist.rank == 0 ? total : 0);
    if (allRanks)
        MPI_Allgatherv(A.data(), counts[dist.rank], MPI_DOUBLE, packed.data(),
                       counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    else
        MPI_Gatherv(A.data(), counts[dist.rank], MPI_DOUBLE, packed.data(),
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!allRanks && dist.rank != 0) return {};
    std::vector<double> full(dist.n * dist.n);
    for (int rank = 0; rank < dist.processes; ++rank) {
        size_t source = offsets[rank];
        for (size_t row : dist.rows[rank]) {
            std::copy_n(packed.data() + source, dist.n, full.data() + row * dist.n);
            source += dist.n;
        }
    }
    return full;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& original, const Distribution& dist) {
    double local[2] = {0.0, 0.0};
    for (size_t r = 0; r < dist.rows[dist.rank].size(); ++r) {
        size_t i = dist.rows[dist.rank][r];
        for (size_t j = 0; j <= i; ++j) {
            double value = 0.0;
            for (size_t k = 0; k <= j; ++k)
                value += L[i * dist.n + k] * L[j * dist.n + k];
            double error = std::fabs(value - original[r * dist.n + j]);
            local[0] = std::max(local[0], error);
            local[1] = std::max(local[1], error / (std::fabs(original[r * dist.n + j]) + 1e-10));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, local, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (dist.rank == 0) {
        std::printf("Max absolute error: %.10e\n", local[0]);
        std::printf("Max relative error: %.10e\n", local[1]);
        if (local[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
    }
    return local[1] <= 1e-6;
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    Distribution dist(n, rank, processes);
    std::vector<double> A(dist.rows[rank].size() * n);
    generatePositiveDefiniteMatrix(A, dist);
    std::vector<double> original;
    if (validate) original = A;
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(A, dist);
    double elapsed = MPI_Wtime() - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops / elapsed / 1e9);
    }
    std::vector<double> full;
    if (validate || printResults) full = collectMatrix(A, dist, validate);
    if (rank == 0 && printResults) print_results(full, "CholeskyL");
    int result = 0;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        bool valid = validateCholesky(full, original, dist);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        result = valid ? 0 : 1;
    }
    MPI_Finalize();
    return result;
}
