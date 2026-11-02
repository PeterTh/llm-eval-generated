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

// Rows are distributed in cyclic blocks. Keeping each block on one rank lets
// its diagonal tile be factored locally before broadcasting it to the others.
struct Distribution {
    size_t n, block;
    int rank, processes;
    std::vector<size_t> rows;
    std::vector<size_t> localIndex;

    Distribution(size_t size, int r, int p) : n(size), rank(r), processes(p) {
        block = std::max<size_t>(1, std::min<size_t>(32, n / (4 * size_t(p))));
        localIndex.resize(n);
        for (size_t i = 0; i < n; ++i) {
            if (owner(i) == rank) {
                localIndex[i] = rows.size();
                rows.push_back(i);
            }
        }
    }
    int owner(size_t row) const { return int((row / block) % size_t(processes)); }
};

static void generateMatrix(std::vector<double>& a, const Distribution& d) {
    const size_t n = d.n;
    // Reproduce the original random sequence, retaining only owned rows.
    std::vector<double> b(d.rows.size() * n), block(d.block * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const bool owned = d.owner(i) == d.rank;
        for (size_t j = 0; j < n; ++j) {
            double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
            if (owned) b[d.localIndex[i]*n+j] = value;
        }
    }
    // Stream rows of B around the ranks to form each local row of B * B^T.
    for (size_t first = 0; first < n; first += d.block) {
        size_t width = std::min(d.block, n-first);
        int root = d.owner(first);
        if (d.rank == root)
            std::copy_n(b.data() + d.localIndex[first]*n, width*n, block.data());
        MPI_Bcast(block.data(), int(width*n), MPI_DOUBLE, root, MPI_COMM_WORLD);
        for (size_t r = 0; r < d.rows.size(); ++r) {
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += b[r*n+k] * block[j*n+k];
                a[r*n+first+j] = sum;
            }
        }
    }
    for (size_t r = 0; r < d.rows.size(); ++r) a[r*n+d.rows[r]] += n;
}

static bool factor(std::vector<double>& a, const Distribution& d) {
    const size_t n = d.n;
    std::vector<double> diagonal(d.block * d.block + 1);
    std::vector<double> send, gathered, panel;
    std::vector<int> counts(d.processes), offsets(d.processes);
    for (size_t first = 0; first < n; first += d.block) {
        const size_t width = std::min(d.block, n - first);
        const size_t end = first + width;
        const int root = d.owner(first);
        int failure = 0;
        if (d.rank == root) {
            for (size_t i = 0; i < width && !failure; ++i) {
                double* row = &a[d.localIndex[first+i]*n];
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += row[first+k] * diagonal[j*width+k];
                    double value = row[first+j] - sum;
                    if (i == j) {
                        if (value <= 0.0) { failure = int(i + 1); break; }
                        value = std::sqrt(value);
                    } else {
                        value /= diagonal[j*width+j];
                    }
                    row[first+j] = value;
                    diagonal[i*width+j] = value;
                }
            }
        }
        if (d.rank == root) diagonal[width*width] = double(failure);
        MPI_Bcast(diagonal.data(), int(width*width + 1), MPI_DOUBLE, root, MPI_COMM_WORLD);
        failure = int(diagonal[width*width]);
        if (failure) {
            if (d.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", first + size_t(failure-1));
            return false;
        }

        // Solve the rectangular panel against the diagonal tile.
        send.clear();
        for (size_t r = 0; r < d.rows.size(); ++r) {
            if (d.rows[r] < end) continue;
            double* row = &a[r*n];
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += row[first+k] * diagonal[j*width+k];
                row[first+j] = (row[first+j] - sum) / diagonal[j*width+j];
                send.push_back(row[first+j]);
            }
        }
        // Allgather just the current panel, then arrange it by global row.
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = end; i < n; ++i) counts[d.owner(i)] += int(width);
        int total = 0;
        for (int p = 0; p < d.processes; ++p) { offsets[p] = total; total += counts[p]; }
        gathered.resize(size_t(total));
        MPI_Allgatherv(send.data(), int(send.size()), MPI_DOUBLE, gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        panel.resize((n-end)*width);
        std::vector<int> position = offsets;
        for (size_t i = end; i < n; ++i) {
            const int p = d.owner(i);
            std::copy_n(gathered.data() + position[p], width, panel.data() + (i-end)*width);
            position[p] += int(width);
        }
        // Update the lower triangle of the trailing matrix in parallel.
        for (size_t r = 0; r < d.rows.size(); ++r) {
            const size_t i = d.rows[r];
            if (i < end) continue;
            double* row = &a[r*n];
            const double* li = panel.data() + (i-end)*width;
            for (size_t j = end; j <= i; ++j) {
                const double* lj = panel.data() + (j-end)*width;
                double sum = 0.0;
                for (size_t k = 0; k < width; ++k) sum += li[k] * lj[k];
                row[j] -= sum;
            }
        }
    }
    for (size_t r = 0; r < d.rows.size(); ++r)
        std::fill(a.begin() + r*n + d.rows[r] + 1, a.begin() + (r+1)*n, 0.0);
    return true;
}

// Broadcast each block of factor rows. This avoids replicating the entire
// factor during validation and gathers it only on rank zero for -r.
static bool report(const std::vector<double>& a, const std::vector<double>& original,
                   const Distribution& d, bool printResults, bool validate) {
    const size_t n = d.n;
    std::vector<double> full;
    if (printResults && d.rank == 0) full.resize(n*n);
    std::vector<double> rows(d.block*n);
    double errors[2] = {0.0, 0.0};
    for (size_t first = 0; first < n; first += d.block) {
        size_t width = std::min(d.block, n-first);
        int root = d.owner(first);
        if (d.rank == root)
            std::copy_n(a.data() + d.localIndex[first]*n, width*n, rows.data());
        MPI_Bcast(rows.data(), int(width*n), MPI_DOUBLE, root, MPI_COMM_WORLD);
        if (printResults && d.rank == 0)
            std::copy_n(rows.data(), width*n, full.data() + first*n);
        if (validate) {
            for (size_t r = 0; r < d.rows.size(); ++r) {
                const size_t i = d.rows[r];
                for (size_t j = 0; j < width; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k <= std::min(i, first+j); ++k)
                        sum += a[r*n+k] * rows[j*n+k];
                    double expected = original[r*n+first+j];
                    double error = std::fabs(sum - expected);
                    errors[0] = std::max(errors[0], error);
                    errors[1] = std::max(errors[1], error / (std::fabs(expected) + 1e-10));
                }
            }
        }
    }
    if (printResults && d.rank == 0) print_results(full, "CholeskyL");
    if (validate) {
        if (d.rank == 0) std::printf("Validating result...\n");
        double global[2];
        MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (d.rank == 0) {
            std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            if (global[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
            std::printf("Validation: %s\n", global[1] > 1e-6 ? "FAILED" : "PASSED");
        }
        return global[1] <= 1e-6;
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = std::strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' || value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) std::printf("Invalid matrix size: %s\n", arg);
                MPI_Finalize(); return 1;
            }
            n = size_t(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    // Check before allocating row metadata as well as before MPI int counts.
    const size_t block = std::max<size_t>(1, std::min<size_t>(32, n / (4 * size_t(processes))));
    if ((n && n > std::numeric_limits<size_t>::max()/n) || n > size_t(INT_MAX)/block) {
        if (rank == 0) std::printf("Matrix size exceeds supported limits\n");
        MPI_Finalize(); return 1;
    }
    Distribution d(n, rank, processes);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }
    std::vector<double> a(d.rows.size()*n), original;
    generateMatrix(a, d);
    if (validate) original = a;
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = factor(a, d);
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", long(maxElapsed*1000));
        double ops = double(n)*double(n)*double(n)/3.0;
        std::printf("Performance: %.3f GFLOPS\n", ops/maxElapsed/1e9);
    }
    bool valid = true;
    if (printResults || validate) valid = report(a, original, d, printResults, validate);
    MPI_Finalize();
    return valid ? 0 : 1;
}
