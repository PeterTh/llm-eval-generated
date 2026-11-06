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

namespace {
// Whole panels of rows are assigned round-robin, so the diagonal block of
// each panel is local to one process.
constexpr size_t maxBlockSize = 64;
size_t blockSize = maxBlockSize;

int owner(size_t row, int processes) {
    return static_cast<int>((row / blockSize) % static_cast<size_t>(processes));
}

size_t localIndex(size_t row, int processes) {
    return (row / blockSize / static_cast<size_t>(processes)) * blockSize + row % blockSize;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// MPI counts are int, so split large contiguous broadcasts into chunks.
void broadcastDoubles(double* data, size_t count, int root) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

bool choleskyDecomposition(std::vector<double>& a, size_t n,
                           const std::vector<size_t>& rows, int rank, int processes) {
    std::vector<double> diagonal(blockSize * blockSize);
    std::vector<double> packed, gathered, panel;
    std::vector<int> counts(processes), displacements(processes);

    for (size_t first = 0; first < n; first += blockSize) {
        const size_t width = std::min(blockSize, n - first);
        const size_t end = first + width;
        const int root = owner(first, processes);
        int failed = 0;
        if (rank == root) {
            // Previous panels have already updated this diagonal block.
            for (size_t i = 0; i < width && !failed; ++i) {
                double* row = a.data() + localIndex(first + i, processes) * n;
                for (size_t j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k)
                        sum += row[first + k] * diagonal[j * width + k];
                    if (i == j) {
                        const double value = row[first + i] - sum;
                        if (value <= 0.0) {
                            failed = 1;
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", first + i);
                            break;
                        }
                        row[first + i] = std::sqrt(value);
                    } else {
                        row[first + j] = (row[first + j] - sum) / diagonal[j * width + j];
                    }
                    diagonal[i * width + j] = row[first + j];
                }
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (failed) return false;
        broadcastDoubles(diagonal.data(), width * width, root);

        // Solve the panel below the diagonal block for every locally owned row.
        packed.clear();
        for (size_t local = 0; local < rows.size(); ++local) {
            if (rows[local] < end) continue;
            double* row = a.data() + local * n;
            for (size_t j = 0; j < width; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k)
                    sum += row[first + k] * diagonal[j * width + k];
                row[first + j] = (row[first + j] - sum) / diagonal[j * width + j];
                packed.push_back(row[first + j]);
            }
        }

        // Exchange the completed panel once, then update local trailing rows.
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = end; i < n; ++i)
            counts[owner(i, processes)] += static_cast<int>(width);
        int total = 0;
        for (int p = 0; p < processes; ++p) {
            displacements[p] = total;
            total += counts[p];
        }
        gathered.resize(static_cast<size_t>(total));
        MPI_Allgatherv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                       gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        panel.resize((n - end) * width);
        std::vector<int> offsets = displacements;
        for (size_t i = end; i < n; ++i) {
            const int p = owner(i, processes);
            std::copy_n(gathered.data() + offsets[p], width, panel.data() + (i - end) * width);
            offsets[p] += static_cast<int>(width);
        }
        for (size_t local = 0; local < rows.size(); ++local) {
            const size_t i = rows[local];
            if (i < end) continue;
            double* row = a.data() + local * n;
            const double* left = panel.data() + (i - end) * width;
            for (size_t j = end; j <= i; ++j) {
                const double* right = panel.data() + (j - end) * width;
                double sum = 0.0;
                for (size_t k = 0; k < width; ++k) sum += left[k] * right[k];
                row[j] -= sum;
            }
        }
    }
    return true;
}

// Reassemble only when results or validation are requested. Broadcasting one
// row block at a time also avoids MPI's int displacement limit for n*n.
std::vector<double> collectMatrix(const std::vector<double>& local, size_t n,
                                  int rank, int processes) {
    std::vector<double> matrix(n * n);
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t count = std::min(blockSize, n - first) * n;
        const int root = owner(first, processes);
        double* destination = matrix.data() + first * n;
        if (rank == root)
            std::copy_n(local.data() + localIndex(first, processes) * n, count, destination);
        broadcastDoubles(destination, count, root);
    }
    return matrix;
}
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' ||
                parsed > static_cast<unsigned long long>(INT_MAX) / maxBlockSize) {
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
    // Keep several row blocks per rank available for smaller matrices.
    blockSize = std::max<size_t>(1, std::min(maxBlockSize,
        (n + 2 * static_cast<size_t>(processes) - 1) / (2 * static_cast<size_t>(processes))));
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<size_t> rows;
    for (size_t i = 0; i < n; ++i)
        if (owner(i, processes) == rank) rows.push_back(i);
    std::vector<double> b(n * n), a(rows.size() * n), original;
    unsigned int seed = 42;
    for (double& value : b) value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t local = 0; local < rows.size(); ++local) {
        const size_t i = rows[local];
        for (size_t j = 0; j < (validate ? n : i + 1); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += b[i * n + k] * b[j * n + k];
            a[local * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
    b.clear();
    b.shrink_to_fit();
    if (validate) original = a;
    for (size_t local = 0; local < rows.size(); ++local)
        std::fill(a.begin() + local * n + rows[local] + 1, a.begin() + (local + 1) * n, 0.0);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, n, rows, rank, processes);
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
        printf("Performance: %.3f GFLOPS\n", ops / (milliseconds / 1000.0) / 1e9);
    }

    std::vector<double> full;
    if (printResults || validate) full = collectMatrix(a, n, rank, processes);
    if (printResults && rank == 0) print_results(full, "CholeskyL");
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        double errors[2] = {0.0, 0.0}, maxErrors[2] = {0.0, 0.0};
        for (size_t local = 0; local < rows.size(); ++local) {
            const size_t i = rows[local];
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += full[i * n + k] * full[j * n + k];
                const double error = std::fabs(sum - original[local * n + j]);
                errors[0] = std::max(errors[0], error);
                errors[1] = std::max(errors[1], error / (std::fabs(original[local * n + j]) + 1e-10));
            }
        }
        MPI_Reduce(errors, maxErrors, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Max absolute error: %.10e\n", maxErrors[0]);
            printf("Max relative error: %.10e\n", maxErrors[1]);
            if (maxErrors[1] > 1e-6) {
                printf("Validation failed: relative error too large\n");
                valid = 0;
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
