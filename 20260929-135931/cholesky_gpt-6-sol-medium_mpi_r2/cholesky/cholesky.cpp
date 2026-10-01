#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The matrix is distributed in cyclic blocks of rows. Each rank stores only its
// own rows; columns in those rows retain their global, row-major numbering.
constexpr size_t blockSize = 64;

int ownerOf(size_t row, int ranks) {
    return static_cast<int>((row / blockSize) % ranks);
}

size_t localIndex(size_t row, int ranks) {
    return (row / blockSize / static_cast<size_t>(ranks)) * blockSize + row % blockSize;
}

size_t localRowCount(size_t n, int rank, int ranks) {
    size_t count = 0;
    for (size_t start = static_cast<size_t>(rank) * blockSize; start < n;
         start += static_cast<size_t>(ranks) * blockSize) {
        count += std::min(blockSize, n - start);
    }
    return count;
}

void broadcastDoubles(double* data, size_t count, int root) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

// Gather only when validation or external result output needs a full matrix.
std::vector<double> gatherMatrix(const std::vector<double>& local, size_t n,
                                 int rank, int ranks) {
    std::vector<int> counts(ranks), offsets(ranks);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        counts[r] = static_cast<int>(localRowCount(n, r, ranks) * n);
        offsets[r] = total;
        total += counts[r];
    }
    std::vector<double> packed(rank == 0 ? n * n : 0);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                rank == 0 ? packed.data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) return {};

    std::vector<double> full(n * n);
    for (int r = 0; r < ranks; ++r) {
        size_t rowIndex = 0;
        for (size_t row = 0; row < n; ++row) {
            if (ownerOf(row, ranks) != r) continue;
            std::copy_n(packed.data() + offsets[r] + rowIndex * n, n,
                        full.data() + row * n);
            ++rowIndex;
        }
    }
    return full;
}

void generatePositiveDefiniteMatrix(std::vector<double>& local, size_t n,
                                    int rank, int ranks) {
    // Preserve the original rand_r sequence and dot-product summation order.
    std::vector<double> B(n * n);
    if (rank == 0) {
        unsigned int seed = 42;
        for (double& value : B) {
            value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }
    broadcastDoubles(B.data(), B.size(), 0);

    for (size_t row = 0; row < n; ++row) {
        if (ownerOf(row, ranks) != rank) continue;
        double* a = local.data() + localIndex(row, ranks) * n;
        const double* bi = B.data() + row * n;
        for (size_t col = 0; col <= row; ++col) {
            const double* bj = B.data() + col * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
            a[col] = sum + (row == col ? static_cast<double>(n) : 0.0);
        }
    }
}

bool choleskyDecomposition(std::vector<double>& local, size_t n,
                           int rank, int ranks) {
    for (size_t start = 0; start < n; start += blockSize) {
        const size_t width = std::min(blockSize, n - start);
        const size_t end = start + width;
        const int diagonalOwner = ownerOf(start, ranks);
        std::vector<double> diagonal(width * width, 0.0);
        int good = 1;

        if (rank == diagonalOwner) {
            // Factor the diagonal block after all preceding panels have updated it.
            for (size_t j = 0; j < width; ++j) {
                double* pivot = local.data() + localIndex(start + j, ranks) * n;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += pivot[start + k] * pivot[start + k];
                const double value = pivot[start + j] - sum;
                if (value <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", start + j);
                    good = 0;
                    break;
                }
                pivot[start + j] = std::sqrt(value);
                for (size_t i = j + 1; i < width; ++i) {
                    double* row = local.data() + localIndex(start + i, ranks) * n;
                    double dot = 0.0;
                    for (size_t k = 0; k < j; ++k) dot += row[start + k] * pivot[start + k];
                    row[start + j] = (row[start + j] - dot) / pivot[start + j];
                }
            }
            if (good) {
                for (size_t i = 0; i < width; ++i) {
                    const double* row = local.data() + localIndex(start + i, ranks) * n;
                    std::copy_n(row + start, i + 1, diagonal.data() + i * width);
                }
            }
        }
        MPI_Bcast(&good, 1, MPI_INT, diagonalOwner, MPI_COMM_WORLD);
        if (!good) return false;
        broadcastDoubles(diagonal.data(), diagonal.size(), diagonalOwner);

        // Triangular solve for every locally owned row below the block.
        for (size_t row = end; row < n; ++row) {
            if (ownerOf(row, ranks) != rank) continue;
            double* a = local.data() + localIndex(row, ranks) * n;
            for (size_t j = 0; j < width; ++j) {
                double dot = 0.0;
                for (size_t k = 0; k < j; ++k) dot += a[start + k] * diagonal[j * width + k];
                a[start + j] = (a[start + j] - dot) / diagonal[j * width + j];
            }
        }

        const size_t trailing = n - end;
        if (trailing == 0) continue;
        std::vector<int> counts(ranks), offsets(ranks);
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            int rows = 0;
            for (size_t row = end; row < n; ++row) rows += ownerOf(row, ranks) == r;
            counts[r] = rows * static_cast<int>(width);
            offsets[r] = total;
            total += counts[r];
        }
        std::vector<double> send(counts[rank]), received(total);
        size_t pos = 0;
        for (size_t row = end; row < n; ++row) {
            if (ownerOf(row, ranks) != rank) continue;
            const double* a = local.data() + localIndex(row, ranks) * n;
            std::copy_n(a + start, width, send.data() + pos);
            pos += width;
        }
        MPI_Allgatherv(send.data(), counts[rank], MPI_DOUBLE, received.data(),
                       counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Transpose the panel so each rank-one update streams through contiguous
        // columns of its local rows.
        std::vector<double> panel(width * trailing);
        for (int r = 0; r < ranks; ++r) {
            size_t packedRow = 0;
            for (size_t row = end; row < n; ++row) {
                if (ownerOf(row, ranks) != r) continue;
                for (size_t k = 0; k < width; ++k) {
                    panel[k * trailing + row - end] = received[offsets[r] + packedRow * width + k];
                }
                ++packedRow;
            }
        }

        // Update the lower triangle of each locally owned trailing row.
        for (size_t row = end; row < n; ++row) {
            if (ownerOf(row, ranks) != rank) continue;
            double* a = local.data() + localIndex(row, ranks) * n;
            const size_t columns = row - end + 1;
            for (size_t k = 0; k < width; ++k) {
                const double factor = a[start + k];
                const double* p = panel.data() + k * trailing;
                for (size_t j = 0; j < columns; ++j) a[end + j] -= factor * p[j];
            }
        }
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
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
            n = static_cast<size_t>(atoi(argv[++i]));
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
    if (n > 0 && n > static_cast<size_t>(INT_MAX) / n) {
        if (rank == 0) fprintf(stderr, "Matrix size exceeds MPI gather limits\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> local(localRowCount(n, rank, ranks) * n, 0.0);
    generatePositiveDefiniteMatrix(local, n, rank, ranks);
    std::vector<double> original;
    if (validate) {
        original = gatherMatrix(local, n, rank, ranks);
        if (rank == 0) {
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j)
                    original[i * n + j] = original[j * n + i];
        }
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(local, n, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / duration / 1e9);
    }
    if (printResults || validate) {
        std::vector<double> result = gatherMatrix(local, n, rank, ranks);
        if (rank == 0) {
            if (printResults) print_results(result, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateCholesky(result, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) {
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }
    MPI_Finalize();
    return 0;
}
