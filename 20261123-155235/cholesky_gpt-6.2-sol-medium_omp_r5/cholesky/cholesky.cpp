#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t maxTileSize = 112;
constexpr size_t smallMatrixSize = 768;

size_t tileSizeFor(size_t n) {
    // Small matrices need more tiles to keep workers busy. Larger tiles
    // amortize task scheduling on larger matrices.
    if (n <= smallMatrixSize) {
        return 64;
    }
    return n <= 3072 ? 96 : maxTileSize;
}

double dotProduct(const double* left, const double* right, size_t count) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (size_t k = 0; k < count; ++k) {
        sum += left[k] * right[k];
    }
    return sum;
}

size_t factorDiagonal(double* tile, size_t stride, size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        double* row = tile + i * stride;
        for (size_t j = 0; j < i; ++j) {
            const double* prev = tile + j * stride;
            row[j] = (row[j] - dotProduct(row, prev, j)) / prev[j];
            tile[j * stride + i] = row[j];
        }
        const double pivot = row[i] - dotProduct(row, row, i);
        if (pivot <= 0.0) {
            return i;
        }
        row[i] = std::sqrt(pivot);
    }
    return rows;
}

void solvePanel(double* panel, const double* diagonal,
                size_t stride, size_t rows, size_t width) {
    // The diagonal tile holds its transpose in its upper triangle.
    for (size_t i = 0; i < rows; ++i) {
        double* row = panel + i * stride;
        for (size_t j = 0; j < width; ++j) {
            const double* diagRow = diagonal + j * stride;
            const double value = row[j] / diagRow[j];
            row[j] = value;
            #pragma omp simd
            for (size_t k = j + 1; k < width; ++k) {
                row[k] -= value * diagRow[k];
            }
        }
    }
}

constexpr size_t updateColumns = 8;

// Hold a small group of output elements in registers across the panel.
template<int Sign, size_t Rows>
void updateTileRows(double* target, const double* left,
                    const double (*packed)[updateColumns], size_t stride, size_t width) {
    double values[Rows][updateColumns];
    for (size_t i = 0; i < Rows; ++i) {
        std::copy_n(target + i * stride, updateColumns, values[i]);
    }
    for (size_t k = 0; k < width; ++k) {
        for (size_t i = 0; i < Rows; ++i) {
            const double entry = Sign * left[i * stride + k];
            #pragma omp simd
            for (size_t c = 0; c < updateColumns; ++c) {
                values[i][c] += entry * packed[k][c];
            }
        }
    }
    for (size_t i = 0; i < Rows; ++i) {
        std::copy_n(values[i], updateColumns, target + i * stride);
    }
}

template<int Sign>
void updateTile(double* target, const double* left, const double* right,
                size_t stride, size_t rows, size_t columns, size_t width) {
    constexpr size_t rowGroup = 4;
    alignas(64) double packed[maxTileSize][updateColumns];
    const bool diagonal = left == right;
    size_t j = 0;
    for (; j + updateColumns <= columns; j += updateColumns) {
        // Transpose a strip of the panel so SIMD loads are contiguous even
        // when the matrix stride would map consecutive rows to one cache set.
        for (size_t c = 0; c < updateColumns; ++c) {
            for (size_t k = 0; k < width; ++k) {
                packed[k][c] = right[(j + c) * stride + k];
            }
        }
        size_t i = diagonal ? j : 0;
        for (; i + rowGroup <= rows; i += rowGroup) {
            updateTileRows<Sign, rowGroup>(target + i * stride + j, left + i * stride,
                                     packed, stride, width);
        }
        for (; i < rows; ++i) {
            updateTileRows<Sign, 1>(target + i * stride + j, left + i * stride,
                              packed, stride, width);
        }
    }
    // The final matrix tile can have a partial group of columns.
    for (size_t c = j; c < columns; ++c) {
        for (size_t i = diagonal ? c : 0; i < rows; ++i) {
            double value = target[i * stride + c];
            for (size_t k = 0; k < width; ++k) {
                value += (Sign * left[i * stride + k]) * right[c * stride + k];
            }
            target[i * stride + c] = value;
        }
    }
}

int threadCount(size_t n, size_t workGroups) {
    // Keep enough work per thread to amortize team and scheduling costs.
    const size_t tileSize = tileSizeFor(n);
    const size_t tiles = (n + tileSize - 1) / tileSize;
    const size_t usefulWorkers = std::min(workGroups, (n <= smallMatrixSize ? 1 : 2) * tiles);
    const size_t available = omp_get_max_threads();
    return static_cast<int>(std::max<size_t>(1, std::min(available, usefulWorkers)));
}

} // namespace

// Decompose A in place into the lower triangular factor L of A = L * L^T.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* matrix = A.data();
    const size_t tileSize = tileSizeFor(n);
    const size_t tiles = (n + tileSize - 1) / tileSize;
    const int workers = threadCount(n, tiles * (tiles + 1) / 4);
    std::atomic<size_t> failedPivot{n};

    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        for (size_t k = 0; k < n; k += tileSize) {
            const size_t width = std::min(tileSize, n - k);
            double* diagonal = matrix + k * n + k;

            // A tile's first element identifies it to the task scheduler.
            #pragma omp task depend(inout: diagonal[0])
            {
                if (failedPivot.load(std::memory_order_relaxed) == n) {
                    const size_t pivot = factorDiagonal(diagonal, n, width);
                    if (pivot != width) {
                        failedPivot.store(k + pivot, std::memory_order_relaxed);
                    }
                }
            }

            for (size_t i = k + width; i < n; i += tileSize) {
                const size_t rows = std::min(tileSize, n - i);
                double* panel = matrix + i * n + k;
                #pragma omp task depend(in: diagonal[0]) depend(inout: panel[0])
                {
                    if (failedPivot.load(std::memory_order_relaxed) == n) {
                        solvePanel(panel, diagonal, n, rows, width);
                    }
                }
            }

            for (size_t i = k + width; i < n; i += tileSize) {
                const size_t rows = std::min(tileSize, n - i);
                const double* left = matrix + i * n + k;
                for (size_t j = k + width; j <= i; j += tileSize) {
                    const size_t columns = std::min(tileSize, n - j);
                    const double* right = matrix + j * n + k;
                    double* target = matrix + i * n + j;

                    #pragma omp task depend(in: left[0], right[0]) depend(inout: target[0])
                    {
                        if (failedPivot.load(std::memory_order_relaxed) == n) {
                            updateTile<-1>(target, left, right, n, rows, columns, width);
                        }
                    }
                }
            }
        }

        // The single region's barrier waits for every tile task.
        if (failedPivot.load(std::memory_order_relaxed) == n) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill_n(matrix + i * n + i + 1, n - i - 1, 0.0);
            }
        }
    }

    const size_t pivot = failedPivot.load(std::memory_order_relaxed);
    if (pivot != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", pivot);
        return false;
    }
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Keep the original random sequence.
    for (double& value : B) {
        value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Form B * B^T + nI. Each tile also writes its symmetric entries.
    const size_t tileSize = tileSizeFor(n);
    const size_t tiles = (n + tileSize - 1) / tileSize;
    const size_t tileCount = tiles * (tiles + 1) / 2;
    const int workers = threadCount(n, tileCount);
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (size_t index = 0; index < tileCount; ++index) {
        // Invert the triangular tile index to distribute equally sized tiles.
        const size_t tileRow = static_cast<size_t>((std::sqrt(8.0 * index + 1) - 1) / 2);
        const size_t i = tileRow * tileSize;
        const size_t j = (index - tileRow * (tileRow + 1) / 2) * tileSize;
        const size_t rows = std::min(tileSize, n - i);
        const size_t columns = std::min(tileSize, n - j);
        double* target = A.data() + i * n + j;
        for (size_t r = 0; r < rows; ++r) {
            std::fill_n(target + r * n, columns, 0.0);
        }
        for (size_t k = 0; k < n; k += tileSize) {
            const size_t width = std::min(tileSize, n - k);
            updateTile<1>(target, B.data() + i * n + k, B.data() + j * n + k,
                          n, rows, columns, width);
        }
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns && j + c < i + r; ++c) {
                A[(j + c) * n + i + r] = target[r * n + c];
            }
        }
        if (i == j) {
            for (size_t r = 0; r < rows; ++r) {
                target[r * n + r] += n;
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    const int workers = threadCount(n, (n + 31) / 32);
    #pragma omp parallel for num_threads(workers) schedule(static, 4) reduction(max:maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        const double* row = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double reconstructed = dotProduct(row, L.data() + j * n, j + 1);
            const double original = A_orig[i * n + j];
            const double error = std::abs(reconstructed - original);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::abs(original) + 1e-10));
            if (i != j) {
                const double upper = A_orig[j * n + i];
                const double upperError = std::abs(reconstructed - upper);
                maxError = std::max(maxError, upperError);
                relError = std::max(relError, upperError / (std::abs(upper) + 1e-10));
            }
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
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
