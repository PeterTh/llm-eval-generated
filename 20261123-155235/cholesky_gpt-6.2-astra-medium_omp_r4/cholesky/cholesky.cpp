#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {
constexpr size_t panelWidth = 32;
constexpr size_t rowGroup = 4;
constexpr size_t colGroup = 8;

int teamSize(size_t n) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
        std::max<size_t>(1, (n + panelWidth - 1) / panelWidth)));
}

// Interleave columns of the transposed panel so SIMD lanes compute
// independent dot products, each accumulating in increasing k order.
void packRows(const double* rows, size_t n, size_t count, size_t depth,
              double* packed) {
    for (size_t k = 0; k < depth; ++k) {
        for (size_t c = 0; c < count; ++c)
            packed[k * colGroup + c] = rows[c * n + k];
        for (size_t c = count; c < colGroup; ++c)
            packed[k * colGroup + c] = 0.0;
    }
}

using Products = double[rowGroup][colGroup];

void dotBlock(const double* rows, size_t n, size_t depth, size_t count,
              const double* packed, Products& __restrict__ sums) {
    if (count < rowGroup) {
        for (size_t r = 0; r < count; ++r)
            for (size_t k = 0; k < depth; ++k)
                for (size_t c = 0; c < colGroup; ++c)
                    sums[r][c] += rows[r * n + k] * packed[k * colGroup + c];
        return;
    }
    for (size_t k = 0; k < depth; ++k) {
        for (size_t r = 0; r < rowGroup; ++r) {
            const double x = rows[r * n + k];
            #pragma omp simd
            for (size_t c = 0; c < colGroup; ++c)
                sums[r][c] += x * packed[k * colGroup + c];
        }
    }
}

// Compute four rows of the current block column. The prefix contributions
// are independent. Within the panel each row needs the earlier pivots.
size_t solveRows(double* a, size_t n, size_t first, size_t stop,
                 size_t i, const double* packed) {
    const size_t count = std::min(rowGroup, n - i);
    double prefix[rowGroup][panelWidth];
    for (size_t j = first; j < stop; j += colGroup) {
        Products sums{};
        const double* group = packed + (j - first) * first;
        dotBlock(a + i * n, n, first, count, group, sums);
        for (size_t r = 0; r < count; ++r)
            std::copy_n(sums[r], std::min(colGroup, stop - j), prefix[r] + j - first);
    }

    for (size_t r = 0; r < count; ++r) {
        double* row = a + (i + r) * n;
        for (size_t j = first; j < stop && j <= i + r; ++j) {
            const double* pivot = a + j * n;
            double sum = prefix[r][j - first];
            for (size_t k = first; k < j; ++k)
                sum += row[k] * pivot[k];
            const double value = row[j] - sum;
            if (j == i + r) {
                if (value <= 0.0) return j;
                row[j] = std::sqrt(value);
                std::fill(row + j + 1, row + n, 0.0);
            } else {
                row[j] = value / pivot[j];
            }
        }
    }
    return n;
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    std::vector<double> packed(panelWidth * n);
    size_t failed = n;
    #pragma omp parallel num_threads(teamSize(n))
    {
        for (size_t first = 0; first < n; first += panelWidth) {
            const size_t stop = std::min(n, first + panelWidth);
            #pragma omp single
            {
                for (size_t j = first; j < stop; j += colGroup)
                    packRows(A.data() + j * n, n, std::min(colGroup, stop - j),
                             first, packed.data() + (j - first) * first);
                for (size_t i = first; i < stop && failed == n; i += rowGroup)
                    failed = solveRows(A.data(), n, first, stop, i, packed.data());
            }
            if (failed != n) break;
            #pragma omp for schedule(static)
            for (size_t i = stop; i < n; i += rowGroup)
                solveRows(A.data(), n, first, stop, i, packed.data());
        }
    }
    if (failed != n)
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failed);
    return failed == n;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B)
        x = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    #pragma omp parallel num_threads(teamSize(n))
    {
        std::vector<double> packed(colGroup * n);
        #pragma omp for schedule(static, 1)
        for (size_t j = 0; j < n; j += colGroup) {
            const size_t cols = std::min(colGroup, n - j);
            packRows(B.data() + j * n, n, cols, n, packed.data());
            for (size_t i = j; i < n; i += rowGroup) {
                Products sums{};
                const size_t rows = std::min(rowGroup, n - i);
                dotBlock(B.data() + i * n, n, n, rows, packed.data(), sums);
                for (size_t r = 0; r < rows; ++r)
                    std::copy_n(sums[r], cols, A.data() + (i + r) * n + j);
            }
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            A[i * n + i] += n;
            for (size_t j = i + 1; j < n; ++j)
                A[i * n + j] = A[j * n + i];
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for schedule(static, 1) num_threads(teamSize(n)) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            for (double original : {A_orig[i * n + j], A_orig[j * n + i]}) {
                const double error = std::fabs(sum - original);
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (std::fabs(original) + 1e-10));
            }
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
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
    double gflops = ops / (std::chrono::duration<double>(end - start).count()) / 1e9;
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
