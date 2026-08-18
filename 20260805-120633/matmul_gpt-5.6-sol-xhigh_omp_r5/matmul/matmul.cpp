#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
  return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
         static_cast<double>(N * N);
}

void initMatrices(std::vector<double> &A, std::vector<double> &B,
                  const size_t N) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      const double value = getPseudoRndValue(N, i, j);
      A[i * N + j] = value;
      B[i * N + j] = value;
    }
  }
}

void matrixMultiply(const std::vector<double> &A, const std::vector<double> &B,
                    std::vector<double> &C, const size_t N) {
  // Output tiles are independent, so the OpenMP loop needs neither atomics
  // nor a reduction.  The 6x8 inner tile stays in SIMD registers across the
  // entire dot product instead of repeatedly loading partial sums from C.
  constexpr size_t rowBlock = 24;
  constexpr size_t columnBlock = 128;
  constexpr size_t microRows = 6;
  constexpr size_t microColumns = 8;

  const double *__restrict a = A.data();
  const double *__restrict b = B.data();
  double *__restrict c = C.data();
  const size_t packedColumnBlocks = (N + microColumns - 1) / microColumns;
  const size_t packedPanelSize = N * microColumns;
  std::unique_ptr<double[]> packedStorage(
      new double[packedColumnBlocks * packedPanelSize]);
  double *__restrict packedB = packedStorage.get();

#pragma omp parallel
  {
    // Pack eight columns at a time.  The multiplication then consumes B
    // as contiguous streams instead of making cache-unfriendly N-strided
    // accesses for every small register tile.
#pragma omp for collapse(2) schedule(static)
    for (size_t columnBlockIndex = 0; columnBlockIndex < packedColumnBlocks;
         ++columnBlockIndex) {
      for (size_t k = 0; k < N; ++k) {
        const size_t columnBase = columnBlockIndex * microColumns;
        double *const packedRow =
            packedB + columnBlockIndex * packedPanelSize + k * microColumns;
        for (size_t lane = 0; lane < microColumns; ++lane) {
          const size_t column = columnBase + lane;
          packedRow[lane] = column < N ? b[k * N + column] : 0.0;
        }
      }
    }

#pragma omp for collapse(2) schedule(static)
    for (size_t rowBase = 0; rowBase < N; rowBase += rowBlock) {
      for (size_t columnBase = 0; columnBase < N; columnBase += columnBlock) {
        const size_t rowEnd = rowBase + rowBlock < N ? rowBase + rowBlock : N;
        const size_t columnEnd =
            columnBase + columnBlock < N ? columnBase + columnBlock : N;
        const size_t fullRowEnd =
            rowBase + ((rowEnd - rowBase) / microRows) * microRows;
        const size_t fullColumnEnd =
            columnBase +
            ((columnEnd - columnBase) / microColumns) * microColumns;

        for (size_t j = columnBase; j < fullColumnEnd; j += microColumns) {
          for (size_t i = rowBase; i < fullRowEnd; i += microRows) {
            __m256d sums00 = _mm256_setzero_pd();
            __m256d sums01 = _mm256_setzero_pd();
            __m256d sums10 = _mm256_setzero_pd();
            __m256d sums11 = _mm256_setzero_pd();
            __m256d sums20 = _mm256_setzero_pd();
            __m256d sums21 = _mm256_setzero_pd();
            __m256d sums30 = _mm256_setzero_pd();
            __m256d sums31 = _mm256_setzero_pd();
            __m256d sums40 = _mm256_setzero_pd();
            __m256d sums41 = _mm256_setzero_pd();
            __m256d sums50 = _mm256_setzero_pd();
            __m256d sums51 = _mm256_setzero_pd();

            const double *const a0 = a + i * N;
            const double *const a1 = a0 + N;
            const double *const a2 = a1 + N;
            const double *const a3 = a2 + N;
            const double *const a4 = a3 + N;
            const double *const a5 = a4 + N;

            for (size_t k = 0; k < N; ++k) {
              const double *const bRow = packedB +
                                         (j / microColumns) * packedPanelSize +
                                         k * microColumns;
              const __m256d b0 = _mm256_loadu_pd(bRow);
              const __m256d b1 = _mm256_loadu_pd(bRow + 4);
              __m256d av = _mm256_broadcast_sd(a0 + k);
              sums00 = _mm256_fmadd_pd(av, b0, sums00);
              sums01 = _mm256_fmadd_pd(av, b1, sums01);
              av = _mm256_broadcast_sd(a1 + k);
              sums10 = _mm256_fmadd_pd(av, b0, sums10);
              sums11 = _mm256_fmadd_pd(av, b1, sums11);
              av = _mm256_broadcast_sd(a2 + k);
              sums20 = _mm256_fmadd_pd(av, b0, sums20);
              sums21 = _mm256_fmadd_pd(av, b1, sums21);
              av = _mm256_broadcast_sd(a3 + k);
              sums30 = _mm256_fmadd_pd(av, b0, sums30);
              sums31 = _mm256_fmadd_pd(av, b1, sums31);
              av = _mm256_broadcast_sd(a4 + k);
              sums40 = _mm256_fmadd_pd(av, b0, sums40);
              sums41 = _mm256_fmadd_pd(av, b1, sums41);
              av = _mm256_broadcast_sd(a5 + k);
              sums50 = _mm256_fmadd_pd(av, b0, sums50);
              sums51 = _mm256_fmadd_pd(av, b1, sums51);
            }

            _mm256_storeu_pd(c + (i + 0) * N + j, sums00);
            _mm256_storeu_pd(c + (i + 0) * N + j + 4, sums01);
            _mm256_storeu_pd(c + (i + 1) * N + j, sums10);
            _mm256_storeu_pd(c + (i + 1) * N + j + 4, sums11);
            _mm256_storeu_pd(c + (i + 2) * N + j, sums20);
            _mm256_storeu_pd(c + (i + 2) * N + j + 4, sums21);
            _mm256_storeu_pd(c + (i + 3) * N + j, sums30);
            _mm256_storeu_pd(c + (i + 3) * N + j + 4, sums31);
            _mm256_storeu_pd(c + (i + 4) * N + j, sums40);
            _mm256_storeu_pd(c + (i + 4) * N + j + 4, sums41);
            _mm256_storeu_pd(c + (i + 5) * N + j, sums50);
            _mm256_storeu_pd(c + (i + 5) * N + j + 4, sums51);
          }
        }

        // Use the same packed, vectorized path for a partial row tile.
        for (size_t i = fullRowEnd; i < rowEnd; ++i) {
          const double *const aRow = a + i * N;
          for (size_t j = columnBase; j < fullColumnEnd; j += microColumns) {
            __m256d sums0 = _mm256_setzero_pd();
            __m256d sums1 = _mm256_setzero_pd();
            for (size_t k = 0; k < N; ++k) {
              const double *const bRow = packedB +
                                         (j / microColumns) * packedPanelSize +
                                         k * microColumns;
              const __m256d av = _mm256_broadcast_sd(aRow + k);
              sums0 = _mm256_fmadd_pd(av, _mm256_loadu_pd(bRow), sums0);
              sums1 = _mm256_fmadd_pd(av, _mm256_loadu_pd(bRow + 4), sums1);
            }
            _mm256_storeu_pd(c + i * N + j, sums0);
            _mm256_storeu_pd(c + i * N + j + 4, sums1);
          }
        }

        // Only the final one to seven columns need the scalar fallback.
        for (size_t i = rowBase; i < rowEnd; ++i) {
          for (size_t j = fullColumnEnd; j < columnEnd; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
              sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
          }
        }
      }
    }
  }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double> &A, const std::vector<double> &B,
                    const std::vector<double> &C, const size_t N) {
  // Check a few random positions
  constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

  for (size_t pi = 0; pi < 5; ++pi) {
    for (size_t pj = 0; pj < 5; ++pj) {
      const size_t i = checkPoints[pi] % N;
      const size_t j = checkPoints[pj] % N;

      double expected = 0.0;
      for (size_t k = 0; k < N; ++k) {
        expected += A[i * N + k] * B[k * N + j];
      }

      const double actual = C[i * N + j];
      const double relError =
          std::abs((actual - expected) / (expected + 1e-10));

      if (relError > 1e-6) {
        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f "
               "(error: %.10e)\n",
               i, j, expected, actual, relError);
        return false;
      }
    }
  }

  return true;
}

void printUsage(const char *progName) {
  printf("Usage: %s [options]\n", progName);
  printf("Options:\n");
  printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
  printf("  -v           Enable validation\n");
  printf("  -r           Print results for external validation\n");
  printf("  -h           Show this help message\n");
}

int main(int argc, char **argv) {
  size_t N = 512;
  bool validate = false;
  bool printResults = false;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      N = atoi(argv[++i]);
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

  printf("Matrix Multiplication Benchmark\n");
  printf("Matrix size: %zu x %zu\n", N, N);
  printf("Validation: %s\n", validate ? "enabled" : "disabled");

  // Allocate matrices
  std::vector<double> A(N * N);
  std::vector<double> B(N * N);
  std::vector<double> C(N * N);

  // Initialize matrices
  printf("Initializing matrices...\n");
  initMatrices(A, B, N);

  // Perform matrix multiplication
  printf("Computing matrix multiplication...\n");
  auto start = std::chrono::high_resolution_clock::now();

  matrixMultiply(A, B, C, N);

  auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> duration = end - start;
  const double milliseconds = duration.count() * 1000.0;

  printf("Computation time: %.3f ms\n", milliseconds);

  // Calculate GFLOPS
  const double gflops = (2.0 * N * N * N) / duration.count() / 1e9;
  printf("Performance: %.3f GFLOPS\n", gflops);

  // Print results for external validation
  if (printResults) {
    print_results(C, "MatrixC");
  }

  // Validation
  if (validate) {
    printf("Validating result...\n");
    bool valid = validateResult(A, B, C, N);

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
