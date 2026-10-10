#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t tileSize = 64;

// Avoid creating more workers than these matrix sizes can keep busy. The
// OpenMP runtime (including OMP_NUM_THREADS) still controls the upper limit.
int workerCount(const size_t n, const size_t blockSize = tileSize) {
    const size_t useful = std::max(size_t{1}, (n + blockSize - 1) / blockSize);
    return static_cast<int>(std::min(useful, static_cast<size_t>(omp_get_max_threads())));
}

// C = X Y^T. All matrices have row-major leading dimension n.
// Eight independent dot products share their input loads in the main kernel.
// Tiling keeps the panel data in cache; SIMD operates on the contiguous depth.
void multiplyTile(double* C, const double* X, const double* Y,
                  const size_t n, const size_t rows, const size_t cols,
                  const size_t depth) {
    size_t i = 0;
    for (; i + 1 < rows; i += 2) {
        const double* x0 = X + i * n;
        const double* x1 = x0 + n;
        size_t j = 0;
        for (; j + 3 < cols; j += 4) {
            const double* y0 = Y + j * n;
            const double* y1 = y0 + n;
            const double* y2 = y1 + n;
            const double* y3 = y2 + n;
            double s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
            double s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
            #pragma omp simd reduction(+:s00,s01,s02,s03,s10,s11,s12,s13)
            for (size_t k = 0; k < depth; ++k) {
                s00 += x0[k] * y0[k];
                s01 += x0[k] * y1[k];
                s02 += x0[k] * y2[k];
                s03 += x0[k] * y3[k];
                s10 += x1[k] * y0[k];
                s11 += x1[k] * y1[k];
                s12 += x1[k] * y2[k];
                s13 += x1[k] * y3[k];
            }
            double* c0 = C + i * n + j;
            double* c1 = c0 + n;
            c0[0] = s00; c0[1] = s01; c0[2] = s02; c0[3] = s03;
            c1[0] = s10; c1[1] = s11; c1[2] = s12; c1[3] = s13;
        }
        for (; j < cols; ++j) {
            const double* y = Y + j * n;
            double s0 = 0.0, s1 = 0.0;
            #pragma omp simd reduction(+:s0,s1)
            for (size_t k = 0; k < depth; ++k) {
                s0 += x0[k] * y[k];
                s1 += x1[k] * y[k];
            }
            C[i * n + j] = s0;
            C[(i + 1) * n + j] = s1;
        }
    }
    if (i < rows) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < depth; ++k) {
                sum += X[i * n + k] * Y[j * n + k];
            }
            C[i * n + j] = sum;
        }
    }
}

// Panel tiles are packed once in transposed order with a padded stride to
// avoid cache-set conflicts. Four rows by eight columns accumulate across
// the depth loop, reusing coefficients without horizontal SIMD reductions.
void updateTrailingTile(double* C, const double* X, const double* Y,
                        const size_t n, const size_t rows, const size_t cols,
                        const size_t depth, const size_t packedStride) {
    size_t i = 0;
    for (; i + 3 < rows; i += 4) {
        size_t j = 0;
        for (; j + 7 < cols; j += 8) {
            double s0[8] = {}, s1[8] = {}, s2[8] = {}, s3[8] = {};
            for (size_t k = 0; k < depth; ++k) {
                const double* right = Y + k * packedStride + j;
                const double x0 = X[i * n + k];
                const double x1 = X[(i + 1) * n + k];
                const double x2 = X[(i + 2) * n + k];
                const double x3 = X[(i + 3) * n + k];
                #pragma omp simd
                for (size_t c = 0; c < 8; ++c) {
                    s0[c] += x0 * right[c];
                    s1[c] += x1 * right[c];
                    s2[c] += x2 * right[c];
                    s3[c] += x3 * right[c];
                }
            }
            #pragma omp simd
            for (size_t c = 0; c < 8; ++c) {
                C[i * n + j + c] -= s0[c];
                C[(i + 1) * n + j + c] -= s1[c];
                C[(i + 2) * n + j + c] -= s2[c];
                C[(i + 3) * n + j + c] -= s3[c];
            }
        }
        if (j < cols) {
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = j; c < cols; ++c) {
                    double sum = 0.0;
                    for (size_t k = 0; k < depth; ++k) {
                        sum += X[(i + r) * n + k] * Y[k * packedStride + c];
                    }
                    C[(i + r) * n + c] -= sum;
                }
            }
        }
    }
    for (; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < depth; ++k) {
                sum += X[i * n + k] * Y[k * packedStride + j];
            }
            C[i * n + j] -= sum;
        }
    }
}

// Factor a diagonal tile after all earlier panel updates have completed.
// Return its size on success, or the local index of a nonpositive pivot.
size_t factorDiagonal(double* tile, const size_t n, const size_t size) {
    double inverse[2 * tileSize];
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += tile[i * n + k] * tile[j * n + k];
            }
            const double value = tile[i * n + j] - sum;
            if (i == j) {
                if (value <= 0.0) {
                    return i;
                }
                tile[i * n + j] = std::sqrt(value);
                inverse[i] = 1.0 / tile[i * n + j];
            } else {
                tile[i * n + j] = value * inverse[j];
            }
        }
    }
    return size;
}

// Solve L_ik L_kk^T = A_ik; rows of this panel tile are independent.
void solvePanel(double* panel, const double* diagonal, double* transposed,
                const size_t n, const size_t rows, const size_t width,
                const size_t packedStride) {
    double inverse[2 * tileSize];
    for (size_t j = 0; j < width; ++j) {
        inverse[j] = 1.0 / diagonal[j * n + j];
    }
    size_t i = 0;
    for (; i + 3 < rows; i += 4) {
        double* row0 = panel + i * n;
        double* row1 = row0 + n;
        double* row2 = row1 + n;
        double* row3 = row2 + n;
        for (size_t j = 0; j < width; ++j) {
            const double* d = diagonal + j * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            #pragma omp simd reduction(+:s0,s1,s2,s3)
            for (size_t k = 0; k < j; ++k) {
                s0 += row0[k] * d[k];
                s1 += row1[k] * d[k];
                s2 += row2[k] * d[k];
                s3 += row3[k] * d[k];
            }
            row0[j] = (row0[j] - s0) * inverse[j];
            row1[j] = (row1[j] - s1) * inverse[j];
            row2[j] = (row2[j] - s2) * inverse[j];
            row3[j] = (row3[j] - s3) * inverse[j];
            double* packedRow = transposed + j * packedStride + i;
            packedRow[0] = row0[j];
            packedRow[1] = row1[j];
            packedRow[2] = row2[j];
            packedRow[3] = row3[j];
        }
    }
    for (; i < rows; ++i) {
        double* row = panel + i * n;
        for (size_t j = 0; j < width; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * diagonal[j * n + k];
            }
            row[j] = (row[j] - sum) * inverse[j];
            transposed[j * packedStride + i] = row[j];
        }
    }
}

} // namespace

// In-place, tiled Cholesky decomposition: A = L L^T. Tasks belonging to
// different tiles can run concurrently. Each tile's first element is its
// dependency token, so updates to a tile occur in increasing panel order.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();
    // Larger panels amortize task overhead once there is ample parallel work.
    const size_t blockSize = n >= 4096 ? 2 * tileSize : tileSize;
    const size_t packedStride = blockSize + 8;
    const size_t packedTileElements = blockSize * packedStride;
    const size_t tiles = (n + blockSize - 1) / blockSize;
    // Only off-diagonal lower tiles need a packed copy. Allocate without
    // initialization: each panel task fills its tile before any reader runs.
    const size_t packedElements = tiles * (tiles == 0 ? 0 : tiles - 1) / 2 * packedTileElements;
    std::unique_ptr<double[]> packedStorage(new double[packedElements]);
    double* packed = packedStorage.get();
    std::atomic<size_t> failedPivot{n};
    const int workers = workerCount(n, blockSize);

    #pragma omp parallel num_threads(workers)
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                for (size_t k = 0; k < n; k += blockSize) {
                    const size_t width = std::min(blockSize, n - k);
                    const size_t diagonal = k * n + k;
                    #pragma omp task firstprivate(k,width,diagonal) depend(inout:data[diagonal]) shared(failedPivot)
                    {
                        if (failedPivot.load(std::memory_order_relaxed) == n) {
                            const size_t bad = factorDiagonal(data + diagonal, n, width);
                            if (bad != width) {
                                failedPivot.store(k + bad, std::memory_order_relaxed);
                            }
                        }
                    }

                    for (size_t i = k + width; i < n; i += blockSize) {
                        const size_t rows = std::min(blockSize, n - i);
                        const size_t panel = i * n + k;
                        const size_t rowTile = i / blockSize;
                        const size_t transposed = (rowTile * (rowTile - 1) / 2 + k / blockSize) * packedTileElements;
                        #pragma omp task firstprivate(rows,width,diagonal,panel,transposed) depend(in:data[diagonal]) depend(inout:data[panel]) shared(failedPivot)
                        {
                            if (failedPivot.load(std::memory_order_relaxed) == n) {
                                solvePanel(data + panel, data + diagonal, packed + transposed, n, rows, width, packedStride);
                            }
                        }
                    }

                    for (size_t j = k + width; j < n; j += blockSize) {
                        const size_t cols = std::min(blockSize, n - j);
                        const size_t right = j * n + k;
                        const size_t colTile = j / blockSize;
                        const size_t rightTransposed = (colTile * (colTile - 1) / 2 + k / blockSize) * packedTileElements;
                        for (size_t i = j; i < n; i += blockSize) {
                            const size_t rows = std::min(blockSize, n - i);
                            const size_t left = i * n + k;
                            const size_t output = i * n + j;
                            #pragma omp task firstprivate(rows,cols,width,left,right,rightTransposed,output) depend(in:data[left],data[right]) depend(inout:data[output]) shared(failedPivot)
                            {
                                if (failedPivot.load(std::memory_order_relaxed) == n) {
                                    updateTrailingTile(data + output, data + left,
                                                       packed + rightTransposed, n, rows, cols, width, packedStride);
                                }
                            }
                        }
                    }
                }
            }
        }

        // All tasks are finished before clearing the upper triangle.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }

    const size_t bad = failedPivot.load(std::memory_order_relaxed);
    if (bad != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", bad);
        return false;
    }
    return true;
}

// Generate the same seeded random B, then compute A = B B^T + n I.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // rand_r mutates its seed: preserve the original sequence and ordering.
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t tiles = (n + tileSize - 1) / tileSize;
    #pragma omp parallel for schedule(dynamic,1) num_threads(workerCount(n))
    for (size_t t = 0; t < tiles * tiles; ++t) {
        const size_t i = (t / tiles) * tileSize;
        const size_t j = (t % tiles) * tileSize;
        if (j > i) {
            continue;
        }
        const size_t rows = std::min(tileSize, n - i);
        const size_t cols = std::min(tileSize, n - j);
        multiplyTile(A.data() + i * n + j, B.data() + i * n,
                     B.data() + j * n, n, rows, cols, n);
        // Each lower tile owns its reflected upper tile as well.
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                if (i + r > j + c) {
                    A[(j + c) * n + i + r] = A[(i + r) * n + j + c];
                }
            }
        }
        if (i == j) {
            for (size_t r = 0; r < rows; ++r) {
                A[(i + r) * n + i + r] += n;
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    const size_t tiles = (n + tileSize - 1) / tileSize;

    // Reconstruct only the lower triangle and compare both symmetric entries.
    // Fuse comparison with reconstruction instead of allocating another matrix.
    #pragma omp parallel for schedule(dynamic,1) num_threads(workerCount(n)) reduction(max:maxError,relError)
    for (size_t t = 0; t < tiles * tiles; ++t) {
        const size_t i = (t / tiles) * tileSize;
        const size_t j = (t % tiles) * tileSize;
        if (j > i) {
            continue;
        }
        const size_t rows = std::min(tileSize, n - i);
        const size_t cols = std::min(tileSize, n - j);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                if (i + r < j + c) {
                    continue;
                }
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < n; ++k) {
                    sum += L[(i + r) * n + k] * L[(j + c) * n + k];
                }
                const double original = A_orig[(i + r) * n + j + c];
                const double error = std::fabs(sum - original);
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (std::fabs(original) + 1e-10));
                if (i + r != j + c) {
                    const double reflected = A_orig[(j + c) * n + i + r];
                    const double reflectedError = std::fabs(sum - reflected);
                    maxError = std::max(maxError, reflectedError);
                    relError = std::max(relError, reflectedError / (std::fabs(reflected) + 1e-10));
                }
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
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
