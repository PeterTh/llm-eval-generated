#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill rows [r0, r0 + rows) of the matrix into mat (local row-major buffer)
void initRows(double* mat, const size_t N, const size_t r0, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, r0 + i, j);
        }
    }
}

// C (rows x N) = A (rows x N) * B (N x N); per-element summation order over k is ascending,
// identical to the serial reference.
void matrixMultiply(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    constexpr size_t JB = 512, KB = 128;
    std::fill(C, C + rows * N, 0.0);
    for (size_t jb = 0; jb < N; jb += JB) {
        const size_t je = std::min(jb + JB, N);
        for (size_t kb = 0; kb < N; kb += KB) {
            const size_t ke = std::min(kb + KB, N);
            size_t i = 0;
            for (; i + 2 <= rows; i += 2) {
                double* __restrict c0 = C + i * N;
                double* __restrict c1 = c0 + N;
                for (size_t k = kb; k < ke; ++k) {
                    const double a0 = A[i * N + k], a1 = A[(i + 1) * N + k];
                    const double* __restrict b = B + k * N;
                    for (size_t j = jb; j < je; ++j) {
                        c0[j] += a0 * b[j];
                        c1[j] += a1 * b[j];
                    }
                }
            }
            for (; i < rows; ++i) {
                double* __restrict c0 = C + i * N;
                for (size_t k = kb; k < ke; ++k) {
                    const double a0 = A[i * N + k];
                    const double* __restrict b = B + k * N;
                    for (size_t j = jb; j < je; ++j) c0[j] += a0 * b[j];
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // A must hold at least the first min(5, N) rows
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
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row distribution (block, remainder spread over first ranks)
    std::vector<size_t> rowStart(nprocs + 1, 0);
    for (int r = 0; r < nprocs; ++r) {
        rowStart[r + 1] = rowStart[r] + N / nprocs + (static_cast<size_t>(r) < N % nprocs ? 1 : 0);
    }
    const size_t myRows = rowStart[rank + 1] - rowStart[rank];

    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(myRows * N);

    if (rank == 0) printf("Initializing matrices...\n");
    initRows(A.data(), N, rowStart[rank], myRows);
    // Each rank generates its own row block of B, then all blocks are shared
    initRows(B.data() + rowStart[rank] * N, N, rowStart[rank], myRows);
    {
        std::vector<int> cnt(nprocs), displ(nprocs);
        bool fits = N * N <= 0x7fffffff;
        for (int r = 0; r < nprocs; ++r) {
            cnt[r] = static_cast<int>((rowStart[r + 1] - rowStart[r]) * N);
            displ[r] = static_cast<int>(rowStart[r] * N);
        }
        if (fits) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, B.data(), cnt.data(), displ.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        } else {
            // Large sizes: broadcast row by row blocks to stay within int counts
            for (int r = 0; r < nprocs; ++r) {
                size_t total = (rowStart[r + 1] - rowStart[r]) * N;
                double* p = B.data() + rowStart[r] * N;
                while (total > 0) {
                    int c = static_cast<int>(std::min<size_t>(total, 1u << 30));
                    MPI_Bcast(p, c, MPI_DOUBLE, r, MPI_COMM_WORLD);
                    p += c;
                    total -= c;
                }
            }
        }
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A.data(), B.data(), C.data(), myRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_ms);
        double gflops = (2.0 * N * N * N) / (max_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full C on rank 0 only if needed
    std::vector<double> Cfull;
    if (printResults || validate) {
        if (rank == 0) Cfull.resize(N * N);
        std::vector<int> cnt(nprocs), displ(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            cnt[r] = static_cast<int>((rowStart[r + 1] - rowStart[r]) * N);
            displ[r] = static_cast<int>(rowStart[r] * N);
        }
        if (N * N <= 0x7fffffff) {
            MPI_Gatherv(C.data(), static_cast<int>(myRows * N), MPI_DOUBLE, Cfull.data(), cnt.data(),
                        displ.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            // Fall back to per-rank chunked point-to-point
            for (int r = 0; r < nprocs; ++r) {
                size_t total = (rowStart[r + 1] - rowStart[r]) * N;
                double* p = (rank == 0) ? Cfull.data() + rowStart[r] * N : C.data();
                while (total > 0) {
                    int c = static_cast<int>(std::min<size_t>(total, 1u << 30));
                    if (r == 0) {
                        if (rank == 0) std::copy(C.data(), C.data() + c, p);
                    } else if (rank == 0) {
                        MPI_Recv(p, c, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else if (rank == r) {
                        MPI_Send(p, c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                    p += c;
                    total -= c;
                }
            }
        }
    }

    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(Cfull, "MatrixC");
        }
        if (validate) {
            printf("Validating result...\n");
            // Rows of A needed for validation
            const size_t vr = std::min<size_t>(5, N);
            std::vector<double> Av(vr * N);
            initRows(Av.data(), N, 0, vr);
            bool valid = validateResult(Av, B, Cfull, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
