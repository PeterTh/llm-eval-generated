#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t PANEL = 64;

void computeRowRanges(const size_t n, const int nprocs, std::vector<size_t>& begins) {
    begins.assign(static_cast<size_t>(nprocs) + 1, 0);
    if (n == 0) {
        return;
    }
    const long double total =
        static_cast<long double>(n) * static_cast<long double>(n + 1) *
        static_cast<long double>(2 * n + 1) / 6.0L;
    long double cum = 0.0L;
    size_t i = 0;
    for (int r = 1; r < nprocs; ++r) {
        const long double want =
            total * static_cast<long double>(r) / static_cast<long double>(nprocs);
        while (i < n && cum < want) {
            const long double w = static_cast<long double>(i + 1) *
                                  static_cast<long double>(i + 1);
            cum += w;
            ++i;
        }
        begins[static_cast<size_t>(r)] = i;
    }
    begins[static_cast<size_t>(nprocs)] = n;
}

}

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           const int rank, const int nprocs) {
    if (n == 0) {
        return true;
    }

    std::vector<size_t> begins;
    computeRowRanges(n, nprocs, begins);
    const size_t lo = begins[static_cast<size_t>(rank)];
    const size_t hi = begins[static_cast<size_t>(rank + 1)];

    std::vector<double> recvbuf(n * PANEL);
    std::vector<double> sendbuf((hi > lo ? hi - lo : 0) * PANEL);
    std::vector<int> counts(static_cast<size_t>(nprocs));
    std::vector<int> displs(static_cast<size_t>(nprocs));

    bool failed = false;
    size_t failIndex = 0;

    for (size_t j = 0; j < n && !failed; j += PANEL) {
        const size_t e = std::min(j + PANEL, n);
        const size_t b = e - j;

        for (size_t c = j; c < e; ++c) {
            double* rc = &A[c * n];

            double sum = 0.0;
            for (size_t k = 0; k < c; ++k) {
                sum += rc[k] * rc[k];
            }
            const double val = rc[c] - sum;
            if (val <= 0.0) {
                failed = true;
                failIndex = c;
                break;
            }
            rc[c] = std::sqrt(val);

            for (size_t k = c + 1; k < n; ++k) {
                rc[k] = 0.0;
            }

            for (size_t i = c + 1; i < e; ++i) {
                double* ri = &A[i * n];
                double s = 0.0;
                for (size_t k = 0; k < c; ++k) {
                    s += ri[k] * rc[k];
                }
                ri[c] = (ri[c] - s) / rc[c];
            }

            const size_t start = std::max(lo, e);
            for (size_t i = start; i < hi; ++i) {
                double* ri = &A[i * n];
                double s = 0.0;
                for (size_t k = 0; k < c; ++k) {
                    s += ri[k] * rc[k];
                }
                ri[c] = (ri[c] - s) / rc[c];
            }
        }

        if (failed) {
            break;
        }

        int off = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t s0 = begins[static_cast<size_t>(r)];
            if (s0 < j) {
                s0 = j;
            }
            const size_t s1 = begins[static_cast<size_t>(r) + 1];
            const int cnt = (s1 > s0)
                                ? static_cast<int>((s1 - s0) * b)
                                : 0;
            counts[static_cast<size_t>(r)] = cnt;
            displs[static_cast<size_t>(r)] = off;
            off += cnt;
        }

        const size_t lstart = std::max(lo, j);
        const size_t lrows = (hi > lstart) ? hi - lstart : 0;
        const int lcnt = static_cast<int>(lrows * b);
        size_t pos = 0;
        for (size_t i = lstart; i < hi; ++i) {
            for (size_t c = j; c < e; ++c) {
                sendbuf[pos++] = A[i * n + c];
            }
        }

        MPI_Allgatherv(sendbuf.data(), lcnt, MPI_DOUBLE, recvbuf.data(),
                       counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        pos = 0;
        for (size_t i = j; i < n; ++i) {
            for (size_t c = j; c < e; ++c) {
                A[i * n + c] = recvbuf[pos++];
            }
        }
    }

    if (failed && rank == 0) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               failIndex);
    }

    return !failed;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n,
                                    const int rank, const int nprocs) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t r0 = (n * static_cast<size_t>(rank)) /
                      static_cast<size_t>(nprocs);
    const size_t r1 = (n * static_cast<size_t>(rank + 1)) /
                      static_cast<size_t>(nprocs);

    for (size_t i = r0; i < r1; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    std::vector<int> counts(static_cast<size_t>(nprocs));
    std::vector<int> displs(static_cast<size_t>(nprocs));
    for (int r = 0; r < nprocs; ++r) {
        const size_t s0 = (n * static_cast<size_t>(r)) /
                          static_cast<size_t>(nprocs);
        const size_t s1 = (n * static_cast<size_t>(r + 1)) /
                          static_cast<size_t>(nprocs);
        counts[static_cast<size_t>(r)] = static_cast<int>((s1 - s0) * n);
        displs[static_cast<size_t>(r)] = static_cast<int>(s0 * n);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, A.data(),
                   counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n,
                      const int rank, const int nprocs) {
    const size_t r0 = (n * static_cast<size_t>(rank)) /
                      static_cast<size_t>(nprocs);
    const size_t r1 = (n * static_cast<size_t>(rank + 1)) /
                      static_cast<size_t>(nprocs);

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = r0; i < r1; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);

            const double rel =
                error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double globalMax = 0.0;
    double globalRel = 0.0;
    MPI_Reduce(&maxError, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&relError, &globalRel, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        printf("Max absolute error: %.10e\n", globalMax);
        printf("Max relative error: %.10e\n", globalRel);

        if (globalRel > 1e-6) {
            printf("Validation failed: relative error too large\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return valid != 0;
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

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    generatePositiveDefiniteMatrix(A, n, rank, nprocs);

    if (validate) {
        A_orig = A;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const bool success = choleskyDecomposition(A, n, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localDuration = t1 - t0;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long ms = static_cast<long>(duration * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double ops =
            static_cast<double>(n) * static_cast<double>(n) *
            static_cast<double>(n) / 3.0;
        const double gflops = ops / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateCholesky(A, A_orig, n, rank, nprocs);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
