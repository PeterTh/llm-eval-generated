#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, i, j);
}

inline void block_range(int n, int nblocks, int idx, int& start, int& sz) {
    int base = n / nblocks;
    int rem = n % nblocks;
    start = idx * base + std::min(idx, rem);
    sz = base + (idx < rem ? 1 : 0);
}

void localMatmul(const double* __restrict__ A, const double* __restrict__ B,
                 double* __restrict__ C, int m, int n, int k,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            double a = A[i * lda + p];
            for (int j = 0; j < n; ++j) {
                C[i * ldc + j] += a * B[p * ldb + j];
            }
        }
    }
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

    int rank, nprocs;
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

    // 2D process grid (as square as possible): pr <= pc, pr * pc = nprocs
    int pr = 1, pc = nprocs;
    for (int p = (int)std::sqrt(nprocs); p >= 1; --p) {
        if (nprocs % p == 0) {
            pr = p;
            pc = nprocs / p;
            break;
        }
    }
    int K = pr; // min(pr, pc)

    int myrow = rank / pc;
    int mycol = rank % pc;

    int r_start, r_sz, c_start, c_sz;
    block_range((int)N, pr, myrow, r_start, r_sz);
    block_range((int)N, pc, mycol, c_start, c_sz);

    int a_start, a_sz;
    block_range((int)N, K, mycol, a_start, a_sz);
    int b_start, b_sz;
    block_range((int)N, K, myrow, b_start, b_sz);

    bool has_A = (mycol < K);
    bool has_B = (myrow < K); // always true since myrow < pr = K

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Process grid: %d x %d\n", pr, pc);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A_local(has_A ? (size_t)r_sz * a_sz : 0);
    std::vector<double> B_local(has_B ? (size_t)b_sz * c_sz : 0);
    std::vector<double> C_local((size_t)r_sz * c_sz, 0.0);

    // Distribute A and B from rank 0
    if (rank == 0) {
        printf("Initializing matrices...\n");
        std::vector<double> fullA(N * N), fullB(N * N);
        initMatrix(fullA, N);
        initMatrix(fullB, N);

        // Copy own blocks
        if (has_A) {
            for (int r = 0; r < r_sz; ++r)
                for (int c = 0; c < a_sz; ++c)
                    A_local[r * a_sz + c] = fullA[(r_start + r) * N + a_start + c];
        }
        if (has_B) {
            for (int r = 0; r < b_sz; ++r)
                for (int c = 0; c < c_sz; ++c)
                    B_local[r * c_sz + c] = fullB[(b_start + r) * N + c_start + c];
        }

        // Send blocks to other processes
        for (int dst = 1; dst < nprocs; ++dst) {
            int di = dst / pc, dj = dst % pc;
            int dr_s, dr_n, dc_s, dc_n;
            block_range((int)N, pr, di, dr_s, dr_n);
            block_range((int)N, pc, dj, dc_s, dc_n);

            if (dj < K) {
                int da_s, da_n;
                block_range((int)N, K, dj, da_s, da_n);
                std::vector<double> buf(dr_n * da_n);
                for (int r = 0; r < dr_n; ++r)
                    for (int c = 0; c < da_n; ++c)
                        buf[r * da_n + c] = fullA[(dr_s + r) * N + da_s + c];
                MPI_Send(buf.data(), dr_n * da_n, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD);
            }
            if (di < K) {
                int db_s, db_n;
                block_range((int)N, K, di, db_s, db_n);
                std::vector<double> buf(db_n * dc_n);
                for (int r = 0; r < db_n; ++r)
                    for (int c = 0; c < dc_n; ++c)
                        buf[r * dc_n + c] = fullB[(db_s + r) * N + dc_s + c];
                MPI_Send(buf.data(), db_n * dc_n, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        if (has_A)
            MPI_Recv(A_local.data(), r_sz * a_sz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (has_B)
            MPI_Recv(B_local.data(), b_sz * c_sz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Row and column communicators for SUMMA
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, myrow, mycol, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, mycol, myrow, &col_comm);

    if (rank == 0) printf("Computing matrix multiplication...\n");

    // Pre-compute max inner block size for broadcast buffers
    int max_inn = 0;
    for (int k = 0; k < K; ++k) {
        int s, sz;
        block_range((int)N, K, k, s, sz);
        max_inn = std::max(max_inn, sz);
    }

    std::vector<double> A_bcast((size_t)r_sz * max_inn);
    std::vector<double> B_bcast((size_t)max_inn * c_sz);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    // SUMMA: C_local += sum_{k=0}^{K-1} A_bcast_k * B_bcast_k
    for (int k = 0; k < K; ++k) {
        int inn_sz;
        { int s; block_range((int)N, K, k, s, inn_sz); }

        if (mycol == k) {
            for (int r = 0; r < r_sz; ++r)
                for (int c = 0; c < a_sz; ++c)
                    A_bcast[r * inn_sz + c] = A_local[r * a_sz + c];
        }
        MPI_Bcast(A_bcast.data(), r_sz * inn_sz, MPI_DOUBLE, k, row_comm);

        if (myrow == k) {
            for (int r = 0; r < b_sz; ++r)
                for (int c = 0; c < c_sz; ++c)
                    B_bcast[r * c_sz + c] = B_local[r * c_sz + c];
        }
        MPI_Bcast(B_bcast.data(), inn_sz * c_sz, MPI_DOUBLE, k, col_comm);

        localMatmul(A_bcast.data(), B_bcast.data(), C_local.data(),
                    r_sz, c_sz, inn_sz, inn_sz, c_sz, c_sz);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C to rank 0
    std::vector<double> fullC;
    if (rank == 0) {
        fullC.resize(N * N);
        for (int r = 0; r < r_sz; ++r)
            for (int c = 0; c < c_sz; ++c)
                fullC[(r_start + r) * N + c_start + c] = C_local[r * c_sz + c];

        for (int src = 1; src < nprocs; ++src) {
            int si = src / pc, sj = src % pc;
            int sr_s, sr_n, sc_s, sc_n;
            block_range((int)N, pr, si, sr_s, sr_n);
            block_range((int)N, pc, sj, sc_s, sc_n);
            std::vector<double> buf(sr_n * sc_n);
            MPI_Recv(buf.data(), sr_n * sc_n, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int r = 0; r < sr_n; ++r)
                for (int c = 0; c < sc_n; ++c)
                    fullC[(sr_s + r) * N + sc_s + c] = buf[r * sc_n + c];
        }
    } else {
        MPI_Send(C_local.data(), r_sz * c_sz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    int result = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(fullC, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            std::vector<double> vA(N * N), vB(N * N);
            initMatrix(vA, N);
            initMatrix(vB, N);

            bool valid = true;
            constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
            for (size_t pi = 0; pi < 5 && valid; ++pi) {
                for (size_t pj = 0; pj < 5 && valid; ++pj) {
                    const size_t i = checkPoints[pi] % N;
                    const size_t j = checkPoints[pj] % N;
                    double expected = 0.0;
                    for (size_t kk = 0; kk < N; ++kk)
                        expected += vA[i * N + kk] * vB[kk * N + j];
                    const double actual = fullC[i * N + j];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, actual, relError);
                        valid = false;
                    }
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return result;
}
