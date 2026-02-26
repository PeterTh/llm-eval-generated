#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Right-looking blocked Cholesky with 1D block-cyclic column distribution
// Block size for panel factorization and trailing update
static constexpr int NB = 64;

// Generate a symmetric positive definite matrix (deterministic with seed 42)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    const int N = (int)n;
    const int nblocks = (N + NB - 1) / NB;
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // 1D block-cyclic column distribution: block b -> rank (b % nprocs)
    int ncols_local = 0;
    for (int b = rank; b < nblocks; b += nprocs) {
        int cs = b * NB;
        int ce = std::min(cs + NB, N);
        ncols_local += ce - cs;
    }
    
    // Column index mappings
    std::vector<int> l2g(ncols_local);
    std::vector<int> g2l(N, -1);
    {
        int lc = 0;
        for (int b = rank; b < nblocks; b += nprocs) {
            int cs = b * NB;
            int ce = std::min(cs + NB, N);
            for (int c = cs; c < ce; c++) {
                l2g[lc] = c;
                g2l[c] = lc;
                lc++;
            }
        }
    }
    
    // Generate matrix (all ranks, deterministic via rand_r with seed 42)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    
    std::vector<double> A_full((size_t)N * N);
    generatePositiveDefiniteMatrix(A_full, n);
    
    std::vector<double> A_orig;
    if (validate && rank == 0) A_orig = A_full;
    
    // Extract local columns into column-major storage
    // local_data[row + lc * N] = A[row][l2g[lc]]
    std::vector<double> local_data((size_t)N * ncols_local);
    for (int lc = 0; lc < ncols_local; lc++) {
        int gc = l2g[lc];
        for (int i = 0; i < N; i++)
            local_data[i + (size_t)lc * N] = A_full[(size_t)i * N + gc];
    }
    A_full.clear();
    A_full.shrink_to_fit();
    
    // Right-looking blocked Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> panel_buf((size_t)N * NB);
    
    for (int b = 0; b < nblocks; b++) {
        const int jb = b * NB;
        const int jend = std::min(jb + NB, N);
        const int nb = jend - jb;
        const int owner = b % nprocs;
        
        // Panel factorization: unblocked Cholesky on A[jb:N, jb:jend]
        if (rank == owner) {
            const int lcs = g2l[jb];
            
            for (int j = 0; j < nb; j++) {
                const int gj = jb + j;
                const int lj = lcs + j;
                
                // Diagonal: L[gj][gj] = sqrt(A[gj][gj] - sum_{k<j} L[gj][jb+k]^2)
                double s = 0.0;
                for (int k = 0; k < j; k++) {
                    double v = local_data[gj + (size_t)(lcs + k) * N];
                    s += v * v;
                }
                double val = local_data[gj + (size_t)lj * N] - s;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %d\n", gj);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                double diag = sqrt(val);
                local_data[gj + (size_t)lj * N] = diag;
                
                // Off-diagonal: L[i][gj] for i > gj
                for (int i = gj + 1; i < N; i++) {
                    s = 0.0;
                    for (int k = 0; k < j; k++)
                        s += local_data[i + (size_t)(lcs + k) * N] *
                             local_data[gj + (size_t)(lcs + k) * N];
                    local_data[i + (size_t)lj * N] =
                        (local_data[i + (size_t)lj * N] - s) / diag;
                }
            }
            
            // Pack panel for broadcast (zero upper triangle)
            for (int j = 0; j < nb; j++) {
                const int gj = jb + j;
                const int lj = lcs + j;
                for (int i = 0; i < gj; i++)
                    panel_buf[i + (size_t)j * N] = 0.0;
                for (int i = gj; i < N; i++)
                    panel_buf[i + (size_t)j * N] = local_data[i + (size_t)lj * N];
            }
        }
        
        // Broadcast panel
        MPI_Bcast(panel_buf.data(), N * nb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Trailing matrix update: A[i][gk] -= sum_j panel[i][j]*panel[gk][j]
        // Only lower triangle (i >= gk), only columns gk >= jend
        int lc_trail = (int)(std::lower_bound(l2g.begin(), l2g.end(), jend) - l2g.begin());
        for (int lc = lc_trail; lc < ncols_local; lc++) {
            const int gk = l2g[lc];
            double* __restrict__ lcol = &local_data[(size_t)lc * N];
            
            for (int j = 0; j < nb; j++) {
                const double L_kj = panel_buf[gk + (size_t)j * N];
                const double* __restrict__ pcol = &panel_buf[(size_t)j * N];
                for (int i = gk; i < N; i++)
                    lcol[i] -= pcol[i] * L_kj;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Zero upper triangle of local columns
    for (int lc = 0; lc < ncols_local; lc++) {
        const int gc = l2g[lc];
        for (int i = 0; i < gc; i++)
            local_data[i + (size_t)lc * N] = 0.0;
    }
    
    // Gather result to rank 0 (column-major local -> row-major full)
    std::vector<double> A_result;
    if (rank == 0) A_result.resize((size_t)N * N, 0.0);
    
    {
        std::vector<double> blk_buf((size_t)N * NB);
        for (int b = 0; b < nblocks; b++) {
            const int bowner = b % nprocs;
            const int cs = b * NB;
            const int ce = std::min(cs + NB, N);
            const int nb_cols = ce - cs;
            
            if (rank == bowner) {
                const int lcs = g2l[cs];
                for (int j = 0; j < nb_cols; j++)
                    for (int i = 0; i < N; i++)
                        blk_buf[i + (size_t)j * N] =
                            local_data[i + (size_t)(lcs + j) * N];
                if (bowner != 0)
                    MPI_Send(blk_buf.data(), N * nb_cols, MPI_DOUBLE, 0, b,
                             MPI_COMM_WORLD);
            }
            if (rank == 0) {
                if (bowner != 0)
                    MPI_Recv(blk_buf.data(), N * nb_cols, MPI_DOUBLE, bowner, b,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < nb_cols; j++)
                    for (int i = 0; i < N; i++)
                        A_result[(size_t)i * N + cs + j] =
                            blk_buf[i + (size_t)j * N];
            }
        }
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        if (printResults) {
            print_results(A_result, "CholeskyL");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A_result, A_orig, n);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
