#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            if (i == j) {
                // Diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
        
        // Zero out upper triangular part
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
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
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    
    if (rank == 0) {
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n), local;
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if(rank==0) { printf("Generating positive definite matrix...\n"); generatePositiveDefiniteMatrix(A, n); }
    MPI_Bcast(A.data(), static_cast<int>(n*n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Distribute rows cyclically. Pivot owner factors its row, then broadcasts it;
    // every rank updates only its local rows.
    local.reserve(((n + ranks - 1) / ranks) * n);
    std::vector<size_t> rows;
    for(size_t i=rank;i<n;i+=ranks) { rows.push_back(i); local.insert(local.end(), A.begin()+i*n, A.begin()+(i+1)*n); }
    if(rank==0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    bool success = true;
    std::vector<double> pivot(n);
    for(size_t k=0;k<n;++k) {
        const int owner = static_cast<int>(k % ranks);
        if(rank==owner) {
            size_t idx=k/ranks;
            double* row=&local[idx*n];
            double d=row[k];
            for(size_t q=0;q<k;++q) d-=row[q]*row[q];
            if(d<=0.0) success=false;
            else row[k]=std::sqrt(d);
            std::copy(row,row+n,pivot.begin());
        }
        MPI_Bcast(&success,1,MPI_C_BOOL,owner,MPI_COMM_WORLD);
        if(!success) break;
        MPI_Bcast(pivot.data(),static_cast<int>(n),MPI_DOUBLE,owner,MPI_COMM_WORLD);
        for(size_t r=0;r<rows.size();++r) {
            size_t i=rows[r]; if(i<=k) continue;
            double* row=&local[r*n]; double v=row[k];
            for(size_t q=0;q<k;++q) v-=row[q]*pivot[q];
            row[k]=v/pivot[k];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    // Reassemble on root for equivalent output and validation.
    std::vector<int> counts(ranks), displs(ranks);
    for(int p=0;p<ranks;++p) { size_t c=0; for(size_t i=p;i<n;i+=ranks) ++c; counts[p]=static_cast<int>(c*n); displs[p]=static_cast<int>(p*n); }
    // Cyclic rows are packed; gather them to root then place by global row.
    int localCount=static_cast<int>(local.size());
    std::vector<int> allcounts(ranks), offsets(ranks);
    MPI_Gather(&localCount,1,MPI_INT,allcounts.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    std::vector<double> packed;
    if(rank==0) { int total=0; for(int p=0;p<ranks;++p){ offsets[p]=total; total+=allcounts[p]; } packed.resize(total); }
    MPI_Gatherv(local.data(),localCount,MPI_DOUBLE,packed.data(),allcounts.data(),offsets.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        for(int p=0;p<ranks;++p) { size_t off=offsets[p]; for(size_t i=p;i<n;i+=ranks) { std::copy_n(packed.data()+off,n,A.data()+i*n); off+=n; } }
        for(size_t i=0;i<n;++i) std::fill(A.begin()+i*n+i+1,A.begin()+(i+1)*n,0.0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if(rank==0) {
        printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize(); return 1;
    }
    if(rank!=0) { MPI_Finalize(); return 0; }
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
