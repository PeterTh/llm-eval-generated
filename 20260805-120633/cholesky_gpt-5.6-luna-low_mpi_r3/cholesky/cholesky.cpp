#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include "../common/results_output.hpp"

static void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += L[i*n+k] * L[j*n+k];
        double error = std::fabs(sum - original[i*n+j]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(original[i*n+j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) { std::printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num> Matrix size (default: 512)\n  -v      Enable validation\n  -r      Print results for external validation\n  -h      Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(INT_MAX)) { if (rank == 0) std::printf("Invalid matrix size\n"); MPI_Finalize(); return 1; }

    const size_t first = (n * static_cast<size_t>(rank)) / world;
    const size_t last = (n * static_cast<size_t>(rank + 1)) / world;
    const size_t localRows = last - first;
    std::vector<double> global, original, A(localRows * n);
    std::vector<int> counts(world), displs(world);
    std::vector<int> rowCounts(world), rowDispls(world);
    for (int r = 0; r < world; ++r) {
        size_t b = n * (n * static_cast<size_t>(r) / world);
        size_t e = n * (n * static_cast<size_t>(r + 1) / world);
        counts[r] = static_cast<int>(e - b); displs[r] = static_cast<int>(b);
        rowCounts[r] = static_cast<int>(e / n - b / n); rowDispls[r] = static_cast<int>(b / n);
    }
    if (rank == 0) { std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled"); global.resize(n*n); generatePositiveDefiniteMatrix(global,n); if (validate) original=global; }
    MPI_Scatterv(rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, A.data(), static_cast<int>(A.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    std::vector<double> column(localRows), fullColumn(n);
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        if (k >= first && k < last) {
            size_t row = (k-first)*n; double sum = 0.0;
            for (size_t q=0; q<k; ++q) sum += A[row+q]*A[row+q];
            double value = A[row+k]-sum;
            if (value <= 0.0) success=false; else A[row+k]=std::sqrt(value);
        }
        int ok = success ? 1 : 0; MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        if (!ok) { success=false; break; }
        // Form this column locally before exchanging it with the other ranks.
        const double diagonal = (k >= first && k < last) ? A[(k-first)*n+k] : 0.0;
        double pivot = 0.0;
        MPI_Allreduce(&diagonal, &pivot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i=first; i<last; ++i) if (i>k) {
            size_t row=(i-first)*n;
            A[row+k] /= pivot;
        }
        for (size_t i=0;i<localRows;++i) column[i]=A[i*n+k];
        MPI_Allgatherv(column.data(), static_cast<int>(localRows), MPI_DOUBLE, fullColumn.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (size_t i=first;i<last;++i) if (i>k) {
            size_t row=(i-first)*n; double factor=fullColumn[i];
            for (size_t j=k+1;j<=i;++j) A[row+j] -= factor*fullColumn[j];
            A[row+k]=factor;
        }
    }
    for (size_t i=first; i<last; ++i)
        for (size_t j=i+1; j<n; ++j) A[(i-first)*n+j]=0.0;
    double elapsed = MPI_Wtime()-start, maxElapsed=0.0;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<double> result;
    if (rank==0) result.resize(n*n);
    MPI_Gatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE, rank==0?result.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int globalOK=success?1:0; MPI_Allreduce(MPI_IN_PLACE,&globalOK,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    int rc=0;
    if (rank==0) { if (!globalOK) { std::printf("Cholesky decomposition failed\n"); rc=1; } else { std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", static_cast<long>(maxElapsed*1000.0), (n*n*n/3.0)/(maxElapsed*1e9)); if(printResults) print_results(result,"CholeskyL"); if(validate) { std::printf("Validating result...\n"); bool v=validateCholesky(result,original,n); std::printf("Validation: %s\n",v?"PASSED":"FAILED"); rc=!v; } } }
    MPI_Finalize(); return rc;
}
