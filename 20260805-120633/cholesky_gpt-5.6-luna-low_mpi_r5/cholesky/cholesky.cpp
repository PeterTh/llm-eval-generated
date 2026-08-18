#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static void usage(const char *p) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

static size_t first_row(size_t n, int rank, int size) { return n * static_cast<size_t>(rank) / size; }
static size_t last_row(size_t n, int rank, int size) { return n * static_cast<size_t>(rank + 1) / size; }
static int owner_of(size_t row, size_t n, int size) {
    int lo = 0, hi = size - 1;
    while (lo < hi) { int m = (lo + hi) / 2; if (row < first_row(n, m + 1, size)) hi = m; else lo = m + 1; }
    return lo;
}

// Matches the original deterministic construction, without replicating the matrix.
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, output_results = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) output_results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    const size_t begin = first_row(n, rank, ranks), end = last_row(n, rank, ranks), rows = end - begin;
    std::vector<double> A(rows * n), original;
    // Generate the same B stream independently on every rank, then form only owned A rows.
    std::vector<double> B(n * n); unsigned int seed = 42;
    for (double &x : B) x = rand_r(&seed) / (double)RAND_MAX - 0.5;
    for (size_t i = begin; i < end; ++i) for (size_t j = 0; j < n; ++j) {
        double s = 0.0; for (size_t k = 0; k < n; ++k) s += B[i*n+k] * B[j*n+k];
        A[(i-begin)*n+j] = s + (i == j ? static_cast<double>(n) : 0.0);
    }
    if (validate) original = A;
    B.clear(); B.shrink_to_fit();
    std::vector<double> pivot(n + 1);
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    bool success = true;
    for (size_t k = 0; k < n; ++k) {
        int owner = owner_of(k, n, ranks);
        if (rank == owner) {
            double sum = 0.0, *row = &A[(k-begin)*n];
            for (size_t q = 0; q < k; ++q) sum += row[q] * row[q];
            double value = row[k] - sum;
            if (value <= 0.0) success = false; else row[k] = std::sqrt(value);
            for (size_t j = k + 1; j < n; ++j) row[j] = 0.0;
            std::copy(row, row + n, pivot.begin());
        }
        int ok = success ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) { success = false; break; }
        MPI_Bcast(pivot.data(), static_cast<int>(n + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (size_t i = std::max(k + 1, begin); i < end; ++i) {
            double *row = &A[(i-begin)*n]; double sum = 0.0;
            for (size_t q = 0; q < k; ++q) sum += row[q] * pivot[q];
            row[k] = (row[k] - sum) / pivot[k];
            for (size_t j = i + 1; j < n; ++j) row[j] = 0.0;
        }
    }
    double elapsed = MPI_Wtime() - start, elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (!rank) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    std::vector<double> full, full_orig;
    if (rank == 0) { full.resize(n*n); if (validate) full_orig.resize(n*n); }
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0;r<ranks;++r) { counts[r]=static_cast<int>((last_row(n,r,ranks)-first_row(n,r,ranks))*n); displs[r]=static_cast<int>(first_row(n,r,ranks)*n); }
    if (!rank) { std::copy(A.begin(), A.end(), full.begin()); if (validate) std::copy(original.begin(), original.end(), full_orig.begin()); }
    MPI_Gatherv(rank ? A.data() : MPI_IN_PLACE, rank ? counts[rank] : 0, MPI_DOUBLE, rank ? nullptr : full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate) MPI_Gatherv(rank ? original.data() : MPI_IN_PLACE, rank ? counts[rank] : 0, MPI_DOUBLE, rank ? nullptr : full_orig.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nComputation time: %.3f ms\nPerformance: %.3f GFLOPS\n", n,n,validate?"enabled":"disabled",elapsed_max*1000.0,(n*n*n/3.0)/(elapsed_max*1e9));
        if (output_results) print_results(full, "CholeskyL");
        if (validate) { double err=0, rel=0; for(size_t i=0;i<n*n;++i){ double e=std::fabs(full[i]-full_orig[i]); err=std::max(err,e); rel=std::max(rel,e/(std::fabs(full_orig[i])+1e-10)); } std::printf("Max absolute error: %.10e\nMax relative error: %.10e\nValidation: %s\n",err,rel,rel<=1e-6?"PASSED":"FAILED"); if(rel>1e-6){MPI_Finalize();return 1;} }
    }
    MPI_Finalize(); return 0;
}
