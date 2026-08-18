#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Contiguous block-row distribution.  Keeping complete rows makes the trailing
// rank-1 update contiguous and avoids communicating matrix blocks.
struct Distribution {
    int rank, size;
    size_t n, first, rows;
    std::vector<int> rowCounts, rowDispls, elemCounts, elemDispls;

    Distribution(size_t order, int r, int p) : rank(r), size(p), n(order) {
        rowCounts.resize(size);
        rowDispls.resize(size);
        elemCounts.resize(size);
        elemDispls.resize(size);
        const size_t q = n / static_cast<size_t>(size);
        const size_t rem = n % static_cast<size_t>(size);
        size_t row = 0;
        for (int i = 0; i < size; ++i) {
            const size_t nr = q + (static_cast<size_t>(i) < rem);
            rowCounts[i] = static_cast<int>(nr);
            rowDispls[i] = static_cast<int>(row);
            elemCounts[i] = static_cast<int>(nr * n);
            elemDispls[i] = static_cast<int>(row * n);
            row += nr;
        }
        first = static_cast<size_t>(rowDispls[rank]);
        rows = static_cast<size_t>(rowCounts[rank]);
    }
};

static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i*n+k] * B[j*n+k];
            A[i*n+j] = sum;
        }
        A[i*n+i] += n;
    }
}

// Right-looking distributed Cholesky.  At step k, every rank computes the
// entries of column k that it owns.  Allgatherv makes that column available for
// the local, cache-friendly update of the remaining lower triangle.
static bool distributedCholesky(std::vector<double>& local, const Distribution& d,
                                MPI_Comm comm) {
    std::vector<double> column(d.n);
    for (size_t k = 0; k < d.n; ++k) {
        double diagonal = 0.0;
        if (k >= d.first && k < d.first + d.rows) {
            double& akk = local[(k - d.first) * d.n + k];
            diagonal = akk;
            if (diagonal > 0.0) akk = std::sqrt(diagonal);
        }
        MPI_Allreduce(MPI_IN_PLACE, &diagonal, 1, MPI_DOUBLE, MPI_SUM, comm);
        const int locallyBad = !(diagonal > 0.0) || !std::isfinite(diagonal);
        int globallyBad = 0;
        MPI_Allreduce(&locallyBad, &globallyBad, 1, MPI_INT, MPI_MAX, comm);
        if (globallyBad) {
            if (d.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            return false;
        }
        const double pivot = std::sqrt(diagonal);

        for (size_t lr = 0; lr < d.rows; ++lr) {
            const size_t i = d.first + lr;
            column[i] = (i == k) ? pivot : (i > k ? local[lr*d.n+k] / pivot : 0.0);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, column.data(),
                       d.rowCounts.data(), d.rowDispls.data(), MPI_DOUBLE, comm);

        for (size_t lr = 0; lr < d.rows; ++lr) {
            const size_t i = d.first + lr;
            if (i <= k) continue;
            double* const row = local.data() + lr*d.n;
            const double lik = column[i];
            row[k] = lik;
            // Only the lower triangle is needed.  This loop is contiguous and
            // is the dominant, readily vectorized part of the factorization.
            for (size_t j = k + 1; j <= i; ++j) row[j] -= lik * column[j];
        }
    }

    // Match the sequential program's full lower-triangular output semantics.
    for (size_t lr = 0; lr < d.rows; ++lr) {
        const size_t i = d.first + lr;
        std::fill(local.begin() + static_cast<std::ptrdiff_t>(lr*d.n+i+1),
                  local.begin() + static_cast<std::ptrdiff_t>((lr+1)*d.n), 0.0);
    }
    return true;
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        const size_t end = std::min(i, j);
        for (size_t k = 0; k <= end; ++k) sum += L[i*n+k] * L[j*n+k];
        const double error = std::fabs(sum - original[i*n+j]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(original[i*n+j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0 || value > std::numeric_limits<size_t>::max()) parseStatus = 1;
            else n = static_cast<size_t>(value);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) parseStatus = 2;
        else parseStatus = 1;
    }
    if (parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (n != 0 && n > static_cast<size_t>(std::numeric_limits<int>::max()) / n)) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI collective counts\n");
        MPI_Finalize();
        return 1;
    }

    Distribution d(n, rank, size);
    std::vector<double> global, original;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\nMPI processes: %d\n", validate ? "enabled" : "disabled", size);
        std::printf("Generating positive definite matrix...\n");
        global.resize(n*n);
        generatePositiveDefiniteMatrix(global, n);
        if (validate) original = global;
    }
    std::vector<double> local(d.rows*n);
    MPI_Scatterv(rank == 0 ? global.data() : nullptr, d.elemCounts.data(), d.elemDispls.data(),
                 MPI_DOUBLE, local.data(), d.elemCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<double>().swap(global);
        std::printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(local, d, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0 && (validate || printResults)) global.resize(n*n);
    if (validate || printResults)
        MPI_Gatherv(local.data(), d.elemCounts[rank], MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, d.elemCounts.data(), d.elemDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(duration * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? ops / duration / 1e9 : 0.0);
        if (printResults) print_results(global, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateCholesky(global, original, n) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
