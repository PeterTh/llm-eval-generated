#include <mpi.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// glibc-compatible rand() (TYPE_3 additive feedback generator) with O(log n)
// jump-ahead, so every rank can generate its slice of the exact same random
// stream the serial program would have produced.
//   r[i] = r[i-31] + r[i-3] (mod 2^32), output = r[i] >> 1
// ****************************************************************************
class GlibcRand {
public:
    static constexpr int N = 31;

    explicit GlibcRand(uint32_t seed) {
        int32_t r[34];
        if (seed == 0) seed = 1;
        r[0] = static_cast<int32_t>(seed);
        for (int i = 1; i < N; ++i) {
            const int64_t hi = r[i - 1] / 127773;
            const int64_t lo = r[i - 1] % 127773;
            int64_t word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            r[i] = static_cast<int32_t>(word);
        }
        for (int i = N; i < 34; ++i) r[i] = r[i - N];
        for (int i = 0; i < N; ++i) base_[i] = static_cast<uint32_t>(r[i + 3]);
        jumpTo(0);
    }

    // Position the generator so that the next call returns draw number m (0-based)
    void jumpTo(uint64_t m) {
        // window[j] = r[m + 313 + j] = r[3 + (m + 310) + j]
        Poly p = powX(m + 310);
        for (int j = 0; j < N; ++j) {
            uint32_t acc = 0;
            for (int i = 0; i < N; ++i) acc += p[i] * base_[i];
            w_[j] = acc;
            p = mulX(p);
        }
        pos_ = 0;
    }

    uint32_t next() {
        const uint32_t v = w_[pos_] + w_[pos_ >= 3 ? pos_ - 3 : pos_ + N - 3];
        w_[pos_] = v;
        pos_ = (pos_ + 1 == N) ? 0 : pos_ + 1;
        return v >> 1;
    }

private:
    using Poly = std::array<uint32_t, N>;

    // x^31 = x^28 + 1  (since r[i+31] = r[i+28] + r[i])
    static Poly mulX(const Poly& a) {
        Poly r{};
        const uint32_t top = a[N - 1];
        for (int i = N - 1; i > 0; --i) r[i] = a[i - 1];
        r[0] = top;
        r[28] += top;
        return r;
    }

    static Poly mul(const Poly& a, const Poly& b) {
        uint32_t c[2 * N - 1] = {};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) c[i + j] += a[i] * b[j];
        for (int d = 2 * N - 2; d >= N; --d) {
            c[d - 3] += c[d];
            c[d - N] += c[d];
        }
        Poly r;
        for (int i = 0; i < N; ++i) r[i] = c[i];
        return r;
    }

    static Poly powX(uint64_t e) {
        Poly result{};
        result[0] = 1;
        Poly b{};
        b[1] = 1;
        while (e) {
            if (e & 1) result = mul(result, b);
            e >>= 1;
            if (e) b = mul(b, b);
        }
        return result;
    }

    Poly base_;
    uint32_t w_[N];
    int pos_;
};

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values, taken from the shared (seed 1)
//   random stream starting at draw number 'first'
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal, const uint64_t first) {
    GlibcRand rng(1);
    rng.jumpTo(first);
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Matrix generation (distributed). Each rank owns a contiguous block of rows
// [rowBegin, rowEnd) and stores a local CSR (cols, rowDelimiters starting at 0).
// ****************************************************************************
struct LocalMatrix {
    std::vector<index_t> cols;
    std::vector<index_t> rowDelimiters;  // rows+1 entries, local offsets
    uint64_t firstNnz = 0;               // global index of first local nonzero
};

// Exact serial semantics of the original generator, restricted to a row range.
// State (nnzAssigned, fillRemaining) enters from the preceding rows.
static void simulateRows(LocalMatrix& m, index_t rowBegin, index_t rowEnd, index_t n, index_t dim,
                         index_t& nnzAssigned, bool& fillRemaining) {
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    GlibcRand rng(8675309);
    rng.jumpTo(static_cast<uint64_t>(rowBegin) * dim);
    m.cols.clear();
    m.rowDelimiters.assign(rowEnd - rowBegin + 1, 0);
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        m.rowDelimiters[i - rowBegin] = static_cast<index_t>(m.cols.size());
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rng.next()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                m.cols.push_back(j);
                nnzAssigned++;
            }
        }
    }
    m.rowDelimiters[rowEnd - rowBegin] = static_cast<index_t>(m.cols.size());
}

// Phase 1: assume no cap / forced fill (pure probabilistic hits) and count hits.
static uint64_t speculateRows(LocalMatrix& m, index_t rowBegin, index_t rowEnd, index_t n, index_t dim) {
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    GlibcRand rng(8675309);
    rng.jumpTo(static_cast<uint64_t>(rowBegin) * dim);
    m.cols.clear();
    m.rowDelimiters.assign(rowEnd - rowBegin + 1, 0);
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        m.rowDelimiters[i - rowBegin] = static_cast<index_t>(m.cols.size());
        for (index_t j = 0; j < dim; ++j) {
            const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
            if (randVal <= prob) m.cols.push_back(j);
        }
    }
    m.rowDelimiters[rowEnd - rowBegin] = static_cast<index_t>(m.cols.size());
    return m.cols.size();
}

void initRandomMatrix(LocalMatrix& m, const index_t rowBegin, const index_t rowEnd, const index_t n,
                      const index_t dim, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    auto rowsOf = [&](int q) {
        return std::pair<index_t, index_t>(static_cast<index_t>(static_cast<uint64_t>(dim) * q / size),
                                           static_cast<index_t>(static_cast<uint64_t>(dim) * (q + 1) / size));
    };

    const uint64_t h = speculateRows(m, rowBegin, rowEnd, n, dim);
    std::vector<unsigned long long> hits(size);
    unsigned long long hmine = h;
    MPI_Allgather(&hmine, 1, MPI_UNSIGNED_LONG_LONG, hits.data(), 1, MPI_UNSIGNED_LONG_LONG, comm);

    // Resolve the sequential dependency (cap at n / forced fill at the end).
    // Ranks whose range cannot be affected keep their speculative result.
    const uint64_t D = static_cast<uint64_t>(dim) * dim;
    index_t cnt = 0;
    bool fill = false;
    for (int q = 0; q < size; ++q) {
        const auto [qb, qe] = rowsOf(q);
        if (fill) {
            if (q == rank) {
                m.firstNnz = cnt;
                m.cols.clear();
                m.rowDelimiters.assign(qe - qb + 1, 0);
                for (index_t i = qb; i < qe; ++i) {
                    m.rowDelimiters[i - qb] = static_cast<index_t>(m.cols.size());
                    for (index_t j = 0; j < dim; ++j) m.cols.push_back(j);
                }
                m.rowDelimiters[qe - qb] = static_cast<index_t>(m.cols.size());
            }
            cnt += static_cast<index_t>(qe - qb) * dim;
        } else if (cnt >= n) {
            if (q == rank) {
                m.firstNnz = cnt;
                m.cols.clear();
                m.rowDelimiters.assign(qe - qb + 1, 0);
            }
        } else {
            const int64_t total = static_cast<int64_t>(cnt) + static_cast<int64_t>(hits[q]);
            const int64_t leftAtEnd = static_cast<int64_t>(D) - static_cast<int64_t>(qe) * dim + 1;
            const bool risky = total >= static_cast<int64_t>(n) || leftAtEnd <= static_cast<int64_t>(n) - total;
            if (risky) {
                unsigned long long st[2] = {cnt, fill};
                if (q == rank) {
                    m.firstNnz = cnt;
                    index_t c = cnt;
                    bool f = fill;
                    simulateRows(m, qb, qe, n, dim, c, f);
                    st[0] = c;
                    st[1] = f;
                }
                MPI_Bcast(st, 2, MPI_UNSIGNED_LONG_LONG, q, comm);
                cnt = static_cast<index_t>(st[0]);
                fill = st[1] != 0;
            } else {
                if (q == rank) m.firstNnz = cnt;
                cnt += static_cast<index_t>(hits[q]);
            }
        }
    }
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t offset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + offset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + offset, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row-block distribution
    const index_t rowBegin = static_cast<index_t>(static_cast<uint64_t>(numRows) * rank / size);
    const index_t rowEnd = static_cast<index_t>(static_cast<uint64_t>(numRows) * (rank + 1) / size);
    const index_t localRows = rowEnd - rowBegin;

    if (rank == 0) printf("Initializing data structures...\n");
    // Dense vector is replicated (it is constant across iterations: no communication in the loop)
    std::vector<double> h_vec(numRows);
    fill(h_vec.data(), numRows, maxVal, 0);

    LocalMatrix mat;
    initRandomMatrix(mat, rowBegin, rowEnd, nItems, numRows, MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(mat.cols.size());
    std::vector<double> h_val(localNnz);
    fill(h_val.data(), localNnz, maxVal, static_cast<uint64_t>(numRows) + mat.firstNnz);
    std::vector<double> h_out(localRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), mat.cols.data(), mat.rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), mat.cols.data(), mat.rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxDurationMs / 1000.0) / 1e9;
        const double avgTime = maxDurationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(size), displs(size);
        for (int q = 0; q < size; ++q) {
            const uint64_t b = static_cast<uint64_t>(numRows) * q / size;
            const uint64_t e = static_cast<uint64_t>(numRows) * (q + 1) / size;
            counts[q] = static_cast<int>(e - b);
            displs[q] = static_cast<int>(b);
        }
        std::vector<double> full(rank == 0 ? numRows : 0);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, full.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "OutputVector");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool localValid = verifyResults(h_reference.data(), h_out.data(), localRows, rowBegin);
        int ok = localValid ? 1 : 0, allOk = 0;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = allOk ? 0 : 1;
    }

    MPI_Finalize();
    return status;
}
