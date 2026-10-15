#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Class: GlibcRand
//
// Purpose:
//   Reproduces glibc's rand() sequence (TYPE_3 additive feedback generator)
//   for a given seed, with O(log k) jump-ahead so that every MPI rank can
//   start at an arbitrary position of the serial random stream. Outputs are
//   bit-identical to srand(seed); rand() x k followed by rand().
//
// ****************************************************************************
class GlibcRand {
    static constexpr int R = 31;
    uint32_t buf[R];
    int pos = 0;

    // Polynomial arithmetic modulo x^31 = x^28 + 1 with coefficients mod 2^32
    static void mulmod(const uint32_t* a, const uint32_t* b, uint32_t* out) {
        uint32_t t[2 * R - 1] = {0};
        for (int i = 0; i < R; ++i) {
            if (a[i] == 0) continue;
            for (int j = 0; j < R; ++j) t[i + j] += a[i] * b[j];
        }
        for (int d = 2 * R - 2; d >= R; --d) {
            t[d - 3] += t[d];
            t[d - R] += t[d];
        }
        for (int i = 0; i < R; ++i) out[i] = t[i];
    }

public:
    GlibcRand(uint32_t seed, uint64_t k) {
        if (seed == 0) seed = 1;
        int32_t r[34];
        r[0] = static_cast<int32_t>(seed);
        for (int i = 1; i < R; ++i) {
            int64_t hi = r[i - 1] / 127773, lo = r[i - 1] % 127773;
            int64_t w = 16807 * lo - 2836 * hi;
            if (w < 0) w += 2147483647;
            r[i] = static_cast<int32_t>(w);
        }
        r[31] = r[0]; r[32] = r[1]; r[33] = r[2];
        uint32_t a[2 * R];
        for (int j = 0; j < R; ++j) a[j] = static_cast<uint32_t>(r[3 + j]);
        for (int j = R; j < 2 * R; ++j) a[j] = a[j - 3] + a[j - 31];

        // out_k = a_{k+341} >> 1; compute x^m mod poly
        const uint64_t m = k + 341;
        uint32_t c[R] = {0}, tmp[R];
        c[0] = 1;
        for (int bit = 63; bit >= 0; --bit) {
            mulmod(c, c, tmp);
            memcpy(c, tmp, sizeof(c));
            if ((m >> bit) & 1) {  // multiply by x
                const uint32_t top = c[R - 1];
                for (int i = R - 1; i > 0; --i) c[i] = c[i - 1];
                c[0] = top;
                c[28] += top;
            }
        }
        for (int q = 0; q < R; ++q) {
            uint32_t s = 0;
            for (int t = 0; t < R; ++t) s += c[t] * a[t + q];
            buf[q] = s;
        }
    }

    uint32_t next() {
        const uint32_t v = buf[pos];
        int p28 = pos + 28;
        if (p28 >= R) p28 -= R;
        buf[pos] = v + buf[p28];
        if (++pos == R) pos = 0;
        return v >> 1;
    }
};

constexpr uint32_t VECTOR_SEED = 1;  // default seed of rand() before any srand()
constexpr uint32_t MATRIX_SEED = 8675309;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values; element i takes the (streamStart+i)-th
//   value of the srand(seed) rand() stream (same values as the serial code).
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal, uint32_t seed, uint64_t streamStart) {
    GlibcRand rng(seed, streamStart);
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: genRows
//
// Purpose:
//   Generates the CSR column indices of rows [r0, r1) exactly as the serial
//   initRandomMatrix would, given nnzStart = number of nonzeroes assigned in
//   rows before r0. rowDel receives (r1-r0+1) offsets local to this range.
//   Returns the global nonzero count after row r1-1.
//   If speculative, the n-limit / fillRemaining logic is skipped (exact
//   whenever that logic cannot trigger within the range).
//
// ****************************************************************************
index_t genRows(std::vector<index_t>& cols, std::vector<index_t>& rowDel, index_t r0, index_t r1,
                index_t nnzStart, const index_t n, const index_t dim, bool speculative) {
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    GlibcRand rng(MATRIX_SEED, static_cast<uint64_t>(r0) * dim);
    cols.clear();
    rowDel.assign(r1 - r0 + 1, 0);
    index_t nnzAssigned = nnzStart;
    bool fillRemaining = false;
    for (index_t i = r0; i < r1; ++i) {
        rowDel[i - r0] = nnzAssigned - nnzStart;
        for (index_t j = 0; j < dim; ++j) {
            const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
            if (speculative) {
                if (randVal <= prob) {
                    cols.push_back(j);
                    nnzAssigned++;
                }
                continue;
            }
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols.push_back(j);
                nnzAssigned++;
            }
        }
    }
    rowDel[r1 - r0] = nnzAssigned - nnzStart;
    return nnzAssigned;
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
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t offset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", offset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       offset + i, ref, res, relError);
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
        printf("Initializing data structures...\n");
        fflush(stdout);
    }

    // Block row distribution
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs + 1, 0);
    for (int r = 0; r < nprocs; ++r) {
        rowCounts[r] = static_cast<int>(numRows / nprocs + (static_cast<index_t>(r) < numRows % nprocs ? 1 : 0));
        rowDispls[r + 1] = rowDispls[r] + rowCounts[r];
    }
    const index_t r0 = rowDispls[rank], r1 = rowDispls[rank + 1];
    const index_t localRows = r1 - r0;

    // Dense vector: each rank generates its slice, then all-gather
    std::vector<double> h_vec(numRows);
    fill(h_vec.data() + r0, localRows, maxVal, VECTOR_SEED, r0);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_vec.data(), rowCounts.data(), rowDispls.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Matrix structure: speculative generation of own rows, prefix over counts
    std::vector<index_t> h_cols, h_rowDelimiters;
    index_t specCount = genRows(h_cols, h_rowDelimiters, r0, r1, 0, nItems, numRows, true);
    std::vector<unsigned long long> counts(nprocs);
    {
        unsigned long long c = specCount;
        MPI_Allgather(&c, 1, MPI_UNSIGNED_LONG_LONG, counts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    }
    std::vector<unsigned long long> startNnz(nprocs + 1, 0);
    for (int r = 0; r < nprocs; ++r) startNnz[r + 1] = startNnz[r] + counts[r];

    // The nnz-limit / fillRemaining logic may trigger in a rank's range only if
    // (conservatively) the following holds; the check is monotone over the range.
    const bool wraps = static_cast<unsigned long long>(numRows) * numRows > 0xFFFFFFFFull;
    int firstTrig = nprocs;
    {
        int trig = 0;
        if (localRows > 0) {
            const long long S = static_cast<long long>(startNnz[rank + 1]);
            const long long leftLast = static_cast<long long>(numRows) * numRows -
                                       (static_cast<long long>(r1 - 1) * numRows + (numRows - 1));
            trig = wraps || S + 1 >= static_cast<long long>(nItems) ||
                   leftLast <= static_cast<long long>(nItems) - S + 1;
        }
        std::vector<int> trigs(nprocs);
        MPI_Allgather(&trig, 1, MPI_INT, trigs.data(), 1, MPI_INT, MPI_COMM_WORLD);
        for (int r = 0; r < nprocs; ++r) {
            if (trigs[r]) { firstTrig = r; break; }
        }
    }
    // Ranks from the first trigger on regenerate with exact state, in sequence
    if (firstTrig < nprocs) {
        unsigned long long cur = startNnz[firstTrig];
        for (int r = firstTrig; r < nprocs; ++r) {
            startNnz[r] = cur;
            unsigned long long end = 0;
            if (r == rank) {
                end = genRows(h_cols, h_rowDelimiters, r0, r1, static_cast<index_t>(cur), nItems, numRows, false);
            }
            MPI_Bcast(&end, 1, MPI_UNSIGNED_LONG_LONG, r, MPI_COMM_WORLD);
            cur = end;
            startNnz[r + 1] = cur;
        }
    }
    const index_t localNnz = static_cast<index_t>(h_cols.size());

    // Matrix values: stream position follows the dense vector's numRows draws
    std::vector<double> h_val(localNnz);
    fill(h_val.data(), localNnz, maxVal, VECTOR_SEED, static_cast<uint64_t>(numRows) + startNnz[rank]);
    std::vector<double> h_out(localRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (root) { printf("Computing reference solution...\n"); fflush(stdout); }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (root) { printf("Computing SpMV...\n"); fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long ms = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %lld ms\n", ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (ms / 1000.0) / 1e9;
        const double avgTime = ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        if (root) full.resize(numRows);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, full.data(), rowCounts.data(),
                    rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) print_results(full, "OutputVector");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (root) { printf("Validating result...\n"); fflush(stdout); }
        int valid = verifyResults(h_reference.data(), h_out.data(), localRows, r0) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = valid ? 0 : 1;
    }

    MPI_Finalize();
    return status;
}
