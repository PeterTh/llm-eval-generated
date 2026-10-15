#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Draw n random values exactly as the serial code did (sequential rand() stream);
// store only those in [keepBegin, keepEnd) into A (A[0] corresponds to keepBegin).
static void fillRange(double* A, const index_t n, const double maxVal, const index_t keepBegin,
                      const index_t keepEnd) {
    for (index_t i = 0; i < n; ++i) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= keepBegin && i < keepEnd) A[i - keepBegin] = v;
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Reproduces the serial random CSR structure. The rand() stream is inherently
//   sequential, so every rank walks it, but only stores the rows [rowBegin,rowEnd)
//   it owns. Column indices are stored in cols (local), rowDelimiters holds
//   offsets relative to the rank's first nonzero. Returns the global index of
//   the rank's first nonzero via nnzBefore.
// ****************************************************************************
static void initRandomMatrix(std::vector<index_t>& cols, std::vector<index_t>& rowDelimiters,
                             const index_t n, const index_t dim, const index_t rowBegin,
                             const index_t rowEnd, index_t& nnzBefore) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);

    nnzBefore = 0;
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        const bool mine = (i >= rowBegin && i < rowEnd);
        if (i == rowBegin) nnzBefore = nnzAssigned;
        if (mine) rowDelimiters[i - rowBegin] = nnzAssigned - nnzBefore;
        if (i >= rowEnd) break;  // nothing further needed locally
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                if (mine) cols.push_back(j);
                nnzAssigned++;
            }
        }
    }
    if (rowBegin >= rowEnd) nnzBefore = 0;
    rowDelimiters[rowEnd - rowBegin] = static_cast<index_t>(cols.size());
}

// CPU reference (OpenMP) for the rank's rows
static void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
                    const double* vec, const index_t nRows, double* out) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(nRows); ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// GPU CSR kernel: TPR (power of two <= 32) threads cooperate on each row.
template <int TPR>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelim, const double* __restrict__ vec,
                           const index_t nRows, double* __restrict__ out) {
    const size_t gt = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = gt / TPR;
    const int lane = threadIdx.x % TPR;
    double t = 0.0;
    const bool active = row < nRows;
    if (active) {
        const index_t end = rowDelim[row + 1];
        for (index_t j = rowDelim[row] + lane; j < end; j += TPR) {
            t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
        }
    }
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off, TPR);
    if (active && lane == 0) out[row] = t;
}

template <int TPR>
static void launch(const double* val, const index_t* cols, const index_t* rd, const double* vec,
                   index_t nRows, double* out) {
    const int block = 256;
    const size_t grid = (static_cast<size_t>(nRows) * TPR + block - 1) / block;
    if (grid == 0) return;
    spmvKernel<TPR><<<static_cast<unsigned>(grid), block>>>(val, cols, rd, vec, nRows, out);
}

static void launchSpmv(int tpr, const double* val, const index_t* cols, const index_t* rd,
                       const double* vec, index_t nRows, double* out) {
    switch (tpr) {
        case 1:  launch<1>(val, cols, rd, vec, nRows, out); break;
        case 2:  launch<2>(val, cols, rd, vec, nRows, out); break;
        case 4:  launch<4>(val, cols, rd, vec, nRows, out); break;
        case 8:  launch<8>(val, cols, rd, vec, nRows, out); break;
        case 16: launch<16>(val, cols, rd, vec, nRows, out); break;
        default: launch<32>(val, cols, rd, vec, nRows, out); break;
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
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t offset) {
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One GPU per rank; ranks on a node are spread round-robin over its devices
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

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
        printf("Parallelization: %d MPI ranks x %d OpenMP threads, CUDA\n", nranks,
               omp_get_max_threads());
        printf("Initializing data structures...\n");
        fflush(stdout);
    }

    // 1D row-block distribution
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const uint64_t b = static_cast<uint64_t>(numRows) * r / nranks;
        const uint64_t e = static_cast<uint64_t>(numRows) * (r + 1) / nranks;
        displs[r] = static_cast<int>(b);
        counts[r] = static_cast<int>(e - b);
    }
    const index_t rowBegin = displs[rank];
    const index_t nLoc = counts[rank];
    const index_t rowEnd = rowBegin + nLoc;

    // Matrix structure (local rows only)
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(nLoc + 1);
    index_t nnzBefore = 0;
    initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows, rowBegin, rowEnd, nnzBefore);
    const index_t nnzLoc = static_cast<index_t>(h_cols.size());

    // Values: replay the default-seeded rand() stream (vector first, then values)
    std::vector<double> h_vec(numRows);
    std::vector<double> h_val(nnzLoc);
    srand(1);
    fillRange(h_vec.data(), numRows, maxVal, 0, numRows);
    fillRange(h_val.data(), nItems, maxVal, nnzBefore, nnzBefore + nnzLoc);
    std::vector<double> h_out(nLoc);

    // Reference solution for local rows
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) { printf("Computing reference solution...\n"); fflush(stdout); }
        h_reference.resize(nLoc);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), nLoc,
                h_reference.data());
    }

    // Device data
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rd;
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * (nnzLoc + 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * (nnzLoc + 1)));
    CUDA_CHECK(cudaMalloc(&d_rd, sizeof(index_t) * (nLoc + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * (numRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * (nLoc + 1)));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * nnzLoc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * nnzLoc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rd, h_rowDelimiters.data(), sizeof(index_t) * (nLoc + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    // Threads per row: smallest power of two >= average row length (capped at warp size)
    const double avg = nLoc ? static_cast<double>(nnzLoc) / nLoc : 1.0;
    int tpr = 1;
    while (tpr < 32 && tpr < avg) tpr <<= 1;

    // Warm-up to exclude one-time launch overhead from the timed region
    launchSpmv(tpr, d_val, d_cols, d_rd, d_vec, nLoc, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) { printf("Computing SpMV...\n"); fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(tpr, d_val, d_cols, d_rd, d_vec, nLoc, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * nLoc, cudaMemcpyDeviceToHost));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> h_all;
        if (rank == 0) h_all.resize(numRows);
        MPI_Gatherv(h_out.data(), static_cast<int>(nLoc), MPI_DOUBLE, h_all.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(h_all, "OutputVector");
    }

    int status = 0;
    if (validate) {
        if (rank == 0) { printf("Validating result...\n"); fflush(stdout); }
        int ok = verifyResults(h_reference.data(), h_out.data(), nLoc, rowBegin) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = allOk ? 0 : 1;
    }

    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rd); cudaFree(d_vec); cudaFree(d_out);
    MPI_Finalize();
    return status;
}
