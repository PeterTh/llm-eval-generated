#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& cols, index_t* rowDelimiters, const index_t n, const index_t dim,
                      const index_t rowBegin, const index_t rowEnd, index_t& nnzBegin,
                      index_t& nnzEnd) {
    // Every rank replays the (inherently sequential) random stream so that the matrix is
    // identical to the serial one, but only stores rows [rowBegin, rowEnd) and stops after them.
    // rowDelimiters has rowEnd-rowBegin+1 entries, relative to nnzBegin. Requires rowBegin < rowEnd.
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);

    nnzBegin = 0;
    bool fillRemaining = false;
    for (index_t i = 0; i < rowEnd; ++i) {
        if (i == rowBegin) nnzBegin = nnzAssigned;
        const bool keep = i >= rowBegin;
        if (keep) rowDelimiters[i - rowBegin] = nnzAssigned - nnzBegin;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                if (keep) cols.push_back(j);
                nnzAssigned++;
            }
        }
    }
    nnzEnd = nnzAssigned;
    rowDelimiters[rowEnd - rowBegin] = nnzEnd - nnzBegin;
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
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ****************************************************************************
// Kernel: spmvKernel
//
// Purpose:
//   CSR SpMV on the GPU; TPR cooperating threads (a power of two <= 32) handle
//   one row, so that column/value loads are coalesced. Partial sums are
//   combined with warp shuffles.
//
// ****************************************************************************
template <int TPR>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t rows,
                           double* __restrict__ out) {
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = gtid / TPR;
    const int lane = threadIdx.x % TPR;
    double t = 0.0;
    if (row < rows) {
        const index_t b = rowDelimiters[row], e = rowDelimiters[row + 1];
        for (index_t j = b + lane; j < e; j += TPR) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
    }
    // TPR-aligned groups never straddle a warp, so the full-mask shuffle is safe
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off, TPR);
    }
    if (row < rows && lane == 0) out[row] = t;
}

template <int TPR>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rd,
                       const double* vec, index_t rows, double* out, cudaStream_t s) {
    constexpr int BLOCK = 256;
    const size_t threads = static_cast<size_t>(rows) * TPR;
    const size_t blocks = (threads + BLOCK - 1) / BLOCK;
    if (blocks == 0) return;
    spmvKernel<TPR><<<static_cast<unsigned>(blocks), BLOCK, 0, s>>>(val, cols, rd, vec, rows, out);
}

static void launchSpmvAuto(int tpr, const double* val, const index_t* cols, const index_t* rd,
                           const double* vec, index_t rows, double* out, cudaStream_t s) {
    switch (tpr) {
        case 1:  launchSpmv<1>(val, cols, rd, vec, rows, out, s); break;
        case 2:  launchSpmv<2>(val, cols, rd, vec, rows, out, s); break;
        case 4:  launchSpmv<4>(val, cols, rd, vec, rows, out, s); break;
        case 8:  launchSpmv<8>(val, cols, rd, vec, rows, out, s); break;
        case 16: launchSpmv<16>(val, cols, rd, vec, rows, out, s); break;
        default: launchSpmv<32>(val, cols, rd, vec, rows, out, s); break;
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
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // Select a GPU for this rank (round-robin over the GPUs of the node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0, nDev = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&nDev));
        if (nDev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % nDev));
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
        printf("Initializing data structures...\n");
        fflush(stdout);
    }

    // 1D row-block distribution
    auto rowStart = [&](int r) {
        return static_cast<index_t>(static_cast<uint64_t>(numRows) * r / nprocs);
    };
    const index_t r0 = rowStart(rank), r1 = rowStart(rank + 1);
    const index_t myRows = r1 - r0;

    std::vector<index_t> h_cols;                          // local column indices
    std::vector<index_t> h_rowDelimiters(myRows + 1, 0);  // local row delimiters (rebased)
    std::vector<double> h_val;                            // local non-zero values
    std::vector<double> h_vec(numRows);                   // full dense vector
    std::vector<double> h_out(myRows);                    // local output slice
    index_t nnzBegin = 0, nnzEnd = 0;

    // The matrix generator reseeds, so it is independent of the fill() stream order.
    if (myRows > 0) {
        h_cols.reserve(static_cast<size_t>(nItems / nprocs) + nItems / (4 * nprocs) + 1024);
        initRandomMatrix(h_cols, h_rowDelimiters.data(), nItems, numRows, r0, r1, nnzBegin, nnzEnd);
    }
    const index_t myNnz = nnzEnd - nnzBegin;

    // Replay the fill() stream: vector first, then values; keep only the local slice of values
    srand(1);
    fill(h_vec.data(), numRows, maxVal);
    h_val.resize(myNnz);
    for (index_t g = 0; g < nnzEnd; ++g) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (g >= nnzBegin) h_val[g - nnzBegin] = v;
    }

    // For validation, compute reference solution (local slice, OpenMP)
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) { printf("Computing reference solution...\n"); fflush(stdout); }
        h_reference.resize(myRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), myRows, h_reference.data());
    }

    // Device buffers
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * (myNnz + 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * (myNnz + 1)));
    CUDA_CHECK(cudaMalloc(&d_rd, sizeof(index_t) * (myRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * (numRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * (myRows + 1)));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * myNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * myNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rd, h_rowDelimiters.data(), sizeof(index_t) * (myRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    // Threads per row chosen from the average row length
    const double avgRow = myRows ? static_cast<double>(myNnz) / myRows : 0.0;
    const int tpr = avgRow < 2 ? 1 : avgRow < 4 ? 2 : avgRow < 8 ? 4 : avgRow < 16 ? 8
                  : avgRow < 32 ? 16 : 32;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (rank == 0) { printf("Computing SpMV...\n"); fflush(stdout); }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmvAuto(tpr, d_val, d_cols, d_rd, d_vec, myRows, d_out, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double local_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ms = 0.0;
    MPI_Reduce(&local_ms, &ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * myRows, cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(ms));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (ms / 1000.0) / 1e9;
        const double avgTime = ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gather the distributed output on rank 0)
    if (printResults) {
        std::vector<double> h_full;
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            displs[r] = static_cast<int>(rowStart(r));
            counts[r] = static_cast<int>(rowStart(r + 1) - rowStart(r));
        }
        if (rank == 0) h_full.resize(numRows);
        MPI_Gatherv(h_out.data(), static_cast<int>(myRows), MPI_DOUBLE, h_full.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(h_full, "OutputVector");
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) { printf("Validating result...\n"); fflush(stdout); }
        const bool valid = verifyResults(h_reference.data(), h_out.data(), myRows);
        int ok = valid ? 1 : 0, allOk = 1;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        exitCode = allOk ? 0 : 1;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rd);
    cudaFree(d_vec);
    cudaFree(d_out);
    MPI_Finalize();
    return exitCode;
}
