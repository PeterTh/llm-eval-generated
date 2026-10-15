#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
// Lock-free equivalent of rand()/srand(): same glibc TYPE_3 generator, private state.
struct Rng {
    random_data rd;
    char state[128];
    Rng() {
        memset(&rd, 0, sizeof(rd));
        initstate_r(1, state, sizeof(state), &rd);
    }
    void seed(unsigned s) { srandom_r(s, &rd); }
    int next() {
        int32_t r;
        random_r(&rd, &r);
        return r;
    }
};

void fill(Rng& rng, double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
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
void initRandomMatrix(Rng& rng, index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    rng.seed(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rng.next()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
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

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                    __LINE__);                                                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                              \
        }                                                                              \
    } while (0)

// ****************************************************************************
// CUDA CSR kernel: TPR threads cooperate on each row (TPR in {4,8,16,32}).
// ****************************************************************************
template <int TPR>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t nRows,
                           double* __restrict__ out) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = gid / TPR;
    const int lane = threadIdx.x % TPR;
    const bool valid = row < nRows;
    index_t b = 0, e = 0;
    if (valid) {
        b = rowDelimiters[row];
        e = rowDelimiters[row + 1];
    }
    double t = 0.0;
    for (index_t j = b + lane; j < e; j += TPR) {
        t += __ldg(val + j) * __ldg(vec + __ldg(cols + j));
    }
#pragma unroll
    for (int o = TPR / 2; o > 0; o >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, o, TPR);
    }
    if (valid && lane == 0) out[row] = t;
}

static void launchSpmv(int tpr, const double* val, const index_t* cols, const index_t* rd,
                       const double* vec, index_t nRows, double* out) {
    const int block = 256;
    const size_t threads = static_cast<size_t>(nRows) * tpr;
    const unsigned grid = static_cast<unsigned>((threads + block - 1) / block);
    if (grid == 0) return;
    switch (tpr) {
        case 4:  spmvKernel<4><<<grid, block>>>(val, cols, rd, vec, nRows, out); break;
        case 8:  spmvKernel<8><<<grid, block>>>(val, cols, rd, vec, nRows, out); break;
        case 16: spmvKernel<16><<<grid, block>>>(val, cols, rd, vec, nRows, out); break;
        default: spmvKernel<32><<<grid, block>>>(val, cols, rd, vec, nRows, out); break;
    }
}

// Chunked point-to-point transfer (keeps MPI counts within int range)
static constexpr size_t CHUNK = size_t(1) << 27;
static void sendBig(const void* p, size_t bytes, int dst) {
    const char* c = static_cast<const char*>(p);
    for (size_t o = 0; o < bytes; o += CHUNK)
        MPI_Send(c + o, static_cast<int>(std::min(CHUNK, bytes - o)), MPI_BYTE, dst, 7,
                 MPI_COMM_WORLD);
}
static void recvBig(void* p, size_t bytes, int src) {
    char* c = static_cast<char*>(p);
    for (size_t o = 0; o < bytes; o += CHUNK)
        MPI_Recv(c + o, static_cast<int>(std::min(CHUNK, bytes - o)), MPI_BYTE, src, 7,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // One GPU per rank, assigned round-robin among ranks on the same node
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank, ndev;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
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
        printf("Initializing data structures...\n");
    }

    // Rank 0 generates the (inherently sequential) random matrix exactly as the original
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        Rng rng;
        fill(rng, h_vec.data(), numRows, maxVal);
        fill(rng, h_val.data(), nItems, maxVal);
        initRandomMatrix(rng, h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
    }

    // Row partition balanced by nonzeros
    std::vector<index_t> rowStart(nprocs + 1, 0);
    if (rank == 0) {
        for (int r = 1; r < nprocs; ++r) {
            const uint64_t target = static_cast<uint64_t>(nItems) * r / nprocs;
            rowStart[r] = static_cast<index_t>(
                std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(),
                                 static_cast<index_t>(target)) - h_rowDelimiters.begin());
            rowStart[r] = std::min(rowStart[r], numRows);
            rowStart[r] = std::max(rowStart[r], rowStart[r - 1]);
        }
        rowStart[nprocs] = numRows;
    }
    MPI_Bcast(rowStart.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t myRow0 = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myRow0;
    std::vector<index_t> l_rd(myRows + 1);
    std::vector<double> l_val;
    std::vector<index_t> l_cols;
    index_t myNnz = 0;

    if (rank == 0) {
        for (int r = nprocs - 1; r >= 0; --r) {
            const index_t r0 = rowStart[r], r1 = rowStart[r + 1];
            const index_t base = h_rowDelimiters[r0];
            const size_t nnz = h_rowDelimiters[r1] - base;
            std::vector<index_t> rd(r1 - r0 + 1);
#pragma omp parallel for schedule(static)
            for (index_t i = 0; i <= r1 - r0; ++i) rd[i] = h_rowDelimiters[r0 + i] - base;
            if (r == 0) {
                l_rd = rd;
                myNnz = static_cast<index_t>(nnz);
                // rank 0's slice starts at nonzero 0: reuse buffers in place
            } else {
                sendBig(rd.data(), rd.size() * sizeof(index_t), r);
                sendBig(h_cols.data() + base, nnz * sizeof(index_t), r);
                sendBig(h_val.data() + base, nnz * sizeof(double), r);
            }
        }
    } else {
        myNnz = 0;
        l_rd.resize(myRows + 1);
        recvBig(l_rd.data(), l_rd.size() * sizeof(index_t), 0);
        myNnz = l_rd[myRows];
        l_cols.resize(myNnz);
        l_val.resize(myNnz);
        recvBig(l_cols.data(), myNnz * sizeof(index_t), 0);
        recvBig(l_val.data(), myNnz * sizeof(double), 0);
    }
    const index_t* src_cols = rank == 0 ? h_cols.data() : l_cols.data();
    const double* src_val = rank == 0 ? h_val.data() : l_val.data();

    // Device buffers
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * (myNnz + 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * (myNnz + 1)));
    CUDA_CHECK(cudaMalloc(&d_rd, sizeof(index_t) * (myRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * (numRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * (myRows + 1)));
    CUDA_CHECK(cudaMemcpy(d_val, src_val, sizeof(double) * myNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, src_cols, sizeof(index_t) * myNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rd, l_rd.data(), sizeof(index_t) * (myRows + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double) * (myRows + 1)));

    const double avg = myRows ? static_cast<double>(myNnz) / myRows : 0.0;
    const int tpr = avg <= 6 ? 4 : avg <= 12 ? 8 : avg <= 24 ? 16 : 32;

    // Warm-up-free timed region: kernels only (data resident on device)
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(tpr, d_val, d_cols, d_rd, d_vec, myRows, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    const long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration = 0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(max_duration);

    // Gather output on rank 0
    std::vector<double> l_out(myRows);
    CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, sizeof(double) * myRows, cudaMemcpyDeviceToHost));
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        counts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        displs[r] = static_cast<int>(rowStart[r]);
    }
    if (rank == 0) h_out.resize(numRows);
    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE, h_out.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rd);
    cudaFree(d_vec);
    cudaFree(d_out);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }
    MPI_Finalize();
    return status;
}
