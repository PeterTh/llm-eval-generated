#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
// Fraction of each rank's rows computed by OpenMP host threads, concurrently with its GPU
constexpr double DEFAULT_CPU_ROW_SHARE = 0.01;
constexpr int WARP_SIZE = 32;
constexpr unsigned int THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

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
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

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
            double randVal = static_cast<double>(rand()) / RAND_MAX;
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
//   Computes sparse matrix-vector multiplication using CSR format.
//   Rows are distributed across OpenMP threads; the whole repeated
//   computation runs in one parallel region to avoid per-repeat fork/join.
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//   repeats: number of times the multiplication is performed
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out, const index_t repeats = 1) {
    #pragma omp parallel
    for (index_t rep = 0; rep < repeats; ++rep) {
        #pragma omp for schedule(static)
        for (index_t i = 0; i < dim; ++i) {
            double t = 0.0;
            for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
                const auto col = cols[j];
                t += val[j] * vec[col];
            }
            out[i] = t;
        }
    }
}

// ****************************************************************************
// Kernel: spmvGpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on the GPU.
//   One warp is assigned to each row; its lanes stride over the row's
//   non-zeros and the partial sums are combined with a warp reduction.
//
// Arguments:
//   val, cols, rowDelimiters, vec: same meaning as in spmvCpu (device memory)
//   rows: number of rows this launch computes, starting at row 0
//   out: output - result from the spmv calculation (device memory)
//
// ****************************************************************************
__global__ void spmvGpu(const double* __restrict__ val, const index_t* __restrict__ cols,
                        const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                        const index_t rows, double* __restrict__ out) {
    const index_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t lane = threadIdx.x % WARP_SIZE;
    const index_t warpsTotal = (gridDim.x * blockDim.x) / WARP_SIZE;

    for (index_t row = globalThread / WARP_SIZE; row < rows; row += warpsTotal) {
        double t = 0.0;
        for (index_t j = rowDelimiters[row] + lane; j < rowDelimiters[row + 1]; j += WARP_SIZE) {
            t += val[j] * vec[cols[j]];
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            t += __shfl_down_sync(0xffffffff, t, offset);
        }
        if (lane == 0) {
            out[row] = t;
        }
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
    printf("  -c <frac>    Fraction of rows per rank computed on OpenMP host threads\n");
    printf("               concurrently with the GPU, in [0, 1) (default: %.2f)\n", DEFAULT_CPU_ROW_SHARE);
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Rows [rowBegin(r), rowBegin(r+1)) are owned by MPI rank r
index_t rowBegin(const int rank, const int size, const index_t dim) {
    return static_cast<index_t>((static_cast<uint64_t>(dim) * static_cast<uint64_t>(rank)) /
                                static_cast<uint64_t>(size));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool isRoot = rank == 0;

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    double cpuShare = DEFAULT_CPU_ROW_SHARE;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses the same arguments)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            cpuShare = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (!(cpuShare >= 0.0 && cpuShare < 1.0)) {
        if (isRoot) {
            fprintf(stderr, "Invalid -c value %g: must be in [0, 1)\n", cpuShare);
        }
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank: ranks on a node are numbered 0..localSize-1
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    // Split host cores between the ranks sharing a node unless the user chose a count
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // The full problem is generated on the root; other ranks receive their row blocks
    std::vector<double> h_val(isRoot ? nItems : 0);                  // Non-zero values
    std::vector<index_t> h_cols(isRoot ? nItems : 0);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);               // Row delimiters
    std::vector<double> h_vec(numRows);                              // Dense vector
    std::vector<double> h_out(isRoot ? numRows : 0);                 // Output vector

    if (isRoot) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Every rank needs the row delimiters and the full dense vector
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Row block partition: per-rank nonzero counts/offsets for scatter, row counts/offsets for gather
    std::vector<int> nnzCounts(size), nnzDispls(size), rowCounts(size), rowDispls(size);
    for (int r = 0; r < size; ++r) {
        const index_t lo = rowBegin(r, size, numRows);
        const index_t hi = rowBegin(r + 1, size, numRows);
        const index_t nnzLo = h_rowDelimiters[lo];
        const index_t nnzHi = h_rowDelimiters[hi];
        if (nnzHi - nnzLo > static_cast<index_t>(INT_MAX)) {
            if (isRoot) {
                fprintf(stderr, "Rank %d would receive more than INT_MAX non-zeros\n", r);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        nnzDispls[r] = static_cast<int>(nnzLo);
        nnzCounts[r] = static_cast<int>(nnzHi - nnzLo);
        rowDispls[r] = static_cast<int>(lo);
        rowCounts[r] = static_cast<int>(hi - lo);
    }

    const index_t rowLo = rowBegin(rank, size, numRows);
    const index_t localRows = rowBegin(rank + 1, size, numRows) - rowLo;
    const index_t localNnz = static_cast<index_t>(nnzCounts[rank]);

    // Local CSR block: values/columns of this rank's nonzeros, row delimiters rebased to 0
    std::vector<double> l_val(localNnz);
    std::vector<index_t> l_cols(localNnz);
    MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 l_val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 l_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<index_t> l_rowDelimiters(localRows + 1);
    for (index_t i = 0; i <= localRows; ++i) {
        l_rowDelimiters[i] = h_rowDelimiters[rowLo + i] - h_rowDelimiters[rowLo];
    }

    // Rows [0, gpuRows) go to the GPU; rows [gpuRows, localRows) go to OpenMP host threads
    const index_t cpuRows = std::min(localRows, static_cast<index_t>(std::lround(localRows * cpuShare)));
    const index_t gpuRows = localRows - cpuRows;

    // Device copies (sizes clamped to 1 so zero-row ranks still allocate valid buffers)
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(localNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(localNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(gpuRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, l_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, l_rowDelimiters.data(), (localRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<double> l_out(localRows);

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // GPU: enqueue every iteration asynchronously, one warp per row
    const size_t threadsNeeded = std::max<size_t>(gpuRows, 1) * WARP_SIZE;
    const unsigned int blocks = static_cast<unsigned int>((threadsNeeded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_val, d_cols, d_rowDelimiters, d_vec, gpuRows, d_out);
    }
    CUDA_CHECK(cudaGetLastError());

    // Host: OpenMP threads compute the remaining rows while the GPU works
    spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data() + gpuRows,
            h_vec.data(), cpuRows, l_out.data() + gpuRows, iterations);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Parallel time is that of the slowest rank
    long long maxMs = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Gather the GPU rows back to the host, then assemble the full output on the root
    CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, gpuRows * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(l_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                h_out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int rc = 0;
    if (isRoot) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    // All ranks exit with the root's status
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
