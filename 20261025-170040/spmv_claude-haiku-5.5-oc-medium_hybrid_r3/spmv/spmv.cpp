#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
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
constexpr int THREADS_PER_BLOCK = 256;
constexpr uint64_t MPI_COUNT_LIMIT = INT_MAX;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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
//   Rows are independent, so they are distributed across OpenMP threads.
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
// Kernel: spmvCsrKernel
//
// Purpose:
//   CSR sparse matrix-vector multiplication on a single GPU. One thread
//   handles one row and accumulates its nonzeros in the same left-to-right
//   order as spmvCpu, so the results are identical to the CPU reference.
//
// Arguments:
//   val: non-zero values local to this device
//   cols: column indices for each non-zero
//   rowDelimiters: local row pointers (rebased to start at 0), size rows+1
//   vec: full dense input vector
//   rows: number of rows owned by this device
//   out: output for the rows owned by this device
//
// ****************************************************************************
__global__ void spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec, const index_t rows,
                              double* __restrict__ out) {
    const index_t stride = blockDim.x * gridDim.x;
    for (index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += stride) {
        double t = 0.0;
        const index_t end = rowDelimiters[i + 1];
        for (index_t j = rowDelimiters[i]; j < end; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Structure: DeviceSlice
//
// Purpose:
//   Holds the portion of a rank's row block that is assigned to one GPU,
//   together with the device buffers that hold it.
//
// ****************************************************************************
struct DeviceSlice {
    int device = 0;
    index_t rowBegin = 0;  // first local row owned by this device
    index_t rows = 0;      // number of local rows owned by this device
    index_t nnz = 0;       // number of non-zeros owned by this device
    double* val = nullptr;
    index_t* cols = nullptr;
    index_t* rowDelimiters = nullptr;
    double* vec = nullptr;
    double* out = nullptr;
};

// ****************************************************************************
// Function: launchSpmv
//
// Purpose:
//   Enqueues one SpMV pass for the rows owned by a device. Must be called
//   with the slice's device as the current device.
//
// ****************************************************************************
void launchSpmv(const DeviceSlice& s) {
    if (s.rows == 0) {
        return;
    }
    const index_t blocks = (s.rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    spmvCsrKernel<<<blocks, THREADS_PER_BLOCK>>>(s.val, s.cols, s.rowDelimiters, s.vec, s.rows,
                                                 s.out);
    CUDA_CHECK(cudaGetLastError());
}

// ****************************************************************************
// Function: partitionRows
//
// Purpose:
//   Splits rows [0, dim) into contiguous blocks holding approximately equal
//   numbers of non-zeros.
//
// Arguments:
//   rowDelimiters: CSR row pointers of size dim+1
//   dim: number of rows
//   parts: number of blocks
//
// Returns:
//   Boundaries of size parts+1; block p is rows [bounds[p], bounds[p+1]).
//
// ****************************************************************************
std::vector<index_t> partitionRows(const index_t* rowDelimiters, const index_t dim, const int parts) {
    const uint64_t nnz = rowDelimiters[dim];
    std::vector<index_t> bounds(parts + 1);
    bounds[0] = 0;
    bounds[parts] = dim;
    for (int p = 1; p < parts; ++p) {
        const index_t target = static_cast<index_t>(nnz * static_cast<uint64_t>(p) / parts);
        const index_t* it = std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target);
        const index_t row = static_cast<index_t>(it - rowDelimiters);
        bounds[p] = std::max(row, bounds[p - 1]);
    }
    return bounds;
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
    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    const bool isRoot = (rank == 0);

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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // MPI counts are 32-bit integers
    if (static_cast<uint64_t>(nItems) > MPI_COUNT_LIMIT ||
        static_cast<uint64_t>(numRows) >= MPI_COUNT_LIMIT) {
        if (isRoot) {
            fprintf(stderr, "Problem too large for 32-bit MPI counts (nnz=%u, rows=%u)\n", nItems,
                    numRows);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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

    // Root generates the problem exactly as the serial code does; the
    // matrix and vector are then distributed to the other ranks.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector (root only)

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.assign(numRows, 0.0);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Row-block partition across MPI ranks, balanced by non-zeros
    const std::vector<index_t> rowBounds = partitionRows(h_rowDelimiters.data(), numRows, numProcs);

    std::vector<int> rowCounts(numProcs), rowDispls(numProcs);
    std::vector<int> nnzCounts(numProcs), nnzDispls(numProcs);
    std::vector<int> ptrCounts(numProcs), ptrDispls(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        const index_t rowBegin = rowBounds[p];
        const index_t rowEnd = rowBounds[p + 1];
        rowCounts[p] = static_cast<int>(rowEnd - rowBegin);
        rowDispls[p] = static_cast<int>(rowBegin);
        nnzDispls[p] = static_cast<int>(h_rowDelimiters[rowBegin]);
        nnzCounts[p] = static_cast<int>(h_rowDelimiters[rowEnd] - h_rowDelimiters[rowBegin]);
        ptrCounts[p] = rowCounts[p] + 1;
        ptrDispls[p] = static_cast<int>(rowBegin);
    }

    const int localRows = rowCounts[rank];
    const int localNnz = nnzCounts[rank];
    std::vector<double> l_val(localNnz);
    std::vector<index_t> l_cols(localNnz);
    std::vector<index_t> l_rowDelimiters(localRows + 1);

    MPI_Scatterv(isRoot ? h_rowDelimiters.data() : nullptr, ptrCounts.data(), ptrDispls.data(),
                 MPI_UINT32_T, l_rowDelimiters.data(), localRows + 1, MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 l_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, l_cols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Rebase local row pointers so they index into l_val / l_cols
    {
        const index_t base = l_rowDelimiters[0];
        for (auto& ptr : l_rowDelimiters) {
            ptr -= base;
        }
    }

    // Determine the GPUs this rank drives. Ranks on the same node share
    // the node's GPUs round-robin.
    int numDevices = 0;
    const cudaError_t devErr = cudaGetDeviceCount(&numDevices);
    if (devErr != cudaSuccess || numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    std::vector<int> devices;
    if (localSize >= numDevices) {
        devices.push_back(localRank % numDevices);
    } else {
        for (int d = localRank; d < numDevices; d += localSize) {
            devices.push_back(d);
        }
    }
    const int numGpus = static_cast<int>(devices.size());

    // Split the local row block across this rank's GPUs, balanced by non-zeros
    const std::vector<index_t> devBounds =
        partitionRows(l_rowDelimiters.data(), static_cast<index_t>(localRows), numGpus);

    std::vector<DeviceSlice> slices(numGpus);
    std::vector<double> l_out(localRows, 0.0);

    // Upload each device's slice and perform one warm-up pass (untimed)
#pragma omp parallel for num_threads(numGpus) schedule(static, 1)
    for (int g = 0; g < numGpus; ++g) {
        DeviceSlice& s = slices[g];
        s.device = devices[g];
        s.rowBegin = devBounds[g];
        s.rows = devBounds[g + 1] - devBounds[g];
        const index_t nnzBegin = l_rowDelimiters[s.rowBegin];
        s.nnz = l_rowDelimiters[s.rowBegin + s.rows] - nnzBegin;

        std::vector<index_t> ptr(s.rows + 1);
        for (index_t i = 0; i <= s.rows; ++i) {
            ptr[i] = l_rowDelimiters[s.rowBegin + i] - nnzBegin;
        }

        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMalloc(&s.val, std::max<size_t>(s.nnz, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.cols, std::max<size_t>(s.nnz, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&s.rowDelimiters, (s.rows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&s.vec, std::max<size_t>(numRows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.out, std::max<size_t>(s.rows, 1) * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(s.val, l_val.data() + nnzBegin, s.nnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.cols, l_cols.data() + nnzBegin, s.nnz * sizeof(index_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.rowDelimiters, ptr.data(), (s.rows + 1) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.vec, h_vec.data(), numRows * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(s.out, 0, std::max<size_t>(s.rows, 1) * sizeof(double)));

        launchSpmv(s);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // For validation, compute reference solution (root only, untimed)
    std::vector<double> h_reference;
    if (isRoot && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(numGpus) schedule(static, 1)
    for (int g = 0; g < numGpus; ++g) {
        const DeviceSlice& s = slices[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        for (index_t iter = 0; iter < iterations; ++iter) {
            launchSpmv(s);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long elapsedMs = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration = std::chrono::milliseconds(elapsedMs);

    // Copy results back and release device memory (untimed)
#pragma omp parallel for num_threads(numGpus) schedule(static, 1)
    for (int g = 0; g < numGpus; ++g) {
        DeviceSlice& s = slices[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        if (s.rows > 0) {
            CUDA_CHECK(cudaMemcpy(l_out.data() + s.rowBegin, s.out, s.rows * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(s.val));
        CUDA_CHECK(cudaFree(s.cols));
        CUDA_CHECK(cudaFree(s.rowDelimiters));
        CUDA_CHECK(cudaFree(s.vec));
        CUDA_CHECK(cudaFree(s.out));
    }

    // Gather the full output vector on root
    MPI_Gatherv(l_out.data(), localRows, MPI_DOUBLE, isRoot ? h_out.data() : nullptr,
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (isRoot) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

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
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
