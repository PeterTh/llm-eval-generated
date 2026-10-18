#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <exception>
#include <limits>

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
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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
    #pragma omp parallel for schedule(dynamic, 64)
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
        if (!std::isfinite(ref) || !std::isfinite(res)) {
            printf("Validation failed at index %u: non-finite result\n", i);
            return false;
        } else if (std::abs(ref) < 1e-10) {
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

static void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunked transfers also support arrays larger than MPI's int count limit.
template <class T>
static void transfer(T* data, size_t count, MPI_Datatype type, int peer, int tag, bool send) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send)
            mpiCheck(MPI_Send(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD));
        else
            mpiCheck(MPI_Recv(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

template <class T>
static T* deviceAllocate(size_t count) {
    T* ptr = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(size_t(1), count) * sizeof(T)));
    return ptr;
}

// A cooperative group of lanes owns a row. Short rows use sub-warps to avoid
// wasting a whole warp, while long rows use all 32 lanes for coalesced loads.
template <int Width>
__global__ void spmvGpu(const double* __restrict__ values,
                        const index_t* __restrict__ columns,
                        const index_t* __restrict__ rows,
                        const double* __restrict__ vector,
                        index_t count, double* __restrict__ output) {
    const unsigned lane = threadIdx.x % Width;
    const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x % 32) / Width * Width);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t row = first; row < count; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        for (int delta = Width / 2; delta; delta /= 2)
            sum += __shfl_down_sync(mask, sum, delta, Width);
        if (lane == 0) output[row] = sum;
    }
}

static void launchSpmv(int width, int blocks, cudaStream_t stream, const double* values,
                       const index_t* columns, const index_t* rows, const double* vector,
                       index_t count, double* output) {
    if (!count) return;
    if (width == 4)
        spmvGpu<4><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, output);
    else if (width == 8)
        spmvGpu<8><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, output);
    else
        spmvGpu<32><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, output);
}

static int run(int argc, char** argv, int rank, int ranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || entries / sparsity > std::numeric_limits<index_t>::max()
        || !std::isfinite(maxVal)) {
        if (rank == 0) fprintf(stderr, "Invalid dimensions, sparsity, or maximum value (CSR uses 32-bit indices).\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each process to a GPU using its node-local MPI rank. Respect a
    // scheduler's per-process CUDA_VISIBLE_DEVICES assignment as well.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0, devices = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % devices;
    cudaCheck(cudaSetDevice(device));
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device));
    mpiCheck(MPI_Comm_free(&localComm));
    if (rank == 0) printf("MPI ranks: %d; OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());

    // Allocate and initialize data structures
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(rank == 0 ? size_t(numRows) + 1 : 0);
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(rank == 0 && printResults ? numRows : 0);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // The matrix and vector are invariant across iterations. Distribute once;
    // no communication or host/device copies are needed in the timed loop.
    for (size_t offset = 0; offset < numRows;) {
        int count = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(h_vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        offset += count;
    }
    std::vector<index_t> cuts(size_t(ranks) + 1);
    if (rank == 0) {
        cuts[0] = 0;
        cuts[ranks] = numRows;
        #pragma omp parallel for schedule(static)
        for (int p = 1; p < ranks; ++p) {
            if (nItems) {
                const index_t target = uint64_t(nItems) * p / ranks;
                cuts[p] = static_cast<index_t>(std::lower_bound(h_rowDelimiters.begin(),
                    h_rowDelimiters.end(), target) - h_rowDelimiters.begin());
            } else {
                cuts[p] = uint64_t(numRows) * p / ranks;
            }
        }
    }
    mpiCheck(MPI_Bcast(cuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    const index_t localRows = cuts[rank + 1] - cuts[rank];
    std::vector<index_t> rows(size_t(localRows) + 1);
    std::vector<double> values;
    std::vector<index_t> columns;
    if (rank == 0) {
        std::copy_n(h_rowDelimiters.data(), size_t(localRows) + 1, rows.data());
        values.assign(h_val.begin(), h_val.begin() + rows.back());
        columns.assign(h_cols.begin(), h_cols.begin() + rows.back());
        for (int p = 1; p < ranks; ++p) {
            const index_t begin = h_rowDelimiters[cuts[p]];
            const size_t count = h_rowDelimiters[cuts[p + 1]] - begin;
            transfer(h_rowDelimiters.data() + cuts[p], size_t(cuts[p + 1] - cuts[p]) + 1,
                     MPI_UINT32_T, p, 0, true);
            if (count) {
                transfer(h_val.data() + begin, count, MPI_DOUBLE, p, 1, true);
                transfer(h_cols.data() + begin, count, MPI_UINT32_T, p, 2, true);
            }
        }
    } else {
        transfer(rows.data(), rows.size(), MPI_UINT32_T, 0, 0, false);
        const size_t count = rows.back() - rows.front();
        values.resize(count);
        columns.resize(count);
        transfer(values.data(), count, MPI_DOUBLE, 0, 1, false);
        transfer(columns.data(), count, MPI_UINT32_T, 0, 2, false);
    }
    const index_t base = rows.front();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows.size(); ++i) rows[i] -= base;
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);
    std::vector<double> localOut(localRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), rows.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    double* d_values = deviceAllocate<double>(values.size());
    index_t* d_columns = deviceAllocate<index_t>(columns.size());
    index_t* d_rows = deviceAllocate<index_t>(rows.size());
    double* d_vector = deviceAllocate<double>(numRows);
    double* d_output = deviceAllocate<double>(localRows);
    if (!values.empty()) {
        cudaCheck(cudaMemcpy(d_values, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_columns, columns.data(), columns.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    cudaCheck(cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_vector, h_vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaCheck(cudaMemsetAsync(d_output, 0, size_t(localRows) * sizeof(double), stream));
    const double average = localRows ? double(values.size()) / localRows : 0;
    const int width = average <= 4 ? 4 : average <= 16 ? 8 : 32;
    const int blocks = static_cast<int>(std::max(uint64_t(1), std::min(
        (uint64_t(localRows) * width + 255) / 256, uint64_t(properties.multiProcessorCount) * 32)));

    // Warm up outside timing, then use a small reusable CUDA graph to amortize
    // launch overhead for long iteration runs without O(iterations) setup.
    if (iterations) launchSpmv(width, blocks, stream, d_values, d_columns, d_rows, d_vector, localRows, d_output);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    constexpr index_t batch = 32;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (iterations >= batch && localRows) {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i)
            launchSpmv(width, blocks, stream, d_values, d_columns, d_rows, d_vector, localRows, d_output);
        cudaCheck(cudaStreamEndCapture(stream, &graph));
        cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(executable, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    index_t iter = 0;
    if (executable) {
        for (; iterations - iter >= batch; iter += batch)
            cudaCheck(cudaGraphLaunch(executable, stream));
    }
    for (; localRows && iter < iterations; ++iter)
        launchSpmv(width, blocks, stream, d_values, d_columns, d_rows, d_vector, localRows, d_output);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (localRows)
        cudaCheck(cudaMemcpy(localOut.data(), d_output, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (executable) cudaCheck(cudaGraphExecDestroy(executable));
    if (graph) cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(d_values));
    cudaCheck(cudaFree(d_columns));
    cudaCheck(cudaFree(d_rows));
    cudaCheck(cudaFree(d_vector));
    cudaCheck(cudaFree(d_output));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        const double gflops = seconds > 0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0;
        const double avgTime = iterations ? seconds * 1000 / iterations : 0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Only collect the global output when it is actually requested.
    if (printResults) {
        if (rank == 0) {
            std::copy(localOut.begin(), localOut.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(h_out.data() + cuts[p], cuts[p + 1] - cuts[p], MPI_DOUBLE, p, 3, false);
            print_results(h_out, "OutputVector");
        } else {
            transfer(localOut.data(), localRows, MPI_DOUBLE, 0, 3, true);
        }
    }

    // Validation
    if (validate) {
        const int localValid = verifyResults(h_reference.data(), localOut.data(), localRows);
        int valid = 0;
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return result;
}
