#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
        if (!std::isfinite(res)) {
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

// A subwarp cooperates on each row. Small rows use fewer lanes, while long
// rows get a full warp with coalesced CSR reads and a register-only reduction.
template <int Lanes>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           index_t count, double* __restrict__ output) {
    const unsigned lane = threadIdx.x & (Lanes - 1);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Lanes;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Lanes;
    for (uint64_t row = first; row < count; row += stride) {
        double sum = 0.0;
        const uint64_t end = rows[row + 1];
        for (uint64_t j = uint64_t(rows[row]) + lane; j < end; j += Lanes)
            sum += values[j] * vector[columns[j]];
        for (int offset = Lanes / 2; offset; offset /= 2)
            sum += __shfl_down_sync(__activemask(), sum, offset, Lanes);
        if (lane == 0) output[row] = sum;
    }
}

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

// Allocate even for an empty partition so that zero-nonzero matrices and more
// MPI ranks than rows follow the same execution path.
template <typename T> struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(size_t(1), count) * sizeof(T)), "cudaMalloc");
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void upload(const std::vector<T>& host) {
        if (!host.empty())
            cudaCheck(cudaMemcpy(data, host.data(), host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "copy to GPU");
    }
};

// Chunk messages to avoid MPI's signed-int count limit for large CSR arrays.
template <typename T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool send, int tag) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

index_t parseCount(const char* text, bool allowZero = false) {
    char* end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || text[0] == '-' || end == text || *end ||
        value > std::numeric_limits<index_t>::max() || (!allowZero && value == 0))
        throw std::runtime_error(std::string("Invalid integer argument: ") + text);
    return static_cast<index_t>(value);
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseCount(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseCount(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseCount(argv[++i], true);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Invalid maximum value");
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
    const uint64_t totalItems = uint64_t(numRows) * numRows / sparsity;
    if (totalItems > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Matrix exceeds the 32-bit CSR nonzero limit");
    const index_t nItems = static_cast<index_t>(totalItems);

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "enumerate CUDA devices");
    if (!deviceCount) throw std::runtime_error("A CUDA GPU is required on every MPI rank");
    // Also works with schedulers exposing just one assigned GPU to each rank.
    cudaCheck(cudaSetDevice(localRank % deviceCount), "select CUDA device");

    if (rank == 0) {
    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (uint64_t(numRows) * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", ranks, omp_get_max_threads());
    }

    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<index_t> boundaries(size_t(ranks) + 1);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        // Keep the original rand() call order and seed exactly. Replicating
        // generation on every rank would waste both memory and O(n^2) work.
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        boundaries[ranks] = numRows;
        #pragma omp parallel for schedule(static)
        for (int p = 1; p < ranks; ++p) {
            if (nItems) {
                const index_t target = uint64_t(nItems) * p / ranks;
                boundaries[p] = static_cast<index_t>(std::lower_bound(
                    h_rowDelimiters.begin(), h_rowDelimiters.end(), target)
                    - h_rowDelimiters.begin());
            } else {
                boundaries[p] = uint64_t(numRows) * p / ranks;
            }
        }
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (size_t offset = 0; offset < numRows;) {
        const int chunk = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        MPI_Bcast(h_vec.data() + offset, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += chunk;
    }

    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> rows(size_t(localRows) + 1), columns;
    std::vector<double> values;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t first = h_rowDelimiters[boundaries[p]];
            const index_t count = h_rowDelimiters[boundaries[p + 1]] - first;
            transfer(h_rowDelimiters.data() + boundaries[p],
                     size_t(boundaries[p + 1] - boundaries[p]) + 1, MPI_UINT32_T, p, true, 0);
            if (count) {
                transfer(h_cols.data() + first, count, MPI_UINT32_T, p, true, 1);
                transfer(h_val.data() + first, count, MPI_DOUBLE, p, true, 2);
            }
        }
        std::copy_n(h_rowDelimiters.begin(), rows.size(), rows.begin());
        columns.assign(h_cols.begin(), h_cols.begin() + rows.back());
        values.assign(h_val.begin(), h_val.begin() + rows.back());
        // Only local CSR storage remains during the benchmark.
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    } else {
        transfer(rows.data(), rows.size(), MPI_UINT32_T, 0, false, 0);
        columns.resize(rows.back() - rows.front());
        values.resize(columns.size());
        transfer(columns.data(), columns.size(), MPI_UINT32_T, 0, false, 1);
        transfer(values.data(), values.size(), MPI_DOUBLE, 0, false, 2);
    }
    const index_t base = rows.front();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows.size(); ++i) rows[i] -= base;
    std::vector<double> localOut(localRows, 0.0);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), rows.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    DeviceArray<double> d_values(values.size()), d_vec(numRows), d_out(localRows);
    DeviceArray<index_t> d_columns(columns.size()), d_rows(rows.size());
    d_values.upload(values);
    d_vec.upload(h_vec);
    d_columns.upload(columns);
    d_rows.upload(rows);
    const double average = localRows ? double(values.size()) / localRows : 0.0;
    const int lanes = average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, localRank % deviceCount), "GPU properties");
    const unsigned blocks = static_cast<unsigned>(std::min(
        (uint64_t(localRows) * lanes + 255) / 256,
        uint64_t(properties.multiProcessorCount) * 32));
    auto launch = [&]() {
        if (!localRows) return;
        switch (lanes) {
        case 4: spmvKernel<4><<<blocks, 256>>>(d_values.data, d_columns.data, d_rows.data, d_vec.data, localRows, d_out.data); break;
        case 8: spmvKernel<8><<<blocks, 256>>>(d_values.data, d_columns.data, d_rows.data, d_vec.data, localRows, d_out.data); break;
        case 16: spmvKernel<16><<<blocks, 256>>>(d_values.data, d_columns.data, d_rows.data, d_vec.data, localRows, d_out.data); break;
        default: spmvKernel<32><<<blocks, 256>>>(d_values.data, d_columns.data, d_rows.data, d_vec.data, localRows, d_out.data); break;
        }
    };
    // Warm up CUDA outside the timed region; no host/device traffic or MPI
    // communication is needed between iterations because the vector is fixed.
    launch();
    cudaCheck(cudaGetLastError(), "launch warmup");
    cudaCheck(cudaDeviceSynchronize(), "complete warmup");
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launch();
    }

    cudaCheck(cudaGetLastError(), "launch SpMV");
    cudaCheck(cudaDeviceSynchronize(), "complete SpMV");
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (iterations && localRows)
        cudaCheck(cudaMemcpy(localOut.data(), d_out.data, size_t(localRows) * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy result from GPU");

    if (rank == 0) {
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate performance metrics
    const double gflops = seconds > 0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
    const double avgTime = iterations ? seconds * 1000.0 / iterations : 0.0;
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        if (rank == 0) {
            std::vector<double> h_out(numRows);
            std::copy(localOut.begin(), localOut.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(h_out.data() + boundaries[p], boundaries[p + 1] - boundaries[p],
                         MPI_DOUBLE, p, false, 3);
            print_results(h_out, "OutputVector");
        } else {
            transfer(localOut.data(), localOut.size(), MPI_DOUBLE, 0, true, 3);
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = verifyResults(h_reference.data(), localOut.data(), localRows) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
