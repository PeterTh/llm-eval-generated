#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
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
    for (int64_t i = 0; i < dim; ++i) {
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
        if (!std::isfinite(ref) || !std::isfinite(res)) return false;
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


static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// A cooperative group per row: narrow groups avoid wasting lanes on sparse rows.
// All lanes participate in shuffles, including groups beyond the last row.
template<int Width>
__global__ void spmvGpu(const double* __restrict__ values,
                        const index_t* __restrict__ columns,
                        const index_t* __restrict__ offsets,
                        const double* __restrict__ vector, index_t rows,
                        double* __restrict__ output) {
    const unsigned lane = threadIdx.x % Width;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    // Round up so every lane in a physical warp executes the same shuffles.
    const uint64_t roundedRows = (uint64_t(rows) + 32 / Width - 1) / (32 / Width) * (32 / Width);
    for (uint64_t row = first; row < roundedRows; row += stride) {
        double sum = 0.0;
        if (row < rows) {
            for (uint64_t j = uint64_t(offsets[row]) + lane; j < offsets[row + 1]; j += Width)
                sum += values[j] * vector[columns[j]];
        }
        for (int delta = Width / 2; delta; delta /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, delta, Width);
        if (lane == 0 && row < rows) output[row] = sum;
    }
}

template<class T>
static T* deviceCopy(const std::vector<T>& host) {
    T* device = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), std::max(size_t(1), host.size()) * sizeof(T)));
    if (!host.empty())
        cudaCheck(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

// Chunked transfers avoid the signed-int count limit of MPI implementations.
template<class T>
static void transfer(T* data, uint64_t count, MPI_Datatype type, int peer, bool send) {
    constexpr uint64_t chunk = 1u << 26;
    for (uint64_t pos = 0; pos < count; pos += chunk) {
        int n = static_cast<int>(std::min(chunk, count - pos));
        if (send) MPI_Send(data + pos, n, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data + pos, n, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static index_t parseIndex(const char* value) {
    char* end = nullptr;
    unsigned long long n = strtoull(value, &end, 10);
    if (*value == '-' || end == value || *end || n == 0 || n > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Expected a positive 32-bit integer");
    return static_cast<index_t>(n);
}

static int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            maxVal = strtod(value, &end);
            if (end == value || *end || !std::isfinite(maxVal)) throw std::runtime_error("Invalid maximum value");
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else throw std::runtime_error("Unknown or incomplete option");
    }
    const uint64_t cells = uint64_t(numRows) * numRows;
    if (cells / sparsity > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Matrix exceeds the 32-bit CSR nonzero limit");
    const index_t nItems = static_cast<index_t>(cells / sparsity);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("Every MPI rank requires a CUDA device");
    // Also works when the scheduler exposes just one GPU to each rank.
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", ranks, omp_get_max_threads());
        printf("Initializing data structures...\n");
    }
    std::vector<double> values, vec(numRows);
    std::vector<index_t> columns, offsets;
    std::vector<index_t> boundaries(size_t(ranks) + 1), nonzeros(size_t(ranks) + 1);
    if (rank == 0) {
        values.resize(nItems);
        columns.resize(nItems);
        offsets.resize(size_t(numRows) + 1);
        // Preserve the original rand() stream exactly, independent of rank/thread count.
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), offsets.data(), nItems, numRows);
        boundaries[ranks] = numRows;
        nonzeros[ranks] = nItems;
        // Balance both nonzeros and row overhead; this also balances an empty matrix.
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = (uint64_t(nItems) + numRows) * p / ranks;
            uint64_t lo = 0, hi = numRows;
            while (lo < hi) {
                uint64_t mid = lo + (hi - lo) / 2;
                if (uint64_t(offsets[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = static_cast<index_t>(lo);
            nonzeros[p] = offsets[lo];
        }
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nonzeros.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (uint64_t pos = 0; pos < numRows; pos += (1u << 26))
        MPI_Bcast(vec.data() + pos, static_cast<int>(std::min(uint64_t(1u << 26), uint64_t(numRows) - pos)), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const index_t rows = boundaries[rank + 1] - boundaries[rank];
    const index_t nnz = nonzeros[rank + 1] - nonzeros[rank];
    std::vector<double> localValues(nnz), out(rows);
    std::vector<index_t> localColumns(nnz), localOffsets(size_t(rows) + 1);
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            transfer(offsets.data() + boundaries[p], uint64_t(boundaries[p + 1]) - boundaries[p] + 1, MPI_UINT32_T, p, true);
            if (nonzeros[p + 1] != nonzeros[p]) {
                transfer(values.data() + nonzeros[p], uint64_t(nonzeros[p + 1]) - nonzeros[p], MPI_DOUBLE, p, true);
                transfer(columns.data() + nonzeros[p], uint64_t(nonzeros[p + 1]) - nonzeros[p], MPI_UINT32_T, p, true);
            }
        }
        std::copy_n(offsets.data(), size_t(rows) + 1, localOffsets.data());
        std::copy_n(values.data(), nnz, localValues.data());
        std::copy_n(columns.data(), nnz, localColumns.data());
        std::vector<double>().swap(values);
        std::vector<index_t>().swap(columns);
        std::vector<index_t>().swap(offsets);
    } else {
        transfer(localOffsets.data(), uint64_t(rows) + 1, MPI_UINT32_T, 0, false);
        transfer(localValues.data(), nnz, MPI_DOUBLE, 0, false);
        transfer(localColumns.data(), nnz, MPI_UINT32_T, 0, false);
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= rows; ++i) localOffsets[i] -= nonzeros[rank];

    double* dValues = deviceCopy(localValues);
    double* dVec = deviceCopy(vec);
    double* dOut = deviceCopy(out);
    index_t* dColumns = deviceCopy(localColumns);
    index_t* dOffsets = deviceCopy(localOffsets);
    std::vector<double> reference;
    if (validate) {
        reference.resize(rows);
        spmvCpu(localValues.data(), localColumns.data(), localOffsets.data(), vec.data(), rows, reference.data());
    }
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const double average = rows ? double(nnz) / rows : 0;
    const int width = average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const unsigned blocks = static_cast<unsigned>(std::min(uint64_t(65535), (uint64_t(rows) * width + 255) / 256));
    auto launch = [&]() {
        if (!rows) return;
        switch (width) {
            case 4: spmvGpu<4><<<blocks, 256, 0, stream>>>(dValues, dColumns, dOffsets, dVec, rows, dOut); break;
            case 8: spmvGpu<8><<<blocks, 256, 0, stream>>>(dValues, dColumns, dOffsets, dVec, rows, dOut); break;
            case 16: spmvGpu<16><<<blocks, 256, 0, stream>>>(dValues, dColumns, dOffsets, dVec, rows, dOut); break;
            default: spmvGpu<32><<<blocks, 256, 0, stream>>>(dValues, dColumns, dOffsets, dVec, rows, dOut); break;
        }
    };
    launch(); // Warm up context, kernel and caches outside the timed region.
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    // Reusable, bounded-size graph reduces CPU launch overhead for short SpMVs.
    const index_t batch = std::min(index_t(32), iterations);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (rows) {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch();
        cudaCheck(cudaStreamEndCapture(stream, &graph));
        cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(executable, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows) {
        for (uint64_t i = 0; i < iterations / batch; ++i)
            cudaCheck(cudaGraphLaunch(executable, stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch();
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / seconds / 1e9);
    }
    if (rows && (validate || printResults))
        cudaCheck(cudaMemcpy(out.data(), dOut, size_t(rows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(numRows);
            std::copy(out.begin(), out.end(), result.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(result.data() + boundaries[p], uint64_t(boundaries[p + 1]) - boundaries[p], MPI_DOUBLE, p, false);
            print_results(result, "OutputVector");
        } else transfer(out.data(), rows, MPI_DOUBLE, 0, true);
    }
    int valid = !validate || verifyResults(reference.data(), out.data(), rows);
    int allValid = 0;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0 && validate) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
    if (executable) cudaCheck(cudaGraphExecDestroy(executable));
    if (graph) cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(dValues));
    cudaCheck(cudaFree(dColumns));
    cudaCheck(cudaFree(dOffsets));
    cudaCheck(cudaFree(dVec));
    cudaCheck(cudaFree(dOut));
    return allValid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
