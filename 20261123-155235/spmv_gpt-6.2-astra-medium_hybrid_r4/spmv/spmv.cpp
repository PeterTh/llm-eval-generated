#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
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
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j) {
            uint64_t numEntriesLeft = uint64_t(dim - i) * dim - j;
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = fillRemaining ? 0.0 : static_cast<double>(rand()) / RAND_MAX;
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
bool verifyResults(const double* reference, const double* result, index_t size, index_t firstRow) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", firstRow + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       firstRow + i, ref, res, relError);
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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
class DeviceArray {
public:
    T* data = nullptr;

    explicit DeviceArray(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
    explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) {
        if (!host.empty())
            checkCuda(cudaMemcpy(data, host.data(), host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "upload");
    }
    DeviceArray(const DeviceArray&) = delete;
    ~DeviceArray() { if (data) checkCuda(cudaFree(data), "cudaFree"); }
};

// Store groups of 32 rows column-major. Adjacent lanes read adjacent entries,
// while every lane traverses its own row in the original CSR order.
constexpr unsigned sliceHeight = 32;

__global__ void multiply(const double* __restrict__ values, const index_t* __restrict__ columns,
                         const size_t* __restrict__ slices, const index_t* __restrict__ lengths,
                         const double* __restrict__ vec, index_t rows, double* out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t entry = slices[row / sliceHeight] + row % sliceHeight;
    double sum = 0;
    const index_t length = lengths[row];
    // Issue several independent gathers per lane to hide vector-load latency.
    #pragma unroll 8
    for (index_t j = 0; j < length; ++j, entry += sliceHeight)
        sum += __dmul_rn(values[entry], vec[columns[entry]]);
    out[row] = sum;
}

// MPI-3 vector collectives use signed int counts and displacements. Window
// large arrays so a 32-bit unsigned CSR offset can cross INT_MAX safely.
template<class T>
void distribute(T* global, T* local, const std::vector<index_t>& cuts, int rank,
                MPI_Datatype datatype, bool gather = false) {
    std::vector<int> counts(cuts.size() - 1), offsets(counts.size());
    for (uint64_t start = 0; start < cuts.back(); start += INT_MAX) {
        uint64_t end = std::min<uint64_t>(start + INT_MAX, cuts.back());
        for (size_t r = 0; r < counts.size(); ++r) {
            uint64_t first = std::clamp<uint64_t>(cuts[r], start, end);
            uint64_t last = std::clamp<uint64_t>(cuts[r+1], start, end);
            offsets[r] = first - start;
            counts[r] = last - first;
        }
        T* buffer = counts[rank] ? local + (start + offsets[rank] - cuts[rank]) : nullptr;
        T* rootBuffer = rank == 0 ? global + start : nullptr;
        if (gather)
            MPI_Gatherv(buffer, counts[rank], datatype, rootBuffer, counts.data(), offsets.data(),
                        datatype, 0, MPI_COMM_WORLD);
        else
            MPI_Scatterv(rootBuffer, counts.data(), offsets.data(), datatype, buffer,
                         counts[rank], datatype, 0, MPI_COMM_WORLD);
    }
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    const uint64_t cells = uint64_t(numRows) * numRows;
    if (!sparsity || cells / sparsity > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or matrix exceeds the CSR index range\n");
        return 1;
    }
    const index_t nItems = cells / sparsity;
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices), "query CUDA devices");
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices), "select CUDA device");

    std::vector<double> vec(numRows), globalValues;
    std::vector<index_t> globalColumns, globalOffsets;
    std::vector<index_t> rowCuts(size_t(ranks) + 1), entryCuts(rowCuts.size());

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - double(nItems) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalOffsets.resize(size_t(numRows) + 1);
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalOffsets.data(), nItems, numRows);

        // Balance both row overhead and nonzeros, including the generator's dense tail.
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int r = 1; r < ranks; ++r) {
            uint64_t target = (work / ranks) * r + (work % ranks) * r / ranks;
            index_t lo = rowCuts[r - 1], hi = numRows;
            while (lo < hi) {
                index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(globalOffsets[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            rowCuts[r] = lo;
            entryCuts[r] = globalOffsets[lo];
        }
        rowCuts.back() = numRows;
        entryCuts.back() = nItems;
    }
    MPI_Bcast(rowCuts.data(), ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(entryCuts.data(), ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    rowCuts.back() = numRows;
    entryCuts.back() = nItems;
    for (uint64_t i = 0; i < numRows; i += INT_MAX)
        MPI_Bcast(vec.data() + i, std::min<uint64_t>(INT_MAX, numRows - i),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t rows = rowCuts[rank+1] - rowCuts[rank];
    const index_t entries = entryCuts[rank+1] - entryCuts[rank];
    std::vector<index_t> offsets, columns;
    std::vector<double> values;
    if (ranks == 1) {
        offsets.swap(globalOffsets);
        columns.swap(globalColumns);
        values.swap(globalValues);
    } else {
        offsets.resize(size_t(rows) + 1);
        columns.resize(entries);
        values.resize(entries);
        distribute(globalOffsets.data(), offsets.data(), rowCuts, rank, MPI_UINT32_T);
        distribute(globalColumns.data(), columns.data(), entryCuts, rank, MPI_UINT32_T);
        distribute(globalValues.data(), values.data(), entryCuts, rank, MPI_DOUBLE);
        std::vector<index_t>().swap(globalOffsets);
        std::vector<index_t>().swap(globalColumns);
        std::vector<double>().swap(globalValues);
    }
    offsets[rows] = entryCuts[rank+1];
    for (auto& offset : offsets) offset -= entryCuts[rank];

    std::vector<double> reference(validate ? rows : 0);
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        spmvCpu(values.data(), columns.data(), offsets.data(), vec.data(), rows, reference.data());
    }

    size_t groups = (size_t(rows) + sliceHeight - 1) / sliceHeight;
    std::vector<size_t> slices(groups + 1);
    std::vector<index_t> lengths(rows);
    #pragma omp parallel for schedule(static)
    for (size_t g = 0; g < groups; ++g) {
        index_t width = 0;
        for (size_t r = g * sliceHeight; r < std::min<size_t>((g + 1) * sliceHeight, rows); ++r) {
            lengths[r] = offsets[r+1] - offsets[r];
            width = std::max(width, lengths[r]);
        }
        slices[g+1] = size_t(width) * sliceHeight;
    }
    for (size_t g = 0; g < groups; ++g) slices[g+1] += slices[g];
    std::vector<double> packedValues(slices.back());
    std::vector<index_t> packedColumns(slices.back());
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        size_t dst = slices[r / sliceHeight] + r % sliceHeight;
        for (index_t src = offsets[r]; src < offsets[r+1]; ++src, dst += sliceHeight) {
            packedValues[dst] = values[src];
            packedColumns[dst] = columns[src];
        }
    }

    std::vector<double>().swap(values);
    std::vector<index_t>().swap(columns);
    DeviceArray<double> dValues(packedValues), dVec(vec), dOut(rows);
    DeviceArray<index_t> dColumns(packedColumns), dLengths(lengths);
    DeviceArray<size_t> dSlices(slices);
    std::vector<double>().swap(packedValues);
    std::vector<index_t>().swap(packedColumns);
    if (!rank) printf("Computing SpMV...\n");
    checkCuda(cudaDeviceSynchronize(), "finish uploads");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows && iterations) {
        // Spread small partitions across more multiprocessors.
        const unsigned threads = rows < 8192 ? sliceHeight : 128;
        for (index_t i = 0; i < iterations; ++i)
            multiply<<<1 + (rows - 1) / threads, threads>>>(dValues.data, dColumns.data,
                dSlices.data, dLengths.data, dVec.data, rows, dOut.data);
        checkCuda(cudaGetLastError(), "launch SpMV");
        checkCuda(cudaDeviceSynchronize(), "complete SpMV");
    }
    double localTime = MPI_Wtime() - start, seconds;
    MPI_Reduce(&localTime, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000 / iterations : 0);
        printf("Performance: %.3f GFLOPS\n", seconds ? 2.0 * nItems * iterations / seconds / 1e9 : 0);
    }

    std::vector<double> out((printResults || validate) ? rows : 0);
    if (!out.empty() && iterations)
        checkCuda(cudaMemcpy(out.data(), dOut.data, out.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "download results");
    if (printResults) {
        std::vector<double> result(rank == 0 ? numRows : 0);
        distribute(result.data(), out.data(), rowCuts, rank, MPI_DOUBLE, true);
        if (!rank) print_results(result, "OutputVector");
    }
    if (validate) {
        if (!rank) printf("Validating result...\n");
        int localValid = verifyResults(reference.data(), out.data(), rows, rowCuts[rank]), valid;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return !valid;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
