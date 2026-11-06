#include <algorithm>
#include <climits>
#include <limits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <mpi.h>

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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                      (static_cast<uint64_t>(i) * dim + j);
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

// Each MPI rank owns a contiguous range of rows and one GPU.  Its CSR row
// offsets are rebased to zero, while column indices address the full vector.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

__global__ void spmvThreadKernel(const double* val, const index_t* cols,
                                 const index_t* offsets, const double* vec,
                                 index_t rows, double* out) {
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        double sum = 0.0;
        for (index_t j = offsets[row]; j < offsets[row + 1]; ++j)
            sum += val[j] * __ldg(vec + cols[j]);
        out[row] = sum;
    }
}

__global__ void spmvWarpKernel(const double* val, const index_t* cols,
                               const index_t* offsets, const double* vec,
                               index_t rows, double* out) {
    const unsigned lane = threadIdx.x & 31;
    const uint64_t first = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x / 32;
    for (uint64_t row = first; row < rows; row += stride) {
        double sum = 0.0;
        const index_t end = offsets[row + 1];
        for (uint64_t j = static_cast<uint64_t>(offsets[row]) + lane; j < end; j += 32)
            sum += val[j] * __ldg(vec + cols[j]);
        for (int delta = 16; delta > 0; delta /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        if (lane == 0) out[row] = sum;
    }
}

class GpuSpmv {
public:
    GpuSpmv(const double* values, const index_t* cols, const index_t* rowOffsets,
            const double* vec, index_t rows, index_t nnz, index_t vecSize, int device)
        : rows_(rows), nnz_(nnz) {
        checkCuda(cudaSetDevice(device), "select device");
        allocate(&dOffsets_, static_cast<size_t>(rows) + 1);
        allocate(&dValues_, nnz);
        allocate(&dCols_, nnz);
        allocate(&dVec_, vecSize);
        allocate(&dOut_, rows);
        copyToDevice(dOffsets_, rowOffsets, static_cast<size_t>(rows) + 1);
        copyToDevice(dValues_, values, nnz);
        copyToDevice(dCols_, cols, nnz);
        copyToDevice(dVec_, vec, vecSize);
        if (rows) checkCuda(cudaMemset(dOut_, 0, static_cast<size_t>(rows) * sizeof(double)), "initialize output");
    }

    GpuSpmv(const GpuSpmv&) = delete;
    GpuSpmv& operator=(const GpuSpmv&) = delete;

    ~GpuSpmv() {
        cudaFree(dOut_);
        cudaFree(dVec_);
        cudaFree(dCols_);
        cudaFree(dValues_);
        cudaFree(dOffsets_);
    }

    void run(index_t iterations) {
        if (rows_ && iterations) {
            constexpr unsigned threads = 256;
            const bool shortRows = static_cast<uint64_t>(nnz_) <= 4ull * rows_;
            const uint64_t workPerBlock = shortRows ? threads : threads / 32;
            const uint64_t needed = (static_cast<uint64_t>(rows_) + workPerBlock - 1) / workPerBlock;
            const unsigned blocks = static_cast<unsigned>(needed < 65535 ? needed : 65535);
            for (index_t i = 0; i < iterations; ++i) {
                if (shortRows)
                    spmvThreadKernel<<<blocks, threads>>>(dValues_, dCols_, dOffsets_, dVec_, rows_, dOut_);
                else
                    spmvWarpKernel<<<blocks, threads>>>(dValues_, dCols_, dOffsets_, dVec_, rows_, dOut_);
            }
            checkCuda(cudaGetLastError(), "launch SpMV");
        }
        checkCuda(cudaDeviceSynchronize(), "execute SpMV");
    }

    void copy_result(double* result) const {
        if (rows_)
            checkCuda(cudaMemcpy(result, dOut_, static_cast<size_t>(rows_) * sizeof(double),
                                 cudaMemcpyDeviceToHost), "copy output to host");
    }

private:
    template <typename T> static void allocate(T** ptr, size_t count) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)), "allocate memory");
    }
    template <typename T> static void copyToDevice(T* dst, const T* src, size_t count) {
        if (count) checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), "copy data to device");
    }
    index_t rows_, nnz_;
    double *dValues_ = nullptr, *dVec_ = nullptr, *dOut_ = nullptr;
    index_t *dCols_ = nullptr, *dOffsets_ = nullptr;
};

// MPI uses int element counts. Transfer larger CSR arrays in bounded chunks.
template <typename T>
void sendArray(const T* data, size_t count, int dest, int tag, MPI_Datatype type) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void receiveArray(T* data, size_t count, int source, int tag, MPI_Datatype type) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void broadcastArray(T* data, size_t count, MPI_Datatype type) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

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

    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || (totalEntries / sparsity) > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or too many nonzeros for 32-bit CSR indices\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(totalEntries / sparsity);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               totalEntries ? 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    std::vector<double> vec(numRows), allValues;
    std::vector<index_t> allCols, allOffsets;
    std::vector<index_t> boundaries(static_cast<size_t>(ranks) + 1, 0);
    if (rank == 0) {
        allValues.resize(nItems);
        allCols.resize(nItems);
        allOffsets.resize(static_cast<size_t>(numRows) + 1);
        // Match the original program's initial C random-number state.
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allOffsets.data(), nItems, numRows);
        for (int r = 1; r < ranks; ++r) {
            if (nItems) {
                const index_t target = static_cast<index_t>((static_cast<uint64_t>(nItems) * r) / ranks);
                boundaries[r] = static_cast<index_t>(std::lower_bound(allOffsets.begin(), allOffsets.end(), target) - allOffsets.begin());
            } else {
                boundaries[r] = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / ranks);
            }
        }
        boundaries[ranks] = numRows;
    }
    broadcastArray(vec.data(), vec.size(), MPI_DOUBLE);

    index_t metadata[4] = {}; // first row, number of rows, first nonzero, number of nonzeros
    std::vector<double> values;
    std::vector<index_t> cols, offsets;
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            index_t first = boundaries[r], last = boundaries[r + 1];
            index_t info[4] = {first, last - first, allOffsets[first], allOffsets[last] - allOffsets[first]};
            if (r == 0) {
                std::copy(info, info + 4, metadata);
                offsets.assign(allOffsets.begin() + first, allOffsets.begin() + static_cast<size_t>(last) + 1);
                cols.assign(allCols.begin() + info[2], allCols.begin() + static_cast<size_t>(info[2]) + info[3]);
                values.assign(allValues.begin() + info[2], allValues.begin() + static_cast<size_t>(info[2]) + info[3]);
            } else {
                MPI_Send(info, 4, MPI_UINT32_T, r, 1, MPI_COMM_WORLD);
                sendArray(allOffsets.data() + first, static_cast<size_t>(info[1]) + 1, r, 2, MPI_UINT32_T);
                if (info[3]) {
                    sendArray(allCols.data() + info[2], info[3], r, 3, MPI_UINT32_T);
                    sendArray(allValues.data() + info[2], info[3], r, 4, MPI_DOUBLE);
                }
            }
        }
        std::vector<double>().swap(allValues);
        std::vector<index_t>().swap(allCols);
        std::vector<index_t>().swap(allOffsets);
    } else {
        MPI_Recv(metadata, 4, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offsets.resize(static_cast<size_t>(metadata[1]) + 1);
        cols.resize(metadata[3]);
        values.resize(metadata[3]);
        receiveArray(offsets.data(), offsets.size(), 0, 2, MPI_UINT32_T);
        receiveArray(cols.data(), cols.size(), 0, 3, MPI_UINT32_T);
        receiveArray(values.data(), values.size(), 0, 4, MPI_DOUBLE);
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < offsets.size(); ++i) offsets[i] -= metadata[2];

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    checkCuda(cudaGetDeviceCount(&deviceCount), "count CUDA devices");
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<double> out(metadata[1], 0.0), reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(metadata[1]);
        spmvCpu(values.data(), cols.data(), offsets.data(), vec.data(), metadata[1], reference.data());
    }

    double elapsed = 0.0;
    {
        GpuSpmv gpu(values.data(), cols.data(), offsets.data(), vec.data(),
                    metadata[1], metadata[3], numRows, localRank % deviceCount);
        if (rank == 0) printf("Computing SpMV...\n");
        // Exclude first-launch setup and possible device-code JIT from timing.
        if (iterations) gpu.run(1);
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        gpu.run(iterations);
        const double localElapsed = MPI_Wtime() - start;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (validate || printResults) gpu.copy_result(out.data());
    }
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double avgTime = iterations ? elapsed * 1000.0 / iterations : 0.0;
        const double gflops = elapsed > 0.0 ? (2.0 * nItems * iterations) / elapsed / 1e9 : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(numRows);
            std::copy(out.begin(), out.end(), result.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t count = static_cast<size_t>(boundaries[r + 1] - boundaries[r]);
                if (count) receiveArray(result.data() + boundaries[r], count, r, 5, MPI_DOUBLE);
            }
            print_results(result, "OutputVector");
        } else {
            sendArray(out.data(), out.size(), 0, 5, MPI_DOUBLE);
        }
    }

    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        uint64_t firstBad = std::numeric_limits<uint64_t>::max();
        #pragma omp parallel for reduction(min:firstBad) schedule(static)
        for (size_t i = 0; i < reference.size(); ++i) {
            const double ref = reference[i], res = out[i];
            if ((std::abs(ref) < 1e-10 && std::abs(res) > MAX_RELATIVE_ERROR) ||
                (std::abs(ref) >= 1e-10 && std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR)) {
                firstBad = std::min(firstBad, static_cast<uint64_t>(metadata[0]) + i);
            }
        }
        uint64_t globalBad;
        MPI_Allreduce(&firstBad, &globalBad, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        if (globalBad != std::numeric_limits<uint64_t>::max()) {
            int owner = firstBad == globalBad ? rank : INT_MAX, ownerRank;
            MPI_Allreduce(&owner, &ownerRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            double pair[2] = {};
            if (rank == ownerRank) {
                size_t i = static_cast<size_t>(globalBad - metadata[0]);
                pair[0] = reference[i]; pair[1] = out[i];
            }
            MPI_Bcast(pair, 2, MPI_DOUBLE, ownerRank, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Validation failed at index %llu: reference %.10e, got %.10e\n",
                       static_cast<unsigned long long>(globalBad), pair[0], pair[1]);
                printf("Validation: FAILED\n");
            }
            status = 1;
        } else if (rank == 0) {
            printf("Validation: PASSED\n");
        }
    }
    MPI_Finalize();
    return status;
}
