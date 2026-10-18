#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>
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

// Use collective distribution for ordinary sizes, with chunked transfers when
// a CSR array exceeds the count/displacement range of MPI_Scatterv.
template <typename T>
void scatterArray(const std::vector<T>& global, std::vector<T>& local,
                  const std::vector<size_t>& offsets, const std::vector<size_t>& sizes,
                  MPI_Datatype type, int rank, int processes) {
    int small = 1;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        counts.resize(processes);
        displacements.resize(processes);
        for (int p = 0; p < processes; ++p) {
            if (offsets[p] > INT_MAX || sizes[p] > INT_MAX) {
                small = 0;
                break;
            }
            counts[p] = static_cast<int>(sizes[p]);
            displacements[p] = static_cast<int>(offsets[p]);
        }
    }
    MPI_Bcast(&small, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (small) {
        MPI_Scatterv(global.data(), counts.data(), displacements.data(), type,
                     local.data(), static_cast<int>(local.size()), type, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        if (!local.empty()) std::copy_n(global.data() + offsets[0], local.size(), local.data());
        for (int p = 1; p < processes; ++p) {
            for (size_t pos = 0; pos < sizes[p];) {
                const int count = static_cast<int>(std::min<size_t>(INT_MAX, sizes[p] - pos));
                MPI_Send(global.data() + offsets[p] + pos, count, type, p, 0, MPI_COMM_WORLD);
                pos += count;
            }
        }
    } else {
        for (size_t pos = 0; pos < local.size();) {
            const int count = static_cast<int>(std::min<size_t>(INT_MAX, local.size() - pos));
            MPI_Recv(local.data() + pos, count, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pos += count;
        }
    }
}

int run(int argc, char** argv, int rank, int processes) {
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
    if (sparsity == 0 || uint64_t(numRows) * numRows / sparsity > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or matrix exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = uint64_t(numRows) * numRows / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, numRows ? 100.0 * (1.0 - static_cast<double>(nItems) / (uint64_t(numRows) * numRows)) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", processes);
    }

    // Allocate and initialize data structures
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(rank == 0 ? size_t(numRows) + 1 : 0);
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Whole rows preserve each dot product's original summation order. Balance
    // nonzero work, while also charging for empty rows and output stores.
    std::vector<index_t> boundaries(size_t(processes) + 1);
    std::vector<index_t> metadata(rank == 0 ? size_t(processes) * 2 : 0);
    std::vector<size_t> rowOffsets, rowSizes, valueOffsets, valueSizes;
    if (rank == 0) {
        boundaries[processes] = numRows;
        const uint64_t totalWork = uint64_t(nItems) + numRows;
        for (int p = 1; p < processes; ++p) {
            const uint64_t target = totalWork * p / processes;
            size_t low = boundaries[p - 1], high = numRows;
            while (low < high) {
                const size_t mid = low + (high - low) / 2;
                if (uint64_t(h_rowDelimiters[mid]) + mid < target) low = mid + 1;
                else high = mid;
            }
            boundaries[p] = static_cast<index_t>(low);
        }
        rowOffsets.resize(processes);
        rowSizes.resize(processes);
        valueOffsets.resize(processes);
        valueSizes.resize(processes);
        for (int p = 0; p < processes; ++p) {
            rowOffsets[p] = boundaries[p];
            rowSizes[p] = size_t(boundaries[p + 1]) - boundaries[p];
            valueOffsets[p] = h_rowDelimiters[boundaries[p]];
            valueSizes[p] = h_rowDelimiters[boundaries[p + 1]] - valueOffsets[p];
            metadata[2 * size_t(p)] = static_cast<index_t>(rowSizes[p]);
            metadata[2 * size_t(p) + 1] = static_cast<index_t>(valueSizes[p]);
        }
    }
    index_t localInfo[2];
    MPI_Scatter(metadata.data(), 2, MPI_UINT32_T, localInfo, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = localInfo[0];
    std::vector<double> localValues(localInfo[1]), localOut(localRows);
    std::vector<index_t> localCols(localInfo[1]), localRowsCSR(localRows);
    localRowsCSR.reserve(size_t(localRows) + 1);
    scatterArray(h_val, localValues, valueOffsets, valueSizes, MPI_DOUBLE, rank, processes);
    scatterArray(h_cols, localCols, valueOffsets, valueSizes, MPI_UINT32_T, rank, processes);
    scatterArray(h_rowDelimiters, localRowsCSR, rowOffsets, rowSizes, MPI_UINT32_T, rank, processes);
    const index_t base = localRowsCSR.empty() ? 0 : localRowsCSR[0];
    for (auto& offset : localRowsCSR) offset -= base;
    localRowsCSR.push_back(localInfo[1]);

    // The input vector is invariant across iterations, so no halo exchange or
    // result communication is needed in the timed loop.
    for (size_t pos = 0; pos < h_vec.size();) {
        const int count = static_cast<int>(std::min<size_t>(INT_MAX, h_vec.size() - pos));
        MPI_Bcast(h_vec.data() + pos, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        pos += count;
    }
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);
    std::vector<index_t>().swap(h_rowDelimiters);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localRowsCSR.data(),
                h_vec.data(), localRows, localOut.data());
    }

    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);

        // Calculate performance metrics using the slowest rank's elapsed time.
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations ? duration * 1000.0 / iterations : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather once, and only if the complete result is actually needed.
    if (validate || printResults) {
        if (rank == 0) {
            h_out.resize(numRows);
            std::copy(localOut.begin(), localOut.end(), h_out.begin());
            for (int p = 1; p < processes; ++p) {
                for (size_t pos = boundaries[p]; pos < boundaries[p + 1];) {
                    const int count = static_cast<int>(std::min<size_t>(INT_MAX, boundaries[p + 1] - pos));
                    MPI_Recv(h_out.data() + pos, count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += count;
                }
            }
        } else {
            for (size_t pos = 0; pos < localOut.size();) {
                const int count = static_cast<int>(std::min<size_t>(INT_MAX, localOut.size() - pos));
                MPI_Send(localOut.data() + pos, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                pos += count;
            }
        }
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int status = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = run(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
