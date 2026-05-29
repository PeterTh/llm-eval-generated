#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA SpMV kernel — grid-stride loop over rows, each thread handles one row.
// Uses __ldg() for read-only cache on vec lookups and shared memory is not
// beneficial for CSR (irregular access), so we keep it simple and efficient.
__global__ void spmvCudaKernel(const double* val, const index_t* cols,
                                const index_t* rowDelimiters,
                                const double* vec, const index_t dim,
                                double* out) {
    index_t tid = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (index_t row = tid; row < dim; row += static_cast<index_t>(gridDim.x * blockDim.x)) {
        double t = 0.0;
        index_t start = rowDelimiters[row];
        index_t end   = rowDelimiters[row + 1];
        for (index_t j = start; j < end; ++j) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
        out[row] = t;
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (OpenMP parallelized)
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
#pragma omp parallel for
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. Encoded in CSR format.
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    srand(8675309);

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
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP parallelized for reference/validation)
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for
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
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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

    // Print configuration (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: Hybrid MPI (%d ranks) + CUDA + OpenMP\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---------------------------------------------------------------
    // Global data allocation and initialization on rank 0
    // ---------------------------------------------------------------
    std::vector<double>  h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double>  h_vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // ---------------------------------------------------------------
    // Broadcast the input vector and row delimiters (needed by all ranks)
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------------
    // Compute 1D block distribution of rows across MPI ranks
    // ---------------------------------------------------------------
    // Each rank gets a contiguous block of rows. We compute start_row
    // and local_dim for this rank.
    index_t startRow = 0, localDim = 0;
    {
        // Balanced block distribution
        index_t baseRows = numRows / static_cast<index_t>(numRanks);
        index_t remainder = numRows % static_cast<index_t>(numRanks);
        index_t rIdx = static_cast<index_t>(rank);
        startRow = baseRows * rIdx + (rIdx < remainder ? rIdx : remainder);
        localDim = baseRows + (rIdx < remainder ? 1 : 0);
    }
    index_t endRow = startRow + localDim;

    // Number of nnz owned by this rank
    index_t localNnz = h_rowDelimiters[endRow] - h_rowDelimiters[startRow];

    // ---------------------------------------------------------------
    // Allocate local host buffers for this rank's portion
    // ---------------------------------------------------------------
    std::vector<double>  local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    // local_rowDelimiters: offset to global nnz index, size localDim+1
    std::vector<index_t> local_rowDelimiters(localDim + 1);

    if (rank == 0) {
        // Rank 0: extract its own portion
        for (index_t i = 0; i < localNnz; ++i) {
            index_t gIdx = h_rowDelimiters[startRow] + i;
            local_val[i]   = h_val[gIdx];
            local_cols[i]  = h_cols[gIdx];
        }
    } else {
        // Other ranks: receive their portion from rank 0
        MPI_Recv(local_val.data(),   localNnz, MPI_DOUBLE,  0, 100 + rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_cols.data(),  localNnz, MPI_UNSIGNED, 0, 101 + rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        // Send portions to other ranks (use a single loop to avoid deadlock)
        index_t rBase = numRows / static_cast<index_t>(numRanks);
        index_t rRem = numRows % static_cast<index_t>(numRanks);
        for (index_t r = 1; r < static_cast<index_t>(numRanks); ++r) {
            index_t rStart = rBase * r + (r < rRem ? r : rRem);
            index_t rEnd = rStart + rBase + (r < rRem ? 1 : 0);
            index_t rNnz = h_rowDelimiters[rEnd] - h_rowDelimiters[rStart];

            MPI_Send(h_val.data()   + h_rowDelimiters[rStart], rNnz, MPI_DOUBLE,  static_cast<int>(r), 100 + static_cast<int>(r), MPI_COMM_WORLD);
            MPI_Send(h_cols.data()  + h_rowDelimiters[rStart], rNnz, MPI_UNSIGNED, static_cast<int>(r), 101 + static_cast<int>(r), MPI_COMM_WORLD);
        }
    }

    // Build local rowDelimiters (offset relative to local data start)
    for (index_t i = 0; i <= localDim; ++i) {
        local_rowDelimiters[i] = h_rowDelimiters[startRow + i] - h_rowDelimiters[startRow];
    }

    // Local output vector
    std::vector<double> local_out(localDim);

    // ---------------------------------------------------------------
    // Reference solution (rank 0 only, for validation)
    // ---------------------------------------------------------------
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // ---------------------------------------------------------------
    // Allocate device memory
    // ---------------------------------------------------------------
    double*  d_val   = nullptr;
    index_t* d_cols  = nullptr;
    index_t* d_rowDel = nullptr;
    double*  d_vec   = nullptr;
    double*  d_out   = nullptr;

    cudaMalloc(&d_val,    localNnz * sizeof(double));
    cudaMalloc(&d_cols,   localNnz * sizeof(index_t));
    cudaMalloc(&d_rowDel, (localDim + 1) * sizeof(index_t));
    cudaMalloc(&d_vec,    numRows * sizeof(double));
    cudaMalloc(&d_out,    localDim * sizeof(double));

    // Copy data to device (one-time, reused across iterations)
    cudaMemcpy(d_val,    local_val.data(),   localNnz * sizeof(double),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols,   local_cols.data(),  localNnz * sizeof(index_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDel, local_rowDelimiters.data(), (localDim + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec,    h_vec.data(),        numRows * sizeof(double),    cudaMemcpyHostToDevice);

    // ---------------------------------------------------------------
    // Launch CUDA kernel for SpMV
    // ---------------------------------------------------------------
    if (rank == 0) printf("Computing SpMV...\n");

    const int blockSize = 256;
    int numBlocks = (localDim + blockSize - 1) / blockSize;
    // Cap to avoid exceeding max grid size; grid-stride handles the rest
    const int maxGridSize = 65535;
    if (numBlocks > maxGridSize) numBlocks = maxGridSize;

    // Sync point before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCudaKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDel, d_vec, localDim, d_out);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back
    cudaMemcpy(local_out.data(), d_out, localDim * sizeof(double), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------------
    // Print timing and performance (all ranks, for visibility)
    // ---------------------------------------------------------------
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Use global nnz for GFLOPS calculation
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // ---------------------------------------------------------------
    // Gather full output for validation / printing (rank 0)
    // ---------------------------------------------------------------
    std::vector<double> full_out(numRows);
    if (rank == 0) {
        // Copy own portion
        std::memcpy(full_out.data(), local_out.data(), localDim * sizeof(double));

        // Receive from other ranks
        index_t rBase = numRows / static_cast<index_t>(numRanks);
        index_t rRem = numRows % static_cast<index_t>(numRanks);
        for (index_t r = 1; r < static_cast<index_t>(numRanks); ++r) {
            index_t rStart = rBase * r + (r < rRem ? r : rRem);
            index_t rDim = rBase + (r < rRem ? 1 : 0);

            MPI_Recv(full_out.data() + rStart, rDim, MPI_DOUBLE, static_cast<int>(r), 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_out.data(), localDim, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
    }

    // ---------------------------------------------------------------
    // Print results (rank 0)
    // ---------------------------------------------------------------
    if (printResults && rank == 0) {
        print_results(full_out, "OutputVector");
    }

    // ---------------------------------------------------------------
    // Validation (rank 0)
    // ---------------------------------------------------------------
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), full_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // ---------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------
    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowDel);
    cudaFree(d_vec);
    cudaFree(d_out);

    MPI_Finalize();
    return 0;
}
