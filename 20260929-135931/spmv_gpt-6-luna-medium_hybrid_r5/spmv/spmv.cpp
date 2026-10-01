#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

__global__ void spmvKernel(const double* val, const index_t* cols,
                           const index_t* rows, const double* vec,
                           index_t rowStart, index_t localRows, double* out) {
    const index_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local < localRows) {
        const index_t row = rowStart + local;
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[local] = sum;
    }
}

static void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
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
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (sparsity == 0 || numRows == 0) { if(rank==0) fprintf(stderr,"n and s must be positive\n"); MPI_Finalize(); return 1; }
    if(rank==0) printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    if(rank==0) printf("Matrix size: %u x %u\n", numRows, numRows);
    if(rank==0) printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    if(rank==0) printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    if(rank==0) { printf("Iterations: %u\n", iterations); printf("Max value: %.2f\n", maxVal); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    if(rank==0) printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if(rank==0) printf("Computing SpMV...\n");
    int deviceCount=0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount < 1) { if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    const index_t rowStart = static_cast<index_t>((static_cast<uint64_t>(numRows)*rank)/world);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(numRows)*(rank+1))/world);
    const index_t localRows=rowEnd-rowStart;
    double *dval=nullptr,*dvec=nullptr,*dout=nullptr;
    index_t *dcols=nullptr,*drows=nullptr;
    cudaCheck(cudaMalloc(&dval, std::max<size_t>(1,nItems)*sizeof(double)), "cudaMalloc values");
    cudaCheck(cudaMalloc(&dcols, std::max<size_t>(1,nItems)*sizeof(index_t)), "cudaMalloc columns");
    cudaCheck(cudaMalloc(&drows, (static_cast<size_t>(numRows)+1)*sizeof(index_t)), "cudaMalloc rows");
    cudaCheck(cudaMalloc(&dvec, static_cast<size_t>(numRows)*sizeof(double)), "cudaMalloc vector");
    cudaCheck(cudaMalloc(&dout, std::max<size_t>(1,localRows)*sizeof(double)), "cudaMalloc output");
    cudaCheck(cudaMemcpy(dval,h_val.data(),nItems*sizeof(double),cudaMemcpyHostToDevice),"copy values");
    cudaCheck(cudaMemcpy(dcols,h_cols.data(),nItems*sizeof(index_t),cudaMemcpyHostToDevice),"copy columns");
    cudaCheck(cudaMemcpy(drows,h_rowDelimiters.data(),(static_cast<size_t>(numRows)+1)*sizeof(index_t),cudaMemcpyHostToDevice),"copy rows");
    cudaCheck(cudaMemcpy(dvec,h_vec.data(),numRows*sizeof(double),cudaMemcpyHostToDevice),"copy vector");
    std::vector<double> localOut(localRows);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if(localRows) spmvKernel<<<(localRows+255)/256,256>>>(dval,dcols,drows,dvec,rowStart,localRows,dout);
        cudaCheck(cudaGetLastError(), "SpMV kernel");
        if(localRows) cudaCheck(cudaMemcpy(localOut.data(),dout,localRows*sizeof(double),cudaMemcpyDeviceToHost),"copy output");
        std::vector<int> counts(world), displs(world);
        for(int p=0;p<world;++p){displs[p]=static_cast<int>((static_cast<uint64_t>(numRows)*p)/world); counts[p]=static_cast<int>((static_cast<uint64_t>(numRows)*(p+1))/world)-displs[p];}
        MPI_Allgatherv(localOut.data(),static_cast<int>(localRows),MPI_DOUBLE,h_out.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double elapsed=duration.count()/1000.0, maxElapsed=0.0;
    MPI_Allreduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    duration=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(maxElapsed));

    if(rank==0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    if(rank==0) { printf("Average time per iteration: %.3f ms\n", avgTime); printf("Performance: %.3f GFLOPS\n", gflops); }
    
    // Print results for external validation
    if (printResults && rank==0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    if (validate) {
        if(rank==0) printf("Validating result...\n");
        const bool validLocal = rank==0 ? verifyResults(h_reference.data(), h_out.data(), numRows) : true;
        int validInt=validLocal?1:0, allValid=0; MPI_Allreduce(&validInt,&allValid,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);

        if (allValid) {
            if(rank==0) printf("Validation: PASSED\n");
            cudaFree(dval); cudaFree(dcols); cudaFree(drows); cudaFree(dvec); cudaFree(dout); MPI_Finalize();
            return 0;
        } else {
            if(rank==0) printf("Validation: FAILED\n");
            cudaFree(dval); cudaFree(dcols); cudaFree(drows); cudaFree(dvec); cudaFree(dout); MPI_Finalize();
            return 1;
        }
    }

    cudaFree(dval); cudaFree(dcols); cudaFree(drows); cudaFree(dvec); cudaFree(dout);
    MPI_Finalize();
    return 0;
}
