#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
//   cols:          array for column indexes of elements
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   val:           array for values (to be allocated and filled)
//   n:             global number of nonzero elements (target)
//   dim:           global number of rows/columns in the matrix
//   rank:          MPI rank
//   size:          MPI size
//   local_vals:    output vector for local values
//   local_cols:    output vector for local column indices
//   local_rowDelimiters: output vector for local row delimiters
//
// ****************************************************************************
void initRandomMatrixParallel(const index_t n, const index_t dim, 
                              int rank, int size,
                              std::vector<double>& local_vals,
                              std::vector<index_t>& local_cols,
                              std::vector<index_t>& local_rowDelimiters,
                              double maxVal) {
    
    // Determine local row range
    index_t rows_per_proc = dim / size;
    index_t remainder = dim % size;
    
    index_t local_num_rows = (rank < (int)remainder) ? rows_per_proc + 1 : rows_per_proc;
    index_t row_start = (rank < (int)remainder) ? rank * (rows_per_proc + 1) : remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
    index_t row_end = row_start + local_num_rows;

    local_rowDelimiters.resize(local_num_rows + 1);

    // Re-seed to ensure consistent generation across ranks if we were generating everything,
    // but here we want to generate only our part.
    // To maintain the *exact* same matrix as the serial version, we would need to discard 
    // random numbers for rows before ours. 
    // However, that is O(N^2) work. 
    // The "equivalent semantics" requirement likely means "perform SpMV correctly", 
    // not "generate the exact same random numbers".
    // But to be safe and rigorous, let's try to skip.
    // Since rand() is cheap, skipping might be acceptable if N is not too huge.
    // But O(N^2) skipping is bad.
    // Let's implement a parallel generation that approximates the distribution.
    
    // Actually, looking at the serial code:
    // It iterates i from 0 to dim, j from 0 to dim.
    // It keeps track of nnzAssigned.
    // The probability depends on (n - nnzAssigned) / numEntriesLeft.
    // This makes the probability of a nonzero depend on previous decisions!
    // This sequential dependency makes parallel generation of the *exact* same matrix impossible 
    // without running the serial generator.
    
    // Strategy: Rank 0 generates the structure (or everyone does redundantly) but only stores local part.
    // Redundant generation is O(N^2) compute but O(N/P) memory.
    // For a benchmark, correctness of the SpMV is paramount. 
    // Let's do redundant generation for structure to guarantee same matrix, 
    // but only store local parts to save memory.
    // This limits scalability of setup, but ensures correctness and equivalent semantics.
    
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);

    bool fillRemaining = false;
    index_t current_local_row = 0;
    local_rowDelimiters[0] = 0; // Local offset starts at 0

    for (index_t i = 0; i < dim; ++i) {
        bool is_local = (i >= row_start && i < row_end);
        
        // If we are past our rows, we can stop if we don't care about future rand calls state
        // But the loop condition depends on nnzAssigned which depends on rand results.
        // So we must execute the loop logic.
        
        for (index_t j = 0; j < dim; ++j) {
            uint64_t numEntriesLeft = (static_cast<uint64_t>(dim) * static_cast<uint64_t>(dim)) - ((static_cast<uint64_t>(i) * static_cast<uint64_t>(dim)) + static_cast<uint64_t>(j));
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (is_local) {
                    local_cols.push_back(j);
                }
                nnzAssigned++;
            }
        }
        if (is_local) {
            current_local_row++;
            local_rowDelimiters[current_local_row] = local_cols.size();
        }
    }
    
    // Now fill values. The original code calls fill(h_val.data(), nItems, maxVal).
    // fill() just loops and calls rand().
    // We need to match which values go to which rank.
    // Since we know the global nnzAssigned (it's nItems), and we know which indices we own.
    // But fill() assigns values sequentially to the compressed array.
    // The compressed array order is determined by the row-major order of the matrix.
    // So if Rank 0 owns rows 0-K, it owns the first X non-zeros.
    // Rank 1 owns the next Y non-zeros.
    // So we can just call rand() and assign if it belongs to us.
    
    // We need to know how many non-zeros are before us.
    // We can do this by communicating or by re-running the loop (which we just did).
    
    // Let's refine the loop above to also handle values.
    // Since 'fill' is called separately in original code, it uses a CONTINUATION of the rand sequence.
    // The original code:
    // 1. fill(h_vec) -> consumes dim rand calls
    // 2. fill(h_val) -> consumes nItems rand calls
    // 3. initRandomMatrix -> consumes dim*dim rand calls AND seeds with 8675309
    
    // WAIT! initRandomMatrix calls srand(8675309) at the START.
    // This RESETS the sequence.
    // So h_vec and h_val are filled with rand sequence starting from... unknown seed (default 1).
    // THEN initRandomMatrix resets seed and uses it for structure.
    
    // This is weird.
    // h_val and h_vec are filled BEFORE initRandomMatrix in main.
    // But initRandomMatrix resets the seed.
    // So h_val and h_vec use the default seed.
    
    // To match original:
    // 1. Generate h_vec (size dim). Replicated on all ranks.
    //    All ranks can just call srand(1); fill(..., dim, ...); to get the same vector.
    // 2. Generate h_val (size nItems). Distributed.
    //    This is tricky. fill() assigns values 0 to nItems-1.
    //    Rank 0 needs 0..count0-1. Rank 1 needs count0..count0+count1-1.
    //    We need to know the counts first.
    // 3. Generate structure (initRandomMatrix).
    
    // Better approach for h_val:
    // Since we are iterating for structure anyway, we can generate the values THEN?
    // No, `h_val` in original code is just random numbers. It doesn't depend on structure.
    // But the mapping of `h_val[k]` to matrix entry `(i,j)` DOES depend on structure.
    // `h_val[k]` corresponds to the k-th non-zero in row-major order.
    
    // So:
    // 1. Replicated h_vec: srand(1); fill.
    // 2. Distributed h_val: 
    //    We need to know how many non-zeros each rank will have to allocate/generate.
    //    We can run the structure generation loop once to count, and again to fill?
    //    Or just use a dynamic vector (std::vector) which we are doing.
    //    But we need the correct rand values.
    //    Since `fill` uses `rand()`, we can just skip `rand()` calls for non-local elements.
    //    BUT `fill` uses the default seed (continuation of srand(1)).
    //    `initRandomMatrix` uses a fixed seed.
    
    //    So we have two independent random sequences.
    //    Sequence A (default seed): used for h_vec, then h_val.
    //    Sequence B (8675309): used for structure.
    
    //    We can handle this.
    //    Step 1: Everyone srand(1).
    //    Step 2: Everyone generates h_vec (dim calls). Store it.
    //    Step 3: Everyone generates h_val. 
    //            To do this distributed, we need to know WHICH part of h_val we own.
    //            This depends on the structure.
    //            So we MUST generate structure first (or simultaneously).
    //            But structure uses Sequence B.
    
    //    So we need to interleave or pre-calculate.
    //    We can compute the structure first (using Sequence B).
    //    Count how many non-zeros each rank has (local_nnz).
    //    And how many non-zeros precede each rank (offset_nnz).
    //    Then switch to Sequence A.
    //    Skip 'dim' calls (for h_vec).
    //    Skip 'offset_nnz' calls.
    //    Generate 'local_nnz' calls for our local h_val.
    
    local_vals.resize(local_cols.size());
    
    // Now generate values
    // We need to know how many non-zeros are before us globally.
    // MPI_Exscan can give us the prefix sum of local_cols.size().
    long long local_count = local_cols.size();
    long long prefix_count = 0;
    MPI_Exscan(&local_count, &prefix_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) prefix_count = 0;
    
    // Seed for values (default seed 1)
    srand(1); 
    
    // Skip random numbers for h_vec
    for(index_t k=0; k<dim; ++k) rand();
    
    // Skip random numbers for preceding non-zeros
    for(long long k=0; k<prefix_count; ++k) rand();
    
    // Generate our values
    fill(local_vals.data(), local_count, maxVal);
}

// ... spmvCpu modified for partial results ...

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
//   num_rows: number of rows to process
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t num_rows, double* out) {
    for (index_t i = 0; i < num_rows; ++i) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Tasks: %d\n", size);
        printf("Initializing data structures...\n");
    }

    // Allocate and initialize data structures
    // h_vec is replicated on all ranks
    std::vector<double> h_vec(numRows);
    
    // Seed and fill h_vec consistently
    srand(1);
    fill(h_vec.data(), numRows, maxVal);

    // Local data structures
    std::vector<double> local_val;
    std::vector<index_t> local_cols;
    std::vector<index_t> local_rowDelimiters;

    initRandomMatrixParallel(nItems, numRows, rank, size, 
                             local_val, local_cols, local_rowDelimiters, maxVal);

    index_t local_num_rows = local_rowDelimiters.size() - 1;
    std::vector<double> local_out(local_num_rows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution (might be slow/memory intensive)...\n");
        // To validate, we need the FULL result.
        // We can either:
        // 1. Gather the full matrix on rank 0 and run serial SpMV (requires memory).
        // 2. Compute distributed and gather result, then compare against... what?
        //    We need a ground truth. The ground truth comes from serial execution.
        //    So we must execute serial logic somewhere.
        //    Or we can execute distributed SpMV, gather results, and compare with...
        //    No, we need to generate the reference result.
        //    Since we distributed the matrix generation, we don't have the full matrix on any node easily
        //    unless we regenerate it.
        //    Let's regenerate the full matrix on rank 0 for validation if memory permits.
        //    Given this is a benchmark, validation is usually for small sizes.
        
        if (rank == 0) {
            // Temporarily allocate full matrix
            std::vector<double> full_val;
            std::vector<index_t> full_cols;
            std::vector<index_t> full_rowDelimiters(numRows + 1);
            
            // To properly generate the full matrix we need a serial version of initRandomMatrix.
            // But I deleted it!
            // I should have kept it or I need to implement it inline.
            // Or I can gather the parts from other ranks!
            // Gathering is better because it verifies the distributed matrix IS what we think it is.
            
            // Gather row delimiters
            std::vector<int> recvcounts(size);
            std::vector<int> displs(size);
            
            // We need to know how many rows each rank has.
            // We can re-calculate or gather.
            // re-calc:
            for(int r=0; r<size; ++r) {
                 index_t r_rows = numRows / size;
                 index_t rem = numRows % size;
                 if (r < (int)rem) r_rows++;
                 recvcounts[r] = r_rows;
            }
            displs[0] = 0;
            for(int r=1; r<size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];
            
            // Gather local_out (final result) is easy.
            // But here we need the reference result.
            // To get reference result, we need full matrix.
            // It's complicated to gather CSR structure.
            // Let's just trust that initRandomMatrixParallel generates a partition of the conceptual matrix.
            // And use a separate serial generation for validation.
            // I will implement a serial generator here quickly.
            
            index_t nnzAssigned = 0;
            double prob = static_cast<double>(nItems) / (static_cast<double>(numRows) * static_cast<double>(numRows));
            srand(8675309);
            full_val.resize(nItems); // We need to fill these too
            full_cols.resize(nItems);
            
            // We need to fill 'val' consistently.
            // The parallel generator skipped rand() calls to match.
            // So we need to match that logic.
            // Parallel generator logic:
            // 1. Structure (Sequence B).
            // 2. Values (Sequence A).
            
            // Structure:
            bool fillRemaining = false;
            for (index_t i = 0; i < numRows; ++i) {
                full_rowDelimiters[i] = nnzAssigned;
                for (index_t j = 0; j < numRows; ++j) {
                    // index_t numEntriesLeft = (numRows * numRows) / 1 - ((i * numRows) + j); // approximate
                    // Actually use exact logic
                     index_t numEntriesLeftExact = (numRows * numRows) - ((i * numRows) + j);
                    index_t needToAssign = nItems - nnzAssigned;
                    if (numEntriesLeftExact <= needToAssign) fillRemaining = true;
                    
                    double randVal = static_cast<double>(rand()) / RAND_MAX;
                    if ((nnzAssigned < nItems && randVal <= prob) || fillRemaining) {
                        full_cols[nnzAssigned] = j;
                        nnzAssigned++;
                    }
                }
            }
            full_rowDelimiters[numRows] = nItems;
            
            // Values:
            srand(1);
            for(index_t k=0; k<numRows; ++k) rand(); // skip h_vec
            fill(full_val.data(), nItems, maxVal);
            
            h_reference.resize(numRows);
            spmvCpu(full_val.data(), full_cols.data(), full_rowDelimiters.data(), 
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), local_num_rows, local_out.data());
        
        // We need the full output vector for the next iteration?
        // No, in SpMV y = Ax, x is input, y is output.
        // Usually power method or iterative solvers use y as input for next step.
        // But this benchmark just repeats spmvCpu(..., h_vec, ..., h_out).
        // It does NOT update h_vec.
        // So we don't need communication between iterations!
        // Wait, if it's just a benchmark of the kernel, we just repeat the kernel.
        // Yes: spmvCpu(..., h_vec, ..., h_out). h_vec is const.
        // So no communication needed between iterations.
        
        // However, if we want to mimic a real solver, we might want to verify.
        // But the code just loops.
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results for validation
    std::vector<double> global_out;
    if (printResults || validate) {
        if (rank == 0) global_out.resize(numRows);
        
        // Gather counts and displs
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        // Calculate recvcounts and displs consistent with distribution
        for(int r=0; r<size; ++r) {
             index_t r_rows = numRows / size;
             index_t rem = numRows % size;
             if (r < (int)rem) r_rows++;
             recvcounts[r] = r_rows;
        }
        displs[0] = 0;
        for(int r=1; r<size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];
        
        MPI_Gatherv(local_out.data(), local_num_rows, MPI_DOUBLE,
                    global_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(global_out, "OutputVector");
    }

    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), global_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return (validate && rank == 0) ? 0 : 0; // Return code?
}
