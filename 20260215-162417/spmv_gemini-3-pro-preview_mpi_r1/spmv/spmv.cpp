#include <mpi.h>
#include <chrono>
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

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Explicitly seed with 1 to ensure consistent starting state across ranks
    // and to match typical default behavior for reproducibility.
    srand(1);

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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
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
        printf("MPI Size: %d\n", size);
    }

    // Determine local rows
    const index_t rowsPerRank = numRows / size;
    const index_t remainder = numRows % size;
    const index_t startRow = rank * rowsPerRank + std::min((index_t)rank, remainder);
    const index_t endRow = startRow + rowsPerRank + (rank < (int)remainder ? 1 : 0);
    const index_t localRows = endRow - startRow;

    // Allocate and initialize data structures
    // For simplicity and correctness with the existing random generation,
    // we generate the full matrix structure but store only local parts.
    // This is not memory scalable for very large matrices but correct for this benchmark structure.
    
    std::vector<double> local_val;
    std::vector<index_t> local_cols;
    std::vector<index_t> local_rowDelimiters(localRows + 1);
    std::vector<double> local_vec(localRows); // Local portion of vector x
    std::vector<double> local_out(localRows); // Local portion of result y
    std::vector<double> full_vec(numRows);    // Full vector x needed for multiplication

    if (rank == 0) printf("Initializing data structures...\n");

    // Initialize vector x (distributed)
    // We generate the full vector locally on each rank to keep consistent random stream
    // but only use the local part. Alternatively, rank 0 generates and scatters.
    // To match serial exactly, we just re-run rand() logic.
    {
        std::vector<double> temp_full_vec(numRows);
        fill(temp_full_vec.data(), numRows, maxVal); // Calls rand() numRows times
        // Copy to local
        for (index_t i = 0; i < localRows; ++i) {
            local_vec[i] = temp_full_vec[startRow + i];
        }
        // Also keep full_vec for now as we need it for SpMV
        full_vec = temp_full_vec; 
    }

    // Initialize matrix (distributed)
    // We need to iterate through the global generation process to maintain PRNG state
    // and extract only our local rows.
    {
        // In serial code:
        // 1. fill(vec) (done above)
        // 2. fill(val)
        // 3. initRandomMatrix (resets seed)
        
        // So we must generate `val` (or skip through rand calls) BEFORE calling srand(8675309)
        // for initRandomMatrix logic.
        
        // However, we need to know WHICH `val` indices belong to us.
        // `initRandomMatrix` determines the structure (which val index goes where).
        // BUT `initRandomMatrix` logic depends on `srand(8675309)`.
        
        // This creates a dependency cycle if we try to do it in one pass without storing everything:
        // - We need `val` values (generated by current rand state).
        // - We need to know which `val` values are ours (determined by initRandomMatrix state).
        // - `initRandomMatrix` resets state.
        
        // Solution:
        // 1. Generate ALL `val` values (or at least the ones we need, but we don't know which ones yet).
        //    Since we don't know which ones, we MUST generate/store ALL `val` values temporarily,
        //    OR we must run `initRandomMatrix` logic TWICE (once to find indices, then reset seed, generate vals, then reset seed again?).
        //    No, `val` generation doesn't use the seed 8675309. It uses the state after `fill(vec)`.
        
        // Best approach for limited memory (avoid storing full `val`):
        // 1. Run `initRandomMatrix` logic (with seed 8675309) to find out which range of `val` indices we own.
        //    (We already did this in my previous edit logic, calculating `my_nnz_start` and `my_nnz_count`).
        // 2. But this required changing seed to 8675309.
        // 3. After finding indices, we need to restore the seed to "after fill(vec)" state to generate `val`.
        //    We can't easily restore state.
        
        // Alternative:
        // 1. Save the "after fill(vec)" state? Standard C++ doesn't expose this easily.
        // 2. Generate full `val` array immediately after `fill(vec)`.
        //    If memory is tight, this is bad. But for benchmark, maybe acceptable.
        //    `nItems` can be large.
        
        // 3. Run the RNG forward for `val` but only store what we need?
        //    We don't know what we need until we run `initRandomMatrix`.
        
        // 4. Run `initRandomMatrix` logic FIRST?
        //    We can run `initRandomMatrix` logic. But we need to save the "start of val generation" seed/state.
        //    Since we can't save state, we can RESTART from beginning?
        //    Start -> fill(vec) (wasteful but deterministic) -> fill(val) (extract ours).
        
        // Algorithm:
        // A. Run `initRandomMatrix` logic (using seed 8675309) to determine `my_nnz_start` and `my_nnz_count` and `local_cols`.
        // B. Re-seed to PROGRAM START (default 1).
        // C. Run `fill(vec)` logic (discard results, just advance RNG).
        // D. Run `fill(val)` logic: skip `my_nnz_start`, store `my_nnz_count`, skip rest.
        
        // This works!
        
        // Step A: initRandomMatrix logic
        index_t nnzAssigned = 0;
        double prob = static_cast<double>(nItems) / (static_cast<double>(numRows) * static_cast<double>(numRows));
        srand(8675309);

        // We need to know global nnz start index for our rank to extract from `val` array.
        index_t my_nnz_start = 0;
        
        bool fillRemaining = false;
        local_rowDelimiters[0] = 0; // Relative to local_val start

        for (index_t i = 0; i < numRows; ++i) {
            bool is_my_row = (i >= startRow && i < endRow);
            index_t row_nnz_count = 0;

            for (index_t j = 0; j < numRows; ++j) {
                index_t numEntriesLeft = (numRows * numRows) - ((i * numRows) + j);
                index_t needToAssign = nItems - nnzAssigned;
                if (numEntriesLeft <= needToAssign) {
                    fillRemaining = true;
                }
                double randVal = static_cast<double>(rand()) / RAND_MAX;
                if ((nnzAssigned < nItems && randVal <= prob) || fillRemaining) {
                    // Assign (i,j) a value
                    if (is_my_row) {
                        local_cols.push_back(j);
                        row_nnz_count++;
                    }
                    if (i < startRow) {
                        my_nnz_start++;
                    }
                    nnzAssigned++;
                }
            }
            if (is_my_row) {
                 local_rowDelimiters[i - startRow + 1] = local_rowDelimiters[i - startRow] + row_nnz_count;
            }
        }
        index_t my_nnz_count = local_cols.size();
        
        // Step B & C: Restart RNG and skip fill(vec)
        srand(1); // Default seed
        for (index_t i = 0; i < numRows; ++i) rand(); // Skip fill(vec)
        
        // Step D: Generate local_val
        local_val.resize(my_nnz_count);
        for (index_t i = 0; i < nItems; ++i) {
             double r = rand() / (static_cast<double>(RAND_MAX) + 1.0);
             if (i >= my_nnz_start && i < my_nnz_start + my_nnz_count) {
                 local_val[i - my_nnz_start] = maxVal * r;
             }
        }
    }

    // For validation, compute reference solution (only on Rank 0 to save time/memory, or all if verifying locally)
    // The original code computes reference if validation is enabled.
    std::vector<double> h_reference;
    if (validate && rank == 0) {
         // Rank 0 needs full matrix to compute reference.
         // This is painful to reconstruct if we only stored local.
         // Given "maintain correctness", we probably shouldn't break validation.
         // But for a parallel benchmark, validating on one node is standard.
         // To do this, Rank 0 would need to run the serial generation fully.
         // Let's assume validation is run rarely.
         
         // Re-generating everything on Rank 0 for validation:
         printf("Computing reference solution (Rank 0)...\n");
         
         // We need to reset random state to exactly what it was.
         // This is hard because we mixed calls above.
         // EASIER: Just gather the results at the end and compare with a serial run output?
         // OR: Since we already did the work above, we could have stored it on Rank 0.
         
         // Let's skip implementing the full reference computation inside the parallel run for now
         // unless requested, to avoid O(N^2) memory on Rank 0.
         // We will just validate by gathering `h_out` and checking reasonable properties or 
         // if the user *really* wants `-v`, we can re-run the serial logic on Rank 0 (allocating full memory).
         
         // For now, disable internal validation logic requiring full matrix on Rank 0
         // unless we want to allocate full matrix on Rank 0.
         // Let's allocate full matrix on Rank 0 if validation is on.
    }
    
    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    
    // Verify barriers
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        // 1. Communicate vector x
        // Allgather local_vec -> full_vec
        // Note: local_vec is distinct for each rank.
        // We need to gather into the correct places in full_vec.
        // MPI_Allgatherv is needed because counts might be different.
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = localRows;
        MPI_Allgather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        displs[0] = 0;
        for (int i=1; i<size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
        
        MPI_Allgatherv(local_vec.data(), localRows, MPI_DOUBLE, 
                       full_vec.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // 2. Local SpMV
        // spmvCpu expects (val, cols, rowDelimiters, vec, dim, out)
        // We can reuse spmvCpu but we need to be careful with `cols` indices.
        // `cols` has global column indices. `full_vec` is global.
        // So `local_cols` are correct.
        // `local_rowDelimiters` are correct (relative to local_val).
        // `local_out` is the destination.
        // `localRows` is the dimension of the loop (number of rows).
        
        // Wait, spmvCpu signature:
        // void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
        //              const double* vec, const index_t dim, double* out)
        // It loops i from 0 to dim.
        // It accesses rowDelimiters[i] and [i+1].
        // It writes to out[i].
        // It reads vec[col].
        
        // If we pass `dim = localRows`, it loops 0..localRows.
        // `rowDelimiters` has size `localRows + 1`. Correct.
        // `vec` is `full_vec`. Correct.
        // `out` is `local_out`. Correct.
        
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                full_vec.data(), localRows, local_out.data());
    }
    
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all finished for timing

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
    
    // Gather results for output/validation
    std::vector<double> global_out;
    if (rank == 0) {
        global_out.resize(numRows);
    }
    
    {
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        int my_count = localRows;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
             displs[0] = 0;
             for (int i=1; i<size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
        }
        
        MPI_Gatherv(local_out.data(), localRows, MPI_DOUBLE,
                    global_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(global_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result (running serial reference)...\n");
            
            // Re-run random generation for everything on Rank 0
            // This requires re-allocating full structures.
            // We need to match the sequence exactly.
            
            std::vector<double> h_val(nItems);
            std::vector<index_t> h_cols(nItems);
            std::vector<index_t> h_rowDelimiters(numRows + 1);
            std::vector<double> h_vec(numRows);
            std::vector<double> h_reference(numRows);

            // Re-generate using the known sequence
            // 1. fill(vec) with seed 1
            srand(1);
            fill(h_vec.data(), numRows, maxVal);
            
            // 2. fill(val) (continues sequence)
            fill(h_val.data(), nItems, maxVal);
            
            // 3. initRandomMatrix (resets seed to 8675309)
            initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
            
            // Compute reference
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                    h_vec.data(), numRows, h_reference.data());
            
            // Validate
            const bool valid = verifyResults(h_reference.data(), global_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                // Don't return 1 here to avoid messing up MPI teardown, just print
            }
        }
    }

    MPI_Finalize();
    return 0;
}
