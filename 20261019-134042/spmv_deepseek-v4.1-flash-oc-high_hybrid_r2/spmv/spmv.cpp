#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
//   Computes sparse matrix-vector multiplication using CSR format.
//   OpenMP parallelizes across rows; each row keeps the original sequential
//   accumulation order so results are bit-for-bit reproducible.
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
// Function: spmvEllKernel
//
// Purpose:
//   CUDA SpMV using an ELL (row-major padded) layout. Consecutive threads own
//   consecutive rows, so at every step k the loads are fully coalesced. Each
//   thread still walks its own row in the original order, therefore the
//   rounding sequence (and the final bits) match the reference CPU kernel.
//   Rows longer than kCap are left to the CPU helper below.
//
// ****************************************************************************
__global__ void spmvEllKernel(const double* __restrict__ ellVal,
                              const index_t* __restrict__ ellCols,
                              const index_t kCap,
                              const index_t* __restrict__ csrOffsets,
                              const double* __restrict__ vec,
                              const index_t numRows,
                              double* __restrict__ out) {
    const index_t row = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < numRows) {
        const index_t start = csrOffsets[row];
        const index_t end = csrOffsets[row + 1];
        const index_t len = end - start;
        if (len <= kCap) {
            double t = 0.0;
            index_t k = 0;
            // Manual 4-way software pipelining: all loads for the next four
            // entries are issued before the (necessarily serial) additions.
            for (; k + 4 <= len; k += 4) {
                const size_t idx0 = static_cast<size_t>(k) * numRows + row;
                const size_t idx1 = idx0 + numRows;
                const size_t idx2 = idx1 + numRows;
                const size_t idx3 = idx2 + numRows;
                const double v0 = __ldg(ellVal + idx0);
                const double v1 = __ldg(ellVal + idx1);
                const double v2 = __ldg(ellVal + idx2);
                const double v3 = __ldg(ellVal + idx3);
                const double x0 = __ldg(vec + ellCols[idx0]);
                const double x1 = __ldg(vec + ellCols[idx1]);
                const double x2 = __ldg(vec + ellCols[idx2]);
                const double x3 = __ldg(vec + ellCols[idx3]);
                t = __dadd_rn(t, __dmul_rn(v0, x0));
                t = __dadd_rn(t, __dmul_rn(v1, x1));
                t = __dadd_rn(t, __dmul_rn(v2, x2));
                t = __dadd_rn(t, __dmul_rn(v3, x3));
            }
            for (; k < len; ++k) {
                const size_t idx = static_cast<size_t>(k) * numRows + row;
                t = __dadd_rn(t, __dmul_rn(__ldg(ellVal + idx), __ldg(vec + ellCols[idx])));
            }
            out[row] = t;
        }
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
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

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks receive the same arguments)
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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---------------------------------------------------------------------
    // Data generation: the root rank reproduces the original sequential RNG
    // stream exactly, then the sparse rows are distributed across MPI ranks.
    // ---------------------------------------------------------------------
    std::vector<double> h_val;                 // Full values (root only)
    std::vector<index_t> h_cols;               // Full columns (root only)
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Every rank needs the row boundaries and the dense input vector. The
    // input vector is constant throughout the benchmark, so it is broadcast
    // once and then reused by all iterations without further communication.
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reference solution (root, exact original algorithm)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Row ownership: contiguous, balanced partition of the rows.
    const index_t firstRow =
        static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) / numProcs);
    const index_t lastRow =
        static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) / numProcs);
    const index_t localRows = lastRow - firstRow;

    // Per-rank nnz counts and displacements, derived from the global CSR.
    std::vector<int> nnzCounts(numProcs);
    std::vector<int> nnzDispls(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        const index_t s = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / numProcs);
        const index_t e = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) / numProcs);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[s]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[e] - h_rowDelimiters[s]);
    }

    const index_t localBase = h_rowDelimiters[firstRow];
    const index_t localNnz = h_rowDelimiters[lastRow] - localBase;

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(localRows + 1);

    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // Local CSR offsets are zero-based; column indices stay global.
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[firstRow + i] - localBase;
    }

    // Root no longer needs the full sparse matrix.
    if (isRoot) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
    }

    std::vector<double> localOut(localRows);

    // ---------------------------------------------------------------------
    // Accelerator setup: one MPI rank drives one GPU. Ranks sharing a node
    // are placed on distinct devices (round-robin if there are more ranks
    // than devices).
    // ---------------------------------------------------------------------
    int localRank = rank;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
    }

    int deviceCount = 0;
    bool useGpu = false;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        const int device = localRank % deviceCount;
        if (cudaSetDevice(device) == cudaSuccess) {
            useGpu = true;
        }
    }

    double* d_ellVal = nullptr;
    index_t* d_ellCols = nullptr;
    index_t* d_csrOffsets = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    // Build a capped ELL (padded row-major) copy of the local matrix. The ELL
    // kernel issues coalesced loads for consecutive rows, which is much faster
    // than CSR's strided row starts while keeping the exact per-row order.
    index_t maxRowLen = 0;
#pragma omp parallel for schedule(static) reduction(max : maxRowLen)
    for (index_t i = 0; i < localRows; ++i) {
        const index_t len = localRowDelimiters[i + 1] - localRowDelimiters[i];
        if (len > maxRowLen) maxRowLen = len;
    }

    index_t kCap = 0;
    if (useGpu && localRows > 0 && maxRowLen > 0) {
        // Bound the ELL storage to ~3x the CSR nnz while covering the bulk of
        // the row-length distribution; the few longer rows use the CPU path.
        const unsigned long long capByMemory =
            (3ULL * static_cast<unsigned long long>(localNnz) + localRows - 1) / localRows;
        kCap = maxRowLen;
        if (static_cast<unsigned long long>(kCap) > capByMemory) {
            kCap = static_cast<index_t>(capByMemory > 0 ? capByMemory : 1);
        }
        if (kCap > maxRowLen) kCap = maxRowLen;
    }
    const bool useEll = useGpu && localRows > 0 && kCap > 0;

    std::vector<double> ellVal;
    std::vector<index_t> ellCols;
    if (useEll) {
        const size_t ellN = static_cast<size_t>(kCap) * localRows;
        ellVal.assign(ellN, 0.0);
        ellCols.assign(ellN, 0);

#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < localRows; ++i) {
            const index_t s = localRowDelimiters[i];
            const index_t len = localRowDelimiters[i + 1] - s;
            const index_t kmax = len < kCap ? len : kCap;
            for (index_t k = 0; k < kmax; ++k) {
                const size_t idx = static_cast<size_t>(k) * localRows + i;
                ellVal[idx] = localVal[s + k];
                ellCols[idx] = localCols[s + k];
            }
        }
    }

    // A handful of rows can be much longer than the rest (the generator fills
    // all remaining entries once it nears the target nnz). Those rows would
    // serialize an entire GPU grid, so OpenMP threads compute them on the host
    // concurrently with the asynchronous GPU kernel. Every row still follows
    // the original sequential accumulation order, hence the bits are identical.
    std::vector<index_t> longRows;
    if (useEll) {
        for (index_t i = 0; i < localRows; ++i) {
            const index_t len = localRowDelimiters[i + 1] - localRowDelimiters[i];
            if (len > kCap) longRows.push_back(i);
        }
    }
    std::vector<double> longOut(longRows.size());

    if (useGpu) {
        const size_t ellCount = useEll ? ellVal.size() : 0;
        const size_t ellValBytes = ellCount * sizeof(double);
        const size_t ellColBytes = ellCount * sizeof(index_t);
        const size_t csrRowBytes = static_cast<size_t>(localRows + 1) * sizeof(index_t);
        const size_t vecBytes = static_cast<size_t>(numRows) * sizeof(double);
        const size_t outBytes = static_cast<size_t>(localRows == 0 ? 1 : localRows) * sizeof(double);

        if (useEll) {
            cudaMalloc(reinterpret_cast<void**>(&d_ellVal), ellValBytes == 0 ? 1 : ellValBytes);
            cudaMalloc(reinterpret_cast<void**>(&d_ellCols), ellColBytes == 0 ? 1 : ellColBytes);
            if (ellCount > 0) {
                cudaMemcpy(d_ellVal, ellVal.data(), ellValBytes, cudaMemcpyHostToDevice);
                cudaMemcpy(d_ellCols, ellCols.data(), ellColBytes, cudaMemcpyHostToDevice);
            }
        }

        cudaMalloc(reinterpret_cast<void**>(&d_csrOffsets), csrRowBytes);
        cudaMalloc(reinterpret_cast<void**>(&d_vec), vecBytes);
        cudaMalloc(reinterpret_cast<void**>(&d_out), outBytes);
        cudaMemcpy(d_csrOffsets, localRowDelimiters.data(), csrRowBytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec, h_vec.data(), vecBytes, cudaMemcpyHostToDevice);
    }

    const int threadsPerBlock = 256;
    const int blocks = static_cast<int>((localRows + threadsPerBlock - 1) / threadsPerBlock);

    auto launchSpMV = [&]() {
        if (useEll) {
            spmvEllKernel<<<blocks, threadsPerBlock>>>(d_ellVal, d_ellCols, kCap,
                                                       d_csrOffsets, d_vec, localRows, d_out);
        }
    };

    auto runSpMV = [&]() {
        if (useGpu) {
            launchSpMV();
            if (!longRows.empty()) {
#pragma omp parallel for schedule(dynamic) if (longRows.size() >= 8)
                for (size_t p = 0; p < longRows.size(); ++p) {
                    const index_t row = longRows[p];
                    const index_t s = localRowDelimiters[row];
                    const index_t e = localRowDelimiters[row + 1];
                    double t = 0.0;
                    for (index_t j = s; j < e; ++j) {
                        t += localVal[j] * h_vec[localCols[j]];
                    }
                    longOut[p] = t;
                }
            }
        } else {
            // No CUDA device on this rank: use the OpenMP CPU implementation.
            if (localRows > 0) {
                spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                        h_vec.data(), localRows, localOut.data());
            }
        }
    };

    // ---------------------------------------------------------------------
    // Benchmark the SpMV iterations. Only the compute loop is timed, exactly
    // as in the original program; setup and data movement are outside it.
    // ---------------------------------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");

    if (useGpu && localRows > 0) {
        // Warm up the device / kernel before timing.
        launchSpMV();
        cudaDeviceSynchronize();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        runSpMV();
    }

    if (useGpu) {
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double localElapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMs = 0.0;
    MPI_Reduce(&localElapsedMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (useEll) {
        cudaMemcpy(localOut.data(), d_out, static_cast<size_t>(localRows) * sizeof(double),
                   cudaMemcpyDeviceToHost);
        for (size_t p = 0; p < longRows.size(); ++p) {
            localOut[longRows[p]] = longOut[p];
        }
    }

    // Gather the distributed output rows on the root rank.
    std::vector<int> outCounts(numProcs);
    std::vector<int> outDispls(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        outDispls[r] = static_cast<int>((static_cast<uint64_t>(numRows) * r) / numProcs);
        outCounts[r] = static_cast<int>((static_cast<uint64_t>(numRows) * (r + 1)) / numProcs) -
                       outDispls[r];
    }

    std::vector<double> h_out;
    if (isRoot) h_out.resize(numRows);

    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, outCounts.data(), outDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));

        // Calculate performance metrics
        const double seconds = elapsedMs / 1000.0;
        const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int resultCode = 0;

    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
                resultCode = 0;
            } else {
                printf("Validation: FAILED\n");
                resultCode = 1;
            }
        }
        MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (useGpu) {
        cudaFree(d_ellVal);
        cudaFree(d_ellCols);
        cudaFree(d_csrOffsets);
        cudaFree(d_vec);
        cudaFree(d_out);
    }

    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&nodeComm);
    }

    MPI_Finalize();
    return resultCode;
}
