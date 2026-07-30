#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Warp-per-row CSR SpMV kernel: each warp computes one row
__global__ void spmv_csr_warp_kernel(
    const double* __restrict__ val,
    const index_t* __restrict__ cols,
    const index_t* __restrict__ rowDelimiters,
    const double* __restrict__ vec,
    const index_t localRows,
    double* __restrict__ out)
{
    const index_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const index_t laneId = threadIdx.x & 31u;

    if (warpId >= localRows) return;

    const index_t start = rowDelimiters[warpId];
    const index_t end = rowDelimiters[warpId + 1];

    double sum = 0.0;
    for (index_t j = start + laneId; j < end; j += 32) {
        sum += val[j] * vec[cols[j]];
    }

    // Warp-level reduction via shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId == 0) {
        out[warpId] = sum;
    }
}

void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

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

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    int failures = 0;
    #pragma omp parallel for reduction(+:failures)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool failed = false;
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                failed = true;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                failed = true;
            }
        }
        if (failed) failures++;
    }
    return failures == 0;
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // GPU setup: assign one GPU per MPI rank (round-robin)
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    if (numGPUs == 0) {
        fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpuId = rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    // Parse command line arguments on rank 0
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResultsFlag = false;

    if (rank == 0) {
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
                printResultsFlag = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 0);
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast parameters to all ranks
    uint32_t bcastParams[5];
    if (rank == 0) {
        bcastParams[0] = numRows;
        bcastParams[1] = sparsity;
        bcastParams[2] = iterations;
        bcastParams[3] = validate ? 1u : 0u;
        bcastParams[4] = printResultsFlag ? 1u : 0u;
    }
    MPI_Bcast(bcastParams, 5, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    numRows = bcastParams[0];
    sparsity = bcastParams[1];
    iterations = bcastParams[2];
    validate = (bcastParams[3] != 0);
    printResultsFlag = (bcastParams[4] != 0);

    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t nItems = (numRows * numRows) / sparsity;

    // Compute row distribution across MPI ranks
    index_t rowsPerRank = numRows / numRanks;
    index_t rem = numRows % numRanks;

    std::vector<index_t> rowStart(numRanks), rowEnd(numRanks);
    for (int r = 0; r < numRanks; r++) {
        rowStart[r] = static_cast<index_t>(r) * rowsPerRank + std::min(static_cast<index_t>(r), rem);
        rowEnd[r] = rowStart[r] + rowsPerRank + (static_cast<index_t>(r) < rem ? 1 : 0);
    }

    index_t myRowStart = rowStart[rank];
    index_t myRowEnd = rowEnd[rank];
    index_t myLocalRows = myRowEnd - myRowStart;

    // Allocate vector on all ranks (needed for Bcast)
    std::vector<double> h_vec(numRows);

    // Generate data on rank 0
    std::vector<double> h_val;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs available: %d\n", numRanks, numGPUs);
        fflush(stdout);

        printf("Initializing data structures...\n");
        fflush(stdout);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            fflush(stdout);
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Compute NNZ distribution for Scatterv
    std::vector<int> nnzCounts(numRanks, 0), nnzDispls(numRanks, 0);
    if (rank == 0) {
        for (int r = 0; r < numRanks; r++) {
            nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
            nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowEnd[r]]) - nnzDispls[r];
        }
    }

    // Distribute nnz counts and displacements to each rank
    int myLocalNnz = 0, myNnzStart = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &myLocalNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzDispls.data(), 1, MPI_INT, &myNnzStart, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local buffers
    std::vector<double> local_val(myLocalNnz);
    std::vector<index_t> local_cols(myLocalNnz);
    std::vector<index_t> local_rowDelimiters(myLocalRows + 1);
    std::vector<double> local_out(myLocalRows, 0.0);

    // Scatter val and cols arrays
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 local_val.data(), myLocalNnz, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 local_cols.data(), myLocalNnz, MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Distribute rowDelimiters (handles overlapping boundaries via Send/Recv)
    if (rank == 0) {
        for (index_t i = 0; i <= myLocalRows; i++) {
            local_rowDelimiters[i] = h_rowDelimiters[myRowStart + i] - static_cast<index_t>(myNnzStart);
        }
        for (int r = 1; r < numRanks; r++) {
            index_t count = rowEnd[r] - rowStart[r] + 1;
            std::vector<index_t> buf(count);
            index_t adj = h_rowDelimiters[rowStart[r]];
            for (index_t i = 0; i < count; i++) {
                buf[i] = h_rowDelimiters[rowStart[r] + i] - adj;
            }
            MPI_Send(buf.data(), static_cast<int>(count), MPI_UINT32_T, r, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_rowDelimiters.data(), static_cast<int>(myLocalRows + 1),
                 MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast dense vector to all ranks
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- GPU memory allocation ----
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelims = nullptr;

    // Allocate device memory for non-zero values and column indices
    if (myLocalNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, static_cast<size_t>(myLocalNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, static_cast<size_t>(myLocalNnz) * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelims, static_cast<size_t>(myLocalRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)));
    if (myLocalRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(myLocalRows) * sizeof(double)));
    }

    // Copy data from host to device
    if (myLocalNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, local_val.data(), myLocalNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), myLocalNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelims, local_rowDelimiters.data(),
                          (myLocalRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Timed SpMV computation on GPU ----
    if (rank == 0) {
        printf("Computing SpMV...\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (myLocalRows > 0 && myLocalNnz > 0) {
        const int threadsPerBlock = 256;
        const int warpsPerBlock = threadsPerBlock / 32;
        const int numBlocks = (static_cast<int>(myLocalRows) + warpsPerBlock - 1) / warpsPerBlock;

        for (index_t iter = 0; iter < iterations; ++iter) {
            spmv_csr_warp_kernel<<<numBlocks, threadsPerBlock>>>(
                d_val, d_cols, d_rowDelims, d_vec,
                myLocalRows, d_out);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Device-to-host copy
    if (myLocalRows > 0) {
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, myLocalRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0
    std::vector<double> h_out;
    std::vector<int> outCounts(numRanks, 0), outDispls(numRanks, 0);
    if (rank == 0) {
        h_out.resize(numRows);
        for (int r = 0; r < numRanks; r++) {
            outCounts[r] = static_cast<int>(rowEnd[r] - rowStart[r]);
            outDispls[r] = static_cast<int>(rowStart[r]);
        }
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(myLocalRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                outCounts.data(), outDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Output and validation on rank 0
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        printf("Computation time: %ld ms\n", duration.count());

        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResultsFlag) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Cleanup GPU memory
    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDelims) CUDA_CHECK(cudaFree(d_rowDelims));
    if (d_vec) CUDA_CHECK(cudaFree(d_vec));
    if (d_out) CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
