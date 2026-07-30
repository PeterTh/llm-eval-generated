#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err_ = call;                                                 \
        if (err_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

__global__ void computeDiagonalKernel(const double *A, const size_t n,
                                       const size_t k, double *result) {
    if (threadIdx.x == 0) {
        const double val = A[k * n + k];
        *result = (val <= 0.0) ? -1.0 : sqrt(val);
    }
}

__global__ void computeOffDiagKernel(double *A, const size_t n, const size_t k,
                                      const double lk_k, const size_t row_start,
                                      const size_t row_end) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = row_start + idx;
    if (i >= row_end) return;
    A[i * n + k] = A[i * n + k] / lk_k;
}

__global__ void updateTrailingKernel(double *A, const double *x,
                                      const size_t n, const size_t k,
                                      const size_t row_start,
                                      const size_t row_end) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = row_start + idx;
    if (i >= row_end) return;
    const double li_k = x[i - (k + 1)];
    const size_t row_i = i * n;
    for (size_t j = k + 1; j <= i; ++j) {
        A[row_i + j] -= li_k * x[j - (k + 1)];
    }
}

__global__ void zeroUpperKernel(double *A, const size_t n, const size_t i,
                                 const size_t start_j) {
    const size_t j = start_j + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        A[i * n + j] = 0.0;
    }
}

void generatePositiveDefiniteMatrix(std::vector<double> &A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double> &L,
                       const std::vector<double> &A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    double maxError = 0.0;
    double relError = 0.0;
#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (error > maxError) maxError = error;
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        if (rel > relError) relError = rel;
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

bool choleskyDecompositionParallel(std::vector<double> &localA, const size_t n,
                                    const size_t localRowStart,
                                    const size_t localRowEnd,
                                    const size_t localRows,
                                    const std::vector<int> &recvCounts,
                                    const std::vector<int> &displs,
                                    const int numRanks, MPI_Comm comm) {
    int myRank;
    MPI_Comm_rank(comm, &myRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    const int device = (deviceCount > 0) ? (myRank % deviceCount) : 0;
    CUDA_CHECK(cudaSetDevice(device));


    double *d_A = nullptr;
    double *d_x = nullptr;
    double *d_diagResult = nullptr;

    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, localRows * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_diagResult, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_A, localA.data(),
                               localRows * n * sizeof(double),
                               cudaMemcpyHostToDevice));
    }

    std::vector<double> fullColK(n, 0.0);

    for (size_t k = 0; k < n; ++k) {
        int owningRank = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t rs = (size_t)displs[r] / n;
            const size_t re = rs + (size_t)recvCounts[r] / n;
            if (k >= rs && k < re) {
                owningRank = r;
                break;
            }
        }

        // Step 1: Diagonal
        double diagVal = 0.0;
        if (myRank == owningRank && localRows > 0) {
            computeDiagonalKernel<<<1, 1>>>(d_A, n, k, d_diagResult);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&diagVal, d_diagResult, sizeof(double),
                                   cudaMemcpyDeviceToHost));
            if (diagVal < 0.0) {
                if (myRank == 0) printf("Error: Not PD at k=%zu\n", k);
                return false;
            }
            const size_t localK = k - localRowStart;
            CUDA_CHECK(cudaMemcpy(d_A + localK * n + k, &diagVal,
                                   sizeof(double), cudaMemcpyHostToDevice));
            localA[localK * n + k] = diagVal;
        }

        // Broadcast diagonal
        if (myRank == owningRank) {
            for (int r = 0; r < numRanks; ++r) {
                if (r != myRank) {
                    MPI_Send(&diagVal, 1, MPI_DOUBLE, r, (int)k, comm);
                }
            }
        } else {
            MPI_Recv(&diagVal, 1, MPI_DOUBLE, owningRank, (int)k, comm,
                      MPI_STATUS_IGNORE);
        }

        // Step 2: Off-diagonal
        const size_t offDiagStart = k + 1;
        if (offDiagStart < n && localRows > 0) {
            const size_t localOffStart = std::max(offDiagStart, localRowStart);
            const size_t localOffEnd = std::min(n, localRowEnd);
            const size_t count = localOffEnd - localOffStart;
            if (count > 0) {
                const int blk = 256;
                const int nblk = (int)((count + blk - 1) / blk);
                computeOffDiagKernel<<<nblk, blk>>>(
                    d_A, n, k, diagVal, localOffStart, localOffEnd);
                CUDA_CHECK(cudaDeviceSynchronize());
                const size_t hostOffset = (localOffStart - localRowStart) * n;
                const size_t devOffset = (localOffStart - localRowStart) * n;
                const size_t copySize =
                    (localOffEnd - localOffStart) * n * sizeof(double);
                CUDA_CHECK(cudaMemcpy(localA.data() + hostOffset,
                                       d_A + devOffset, copySize,
                                       cudaMemcpyDeviceToHost));
            }
        }

        // Step 3: Column exchange
        std::vector<double> localColPart(localRows);
        if (localRows > 0) {
            for (size_t i = 0; i < localRows; ++i) {
                localColPart[i] = localA[i * n + k];
            }
        }

        if (myRank == 0) {
            const size_t rs0 = (size_t)displs[0] / n;
            for (size_t i = 0; i < localRows; ++i) {
                fullColK[rs0 + i] = localA[i * n + k];
            }
            std::vector<MPI_Request> reqs(numRanks - 1);
            for (int r = 1; r < numRanks; ++r) {
                const size_t rs = (size_t)displs[r] / n;
                const int cnt = recvCounts[r] / (int)n;
                MPI_Irecv(fullColK.data() + rs, cnt, MPI_DOUBLE, r, (int)k, comm,
                           &reqs[r - 1]);
            }
            MPI_Waitall(numRanks - 1, reqs.data(), MPI_STATUSES_IGNORE);
            for (int r = 1; r < numRanks; ++r) {
                MPI_Isend(fullColK.data(), (int)n, MPI_DOUBLE, r, (int)k, comm,
                           &reqs[r - 1]);
            }
            MPI_Waitall(numRanks - 1, reqs.data(), MPI_STATUSES_IGNORE);
        } else {
            MPI_Request req;
            MPI_Isend(localColPart.data(), (int)localRows, MPI_DOUBLE, 0, (int)k,
                       comm, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            MPI_Irecv(fullColK.data(), (int)n, MPI_DOUBLE, 0, (int)k, comm, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }

        // Step 4: Update
        if (k + 1 < n && localRows > 0) {
            const size_t updateStart = std::max(k + 1, localRowStart);
            const size_t updateEnd = std::min(n, localRowEnd);
            const size_t updateCount = updateEnd - updateStart;
            if (updateCount > 0) {
                const size_t vecLen = n - k - 1;
                CUDA_CHECK(cudaMemcpy(d_x, fullColK.data() + k + 1,
                                       vecLen * sizeof(double),
                                       cudaMemcpyHostToDevice));
                const int blk = 256;
                const int nblk = (int)((updateCount + blk - 1) / blk);
                updateTrailingKernel<<<nblk, blk>>>(
                    d_A, d_x, n, k, updateStart, updateEnd);
                CUDA_CHECK(cudaDeviceSynchronize());
                const size_t hostOffset = (updateStart - localRowStart) * n;
                const size_t devOffset = (updateStart - localRowStart) * n;
                const size_t copySize =
                    (updateEnd - updateStart) * n * sizeof(double);
                CUDA_CHECK(cudaMemcpy(localA.data() + hostOffset,
                                       d_A + devOffset, copySize,
                                       cudaMemcpyDeviceToHost));
            }
        }
    }

    // Zero upper triangle
    if (localRows > 0) {
        for (size_t i = localRowStart; i < localRowEnd; ++i) {
            const size_t zeroStart = i + 1;
            if (zeroStart < n) {
                const size_t count = n - zeroStart;
                const int blk = 256;
                const int nblk = (int)((count + blk - 1) / blk);
                zeroUpperKernel<<<nblk, blk>>>(d_A, n, i, zeroStart);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(localA.data(), d_A,
                               localRows * n * sizeof(double),
                               cudaMemcpyDeviceToHost));
    }

    if (localRows > 0) {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_diagResult));
    }

    return true;
}

int main(int argc, char **argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        }
    }

    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices: %d\n", deviceCount);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A;
    if (rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const size_t baseRows = n / numRanks;
    const size_t extraRows = n % numRanks;

    std::vector<int> sendCounts(numRanks);
    std::vector<int> displs(numRanks);
    size_t cumDisp = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t rows = r < (int)extraRows ? baseRows + 1 : baseRows;
        sendCounts[r] = (int)(rows * n);
        displs[r] = (int)cumDisp;
        cumDisp += rows * n;
    }

    const size_t localRows = rank < (int)extraRows ? baseRows + 1 : baseRows;
    const size_t localRowStart = (size_t)displs[rank] / n;
    const size_t localRowEnd = localRowStart + localRows;

    std::vector<double> localA(localRows * n);
    MPI_Scatterv(A.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                  localA.data(), (int)(localRows * n), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

    std::vector<double> A_orig;
    if (validate && rank == 0) {
        A_orig = A;
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecompositionParallel(
        localA, n, localRowStart, localRowEnd, localRows,
        sendCounts, displs, numRanks, MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    MPI_Barrier(MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) A.resize(n * n);
    MPI_Gatherv(localA.data(), (int)(localRows * n), MPI_DOUBLE,
                 A.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / ((double)duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
