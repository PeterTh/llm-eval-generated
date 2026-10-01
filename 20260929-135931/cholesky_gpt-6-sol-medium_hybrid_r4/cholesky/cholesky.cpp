#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t result, MPI_Comm comm) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        MPI_Abort(comm, 1);
    }
}
static void checkBlas(cublasStatus_t result, MPI_Comm comm) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(result));
        MPI_Abort(comm, 1);
    }
}

static int localColumns(int n, int block, int rank, int ranks) {
    int count = 0;
    for (int tile = rank; tile * block < n; tile += ranks)
        count += std::min(block, n - tile * block);
    return count;
}
static int globalRow(int localCol, int n, int block, int rank, int ranks) {
    (void)n;
    return ((localCol / block) * ranks + rank) * block + localCol % block;
}

__global__ static void solvePanel(double* a, const double* diag, int n,
                                  int localN, int block, int rank, int ranks,
                                  int k, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= localN || ((col / block) * ranks + rank) * block + col % block < k + width) return;
    double* row = a + static_cast<size_t>(col) * n + k;
    for (int j = 0; j < width; ++j) {
        double value = row[j];
        for (int p = 0; p < j; ++p) value -= diag[j + p * block] * row[p];
        row[j] = value / diag[j + j * block];
    }
}
__global__ static void packPanel(const double* a, double* packed, int n,
                                 int localN, int k, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < localN * width)
        packed[index] = a[static_cast<size_t>(index / width) * n + k + index % width];
}
__global__ static void packResult(const double* a, double* packed, int n,
                                  int localN, int block, int rank, int ranks) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(n) * localN) return;
    int col = index / n;
    int j = ((col / block) * ranks + rank) * block + col % block;
    int i = index % n;
    packed[index] = i <= j ? a[static_cast<size_t>(col) * n + i] : 0.0;
}

// Generate exactly the same seeded matrix as the sequential benchmark.
static void generatePositiveDefiniteMatrix(std::vector<double>& a, int n) {
    std::vector<double> b(static_cast<size_t>(n) * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += b[static_cast<size_t>(i) * n + k] * b[static_cast<size_t>(j) * n + k];
            a[static_cast<size_t>(i) * n + j] = sum + (i == j ? n : 0);
        }
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& orig, int n) {
    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k <= std::min(i, j); ++k)
                sum += l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
            double error = fabs(sum - orig[static_cast<size_t>(i) * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(orig[static_cast<size_t>(i) * n + j]) + 1e-10));
        }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            long parsed = strtol(argv[++i], &end, 10);
            if (*end || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
                if (!rank) fprintf(stderr, "Invalid matrix size\n");
                MPI_Abort(comm, 1);
            }
            n = static_cast<int>(parsed);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI counts are int, and each local device column has n elements.
    if (static_cast<long long>(n) * n > std::numeric_limits<int>::max()) {
        if (!rank) fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Abort(comm, 1);
    }
    MPI_Comm shared;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank, devices = 0;
    MPI_Comm_rank(shared, &localRank);
    checkCuda(cudaGetDeviceCount(&devices), comm);
    if (!devices) { fprintf(stderr, "Rank %d: CUDA device required\n", rank); MPI_Abort(comm, 1); }
    checkCuda(cudaSetDevice(localRank % devices), comm);
    MPI_Comm_free(&shared);

    constexpr int block = 64;
    int localN = localColumns(n, block, rank, ranks);
    std::vector<int> cols(ranks), counts(ranks), offsets(ranks);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        cols[r] = localColumns(n, block, r, ranks);
        counts[r] = cols[r] * n;
        offsets[r] = total;
        total += counts[r];
    }
    std::vector<double> a, original, send;
    if (!rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
        a.resize(static_cast<size_t>(n) * n);
        generatePositiveDefiniteMatrix(a, n);
        if (validate) original = a;
        send.resize(a.size());
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r)
            for (int col = 0; col < cols[r]; ++col) {
                int row = globalRow(col, n, block, r, ranks);
                for (int i = 0; i < n; ++i)
                    send[static_cast<size_t>(offsets[r]) + static_cast<size_t>(col) * n + i] = a[static_cast<size_t>(row) * n + i];
            }
        printf("Computing Cholesky decomposition...\n");
    }
    std::vector<double> hostLocal(static_cast<size_t>(localN) * n);
    MPI_Scatterv(rank == 0 ? send.data() : nullptr, counts.data(), offsets.data(), MPI_DOUBLE,
                 hostLocal.data(), counts[rank], MPI_DOUBLE, 0, comm);
    send.clear();
    double *deviceA = nullptr, *devicePanel = nullptr, *devicePacked = nullptr, *deviceDiag = nullptr;
    checkCuda(cudaMalloc(&deviceA, std::max<size_t>(1, hostLocal.size()) * sizeof(double)), comm);
    checkCuda(cudaMalloc(&devicePanel, static_cast<size_t>(n) * block * sizeof(double)), comm);
    checkCuda(cudaMalloc(&devicePacked, std::max<size_t>(1, static_cast<size_t>(localN) * n) * sizeof(double)), comm);
    checkCuda(cudaMalloc(&deviceDiag, block * block * sizeof(double)), comm);
    checkCuda(cudaMemcpy(deviceA, hostLocal.data(), hostLocal.size() * sizeof(double), cudaMemcpyHostToDevice), comm);
    cublasHandle_t blas;
    checkBlas(cublasCreate(&blas), comm);
    // Trigger CUDA and cuBLAS kernel initialization before the timed factorization.
    const double one = 1.0, zero = 0.0;
    checkCuda(cudaMemset(deviceDiag, 0, sizeof(double)), comm);
    checkCuda(cudaMemcpy(deviceDiag, &one, sizeof(double), cudaMemcpyHostToDevice), comm);
    checkCuda(cudaMemset(devicePacked, 0, sizeof(double)), comm);
    checkBlas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                          &zero, deviceDiag, 1, deviceDiag, 1,
                          &one, devicePacked, 1), comm);
    solvePanel<<<1, 1>>>(deviceA, deviceDiag, n, 0, block, rank, ranks, 0, 0);
    packPanel<<<1, 1>>>(deviceA, devicePacked, n, 0, 0, 1);
    checkCuda(cudaDeviceSynchronize(), comm);
    std::vector<double> diagonal(block * block), diagonalOutput(block * block), localPanel(static_cast<size_t>(localN) * block);
    std::vector<double> received(static_cast<size_t>(n) * block), panel(static_cast<size_t>(n) * block);
    std::vector<int> panelCounts(ranks), panelOffsets(ranks);
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    bool success = true;
    for (int k = 0; k < n; k += block) {
        int width = std::min(block, n - k);
        int tile = k / block, owner = tile % ranks;
        if (rank == owner) {
            int first = (tile / ranks) * block;
            checkCuda(cudaMemcpy2D(diagonal.data(), block * sizeof(double),
                                   deviceA + static_cast<size_t>(first) * n + k, n * sizeof(double),
                                   width * sizeof(double), width, cudaMemcpyDeviceToHost), comm);
            for (int j = 0; j < width; ++j) {
                double value = diagonal[j + j * block];
                for (int p = 0; p < j; ++p) value -= diagonal[j + p * block] * diagonal[j + p * block];
                if (!(value > 0.0)) { success = false; break; }
                diagonal[j + j * block] = sqrt(value);
                for (int i = j + 1; i < width; ++i) {
                    double x = diagonal[i + j * block];
                    for (int p = 0; p < j; ++p) x -= diagonal[i + p * block] * diagonal[j + p * block];
                    diagonal[i + j * block] = x / diagonal[j + j * block];
                }
            }
            if (success) {
                for (int j = 0; j < width; ++j)
                    for (int i = 0; i < width; ++i)
                        diagonalOutput[i + j * block] = j >= i ? diagonal[j + i * block] : 0.0;
                checkCuda(cudaMemcpy2D(deviceA + static_cast<size_t>(first) * n + k, n * sizeof(double),
                                       diagonalOutput.data(), block * sizeof(double), width * sizeof(double), width,
                                       cudaMemcpyHostToDevice), comm);
            }
        }
        int ok = success ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, owner, comm);
        if (!ok) { success = false; if (!rank) printf("Error: Matrix is not positive definite at diagonal element %d\n", k); break; }
        MPI_Bcast(diagonal.data(), block * block, MPI_DOUBLE, owner, comm);
        checkCuda(cudaMemcpy(deviceDiag, diagonal.data(), block * block * sizeof(double), cudaMemcpyHostToDevice), comm);
        int firstTrailing = 0;
        while (firstTrailing < localN && globalRow(firstTrailing, n, block, rank, ranks) < k + width) ++firstTrailing;
        if (localN) {
            solvePanel<<<(localN + 127) / 128, 128>>>(deviceA, deviceDiag, n, localN, block, rank, ranks, k, width);
            packPanel<<<(localN * width + 255) / 256, 256>>>(deviceA, devicePacked, n, localN, k, width);
            checkCuda(cudaGetLastError(), comm);
            checkCuda(cudaMemcpy(localPanel.data(), devicePacked, static_cast<size_t>(localN) * width * sizeof(double), cudaMemcpyDeviceToHost), comm);
        }
        int panelTotal = 0;
        for (int r = 0; r < ranks; ++r) {
            panelCounts[r] = cols[r] * width;
            panelOffsets[r] = panelTotal;
            panelTotal += panelCounts[r];
        }
        MPI_Allgatherv(localPanel.data(), localN * width, MPI_DOUBLE, received.data(),
                       panelCounts.data(), panelOffsets.data(), MPI_DOUBLE, comm);
        #pragma omp parallel for schedule(static) if(n >= 2048)
        for (int r = 0; r < ranks; ++r)
            for (int col = 0; col < cols[r]; ++col) {
                int row = globalRow(col, n, block, r, ranks);
                for (int q = 0; q < width; ++q)
                    panel[static_cast<size_t>(q) * n + row] = received[panelOffsets[r] + col * width + q];
            }
        checkCuda(cudaMemcpy(devicePanel, panel.data(), static_cast<size_t>(n) * width * sizeof(double), cudaMemcpyHostToDevice), comm);
        if (firstTrailing < localN && k + width < n) {
            const double alpha = -1.0, beta = 1.0;
            checkBlas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n - k - width, localN - firstTrailing, width,
                                  &alpha, devicePanel + k + width, n,
                                  deviceA + static_cast<size_t>(firstTrailing) * n + k, n,
                                  &beta, deviceA + static_cast<size_t>(firstTrailing) * n + k + width, n), comm);
        }
    }
    checkCuda(cudaDeviceSynchronize(), comm);
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!success) { if (!rank) printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (!rank) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Performance: %.3f GFLOPS\n", (static_cast<double>(n) * n * n / 3.0) / maxElapsed / 1e9);
    }
    if (validate || printResults) {
        if (localN) {
            packResult<<<(static_cast<size_t>(localN) * n + 255) / 256, 256>>>(deviceA, devicePacked, n, localN, block, rank, ranks);
            checkCuda(cudaGetLastError(), comm);
            checkCuda(cudaMemcpy(hostLocal.data(), devicePacked, hostLocal.size() * sizeof(double), cudaMemcpyDeviceToHost), comm);
        }
        if (!rank) send.resize(static_cast<size_t>(n) * n);
        MPI_Gatherv(hostLocal.data(), counts[rank], MPI_DOUBLE, rank == 0 ? send.data() : nullptr,
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        if (!rank) {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < ranks; ++r)
                for (int col = 0; col < cols[r]; ++col) {
                    int row = globalRow(col, n, block, r, ranks);
                    memcpy(a.data() + static_cast<size_t>(row) * n,
                           send.data() + static_cast<size_t>(offsets[r]) + static_cast<size_t>(col) * n,
                           static_cast<size_t>(n) * sizeof(double));
                }
            if (printResults) print_results(a, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(a, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                success = valid;
            }
        }
    }
    int result = success ? 0 : 1;
    MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    cublasDestroy(blas);
    cudaFree(deviceDiag);
    cudaFree(devicePacked);
    cudaFree(devicePanel);
    cudaFree(deviceA);
    MPI_Finalize();
    return result;
}
