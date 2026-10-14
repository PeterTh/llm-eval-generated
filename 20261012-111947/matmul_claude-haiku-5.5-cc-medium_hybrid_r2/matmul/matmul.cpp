#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Abort the whole job on any CUDA failure
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// CUDA tiling: each thread block computes a BM x BN tile of C in steps of BK over k,
// with each thread holding a TM x TN register block of C
constexpr size_t BM = 64;
constexpr size_t BN = 64;
constexpr size_t BK = 16;
constexpr size_t TM = 4;
constexpr size_t TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);

constexpr size_t GPU_CHUNK = 128;  // rows of C handed out per dynamic work item

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// First row of the block owned by rank r of P (rows are split as evenly as possible)
constexpr size_t rowBegin(const size_t N, const size_t r, const size_t P) noexcept {
    return N * r / P;
}

// Fill the rows [rowBegin, rowEnd) of the global matrix into a local block
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// C[rows x N] = A[rows x N] * B[N x N]. Each C element accumulates its k-terms in
// increasing k order, exactly as the sequential reference does. The explicit
// __dmul_rn/__dadd_rn keep the same rounding as the host code (no FMA contraction).
__global__ void __launch_bounds__(THREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BM][BK];
    __shared__ double Bs[BK][BN];

    const size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const size_t rowBase = blockIdx.y * BM;
    const size_t colBase = blockIdx.x * BN;
    double acc[TM][TN] = {};

    for (size_t kt = 0; kt < N; kt += BK) {
        const size_t kmax = (N - kt < BK) ? N - kt : BK;

        for (size_t idx = tid; idx < BM * BK; idx += THREADS) {
            const size_t m = idx / BK;
            const size_t kk = idx % BK;
            const size_t gr = rowBase + m;
            As[m][kk] = (gr < rows && kk < kmax) ? A[gr * N + kt + kk] : 0.0;
        }
        for (size_t idx = tid; idx < BK * BN; idx += THREADS) {
            const size_t kk = idx / BN;
            const size_t n = idx % BN;
            const size_t gc = colBase + n;
            Bs[kk][n] = (kk < kmax && gc < N) ? B[(kt + kk) * N + gc] : 0.0;
        }
        __syncthreads();

        // Each C element still accumulates its k-terms in increasing k order
        for (size_t kk = 0; kk < kmax; ++kk) {
            double a[TM];
            double b[TN];
            for (size_t r = 0; r < TM; ++r) a[r] = As[threadIdx.y * TM + r][kk];
            for (size_t c = 0; c < TN; ++c) b[c] = Bs[kk][threadIdx.x * TN + c];
            for (size_t r = 0; r < TM; ++r) {
                for (size_t c = 0; c < TN; ++c) {
                    acc[r][c] = __dadd_rn(acc[r][c], __dmul_rn(a[r], b[c]));
                }
            }
        }
        __syncthreads();
    }

    for (size_t r = 0; r < TM; ++r) {
        const size_t gr = rowBase + threadIdx.y * TM + r;
        if (gr >= rows) continue;
        for (size_t c = 0; c < TN; ++c) {
            const size_t gc = colBase + threadIdx.x * TN + c;
            if (gc < N) C[gr * N + gc] = acc[r][c];
        }
    }
}

// Drives one GPU: repeatedly claims a chunk of local rows and computes it on the device
void gpuWorker(const int device, const double* Al, const double* B, double* Cl,
               const size_t localRows, const size_t N, std::atomic<size_t>& nextRow) {
    CUDA_CHECK(cudaSetDevice(device));

    const size_t chunkMax = std::min(GPU_CHUNK, localRows);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, chunkMax * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, chunkMax * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(BN / TN, BM / TM);
    for (size_t begin = nextRow.fetch_add(GPU_CHUNK); begin < localRows; begin = nextRow.fetch_add(GPU_CHUNK)) {
        const size_t rows = std::min(GPU_CHUNK, localRows - begin);
        CUDA_CHECK(cudaMemcpy(dA, Al + begin * N, rows * N * sizeof(double), cudaMemcpyHostToDevice));

        const dim3 grid((N + BN - 1) / BN, (rows + BM - 1) / BM);
        matmulKernel<<<grid, block>>>(dA, dB, dC, rows, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(Cl + begin * N, dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// Computes this rank's rows of C on all of its GPUs, one OpenMP thread per GPU
void computeLocalRows(const std::vector<int>& devices, const double* Al, const double* B,
                      double* Cl, const size_t localRows, const size_t N) {
    std::atomic<size_t> nextRow{0};
    const int numWorkers = static_cast<int>(devices.size());

    #pragma omp parallel for schedule(static, 1) num_threads(numWorkers)
    for (int d = 0; d < numWorkers; ++d) {
        gpuWorker(devices[d], Al, B, Cl, localRows, N, nextRow);
    }
}

// GPUs are split round-robin across the ranks of a node. When there are more ranks than
// GPUs, ranks share a device.
std::vector<int> assignDevices(const int nodeRank, const int nodeSize, const int numDevices) {
    std::vector<int> devices;
    if (numDevices >= nodeSize) {
        for (int d = nodeRank; d < numDevices; d += nodeSize) {
            devices.push_back(d);
        }
    } else {
        devices.push_back(nodeRank % numDevices);
    }
    return devices;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Simple validation: compute a single element and compare.
// A is regenerated from its generator, since the full A matrix is not kept on the root rank.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select the GPUs this rank drives
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const std::vector<int> devices = assignDevices(nodeRank, nodeSize, numDevices);
    // Create the CUDA contexts now so that their setup cost is not part of the timed computation
    for (const int device : devices) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(nullptr));
    }

    // Row block owned by this rank
    const size_t rowStart = rowBegin(N, rank, size);
    const size_t rowEnd = rowBegin(N, rank + 1, size);
    const size_t localRows = rowEnd - rowStart;

    // Allocate matrices: B is needed in full on every rank, A and C only for the local rows
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);

    // Pin the host matrices so that host-device copies run as DMA transfers
    if (!A.empty()) CUDA_CHECK(cudaHostRegister(A.data(), A.size() * sizeof(double), cudaHostRegisterDefault));
    if (!B.empty()) CUDA_CHECK(cudaHostRegister(B.data(), B.size() * sizeof(double), cudaHostRegisterDefault));
    if (!C.empty()) CUDA_CHECK(cudaHostRegister(C.data(), C.size() * sizeof(double), cudaHostRegisterDefault));

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    initMatrixRows(A, N, rowStart, rowEnd);
    initMatrixRows(B, N, 0, N);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    computeLocalRows(devices, A.data(), B.data(), C.data(), localRows, N);

    auto end = std::chrono::high_resolution_clock::now();

    // The slowest rank determines the time
    double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(maxElapsed));

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C on the root rank when it is needed for output or validation
    std::vector<double> Cfull;
    if (printResults || validate) {
        if (root) Cfull.resize(N * N);

        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t b = rowBegin(N, r, size);
            const size_t e = rowBegin(N, r + 1, size);
            counts[r] = static_cast<int>((e - b) * N);
            displs[r] = static_cast<int>(b * N);
        }
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    Cfull.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (root && printResults) {
        print_results(Cfull, "MatrixC");
    }

    // Validation
    int status = 0;
    if (root && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(B, Cfull, N);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    // All ranks exit with the same status
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!A.empty()) CUDA_CHECK(cudaHostUnregister(A.data()));
    if (!B.empty()) CUDA_CHECK(cudaHostUnregister(B.data()));
    if (!C.empty()) CUDA_CHECK(cudaHostUnregister(C.data()));
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return status;
}
