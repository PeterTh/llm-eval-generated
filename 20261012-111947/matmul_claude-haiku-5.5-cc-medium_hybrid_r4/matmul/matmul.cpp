#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill rows [rowBegin, rowEnd) of an N-column matrix. `mat` points at row rowBegin,
// so a rank only needs storage for its own rows.
void initRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// GPU tiling: each thread block computes a TILE x TILE patch of C with
// THREADS x THREADS threads, each thread owning MICRO x MICRO outputs.
constexpr int TILE = 64;
constexpr int THREADS = 16;
constexpr int MICRO = TILE / THREADS;
constexpr int BK = 16;  // k-depth staged in shared memory per step

// C = A * B for `rows` rows of A (row-major, N columns) and full B (N x N).
// Every element accumulates over k in ascending order, rounding each product
// and each sum separately (no FMA contraction), which matches the sequential
// host loop bit for bit. Out-of-range staging entries are zero, which leaves
// the accumulators unchanged.
__global__ void __launch_bounds__(THREADS * THREADS)
gemmKernel(const double* __restrict__ A, const double* __restrict__ B,
           double* __restrict__ C, const int rows, const int N) {
    __shared__ double As[BK][TILE];
    __shared__ double Bs[BK][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS + tx;
    const int rowBase = blockIdx.y * TILE;
    const int colBase = blockIdx.x * TILE;

    double acc[MICRO][MICRO] = {};

    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int idx = tid; idx < TILE * BK; idx += THREADS * THREADS) {
            const int m = idx / BK;
            const int kk = idx % BK;
            const int gr = rowBase + m;
            const int gk = k0 + kk;
            As[kk][m] = (gr < rows && gk < N) ? A[(size_t)gr * N + gk] : 0.0;
        }
        for (int idx = tid; idx < BK * TILE; idx += THREADS * THREADS) {
            const int kk = idx / TILE;
            const int n = idx % TILE;
            const int gk = k0 + kk;
            const int gc = colBase + n;
            Bs[kk][n] = (gk < N && gc < N) ? B[(size_t)gk * N + gc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[MICRO];
            double b[MICRO];
            #pragma unroll
            for (int r = 0; r < MICRO; ++r) a[r] = As[kk][r * THREADS + ty];
            #pragma unroll
            for (int c = 0; c < MICRO; ++c) b[c] = Bs[kk][c * THREADS + tx];
            #pragma unroll
            for (int r = 0; r < MICRO; ++r) {
                #pragma unroll
                for (int c = 0; c < MICRO; ++c) {
                    acc[r][c] = __dadd_rn(acc[r][c], __dmul_rn(a[r], b[c]));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < MICRO; ++r) {
        #pragma unroll
        for (int c = 0; c < MICRO; ++c) {
            const int gr = rowBase + r * THREADS + ty;
            const int gc = colBase + c * THREADS + tx;
            if (gr < rows && gc < N) C[(size_t)gr * N + gc] = acc[r][c];
        }
    }
}

// Computes `rows` rows of C on GPU `device`. hA and hC are this device's row blocks.
void gemmOnDevice(int device, const double* hA, const double* hB, double* hC,
                  const size_t rows, const size_t N) {
    CUDA_CHECK(cudaSetDevice(device));
    if (rows == 0) return;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, hA, rows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, N * N * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(THREADS, THREADS);
    const dim3 grid(static_cast<unsigned int>((N + TILE - 1) / TILE),
                    static_cast<unsigned int>((rows + TILE - 1) / TILE));
    gemmKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(rows), static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    // Synchronous device-to-host copy also waits for the kernel to finish.
    CUDA_CHECK(cudaMemcpy(hC, dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// Page-locked host memory so host<->device copies run at full PCIe speed.
double* allocPinned(const size_t count) {
    double* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(count, 1) * sizeof(double)));
    return p;
}

// Simple validation: compute a few elements and compare. A values come from
// getPseudoRndValue, which is what initRows stored for A.
bool validateResult(const double* B, const double* C, const size_t N) {
    if (N == 0) return true;

    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                const double a = getPseudoRndValue(N, i, k);
                expected += a * B[k * N + j];
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool root = rank == 0;

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int exitCode = -1;  // >= 0 means exit right after parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            exitCode = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // GPUs on each node are shared round-robin among that node's ranks.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int visibleDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visibleDevices));
    if (visibleDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    for (int d = localRank; d < visibleDevices; d += localSize) {
        devices.push_back(d);
    }
    const int localCount = static_cast<int>(devices.size());

    // Rows are split over all GPUs in the cluster in rank order, so a rank
    // with more GPUs receives proportionally more rows.
    std::vector<int> deviceCounts(numRanks);
    MPI_Allgather(&localCount, 1, MPI_INT, deviceCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    size_t totalDevices = 0;
    size_t firstDevice = 0;
    for (int r = 0; r < numRanks; ++r) {
        if (r < rank) firstDevice += deviceCounts[r];
        totalDevices += deviceCounts[r];
    }
    auto blockStart = [N, totalDevices](size_t g) { return N * g / totalDevices; };

    const size_t rowBegin = blockStart(firstDevice);
    const size_t rowEnd = blockStart(firstDevice + localCount);
    const size_t localRows = rowEnd - rowBegin;

    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowDispls(numRanks);
    {
        size_t first = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t end = first + deviceCounts[r];
            rowDispls[r] = static_cast<int>(blockStart(first));
            rowCounts[r] = static_cast<int>(blockStart(end) - blockStart(first));
            first = end;
        }
    }

    if (root) {
        printf("Devices: %zu GPU(s) across %d rank(s)\n", totalDevices, numRanks);
    }

    // Allocate matrices (B in full on every rank, A and C only for local rows)
    double* A = allocPinned(localRows * N);
    double* B = allocPinned(N * N);
    double* C = allocPinned(localRows * N);

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    initRows(A, N, rowBegin, rowEnd);
    initRows(B, N, 0, N);

    // Create CUDA contexts before timing so one-time setup isn't measured
    const int teamSize = std::max(1, localCount);
    #pragma omp parallel for num_threads(teamSize) schedule(static, 1)
    for (int t = 0; t < localCount; ++t) {
        CUDA_CHECK(cudaSetDevice(devices[t]));
        CUDA_CHECK(cudaFree(nullptr));
    }

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    // One host thread per local GPU drives that device's transfers and kernel
    #pragma omp parallel for num_threads(teamSize) schedule(static, 1)
    for (int t = 0; t < localCount; ++t) {
        const size_t rowStart = blockStart(firstDevice + t) - rowBegin;
        const size_t rowStop = blockStart(firstDevice + t + 1) - rowBegin;
        gemmOnDevice(devices[t], A + rowStart * N, B, C + rowStart * N, rowStop - rowStart, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double seconds = MPI_Wtime() - start;

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C on rank 0 only when it is needed for output or validation
    std::vector<double> fullC;
    if (validate || printResults) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);
        if (root) fullC.resize(N * N);
        MPI_Gatherv(C, static_cast<int>(localRows), rowType, fullC.data(),
                    rowCounts.data(), rowDispls.data(), rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(fullC, "MatrixC");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            const bool valid = validateResult(B, fullC.data(), N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFreeHost(C));
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return status;
}
