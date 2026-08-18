#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

// A panel is kept on one MPI rank.  This is large enough to make the trailing
// update a high-arithmetic-intensity GEMM, while keeping the panel broadcasts
// small enough to scale across nodes.
constexpr int kPanelSize = 128;

[[noreturn]] void abortAll(const int rank, const char* const where,
                           const char* const detail) {
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, where, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const int rank, const char* const where) {
    if (status != cudaSuccess) {
        abortAll(rank, where, cudaGetErrorString(status));
    }
}

void checkCublas(const cublasStatus_t status, const int rank, const char* const where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        abortAll(rank, where, "cuBLAS operation failed");
    }
}

void checkCusolver(const cusolverStatus_t status, const int rank, const char* const where) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        abortAll(rank, where, "cuSOLVER operation failed");
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const size_t elements, const int rank) { allocate(elements, rank); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            // There is nothing useful to do with a destructor-time CUDA error.
            cudaFree(ptr_);
        }
    }

    void allocate(const size_t elements, const int rank) {
        if (ptr_ != nullptr) {
            checkCuda(cudaFree(ptr_), rank, "cudaFree");
        }
        // cudaMalloc(0) is not portable.  Empty row partitions still take part
        // in collective operations, so give them a harmless one-element buffer.
        checkCuda(cudaMalloc(&ptr_, std::max<size_t>(elements, 1) * sizeof(T)),
                  rank, "cudaMalloc");
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

struct BlasContext {
    explicit BlasContext(const int rank) : rank(rank) {
        checkCublas(cublasCreate(&blas), rank, "cublasCreate");
        checkCusolver(cusolverDnCreate(&solver), rank, "cusolverDnCreate");
    }

    ~BlasContext() {
        if (solver != nullptr) {
            cusolverDnDestroy(solver);
        }
        if (blas != nullptr) {
            cublasDestroy(blas);
        }
    }

    const int rank;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
};

size_t rowStart(const int rank, const int ranks, const int n) {
    return static_cast<size_t>(n) * static_cast<size_t>(rank) /
           static_cast<size_t>(ranks);
}

size_t rowEnd(const int rank, const int ranks, const int n) {
    return rowStart(rank + 1, ranks, n);
}

int ownerOfRow(const int row, const int ranks, const int n) {
    // This also handles empty partitions when there are more ranks than rows.
    return static_cast<int>((static_cast<long long>(row + 1) * ranks - 1) / n);
}

unsigned int lcgAdvance(unsigned int state, unsigned long long steps) {
    // rand_r() in glibc advances this LCG three times per generated value.
    // Exponentiation of affine transforms gives each MPI rank its exact segment
    // of the original rand_r(&seed) stream without serializing initialization.
    constexpr unsigned int a = 1103515245u;
    constexpr unsigned int c = 12345u;
    unsigned int accumulatedA = 1u;
    unsigned int accumulatedC = 0u;
    unsigned int stepA = a;
    unsigned int stepC = c;

    while (steps != 0) {
        if ((steps & 1u) != 0) {
            accumulatedC = accumulatedA * stepC + accumulatedC;
            accumulatedA *= stepA;
        }
        stepC = stepA * stepC + stepC;
        stepA *= stepA;
        steps >>= 1u;
    }
    return accumulatedA * state + accumulatedC;
}

unsigned int compatibleRandR(unsigned int& seed) {
    // This is the POSIX rand_r implementation used by glibc, reproduced here
    // so parallel generation remains bit-for-bit compatible with the original
    // serial seed stream on the benchmark platform.
    unsigned int next = seed;
    next = next * 1103515245u + 12345u;
    int result = static_cast<int>((next / 65536u) % 2048u);
    next = next * 1103515245u + 12345u;
    result = (result << 10) ^ static_cast<int>((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result = (result << 10) ^ static_cast<int>((next / 65536u) % 1024u);
    seed = next;
    return static_cast<unsigned int>(result);
}

void generateLocalB(std::vector<double>& localB, const int n, const size_t firstRow) {
    const size_t firstValue = firstRow * static_cast<size_t>(n);
    const size_t localRows = localB.size() / static_cast<size_t>(n);
    // Each output row begins at a jump-ahead point in the same serial random
    // stream.  This lets OpenMP generate owned rows concurrently without
    // changing any generated value or the serial order within an inner product.
#pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(localRows); ++row) {
        unsigned int seed = lcgAdvance(
            42u, 3ull * (firstValue + static_cast<size_t>(row) * n));
        double* const output = localB.data() + static_cast<size_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            output[col] = static_cast<double>(compatibleRandR(seed)) /
                              static_cast<double>(RAND_MAX) -
                          0.5;
        }
    }
}

// The original generator is retained for validation.  Its random sequence and
// each individual inner-product order are unchanged; only independent output
// rows run concurrently with OpenMP.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const int n) {
    std::vector<double> B(static_cast<size_t>(n) * n);
    unsigned int seed = 42u;
    for (double& value : B) {
        value = static_cast<double>(rand_r(&seed)) / static_cast<double>(RAND_MAX) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += B[static_cast<size_t>(i) * n + k] *
                       B[static_cast<size_t>(j) * n + k];
            }
            A[static_cast<size_t>(i) * n + j] = sum;
        }
        A[static_cast<size_t>(i) * n + i] += n;
    }
}

__global__ void addDiagonalKernel(double* const matrix, const int rows, const int n,
                                  const int globalFirstRow) {
    const int localRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow < rows) {
        matrix[static_cast<size_t>(localRow) * n + globalFirstRow + localRow] += n;
    }
}

void makeColumnMajorDiagonal(const std::vector<double>& rowMajor,
                             std::vector<double>& columnMajor, const int width) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            columnMajor[static_cast<size_t>(col) * width + row] =
                rowMajor[static_cast<size_t>(row) * width + col];
        }
    }
}

void makeLowerRowMajor(const std::vector<double>& columnMajor,
                       std::vector<double>& rowMajor, const int width) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            rowMajor[static_cast<size_t>(row) * width + col] =
                row >= col ? columnMajor[static_cast<size_t>(col) * width + row] : 0.0;
        }
    }
}

bool distributedCholesky(DeviceBuffer<double>& deviceA, const int n,
                         const int rank, const int ranks, const size_t localFirst,
                         const size_t localLast, BlasContext& context) {
    const size_t localRows = localLast - localFirst;
    DeviceBuffer<double> deviceDiagonal(static_cast<size_t>(kPanelSize) * kPanelSize, rank);
    DeviceBuffer<double> devicePanel(std::max<size_t>(localRows, 1) * kPanelSize, rank);
    DeviceBuffer<double> deviceAllPanel(static_cast<size_t>(n) * kPanelSize, rank);
    DeviceBuffer<int> deviceInfo(1, rank);
    int maxWorkspaceElements = 0;
    checkCusolver(cusolverDnDpotrf_bufferSize(context.solver, CUBLAS_FILL_MODE_LOWER,
                                              kPanelSize, deviceDiagonal.data(), kPanelSize,
                                              &maxWorkspaceElements),
                  rank, "cusolverDnDpotrf maximum workspace size");
    DeviceBuffer<double> potrfWorkspace(maxWorkspaceElements, rank);

    std::vector<double> diagonalRowMajor(static_cast<size_t>(kPanelSize) * kPanelSize);
    std::vector<double> diagonalColumnMajor(static_cast<size_t>(kPanelSize) * kPanelSize);
    std::vector<double> localPanel(std::max<size_t>(localRows, 1) * kPanelSize);
    std::vector<double> allPanel(static_cast<size_t>(n) * kPanelSize);
    std::vector<int> receiveCounts(ranks);
    std::vector<int> displacements(ranks);

    for (int panelStart = 0; panelStart < n;) {
        const int owner = ownerOfRow(panelStart, ranks, n);
        const size_t ownerLast = rowEnd(owner, ranks, n);
        const int panelWidth = std::min(
            {kPanelSize, n - panelStart, static_cast<int>(ownerLast - panelStart)});
        const int panelEnd = panelStart + panelWidth;
        int panelSuccess = 1;
        int failureIndex = -1;

        if (rank == owner) {
            const size_t localOffset = static_cast<size_t>(panelStart) - localFirst;
            checkCuda(cudaMemcpy2D(diagonalRowMajor.data(), panelWidth * sizeof(double),
                                   deviceA.data() + localOffset * n + panelStart,
                                   static_cast<size_t>(n) * sizeof(double),
                                   panelWidth * sizeof(double), panelWidth,
                                   cudaMemcpyDeviceToHost),
                      rank, "copy diagonal panel from device");
            makeColumnMajorDiagonal(diagonalRowMajor, diagonalColumnMajor, panelWidth);
            checkCuda(cudaMemcpy(deviceDiagonal.data(), diagonalColumnMajor.data(),
                                 static_cast<size_t>(panelWidth) * panelWidth * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      rank, "copy diagonal panel to device");

            int workspaceElements = 0;
            checkCusolver(cusolverDnDpotrf_bufferSize(
                              context.solver, CUBLAS_FILL_MODE_LOWER, panelWidth,
                              deviceDiagonal.data(), panelWidth, &workspaceElements),
                          rank, "cusolverDnDpotrf_bufferSize");
            checkCusolver(cusolverDnDpotrf(context.solver, CUBLAS_FILL_MODE_LOWER,
                                            panelWidth, deviceDiagonal.data(), panelWidth,
                                            potrfWorkspace.data(), workspaceElements,
                                            deviceInfo.data()),
                          rank, "cusolverDnDpotrf");
            checkCuda(cudaMemcpy(&failureIndex, deviceInfo.data(), sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      rank, "copy POTRF status");
            if (failureIndex != 0) {
                panelSuccess = 0;
                failureIndex = panelStart + std::max(failureIndex - 1, 0);
            } else {
                checkCuda(cudaMemcpy(diagonalColumnMajor.data(), deviceDiagonal.data(),
                                     static_cast<size_t>(panelWidth) * panelWidth * sizeof(double),
                                     cudaMemcpyDeviceToHost),
                          rank, "copy factored diagonal panel");
                makeLowerRowMajor(diagonalColumnMajor, diagonalRowMajor, panelWidth);
                checkCuda(cudaMemcpy2D(deviceA.data() + localOffset * n + panelStart,
                                       static_cast<size_t>(n) * sizeof(double),
                                       diagonalRowMajor.data(), panelWidth * sizeof(double),
                                       panelWidth * sizeof(double), panelWidth,
                                       cudaMemcpyHostToDevice),
                          rank, "store factored diagonal panel");
            }
        }

        MPI_Bcast(&panelSuccess, 1, MPI_INT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&failureIndex, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (panelSuccess == 0) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                            failureIndex);
            }
            return false;
        }

        // Broadcast a host-staged panel.  This works both with conventional MPI
        // and CUDA-aware MPI implementations, without making CUDA awareness a
        // hidden deployment requirement.
        MPI_Bcast(diagonalRowMajor.data(), panelWidth * panelWidth, MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);
        makeColumnMajorDiagonal(diagonalRowMajor, diagonalColumnMajor, panelWidth);
        checkCuda(cudaMemcpy(deviceDiagonal.data(), diagonalColumnMajor.data(),
                             static_cast<size_t>(panelWidth) * panelWidth * sizeof(double),
                             cudaMemcpyHostToDevice),
                  rank, "broadcast diagonal panel to device");

        const size_t firstTrailingRow = std::max(localFirst, static_cast<size_t>(panelEnd));
        const size_t localTrailingRows = firstTrailingRow < localLast
                                             ? localLast - firstTrailingRow
                                             : 0;
        if (localTrailingRows != 0) {
            const size_t localOffset = firstTrailingRow - localFirst;
            checkCuda(cudaMemcpy2D(devicePanel.data(), panelWidth * sizeof(double),
                                   deviceA.data() + localOffset * n + panelStart,
                                   static_cast<size_t>(n) * sizeof(double),
                                   panelWidth * sizeof(double), localTrailingRows,
                                   cudaMemcpyDeviceToDevice),
                      rank, "pack panel below diagonal");
            const double one = 1.0;
            // Row-major X is column-major X^T.  X = Aik * inv(Lkk^T), so the
            // transposed representation is solved by Lkk * X^T = Aik^T.
            checkCublas(cublasDtrsm(context.blas, CUBLAS_SIDE_LEFT,
                                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT, panelWidth,
                                    static_cast<int>(localTrailingRows), &one,
                                    deviceDiagonal.data(), panelWidth,
                                    devicePanel.data(), panelWidth),
                        rank, "cublasDtrsm panel solve");
            checkCuda(cudaMemcpy2D(deviceA.data() + localOffset * n + panelStart,
                                   static_cast<size_t>(n) * sizeof(double),
                                   devicePanel.data(), panelWidth * sizeof(double),
                                   panelWidth * sizeof(double), localTrailingRows,
                                   cudaMemcpyDeviceToDevice),
                      rank, "unpack solved panel");
            checkCuda(cudaMemcpy(localPanel.data(), devicePanel.data(),
                                 localTrailingRows * panelWidth * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      rank, "copy solved panel for MPI");
        }

        const int trailingRows = n - panelEnd;
        for (int process = 0; process < ranks; ++process) {
            const size_t processFirst = std::max(
                rowStart(process, ranks, n), static_cast<size_t>(panelEnd));
            const size_t processLast = rowEnd(process, ranks, n);
            const size_t rows = processFirst < processLast ? processLast - processFirst : 0;
            receiveCounts[process] = static_cast<int>(rows * panelWidth);
            displacements[process] = static_cast<int>(
                (processFirst - static_cast<size_t>(panelEnd)) * panelWidth);
        }
        MPI_Allgatherv(localTrailingRows == 0 ? nullptr : localPanel.data(),
                       static_cast<int>(localTrailingRows * panelWidth), MPI_DOUBLE,
                       allPanel.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (localTrailingRows != 0 && trailingRows != 0) {
            checkCuda(cudaMemcpy(deviceAllPanel.data(), allPanel.data(),
                                 static_cast<size_t>(trailingRows) * panelWidth * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      rank, "copy gathered panel to device");
            const double minusOne = -1.0;
            const double one = 1.0;
            // The local trailing rectangle is row-major m-by-n.  In its
            // column-major transpose view this is C^T = Lall * Llocal^T.
            // ldc=n preserves the original full-row pitch of deviceA.
            const size_t localOffset = firstTrailingRow - localFirst;
            checkCublas(cublasDgemm(context.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    trailingRows, static_cast<int>(localTrailingRows),
                                    panelWidth, &minusOne, deviceAllPanel.data(), panelWidth,
                                    devicePanel.data(), panelWidth, &one,
                                    deviceA.data() + localOffset * n + panelEnd, n),
                        rank, "cublasDgemm trailing update");
        }

        panelStart = panelEnd;
    }

    // MPI does not imply completion of queued device work.  Complete it before
    // the benchmark's end barrier and before any result is copied to the host.
    checkCuda(cudaDeviceSynchronize(), rank, "finish device factorization");
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      const int n) {
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            const int limit = std::min(i, j);
            for (int k = 0; k <= limit; ++k) {
                sum += L[static_cast<size_t>(i) * n + k] *
                       L[static_cast<size_t>(j) * n + k];
            }
            const double error = std::fabs(sum - original[static_cast<size_t>(i) * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError,
                                error / (std::fabs(original[static_cast<size_t>(i) * n + j]) +
                                         1e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* const programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortAll(rank, "MPI_Init_thread", "MPI implementation lacks MPI_THREAD_FUNNELED");
    }

    size_t requestedSize = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentError = false;
    bool showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                argumentError = true;
            } else {
                requestedSize = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentError = true;
        }
    }

    // MPI collectives use int counts and cuBLAS uses int dimensions.  This
    // bound is far above the practical memory limit of the replicated B panel.
    if (requestedSize > 46340) {
        argumentError = true;
    }
    int commandState = argumentError ? 1 : (showHelp ? 2 : 0);
    int globalCommandState = 0;
    MPI_Allreduce(&commandState, &globalCommandState, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (globalCommandState != 0) {
        if (rank == 0) {
            if (globalCommandState == 1) {
                std::printf("Invalid command-line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return globalCommandState == 2 ? 0 : 1;
    }
    const int n = static_cast<int>(requestedSize);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount == 0) {
        abortAll(rank, "CUDA initialization", "no CUDA accelerator is visible");
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), rank, "cudaSetDevice");
    checkCuda(cudaFree(nullptr), rank, "CUDA context initialization");

    const size_t localFirst = rowStart(rank, ranks, n);
    const size_t localLast = rowEnd(rank, ranks, n);
    const size_t localRows = localLast - localFirst;
    const size_t localElements = localRows * static_cast<size_t>(n);
    const size_t totalElements = static_cast<size_t>(n) * n;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel execution: %d MPI ranks, CUDA accelerators, OpenMP host work\n",
                    ranks);
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> original;
    if (validate && rank == 0) {
        original.resize(totalElements);
        generatePositiveDefiniteMatrix(original, n);
    }

    // Build A=B*B^T in distributed rows.  B is all-gathered once, then the
    // accelerator does the dense product; only A's owned rows remain resident.
    std::vector<double> localB(localElements);
    std::vector<double> fullB(totalElements);
    generateLocalB(localB, n, localFirst);
    std::vector<int> bCounts(ranks);
    std::vector<int> bDisplacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        const size_t rows = rowEnd(process, ranks, n) - rowStart(process, ranks, n);
        bCounts[process] = static_cast<int>(rows * static_cast<size_t>(n));
        bDisplacements[process] = static_cast<int>(rowStart(process, ranks, n) * n);
    }
    MPI_Allgatherv(localB.empty() ? nullptr : localB.data(), static_cast<int>(localElements),
                   MPI_DOUBLE, fullB.data(), bCounts.data(), bDisplacements.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    BlasContext context(rank);
    DeviceBuffer<double> deviceA(localElements, rank);
    {
        // B is needed only to create A.  Releasing both device copies before
        // factorization leaves the GPU memory budget for the distributed factor
        // and permits substantially larger matrices per accelerator.
        DeviceBuffer<double> deviceLocalB(localElements, rank);
        DeviceBuffer<double> deviceFullB(totalElements, rank);
        if (!localB.empty()) {
            checkCuda(cudaMemcpy(deviceLocalB.data(), localB.data(),
                                 localElements * sizeof(double), cudaMemcpyHostToDevice),
                      rank, "copy local B to device");
        }
        checkCuda(cudaMemcpy(deviceFullB.data(), fullB.data(), totalElements * sizeof(double),
                             cudaMemcpyHostToDevice),
                  rank, "copy global B to device");

        if (localRows != 0) {
            const double one = 1.0;
            const double zero = 0.0;
            // Row-major A is the column-major transpose.  This computes
            // A^T=B*B_local^T and stores it directly in the row-major A buffer.
            checkCublas(cublasDgemm(context.blas, CUBLAS_OP_T, CUBLAS_OP_N, n,
                                    static_cast<int>(localRows), n, &one,
                                    deviceFullB.data(), n, deviceLocalB.data(), n, &zero,
                                    deviceA.data(), n),
                        rank, "cublasDgemm SPD matrix generation");
            const int threads = 256;
            const int blocks = (static_cast<int>(localRows) + threads - 1) / threads;
            addDiagonalKernel<<<blocks, threads>>>(deviceA.data(), static_cast<int>(localRows), n,
                                                   static_cast<int>(localFirst));
            checkCuda(cudaGetLastError(), rank, "add diagonal kernel launch");
        }
        checkCuda(cudaDeviceSynchronize(), rank, "finish SPD matrix generation");
    }
    localB.clear();
    localB.shrink_to_fit();
    fullB.clear();
    fullB.shrink_to_fit();

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(deviceA, n, rank, ranks, localFirst, localLast,
                                             context);
    MPI_Barrier(MPI_COMM_WORLD);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(duration * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = duration > 0.0 ? ops / duration / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<double> localResult(localElements);
        if (!localResult.empty()) {
            checkCuda(cudaMemcpy(localResult.data(), deviceA.data(),
                                 localElements * sizeof(double), cudaMemcpyDeviceToHost),
                      rank, "copy local factor to host");
#pragma omp parallel for schedule(static)
            for (size_t row = 0; row < localRows; ++row) {
                const size_t globalRow = localFirst + row;
                for (int col = static_cast<int>(globalRow) + 1; col < n; ++col) {
                    localResult[row * n + col] = 0.0;
                }
            }
        }

        std::vector<double> factor;
        if (rank == 0) {
            factor.resize(totalElements);
        }
        MPI_Gatherv(localResult.empty() ? nullptr : localResult.data(),
                    static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? factor.data() : nullptr, bCounts.data(), bDisplacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(factor, "CholeskyL");
            }
            if (validate) {
                std::printf("Validating result...\n");
                if (validateCholesky(factor, original, n)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
