#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/results_output.hpp"

namespace {
constexpr int blockSize = 256;
int rankId = 0;
void cudaCheck(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", rankId, cudaGetErrorString(code));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
void blasCheck(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: cuBLAS error %d\n", rankId, int(code));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
struct Buffer {
    double *device = nullptr, *host = nullptr;
    void allocate(size_t count, bool pinned = false) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(double)));
        if (pinned) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(double)));
    }
    void release() {
        if (device) cudaCheck(cudaFree(device));
        if (host) cudaCheck(cudaFreeHost(host));
        device = host = nullptr;
    }
};
__global__ void addDiagonal(double* tile, int width, int n) {
    int i = threadIdx.x;
    if (i < width) tile[i + i * blockSize] += n;
}

// A persistent OpenMP team handles the small CPU panel. Each column depends on
// the preceding column; rows below its diagonal are independent.
bool factorDiagonal(double* a, int width) {
    bool good = true;
    // Contiguous row dot products permit SIMD and avoid stride-256 CPU loads.
    std::vector<double> panel(size_t(width) * width);
    double* p = panel.data();
    #pragma omp parallel shared(good)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < width; ++i)
            for (int j = 0; j <= i; ++j) p[i * width + j] = a[i + j * blockSize];
        for (int k = 0; k < width; ++k) {
            #pragma omp single
            {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int s = 0; s < k; ++s) sum += p[k * width + s] * p[k * width + s];
                const double value = p[k * width + k] - sum;
                if (!(value > 0.0) || !std::isfinite(value)) good = false;
                else p[k * width + k] = std::sqrt(value);
            }
            #pragma omp for schedule(static)
            for (int i = k + 1; i < width; ++i) {
                if (good) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int s = 0; s < k; ++s) sum += p[i * width + s] * p[k * width + s];
                    p[i * width + k] = (p[i * width + k] - sum) / p[k * width + k];
                }
            }
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < width; ++i)
                a[i + j * blockSize] = i >= j ? p[i * width + j] : 0.0;
    }
    return good;
}

struct Matrix {
    int n, tiles, processRows, processCols, row, col;
    MPI_Comm rowComm, colComm;
    cudaStream_t stream;
    cublasHandle_t handle;
    std::vector<Buffer> data, left, right;
    Buffer diagonal;
    double** updatePointers = nullptr;
    size_t updateCapacity = 0;
    int width(int i) const { return std::min(blockSize, n - i * blockSize); }
    int owner(int i, int j) const { return (i % processRows) * processCols + j % processCols; }
    Buffer& tile(int i, int j) { return data[size_t(i) * tiles + j]; }
    Matrix(int size, int processes) : n(size), tiles((size - 1) / blockSize + 1) {
        int dims[2] = {0, 0};
        MPI_Dims_create(processes, 2, dims);
        processRows = dims[0]; processCols = dims[1];
        row = rankId / processCols; col = rankId % processCols;
        MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blasCheck(cublasCreate(&handle));
        blasCheck(cublasSetStream(handle, stream));
        // Double precision throughout; no reduced-precision tensor math.
        blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        data.resize(size_t(tiles) * tiles);
        left.resize(tiles); right.resize(tiles);
        const size_t count = size_t(blockSize) * blockSize;
        for (int i = 0; i < tiles; ++i) {
            if (i % processRows == row) left[i].allocate(count, true);
            if (i % processCols == col) right[i].allocate(count, true);
            for (int j = 0; j <= i; ++j)
                if (owner(i, j) == rankId) {
                    tile(i, j).allocate(count);
                    ++updateCapacity;
                }
        }
        diagonal.allocate(count, true);
        if (updateCapacity) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&updatePointers),
            3 * updateCapacity * sizeof(double*)));
    }
    ~Matrix() {
        for (auto& b : data) b.release();
        for (auto& b : left) b.release();
        for (auto& b : right) b.release();
        diagonal.release();
        if (updatePointers) cudaCheck(cudaFree(updatePointers));
        blasCheck(cublasDestroy(handle));
        cudaCheck(cudaStreamDestroy(stream));
        MPI_Comm_free(&rowComm); MPI_Comm_free(&colComm);
    }
    void generate() {
        // Preserve the exact original rand_r sequence. Retain only B row blocks
        // needed by this process row/column, rather than replicating B on GPUs.
        std::vector<Buffer> randomRows(tiles);
        std::vector<double> host(size_t(n) * blockSize);
        unsigned int seed = 42;
        for (int i = 0; i < tiles; ++i) {
            const bool needed = i % processRows == row || i % processCols == col;
            const size_t count = size_t(n) * width(i);
            for (size_t s = 0; s < count; ++s) {
                const double value = rand_r(&seed) / double(RAND_MAX) - 0.5;
                if (needed) host[s] = value;
            }
            if (needed) {
                randomRows[i].allocate(count);
                cudaCheck(cudaMemcpy(randomRows[i].device, host.data(), count * sizeof(double), cudaMemcpyHostToDevice));
            }
        }
        const double one = 1.0, zero = 0.0;
        for (int i = row; i < tiles; i += processRows)
            for (int j = col; j <= i; j += processCols) {
                cudaCheck(cudaMemsetAsync(tile(i,j).device, 0, size_t(blockSize)*blockSize*sizeof(double), stream));
                blasCheck(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    width(i), width(j), n, &one, randomRows[i].device, n,
                    randomRows[j].device, n, &zero, tile(i,j).device, blockSize));
                if (i == j) addDiagonal<<<1, blockSize, 0, stream>>>(tile(i,j).device, width(i), n);
            }
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaStreamSynchronize(stream));
        for (auto& b : randomRows) b.release();
        // Warm up the triangular-solve implementation outside the timed region.
        std::fill(diagonal.host, diagonal.host + blockSize * blockSize, 0.0);
        for (int i = 0; i < blockSize; ++i) diagonal.host[i + i * blockSize] = 1.0;
        cudaCheck(cudaMemcpyAsync(diagonal.device, diagonal.host,
            size_t(blockSize)*blockSize*sizeof(double), cudaMemcpyHostToDevice, stream));
        Buffer scratch;
        scratch.allocate(size_t(blockSize)*blockSize);
        cudaCheck(cudaMemsetAsync(scratch.device, 0, size_t(blockSize)*blockSize*sizeof(double), stream));
        blasCheck(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, blockSize, blockSize, &one,
            diagonal.device, blockSize, scratch.device, blockSize));
        if (updateCapacity) {
            double* pointers[] = {diagonal.device, diagonal.device, scratch.device};
            cudaCheck(cudaMemcpyAsync(updatePointers, pointers, sizeof(pointers), cudaMemcpyHostToDevice, stream));
            blasCheck(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                blockSize, blockSize, blockSize, &one,
                const_cast<const double**>(updatePointers), blockSize,
                const_cast<const double**>(updatePointers + 1), blockSize,
                &zero, updatePointers + 2, blockSize, 1));
        }
        blasCheck(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            blockSize, blockSize, &one, diagonal.device, blockSize, &zero,
            scratch.device, blockSize));
        cudaCheck(cudaStreamSynchronize(stream));
        scratch.release();
    }
    bool factor() {
        const size_t bytes = size_t(blockSize) * blockSize * sizeof(double);
        const int count = blockSize * blockSize;
        const double one = 1.0, minusOne = -1.0;
        for (int k = 0; k < tiles; ++k) {
            int good = 1;
            const int root = owner(k,k);
            if (rankId == root) {
                cudaCheck(cudaMemcpyAsync(diagonal.host, tile(k,k).device, bytes, cudaMemcpyDeviceToHost, stream));
                cudaCheck(cudaStreamSynchronize(stream));
                good = factorDiagonal(diagonal.host, width(k));
                if (good) cudaCheck(cudaMemcpyAsync(tile(k,k).device, diagonal.host, bytes, cudaMemcpyHostToDevice, stream));
            }
            MPI_Bcast(&good, 1, MPI_INT, root, MPI_COMM_WORLD);
            if (!good) {
                if (rankId == 0) printf("Error: Matrix is not positive definite in diagonal block %d\n", k);
                return false;
            }
            MPI_Bcast(diagonal.host, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
            cudaCheck(cudaMemcpyAsync(diagonal.device, diagonal.host, bytes, cudaMemcpyHostToDevice, stream));
            // Owners solve all blocks of the current panel on their GPUs.
            if (col == k % processCols)
                for (int i = k + 1; i < tiles; ++i)
                    if (i % processRows == row) {
                        blasCheck(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, width(i), width(k),
                            &one, diagonal.device, blockSize, tile(i,k).device, blockSize));
                        cudaCheck(cudaMemcpyAsync(left[i].host, tile(i,k).device, bytes, cudaMemcpyDeviceToHost, stream));
                    }
            cudaCheck(cudaStreamSynchronize(stream));
            // Broadcast panels along process rows, then along process columns.
            // Host staging works with MPI implementations lacking CUDA awareness.
            std::vector<MPI_Request> requests;
            for (int i = k + 1; i < tiles; ++i)
                if (i % processRows == row) {
                    requests.emplace_back();
                    MPI_Ibcast(left[i].host, count, MPI_DOUBLE, k % processCols, rowComm, &requests.back());
                }
            if (!requests.empty()) MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
            for (int j = k + 1; j < tiles; ++j)
                if (j % processCols == col) {
                    if (j % processRows == row) std::memcpy(right[j].host, left[j].host, bytes);
                    requests.emplace_back();
                    MPI_Ibcast(right[j].host, count, MPI_DOUBLE, j % processRows, colComm, &requests.back());
                }
            if (!requests.empty()) MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            for (int i = k + 1; i < tiles; ++i) {
                if (i % processRows == row) cudaCheck(cudaMemcpyAsync(left[i].device, left[i].host, bytes, cudaMemcpyHostToDevice, stream));
                if (i % processCols == col) cudaCheck(cudaMemcpyAsync(right[i].device, right[i].host, bytes, cudaMemcpyHostToDevice, stream));
            }
            // Batch independent off-diagonal updates to avoid one launch per
            // small tile. The final partial row uses a separate batch.
            std::vector<double*> aPointers[2], bPointers[2], cPointers[2];
            for (int i = k + 1; i < tiles; ++i)
                if (i % processRows == row)
                    for (int j = k + 1; j <= i; ++j)
                        if (j % processCols == col) {
                            if (i == j) {
                                blasCheck(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N, width(i), width(k), &minusOne,
                                    left[i].device, blockSize, &one, tile(i,j).device, blockSize));
                            } else {
                                const int group = width(i) == blockSize ? 0 : 1;
                                aPointers[group].push_back(left[i].device);
                                bPointers[group].push_back(right[j].device);
                                cPointers[group].push_back(tile(i,j).device);
                            }
                        }
            size_t offset = 0;
            for (int group = 0; group < 2; ++group) {
                const size_t batch = aPointers[group].size();
                if (!batch) continue;
                cudaCheck(cudaMemcpyAsync(updatePointers + offset, aPointers[group].data(),
                    batch*sizeof(double*), cudaMemcpyHostToDevice, stream));
                cudaCheck(cudaMemcpyAsync(updatePointers + updateCapacity + offset, bPointers[group].data(),
                    batch*sizeof(double*), cudaMemcpyHostToDevice, stream));
                cudaCheck(cudaMemcpyAsync(updatePointers + 2*updateCapacity + offset, cPointers[group].data(),
                    batch*sizeof(double*), cudaMemcpyHostToDevice, stream));
                blasCheck(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    group == 0 ? blockSize : width(tiles-1), blockSize, width(k), &minusOne,
                    const_cast<const double**>(updatePointers + offset), blockSize,
                    const_cast<const double**>(updatePointers + updateCapacity + offset), blockSize,
                    &one, updatePointers + 2*updateCapacity + offset, blockSize, int(batch)));
                offset += batch;
            }
            // The next panel can reuse host buffers only after their H2D copies
            // finish. This also establishes the dependency on trailing updates.
            cudaCheck(cudaStreamSynchronize(stream));
        }
        return true;
    }
    std::vector<double> gather(bool symmetric) {
        std::vector<double> result;
        if (rankId == 0) result.resize(size_t(n) * n, 0.0);
        const int count = blockSize * blockSize;
        for (int i = 0; i < tiles; ++i)
            for (int j = 0; j <= i; ++j) {
                const int source = owner(i,j);
                if (rankId == source) {
                    cudaCheck(cudaMemcpyAsync(diagonal.host, tile(i,j).device, size_t(count)*sizeof(double), cudaMemcpyDeviceToHost, stream));
                    cudaCheck(cudaStreamSynchronize(stream));
                    if (source != 0) MPI_Send(diagonal.host, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
                if (rankId == 0) {
                    if (source != 0) MPI_Recv(diagonal.host, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    #pragma omp parallel for schedule(static)
                    for (int r = 0; r < width(i); ++r)
                        for (int c = 0; c < width(j); ++c) {
                            const size_t gr = size_t(i) * blockSize + r, gc = size_t(j) * blockSize + c;
                            if (gr >= gc) {
                                result[gr * n + gc] = diagonal.host[r + c * blockSize];
                                if (symmetric) result[gc * n + gr] = result[gr * n + gc];
                            }
                        }
                }
            }
        return result;
    }
};

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n) {
    Buffer factor, product;
    factor.allocate(size_t(n)*n); product.allocate(size_t(n)*n);
    cudaCheck(cudaMemcpy(factor.device, l.data(), size_t(n)*n*sizeof(double), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    blasCheck(cublasCreate(&handle));
    const double one = 1.0, zero = 0.0;
    // Row-major L is column-major L^T. The symmetric product has either layout.
    blasCheck(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
        &one, factor.device, n, factor.device, n, &zero, product.device, n));
    std::vector<double> reconstructed(size_t(n)*n);
    cudaCheck(cudaMemcpy(reconstructed.data(), product.device, size_t(n)*n*sizeof(double), cudaMemcpyDeviceToHost));
    double maxError = 0.0, relError = 0.0;
    int finite = 1;
    #pragma omp parallel for reduction(max:maxError,relError) reduction(&:finite) schedule(static)
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        const double error = std::fabs(reconstructed[i] - original[i]);
        finite &= std::isfinite(error);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(original[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    factor.release(); product.release(); blasCheck(cublasDestroy(handle));
    return finite && relError <= 1e-6;
}
void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || *end || value <= 0 || value > INT_MAX ||
                size_t(value) > std::numeric_limits<size_t>::max() / size_t(value) / sizeof(double)) {
                if (rankId == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = int(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rankId == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rankId == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    // Bind one rank to each visible accelerator, using node-local ranks. CUDA
    // visibility masks supplied by cluster launchers are respected.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &localComm);
    int localRank, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", rankId);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);
    // Excessive CPU threads make the small diagonal panel slower. Respect an
    // explicit user setting; otherwise cap the team to the available CPU cores.
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(4, omp_get_num_procs()));
    int status = 0;
    try {
        Matrix matrix(n, processes);
        if (rankId == 0) {
            printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
            printf("MPI grid: %d x %d; CUDA tile size: %d; OpenMP threads: %d\n",
                matrix.processRows, matrix.processCols, blockSize, omp_get_max_threads());
            printf("Generating positive definite matrix...\n");
        }
        matrix.generate();
        std::vector<double> original;
        if (validate) original = matrix.gather(true);
        if (rankId == 0) printf("Computing Cholesky decomposition...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        const bool success = matrix.factor();
        const double elapsed = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!success) {
            if (rankId == 0) printf("Cholesky decomposition failed\n");
            status = 1;
        } else {
            if (rankId == 0) {
                printf("Computation time: %.3f ms\n", duration * 1000.0);
                printf("Performance: %.3f GFLOPS\n", double(n)*n*n / (3.0*duration*1e9));
            }
            std::vector<double> result;
            if (validate || printResults) result = matrix.gather(false);
            if (rankId == 0) {
                if (printResults) print_results(result, "CholeskyL");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateCholesky(result, original, n);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rankId, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
