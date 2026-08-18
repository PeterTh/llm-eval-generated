#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

namespace {

constexpr std::size_t kDefaultBlockSize = 128;

#ifdef __CUDACC__

// Each thread updates one element of the local lower-triangular trailing
// matrix.  The panel is replicated on every rank, so this kernel performs all
// of the O(n^3) work without MPI communication inside a CUDA launch.
__global__ void updateTrailingKernel(double* matrix,
                                     const double* panel,
                                     const std::size_t n,
                                     const std::size_t local_rows,
                                     const std::size_t row_begin,
                                     const std::size_t k,
                                     const std::size_t first,
                                     const std::size_t block_size) {
    const std::size_t j = first +
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t local_i =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (local_i >= local_rows || j >= n) {
        return;
    }

    const std::size_t i = row_begin + local_i;
    if (i < first || j > i) {
        return;
    }

    const double* const matrix_row = matrix + local_i * n;
    const double* const panel_row = panel + (j - first) * block_size;
    double sum = 0.0;
    for (std::size_t t = 0; t < block_size; ++t) {
        sum += matrix_row[k + t] * panel_row[t];
    }
    matrix[local_i * n + j] -= sum;
}

// Solve A21 = A21 * inv(L11^T).  One CUDA thread owns one row; the short
// triangular solve has a sequential dependency in the block dimension, while
// thousands of rows are solved concurrently.
__global__ void solvePanelKernel(double* matrix,
                                 const double* diagonal,
                                 const std::size_t n,
                                 const std::size_t local_rows,
                                 const std::size_t row_begin,
                                 const std::size_t k,
                                 const std::size_t first,
                                 const std::size_t block_size) {
    const std::size_t local_i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_i >= local_rows) {
        return;
    }

    const std::size_t i = row_begin + local_i;
    if (i < first) {
        return;
    }

    double* const row = matrix + local_i * n;
    for (std::size_t col = 0; col < block_size; ++col) {
        double sum = 0.0;
        for (std::size_t t = 0; t < col; ++t) {
            sum += row[k + t] * diagonal[col * block_size + t];
        }
        row[k + col] = (row[k + col] - sum) /
                       diagonal[col * block_size + col];
    }
}

__global__ void zeroUpperKernel(double* matrix,
                                const std::size_t n,
                                const std::size_t local_rows,
                                const std::size_t row_begin) {
    const std::size_t local_i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_i >= local_rows) {
        return;
    }

    const std::size_t i = row_begin + local_i;
    double* const row = matrix + local_i * n;
    for (std::size_t j = i + 1; j < n; ++j) {
        row[j] = 0.0;
    }
}

#endif

std::size_t checkedProduct(const std::size_t a, const std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return 0;
    }
    return a * b;
}

bool fitsMpiInt(const std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

struct RowRange {
    std::size_t begin;
    std::size_t end;

    std::size_t size() const { return end - begin; }
};

RowRange rowsForRank(const std::size_t n, const int rank, const int ranks) {
    const std::size_t p = static_cast<std::size_t>(rank);
    const std::size_t count = static_cast<std::size_t>(ranks);
    return {n * p / count, n * (p + 1) / count};
}

int ownerOfRow(const std::size_t n, const std::size_t row, const int ranks) {
    // This inverse of rowsForRank is exact for the quotient/remainder-free
    // partition used above, including ranks with no rows when ranks > n.
    int low = 0;
    int high = ranks - 1;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (rowsForRank(n, middle, ranks).end <= row) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

void generatePositiveDefiniteMatrix(std::vector<double>& matrix,
                                    const std::size_t n) {
    // Keep the original deterministic generator and summation order.  Only
    // the independent output entries are parallelized with OpenMP.
    const std::size_t elements = checkedProduct(n, n);
    std::vector<double> b(elements);
    unsigned int seed = 42;
    for (std::size_t i = 0; i < elements; ++i) {
        b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += b[static_cast<std::size_t>(i) * n + k] *
                       b[j * n + k];
            }
            matrix[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }

#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        matrix[static_cast<std::size_t>(i) * n +
               static_cast<std::size_t>(i)] += static_cast<double>(n);
    }
}

bool factorDiagonalTile(std::vector<double>& tile, const std::size_t n) {
    bool positive_definite = true;

    // Keep one OpenMP team alive across the panel's dependency chain.  The
    // diagonal step is serialized, while each below-diagonal column update is
    // distributed over the team.
#pragma omp parallel shared(positive_definite)
    {
        for (std::size_t j = 0; j < n; ++j) {
#pragma omp single nowait
            {
                if (positive_definite) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < j; ++k) {
                        sum += tile[j * n + k] * tile[j * n + k];
                    }
                    const double value = tile[j * n + j] - sum;
                    if (!(value > 0.0)) {
                        positive_definite = false;
                    } else {
                        tile[j * n + j] = std::sqrt(value);
                    }
                }
            }
#pragma omp barrier
            if (positive_definite) {
#pragma omp for schedule(static) nowait
                for (std::int64_t i = static_cast<std::int64_t>(j + 1);
                     i < static_cast<std::int64_t>(n); ++i) {
                    double off_diagonal_sum = 0.0;
                    for (std::size_t k = 0; k < j; ++k) {
                        off_diagonal_sum +=
                            tile[static_cast<std::size_t>(i) * n + k] *
                            tile[j * n + k];
                    }
                    tile[static_cast<std::size_t>(i) * n + j] =
                        (tile[static_cast<std::size_t>(i) * n + j] -
                         off_diagonal_sum) /
                        tile[j * n + j];
                }
            }
#pragma omp barrier
        }
    }
    return positive_definite;
}

void solvePanelHost(std::vector<double>& matrix,
                    const RowRange range,
                    const std::size_t n,
                    const std::size_t k,
                    const std::size_t first,
                    const std::size_t block_size,
                    const std::vector<double>& diagonal) {
#pragma omp parallel for schedule(static)
    for (std::int64_t local_i = 0;
         local_i < static_cast<std::int64_t>(range.size()); ++local_i) {
        const std::size_t i = range.begin + static_cast<std::size_t>(local_i);
        if (i < first) {
            continue;
        }

        double* const row = matrix.data() +
                            static_cast<std::size_t>(local_i) * n;
        for (std::size_t col = 0; col < block_size; ++col) {
            double sum = 0.0;
            for (std::size_t t = 0; t < col; ++t) {
                sum += row[k + t] * diagonal[col * block_size + t];
            }
            row[k + col] = (row[k + col] - sum) /
                           diagonal[col * block_size + col];
        }
    }
}

void updateTrailingHost(std::vector<double>& matrix,
                        const RowRange range,
                        const std::size_t n,
                        const std::size_t k,
                        const std::size_t first,
                        const std::size_t block_size,
                        const std::vector<double>& panel) {
#pragma omp parallel for schedule(static)
    for (std::int64_t local_i = 0;
         local_i < static_cast<std::int64_t>(range.size()); ++local_i) {
        const std::size_t local_row = static_cast<std::size_t>(local_i);
        const std::size_t i = range.begin + local_row;
        if (i < first) {
            continue;
        }

        double* const row = matrix.data() + local_row * n;
        for (std::size_t j = first; j <= i; ++j) {
            double sum = 0.0;
            const double* const panel_row = panel.data() +
                                             (j - first) * block_size;
            for (std::size_t t = 0; t < block_size; ++t) {
                sum += row[k + t] * panel_row[t];
            }
            row[j] -= sum;
        }
    }
}

void zeroUpperHost(std::vector<double>& matrix,
                   const RowRange range,
                   const std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::int64_t local_i = 0;
         local_i < static_cast<std::int64_t>(range.size()); ++local_i) {
        const std::size_t i = range.begin + static_cast<std::size_t>(local_i);
        double* const row = matrix.data() +
                            static_cast<std::size_t>(local_i) * n;
        for (std::size_t j = i + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }
}

bool validateCholesky(const std::vector<double>& lower,
                      const std::vector<double>& original,
                      const std::size_t n) {
    std::vector<double> reconstructed(checkedProduct(n, n));

#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += lower[static_cast<std::size_t>(i) * n + k] *
                       lower[j * n + k];
            }
            reconstructed[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }

    double max_error = 0.0;
    double relative_error = 0.0;
#pragma omp parallel for reduction(max:max_error,relative_error) schedule(static)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(checkedProduct(n, n)); ++index) {
        const std::size_t i = static_cast<std::size_t>(index);
        const double error = std::fabs(reconstructed[i] - original[i]);
        max_error = std::max(max_error, error);
        const double relative = error / (std::fabs(original[i]) + 1e-10);
        relative_error = std::max(relative_error, relative);
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

#ifdef __CUDACC__

class CudaLocalMatrix {
public:
    CudaLocalMatrix() = default;
    CudaLocalMatrix(const CudaLocalMatrix&) = delete;
    CudaLocalMatrix& operator=(const CudaLocalMatrix&) = delete;

    ~CudaLocalMatrix() { release(); }

    bool initialize(const std::vector<double>& host_matrix,
                    const RowRange range,
                    const std::size_t n,
                    const std::size_t max_block,
                    const int local_rank) {
        n_ = n;
        range_ = range;
        max_block_ = max_block;

        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0 || range.size() == 0) {
            cudaGetLastError();
            return false;
        }

        error = cudaSetDevice(local_rank % device_count);
        if (error != cudaSuccess) {
            cudaGetLastError();
            return false;
        }

        const std::size_t matrix_elements = checkedProduct(range.size(), n);
        const std::size_t panel_elements = checkedProduct(n, max_block);
        if (matrix_elements == 0 || panel_elements == 0) {
            return false;
        }

        if (cudaMalloc(reinterpret_cast<void**>(&matrix_),
                       matrix_elements * sizeof(double)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&panel_),
                       panel_elements * sizeof(double)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&diagonal_),
                       max_block * max_block * sizeof(double)) != cudaSuccess) {
            release();
            return false;
        }

        if (cudaMemcpy(matrix_, host_matrix.data(),
                       matrix_elements * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            release();
            return false;
        }
        enabled_ = true;
        return true;
    }

    void release() {
        if (matrix_ != nullptr) {
            cudaFree(matrix_);
        }
        if (panel_ != nullptr) {
            cudaFree(panel_);
        }
        if (diagonal_ != nullptr) {
            cudaFree(diagonal_);
        }
        matrix_ = nullptr;
        panel_ = nullptr;
        diagonal_ = nullptr;
        enabled_ = false;
    }

    bool downloadTile(std::vector<double>& tile,
                      const std::size_t row,
                      const std::size_t column,
                      const std::size_t block_size) const {
        return cudaMemcpy2D(tile.data(), block_size * sizeof(double),
                            matrix_ + (row - range_.begin) * n_ + column,
                            n_ * sizeof(double),
                            block_size * sizeof(double), block_size,
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    bool uploadTile(const std::vector<double>& tile,
                    const std::size_t row,
                    const std::size_t column,
                    const std::size_t block_size) const {
        return cudaMemcpy2D(matrix_ + (row - range_.begin) * n_ + column,
                            n_ * sizeof(double), tile.data(),
                            block_size * sizeof(double),
                            block_size * sizeof(double), block_size,
                            cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool uploadDiagonal(const std::vector<double>& tile,
                        const std::size_t block_size) const {
        return cudaMemcpy(diagonal_, tile.data(),
                          block_size * block_size * sizeof(double),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool solvePanel(const std::size_t k,
                    const std::size_t first,
                    const std::size_t block_size) const {
        if (range_.size() == 0) {
            return true;
        }
        constexpr unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            (range_.size() + threads - 1) / threads);
        solvePanelKernel<<<blocks, threads>>>(matrix_, diagonal_, n_,
                                               range_.size(), range_.begin,
                                               k, first, block_size);
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    bool downloadPanel(std::vector<double>& panel,
                       const std::size_t row,
                       const std::size_t rows,
                       const std::size_t column,
                       const std::size_t block_size) const {
        if (rows == 0) {
            return true;
        }
        return cudaMemcpy2D(panel.data(), block_size * sizeof(double),
                            matrix_ + (row - range_.begin) * n_ + column,
                            n_ * sizeof(double),
                            block_size * sizeof(double), rows,
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    bool uploadPanel(const std::vector<double>& panel,
                     const std::size_t rows,
                     const std::size_t block_size) const {
        return cudaMemcpy(panel_, panel.data(),
                          rows * block_size * sizeof(double),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool updateTrailing(const std::size_t k,
                        const std::size_t first,
                        const std::size_t block_size) const {
        if (range_.size() == 0 || first >= n_) {
            return true;
        }
        constexpr unsigned int x_threads = 32;
        constexpr unsigned int y_threads = 8;
        const dim3 threads(x_threads, y_threads, 1);
        const dim3 blocks(
            static_cast<unsigned int>((n_ - first + x_threads - 1) /
                                       x_threads),
            static_cast<unsigned int>((range_.size() + y_threads - 1) /
                                       y_threads),
            1);
        updateTrailingKernel<<<blocks, threads>>>(
            matrix_, panel_, n_, range_.size(), range_.begin, k, first,
            block_size);
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    bool zeroUpper() const {
        if (range_.size() == 0) {
            return true;
        }
        constexpr unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            (range_.size() + threads - 1) / threads);
        zeroUpperKernel<<<blocks, threads>>>(matrix_, n_, range_.size(),
                                             range_.begin);
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    bool download(std::vector<double>& host_matrix) const {
        return cudaMemcpy(host_matrix.data(), matrix_,
                          host_matrix.size() * sizeof(double),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

private:
    bool enabled_ = false;
    std::size_t n_ = 0;
    std::size_t max_block_ = 0;
    RowRange range_{0, 0};
    double* matrix_ = nullptr;
    double* panel_ = nullptr;
    double* diagonal_ = nullptr;
};

#else

// This keeps a host-only compiler invocation usable for environments that do
// not have a CUDA compiler.  The CMake target below always builds the real
// CUDA kernels when CUDA is available.
class CudaLocalMatrix {
public:
    bool initialize(const std::vector<double>&,
                    const RowRange,
                    const std::size_t,
                    const std::size_t,
                    const int) {
        return false;
    }
    void release() {}
    bool downloadTile(std::vector<double>&, std::size_t, std::size_t,
                      std::size_t) const { return false; }
    bool uploadTile(const std::vector<double>&, std::size_t, std::size_t,
                    std::size_t) const { return false; }
    bool uploadDiagonal(const std::vector<double>&, std::size_t) const {
        return false;
    }
    bool solvePanel(std::size_t, std::size_t, std::size_t) const {
        return false;
    }
    bool downloadPanel(std::vector<double>&, std::size_t, std::size_t,
                       std::size_t, std::size_t) const { return false; }
    bool uploadPanel(const std::vector<double>&, std::size_t,
                     std::size_t) const { return false; }
    bool updateTrailing(std::size_t, std::size_t, std::size_t) const {
        return false;
    }
    bool zeroUpper() const { return false; }
    bool download(std::vector<double>&) const { return false; }
};

#endif

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, std::size_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::size_t n = 512;
    bool validate = false;
    bool print_results_flag = false;
    bool parse_ok = true;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            parse_ok = parseSize(argv[++i], n);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_flag = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            parse_ok = false;
        }
        if (!parse_ok) {
            break;
        }
    }

    if (show_help && parse_ok) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (!parse_ok) {
        if (rank == 0) {
            std::printf("Invalid command line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const std::size_t matrix_elements = checkedProduct(n, n);
    if (matrix_elements == 0 || !fitsMpiInt(matrix_elements)) {
        if (rank == 0) {
            std::printf("Matrix is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    const RowRange local_range = rowsForRank(n, rank, ranks);
    const std::size_t local_elements = checkedProduct(local_range.size(), n);
    if (local_elements == 0 && local_range.size() != 0) {
        if (rank == 0) {
            std::printf("Matrix allocation size overflow\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
    }

    std::vector<double> global_matrix;
    std::vector<double> original;
    if (rank == 0) {
        global_matrix.resize(matrix_elements);
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(global_matrix, n);
        if (validate) {
            original = global_matrix;
        }
    }

    std::vector<int> scatter_counts(static_cast<std::size_t>(ranks));
    std::vector<int> scatter_displacements(static_cast<std::size_t>(ranks));
    for (int p = 0; p < ranks; ++p) {
        const RowRange range = rowsForRank(n, p, ranks);
        const std::size_t count = checkedProduct(range.size(), n);
        const std::size_t displacement = checkedProduct(range.begin, n);
        if (!fitsMpiInt(count) || !fitsMpiInt(displacement)) {
            if (rank == 0) {
                std::printf("MPI scatter layout exceeds MPI count limits\n");
            }
            MPI_Finalize();
            return 1;
        }
        scatter_counts[static_cast<std::size_t>(p)] =
            static_cast<int>(count);
        scatter_displacements[static_cast<std::size_t>(p)] =
            static_cast<int>(displacement);
    }

    std::vector<double> local_matrix(local_elements);
    MPI_Scatterv(rank == 0 ? global_matrix.data() : nullptr,
                 scatter_counts.data(), scatter_displacements.data(),
                 MPI_DOUBLE, local_matrix.data(),
                 scatter_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    std::vector<double>().swap(global_matrix);

    MPI_Comm local_communicator = MPI_COMM_NULL;
    int local_rank = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_communicator);
    MPI_Comm_rank(local_communicator, &local_rank);

    std::size_t block_size = kDefaultBlockSize;
    block_size = std::min(block_size, n);
    // A panel never crosses a row ownership boundary.  That keeps each
    // diagonal tile on one rank while preserving large CUDA update panels.
    const std::size_t max_panel_elements = checkedProduct(n, block_size);
    if (max_panel_elements == 0) {
        if (rank == 0) {
            std::printf("Panel allocation size overflow\n");
        }
        MPI_Comm_free(&local_communicator);
        MPI_Finalize();
        return 1;
    }

    CudaLocalMatrix cuda_matrix;
    const bool cuda_enabled = cuda_matrix.initialize(
        local_matrix, local_range, n, block_size, local_rank);
    int cuda_rank = cuda_enabled ? 1 : 0;
    int cuda_ranks = 0;
    MPI_Allreduce(&cuda_rank, &cuda_ranks, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("CUDA update ranks: %d/%d\n", cuda_ranks, ranks);
    }

    std::vector<double> diagonal;
    std::vector<double> panel(max_panel_elements);
    std::vector<double> local_panel;
    std::vector<int> panel_counts(static_cast<std::size_t>(ranks));
    std::vector<int> panel_displacements(static_cast<std::size_t>(ranks));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    bool success = true;

    for (std::size_t k = 0; k < n && success;) {
        const int owner = ownerOfRow(n, k, ranks);
        const RowRange owner_range = rowsForRank(n, owner, ranks);
        const std::size_t block_size_for_panel =
            std::min(block_size, owner_range.end - k);
        const std::size_t first = k + block_size_for_panel;
        diagonal.assign(block_size_for_panel * block_size_for_panel, 0.0);

        if (rank == owner) {
            const std::size_t owner_row = k;
            if (cuda_enabled) {
                success = cuda_matrix.downloadTile(
                    diagonal, owner_row, k, block_size_for_panel);
            } else {
                const std::size_t local_row = k - local_range.begin;
                for (std::size_t i = 0; i < block_size_for_panel; ++i) {
                    std::memcpy(
                        diagonal.data() + i * block_size_for_panel,
                        local_matrix.data() + (local_row + i) * n + k,
                        block_size_for_panel * sizeof(double));
                }
            }
            if (success) {
                success = factorDiagonalTile(diagonal, block_size_for_panel);
            }
            if (success && cuda_enabled) {
                success = cuda_matrix.uploadTile(
                    diagonal, owner_row, k, block_size_for_panel);
            }
            if (success && !cuda_enabled) {
                const std::size_t local_row = k - local_range.begin;
                for (std::size_t i = 0; i < block_size_for_panel; ++i) {
                    std::memcpy(
                        local_matrix.data() + (local_row + i) * n + k,
                        diagonal.data() + i * block_size_for_panel,
                        block_size_for_panel * sizeof(double));
                }
            }
        }

        int factor_ok = success ? 1 : 0;
        MPI_Bcast(&factor_ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!factor_ok) {
            success = false;
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            }
            break;
        }
        success = true;
        MPI_Bcast(diagonal.data(),
                  static_cast<int>(block_size_for_panel *
                                   block_size_for_panel),
                  MPI_DOUBLE, owner, MPI_COMM_WORLD);

        if (cuda_enabled) {
            success = cuda_matrix.uploadDiagonal(diagonal,
                                                  block_size_for_panel);
            if (success) {
                success = cuda_matrix.solvePanel(
                    k, first, block_size_for_panel);
            }
        } else {
            solvePanelHost(local_matrix, local_range, n, k, first,
                           block_size_for_panel, diagonal);
        }

        const std::size_t local_panel_begin =
            std::max(local_range.begin, first);
        const std::size_t local_panel_end =
            std::min(local_range.end, n);
        const std::size_t local_panel_rows =
            local_panel_end > local_panel_begin
                ? local_panel_end - local_panel_begin
                : 0;
        local_panel.resize(local_panel_rows * block_size_for_panel);
        if (success && local_panel_rows != 0) {
            if (cuda_enabled) {
                success = cuda_matrix.downloadPanel(
                    local_panel, local_panel_begin, local_panel_rows, k,
                    block_size_for_panel);
            } else {
                for (std::size_t i = 0; i < local_panel_rows; ++i) {
                    std::memcpy(
                        local_panel.data() + i * block_size_for_panel,
                        local_matrix.data() +
                            (local_panel_begin - local_range.begin + i) * n + k,
                        block_size_for_panel * sizeof(double));
                }
            }
        }

        for (int p = 0; p < ranks; ++p) {
            const RowRange range = rowsForRank(n, p, ranks);
            const std::size_t begin = std::max(range.begin, first);
            const std::size_t end = std::min(range.end, n);
            const std::size_t rows = end > begin ? end - begin : 0;
            const std::size_t count = checkedProduct(rows,
                                                     block_size_for_panel);
            const std::size_t displacement = checkedProduct(
                begin > first ? begin - first : 0, block_size_for_panel);
            if (!fitsMpiInt(count) || !fitsMpiInt(displacement)) {
                success = false;
            }
            panel_counts[static_cast<std::size_t>(p)] =
                static_cast<int>(count);
            panel_displacements[static_cast<std::size_t>(p)] =
                static_cast<int>(displacement);
        }

        int local_ok = success ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (!global_ok) {
            success = false;
            if (rank == 0) {
                std::printf("Hybrid panel communication failed\n");
            }
            break;
        }

        MPI_Allgatherv(local_panel.empty() ? nullptr : local_panel.data(),
                       panel_counts[static_cast<std::size_t>(rank)],
                       MPI_DOUBLE, panel.data(), panel_counts.data(),
                       panel_displacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (first < n) {
            if (cuda_enabled) {
                success = cuda_matrix.uploadPanel(
                    panel, n - first, block_size_for_panel);
                if (success) {
                    success = cuda_matrix.updateTrailing(
                        k, first, block_size_for_panel);
                }
            } else {
                updateTrailingHost(local_matrix, local_range, n, k, first,
                                   block_size_for_panel, panel);
            }
        }

        local_ok = success ? 1 : 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (!global_ok) {
            success = false;
            if (rank == 0) {
                std::printf("CUDA trailing update failed\n");
            }
            break;
        }
        k = first;
    }

    if (success) {
        if (cuda_enabled) {
            success = cuda_matrix.zeroUpper();
        } else {
            zeroUpperHost(local_matrix, local_range, n);
        }
    }

    int local_success = success ? 1 : 0;
    int global_success = 0;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    success = global_success != 0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        cuda_matrix.release();
        MPI_Comm_free(&local_communicator);
        MPI_Finalize();
        return 1;
    }

    std::vector<double> result;
    std::vector<int> gather_counts;
    std::vector<int> gather_displacements;
    if (validate || print_results_flag) {
        if (cuda_enabled) {
            success = cuda_matrix.download(local_matrix);
        }
        local_success = success ? 1 : 0;
        MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (!global_success) {
            if (rank == 0) {
                std::printf("Failed to copy the result from CUDA\n");
            }
            cuda_matrix.release();
            MPI_Comm_free(&local_communicator);
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            result.resize(matrix_elements);
            gather_counts = scatter_counts;
            gather_displacements = scatter_displacements;
        }
        MPI_Gatherv(local_matrix.data(),
                    scatter_counts[static_cast<std::size_t>(rank)],
                    MPI_DOUBLE, rank == 0 ? result.data() : nullptr,
                    rank == 0 ? gather_counts.data() : nullptr,
                    rank == 0 ? gather_displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double milliseconds = elapsed_max * 1000.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / elapsed_max / 1.0e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (print_results_flag) {
            print_results(result, "CholeskyL");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(result, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            success = valid;
        }
    }

    int root_success = success ? 1 : 0;
    MPI_Bcast(&root_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cuda_matrix.release();
    MPI_Comm_free(&local_communicator);
    MPI_Finalize();
    return root_success ? 0 : 1;
}
