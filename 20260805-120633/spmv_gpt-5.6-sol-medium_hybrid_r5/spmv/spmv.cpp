#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        const cudaError_t error_ = (call);                                        \
        if (error_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank,    \
                         __FILE__, __LINE__, cudaGetErrorString(error_));          \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
        }                                                                        \
    } while (false)

void fill(double* values, index_t size, double max_value) {
    // Preserve the original benchmark's deterministic random stream.
    for (index_t i = 0; i < size; ++i) {
        values[i] = max_value * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* columns, index_t* row_offsets, index_t nnz,
                      index_t dimension) {
    index_t assigned = 0;
    const double probability = static_cast<double>(nnz) /
                               (static_cast<double>(dimension) * dimension);
    srand(8675309);
    bool fill_remaining = false;
    for (index_t row = 0; row < dimension; ++row) {
        row_offsets[row] = assigned;
        for (index_t column = 0; column < dimension; ++column) {
            const uint64_t position = static_cast<uint64_t>(row) * dimension + column;
            const uint64_t entries_left = static_cast<uint64_t>(dimension) * dimension - position;
            const uint64_t needed = nnz - assigned;
            if (entries_left <= needed) fill_remaining = true;
            const double random_value = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < nnz && random_value <= probability) || fill_remaining) {
                columns[assigned++] = column;
            }
        }
    }
    row_offsets[dimension] = nnz;
}

void spmvCpu(const double* values, const index_t* columns, const index_t* row_offsets,
             const double* vector, index_t rows, double* output) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        double sum = 0.0;
        for (index_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
            sum += values[entry] * vector[columns[entry]];
        }
        output[row] = sum;
    }
}

__global__ void spmvCsrWarpKernel(const double* __restrict__ values,
                                  const index_t* __restrict__ columns,
                                  const index_t* __restrict__ row_offsets,
                                  const double* __restrict__ vector,
                                  index_t rows, double* __restrict__ output) {
    constexpr unsigned FULL_WARP = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31u;
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;

    double sum = 0.0;
    for (index_t entry = row_offsets[row] + lane; entry < row_offsets[row + 1]; entry += 32) {
        sum += values[entry] * __ldg(vector + columns[entry]);
    }
    sum += __shfl_down_sync(FULL_WARP, sum, 16);
    sum += __shfl_down_sync(FULL_WARP, sum, 8);
    sum += __shfl_down_sync(FULL_WARP, sum, 4);
    sum += __shfl_down_sync(FULL_WARP, sum, 2);
    sum += __shfl_down_sync(FULL_WARP, sum, 1);
    if (lane == 0) output[row] = sum;
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    index_t first_bad = size;
#pragma omp parallel for reduction(min : first_bad) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const bool bad = std::abs(ref) < 1e-10
                             ? std::abs(res) > MAX_RELATIVE_ERROR
                             : std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        if (bad) first_bad = std::min(first_bad, static_cast<index_t>(i));
    }
    if (first_bad == size) return true;
    const double ref = reference[first_bad];
    const double res = result[first_bad];
    std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                first_bad, ref, res);
    return false;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum element value (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t num_rows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double max_value = 1.0;
    bool validate = false;
    bool print_results_requested = false;
    bool help = false;
    bool arguments_valid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_rows = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            max_value = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            arguments_valid = false;
        }
    }
    if (help || !arguments_valid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return arguments_valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const uint64_t total_entries = static_cast<uint64_t>(num_rows) * num_rows;
    const uint64_t nnz64 = sparsity == 0 ? 0 : total_entries / sparsity;
    if (num_rows == 0 || sparsity == 0 || iterations == 0 || nnz64 > std::numeric_limits<index_t>::max() ||
        nnz64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Invalid dimensions: require positive arguments and at most INT_MAX nonzeros\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const index_t total_nnz = static_cast<index_t>(nnz64);

    // Contiguous rows keep output communication-free and preserve CSR ordering.
    std::vector<int> row_counts(ranks), row_displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const uint64_t begin = static_cast<uint64_t>(num_rows) * r / ranks;
        const uint64_t end = static_cast<uint64_t>(num_rows) * (r + 1) / ranks;
        row_displacements[r] = static_cast<int>(begin);
        row_counts[r] = static_cast<int>(end - begin);
    }
    const index_t local_rows = static_cast<index_t>(row_counts[rank]);

    std::vector<double> full_values;
    std::vector<index_t> full_columns;
    std::vector<index_t> full_row_offsets;
    std::vector<double> dense_vector(num_rows);
    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", num_rows, num_rows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", total_nnz,
                    100.0 * (1.0 - static_cast<double>(total_nnz) / total_entries));
        std::printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations,
                    max_value, validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s) per rank\n",
                    ranks, omp_get_max_threads());
        std::printf("Initializing data structures...\n");
        full_values.resize(total_nnz);
        full_columns.resize(total_nnz);
        full_row_offsets.resize(static_cast<size_t>(num_rows) + 1);
        fill(dense_vector.data(), num_rows, max_value);
        fill(full_values.data(), total_nnz, max_value);
        initRandomMatrix(full_columns.data(), full_row_offsets.data(), total_nnz, num_rows);
    }
    MPI_Bcast(dense_vector.data(), static_cast<int>(num_rows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> nnz_counts(ranks), nnz_displacements(ranks), offset_counts(ranks);
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r) {
            const index_t begin = full_row_offsets[row_displacements[r]];
            const index_t end = full_row_offsets[row_displacements[r] + row_counts[r]];
            nnz_displacements[r] = static_cast<int>(begin);
            nnz_counts[r] = static_cast<int>(end - begin);
            offset_counts[r] = row_counts[r] + 1;
        }
    }
    int local_nnz_count = 0;
    MPI_Scatter(nnz_counts.data(), 1, MPI_INT, &local_nnz_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t local_nnz = static_cast<index_t>(local_nnz_count);
    std::vector<double> local_values(local_nnz);
    std::vector<index_t> local_columns(local_nnz);
    std::vector<index_t> local_row_offsets(static_cast<size_t>(local_rows) + 1);
    MPI_Scatterv(full_values.data(), nnz_counts.data(), nnz_displacements.data(), MPI_DOUBLE,
                 local_values.data(), local_nnz_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(full_columns.data(), nnz_counts.data(), nnz_displacements.data(), MPI_UINT32_T,
                 local_columns.data(), local_nnz_count, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(full_row_offsets.data(), offset_counts.data(), row_displacements.data(), MPI_UINT32_T,
                 local_row_offsets.data(), static_cast<int>(local_rows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t offset_base = local_row_offsets[0];
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row <= static_cast<int64_t>(local_rows); ++row) {
        local_row_offsets[row] -= offset_base;
    }

    // Assign node-local ranks round-robin to the available accelerators.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA accelerator is available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    double *device_values = nullptr, *device_vector = nullptr, *device_output = nullptr;
    index_t *device_columns = nullptr, *device_row_offsets = nullptr;
    if (local_nnz != 0) {
        CUDA_CHECK(cudaMalloc(&device_values, static_cast<size_t>(local_nnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&device_columns, static_cast<size_t>(local_nnz) * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(device_values, local_values.data(), static_cast<size_t>(local_nnz) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_columns, local_columns.data(), static_cast<size_t>(local_nnz) * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&device_row_offsets, (static_cast<size_t>(local_rows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&device_vector, static_cast<size_t>(num_rows) * sizeof(double)));
    if (local_rows != 0) CUDA_CHECK(cudaMalloc(&device_output, static_cast<size_t>(local_rows) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(device_row_offsets, local_row_offsets.data(), (static_cast<size_t>(local_rows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_vector, dense_vector.data(), static_cast<size_t>(num_rows) * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((static_cast<uint64_t>(local_rows) * 32 + threads - 1) / threads);
    if (local_rows != 0) spmvCsrWarpKernel<<<blocks, threads>>>(device_values, device_columns,
        device_row_offsets, device_vector, local_rows, device_output);
    CUDA_CHECK(cudaDeviceSynchronize()); // warm-up/JIT is deliberately outside timing

    if (rank == 0) std::printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        if (local_rows != 0) spmvCsrWarpKernel<<<blocks, threads>>>(device_values, device_columns,
            device_row_offsets, device_vector, local_rows, device_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_output(local_rows);
    if (local_rows != 0) CUDA_CHECK(cudaMemcpy(local_output.data(), device_output,
        static_cast<size_t>(local_rows) * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> output;
    if (rank == 0 && (validate || print_results_requested)) output.resize(num_rows);
    if (validate || print_results_requested) {
        MPI_Gatherv(local_output.data(), static_cast<int>(local_rows), MPI_DOUBLE, output.data(),
                    row_counts.data(), row_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_vector));
    CUDA_CHECK(cudaFree(device_row_offsets));
    CUDA_CHECK(cudaFree(device_columns));
    CUDA_CHECK(cudaFree(device_values));

    int exit_code = EXIT_SUCCESS;
    if (rank == 0) {
        const double milliseconds = elapsed * 1.0e3;
        const double average_ms = milliseconds / iterations;
        const double gflops = 2.0 * total_nnz * iterations / elapsed / 1.0e9;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.3f ms\n", average_ms);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (print_results_requested) print_results(output, "OutputVector");
        if (validate) {
            std::printf("Computing reference solution...\n");
            std::vector<double> reference(num_rows);
            spmvCpu(full_values.data(), full_columns.data(), full_row_offsets.data(),
                    dense_vector.data(), num_rows, reference.data());
            std::printf("Validating result...\n");
            if (verifyResults(reference.data(), output.data(), num_rows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exit_code = EXIT_FAILURE;
            }
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
