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

#include "../common/results_output.hpp"

namespace {

constexpr std::size_t kBlockSize = 128;
constexpr int kScatterTag = 4101;
constexpr int kOriginalGatherTag = 4102;
constexpr int kResultGatherTag = 4103;

struct RowPartition {
    std::vector<std::size_t> begin;
    std::vector<std::size_t> rows;
    std::size_t local_begin = 0;
    std::size_t local_rows = 0;
};

std::size_t row_begin(const std::size_t n, const int process_count, const int rank) {
    const std::size_t p = static_cast<std::size_t>(process_count);
    const std::size_t r = static_cast<std::size_t>(rank);
    const std::size_t base = n / p;
    const std::size_t remainder = n % p;
    return base * r + std::min(r, remainder);
}

RowPartition make_partition(const std::size_t n, const int process_count, const int rank) {
    RowPartition partition;
    partition.begin.resize(static_cast<std::size_t>(process_count));
    partition.rows.resize(static_cast<std::size_t>(process_count));

    for (int r = 0; r < process_count; ++r) {
        partition.begin[static_cast<std::size_t>(r)] = row_begin(n, process_count, r);
        const std::size_t next = r + 1 < process_count
            ? row_begin(n, process_count, r + 1)
            : n;
        partition.rows[static_cast<std::size_t>(r)] = next - partition.begin[static_cast<std::size_t>(r)];
    }

    partition.local_begin = partition.begin[static_cast<std::size_t>(rank)];
    partition.local_rows = partition.rows[static_cast<std::size_t>(rank)];
    return partition;
}

std::size_t mpi_chunk_size() {
    return static_cast<std::size_t>(std::numeric_limits<int>::max());
}

void send_doubles(const double* data, const std::size_t count, const int destination,
                  const int tag, MPI_Comm communicator) {
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t chunk = std::min(mpi_chunk_size(), count - offset);
        MPI_Send(data + offset, static_cast<int>(chunk), MPI_DOUBLE, destination, tag, communicator);
        offset += chunk;
    }
}

void receive_doubles(double* data, const std::size_t count, const int source,
                     const int tag, MPI_Comm communicator) {
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t chunk = std::min(mpi_chunk_size(), count - offset);
        MPI_Recv(data + offset, static_cast<int>(chunk), MPI_DOUBLE, source, tag,
                 communicator, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

// Gather contiguous row ranges only when validation or result printing needs the full matrix.
void gather_rows(const std::vector<double>& local_rows, std::vector<double>& global_rows,
                 const RowPartition& partition, const std::size_t n, const int rank,
                 const int tag, MPI_Comm communicator) {
    const std::size_t local_elements = partition.local_rows * n;
    if (rank == 0) {
        if (local_elements != 0) {
            std::memcpy(global_rows.data() + partition.local_begin * n,
                        local_rows.data(), local_elements * sizeof(double));
        }
        for (int r = 1; r < static_cast<int>(partition.rows.size()); ++r) {
            const std::size_t elements = partition.rows[static_cast<std::size_t>(r)] * n;
            if (elements != 0) {
                receive_doubles(global_rows.data() + partition.begin[static_cast<std::size_t>(r)] * n,
                                elements, r, tag, communicator);
            }
        }
    } else if (local_elements != 0) {
        send_doubles(local_rows.data(), local_elements, 0, tag, communicator);
    }
}

// Generate the same B as the original benchmark, preserving rand_r's serial sequence,
// and stream each contiguous row range directly to its owning rank.
void generate_and_distribute_random_rows(std::vector<double>& local_random,
                                         const RowPartition& partition, const std::size_t n,
                                         const int rank, MPI_Comm communicator) {
    if (n == 0) {
        return;
    }

    const std::size_t local_elements = partition.local_rows * n;
    if (rank != 0) {
        if (local_elements != 0) {
            receive_doubles(local_random.data(), local_elements, 0, kScatterTag, communicator);
        }
        return;
    }

    const std::size_t max_rows = *std::max_element(partition.rows.begin(), partition.rows.end());
    const std::size_t max_rows_per_message = std::max<std::size_t>(1, mpi_chunk_size() / n);
    std::vector<double> random_chunk(std::min(max_rows, max_rows_per_message) * n);
    unsigned int seed = 42;

    for (int owner = 0; owner < static_cast<int>(partition.rows.size()); ++owner) {
        const std::size_t owner_rows = partition.rows[static_cast<std::size_t>(owner)];
        for (std::size_t offset = 0; offset < owner_rows;) {
            const std::size_t chunk_rows = std::min(max_rows_per_message, owner_rows - offset);
            const std::size_t chunk_elements = chunk_rows * n;
            for (std::size_t i = 0; i < chunk_elements; ++i) {
                random_chunk[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
            }

            if (owner == 0) {
                std::memcpy(local_random.data() + offset * n, random_chunk.data(),
                            chunk_elements * sizeof(double));
            } else {
                send_doubles(random_chunk.data(), chunk_elements, owner,
                             kScatterTag, communicator);
            }
            offset += chunk_rows;
        }
    }
}

// Form the local rows of A = B B^T + n I. B is distributed by rows and is broadcast
// in row chunks, so no rank needs a second full copy of B.
void generate_positive_definite_rows(std::vector<double>& local_matrix,
                                     const std::vector<double>& local_random,
                                     const RowPartition& partition, const std::size_t n,
                                     const int rank, MPI_Comm communicator) {
    if (n == 0) {
        return;
    }

    const std::size_t max_rows_per_broadcast =
        std::max<std::size_t>(1, mpi_chunk_size() / n);
    const std::size_t max_rows = *std::max_element(partition.rows.begin(), partition.rows.end());
    std::vector<double> random_chunk(std::min(max_rows, max_rows_per_broadcast) * n);

    for (int owner = 0; owner < static_cast<int>(partition.rows.size()); ++owner) {
        const std::size_t owner_rows = partition.rows[static_cast<std::size_t>(owner)];
        const std::size_t owner_begin = partition.begin[static_cast<std::size_t>(owner)];

        for (std::size_t offset = 0; offset < owner_rows;) {
            const std::size_t chunk_rows = std::min(max_rows_per_broadcast, owner_rows - offset);
            const std::size_t chunk_elements = chunk_rows * n;

            if (rank == owner) {
                std::memcpy(random_chunk.data(), local_random.data() + offset * n,
                            chunk_elements * sizeof(double));
            }
            MPI_Bcast(random_chunk.data(), static_cast<int>(chunk_elements), MPI_DOUBLE,
                      owner, communicator);

            for (std::size_t local_i = 0; local_i < partition.local_rows; ++local_i) {
                double* output_row = local_matrix.data() + local_i * n;
                const double* b_row = local_random.data() + local_i * n;
                for (std::size_t j = 0; j < chunk_rows; ++j) {
                    const double* other_row = random_chunk.data() + j * n;
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        sum += b_row[k] * other_row[k];
                    }
                    output_row[owner_begin + offset + j] = sum;
                }
            }
            offset += chunk_rows;
        }
    }

    for (std::size_t local_i = 0; local_i < partition.local_rows; ++local_i) {
        const std::size_t global_i = partition.local_begin + local_i;
        local_matrix[local_i * n + global_i] += static_cast<double>(n);
    }
}

void pack_rows(const std::vector<double>& matrix, const std::size_t n,
               const std::size_t local_begin, const std::size_t first_row,
               const std::size_t last_row, const std::size_t first_column,
               const std::size_t column_count, std::vector<double>& packed) {
    if (first_row >= last_row || column_count == 0) {
        packed.clear();
        return;
    }
    const std::size_t row_count = last_row - first_row;
    packed.resize(row_count * column_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        const double* source = matrix.data() + (first_row - local_begin + row) * n + first_column;
        std::memcpy(packed.data() + row * column_count, source,
                    column_count * sizeof(double));
    }
}

void make_allgatherv_layout(const RowPartition& partition, const std::size_t first_row,
                            const std::size_t last_row, const std::size_t width,
                            std::vector<int>& counts, std::vector<int>& displacements) {
    const std::size_t process_count = partition.rows.size();
    counts.resize(process_count);
    displacements.resize(process_count);
    for (std::size_t r = 0; r < process_count; ++r) {
        const std::size_t local_first = std::max(first_row, partition.begin[r]);
        const std::size_t local_last = std::min(last_row, partition.begin[r] + partition.rows[r]);
        const bool intersects = local_first < local_last;
        const std::size_t count = intersects ? (local_last - local_first) * width : 0;
        const std::size_t displacement = intersects ? (local_first - first_row) * width : 0;
        if (count > mpi_chunk_size() || displacement > mpi_chunk_size()) {
            std::fprintf(stderr, "MPI block is too large for this MPI implementation\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(displacement);
    }
}

bool factor_diagonal_block(std::vector<double>& block, const std::size_t block_size,
                           const std::size_t global_begin, const int rank) {
    if (rank != 0) {
        return true;
    }

    for (std::size_t i = 0; i < block_size; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                sum += block[i * block_size + k] * block[j * block_size + k];
            }

            if (i == j) {
                const double value = block[i * block_size + i] - sum;
                if (value <= 0.0) {
                    std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                                global_begin + i);
                    return false;
                }
                block[i * block_size + i] = std::sqrt(value);
            } else {
                block[i * block_size + j] =
                    (block[i * block_size + j] - sum) / block[j * block_size + j];
            }
        }
    }
    return true;
}

bool distributed_cholesky(std::vector<double>& matrix, const std::size_t n,
                          const RowPartition& partition, const int rank,
                          MPI_Comm communicator) {
    for (std::size_t block_begin = 0; block_begin < n; block_begin += kBlockSize) {
        const std::size_t block_end = std::min(n, block_begin + kBlockSize);
        const std::size_t block_size = block_end - block_begin;

        // Assemble the small diagonal block on every rank, factor it on rank 0,
        // and broadcast the resulting lower triangular block.
        std::vector<int> diagonal_counts;
        std::vector<int> diagonal_displacements;
        make_allgatherv_layout(partition, block_begin, block_end, block_size,
                               diagonal_counts, diagonal_displacements);
        const std::size_t local_diagonal_first = std::max(block_begin, partition.local_begin);
        const std::size_t local_diagonal_last = std::min(
            block_end, partition.local_begin + partition.local_rows);
        std::vector<double> packed_diagonal;
        pack_rows(matrix, n, partition.local_begin, local_diagonal_first,
                  local_diagonal_last, block_begin, block_size, packed_diagonal);
        std::vector<double> diagonal(block_size * block_size);
        MPI_Allgatherv(packed_diagonal.empty() ? nullptr : packed_diagonal.data(),
                       static_cast<int>(packed_diagonal.size()), MPI_DOUBLE,
                       diagonal.data(), diagonal_counts.data(), diagonal_displacements.data(),
                       MPI_DOUBLE, communicator);

        int factor_success = factor_diagonal_block(diagonal, block_size, block_begin, rank) ? 1 : 0;
        MPI_Bcast(&factor_success, 1, MPI_INT, 0, communicator);
        if (factor_success == 0) {
            return false;
        }
        MPI_Bcast(diagonal.data(), static_cast<int>(diagonal.size()), MPI_DOUBLE, 0, communicator);

        // Store L_kk on its owning ranks and solve the panel below it.
        for (std::size_t local_i = 0; local_i < partition.local_rows; ++local_i) {
            const std::size_t global_i = partition.local_begin + local_i;
            double* row = matrix.data() + local_i * n;
            if (global_i >= block_begin && global_i < block_end) {
                const std::size_t diagonal_row = global_i - block_begin;
                for (std::size_t j = 0; j < block_size; ++j) {
                    row[block_begin + j] = j <= diagonal_row
                        ? diagonal[diagonal_row * block_size + j]
                        : 0.0;
                }
            } else if (global_i >= block_end) {
                for (std::size_t j = 0; j < block_size; ++j) {
                    double value = row[block_begin + j];
                    for (std::size_t k = 0; k < j; ++k) {
                        value -= row[block_begin + k] * diagonal[j * block_size + k];
                    }
                    row[block_begin + j] = value / diagonal[j * block_size + j];
                }
            }
        }

        // Replicate L_(end:n,k:end) so each rank can update its local trailing rows.
        if (block_end < n) {
            std::vector<int> panel_counts;
            std::vector<int> panel_displacements;
            make_allgatherv_layout(partition, block_end, n, block_size,
                                   panel_counts, panel_displacements);
            const std::size_t local_panel_first = std::max(block_end, partition.local_begin);
            const std::size_t local_panel_last = partition.local_begin + partition.local_rows;
            std::vector<double> packed_panel;
            pack_rows(matrix, n, partition.local_begin, local_panel_first, local_panel_last,
                      block_begin, block_size, packed_panel);
            std::vector<double> panel((n - block_end) * block_size);
            MPI_Allgatherv(packed_panel.empty() ? nullptr : packed_panel.data(),
                           static_cast<int>(packed_panel.size()), MPI_DOUBLE,
                           panel.data(), panel_counts.data(), panel_displacements.data(),
                           MPI_DOUBLE, communicator);

            for (std::size_t local_i = 0; local_i < partition.local_rows; ++local_i) {
                const std::size_t global_i = partition.local_begin + local_i;
                if (global_i < block_end) {
                    continue;
                }
                double* row = matrix.data() + local_i * n;
                const double* left_panel = row + block_begin;
                for (std::size_t j = block_end; j < n; ++j) {
                    const double* other_panel = panel.data() + (j - block_end) * block_size;
                    double update = 0.0;
                    for (std::size_t k = 0; k < block_size; ++k) {
                        update += left_panel[k] * other_panel[k];
                    }
                    row[j] -= update;
                }
            }
        }
    }

    // The benchmark exposes the full lower-triangular matrix, so clear all
    // entries above the diagonal after the distributed factorization.
    for (std::size_t local_i = 0; local_i < partition.local_rows; ++local_i) {
        const std::size_t global_i = partition.local_begin + local_i;
        double* row = matrix.data() + local_i * n;
        for (std::size_t j = global_i + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }
    return true;
}

bool validate_cholesky(const std::vector<double>& lower,
                       const std::vector<double>& original, const std::size_t n) {
    std::vector<double> reconstructed(n * n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += lower[i * n + k] * lower[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double max_error = 0.0;
    double relative_error = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - original[i]);
        max_error = std::max(max_error, error);
        relative_error = std::max(relative_error,
                                  error / (std::fabs(original[i]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    std::size_t n = 512;
    bool validate = false;
    bool print_result_output = false;
    bool parse_success = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_result_output = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                print_usage(argv[0]);
            }
            parse_success = false;
        }
    }
    if (!parse_success) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        if (rank == 0) {
            std::printf("Matrix size is too large\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (n > mpi_chunk_size()) {
        if (rank == 0) {
            std::printf("Matrix size exceeds the supported MPI count range\n");
        }
        MPI_Finalize();
        return 1;
    }

    const RowPartition partition = make_partition(n, process_count, rank);
    const std::size_t local_elements = partition.local_rows * n;
    const std::size_t global_elements = n * n;

    std::vector<double> local_matrix(local_elements);
    std::vector<double> local_random(local_elements);

    if (rank == 0) {
        std::printf("Generating positive definite matrix...\n");
    }
    generate_and_distribute_random_rows(local_random, partition, n, rank, MPI_COMM_WORLD);
    generate_positive_definite_rows(local_matrix, local_random, partition, n, rank,
                                    MPI_COMM_WORLD);
    std::vector<double>().swap(local_random);

    std::vector<double> original;
    const bool need_full_result = validate || print_result_output;
    if (validate) {
        original.resize(global_elements);
        gather_rows(local_matrix, original, partition, n, rank,
                    kOriginalGatherTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributed_cholesky(local_matrix, n, partition, rank,
                                               MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maximum_elapsed = 0.0;
    MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_success = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_success == 0) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long duration_ms = static_cast<long>(maximum_elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", duration_ms);
        const double operations = static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        const double gflops = maximum_elapsed > 0.0
            ? operations / maximum_elapsed / 1e9
            : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (need_full_result) {
        if (rank == 0) {
            result.resize(global_elements);
        }
        gather_rows(local_matrix, result, partition, n, rank,
                    kResultGatherTag, MPI_COMM_WORLD);
    }

    if (rank == 0 && print_result_output) {
        print_results(result, "CholeskyL");
    }

    int validation_success = 1;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
            validation_success = validate_cholesky(result, original, n) ? 1 : 0;
            std::printf("Validation: %s\n", validation_success ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validation_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validation_success == 1 ? 0 : 1;
}
