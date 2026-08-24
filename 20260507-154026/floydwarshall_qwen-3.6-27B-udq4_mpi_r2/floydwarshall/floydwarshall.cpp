#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; k++) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Parse arguments on rank 0, then broadcast ---
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Abort(MPI_COMM_WORLD, 0);
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }

    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int vf = validate ? 1 : 0, pf = printResults ? 1 : 0;
    MPI_Bcast(&vf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = vf != 0;
    printResults = pf != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // --- 2-D block decomposition of the N×N matrix ---
    int qr = 1;
    for (int i = (int)std::sqrt((double)size); i >= 1; --i)
        if (size % i == 0) { qr = i; break; }
    int qc = size / qr;
    int my_qr = rank / qc, my_qc = rank % qc;

    size_t row_blk = (numNodes + qr - 1) / qr;
    size_t col_blk = (numNodes + qc - 1) / qc;

    size_t my_row_start = my_qr * row_blk;
    size_t my_row_end   = std::min(my_row_start + row_blk, numNodes);
    size_t my_col_start = my_qc * col_blk;
    size_t my_col_end   = std::min(my_col_start + col_blk, numNodes);
    size_t my_rows = my_row_end - my_row_start;
    size_t my_cols = my_col_end - my_col_start;

    // Local blocks (row-major: local[li*my_cols + lj] == global[my_row_start+li][my_col_start+lj])
    std::vector<unsigned int> dist_local(my_rows * my_cols);
    std::vector<unsigned int> path_local(my_rows * my_cols);

    // --- Initialize distance matrix (deterministic, same seed everywhere) ---
    if (rank == 0) printf("Initializing graph...\n");

    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    for (size_t j = 0; j < numNodes; ++j) {
        if (j < my_col_start || j >= my_col_end) {
            for (size_t i = 0; i < numNodes; ++i) rand_r(&seed);
            continue;
        }
        for (size_t i = 0; i < my_row_start; ++i) rand_r(&seed);

        size_t lj = j - my_col_start;
        for (size_t li = 0; li < my_rows; ++li) {
            dist_local[li * my_cols + lj] =
                1u + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
        for (size_t i = my_row_end; i < numNodes; ++i) rand_r(&seed);
    }

    // Zero diagonal
    for (size_t li = 0; li < my_rows; ++li) {
        size_t gi = my_row_start + li;
        if (gi >= my_col_start && gi < my_col_end)
            dist_local[li * my_cols + (gi - my_col_start)] = 0;
    }

    // Initialize path: path[i][j] = j
    for (size_t li = 0; li < my_rows; ++li)
        for (size_t lj = 0; lj < my_cols; ++lj)
            path_local[li * my_cols + lj] = (unsigned int)(my_col_start + lj);

    // --- Set up row / column MPI communicators ---
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_qr, my_qc, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_qc, my_qr, &col_comm);

    // --- Parallel Floyd-Warshall ---
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned int> col_k_piece(my_rows);  // dist[i][k] for my rows
    std::vector<unsigned int> row_k_piece(my_cols);  // dist[k][j] for my cols

    for (size_t k = 0; k < numNodes; ++k) {
        int k_qr = (int)(k / row_blk);
        int k_qc = (int)(k / col_blk);

        // Broadcast column k within each row communicator
        {
            int root = k_qc;
            if (my_qc == k_qc) {
                size_t kcl = k % col_blk;
                for (size_t li = 0; li < my_rows; ++li)
                    col_k_piece[li] = dist_local[li * my_cols + kcl];
            }
            MPI_Bcast(col_k_piece.data(), (int)my_rows, MPI_UNSIGNED, root, row_comm);
        }

        // Broadcast row k within each column communicator
        {
            int root = k_qr;
            if (my_qr == k_qr) {
                size_t krl = k % row_blk;
                for (size_t lj = 0; lj < my_cols; ++lj)
                    row_k_piece[lj] = dist_local[krl * my_cols + lj];
            }
            MPI_Bcast(row_k_piece.data(), (int)my_cols, MPI_UNSIGNED, root, col_comm);
        }

        // Update local block: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        for (size_t li = 0; li < my_rows; ++li) {
            unsigned int dik = col_k_piece[li];
            for (size_t lj = 0; lj < my_cols; ++lj) {
                unsigned int nd = dik + row_k_piece[lj];
                size_t idx = li * my_cols + lj;
                if (nd < dist_local[idx]) {
                    dist_local[idx] = nd;
                    path_local[idx] = (unsigned int)k;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    MPI_Barrier(MPI_COMM_WORLD);

    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // --- Gather full matrices on rank 0 ---
    std::vector<int> recv_counts(size), displs(size);
    int total = 0;
    for (int r = 0; r < size; ++r) {
        int r_qr = r / qc, r_qc = r % qc;
        size_t r_rs = r_qr * row_blk;
        size_t r_re = std::min(r_rs + row_blk, numNodes);
        size_t r_cs = r_qc * col_blk;
        size_t r_ce = std::min(r_cs + col_blk, numNodes);
        recv_counts[r] = (int)((r_re - r_rs) * (r_ce - r_cs));
        displs[r] = total;
        total += recv_counts[r];
    }

    std::vector<unsigned int> dist_gathered(total), path_gathered(total);
    MPI_Gatherv(dist_local.data(),  (int)(my_rows * my_cols), MPI_UNSIGNED,
                dist_gathered.data(), recv_counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(path_local.data(),  (int)(my_rows * my_cols), MPI_UNSIGNED,
                path_gathered.data(), recv_counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Reconstruct flat arrays on rank 0
    std::vector<unsigned int> dist_full, path_full;
    if (rank == 0) {
        dist_full.resize(numNodes * numNodes);
        path_full.resize(numNodes * numNodes);

        int off = 0;
        for (int r = 0; r < size; ++r) {
            int r_qr = r / qc, r_qc = r % qc;
            size_t r_rs = r_qr * row_blk;
            size_t r_re = std::min(r_rs + row_blk, numNodes);
            size_t r_cs = r_qc * col_blk;
            size_t r_ce = std::min(r_cs + col_blk, numNodes);
            size_t rr = r_re - r_rs, rc = r_ce - r_cs;

            for (size_t li = 0; li < rr; ++li)
                for (size_t lj = 0; lj < rc; ++lj) {
                    size_t gi = r_rs + li, gj = r_cs + lj;
                    dist_full[idx2(gi, gj, numNodes)] = dist_gathered[off + (int)(li * rc + lj)];
                    path_full[idx2(gi, gj, numNodes)] = path_gathered[off + (int)(li * rc + lj)];
                }
            off += recv_counts[r];
        }
    }

    // --- Results & validation on rank 0 ---
    if (printResults && rank == 0) {
        print_results_int(dist_full, "DistanceMatrix");
    }

    if (validate && rank == 0) {
        printf("Validating result...\n");
        if (validateResult(dist_full, numNodes))
            printf("Validation: PASSED\n");
        else
            printf("Validation: FAILED\n");
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();

    return 0;
}
