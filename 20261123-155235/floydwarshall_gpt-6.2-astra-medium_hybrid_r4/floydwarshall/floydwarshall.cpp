#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;
constexpr int LANES_Y = 8;
// Every real distance is bounded by its direct edge (at most 201).
using Distance = unsigned char;
constexpr Distance PAD = 255;

void fail(const char* message) {
    fprintf(stderr, "floydwarshall: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void check(cudaError_t result) {
    if (result != cudaSuccess) fail(cudaGetErrorString(result));
}

unsigned int seedAt(uint64_t index) {
    unsigned int seed = 42;
#ifdef __GLIBC__
    // glibc rand_r advances its 32-bit LCG three times per sample.
    // Exponentiate the affine transition modulo 2^32 to seek to a row.
    unsigned int mul = 1103515245u, add = 12345u;
    for (uint64_t steps = 3 * index; steps; steps >>= 1) {
        if (steps & 1) seed = mul * seed + add;
        add *= mul + 1;
        mul *= mul;
    }
#else
    for (uint64_t i = 0; i < index; ++i) rand_r(&seed);
#endif
    return seed;
}

void initialize(std::vector<Distance>& local, int first, int n, int stride) {
    const int rows = local.size() / stride;
    #pragma omp parallel
    {
        int thread = omp_get_thread_num(), threads = omp_get_num_threads();
        int begin = int(int64_t(rows) * thread / threads);
        int end = int(int64_t(rows) * (thread + 1) / threads);
        unsigned int seed = begin < end && first + begin < n
            ? seedAt(uint64_t(first + begin) * n) : 0;
        for (int i = begin; i < end; ++i) {
            Distance* row = local.data() + size_t(i) * stride;
            int j = 0;
            if (first + i < n) {
                for (; j < n; ++j)
                    row[j] = 1 + static_cast<unsigned int>(double(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                row[first + i] = 0;
            }
            std::fill(row + j, row + stride, PAD);
        }
    }
}

// Close the diagonal tile in shared memory. Pivot operands stay read-only
// during their iteration, including for padded vertices.
__global__ void closePivot(Distance* tile, int stride) {
    __shared__ unsigned int d[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < TILE; r += LANES_Y)
        d[r][x] = tile[size_t(r) * stride + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += LANES_Y)
            if (r != k && x != k)
                d[r][x] = min(d[r][x], d[r][k] + d[k][x]);
        __syncthreads();
    }
    for (int r = y; r < TILE; r += LANES_Y)
        tile[size_t(r) * stride + x] = d[r][x];
}

enum class Phase { Row, Column, Remaining };

// All three updates are tile min-plus products. Stage both operands once,
// then accumulate four output rows per thread without inner-loop barriers.
template<Phase phase>
__global__ void relax(Distance* local, const Distance* panel, int stride,
                      int pivot, int firstTile) {
    int col = phase == Phase::Column ? pivot : int(blockIdx.x);
    int row = phase == Phase::Row ? 0 : int(blockIdx.y);
    if ((phase != Phase::Column && col == pivot) ||
        (phase != Phase::Row && firstTile + row == pivot)) return;
    Distance* rows = local + size_t(row) * TILE * stride;
    Distance* output = rows + col * TILE;
    const Distance* left = rows + pivot * TILE;
    const Distance* right = panel + col * TILE;
    __shared__ unsigned int a[TILE][TILE], b[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    unsigned int best[TILE / LANES_Y];
    #pragma unroll
    for (int i = 0; i < TILE / LANES_Y; ++i) {
        int r = y + i * LANES_Y;
        size_t pos = size_t(r) * stride + x;
        a[r][x] = left[pos];
        b[r][x] = right[pos];
        best[i] = phase == Phase::Column ? a[r][x] :
                  phase == Phase::Row ? b[r][x] : output[pos];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int rhs = b[k][x];
        #pragma unroll
        for (int i = 0; i < TILE / LANES_Y; ++i)
            best[i] = min(best[i], a[y + i * LANES_Y][k] + rhs);
    }
    #pragma unroll
    for (int i = 0; i < TILE / LANES_Y; ++i)
        output[size_t(y + i * LANES_Y) * stride + x] = best[i];
}

bool validateResult(const std::vector<Distance>& local, int n, int firstRow,
                    int rows, int stride, int rank, MPI_Comm comm) {
    // Rank zero owns the first tile, which includes all sampled sources.
    int sample = std::min(10, n);
    std::vector<Distance> sources(size_t(sample) * n);
    for (int i = 0; i < sample; ++i) {
        Distance* src = sources.data() + size_t(i) * n;
        if (rank == 0) std::copy_n(local.data() + size_t(i) * stride, n, src);
        MPI_Bcast(src, n, MPI_UNSIGNED_CHAR, 0, comm);
    }
    int invalid = 0;
    #pragma omp parallel for reduction(|:invalid)
    for (int r = 0; r < rows; ++r) {
        const int k = firstRow + r;
        const Distance* row = local.data() + size_t(r) * stride;
        invalid |= row[k] != 0;
        for (int i = 0; i < sample; ++i) {
            const Distance* src = sources.data() + size_t(i) * n;
            for (int j = 0; j < sample; ++j)
                invalid |= src[k] + row[j] < src[j];
        }
    }
    int anyInvalid;
    MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm);
    return anyInvalid == 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int tileStart(int rank, int tiles, int ranks) {
    return rank * (tiles / ranks) + std::min(rank, tiles % ranks);
}

// Bound each MPI count independently of the global matrix size.
constexpr size_t MPI_CHUNK = 1u << 26;

void solve(Distance* local, int tiles, int first, int localTiles,
           int rank, int ranks, MPI_Comm comm, Distance* received, Distance* staging) {
    int stride = tiles * TILE;
    size_t panelSize = size_t(TILE) * stride;
    dim3 threads(TILE, LANES_Y);
    int owner = 0;
    for (int pivot = 0; pivot < tiles; ++pivot) {
        if (pivot == tileStart(owner + 1, tiles, ranks)) ++owner;
        Distance* panel = received;
        if (rank == owner) {
            panel = local + size_t(pivot - first) * panelSize;
            closePivot<<<1, threads>>>(panel + pivot * TILE, stride);
            if (tiles > 1)
                relax<Phase::Row><<<tiles, threads>>>(panel, panel, stride, pivot, pivot);
            if (staging)
                check(cudaMemcpy(staging, panel, panelSize * sizeof(Distance), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) {
            // MPI cannot observe pending work on the CUDA default stream.
            if (!staging) check(cudaDeviceSynchronize());
            Distance* buffer = staging ? staging : panel;
            for (size_t off = 0; off < panelSize; off += MPI_CHUNK)
                MPI_Bcast(buffer + off, std::min(MPI_CHUNK, panelSize - off),
                          MPI_UNSIGNED_CHAR, owner, comm);
            if (staging && rank != owner)
                check(cudaMemcpy(received, staging, panelSize * sizeof(Distance), cudaMemcpyHostToDevice));
        }
        // grid.y is limited to 65535 on CUDA devices.
        for (int row = 0; tiles > 1 && row < localTiles; row += 65535) {
            int count = std::min(65535, localTiles - row);
            Distance* chunk = local + size_t(row) * panelSize;
            relax<Phase::Column><<<dim3(1, count), threads>>>(chunk, panel, stride, pivot, first + row);
            relax<Phase::Remaining><<<dim3(tiles, count), threads>>>(chunk, panel, stride, pivot, first + row);
        }
        check(cudaGetLastError());
    }
    check(cudaDeviceSynchronize());
}

int benchmark(int n, bool validate, bool results, int rank, int ranks, MPI_Comm comm) {
    int tiles = (n + (TILE - 1)) / TILE;
    int first = tileStart(rank, tiles, ranks);
    int localTiles = tileStart(rank + 1, tiles, ranks) - first;
    int stride = tiles * TILE;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %d\n", n);
        printf("Validation: %s\nInitializing graph...\n", validate ? "enabled" : "disabled");
    }
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, devices;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("no CUDA devices available");
    check(cudaSetDevice(nodeRank % devices));

    size_t panelSize = size_t(TILE) * stride;
    std::vector<Distance> host(size_t(localTiles) * panelSize);
    Distance *local = nullptr, *received = nullptr, *staging = nullptr;
    if (n) {
        initialize(host, first * TILE, n, stride);
        check(cudaMalloc(&local, host.size() * sizeof(Distance)));
        check(cudaMemcpy(local, host.data(), host.size() * sizeof(Distance), cudaMemcpyHostToDevice));
        if (ranks > 1) {
            check(cudaMalloc(&received, panelSize * sizeof(Distance)));
            bool deviceMPI = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
            deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
            if (!deviceMPI) check(cudaMallocHost(&staging, panelSize * sizeof(Distance)));
        }
    }
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    solve(local, tiles, first, localTiles, rank, ranks, comm, received, staging);
    double elapsed = MPI_Wtime() - start, maximum;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(maximum * 1000));
        printf("Performance: %.3f GOPS\n", double(n) * n * n / maximum / 1e9);
    }
    if ((results || validate) && n)
        check(cudaMemcpy(host.data(), local, host.size() * sizeof(Distance), cudaMemcpyDeviceToHost));
    if (local) check(cudaFree(local));
    if (received) check(cudaFree(received));
    if (staging) check(cudaFreeHost(staging));

    int status = 0;
    int rows = std::min(localTiles * TILE, n - first * TILE);
    if (results) {
        // Pack real vertices in the original row-major, 32-bit output format.
        std::vector<unsigned int> dist(size_t(rank == 0 ? n : rows) * n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i)
            std::copy_n(host.data() + size_t(i) * stride, n, dist.data() + size_t(i) * n);
        if (rank == 0) {
            for (int peer = 1; peer < ranks; ++peer) {
                size_t begin = size_t(tileStart(peer, tiles, ranks) * TILE) * n;
                size_t end = size_t(std::min(n, tileStart(peer + 1, tiles, ranks) * TILE)) * n;
                for (size_t off = begin; off < end; off += MPI_CHUNK)
                    MPI_Recv(dist.data() + off, std::min(MPI_CHUNK, end - off), MPI_UNSIGNED,
                             peer, 0, comm, MPI_STATUS_IGNORE);
            }
            print_results_int(dist, "DistanceMatrix");
        } else {
            for (size_t off = 0; off < dist.size(); off += MPI_CHUNK)
                MPI_Send(dist.data() + off, std::min(MPI_CHUNK, dist.size() - off),
                         MPI_UNSIGNED, 0, 0, comm);
        }
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        status = validateResult(host, n, first * TILE, rows, stride, rank, comm) ? 0 : 1;
        if (rank == 0) printf("Validation: %s\n", status ? "FAILED" : "PASSED");
    }
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || value < 0 || value > INT_MAX - TILE + 1)
                fail("invalid number of nodes");
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            results = true;
        } else {
            bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }

    int active = std::min(ranks, std::max(1, (n + (TILE - 1)) / TILE));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (rank < active) {
        status = benchmark(n, validate, results, rank, active, comm);
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
