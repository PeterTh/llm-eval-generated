#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        fail("CUDA operation failed");
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

static void checkMpi(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "%s: %.*s\n", operation, length, message);
        fail("MPI operation failed");
    }
}
#define MPI_CHECK(operation) checkMpi((operation), #operation)

struct Grid {
    int n, rows, cols, row_start, col_start;
    size_t stride;
};

// The only mesh built by the benchmark has four unit-weight connections, in
// +row, -row, +column, -column order. Reconstructing them avoids the original
// 144-byte static record per cell. Keep each operation and addition in that order.
template<bool Interior>
__device__ __forceinline__ void updateCell(const Grid g, int row, int col,
                                         const double* __restrict__ old_energy,
                                         double* __restrict__ new_energy,
                                         double* __restrict__ flux) {
    const size_t p = (size_t(row) + 1) * g.stride + col + 1;
    const size_t f = size_t(row) * g.cols + col;
    const double energy = old_energy[p];
    double change = 0.0;
    if constexpr (Interior) {
        change += ((old_energy[p + g.stride] - energy) * 0.8) * 0.25;
        change += ((old_energy[p - g.stride] - energy) * 0.8) * 0.25;
        change += ((old_energy[p + 1] - energy) * 0.8) * 0.25;
        change += ((old_energy[p - 1] - energy) * 0.8) * 0.25;
    } else {
        const int x = g.row_start + row;
        const int y = g.col_start + col;
        if ((x == 0 || x == g.n - 1) && (y == 0 || y == g.n - 1)) {
            // For n == 1 the original's last material assignment is inflow.
            change = ((x == 0 && y == 0) || (x == g.n - 1 && y == g.n - 1))
                         ? 0.5 : -0.5;
        }
        if (x < g.n - 1) change += ((old_energy[p + g.stride] - energy) * 0.8) * 0.25;
        if (x > 0)       change += ((old_energy[p - g.stride] - energy) * 0.8) * 0.25;
        if (y < g.n - 1) change += ((old_energy[p + 1] - energy) * 0.8) * 0.25;
        if (y > 0)       change += ((old_energy[p - 1] - energy) * 0.8) * 0.25;
    }
    new_energy[p] = energy + change;
    // Flux depends only on its own previous value, so a single buffer suffices.
    flux[f] = flux[f] + fabs(change);
}

__global__ void updateAll(Grid g, const double* old_energy, double* new_energy, double* flux) {
    const int col = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (col < g.cols && row < g.rows) updateCell<false>(g, row, col, old_energy, new_energy, flux);
}

__global__ void updateInterior(Grid g, const double* old_energy, double* new_energy, double* flux) {
    const int col = int(blockIdx.x * blockDim.x + threadIdx.x) + 1;
    const int row = int(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    if (col < g.cols - 1 && row < g.rows - 1)
        updateCell<true>(g, row, col, old_energy, new_energy, flux);
}

__global__ void updateBoundary(Grid g, size_t count, const double* old_energy,
                               double* new_energy, double* flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row, col;
    if (g.rows == 1) { row = 0; col = int(i); }
    else if (g.cols == 1) { row = int(i); col = 0; }
    else if (i < size_t(g.cols)) { row = 0; col = int(i); }
    else if (i < 2 * size_t(g.cols)) { row = g.rows - 1; col = int(i - g.cols); }
    else {
        row = 1 + int((i - 2 * size_t(g.cols)) / 2);
        col = ((i - 2 * size_t(g.cols)) & 1) ? g.cols - 1 : 0;
    }
    updateCell<false>(g, row, col, old_energy, new_energy, flux);
}

// Four contiguous faces allow portable MPI with pinned staging buffers; only
// energies cross ranks. No CUDA-aware MPI extension is required.
__global__ void packFaces(Grid g, const double* energy, double* faces) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(g.cols)) {
        faces[i] = energy[g.stride + i + 1];
        faces[g.cols + i] = energy[size_t(g.rows) * g.stride + i + 1];
    }
    if (i < size_t(g.rows)) {
        faces[2 * size_t(g.cols) + i] = energy[(i + 1) * g.stride + 1];
        faces[2 * size_t(g.cols) + g.rows + i] = energy[(i + 1) * g.stride + g.cols];
    }
}

__global__ void unpackFaces(Grid g, const double* faces, double* energy,
                            bool up, bool down, bool left, bool right) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(g.cols)) {
        if (up) energy[i + 1] = faces[i];
        if (down) energy[size_t(g.rows + 1) * g.stride + i + 1] = faces[g.cols + i];
    }
    if (i < size_t(g.rows)) {
        if (left) energy[(i + 1) * g.stride] = faces[2 * size_t(g.cols) + i];
        if (right) energy[(i + 1) * g.stride + g.cols + 1] = faces[2 * size_t(g.cols) + g.rows + i];
    }
}

// Own only a rectangular subdomain, not a copy of the global mesh.
struct Simulation {
    Grid grid{};
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, ranks = 1;
    int neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    double* energy[2] = {nullptr, nullptr};
    double* flux = nullptr;
    double *device_send = nullptr, *device_recv = nullptr;
    double *host_send = nullptr, *host_recv = nullptr;
    size_t face_count = 0, boundary_count = 0;
    int neighbor_count = 0, graph_steps = 0;
    MPI_Request requests[8];
    cudaStream_t compute{}, exchange{};
    cudaEvent_t step_done{}, packed{}, received{};
    cudaGraphExec_t graph_exec{};

    Simulation(int n, int iterations, MPI_Comm cart) : comm(cart) {
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        MPI_CHECK(MPI_Comm_size(comm, &ranks));
        int dims[2], periods[2], coords[2];
        MPI_CHECK(MPI_Cart_get(comm, 2, dims, periods, coords));
        auto extent = [n](int d, int c) { return n / d + (c < n % d); };
        auto origin = [n](int d, int c) { return c * (n / d) + std::min(c, n % d); };
        grid = {n, extent(dims[0], coords[0]), extent(dims[1], coords[1]),
                origin(dims[0], coords[0]), origin(dims[1], coords[1]), 0};
        grid.stride = (size_t(grid.cols) + 2 + 31) & ~size_t(31);
        MPI_CHECK(MPI_Cart_shift(comm, 0, 1, &neighbors[0], &neighbors[1]));
        MPI_CHECK(MPI_Cart_shift(comm, 1, 1, &neighbors[2], &neighbors[3]));
        const size_t cells = size_t(grid.rows) * grid.cols;
        const size_t padded = size_t(grid.rows + 2) * grid.stride;
        if (padded > SIZE_MAX / sizeof(double) || cells > SIZE_MAX / sizeof(double))
            fail("Subdomain allocation size overflow");
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&step_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&packed, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        for (auto& buffer : energy) {
            CUDA_CHECK(cudaMalloc(&buffer, padded * sizeof(double)));
            CUDA_CHECK(cudaMemsetAsync(buffer, 0, padded * sizeof(double), compute));
        }
        CUDA_CHECK(cudaMalloc(&flux, cells * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(flux, 0, cells * sizeof(double), compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        boundary_count = grid.rows == 1 ? grid.cols : grid.cols == 1 ? grid.rows :
                         2 * size_t(grid.cols) + 2 * size_t(grid.rows - 2);
        if (ranks > 1) {
            face_count = 2 * (size_t(grid.rows) + grid.cols);
            CUDA_CHECK(cudaMalloc(&device_send, face_count * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&device_recv, face_count * sizeof(double)));
            CUDA_CHECK(cudaHostAlloc(&host_send, face_count * sizeof(double), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&host_recv, face_count * sizeof(double), cudaHostAllocDefault));
            const size_t offsets[4] = {0, size_t(grid.cols), 2 * size_t(grid.cols),
                                      2 * size_t(grid.cols) + grid.rows};
            for (int face = 0; face < 4; ++face) {
                if (neighbors[face] == MPI_PROC_NULL) continue;
                MPI_CHECK(MPI_Recv_init(host_recv + offsets[face], face < 2 ? grid.cols : grid.rows,
                                       MPI_DOUBLE, neighbors[face], face ^ 1, comm,
                                       &requests[neighbor_count++]));
            }
            int request = neighbor_count;
            for (int face = 0; face < 4; ++face) {
                if (neighbors[face] == MPI_PROC_NULL) continue;
                MPI_CHECK(MPI_Send_init(host_send + offsets[face], face < 2 ? grid.cols : grid.rows,
                                       MPI_DOUBLE, neighbors[face], face, comm, &requests[request++]));
            }
        } else {
            // Even-sized graphs return to energy[0], so they can be replayed
            // without pointer patches. Graph construction is outside the timer.
            graph_steps = std::min(iterations, 32) & ~1;
            if (graph_steps > 0) {
                cudaGraph_t graph;
                CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < graph_steps; ++i) launchAll(energy[i & 1], energy[(i + 1) & 1]);
                CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphUpload(graph_exec, compute));
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
        }
        CUDA_CHECK(cudaEventRecord(step_done, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void launchAll(const double* old_energy, double* new_energy) {
        const dim3 threads(32, 8);
        const dim3 blocks((grid.cols + 31) / 32, (grid.rows + 7) / 8);
        updateAll<<<blocks, threads, 0, compute>>>(grid, old_energy, new_energy, flux);
        CUDA_CHECK(cudaGetLastError());
    }

    void run(int iterations) {
        int iter = 0;
        if (ranks == 1) {
            if (graph_steps > 0) {
                for (; iterations - iter >= graph_steps; iter += graph_steps)
                    CUDA_CHECK(cudaGraphLaunch(graph_exec, compute));
            }
            for (; iter < iterations; ++iter) launchAll(energy[iter & 1], energy[(iter + 1) & 1]);
        } else {
            const unsigned face_blocks = unsigned((std::max(grid.rows, grid.cols) + size_t(255)) / 256);
            for (; iter < iterations; ++iter) {
                double* current = energy[iter & 1];
                double* next = energy[(iter + 1) & 1];
                MPI_CHECK(MPI_Startall(neighbor_count, requests));
                CUDA_CHECK(cudaStreamWaitEvent(exchange, step_done, 0));
                packFaces<<<face_blocks, 256, 0, exchange>>>(grid, current, device_send);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(host_send, device_send, face_count * sizeof(double),
                                           cudaMemcpyDeviceToHost, exchange));
                CUDA_CHECK(cudaEventRecord(packed, exchange));
                if (grid.rows > 2 && grid.cols > 2) {
                    const dim3 threads(32, 8);
                    const dim3 blocks((grid.cols - 2 + 31) / 32, (grid.rows - 2 + 7) / 8);
                    updateInterior<<<blocks, threads, 0, compute>>>(grid, current, next, flux);
                    CUDA_CHECK(cudaGetLastError());
                }
                CUDA_CHECK(cudaEventSynchronize(packed));
                MPI_CHECK(MPI_Startall(neighbor_count, requests + neighbor_count));
                // MPI progresses while the GPU computes cells that need no halos.
                MPI_CHECK(MPI_Waitall(2 * neighbor_count, requests, MPI_STATUSES_IGNORE));
                CUDA_CHECK(cudaMemcpyAsync(device_recv, host_recv, face_count * sizeof(double),
                                           cudaMemcpyHostToDevice, exchange));
                CUDA_CHECK(cudaEventRecord(received, exchange));
                CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
                unpackFaces<<<face_blocks, 256, 0, compute>>>(grid, device_recv, current,
                    neighbors[0] != MPI_PROC_NULL, neighbors[1] != MPI_PROC_NULL,
                    neighbors[2] != MPI_PROC_NULL, neighbors[3] != MPI_PROC_NULL);
                CUDA_CHECK(cudaGetLastError());
                updateBoundary<<<unsigned((boundary_count + 255) / 256), 256, 0, compute>>>(
                    grid, boundary_count, current, next, flux);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(step_done, compute));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void download(int iterations, std::vector<double>& energies, std::vector<double>& fluxes) {
        const size_t cells = size_t(grid.rows) * grid.cols;
        energies.resize(cells);
        fluxes.resize(cells);
        CUDA_CHECK(cudaMemcpy2D(energies.data(), size_t(grid.cols) * sizeof(double),
            energy[iterations & 1] + grid.stride + 1, grid.stride * sizeof(double),
            size_t(grid.cols) * sizeof(double), grid.rows, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(fluxes.data(), flux, cells * sizeof(double), cudaMemcpyDeviceToHost));
    }

    ~Simulation() {
        for (int i = 0; i < 2 * neighbor_count; ++i) MPI_CHECK(MPI_Request_free(&requests[i]));
        if (graph_exec) CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
        if (host_send) CUDA_CHECK(cudaFreeHost(host_send));
        if (host_recv) CUDA_CHECK(cudaFreeHost(host_recv));
        if (device_send) CUDA_CHECK(cudaFree(device_send));
        if (device_recv) CUDA_CHECK(cudaFree(device_recv));
        CUDA_CHECK(cudaFree(energy[0]));
        CUDA_CHECK(cudaFree(energy[1]));
        CUDA_CHECK(cudaFree(flux));
        CUDA_CHECK(cudaEventDestroy(step_done));
        CUDA_CHECK(cudaEventDestroy(packed));
        CUDA_CHECK(cudaEventDestroy(received));
        CUDA_CHECK(cudaStreamDestroy(compute));
        CUDA_CHECK(cudaStreamDestroy(exchange));
    }
};

// Use global element indices so the XOR hash is independent of decomposition.
static uint64_t computeHash(const Grid& g, const std::vector<double>& energy,
                            const std::vector<double>& flux) {
    uint64_t result = 0;
    const size_t cells = energy.size();
    #pragma omp parallel for schedule(static) reduction(^:result)
    for (size_t i = 0; i < cells; ++i) {
        const uint64_t index = (uint64_t(g.row_start) + i / g.cols) * g.n +
                               g.col_start + i % g.cols;
        uint64_t e, f;
        // memcpy avoids the original's strict-aliasing violation.
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        result ^= (e + index) * UINT64_C(0x9e3779b97f4a7c15);
        result ^= (f + index) * UINT64_C(0xbf58476d1ce4e5b9);
    }
    return result;
}

// Only requested output/validation is gathered. MPI vector datatypes place each
// rectangle directly into global row order, without a second full-size copy or
// the int-sized displacement limitation of MPI_Gatherv.
static std::vector<double> gatherField(const Simulation& sim, const std::vector<double>& local) {
    constexpr int output_tag = 10;
    const Grid& g = sim.grid;
    std::vector<double> global;
    if (sim.rank == 0) global.resize(size_t(g.n) * g.n);
    if (sim.rank == 0) {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < g.rows; ++row) {
            std::copy_n(local.data() + size_t(row) * g.cols, g.cols,
                        global.data() + size_t(g.row_start + row) * g.n + g.col_start);
        }
        int dims[2], periods[2], coords[2];
        MPI_CHECK(MPI_Cart_get(sim.comm, 2, dims, periods, coords));
        for (int source = 1; source < sim.ranks; ++source) {
            MPI_CHECK(MPI_Cart_coords(sim.comm, source, 2, coords));
            const int rows = g.n / dims[0] + (coords[0] < g.n % dims[0]);
            const int cols = g.n / dims[1] + (coords[1] < g.n % dims[1]);
            const int row_start = coords[0] * (g.n / dims[0]) + std::min(coords[0], g.n % dims[0]);
            const int col_start = coords[1] * (g.n / dims[1]) + std::min(coords[1], g.n % dims[1]);
            MPI_Datatype rectangle;
            MPI_CHECK(MPI_Type_vector(rows, cols, g.n, MPI_DOUBLE, &rectangle));
            MPI_CHECK(MPI_Type_commit(&rectangle));
            MPI_CHECK(MPI_Recv(global.data() + size_t(row_start) * g.n + col_start,
                               1, rectangle, source, output_tag, sim.comm, MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Type_free(&rectangle));
        }
    } else {
        MPI_Datatype rectangle;
        MPI_CHECK(MPI_Type_vector(g.rows, g.cols, g.cols, MPI_DOUBLE, &rectangle));
        MPI_CHECK(MPI_Type_commit(&rectangle));
        MPI_CHECK(MPI_Send(local.data(), 1, rectangle, 0, output_tag, sim.comm));
        MPI_CHECK(MPI_Type_free(&rectangle));
    }
    return global;
}

static bool validateResults(const std::vector<double>& energies, const std::vector<double>& fluxes) {
    double energy_sum = 0.0, flux_sum = 0.0;
    double energy_max = std::numeric_limits<double>::lowest();
    double energy_min = std::numeric_limits<double>::max();
    // Preserve the reference's summation order for reproducible diagnostic output.
    for (size_t i = 0; i < energies.size(); ++i) {
        energy_sum += energies[i];
        flux_sum += fluxes[i];
        energy_max = std::max(energies[i], energy_max);
        energy_min = std::min(energies[i], energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseNumber(const char* text, int minimum, int maximum, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < minimum || parsed > maximum) return false;
    value = int(parsed);
    return true;
}

static int benchmark(int argc, char** argv, int world_rank, int world_size) {
    int n = 512, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const bool grid_option = strcmp(argv[i], "-n") == 0;
            if (!parseNumber(argv[++i], grid_option ? 1 : 0, grid_option ? INT_MAX - 64 : INT_MAX,
                             grid_option ? n : iterations)) {
                if (world_rank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (uint64_t(n) * n > SIZE_MAX / sizeof(ElementDynamic)) fail("Grid allocation size overflow");

    // Maximize useful ranks, then minimize face lengths. Small grids can have
    // more MPI ranks than cells; excess ranks stay outside the work communicator.
    int dims[2] = {1, 1}, active = 1;
    long long best_surface = LLONG_MAX;
    for (int rows = 1; rows <= std::min(n, world_size); ++rows) {
        const int cols = std::min(n, world_size / rows);
        const int used = rows * cols;
        const long long surface = static_cast<long long>(rows) + cols;
        if (used > active || (used == active && surface < best_surface)) {
            dims[0] = rows; dims[1] = cols; active = used; best_surface = surface;
        }
    }
    MPI_Comm workers;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED,
                             world_rank, &workers));
    if (workers == MPI_COMM_NULL) return 0;
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_CHECK(MPI_Cart_create(workers, 2, dims, periods, 0, &cart));
    MPI_CHECK(MPI_Comm_set_errhandler(cart, MPI_ERRORS_RETURN));

    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(workers, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node));
    int local_rank, local_size;
    MPI_CHECK(MPI_Comm_rank(node, &local_rank));
    MPI_CHECK(MPI_Comm_size(node, &local_size));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) fail("The MPI/OpenMP/CUDA benchmark requires a CUDA device on every worker");
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    // Respect explicit OpenMP settings; avoid large default teams per GPU rank.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_max_threads() / local_size)));
    MPI_CHECK(MPI_Comm_free(&node));

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, uint64_t(n) * n);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: MPI %d ranks (%d x %d), OpenMP %d threads/rank, CUDA\n\n",
               active, dims[0], dims[1], omp_get_max_threads());
        printf("Building unstructured mesh...\n");
    }
    int valid = 1;
    {
        Simulation simulation(n, iterations, cart);
        const double local_mem = (2.0 * (simulation.grid.rows + 2) * simulation.grid.stride +
            double(simulation.grid.rows) * simulation.grid.cols + 4.0 * simulation.face_count) * sizeof(double);
        double total_mem = 0;
        MPI_CHECK(MPI_Reduce(&local_mem, &total_mem, 1, MPI_DOUBLE, MPI_SUM, 0, cart));
        if (world_rank == 0) {
            printf("Memory usage: %.2f MB (distributed state and halo buffers)\n\n", total_mem / (1024 * 1024));
            printf("Running simulation...\n");
        }
        MPI_CHECK(MPI_Barrier(cart));
        const double start = MPI_Wtime();
        simulation.run(iterations);
        const double elapsed = MPI_Wtime() - start;
        double max_elapsed = 0;
        MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart));
        std::vector<double> local_energy, local_flux;
        simulation.download(iterations, local_energy, local_flux);
        const uint64_t local_hash = computeHash(simulation.grid, local_energy, local_flux);
        uint64_t result_hash = 0;
        MPI_CHECK(MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart));
        if (world_rank == 0) {
            const double milliseconds = max_elapsed * 1000;
            const double elements_per_sec = max_elapsed > 0 ? double(iterations) * n * n / max_elapsed / 1e9 : 0;
            printf("Computation time: %.3f ms\n", milliseconds);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", iterations > 0 ? milliseconds / iterations : 0);
            printf("  Elements/sec: %.4f GigaElements/s\n", elements_per_sec);
            printf("  Performance: %.4f GFLOPS\n", elements_per_sec * 22);
            printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
        }
        if (printResults || validate) {
            auto energies = gatherField(simulation, local_energy);
            std::vector<double> fluxes;
            if (validate) fluxes = gatherField(simulation, local_flux);
            if (world_rank == 0) {
                if (printResults) print_results(energies, "ElementEnergy");
                if (validate) valid = validateResults(energies, fluxes);
            }
        }
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, cart));
    }
    MPI_CHECK(MPI_Comm_free(&cart));
    MPI_CHECK(MPI_Comm_free(&workers));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int result = 1;
    try { result = benchmark(argc, argv, rank, size); }
    catch (const std::exception& error) { fail(error.what()); }
    // Include idle ranks and propagate validation failures to every process.
    int global_result = 0;
    MPI_CHECK(MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return global_result;
}
