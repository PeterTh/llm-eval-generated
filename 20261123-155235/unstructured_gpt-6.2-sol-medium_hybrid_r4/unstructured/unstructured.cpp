#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <mpi.h>
#include <omp.h>
#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

namespace {

constexpr int block_size = 256;
enum Side { top, bottom, left, right, sides };
constexpr Side opposite[] = {bottom, top, right, left};

[[noreturn]] void abortRun(const char* message, const char* detail = "") {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s%s\n", rank, message, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        abortRun("CUDA: ", cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count) {
            checkCuda(cudaMalloc(&data, count * sizeof(T)));
        }
    }
    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

struct Tile {
    int row, col, rows, cols;
};

struct Partition {
    int rows, cols;

    Tile tile(int rank, int grid) const {
        const int row = rank / cols;
        const int col = rank % cols;
        const int height = grid / rows;
        const int width = grid / cols;
        const int extra_rows = grid % rows;
        const int extra_cols = grid % cols;
        return {row * height + std::min(row, extra_rows),
                col * width + std::min(col, extra_cols),
                height + (row < extra_rows), width + (col < extra_cols)};
    }
};

Partition partitionGrid(int grid, int ranks) {
    Partition best{1, 1};
    // Maximize the number of nonempty tiles, favoring a square process grid.
    for (int rows = 1; rows <= grid && rows <= ranks / rows; ++rows) {
        const int cols = std::min(grid, ranks / rows);
        if (rows * cols >= best.rows * best.cols) {
            best = {rows, cols};
        }
    }
    return best;
}

struct Halos {
    int offsets[sides + 1] = {};
    int count() const { return offsets[sides]; }
};

struct Interior {
    int row, col, rows, cols;
    int count() const { return rows * cols; }
};

struct Mesh {
    int rows, cols, cells;
    Halos halos;
    Interior interior;
};

[[maybe_unused]] int hostThreads(int work, int grain) {
    return std::min({omp_get_max_threads(), omp_get_num_procs(), 1 + (work - 1) / grain});
}

void buildMesh(const Tile& tile, int grid, const Mesh& mesh,
               int* neighbors, signed char* sources) {
    const auto plane = static_cast<size_t>(mesh.cells);
    #pragma omp parallel for schedule(static) num_threads(hostThreads(mesh.cells, 65536))
    for (int row = 0; row < mesh.rows; ++row) {
        const int global_row = tile.row + row;
        const int row_start = row * mesh.cols;
        // Direction-major storage coalesces loads across GPU threads.
        // Preserve the reference order: +row, -row, +column, -column.
        for (int col = 0; col < mesh.cols; ++col) {
            const int cell = row_start + col;
            const int global_col = tile.col + col;
            neighbors[cell] = global_row + 1 == grid ? -1 :
                (row + 1 < mesh.rows ? cell + mesh.cols : mesh.cells + mesh.halos.offsets[bottom] + col);
            neighbors[plane + cell] = global_row == 0 ? -1 :
                (row > 0 ? cell - mesh.cols : mesh.cells + mesh.halos.offsets[top] + col);
            neighbors[2 * plane + cell] = global_col + 1 == grid ? -1 :
                (col + 1 < mesh.cols ? cell + 1 : mesh.cells + mesh.halos.offsets[right] + row);
            neighbors[3 * plane + cell] = global_col == 0 ? -1 :
                (col > 0 ? cell - 1 : mesh.cells + mesh.halos.offsets[left] + row);
            sources[cell] = 0;
            if ((global_row == 0 || global_row == grid - 1) &&
                (global_col == 0 || global_col == grid - 1)) {
                sources[cell] = (global_row == global_col) ? 1 : -1;
            }
        }
    }
}

enum class Region { all, interior, boundary };

template <Region region>
__global__ void advance(Mesh mesh, int work, const int* __restrict__ neighbors,
                        const signed char* __restrict__ sources,
                        const double* __restrict__ current, double* __restrict__ next,
                        double* __restrict__ accumulated_flux) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= work) {
        return;
    }

    int cell = item;
    if constexpr (region == Region::interior) {
        const auto inner = mesh.interior;
        cell = (inner.row + item / inner.cols) * mesh.cols + inner.col + item % inner.cols;
    }
    if constexpr (region == Region::boundary) {
        const auto inner = mesh.interior;
        const int first = inner.row * mesh.cols;
        const int strip_width = mesh.cols - inner.cols;
        const int middle = inner.rows * strip_width;
        if (item >= first + middle) {
            cell = item + inner.rows * inner.cols;
        } else if (item >= first) {
            const int offset = item - first;
            const int col = offset % strip_width;
            cell = first + (offset / strip_width) * mesh.cols + col +
                   (col >= inner.col ? inner.cols : 0);
        }
    }

    const double energy = current[cell];
    double flux = 0.5 * sources[cell];
    #pragma unroll
    for (int direction = 0; direction < sides; ++direction) {
        const int neighbor = neighbors[static_cast<size_t>(direction) * mesh.cells + cell];
        if (neighbor >= 0) {
            flux += ((current[neighbor] - energy) * 0.8) * 0.25;
        }
    }
    next[cell] = energy + flux;
    // Each thread owns its cell's flux. Only energy is needed by neighbors.
    accumulated_flux[cell] += fabs(flux);
}

__global__ void packHalos(Mesh mesh, int count, const double* energy, double* packed) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) {
        return;
    }

    const auto& offsets = mesh.halos.offsets;
    int cell;
    if (item < offsets[bottom]) {
        cell = item;
    } else if (item < offsets[left]) {
        cell = (mesh.rows - 1) * mesh.cols + (item - offsets[bottom]);
    } else if (item < offsets[right]) {
        cell = (item - offsets[left]) * mesh.cols;
    } else {
        cell = (item - offsets[right]) * mesh.cols + mesh.cols - 1;
    }
    packed[item] = energy[cell];
}

int blocksFor(int count) {
    return 1 + (count - 1) / block_size;
}

bool cudaAwareMpi() {
#if defined(OMPI_MAJOR_VERSION) && defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

double runSimulation(const Mesh& mesh, const int* neighbors, const signed char* sources,
                     double*& current, double*& next, double* flux, int iterations,
                     const int (&peer)[sides], MPI_Comm comm) {
    const int halo_count = mesh.halos.count();
    const bool exchange = halo_count && iterations > 1;
    const bool direct = exchange && cudaAwareMpi();
    const size_t halo_bytes = static_cast<size_t>(halo_count) * sizeof(double);
    DeviceBuffer<double> packed(exchange ? halo_count : 0);
    double* host_halos = nullptr;
    cudaStream_t compute, transfer = nullptr;
    cudaEvent_t energy_ready = nullptr, halo_ready = nullptr;

    checkCuda(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    if (exchange) {
        checkCuda(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        checkCuda(cudaEventCreateWithFlags(&energy_ready, cudaEventDisableTiming));
        if (!direct) {
            checkCuda(cudaEventCreateWithFlags(&halo_ready, cudaEventDisableTiming));
            checkCuda(cudaMallocHost(&host_halos, 3 * halo_bytes));
        }
    }

    // Resolve any CUDA lazy module loading before timing the simulation.
    if (iterations > 0) {
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, advance<Region::all>));
        if (exchange) {
            checkCuda(cudaFuncGetAttributes(&attributes, packHalos));
            checkCuda(cudaFuncGetAttributes(&attributes, advance<Region::interior>));
            checkCuda(cudaFuncGetAttributes(&attributes, advance<Region::boundary>));
        }
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        // All ghost cells are initially zero, just like the owned cells.
        if (!halo_count || iter == 0) {
            advance<Region::all><<<blocksFor(mesh.cells), block_size, 0, compute>>>(
                mesh, mesh.cells, neighbors, sources, current, next, flux);
        } else {
            const int* offsets = mesh.halos.offsets;
            // Alternate receive buffers so MPI can post receives during the
            // previous iteration's host-to-device copy.
            double* receive = direct ? current + mesh.cells :
                host_halos + static_cast<size_t>(1 + (iter & 1)) * halo_count;
            const double* send = direct ? packed.data : host_halos;
            MPI_Request requests[2 * sides];
            int request_count = 0;
            for (int side = 0; side < sides; ++side) {
                if (peer[side] != MPI_PROC_NULL) {
                    MPI_Irecv(receive + offsets[side], offsets[side + 1] - offsets[side],
                              MPI_DOUBLE, peer[side], opposite[side], comm,
                              &requests[request_count++]);
                }
            }

            checkCuda(cudaStreamWaitEvent(transfer, energy_ready));
            packHalos<<<blocksFor(halo_count), block_size, 0, transfer>>>(
                mesh, halo_count, current, packed.data);
            if (!direct) {
                checkCuda(cudaMemcpyAsync(host_halos, packed.data, halo_bytes,
                                          cudaMemcpyDeviceToHost, transfer));
            }

            const int interior_count = mesh.interior.count();
            if (interior_count) {
                advance<Region::interior><<<blocksFor(interior_count), block_size, 0, compute>>>(
                    mesh, interior_count, neighbors, sources, current, next, flux);
            }

            checkCuda(cudaStreamSynchronize(transfer));
            for (int side = 0; side < sides; ++side) {
                if (peer[side] != MPI_PROC_NULL) {
                    MPI_Isend(send + offsets[side], offsets[side + 1] - offsets[side],
                              MPI_DOUBLE, peer[side], side, comm, &requests[request_count++]);
                }
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

            if (!direct) {
                checkCuda(cudaMemcpyAsync(current + mesh.cells, receive, halo_bytes,
                                          cudaMemcpyHostToDevice, transfer));
                checkCuda(cudaEventRecord(halo_ready, transfer));
                checkCuda(cudaStreamWaitEvent(compute, halo_ready));
            }
            const int boundary_count = mesh.cells - interior_count;
            advance<Region::boundary><<<blocksFor(boundary_count), block_size, 0, compute>>>(
                mesh, boundary_count, neighbors, sources, current, next, flux);
        }
        if (exchange && iter + 1 < iterations) {
            checkCuda(cudaEventRecord(energy_ready, compute));
        }
        std::swap(current, next);
    }

    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (exchange) {
        if (!direct) {
            checkCuda(cudaFreeHost(host_halos));
            checkCuda(cudaEventDestroy(halo_ready));
        }
        checkCuda(cudaEventDestroy(energy_ready));
        checkCuda(cudaStreamDestroy(transfer));
    }
    checkCuda(cudaStreamDestroy(compute));

    return duration;
}

struct Summary {
    uint64_t hash = 0;
    double energy = 0.0;
    double flux = 0.0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
};

template <bool validate>
struct Combine {
    __device__ Summary operator()(Summary a, const Summary& b) const {
        a.hash ^= b.hash;
        if constexpr (validate) {
            a.energy += b.energy;
            a.flux += b.flux;
            a.minimum = fmin(a.minimum, b.minimum);
            a.maximum = fmax(a.maximum, b.maximum);
        }
        return a;
    }
};

template <bool validate>
__global__ void summarize(const double* energy, const double* flux, int cells,
                          Tile tile, int grid, Summary* partial) {
    Summary result;
    for (size_t cell = blockIdx.x * blockDim.x + threadIdx.x;
         cell < static_cast<size_t>(cells); cell += gridDim.x * blockDim.x) {
        const auto local = static_cast<int>(cell);
        const uint64_t global_index =
            static_cast<uint64_t>(tile.row + local / tile.cols) * grid +
            tile.col + local % tile.cols;
        const double e = energy[cell];
        const double f = flux[cell];
        result.hash ^= (static_cast<uint64_t>(__double_as_longlong(e)) + global_index) *
                       0x9e3779b97f4a7c15ULL;
        result.hash ^= (static_cast<uint64_t>(__double_as_longlong(f)) + global_index) *
                       0xbf58476d1ce4e5b9ULL;
        if constexpr (validate) {
            result.energy += e;
            result.flux += f;
            result.minimum = fmin(result.minimum, e);
            result.maximum = fmax(result.maximum, e);
        }
    }

    using Reduction = cub::BlockReduce<Summary, block_size>;
    __shared__ typename Reduction::TempStorage shared;
    const Summary total = Reduction(shared).Reduce(result, Combine<validate>{});
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = total;
    }
}

Summary collectSummary(const double* energy, const double* flux, int cells,
                       const Tile& tile, int grid, bool validate, MPI_Comm comm) {
    const int blocks = std::min(4096, blocksFor(cells));
    DeviceBuffer<Summary> partial(blocks);
    if (validate) {
        summarize<true><<<blocks, block_size>>>(energy, flux, cells, tile, grid, partial.data);
    } else {
        summarize<false><<<blocks, block_size>>>(energy, flux, cells, tile, grid, partial.data);
    }
    checkCuda(cudaGetLastError());

    std::vector<Summary> host(blocks);
    checkCuda(cudaMemcpy(host.data(), partial.data, blocks * sizeof(Summary), cudaMemcpyDeviceToHost));

    uint64_t hash = 0;
    double energy_sum = 0.0, flux_sum = 0.0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    #pragma omp parallel for schedule(static) num_threads(hostThreads(blocks, 1024)) \
        reduction(^:hash) reduction(+:energy_sum,flux_sum) reduction(min:minimum) reduction(max:maximum)
    for (int i = 0; i < blocks; ++i) {
        hash ^= host[i].hash;
        if (validate) {
            energy_sum += host[i].energy;
            flux_sum += host[i].flux;
            minimum = std::min(minimum, host[i].minimum);
            maximum = std::max(maximum, host[i].maximum);
        }
    }

    Summary global;
    MPI_Reduce(&hash, &global.hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (validate) {
        const double sums[] = {energy_sum, flux_sum};
        double global_sums[2] = {};
        MPI_Reduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
        global.energy = global_sums[0];
        global.flux = global_sums[1];
        MPI_Reduce(&minimum, &global.minimum, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maximum, &global.maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    }
    return global;
}

bool validateResults(const Summary& result) {
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", result.energy);
    printf("  Flux sum: %.2f\n", result.flux);
    printf("  Energy range: [%.6f, %.6f]\n", result.minimum, result.maximum);

    if (!std::isfinite(result.energy)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(result.energy) > 1e-8) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(result.flux)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(result.minimum) || !std::isfinite(result.maximum)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

void printGlobalResults(const double* energy, const Tile& local_tile, int cells,
                        const Partition& partition, int grid, int rank, MPI_Comm comm) {
    constexpr int results_tag = sides;
    if (rank != 0) {
        auto local = std::unique_ptr<double[]>(new double[cells]);
        checkCuda(cudaMemcpy(local.get(), energy, static_cast<size_t>(cells) * sizeof(double),
                             cudaMemcpyDeviceToHost));
        MPI_Send(local.get(), cells, MPI_DOUBLE, 0, results_tag, comm);
        return;
    }
    const int ranks = partition.rows * partition.cols;
    std::vector<double> global(static_cast<size_t>(grid) * grid);
    std::vector<MPI_Request> requests(ranks - 1);
    for (int sender = 1; sender < ranks; ++sender) {
        const Tile tile = partition.tile(sender, grid);
        MPI_Datatype rectangle;
        MPI_Type_vector(tile.rows, tile.cols, grid, MPI_DOUBLE, &rectangle);
        MPI_Type_commit(&rectangle);
        MPI_Irecv(global.data() + static_cast<size_t>(tile.row) * grid + tile.col,
                  1, rectangle, sender, results_tag, comm, &requests[sender - 1]);
        MPI_Type_free(&rectangle);
    }

    const size_t row_bytes = static_cast<size_t>(local_tile.cols) * sizeof(double);
    checkCuda(cudaMemcpy2D(global.data(), static_cast<size_t>(grid) * sizeof(double),
                           energy, row_bytes, row_bytes, local_tile.rows, cudaMemcpyDeviceToHost));
    MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE);
    print_results(global, "ElementEnergy");
}

void bindGpu(MPI_Comm comm) {
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int node_rank;
    MPI_Comm_rank(node, &node_rank);
    MPI_Comm_free(&node);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        abortRun("No CUDA device is visible");
    }
    checkCuda(cudaSetDevice(node_rank % devices));
}

bool benchmark(int grid, int iterations, bool validate, bool printResults,
               const Partition& partition, int rank, MPI_Comm comm) {
    bindGpu(comm);
    const uint64_t elements = static_cast<uint64_t>(grid) * grid;
    const Tile tile = partition.tile(rank, grid);
    const int peer[sides] = {
        tile.row > 0 ? rank - partition.cols : MPI_PROC_NULL,
        tile.row + tile.rows < grid ? rank + partition.cols : MPI_PROC_NULL,
        tile.col > 0 ? rank - 1 : MPI_PROC_NULL,
        tile.col + tile.cols < grid ? rank + 1 : MPI_PROC_NULL
    };
    const int lengths[sides] = {tile.cols, tile.cols, tile.rows, tile.rows};
    const size_t cells = static_cast<size_t>(tile.rows) * tile.cols;
    size_t state_cells = cells;
    Mesh mesh{tile.rows, tile.cols, 0, {}, {}};
    for (int side = 0; side < sides; ++side) {
        if (peer[side] != MPI_PROC_NULL) {
            state_cells += lengths[side];
        }
        if (state_cells > static_cast<size_t>(std::numeric_limits<int>::max())) {
            abortRun("Local mesh index limit exceeded. Use more MPI ranks.");
        }
        mesh.halos.offsets[side + 1] = static_cast<int>(state_cells - cells);
    }
    mesh.cells = static_cast<int>(cells);
    const int first_row = peer[top] != MPI_PROC_NULL;
    const int first_col = peer[left] != MPI_PROC_NULL;
    mesh.interior = {first_row, first_col,
                     std::max(0, tile.rows - first_row - (peer[bottom] != MPI_PROC_NULL)),
                     std::max(0, tile.cols - first_col - (peer[right] != MPI_PROC_NULL))};

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", grid, grid, elements);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    DeviceBuffer<int> neighbors(sides * cells);
    DeviceBuffer<signed char> sources(cells);
    {
        auto host_neighbors = std::unique_ptr<int[]>(new int[sides * cells]);
        auto host_sources = std::unique_ptr<signed char[]>(new signed char[cells]);
        buildMesh(tile, grid, mesh, host_neighbors.get(), host_sources.get());
        checkCuda(cudaMemcpy(neighbors.data, host_neighbors.get(), sides * cells * sizeof(int),
                             cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(sources.data, host_sources.get(), cells * sizeof(signed char),
                             cudaMemcpyHostToDevice));
    }

    DeviceBuffer<double> energy_a(state_cells), energy_b(state_cells), flux(cells);
    checkCuda(cudaMemset(energy_a.data, 0, state_cells * sizeof(double)));
    checkCuda(cudaMemset(flux.data, 0, cells * sizeof(double)));
    // Nonblocking compute streams are independent of initialization on the default stream.
    checkCuda(cudaStreamSynchronize(nullptr));

    if (rank == 0) {
        const double static_mb = elements * (sides * sizeof(int) + sizeof(signed char)) / (1024.0 * 1024.0);
        const int packed_buffers = iterations > 1;
        const uint64_t halo_cells = 2 * static_cast<uint64_t>(grid) * ((partition.rows - 1) + (partition.cols - 1));
        const double dynamic_mb = sizeof(double) *
            (3 * elements + (2 + packed_buffers) * halo_cells) / (1024.0 * 1024.0);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               static_mb + dynamic_mb, static_mb, dynamic_mb);
        printf("Running simulation...\n");
    }

    double* current = energy_a.data;
    double* next = energy_b.data;
    const double duration = runSimulation(mesh, neighbors.data, sources.data, current, next,
                                          flux.data, iterations, peer, comm);
    if (rank == 0) {
        const int measured_iters = iterations > 1 ? iterations - 1 : 1;
        const double giga_elems_per_sec = measured_iters * static_cast<double>(elements) / duration / 1e9;
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration * 1000.0 / measured_iters);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
    }

    const Summary summary = collectSummary(current, flux.data, mesh.cells, tile, grid, validate, comm);
    if (rank == 0) {
        printf("  Result hash: %016" PRIX64 "\n\n", summary.hash);
    }
    if (printResults) {
        printGlobalResults(current, tile, mesh.cells, partition, grid, rank, comm);
    }
    return rank != 0 || !validate || validateResults(summary);
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        abortRun("MPI does not provide MPI_THREAD_FUNNELED");
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int grid = 512, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            grid = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) {
                    printf("Unknown option: %s\n", argv[i]);
                }
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    if (grid <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Grid size must be positive\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const Partition partition = partitionGrid(grid, size);
    MPI_Comm active = MPI_COMM_NULL;
    if (partition.rows * partition.cols == size) {
        active = MPI_COMM_WORLD;
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, rank < partition.rows * partition.cols ? 0 : MPI_UNDEFINED,
                       rank, &active);
    }
    int status = EXIT_SUCCESS;
    if (active != MPI_COMM_NULL) {
        try {
            if (!benchmark(grid, iterations, validate, printResults, partition, rank, active)) {
                status = EXIT_FAILURE;
            }
        } catch (const std::exception& error) {
            abortRun("Benchmark failed: ", error.what());
        }
        if (active != MPI_COMM_WORLD) {
            MPI_Comm_free(&active);
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
