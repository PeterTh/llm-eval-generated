#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using val_t = double;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// buildSquare2D always produces this connectivity. Encoding it implicitly avoids
// streaming 144 bytes of redundant indices/coefficients for every cell update.
// Energy is double-buffered, but cumulative flux can be updated in place: no
// other cell reads it. Only energies require MPI halos.
__global__ void update(const val_t* __restrict__ energy,
                       val_t* __restrict__ next, val_t* __restrict__ flux,
                       int n, int row_offset, size_t first, size_t count) {
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += size_t(blockDim.x) * gridDim.x) {
        const size_t i = first + k;
        const int y = i % n;
        const int x = row_offset + i / n;
        const size_t p = i + n; // one halo row before owned rows
        const val_t e = energy[p];
        val_t total = 0.0;
        if (x == 0 || x == n - 1) {
            if (y == 0) total = (x == 0) ? 0.5 : -0.5;
            if (y == n - 1) total = (x == n - 1) ? 0.5 : -0.5;
        }
        // Preserve reference order (+x, -x, +y, -y), including its arithmetic.
        if (x + 1 < n) total += (energy[p + n] - e) * 0.8 * 1.0 * 0.25;
        if (x > 0)     total += (energy[p - n] - e) * 0.8 * 1.0 * 0.25;
        if (y + 1 < n) total += (energy[p + 1] - e) * 0.8 * 1.0 * 0.25;
        if (y > 0)     total += (energy[p - 1] - e) * 0.8 * 1.0 * 0.25;
        next[p] = e + total;
        flux[i] += fabs(total);
    }
}

static void launchUpdate(const val_t* energy, val_t* next, val_t* flux,
                         int n, int offset, size_t first, size_t count,
                         cudaStream_t stream) {
    if (!count) return;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    update<<<blocks, 256, 0, stream>>>(energy, next, flux, n, offset, first, count);
}

struct World {
    int n, offset, rows, rank, ranks;
    MPI_Comm comm;
    val_t *energy = nullptr, *next = nullptr, *flux = nullptr;
    val_t* halo = nullptr;
    cudaStream_t compute, transfer;
    cudaEvent_t received;
    std::vector<val_t> final_energy, final_flux;

    World(int width, MPI_Comm communicator) : n(width), comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        rows = n / ranks + (rank < n % ranks);
        offset = rank * (n / ranks) + std::min(rank, n % ranks);
        const size_t cells = size_t(rows) * n;
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&energy), (cells + 2 * size_t(n)) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next), (cells + 2 * size_t(n)) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&flux), cells * sizeof(val_t)));
        CUDA_CHECK(cudaMemsetAsync(energy, 0, (cells + 2 * size_t(n)) * sizeof(val_t), compute));
        CUDA_CHECK(cudaMemsetAsync(next, 0, (cells + 2 * size_t(n)) * sizeof(val_t), compute));
        CUDA_CHECK(cudaMemsetAsync(flux, 0, cells * sizeof(val_t), compute));
        if (ranks > 1)
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&halo), 4 * size_t(n) * sizeof(val_t)));
        // Host result storage is rank-local and allocated outside the timer.
        final_energy.resize(cells);
        final_flux.resize(cells);
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void simulate(int iterations) {
        const size_t cells = size_t(rows) * n;
        int iter = 0;
        // Amortize host launch overhead for a single-rank GPU run. An even
        // batch returns to the same ping-pong buffers and can be replayed.
        if (ranks == 1 && iterations >= 32) {
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            for (int j = 0; j < 32; ++j) {
                launchUpdate(energy, next, flux, n, offset, 0, cells, compute);
                std::swap(energy, next);
            }
            CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (; iter <= iterations - 32; iter += 32)
                CUDA_CHECK(cudaGraphLaunch(executable, compute));
            CUDA_CHECK(cudaStreamSynchronize(compute));
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        for (; iter < iterations; ++iter) {
            if (ranks == 1) {
                launchUpdate(energy, next, flux, n, offset, 0, cells, compute);
            } else {
                // Pinned staging works with any MPI implementation, including
                // those without CUDA-aware support. Overlap halo exchange with
                // all rows whose stencil is wholly owned by this rank.
                const int above = rank ? rank - 1 : MPI_PROC_NULL;
                const int below = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
                MPI_Request requests[4];
                MPI_Irecv(halo + 2 * size_t(n), n, MPI_DOUBLE, above, 1, comm, &requests[0]);
                MPI_Irecv(halo + 3 * size_t(n), n, MPI_DOUBLE, below, 0, comm, &requests[1]);
                if (rows > 2)
                    launchUpdate(energy, next, flux, n, offset, n, size_t(rows - 2) * n, compute);
                if (above != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halo, energy + n, size_t(n) * sizeof(val_t),
                                               cudaMemcpyDeviceToHost, transfer));
                if (below != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(halo + n, energy + cells, size_t(n) * sizeof(val_t),
                                               cudaMemcpyDeviceToHost, transfer));
                CUDA_CHECK(cudaStreamSynchronize(transfer));
                MPI_Isend(halo, n, MPI_DOUBLE, above, 0, comm, &requests[2]);
                MPI_Isend(halo + n, n, MPI_DOUBLE, below, 1, comm, &requests[3]);
                MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
                if (above != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy, halo + 2 * size_t(n), size_t(n) * sizeof(val_t),
                                               cudaMemcpyHostToDevice, transfer));
                if (below != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(energy + cells + n, halo + 3 * size_t(n), size_t(n) * sizeof(val_t),
                                               cudaMemcpyHostToDevice, transfer));
                CUDA_CHECK(cudaEventRecord(received, transfer));
                CUDA_CHECK(cudaStreamWaitEvent(compute, received, 0));
                launchUpdate(energy, next, flux, n, offset, 0, n, compute);
                if (rows > 1)
                    launchUpdate(energy, next, flux, n, offset, size_t(rows - 1) * n, n, compute);
                // Next iteration may send the rows just written by these kernels.
                CUDA_CHECK(cudaStreamSynchronize(compute));
            }
            std::swap(energy, next);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void download() {
        const size_t bytes = final_energy.size() * sizeof(val_t);
        CUDA_CHECK(cudaMemcpy(final_energy.data(), energy + n, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(final_flux.data(), flux, bytes, cudaMemcpyDeviceToHost));
    }

    ~World() {
        CUDA_CHECK(cudaFree(energy));
        CUDA_CHECK(cudaFree(next));
        CUDA_CHECK(cudaFree(flux));
        if (halo) CUDA_CHECK(cudaFreeHost(halo));
        CUDA_CHECK(cudaEventDestroy(received));
        CUDA_CHECK(cudaStreamDestroy(transfer));
        CUDA_CHECK(cudaStreamDestroy(compute));
    }
};

static uint64_t computeHash(const World& world) {
    uint64_t local = 0, result = 0;
    const size_t base = size_t(world.offset) * world.n;
    #pragma omp parallel for reduction(^:local) schedule(static)
    for (size_t i = 0; i < world.final_energy.size(); ++i) {
        uint64_t e, f;
        memcpy(&e, &world.final_energy[i], sizeof(e));
        memcpy(&f, &world.final_flux[i], sizeof(f));
        local ^= (e + base + i) * 0x9e3779b97f4a7c15ULL;
        local ^= (f + base + i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Reduce(&local, &result, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return result;
}

static bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    int finite = 1;
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(min:energy_min,finite) reduction(max:energy_max) schedule(static)
    for (size_t i = 0; i < world.final_energy.size(); ++i) {
        const val_t e = world.final_energy[i], f = world.final_flux[i];
        energy_sum += e;
        flux_sum += f;
        energy_min = std::min(energy_min, e);
        energy_max = std::max(energy_max, e);
        finite = std::min(finite, int(std::isfinite(e) && std::isfinite(f)));
    }
    val_t sums[2] = {energy_sum, flux_sum}, all_sums[2], lo, hi;
    int all_finite;
    MPI_Allreduce(sums, all_sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    MPI_Allreduce(&energy_min, &lo, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    MPI_Allreduce(&energy_max, &hi, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    MPI_Allreduce(&finite, &all_finite, 1, MPI_INT, MPI_MIN, world.comm);
    const bool valid = all_finite && std::isfinite(all_sums[0]) && std::isfinite(all_sums[1]);
    if (world.rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
               "  Energy range: [%.6f, %.6f]\n", all_sums[0], all_sums[1], lo, hi);
        if (std::abs(all_sums[0]) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

static void printResults(const World& world) {
    // Gather only for the explicitly requested, order-sensitive external report.
    // Chunked point-to-point transfers avoid MPI's int-sized Gatherv counts.
    std::vector<val_t> all;
    if (world.rank == 0) {
        all.resize(size_t(world.n) * world.n);
        std::copy(world.final_energy.begin(), world.final_energy.end(), all.begin());
    }
    const size_t chunk = size_t(1) << 26;
    if (world.rank == 0) {
        for (int r = 1; r < world.ranks; ++r) {
            const size_t rows = world.n / world.ranks + (r < world.n % world.ranks);
            const size_t start = size_t(r * (world.n / world.ranks) + std::min(r, world.n % world.ranks)) * world.n;
            for (size_t i = 0; i < rows * world.n; i += chunk)
                MPI_Recv(all.data() + start + i, int(std::min(chunk, rows * world.n - i)),
                         MPI_DOUBLE, r, 2, world.comm, MPI_STATUS_IGNORE);
        }
        print_results(all, "ElementEnergy");
    } else {
        for (size_t i = 0; i < world.final_energy.size(); i += chunk)
            MPI_Send(world.final_energy.data() + i, int(std::min(chunk, world.final_energy.size() - i)),
                     MPI_DOUBLE, 0, 2, world.comm);
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInteger(const char* text, int& value) {
    char* end;
    const long parsed = strtol(text, &end, 10);
    if (!*text || *end || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10;
    bool validate = false, report = false, help = false, valid_args = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            valid_args = parseInteger(argv[++i], n) && n > 0 && valid_args;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            valid_args = parseInteger(argv[++i], iterations) && valid_args;
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) report = true;
        else if (strcmp(argv[i], "-h") == 0) help = true;
        else valid_args = false;
    }
    if (!valid_args || help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return valid_args ? 0 : 1;
    }

    // At least one owned row per participating rank, even for tiny test grids.
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < n ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    MPI_Comm local;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, local_size, devices;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: CUDA device required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    // Honor explicit OpenMP settings; avoid oversubscribing a node by default.
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, omp_get_num_procs() / local_size));
    MPI_Comm_free(&local);

    int status = 0;
    {
        World world(n, active);
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %zu elements\n", n, n, size_t(n) * n);
            printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("Parallel execution: %d MPI ranks, CUDA, OpenMP\n", world.ranks);
            printf("Device storage per rank: %.2f MB (implicit connectivity)\n\n",
                   (3 * world.final_energy.size() + 4 * size_t(n)) * sizeof(val_t) / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(active);
        const double start = MPI_Wtime();
        world.simulate(iterations);
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);
        world.download();
        const uint64_t hash = computeHash(world);
        if (rank == 0) {
            const double rate = double(iterations) * n * n / std::max(seconds, 1e-12) / 1e9;
            printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000.0);
            printf("  Time per iteration: %.4f ms\n", seconds * 1000.0 / std::max(iterations, 1));
            printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22.0);
            printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));
        }
        if (report) printResults(world);
        if (validate && !validateResults(world)) status = 1;
    }
    MPI_Comm_free(&active);
    MPI_Finalize();
    return status;
}
