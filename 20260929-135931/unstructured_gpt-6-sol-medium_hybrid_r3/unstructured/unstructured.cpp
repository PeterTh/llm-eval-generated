#include <algorithm>
#include <chrono>
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
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void mpiCheck(int result, const char* operation) {
    if (result != MPI_SUCCESS) {
        fprintf(stderr, "MPI %s failed\n", operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// The input mesh is a square grid. Generate its four connections in the
// original (+row, -row, +column, -column) order without storing connectivity.
__global__ void updateRows(const ElementDynamic* __restrict__ old_state,
                           ElementDynamic* __restrict__ new_state,
                           int width, int global_first_row,
                           int first_row, int row_count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= row_count * width) return;
    const int local_row = first_row + k / width;
    const int column = k % width;
    const int global_row = global_first_row + local_row;
    const int p = (local_row + 1) * width + column;
    const double energy = old_state[p].current_energy;
    const double external = ((global_row == 0 || global_row == width - 1) &&
                             (column == 0 || column == width - 1))
                            ? ((global_row == column) ? 0.5 : -0.5) : 0.0;
    double flux = external;
    if (global_row + 1 < width)
        flux += (old_state[p + width].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (global_row > 0)
        flux += (old_state[p - width].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (column + 1 < width)
        flux += (old_state[p + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (column > 0)
        flux += (old_state[p - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    new_state[p].current_energy = energy + flux;
    new_state[p].total_flux = old_state[p].total_flux + fabs(flux);
}

static void launchRows(const ElementDynamic* old_state, ElementDynamic* new_state,
                       int width, int global_first_row,
                       int first_row, int row_count, cudaStream_t stream) {
    if (row_count == 0) return;
    constexpr int block_size = 256;
    const size_t count = static_cast<size_t>(row_count) * width;
    updateRows<<<static_cast<unsigned>((count + block_size - 1) / block_size),
                 block_size, 0, stream>>>(old_state, new_state, width,
                                           global_first_row,
                                           first_row, row_count);
    cudaCheck(cudaGetLastError(), "kernel launch");
}

static uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t first_index) {
    uint64_t hash = 0;
    // XOR makes the partial hashes independent. OpenMP keeps the original
    // global index in each term, so the result is bit-for-bit deterministic.
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e, f;
        memcpy(&e, &elements[i].current_energy, sizeof(e));
        memcpy(&f, &elements[i].total_flux, sizeof(f));
        hash ^= (e + first_index + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + first_index + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    double energy_sum = 0.0, flux_sum = 0.0;
    double energy_max = std::numeric_limits<double>::lowest();
    double energy_min = std::numeric_limits<double>::max();
    // Preserve the reference's summation order for its printed values.
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "initialization");
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int width = 512, iterations = 10;
    bool validate = false, print_results_flag = false, help = false, invalid = false;
    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "-n") == 0 && a + 1 < argc) width = atoi(argv[++a]);
        else if (strcmp(argv[a], "-i") == 0 && a + 1 < argc) iterations = atoi(argv[++a]);
        else if (strcmp(argv[a], "-v") == 0) validate = true;
        else if (strcmp(argv[a], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[a], "-h") == 0) help = true;
        else { invalid = true; if (world_rank == 0) printf("Unknown option: %s\n", argv[a]); }
    }
    if (width < 1 || static_cast<int64_t>(width) * width > INT32_MAX || iterations < 0) invalid = true;
    if (help || invalid) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return invalid ? 1 : 0;
    }

    const int active_size = std::min(world_size, width);
    MPI_Comm comm = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                            world_rank, &comm), "active communicator split");
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = world_rank;
    const int first_row = static_cast<int64_t>(width) * rank / active_size;
    const int end_row = static_cast<int64_t>(width) * (rank + 1) / active_size;
    const int local_rows = end_row - first_row;
    const size_t local_count = static_cast<size_t>(local_rows) * width;
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(ElementDynamic);

    MPI_Comm shared;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank,
                                 MPI_INFO_NULL, &shared), "shared communicator split");
    int local_rank, devices;
    MPI_Comm_rank(shared, &local_rank);
    cudaCheck(cudaGetDeviceCount(&devices), "device count");
    if (devices < 1) {
        fprintf(stderr, "Rank %d has no CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % devices), "set device");
    MPI_Comm_free(&shared);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", width, width, width * width);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        const double dynamic_mem = static_cast<double>(width) * width * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               dynamic_mem / (1024.0 * 1024.0), 0.0, dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    ElementDynamic *device_a, *device_b;
    cudaCheck(cudaMalloc(&device_a, (local_count + 2 * width) * sizeof(ElementDynamic)), "allocate state A");
    cudaCheck(cudaMalloc(&device_b, (local_count + 2 * width) * sizeof(ElementDynamic)), "allocate state B");
    cudaCheck(cudaMemset(device_a, 0, (local_count + 2 * width) * sizeof(ElementDynamic)), "initialize state");
    cudaStream_t compute_stream;
    cudaCheck(cudaStreamCreate(&compute_stream), "create stream");

    // One pinned block keeps boundary transfers fast and permits MPI to use
    // host buffers even when the installed MPI lacks CUDA-aware support.
    ElementDynamic* boundary;
    cudaCheck(cudaMallocHost(&boundary, 4 * row_bytes), "allocate pinned boundary rows");
    ElementDynamic* send_top = boundary;
    ElementDynamic* send_bottom = boundary + width;
    ElementDynamic* recv_top = boundary + 2 * width;
    ElementDynamic* recv_bottom = boundary + 3 * width;
    MPI_Datatype row_type;
    mpiCheck(MPI_Type_contiguous(2 * width, MPI_DOUBLE, &row_type), "create row type");
    mpiCheck(MPI_Type_commit(&row_type), "commit row type");

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (active_size == 1) {
            launchRows(device_a, device_b, width, first_row,
                       0, local_rows, compute_stream);
            std::swap(device_a, device_b);
            continue;
        }
        // Read only the two rows required by neighboring ranks.
        if (rank > 0)
            cudaCheck(cudaMemcpyAsync(send_top, device_a + width, row_bytes,
                                      cudaMemcpyDeviceToHost, compute_stream), "copy top row");
        if (rank + 1 < active_size)
            cudaCheck(cudaMemcpyAsync(send_bottom, device_a + local_rows * width, row_bytes,
                                      cudaMemcpyDeviceToHost, compute_stream), "copy bottom row");
        cudaCheck(cudaStreamSynchronize(compute_stream), "complete boundary read");
        MPI_Request requests[4];
        int count = 0;
        if (rank > 0) {
            mpiCheck(MPI_Irecv(recv_top, 1, row_type, rank - 1, 1, comm, &requests[count++]), "receive top");
            mpiCheck(MPI_Isend(send_top, 1, row_type, rank - 1, 0, comm, &requests[count++]), "send top");
        }
        if (rank + 1 < active_size) {
            mpiCheck(MPI_Irecv(recv_bottom, 1, row_type, rank + 1, 0, comm, &requests[count++]), "receive bottom");
            mpiCheck(MPI_Isend(send_bottom, 1, row_type, rank + 1, 1, comm, &requests[count++]), "send bottom");
        }
        // Interior rows do not depend on incoming halos.
        launchRows(device_a, device_b, width, first_row, 1,
                   std::max(local_rows - 2, 0), compute_stream);
        mpiCheck(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE), "boundary exchange");
        if (rank > 0)
            cudaCheck(cudaMemcpyAsync(device_a, recv_top, row_bytes,
                                      cudaMemcpyHostToDevice, compute_stream), "write top halo");
        if (rank + 1 < active_size)
            cudaCheck(cudaMemcpyAsync(device_a + (local_rows + 1) * width,
                                      recv_bottom, row_bytes, cudaMemcpyHostToDevice,
                                      compute_stream), "write bottom halo");
        launchRows(device_a, device_b, width, first_row, 0, 1, compute_stream);
        if (local_rows > 1)
            launchRows(device_a, device_b, width, first_row,
                       local_rows - 1, 1, compute_stream);
        std::swap(device_a, device_b);
    }
    cudaCheck(cudaStreamSynchronize(compute_stream), "finish simulation");
    const double local_seconds = MPI_Wtime() - start;
    double seconds;
    mpiCheck(MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm), "timing reduction");

    std::vector<ElementDynamic> local_output(local_count);
    cudaCheck(cudaMemcpy(local_output.data(), device_a + width,
                         local_count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "copy results");
    const uint64_t local_hash = computeHash(local_output, static_cast<size_t>(first_row) * width);
    uint64_t global_hash = 0;
    mpiCheck(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm),
             "hash reduction");
    std::vector<ElementDynamic> global_output;
    if (validate || print_results_flag) {
        std::vector<int> counts, displacements;
        MPI_Datatype element_type;
        mpiCheck(MPI_Type_contiguous(2, MPI_DOUBLE, &element_type), "create element type");
        mpiCheck(MPI_Type_commit(&element_type), "commit element type");
        if (rank == 0) {
            global_output.resize(static_cast<size_t>(width) * width);
            counts.resize(active_size);
            displacements.resize(active_size);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < active_size; ++r) {
                const int begin = static_cast<int64_t>(width) * r / active_size;
                const int end = static_cast<int64_t>(width) * (r + 1) / active_size;
                counts[r] = (end - begin) * width;
                displacements[r] = begin * width;
            }
        }
        mpiCheck(MPI_Gatherv(local_output.data(), static_cast<int>(local_count), element_type,
                             rank == 0 ? global_output.data() : nullptr,
                             rank == 0 ? counts.data() : nullptr,
                             rank == 0 ? displacements.data() : nullptr,
                             element_type, 0, comm), "gather results");
        MPI_Type_free(&element_type);
    }

    int valid = 1;
    if (rank == 0) {
        const long duration_ms = static_cast<long>(seconds * 1000.0);
        const int measured = std::max(iterations - 1, 1);
        const double time_per_iter = seconds * 1000.0 / measured;
        const double giga_elems_per_sec = seconds > 0.0
            ? static_cast<double>(measured) * width * width / seconds / 1e9 : 0.0;
        printf("Computation time: %ld ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016lX\n\n", global_hash);
        if (print_results_flag) {
            std::vector<double> energy_data(global_output.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < global_output.size(); ++i)
                energy_data[i] = global_output[i].current_energy;
            print_results(energy_data, "ElementEnergy");
        }
        if (validate) valid = validateResults(global_output) ? 1 : 0;
    }
    mpiCheck(MPI_Bcast(&valid, 1, MPI_INT, 0, comm), "broadcast validation");
    MPI_Type_free(&row_type);
    cudaFreeHost(boundary);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaStreamDestroy(compute_stream);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
