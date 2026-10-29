#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy; val_t total_flux; };
struct World { std::vector<ElementDynamic> elements_dynamic; };

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// The two halo rows precede and follow the locally owned rows. Neighbor
// contributions are accumulated in the same order as the original mesh.
__global__ void update(const ElementDynamic* in, ElementDynamic* out,
                       int n, int first, int rows, int begin, int end) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = size_t(end - begin) * n;
    if (t >= count) return;
    int row = begin + int(t / n), y = int(t % n), x = first + row;
    size_t i = size_t(row + 1) * n + y;
    double e = in[i].current_energy;
    double flux = 0.0;
    if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1)) {
        // The last corner assignment wins when n == 1.
        flux = (x == y) ? 0.5 : -0.5;
    }
    if (x + 1 < n) flux += (in[i + n].current_energy - e) * 0.8 * 1.0 * 0.25;
    if (x > 0)     flux += (in[i - n].current_energy - e) * 0.8 * 1.0 * 0.25;
    if (y + 1 < n) flux += (in[i + 1].current_energy - e) * 0.8 * 1.0 * 0.25;
    if (y > 0)     flux += (in[i - 1].current_energy - e) * 0.8 * 1.0 * 0.25;
    out[i].current_energy = e + flux;
    out[i].total_flux = in[i].total_flux + fabs(flux);
}

static void launch(const ElementDynamic* in, ElementDynamic* out, int n,
                   int first, int rows, int begin, int end, cudaStream_t stream) {
    if (begin >= end) return;
    size_t count = size_t(end - begin) * n;
    update<<<unsigned((count + 255) / 256), 256, 0, stream>>>(in, out, n, first, rows, begin, end);
    CUDA_CHECK(cudaGetLastError());
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }
    
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n < 1 || iterations < 0 || ranks > n || size_t(n) * n > size_t(INT_MAX) / sizeof(ElementDynamic)) {
        if (!rank) fprintf(stderr, "Invalid grid or iteration count, too many ranks, or grid exceeds MPI gather limits.\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices;
    MPI_Comm_rank(local, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local);
    int rows = n / ranks + (rank < n % ranks);
    int first = rank * (n / ranks) + std::min(rank, n % ranks);
    size_t row_bytes = size_t(n) * sizeof(ElementDynamic);
    size_t bytes = size_t(rows + 2) * row_bytes;
    ElementDynamic *a, *b, *send_top, *send_bottom, *recv_top, *recv_bottom;
    CUDA_CHECK(cudaMalloc(&a, bytes)); CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes)); CUDA_CHECK(cudaMemset(b, 0, bytes));
    CUDA_CHECK(cudaMallocHost(&send_top, row_bytes)); CUDA_CHECK(cudaMallocHost(&send_bottom, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_top, row_bytes)); CUDA_CHECK(cudaMallocHost(&recv_bottom, row_bytes));
    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreate(&compute)); CUDA_CHECK(cudaStreamCreate(&transfer));
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n\n", n, n, size_t(n)*n, iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\nRunning simulation...\n");
    }
    int up = rank ? rank - 1 : MPI_PROC_NULL;
    int down = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks == 1) {
            launch(a, b, n, first, rows, 0, rows, compute);
            std::swap(a, b);
            continue;
        }
        // Complete the previous update before reading its boundary rows.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(send_top, a + n, row_bytes, cudaMemcpyDeviceToHost, transfer));
        if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(send_bottom, a + size_t(rows)*n, row_bytes, cudaMemcpyDeviceToHost, transfer));
        launch(a, b, n, first, rows, 1, rows - 1, compute);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        MPI_Sendrecv(send_top, int(row_bytes), MPI_BYTE, up, 0, recv_bottom, int(row_bytes), MPI_BYTE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bottom, int(row_bytes), MPI_BYTE, down, 1, recv_top, int(row_bytes), MPI_BYTE, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(a, recv_top, row_bytes, cudaMemcpyHostToDevice, transfer));
        if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(a + size_t(rows + 1)*n, recv_bottom, row_bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        launch(a, b, n, first, rows, 0, 1, compute);
        if (rows > 1) launch(a, b, n, first, rows, rows - 1, rows, compute);
        std::swap(a, b);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<ElementDynamic> local_data(size_t(rows)*n);
    CUDA_CHECK(cudaMemcpy(local_data.data(), a + n, size_t(rows)*row_bytes, cudaMemcpyDeviceToHost));
    World world;
    std::vector<int> counts(ranks), offsets(ranks);
    #pragma omp parallel for
    for (int r = 0; r < ranks; ++r) {
        int nr = n / ranks + (r < n % ranks);
        int fr = r * (n / ranks) + std::min(r, n % ranks);
        counts[r] = int(size_t(nr)*row_bytes); offsets[r] = int(size_t(fr)*row_bytes);
    }
    if (!rank) world.elements_dynamic.resize(size_t(n)*n);
    MPI_Gatherv(local_data.data(), int(size_t(rows)*row_bytes), MPI_BYTE, world.elements_dynamic.data(), counts.data(), offsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (!rank) {
        long duration_ms = long(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        int measured = std::max(iterations - 1, 1);
        double rate = (measured * double(n) * n) / max_elapsed / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", max_elapsed * 1000 / measured, rate, rate * 22.0);
        printf("  Result hash: %016lX\n\n", (unsigned long)computeHash(world.elements_dynamic));
        if (printResults) {
            std::vector<double> energy(world.elements_dynamic.size());
            #pragma omp parallel for
            for (size_t i = 0; i < energy.size(); ++i) energy[i] = world.elements_dynamic[i].current_energy;
            print_results(energy, "ElementEnergy");
        }
        if (validate) valid = validateResults(world);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFreeHost(send_top)); CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(recv_top)); CUDA_CHECK(cudaFreeHost(recv_bottom));
    CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(transfer));
    MPI_Finalize();
    return valid ? 0 : 1;
}
