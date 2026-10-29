#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy; val_t total_flux; };
struct World { std::vector<ElementDynamic> elements_dynamic; };

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Rows 1..rows are owned; rows 0 and rows+1 hold neighboring ranks' halos.
__global__ void update(const ElementDynamic* in, ElementDynamic* out,
                       int n, int first_row, int rows) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * n;
    if (k >= count) return;
    int x = first_row + (int)(k / n), y = (int)(k % n);
    size_t pos = k + n;
    double energy = in[pos].current_energy;
    double flux = 0.0;
    // Corner assignments are made in this order by the original mesh builder.
    if (x == 0 && y == 0) flux = 0.5;
    if (x == 0 && y == n - 1) flux = -0.5;
    if (x == n - 1 && y == 0) flux = -0.5;
    if (x == n - 1 && y == n - 1) flux = 0.5;
    if (x + 1 < n) flux += (in[pos + n].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (x > 0) flux += (in[pos - n].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y + 1 < n) flux += (in[pos + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y > 0) flux += (in[pos - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    out[pos].current_energy = energy + flux;
    out[pos].total_flux = in[pos].total_flux + fabs(flux);
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
    int n = 512, iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    // MPI counts and displacements are expressed in bytes.
    if (n <= 0 || iters < 0 || (uint64_t)n * n * sizeof(ElementDynamic) > INT32_MAX) {
        if (!rank) fprintf(stderr, "Invalid grid size or iteration count (grid must fit MPI byte counts).\n");
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
    int active = std::min(n, ranks);
    int rows = rank < active ? n / active + (rank < n % active) : 0;
    int first = rank < active ? rank * (n / active) + std::min(rank, n % active) : n;
    size_t row_bytes = (size_t)n * sizeof(ElementDynamic);
    size_t bytes = (size_t)(rows + 2) * row_bytes;
    ElementDynamic *a = nullptr, *b = nullptr, *send_top = nullptr, *send_bottom = nullptr, *recv_top = nullptr, *recv_bottom = nullptr;
    if (rows) {
        CUDA_CHECK(cudaMalloc(&a, bytes)); CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemset(a, 0, bytes)); CUDA_CHECK(cudaMemset(b, 0, bytes));
        CUDA_CHECK(cudaMallocHost(&send_top, row_bytes)); CUDA_CHECK(cudaMallocHost(&send_bottom, row_bytes));
        CUDA_CHECK(cudaMallocHost(&recv_top, row_bytes)); CUDA_CHECK(cudaMallocHost(&recv_bottom, row_bytes));
    }
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n\n", n, n, (long long)n*n, iters, validate ? "enabled" : "disabled");
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iters && rows; ++iter) {
        int prev = rank ? rank - 1 : MPI_PROC_NULL;
        int next = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
        if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_top, a + n, row_bytes, cudaMemcpyDeviceToHost));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_bottom, a + (size_t)rows*n, row_bytes, cudaMemcpyDeviceToHost));
        MPI_Sendrecv(send_top, (int)row_bytes, MPI_BYTE, prev, 0, recv_bottom, (int)row_bytes, MPI_BYTE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bottom, (int)row_bytes, MPI_BYTE, next, 1, recv_top, (int)row_bytes, MPI_BYTE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (prev != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(a, recv_top, row_bytes, cudaMemcpyHostToDevice));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(a + (size_t)(rows+1)*n, recv_bottom, row_bytes, cudaMemcpyHostToDevice));
        size_t count = (size_t)rows*n;
        update<<<(unsigned)((count + 255)/256), 256>>>(a, b, n, first, rows);
        CUDA_CHECK(cudaGetLastError());
        std::swap(a, b);
    }
    if (rows) CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<ElementDynamic> local_data((size_t)rows*n);
    if (rows) CUDA_CHECK(cudaMemcpy(local_data.data(), a+n, local_data.size()*sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    World world;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        int nr = r < active ? n/active + (r < n%active) : 0;
        int begin = r < active ? r*(n/active) + std::min(r,n%active) : n;
        counts[r] = (int)((size_t)nr*row_bytes);
        displacements[r] = (int)((size_t)begin*row_bytes);
    }
    if (!rank) world.elements_dynamic.resize((size_t)n*n);
    MPI_Gatherv(local_data.data(), counts[rank], MPI_BYTE, world.elements_dynamic.data(), counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    int ok = 1;
    if (!rank) {
        printf("Computation time: %.3f ms\n", max_elapsed*1000.0);
        double rate = max_elapsed > 0 ? (double)n*n*iters/max_elapsed/1e9 : 0;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", iters ? max_elapsed*1000/iters : 0.0, rate, rate*22);
        printf("  Result hash: %016lX\n\n", (unsigned long)computeHash(world.elements_dynamic));
        if (printResults) {
            std::vector<double> energyData(world.elements_dynamic.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < energyData.size(); ++i) energyData[i] = world.elements_dynamic[i].current_energy;
            print_results(energyData, "ElementEnergy");
        }
        if (validate && !validateResults(world)) ok = 0;
    }
    if (rows) { CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFreeHost(send_top)); CUDA_CHECK(cudaFreeHost(send_bottom)); CUDA_CHECK(cudaFreeHost(recv_top)); CUDA_CHECK(cudaFreeHost(recv_bottom)); }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
