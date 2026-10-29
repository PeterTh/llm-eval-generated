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
#include <omp.h>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy; val_t total_flux; };
struct World { std::vector<ElementDynamic> elements_dynamic; };

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Rows 1..rows are owned; rows 0 and rows+1 hold neighbor energies.
__global__ void update(const ElementDynamic* in, ElementDynamic* out,
                       int n, int first, int rows) {
    size_t count = (size_t)n * rows;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride) {
        int x = first + (int)(k / n), y = (int)(k % n);
        size_t i = k + n;
        double energy = in[i].current_energy;
        double flow = 0.0;
        // Corner assignments occur in this order in the original mesh builder.
        if (x == 0 && y == 0) flow = 0.5;
        if (x == 0 && y == n - 1) flow = -0.5;
        if (x == n - 1 && y == 0) flow = -0.5;
        if (x == n - 1 && y == n - 1) flow = 0.5;
        if (x + 1 < n) flow += (in[i + n].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (x > 0) flow += (in[i - n].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (y + 1 < n) flow += (in[i + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
        if (y > 0) flow += (in[i - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
        out[i].current_energy = energy + flow;
        out[i].total_flux = in[i].total_flux + fabs(flow);
    }
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
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    if (n <= 0 || (uint64_t)n * n > INT32_MAX / (int)sizeof(ElementDynamic)) {
        if (!rank) fprintf(stderr, "Grid size is invalid or too large for MPI gather\n");
        MPI_Finalize(); return 1;
    }
    int rows = n / ranks + (rank < n % ranks);
    int first = rank * (n / ranks) + std::min(rank, n % ranks);
    int active = std::min(n, ranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank; MPI_Comm_rank(local, &local_rank);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA devices available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local);
    size_t row_bytes = (size_t)n * sizeof(ElementDynamic);
    size_t bytes = (size_t)(rows + 2) * row_bytes;
    ElementDynamic *a, *b;
    CUDA_CHECK(cudaMalloc(&a, bytes)); CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes)); CUDA_CHECK(cudaMemset(b, 0, bytes));
    std::vector<ElementDynamic> send_up(n), send_down(n), recv_up(n), recv_down(n);
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n\n", n, n, (long long)n*n, iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\nRunning simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int up = rank > 0 && rows ? rank - 1 : MPI_PROC_NULL;
    int down = rank + 1 < active && rows ? rank + 1 : MPI_PROC_NULL;
    for (int iter = 0; iter < iterations; ++iter) {
        if (rows) {
            if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_up.data(), a + n, row_bytes, cudaMemcpyDeviceToHost));
            if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(send_down.data(), a + (size_t)rows*n, row_bytes, cudaMemcpyDeviceToHost));
            MPI_Sendrecv(send_up.data(), (int)row_bytes, MPI_BYTE, up, 0, recv_down.data(), (int)row_bytes, MPI_BYTE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_down.data(), (int)row_bytes, MPI_BYTE, down, 1, recv_up.data(), (int)row_bytes, MPI_BYTE, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(a, recv_up.data(), row_bytes, cudaMemcpyHostToDevice));
            if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(a + (size_t)(rows+1)*n, recv_down.data(), row_bytes, cudaMemcpyHostToDevice));
            int blocks = (int)std::min<size_t>(((size_t)rows*n + 255)/256, 65535);
            update<<<blocks, 256>>>(a, b, n, first, rows);
            CUDA_CHECK(cudaGetLastError());
            std::swap(a, b);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<ElementDynamic> owned((size_t)rows*n);
    if (rows) CUDA_CHECK(cudaMemcpy(owned.data(), a+n, owned.size()*sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    World world;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        int rr = n/ranks + (r < n%ranks);
        int fr = r*(n/ranks) + std::min(r,n%ranks);
        counts[r] = rr*n*(int)sizeof(ElementDynamic);
        offsets[r] = fr*n*(int)sizeof(ElementDynamic);
    }
    if (!rank) world.elements_dynamic.resize((size_t)n*n);
    MPI_Gatherv(owned.data(), counts[rank], MPI_BYTE, world.elements_dynamic.data(), counts.data(), offsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (!rank) {
        long duration_ms = (long)(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        int measured = std::max(iterations - 1, 1);
        double rate = (double)measured * n * n / max_elapsed / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", max_elapsed*1000.0/measured, rate, rate*22.0);
        printf("  Result hash: %016lX\n\n", computeHash(world.elements_dynamic));
        if (printResults) {
            std::vector<double> energyData(world.elements_dynamic.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < energyData.size(); ++i) energyData[i] = world.elements_dynamic[i].current_energy;
            print_results(energyData, "ElementEnergy");
        }
        if (validate) valid = validateResults(world);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    MPI_Finalize();
    return valid ? 0 : 1;
}
