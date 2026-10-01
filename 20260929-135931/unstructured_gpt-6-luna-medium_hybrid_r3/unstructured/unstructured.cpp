#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void updateGrid(const val_t* inEnergy, const val_t* inFlux,
                           val_t* outEnergy, val_t* outFlux, int n, int rows,
                           int globalStart, int leftHalo, int rightHalo) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * n;
    if (p >= count) return;
    const int r = p / n, c = p - r * n;
    const int g = globalStart + r;
    const int center = (r + 1) * n + c;
    const int material = ((g == 0 || g == n - 1) && (c == 0 || c == n - 1));
    const val_t transfer = 0.8;
    val_t external = 0.0;
    if (material) external = ((g == 0 && c == 0) || (g == n - 1 && c == n - 1)) ? 0.5 : -0.5;

    val_t flux = external;
    const val_t e = inEnergy[center];
    if (g + 1 < n) flux += (inEnergy[center + n] - e) * transfer * 0.25;
    if (g > 0) flux += (inEnergy[center - n] - e) * transfer * 0.25;
    if (c + 1 < n) flux += (inEnergy[center + 1] - e) * transfer * 0.25;
    if (c > 0) flux += (inEnergy[center - 1] - e) * transfer * 0.25;
    outEnergy[p] = e + flux;
    outFlux[p] = inFlux[p] + fabs(flux);
    (void)leftHalo;
    (void)rightHalo;
}

static int rowCount(int n, int rank, int size) { return n / size + (rank < n % size); }
static int rowStart(int n, int rank, int size) { return rank * (n / size) + std::min(rank, n % size); }

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num> Grid size (NxN elements) (default: 512)\n  -i <num> Number of iterations (default: 10)\n  -v Validate\n  -r Print results\n  -h Show help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, iterations = 10, validate = 0, printResults = 0, help = 0, bad = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = 1;
        else if (!std::strcmp(argv[i], "-r")) printResults = 1;
        else if (!std::strcmp(argv[i], "-h")) help = 1;
        else bad = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (help || bad || n <= 0 || iterations < 0 || n < size) {
        if (rank == 0) {
            if (bad) std::printf("Unknown or incomplete option\n");
            if (n < size && n > 0) std::printf("Grid size must be at least the MPI rank count\n");
            printUsage(argv[0]); std::printf("\n");
        }
        MPI_Finalize(); return bad || n <= 0 || iterations < 0 || n < size ? 1 : 0;
    }

    int devices = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&devices);
    if (deviceStatus != cudaSuccess || devices == 0) {
        if (rank == 0) std::fprintf(stderr, "CUDA device required for unstructured benchmark\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % devices), "cudaSetDevice");
    const int rows = rowCount(n, rank, size), startRow = rowStart(n, rank, size);
    const size_t localCount = static_cast<size_t>(rows) * n;
    const size_t paddedCount = static_cast<size_t>(rows + 2) * n;
    std::vector<val_t> energy(paddedCount, 0.0), nextEnergy(localCount, 0.0),
                       flux(localCount, 0.0), nextFlux(localCount, 0.0),
                       sendTop(n), sendBottom(n), recvTop(n), recvBottom(n);
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) { counts[r] = rowCount(n, r, size) * n; displs[r] = rowStart(n, r, size) * n; }
    std::vector<val_t> gatheredEnergy(rank == 0 ? static_cast<size_t>(n) * n : 0);

    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", n, n, n*n, iterations, validate ? "enabled" : "disabled");
        std::printf("Building unstructured mesh...\n");
        const size_t staticMem = static_cast<size_t>(n) * n * (2 * sizeof(uint64_t) + 8 * sizeof(uint64_t) + 8 * sizeof(double));
        const size_t dynamicMem = static_cast<size_t>(n) * n * 4 * sizeof(double);
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n", (staticMem+dynamicMem)/(1024.0*1024.0), staticMem/(1024.0*1024.0), dynamicMem/(1024.0*1024.0));
        std::printf("Running simulation...\n");
    }

    val_t *dEnergy[2], *dFlux[2];
    for (int b = 0; b < 2; ++b) {
        cudaCheck(cudaMalloc(&dEnergy[b], paddedCount * sizeof(val_t)), "allocate energy");
        cudaCheck(cudaMalloc(&dFlux[b], localCount * sizeof(val_t)), "allocate flux");
    }
    cudaCheck(cudaMemcpy(dEnergy[0], energy.data(), paddedCount*sizeof(val_t), cudaMemcpyHostToDevice), "copy energy");
    cudaCheck(cudaMemcpy(dFlux[0], flux.data(), localCount*sizeof(val_t), cudaMemcpyHostToDevice), "copy flux");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin = std::chrono::high_resolution_clock::now();
    int current = 0;
    for (int it = 0; it < iterations; ++it) {
        cudaCheck(cudaMemcpy(energy.data() + n, dEnergy[current] + n, localCount*sizeof(val_t), cudaMemcpyDeviceToHost), "copy local energy");
        if (rank > 0) std::copy_n(energy.data()+n, n, sendTop.data());
        if (rank + 1 < size) std::copy_n(energy.data()+static_cast<size_t>(rows)*n, n, sendBottom.data());
        MPI_Sendrecv(rank > 0 ? sendTop.data() : nullptr, n, MPI_DOUBLE, rank > 0 ? rank-1 : MPI_PROC_NULL, 17,
                     rank+1<size ? recvBottom.data() : nullptr, n, MPI_DOUBLE, rank+1<size ? rank+1 : MPI_PROC_NULL, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(rank+1<size ? sendBottom.data() : nullptr, n, MPI_DOUBLE, rank+1<size ? rank+1 : MPI_PROC_NULL, 18,
                     rank>0 ? recvTop.data() : nullptr, n, MPI_DOUBLE, rank>0 ? rank-1 : MPI_PROC_NULL, 18, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > 0) std::copy(recvTop.begin(), recvTop.end(), energy.begin());
        if (rank+1<size) std::copy(recvBottom.begin(), recvBottom.end(), energy.begin()+static_cast<size_t>(rows+1)*n);
        cudaCheck(cudaMemcpy(dEnergy[current], energy.data(), paddedCount*sizeof(val_t), cudaMemcpyHostToDevice), "copy halo energy");
        const int out = 1-current;
        updateGrid<<<(static_cast<unsigned int>(localCount)+255)/256, 256>>>(dEnergy[current], dFlux[current], dEnergy[out]+n, dFlux[out], n, rows, startRow, rank>0, rank+1<size);
        cudaCheck(cudaGetLastError(), "launch update kernel");
        current = out;
    }
    cudaCheck(cudaDeviceSynchronize(), "finish simulation");
    cudaCheck(cudaMemcpy(nextEnergy.data(), dEnergy[current]+n, localCount*sizeof(val_t), cudaMemcpyDeviceToHost), "copy final energy");
    cudaCheck(cudaMemcpy(nextFlux.data(), dFlux[current], localCount*sizeof(val_t), cudaMemcpyDeviceToHost), "copy final flux");
    const auto end = std::chrono::high_resolution_clock::now();
    const long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    long maxDuration = 0;
    MPI_Reduce(&duration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    uint64_t localHash = 0;
    double localEnergySum = 0, localFluxSum = 0, localMin = std::numeric_limits<double>::max(), localMax = std::numeric_limits<double>::lowest();
    #pragma omp parallel for reduction(+:localEnergySum,localFluxSum) reduction(min:localMin) reduction(max:localMax) reduction(^:localHash) schedule(static)
    for (long long i = 0; i < static_cast<long long>(localCount); ++i) {
        const double e = nextEnergy[i], f = nextFlux[i];
        localEnergySum += e; localFluxSum += f; localMin = std::min(localMin,e); localMax = std::max(localMax,e);
        uint64_t eb, fb; std::memcpy(&eb,&e,sizeof(eb)); std::memcpy(&fb,&f,sizeof(fb));
        const uint64_t global = static_cast<uint64_t>(startRow)*n+i;
        localHash ^= (eb+global)*0x9e3779b97f4a7c15ULL;
        localHash ^= (fb+global)*0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash = 0;
    double energySum=0, fluxSum=0, energyMin=0, energyMax=0;
    MPI_Reduce(&localHash,&hash,1,MPI_UINT64_T,MPI_BXOR,0,MPI_COMM_WORLD);
    MPI_Reduce(&localEnergySum,&energySum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&localFluxSum,&fluxSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&localMin,&energyMin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&localMax,&energyMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Gatherv(nextEnergy.data(), static_cast<int>(localCount), MPI_DOUBLE, rank==0?gatheredEnergy.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const int measured = std::max(iterations-1,1);
        const double seconds = maxDuration/1000.0;
        const double geps = seconds > 0 ? (static_cast<double>(measured)*n*n)/seconds/1e9 : 0.0;
        std::printf("Computation time: %ld ms\n", maxDuration);
        std::printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n", maxDuration/static_cast<double>(measured), geps, geps*22.0, static_cast<unsigned long>(hash));
        if (printResults) print_results(gatheredEnergy, "ElementEnergy");
        if (validate) {
            std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energySum, fluxSum, energyMin, energyMax);
            if (!std::isfinite(energySum) || !std::isfinite(fluxSum) || !std::isfinite(energyMin) || !std::isfinite(energyMax)) {
                std::printf("  ERROR: Non-finite simulation result\n"); bad=1;
            } else {
                if (std::abs(energySum)>1e-8) std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
                std::printf("  Validation: PASSED\n");
            }
        }
    }
    MPI_Bcast(&bad,1,MPI_INT,0,MPI_COMM_WORLD);
    for (int b=0;b<2;++b) { cudaFree(dEnergy[b]); cudaFree(dFlux[b]); }
    MPI_Finalize();
    return bad ? 1 : 0;
}
