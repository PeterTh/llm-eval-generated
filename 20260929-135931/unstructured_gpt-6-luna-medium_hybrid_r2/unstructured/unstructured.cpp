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
struct ElementDynamic { val_t current_energy, total_flux; };

__global__ void update(const val_t* cur, val_t* next, val_t* accumulated,
                       int n, int rows, int first, int root) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * root;
    if (k >= count) return;
    const int r = k / root, c = k % root, gr = first + r;
    const int p = (r + 1) * root + c;
    const val_t e = cur[p];
    val_t ext = 0.0;
    if (gr == 0 && c == 0) ext = 0.5;
    if (gr == 0 && c == root - 1) ext = -0.5;
    if (gr == root - 1 && c == 0) ext = -0.5;
    if (gr == root - 1 && c == root - 1) ext = 0.5;
    val_t f = ext;
    if (gr + 1 < root) f += (cur[p + root] - e) * 0.2;
    if (gr > 0) f += (cur[p - root] - e) * 0.2;
    if (c + 1 < root) f += (cur[p + 1] - e) * 0.2;
    if (c > 0) f += (cur[p - 1] - e) * 0.2;
    next[p] = e + f;
    accumulated[k] += fabs(f);
}

static void cudaCheck(cudaError_t e) { if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } }

void runSimulation(int root, int iters, int rank, int size, int first, int rows,
                   std::vector<ElementDynamic>& result) {
    const size_t n = static_cast<size_t>(rows) * root;
    std::vector<val_t> hostA((rows + 2) * root, 0.0), hostB((rows + 2) * root, 0.0), flux(n, 0.0);
    val_t *a, *b, *dflux;
    cudaCheck(cudaMalloc(&a, hostA.size() * sizeof(val_t)));
    cudaCheck(cudaMalloc(&b, hostB.size() * sizeof(val_t)));
    cudaCheck(cudaMalloc(&dflux, n * sizeof(val_t)));
    cudaCheck(cudaMemset(dflux, 0, n * sizeof(val_t)));
    int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int down = rank + 1 == size ? MPI_PROC_NULL : rank + 1;
    for (int it = 0; it < iters; ++it) {
        cudaCheck(cudaMemcpy(a, hostA.data(), hostA.size() * sizeof(val_t), cudaMemcpyHostToDevice));
        if (rank > 0) MPI_Sendrecv(hostA.data() + root, root, MPI_DOUBLE, up, 11,
                                  hostA.data(), root, MPI_DOUBLE, up, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank + 1 < size) MPI_Sendrecv(hostA.data() + rows * root, root, MPI_DOUBLE, down, 12,
                                         hostA.data() + (rows + 1) * root, root, MPI_DOUBLE, down, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaCheck(cudaMemcpy(a, hostA.data(), hostA.size() * sizeof(val_t), cudaMemcpyHostToDevice));
        update<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(a, b, dflux, root, rows, first, root);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        std::swap(a, b);
        // Copy the updated owned rows back for the next halo exchange.
        cudaCheck(cudaMemcpy(hostA.data() + root, a + root, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    cudaCheck(cudaMemcpy(flux.data(), dflux, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    result.resize(n);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        result[i] = {hostA[root + i], flux[i]};
    cudaFree(a); cudaFree(b); cudaFree(dflux);
}

uint64_t computeHash(const std::vector<ElementDynamic>& e) {
    uint64_t h = 0;
    for (size_t i = 0; i < e.size(); ++i) {
        uint64_t x, y; memcpy(&x, &e[i].current_energy, 8); memcpy(&y, &e[i].total_flux, 8);
        h ^= (x + i) * 0x9e3779b97f4a7c15ULL; h ^= (y + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    int root = 512, iters = 10, validate = 0, printResults = 0, bad = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) root = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = 1;
        else if (!strcmp(argv[i], "-r")) printResults = 1;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printf("Usage: %s [-n grid] [-i iterations] [-v] [-r] [-h]\n", argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) fprintf(stderr, "Unknown option: %s\n", argv[i]); bad = 1; }
    }
    if (root <= 0 || iters < 0 || root < size) bad = 1;
    int anybad; MPI_Allreduce(&bad, &anybad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anybad) { MPI_Finalize(); return 1; }
    int q = root / size, rem = root % size;
    int rows = q + (rank < rem), first = rank * q + std::min(rank, rem);
    int n = root * root;
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", root, root, n, iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\nMemory usage: %.2f MB (distributed)\n\nRunning simulation...\n", n * (sizeof(ElementDynamic) * 2 + 4 * sizeof(val_t)) / (1024.0 * 1024.0));
    }
    int deviceCount = 0; cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount) cudaCheck(cudaSetDevice(rank % deviceCount));
    std::vector<ElementDynamic> local;
    auto start = std::chrono::high_resolution_clock::now();
    runSimulation(root, iters, rank, size, first, rows, local);
    auto end = std::chrono::high_resolution_clock::now();
    int *counts = nullptr, *displs = nullptr;
    if (!rank) { counts = new int[size]; displs = new int[size]; for (int r = 0, off = 0; r < size; ++r) { counts[r] = 2 * (root / size + (r < rem)) * root; displs[r] = off; off += counts[r]; } }
    std::vector<ElementDynamic> all(rank == 0 ? n : 0);
    MPI_Gatherv(local.data(), static_cast<int>(local.size() * 2), MPI_DOUBLE, all.data(), counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Computation time: %ld ms\n", ms);
        int measured = std::max(iters - 1, 1); double eps = (measured * static_cast<double>(n)) / (ms / 1000.0) / 1e9;
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n", static_cast<double>(ms) / measured, eps, eps * 22, computeHash(all));
        if (printResults) { std::vector<double> data(n); for (int i = 0; i < n; ++i) data[i] = all[i].current_energy; print_results(data, "ElementEnergy"); }
        if (validate) { double es = 0, fs = 0, emin = INFINITY, emax = -INFINITY; for (auto& e : all) { es += e.current_energy; fs += e.total_flux; emin = std::min(emin, e.current_energy); emax = std::max(emax, e.current_energy); }
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", es, fs, emin, emax);
            if (!std::isfinite(es) || !std::isfinite(fs) || !std::isfinite(emin) || !std::isfinite(emax)) { printf("  ERROR: non-finite result\n"); bad = 1; } else { if (std::abs(es) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n"); printf("  Validation: PASSED\n"); }
        }
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD); delete[] counts; delete[] displs; MPI_Finalize(); return bad;
}
