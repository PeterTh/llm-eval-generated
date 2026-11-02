#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each thread owns one body. All threads in a block cooperate in loading
// position tiles, and read only the positions from the start of this step.
__global__ void advance(const double4* positions, const double4* velocities,
                        double4* nextPositions, double4* nextVelocities,
                        int n, int first, int localCount) {
    __shared__ double4 tile[BLOCK_SIZE];
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = local < localCount;
    double4 p = active ? positions[first + local] : make_double4(0, 0, 0, 0);
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                double dx = tile[k].x - p.x;
                double dy = tile[k].y - p.y;
                double dz = tile[k].z - p.z;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (active) {
        double4 v = velocities[local];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        p.x += v.x * DT; p.y += v.y * DT; p.z += v.z * DT;
        nextVelocities[local] = v;
        nextPositions[first + local] = p;
    }
}

// One partial energy per local body; the host combines these across ranks.
__global__ void energyKernel(const double4* positions, const double4* velocities,
                             double* energy, int n, int first, int localCount) {
    __shared__ double4 tile[BLOCK_SIZE];
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    int i = first + local;
    bool active = local < localCount;
    double4 p = active ? positions[i] : make_double4(0, 0, 0, 0);
    double e = 0.0;
    if (active) {
        double4 v = velocities[local];
        e = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = max(0, i + 1 - base); k < count; ++k) {
                double dx = tile[k].x - p.x, dy = tile[k].y - p.y, dz = tile[k].z - p.z;
                e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (active) energy[local] = e;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
           "  -s <num>     Number of simulation steps (default: 10)\n"
           "  -v           Enable validation (checks energy conservation)\n"
           "  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n < 0 || n > INT_MAX / 4) {
        if (!rank) fprintf(stderr, "Number of bodies is out of range\n");
        MPI_Finalize(); return 1;
    }
    if (!rank) printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                      n, steps, validate ? "enabled" : "disabled");

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        int size = n / ranks + (r < n % ranks);
        counts[r] = 4 * size; offsets[r] = 4 * offset; offset += size;
    }
    int first = offsets[rank] / 4, localCount = counts[rank] / 4;
    size_t fullBytes = std::max(n, 1) * sizeof(double4);
    size_t localBytes = std::max(localCount, 1) * sizeof(double4);
    double4 *hostPos, *hostVel;
    CUDA_CHECK(cudaMallocHost(&hostPos, fullBytes));
    CUDA_CHECK(cudaMallocHost(&hostVel, fullBytes));
    // rand_r is intentionally serial: this preserves the original seed and order.
    unsigned int seed = 42;
    std::vector<double> initialValues((size_t)n * 6);
    for (double &value : initialValues)
        value = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    int initThreads = std::min(omp_get_max_threads(), std::max(2, (n + 4095) / 4096));
    #pragma omp parallel for schedule(static) num_threads(initThreads)
    for (int i = 0; i < n; ++i) {
        const double *values = initialValues.data() + 6ULL * i;
        hostPos[i] = make_double4(values[0], values[1], values[2], 0);
        hostVel[i] = make_double4(values[3], values[4], values[5], 0);
    }
    initialValues.clear();
    initialValues.shrink_to_fit();
    double4 *pos, *nextPos, *vel;
    CUDA_CHECK(cudaMalloc(&pos, fullBytes));
    CUDA_CHECK(cudaMalloc(&nextPos, fullBytes));
    CUDA_CHECK(cudaMalloc(&vel, localBytes));
    if (n) CUDA_CHECK(cudaMemcpy(pos, hostPos, n * sizeof(double4), cudaMemcpyHostToDevice));
    if (localCount) CUDA_CHECK(cudaMemcpy(vel, hostVel + first, localCount * sizeof(double4), cudaMemcpyHostToDevice));
    int blocks = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        if (localCount) {
            advance<<<blocks, BLOCK_SIZE>>>(pos, vel, nextPos, vel, n, first, localCount);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks > 1) {
            if (localCount) CUDA_CHECK(cudaMemcpy(hostPos + first, nextPos + first,
                                                  localCount * sizeof(double4), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hostPos, counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            if (n) CUDA_CHECK(cudaMemcpy(nextPos, hostPos, n * sizeof(double4), cudaMemcpyHostToDevice));
        }
        std::swap(pos, nextPos);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    long local_ms = duration.count(), global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Simulation time: %ld ms\n", global_ms);

    if (printResults || validate) {
        if (localCount) {
            CUDA_CHECK(cudaMemcpy(hostPos + first, pos + first, localCount * sizeof(double4), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostVel + first, vel, localCount * sizeof(double4), cudaMemcpyDeviceToHost));
        }
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hostPos + first, counts[rank], MPI_DOUBLE,
                    hostPos, counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hostVel + first, counts[rank], MPI_DOUBLE,
                    hostVel, counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (printResults && !rank) {
        std::vector<double> data((size_t)n * 6);
        #pragma omp parallel for schedule(static) num_threads(initThreads)
        for (int i = 0; i < n; ++i) {
            data[6ULL*i] = hostPos[i].x; data[6ULL*i+1] = hostPos[i].y; data[6ULL*i+2] = hostPos[i].z;
            data[6ULL*i+3] = hostVel[i].x; data[6ULL*i+4] = hostVel[i].y; data[6ULL*i+5] = hostVel[i].z;
        }
        print_results(data, "Bodies");
    }
    int status = 0;
    if (validate) {
        int bad = n;
        if (!rank) {
            printf("Validating simulation results...\n");
            #pragma omp parallel for reduction(min:bad) schedule(static) num_threads(initThreads)
            for (int i = 0; i < n; ++i) {
                double4 p = hostPos[i], v = hostVel[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                    !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
                    std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6 ||
                    std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) bad = std::min(bad, i);
            }
            if (bad < n) {
                double4 p = hostPos[bad], v = hostVel[bad];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                    !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
                    printf("Validation failed: found NaN or Inf value in body state\n");
                else if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6)
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                else printf("Validation failed: body velocity exceeds reasonable bounds\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!status) {
            double *deviceEnergy;
            CUDA_CHECK(cudaMalloc(&deviceEnergy, std::max(localCount, 1) * sizeof(double)));
            std::vector<double> partial(localCount);
            if (localCount) {
                energyKernel<<<blocks, BLOCK_SIZE>>>(pos, vel, deviceEnergy, n, first, localCount);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(partial.data(), deviceEnergy, localCount * sizeof(double), cudaMemcpyDeviceToHost));
            }
            double localEnergy = 0.0, totalEnergy = 0.0;
            #pragma omp parallel for reduction(+:localEnergy) schedule(static) num_threads(initThreads)
            for (int i = 0; i < localCount; ++i) localEnergy += partial[i];
            MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!rank) printf("Final energy: %.6f\nValidation: PASSED\n", totalEnergy);
            CUDA_CHECK(cudaFree(deviceEnergy));
        } else if (!rank) printf("Validation: FAILED\n");
    }
    CUDA_CHECK(cudaFree(pos)); CUDA_CHECK(cudaFree(nextPos)); CUDA_CHECK(cudaFree(vel));
    CUDA_CHECK(cudaFreeHost(hostPos)); CUDA_CHECK(cudaFreeHost(hostVel));
    MPI_Finalize();
    return status;
}
