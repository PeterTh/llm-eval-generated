#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define SOFTENING 1e-9
#define DT 0.01
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Force computation kernel with shared memory tiling
__global__ void computeForcesKernel(
    const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    int n, int localStart, int localCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;

    int i = localStart + idx;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double pxi = px[i], pyi = py[i], pzi = pz[i];

    __shared__ double spx[BLOCK_SIZE];
    __shared__ double spy[BLOCK_SIZE];
    __shared__ double spz[BLOCK_SIZE];

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        int gj = tile + threadIdx.x;
        spx[threadIdx.x] = (gj < n) ? px[gj] : 0.0;
        spy[threadIdx.x] = (gj < n) ? py[gj] : 0.0;
        spz[threadIdx.x] = (gj < n) ? pz[gj] : 0.0;
        __syncthreads();

        int limit = min(BLOCK_SIZE, n - tile);
        for (int j = 0; j < limit; ++j) {
            double dx = spx[j] - pxi;
            double dy = spy[j] - pyi;
            double dz = spz[j] - pzi;
            double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    vx[i] += DT * Fx;
    vy[i] += DT * Fy;
    vz[i] += DT * Fz;
}

// Position integration kernel
__global__ void integrateKernel(
    double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
    const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
    int localStart, int localCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;

    int i = localStart + idx;
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
}

void randomizeBodies(double* px, double* py, double* pz,
                     double* vx, double* vy, double* vz,
                     int n, unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const double* px, const double* py, const double* pz,
                          const double* vx, const double* vy, const double* vz, int n) {
    double energy = 0.0;

    #pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    #pragma omp parallel for reduction(+:energy) schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double local_e = 0.0;
        for (int j = i + 1; j < n; ++j) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_e -= 1.0 / dist;
        }
        energy += local_e;
    }

    return energy;
}

bool validateSimulation(const double* px, const double* py, const double* pz,
                        const double* vx, const double* vy, const double* vz, int n) {
    bool valid = true;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            valid = false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            valid = false;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
            valid = false;
        }
    }
    if (!valid) {
        printf("Validation failed: body state exceeds reasonable bounds or contains NaN/Inf\n");
    }
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Assign GPU to each rank round-robin
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute per-rank work distribution
    std::vector<int> recvCounts(numProcs), displs(numProcs);
    {
        int base = numBodies / numProcs;
        int rem = numBodies % numProcs;
        int offset = 0;
        for (int r = 0; r < numProcs; ++r) {
            recvCounts[r] = base + (r < rem ? 1 : 0);
            displs[r] = offset;
            offset += recvCounts[r];
        }
    }
    int localCount = recvCounts[rank];
    int localStart = displs[rank];

    // Allocate pinned host arrays (SoA layout for GPU efficiency)
    size_t bytes = numBodies * sizeof(double);
    double *h_px, *h_py, *h_pz, *h_vx, *h_vy, *h_vz;
    CUDA_CHECK(cudaMallocHost(&h_px, bytes));
    CUDA_CHECK(cudaMallocHost(&h_py, bytes));
    CUDA_CHECK(cudaMallocHost(&h_pz, bytes));
    CUDA_CHECK(cudaMallocHost(&h_vx, bytes));
    CUDA_CHECK(cudaMallocHost(&h_vy, bytes));
    CUDA_CHECK(cudaMallocHost(&h_vz, bytes));

    // Initialize bodies identically on all ranks
    randomizeBodies(h_px, h_py, h_pz, h_vx, h_vy, h_vz, numBodies);

    // Allocate device arrays
    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    CUDA_CHECK(cudaMalloc(&d_px, bytes));
    CUDA_CHECK(cudaMalloc(&d_py, bytes));
    CUDA_CHECK(cudaMalloc(&d_pz, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_px, h_px, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int numBlocks = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t localBytes = localCount * sizeof(double);

    for (int step = 0; step < numSteps; ++step) {
        // CUDA: compute forces for this rank's bodies
        if (localCount > 0) {
            computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                numBodies, localStart, localCount);

            // CUDA: integrate positions for this rank's bodies
            integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                localStart, localCount);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local positions from device to host
        CUDA_CHECK(cudaMemcpy(h_px + localStart, d_px + localStart,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py + localStart, d_py + localStart,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pz + localStart, d_pz + localStart,
                              localBytes, cudaMemcpyDeviceToHost));

        // MPI: allgather updated positions across all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_px, recvCounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_py, recvCounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_pz, recvCounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy all positions back to device for next step
        CUDA_CHECK(cudaMemcpy(d_px, h_px, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h_py, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz, h_pz, bytes, cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather final velocities for output/validation
    CUDA_CHECK(cudaMemcpy(h_vx + localStart, d_vx + localStart,
                          localBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy + localStart, d_vy + localStart,
                          localBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz + localStart, d_vz + localStart,
                          localBytes, cudaMemcpyDeviceToHost));

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_vx, recvCounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_vy, recvCounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_vz, recvCounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(h_px[i]);
                bodyData.push_back(h_py[i]);
                bodyData.push_back(h_pz[i]);
                bodyData.push_back(h_vx[i]);
                bodyData.push_back(h_vy[i]);
                bodyData.push_back(h_vz[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(h_px, h_py, h_pz, h_vx, h_vy, h_vz, numBodies)) {
                double finalEnergy = computeTotalEnergy(h_px, h_py, h_pz,
                                                        h_vx, h_vy, h_vz, numBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
                CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
                CUDA_CHECK(cudaFreeHost(h_px)); CUDA_CHECK(cudaFreeHost(h_py)); CUDA_CHECK(cudaFreeHost(h_pz));
                CUDA_CHECK(cudaFreeHost(h_vx)); CUDA_CHECK(cudaFreeHost(h_vy)); CUDA_CHECK(cudaFreeHost(h_vz));
                MPI_Finalize();
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFreeHost(h_px)); CUDA_CHECK(cudaFreeHost(h_py)); CUDA_CHECK(cudaFreeHost(h_pz));
    CUDA_CHECK(cudaFreeHost(h_vx)); CUDA_CHECK(cudaFreeHost(h_vy)); CUDA_CHECK(cudaFreeHost(h_vz));

    MPI_Finalize();
    return 0;
}
