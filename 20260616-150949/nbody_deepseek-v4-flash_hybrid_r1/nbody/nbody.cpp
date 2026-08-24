#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                  \
} while(0)

struct Body {
    double x, y, z, vx, vy, vz;
};

void randomizeBodies(Body* bodies, int n, unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        bodies[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: force computation with shared-memory tiling
// Each thread handles one local body; tiles of all-body positions are loaded
// into shared memory to reduce global-memory traffic.
// ---------------------------------------------------------------------------
__global__ void bodyForceKernel(
    double* velX, double* velY, double* velZ,
    const double* posX, const double* posY, const double* posZ,
    int n, int localStart, int localCount)
{
    extern __shared__ double sdata[];
    double* sX = sdata;
    double* sY = &sdata[blockDim.x];
    double* sZ = &sdata[2 * blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Own position for this thread's body
    double px, py, pz;
    if (idx < localCount) {
        int gi = localStart + idx;
        px = posX[gi];
        py = posY[gi];
        pz = posZ[gi];
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;

    // Tile over all bodies: each tile fills one blockDim.x-sized shared-memory slab
    for (int tile = 0; tile < n; tile += blockDim.x) {
        int j = tile + tid;
        if (j < n) {
            sX[tid] = posX[j];
            sY[tid] = posY[j];
            sZ[tid] = posZ[j];
        }
        __syncthreads();

        if (idx < localCount) {
            int jMax = blockDim.x < (n - tile) ? blockDim.x : (n - tile);
            for (int jj = 0; jj < jMax; ++jj) {
                double dx = sX[jj] - px;
                double dy = sY[jj] - py;
                double dz = sZ[jj] - pz;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = rsqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (idx < localCount) {
        velX[idx] += DT * fx;
        velY[idx] += DT * fy;
        velZ[idx] += DT * fz;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main : hybrid MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Assign GPU: round-robin over available devices
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 0) {
        CUDA_CHECK(cudaSetDevice(mpiRank % numDevices));
    }

    // Default parameters
    int numBodies = 1024;
    int numSteps = 10;
    int validate   = 0;
    int printRes   = 0;

    // Rank 0 parses command-line and broadcasts
    if (mpiRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printRes = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printRes,  1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- distribute bodies across MPI ranks ----
    int baseCount  = numBodies / mpiSize;
    int remainder  = numBodies % mpiSize;
    int localCount = baseCount + (mpiRank < remainder ? 1 : 0);
    int localStart = mpiRank * baseCount + std::min(mpiRank, remainder);

    std::vector<int> counts(mpiSize), displs(mpiSize);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < mpiSize; ++i)
        displs[i] = displs[i - 1] + counts[i - 1];

    if (mpiRank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", mpiSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- pinned host memory (faster GPU transfers) ----
    double *h_posX, *h_posY, *h_posZ;
    double *h_velX, *h_velY, *h_velZ;
    double *h_allPosX, *h_allPosY, *h_allPosZ;

    CUDA_CHECK(cudaMallocHost(&h_posX,    localCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_posY,    localCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_posZ,    localCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_velX,    localCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_velY,    localCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_velZ,    localCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_allPosX, numBodies  * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_allPosY, numBodies  * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_allPosZ, numBodies  * sizeof(double)));

    // ---- GPU memory ----
    double *d_posX, *d_posY, *d_posZ;
    double *d_velX, *d_velY, *d_velZ;

    CUDA_CHECK(cudaMalloc(&d_posX, numBodies  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posY, numBodies  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posZ, numBodies  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velX, localCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velY, localCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velZ, localCount * sizeof(double)));

    // ---- initialise on rank 0, then scatter ----
    double *g_posX = nullptr, *g_posY = nullptr, *g_posZ = nullptr;
    double *g_velX = nullptr, *g_velY = nullptr, *g_velZ = nullptr;

    if (mpiRank == 0) {
        CUDA_CHECK(cudaMallocHost(&g_posX, numBodies * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&g_posY, numBodies * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&g_posZ, numBodies * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&g_velX, numBodies * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&g_velY, numBodies * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&g_velZ, numBodies * sizeof(double)));

        std::vector<Body> allBodies(numBodies);
        randomizeBodies(allBodies.data(), numBodies);
        for (int i = 0; i < numBodies; ++i) {
            g_posX[i] = allBodies[i].x;
            g_posY[i] = allBodies[i].y;
            g_posZ[i] = allBodies[i].z;
            g_velX[i] = allBodies[i].vx;
            g_velY[i] = allBodies[i].vy;
            g_velZ[i] = allBodies[i].vz;
        }
    }

    MPI_Scatterv(g_posX, counts.data(), displs.data(), MPI_DOUBLE,
                 h_posX, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g_posY, counts.data(), displs.data(), MPI_DOUBLE,
                 h_posY, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g_posZ, counts.data(), displs.data(), MPI_DOUBLE,
                 h_posZ, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g_velX, counts.data(), displs.data(), MPI_DOUBLE,
                 h_velX, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g_velY, counts.data(), displs.data(), MPI_DOUBLE,
                 h_velY, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(g_velZ, counts.data(), displs.data(), MPI_DOUBLE,
                 h_velZ, localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        CUDA_CHECK(cudaFreeHost(g_posX)); CUDA_CHECK(cudaFreeHost(g_posY));
        CUDA_CHECK(cudaFreeHost(g_posZ));
        CUDA_CHECK(cudaFreeHost(g_velX)); CUDA_CHECK(cudaFreeHost(g_velY));
        CUDA_CHECK(cudaFreeHost(g_velZ));
    }

    // ---- main simulation loop ----
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int BLOCK = 256;

    for (int step = 0; step < numSteps; ++step) {
        // 1. Allgather: every rank gets all positions
        MPI_Allgatherv(h_posX, localCount, MPI_DOUBLE,
                       h_allPosX, counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_posY, localCount, MPI_DOUBLE,
                       h_allPosY, counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_posZ, localCount, MPI_DOUBLE,
                       h_allPosZ, counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // 2. Upload positions to GPU
        CUDA_CHECK(cudaMemcpy(d_posX, h_allPosX, numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posY, h_allPosY, numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posZ, h_allPosZ, numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));

        // 3. Upload local velocities to GPU
        CUDA_CHECK(cudaMemcpy(d_velX, h_velX, localCount * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_velY, h_velY, localCount * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_velZ, h_velZ, localCount * sizeof(double),
                              cudaMemcpyHostToDevice));

        // 4. Launch CUDA kernel (force computation + velocity update)
        int grid = (localCount + BLOCK - 1) / BLOCK;
        size_t smem = 3 * BLOCK * sizeof(double);
        bodyForceKernel<<<grid, BLOCK, smem>>>(
            d_velX, d_velY, d_velZ,
            d_posX, d_posY, d_posZ,
            numBodies, localStart, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 5. Download updated velocities
        CUDA_CHECK(cudaMemcpy(h_velX, d_velX, localCount * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_velY, d_velY, localCount * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_velZ, d_velZ, localCount * sizeof(double),
                              cudaMemcpyDeviceToHost));

        // 6. Integrate positions (OpenMP)
        #pragma omp parallel for
        for (int i = 0; i < localCount; ++i) {
            h_posX[i] += h_velX[i] * DT;
            h_posY[i] += h_velY[i] * DT;
            h_posZ[i] += h_velZ[i] * DT;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDuration = static_cast<long long>(duration.count());
    long long globalDuration;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG_INT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (mpiRank == 0)
        printf("Simulation time: %lld ms\n", globalDuration);

    // ---- optional: print results for external validation ----
    if (printRes) {
        double *gX = nullptr, *gY = nullptr, *gZ = nullptr;
        double *gVX = nullptr, *gVY = nullptr, *gVZ = nullptr;

        if (mpiRank == 0) {
            CUDA_CHECK(cudaMallocHost(&gX,  numBodies * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&gY,  numBodies * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&gZ,  numBodies * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&gVX, numBodies * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&gVY, numBodies * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&gVZ, numBodies * sizeof(double)));
        }

        MPI_Gatherv(h_posX, localCount, MPI_DOUBLE,
                    gX, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_posY, localCount, MPI_DOUBLE,
                    gY, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_posZ, localCount, MPI_DOUBLE,
                    gZ, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_velX, localCount, MPI_DOUBLE,
                    gVX, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_velY, localCount, MPI_DOUBLE,
                    gVY, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_velZ, localCount, MPI_DOUBLE,
                    gVZ, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (mpiRank == 0) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(gX[i]);  bodyData.push_back(gY[i]);
                bodyData.push_back(gZ[i]);
                bodyData.push_back(gVX[i]); bodyData.push_back(gVY[i]);
                bodyData.push_back(gVZ[i]);
            }
            print_results(bodyData, "Bodies");

            CUDA_CHECK(cudaFreeHost(gX));  CUDA_CHECK(cudaFreeHost(gY));
            CUDA_CHECK(cudaFreeHost(gZ));
            CUDA_CHECK(cudaFreeHost(gVX)); CUDA_CHECK(cudaFreeHost(gVY));
            CUDA_CHECK(cudaFreeHost(gVZ));
        }
    }

    // ---- optional: validate ----
    if (validate) {
        // Check for NaN / Inf / extreme values on local bodies
        int invalid = 0;
        #pragma omp parallel for reduction(+:invalid)
        for (int i = 0; i < localCount; ++i) {
            if (!std::isfinite(h_posX[i]) || !std::isfinite(h_posY[i]) ||
                !std::isfinite(h_posZ[i]) || !std::isfinite(h_velX[i]) ||
                !std::isfinite(h_velY[i]) || !std::isfinite(h_velZ[i]))
                ++invalid;
            const double maxPos = 1e6, maxVel = 1e6;
            if (std::abs(h_posX[i]) > maxPos || std::abs(h_posY[i]) > maxPos ||
                std::abs(h_posZ[i]) > maxPos || std::abs(h_velX[i]) > maxVel ||
                std::abs(h_velY[i]) > maxVel || std::abs(h_velZ[i]) > maxVel)
                ++invalid;
        }

        int globalValid;
        MPI_Allreduce(&invalid, &globalValid, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);

        if (mpiRank == 0 && globalValid != 0)
            printf("Validation failed: found NaN, Inf, or extreme values\n");

        // Total energy
        double kin = 0.0;
        #pragma omp parallel for reduction(+:kin)
        for (int i = 0; i < localCount; ++i)
            kin += 0.5 * (h_velX[i] * h_velX[i] + h_velY[i] * h_velY[i] +
                          h_velZ[i] * h_velZ[i]);

        double pot = 0.0;
        #pragma omp parallel for reduction(+:pot)
        for (int i = localStart; i < localStart + localCount; ++i)
            for (int j = i + 1; j < numBodies; ++j) {
                double dx = h_allPosX[j] - h_allPosX[i];
                double dy = h_allPosY[j] - h_allPosY[i];
                double dz = h_allPosZ[j] - h_allPosZ[i];
                pot -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }

        double localTotal = kin + pot;
        double totalEnergy;
        MPI_Reduce(&localTotal, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        if (mpiRank == 0) {
            if (globalValid == 0) {
                printf("Validating simulation results...\n");
                printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // ---- cleanup ----
    CUDA_CHECK(cudaFreeHost(h_posX));    CUDA_CHECK(cudaFreeHost(h_posY));
    CUDA_CHECK(cudaFreeHost(h_posZ));
    CUDA_CHECK(cudaFreeHost(h_velX));    CUDA_CHECK(cudaFreeHost(h_velY));
    CUDA_CHECK(cudaFreeHost(h_velZ));
    CUDA_CHECK(cudaFreeHost(h_allPosX)); CUDA_CHECK(cudaFreeHost(h_allPosY));
    CUDA_CHECK(cudaFreeHost(h_allPosZ));
    CUDA_CHECK(cudaFree(d_posX));        CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));        CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));

    MPI_Finalize();
    return 0;
}
