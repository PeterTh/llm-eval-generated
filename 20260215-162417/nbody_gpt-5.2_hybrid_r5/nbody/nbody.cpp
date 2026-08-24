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

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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

struct Pos3 {
    double x, y, z;
};

__global__ void nbody_step_kernel(const double* __restrict__ allPos, double* __restrict__ localPos,
                                 double* __restrict__ localVel, int n, int nLocal) {
    extern __shared__ Pos3 shPos[];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (li < nLocal);

    double ix = 0.0, iy = 0.0, iz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    if (active) {
        ix = localPos[3 * li + 0];
        iy = localPos[3 * li + 1];
        iz = localPos[3 * li + 2];
        vx = localVel[3 * li + 0];
        vy = localVel[3 * li + 1];
        vz = localVel[3 * li + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const int j = tileStart + threadIdx.x;
        if (j < n) {
            shPos[threadIdx.x].x = allPos[3 * j + 0];
            shPos[threadIdx.x].y = allPos[3 * j + 1];
            shPos[threadIdx.x].z = allPos[3 * j + 2];
        }
        __syncthreads();

        const int tileCount = min(blockDim.x, n - tileStart);
        if (active) {
            #pragma unroll 4
            for (int k = 0; k < tileCount; ++k) {
                const double dx = shPos[k].x - ix;
                const double dy = shPos[k].y - iy;
                const double dz = shPos[k].z - iz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        vx += DT * Fx;
        vy += DT * Fy;
        vz += DT * Fz;

        localVel[3 * li + 0] = vx;
        localVel[3 * li + 1] = vy;
        localVel[3 * li + 2] = vz;

        localPos[3 * li + 0] = ix + vx * DT;
        localPos[3 * li + 1] = iy + vy * DT;
        localPos[3 * li + 2] = iz + vz * DT;
    }
}

static void computeCountsDispls(int n, int worldSize, std::vector<int>& countsBodies, std::vector<int>& displsBodies) {
    countsBodies.assign(worldSize, 0);
    displsBodies.assign(worldSize, 0);
    const int base = n / worldSize;
    const int rem = n % worldSize;
    int disp = 0;
    for (int r = 0; r < worldSize; ++r) {
        const int cnt = base + (r < rem ? 1 : 0);
        countsBodies[r] = cnt;
        displsBodies[r] = disp;
        disp += cnt;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    int validate = 0;
    int printResults = 0;

    // action: 1=run, 0=help/exit success, -1=error/exit failure
    int action = 1;

    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                action = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                action = -1;
            }
        }
    }

    MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (action != 1) {
        MPI_Finalize();
        return (action == 0) ? 0 : 1;
    }
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Map ranks to devices within a node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int devCount = 0;
    cudaCheck(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount <= 0) {
        if (worldRank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devCount), "cudaSetDevice");
    cudaCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte), "cudaDeviceSetSharedMemConfig");

    MPI_Comm_free(&localComm);

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
    }

    std::vector<int> countsBodies, displsBodies;
    computeCountsDispls(numBodies, worldSize, countsBodies, displsBodies);

    const int localBodies = countsBodies[worldRank];
    const int localPosCount = localBodies * 3;

    // Host buffers (positions are exchanged every step).
    double* h_localPos = nullptr;
    double* h_localVel = nullptr;
    double* h_allPos = nullptr;
    cudaCheck(cudaHostAlloc((void**)&h_localPos, (localPosCount > 0 ? localPosCount : 1) * sizeof(double), cudaHostAllocDefault),
              "cudaHostAlloc h_localPos");
    cudaCheck(cudaHostAlloc((void**)&h_localVel, (localPosCount > 0 ? localPosCount : 1) * sizeof(double), cudaHostAllocDefault),
              "cudaHostAlloc h_localVel");
    cudaCheck(cudaHostAlloc((void**)&h_allPos, (numBodies * 3 > 0 ? numBodies * 3 : 1) * sizeof(double), cudaHostAllocDefault),
              "cudaHostAlloc h_allPos");

    std::vector<double> rootAllPos;
    std::vector<double> rootAllVel;

    if (worldRank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);

        rootAllPos.resize((size_t)numBodies * 3);
        rootAllVel.resize((size_t)numBodies * 3);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            rootAllPos[3 * i + 0] = bodies[i].pos.x;
            rootAllPos[3 * i + 1] = bodies[i].pos.y;
            rootAllPos[3 * i + 2] = bodies[i].pos.z;
            rootAllVel[3 * i + 0] = bodies[i].vel.x;
            rootAllVel[3 * i + 1] = bodies[i].vel.y;
            rootAllVel[3 * i + 2] = bodies[i].vel.z;
        }
    }

    // Scatter initial state.
    std::vector<int> countsPos(worldSize), displsPos(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        countsPos[r] = countsBodies[r] * 3;
        displsPos[r] = displsBodies[r] * 3;
    }

    MPI_Scatterv(worldRank == 0 ? rootAllPos.data() : nullptr, countsPos.data(), displsPos.data(), MPI_DOUBLE,
                h_localPos, localPosCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(worldRank == 0 ? rootAllVel.data() : nullptr, countsPos.data(), displsPos.data(), MPI_DOUBLE,
                h_localVel, localPosCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Build global position vector for the first step.
    MPI_Allgatherv(h_localPos, localPosCount, MPI_DOUBLE, h_allPos, countsPos.data(), displsPos.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Device buffers.
    double* d_allPos = nullptr;
    double* d_localPos = nullptr;
    double* d_localVel = nullptr;

    cudaCheck(cudaMalloc((void**)&d_allPos, (size_t)numBodies * 3 * sizeof(double)), "cudaMalloc d_allPos");
    cudaCheck(cudaMalloc((void**)&d_localPos, (size_t)(localPosCount > 0 ? localPosCount : 1) * sizeof(double)),
              "cudaMalloc d_localPos");
    cudaCheck(cudaMalloc((void**)&d_localVel, (size_t)(localPosCount > 0 ? localPosCount : 1) * sizeof(double)),
              "cudaMalloc d_localVel");

    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    if (localPosCount > 0) {
        cudaCheck(cudaMemcpyAsync(d_localPos, h_localPos, (size_t)localPosCount * sizeof(double), cudaMemcpyHostToDevice, stream),
                  "H2D localPos");
        cudaCheck(cudaMemcpyAsync(d_localVel, h_localVel, (size_t)localPosCount * sizeof(double), cudaMemcpyHostToDevice, stream),
                  "H2D localVel");
    }
    cudaCheck(cudaStreamSynchronize(stream), "sync after init copies");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int threads = 256;
    const int blocks = (localBodies + threads - 1) / threads;
    const size_t shmemBytes = (size_t)threads * sizeof(Pos3);

    for (int step = 0; step < numSteps; ++step) {
        cudaCheck(cudaMemcpyAsync(d_allPos, h_allPos, (size_t)numBodies * 3 * sizeof(double), cudaMemcpyHostToDevice, stream),
                  "H2D allPos");

        if (localBodies > 0) {
            nbody_step_kernel<<<blocks, threads, shmemBytes, stream>>>(d_allPos, d_localPos, d_localVel, numBodies, localBodies);
            cudaCheck(cudaGetLastError(), "kernel launch");
            cudaCheck(cudaMemcpyAsync(h_localPos, d_localPos, (size_t)localPosCount * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "D2H localPos");
        }

        cudaCheck(cudaStreamSynchronize(stream), "sync step");

        // Exchange updated positions for next iteration.
        MPI_Allgatherv(h_localPos, localPosCount, MPI_DOUBLE, h_allPos, countsPos.data(), displsPos.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    double localDurationMs = std::chrono::duration<double, std::milli>(end - start).count();
    double globalDurationMs = 0.0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Simulation time: %.17g ms\n", globalDurationMs);
    }

    // Gather results to rank 0 for printing/validation.
    if (printResults || validate) {
        if (localBodies > 0) {
            cudaCheck(cudaMemcpyAsync(h_localVel, d_localVel, (size_t)localPosCount * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "D2H localVel");
        }
        cudaCheck(cudaStreamSynchronize(stream), "sync final D2H");

        std::vector<double> allPos;
        std::vector<double> allVel;
        if (worldRank == 0) {
            allPos.resize((size_t)numBodies * 3);
            allVel.resize((size_t)numBodies * 3);
        }

        MPI_Gatherv(h_localPos, localPosCount, MPI_DOUBLE, worldRank == 0 ? allPos.data() : nullptr, countsPos.data(),
                   displsPos.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_localVel, localPosCount, MPI_DOUBLE, worldRank == 0 ? allVel.data() : nullptr, countsPos.data(),
                   displsPos.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (worldRank == 0) {
            if (printResults) {
                std::vector<double> bodyData((size_t)numBodies * 6);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    bodyData[6 * i + 0] = allPos[3 * i + 0];
                    bodyData[6 * i + 1] = allPos[3 * i + 1];
                    bodyData[6 * i + 2] = allPos[3 * i + 2];
                    bodyData[6 * i + 3] = allVel[3 * i + 0];
                    bodyData[6 * i + 4] = allVel[3 * i + 1];
                    bodyData[6 * i + 5] = allVel[3 * i + 2];
                }
                print_results(bodyData, "Bodies");
            }

            if (validate) {
                printf("Validating simulation results...\n");

                std::vector<Body> bodies(numBodies);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    bodies[i].pos.x = allPos[3 * i + 0];
                    bodies[i].pos.y = allPos[3 * i + 1];
                    bodies[i].pos.z = allPos[3 * i + 2];
                    bodies[i].vel.x = allVel[3 * i + 0];
                    bodies[i].vel.y = allVel[3 * i + 1];
                    bodies[i].vel.z = allVel[3 * i + 2];
                }

                if (validateSimulation(bodies)) {
                    double finalEnergy = computeTotalEnergy(bodies);
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
    }

    cudaFreeHost(h_localPos);
    cudaFreeHost(h_localVel);
    cudaFreeHost(h_allPos);

    cudaFree(d_allPos);
    cudaFree(d_localPos);
    cudaFree(d_localVel);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    return 0;
}
