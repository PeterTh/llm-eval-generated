#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 256;
constexpr int BLOCK = 256;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

__global__ void computeForcesKernel(const double* __restrict__ pos,
                                    double* __restrict__ force,
                                    int n, int iBegin, int iEnd,
                                    double softening) {
    extern __shared__ double tile[];

    const int chunk = (iEnd - iBegin + gridDim.x - 1) / gridDim.x;
    const int local = blockIdx.x * chunk + threadIdx.x;
    const int idx = iBegin + local;
    const bool active = threadIdx.x < chunk && idx < iEnd;

    double ix = 0.0, iy = 0.0, iz = 0.0;
    if (active) {
        ix = pos[3 * (size_t)idx + 0];
        iy = pos[3 * (size_t)idx + 1];
        iz = pos[3 * (size_t)idx + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += TILE) {
        const int t = min(TILE, n - base);

        for (int k = threadIdx.x; k < t; k += blockDim.x) {
            const size_t j = (size_t)(base + k);
            tile[3 * k + 0] = pos[3 * j + 0];
            tile[3 * k + 1] = pos[3 * j + 1];
            tile[3 * k + 2] = pos[3 * j + 2];
        }
        __syncthreads();

        if (active) {
            for (int k = 0; k < t; ++k) {
                const double dx = tile[3 * k + 0] - ix;
                const double dy = tile[3 * k + 1] - iy;
                const double dz = tile[3 * k + 2] - iz;
                const double distSqr = dx * dx + dy * dy + dz * dz + softening;
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
        force[3 * (size_t)idx + 0] = Fx;
        force[3 * (size_t)idx + 1] = Fy;
        force[3 * (size_t)idx + 2] = Fz;
    }
}

void integrateBodies(std::vector<double>& pos, std::vector<double>& vel,
                     const std::vector<double>& force, int iBegin, int localCount) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < localCount; ++t) {
        const size_t idx = (size_t)iBegin + t;

        const double vx = vel[3 * idx + 0] + DT * force[3 * (size_t)t + 0];
        const double vy = vel[3 * idx + 1] + DT * force[3 * (size_t)t + 1];
        const double vz = vel[3 * idx + 2] + DT * force[3 * (size_t)t + 2];

        vel[3 * idx + 0] = vx;
        vel[3 * idx + 1] = vy;
        vel[3 * idx + 2] = vz;

        pos[3 * idx + 0] += vx * DT;
        pos[3 * idx + 1] += vy * DT;
        pos[3 * idx + 2] += vz * DT;
    }
}

void randomizeBodies(std::vector<double>& pos, std::vector<double>& vel,
                     size_t n, unsigned int seed = 42) {
    for (size_t i = 0; i < n; ++i) {
        pos[3 * i + 0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[3 * i + 1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[3 * i + 2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[3 * i + 0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[3 * i + 1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[3 * i + 2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<double>& pos,
                          const std::vector<double>& vel, size_t n) {
    double energy = 0.0;

    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (vel[3 * i + 0] * vel[3 * i + 0] +
                         vel[3 * i + 1] * vel[3 * i + 1] +
                         vel[3 * i + 2] * vel[3 * i + 2]);
    }

    for (size_t i = 0; i < n; ++i) {
        const double ix = pos[3 * i + 0];
        const double iy = pos[3 * i + 1];
        const double iz = pos[3 * i + 2];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = pos[3 * j + 0] - ix;
            const double dy = pos[3 * j + 1] - iy;
            const double dz = pos[3 * j + 2] - iz;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<double>& pos,
                        const std::vector<double>& vel, size_t n) {
    const double maxPos = 1e6;
    const double maxVel = 1e6;
    bool finiteOk = true;
    bool posOk = true;
    bool velOk = true;

    #pragma omp parallel for schedule(static) reduction(&&:finiteOk, posOk, velOk)
    for (size_t i = 0; i < n; ++i) {
        const double px = pos[3 * i + 0];
        const double py = pos[3 * i + 1];
        const double pz = pos[3 * i + 2];
        const double vx = vel[3 * i + 0];
        const double vy = vel[3 * i + 1];
        const double vz = vel[3 * i + 2];

        finiteOk = finiteOk && std::isfinite(px) && std::isfinite(py) && std::isfinite(pz) &&
                   std::isfinite(vx) && std::isfinite(vy) && std::isfinite(vz);

        posOk = posOk && std::abs(px) <= maxPos && std::abs(py) <= maxPos &&
                std::abs(pz) <= maxPos;

        velOk = velOk && std::abs(vx) <= maxVel && std::abs(vy) <= maxVel &&
                std::abs(vz) <= maxVel;
    }

    if (!finiteOk) {
        printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (!posOk) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (!velOk) {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (numBodies < 0) numBodies = 0;
    if (numSteps < 0) numSteps = 0;
    const size_t n = (size_t)numBodies;

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &nodeRank);
        MPI_Comm_size(nodeComm, &nodeSize);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, device));
    const int smCount = std::max(1, deviceProps.multiProcessorCount);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(size), displs(size);
    std::vector<int> counts3(size), displs3(size);
    const int base = (int)(n / (size_t)size);
    const int rem = (int)(n % (size_t)size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
        counts3[r] = 3 * counts[r];
        displs3[r] = 3 * displs[r];
    }

    const int iBegin = displs[rank];
    const int localCount = counts[rank];

    if (getenv("OMP_NUM_THREADS") == nullptr) {
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (cores < 1) cores = 1;
        int perRank = (int)(cores / (long)std::max(1, nodeSize));
        if (perRank < 1) perRank = 1;
        int workThreads = localCount / 1024;
        if (workThreads < 1) workThreads = 1;
        omp_set_num_threads(std::min(perRank, workThreads));
    }

    std::vector<double> hpos(3 * n, 0.0);
    std::vector<double> hvel(3 * n, 0.0);
    randomizeBodies(hpos, hvel, n);

    const size_t bufBodies = (n > 0 ? n : 1);
    const size_t fullBytes = 3 * n * sizeof(double);
    const size_t localBytes = 3 * (size_t)localCount * sizeof(double);

    std::vector<double> hforce(3 * (size_t)localCount, 0.0);

    double* dpos = nullptr;
    double* dforce = nullptr;
    CUDA_CHECK(cudaMalloc(&dpos, 3 * bufBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dforce, 3 * bufBodies * sizeof(double)));

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), fullBytes, cudaMemcpyHostToDevice));
    }

    const int blocksNeeded = (localCount + BLOCK - 1) / BLOCK;
    int blocks = blocksNeeded;
    if (localCount > BLOCK) {
        blocks = ((blocksNeeded + smCount - 1) / smCount) * smCount;
    }
    const size_t shmem = 3 * (size_t)TILE * sizeof(double);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            computeForcesKernel<<<blocks, BLOCK, shmem>>>(dpos, dforce, (int)n,
                                                          iBegin, iBegin + localCount,
                                                          SOFTENING);
            CUDA_CHECK(cudaMemcpy(hforce.data(), dforce + 3 * (size_t)iBegin,
                                  localBytes, cudaMemcpyDeviceToHost));
            integrateBodies(hpos, hvel, hforce, iBegin, localCount);
        }

        if (step + 1 < numSteps && n > 0) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hpos.data(),
                           counts3.data(), displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), fullBytes, cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if ((printResults || validate) && n > 0) {
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hpos.data(),
                        counts3.data(), displs3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, hvel.data(),
                        counts3.data(), displs3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(hpos.data() + 3 * (size_t)iBegin, 3 * localCount, MPI_DOUBLE,
                        hpos.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
            MPI_Gatherv(hvel.data() + 3 * (size_t)iBegin, 3 * localCount, MPI_DOUBLE,
                        hvel.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
        }
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", global_ms);

        if (printResults) {
            std::vector<double> bodyData(6 * n);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                bodyData[6 * i + 0] = hpos[3 * i + 0];
                bodyData[6 * i + 1] = hpos[3 * i + 1];
                bodyData[6 * i + 2] = hpos[3 * i + 2];
                bodyData[6 * i + 3] = hvel[3 * i + 0];
                bodyData[6 * i + 4] = hvel[3 * i + 1];
                bodyData[6 * i + 5] = hvel[3 * i + 2];
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(hpos, hvel, n)) {
                double finalEnergy = computeTotalEnergy(hpos, hvel, n);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(dpos));
    CUDA_CHECK(cudaFree(dforce));

    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&nodeComm);
    }
    MPI_Finalize();

    return exitCode;
}
